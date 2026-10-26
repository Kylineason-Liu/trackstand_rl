"""Motor node: 50 Hz policy input, 100 Hz feedback, 200 Hz MIT/C0 control.

One SocketCAN owner performs preflight, receives feedback, polls A3/A2/A1,
checks the 200 Hz safety gate, and sends commands. The ROS interface matches
motor_io; versioned motor_control telemetry exposes native MIT command/feedback data.
This node owns its state and ROS handling; it does not import either legacy motor node.
Firmware PD frequency and MIT current limiting are not established by the
local protocol; F0 ranges do not implement the previous host torque clamp.
V1 position clip/slew is performed by policy_controller at 50 Hz; this node
holds those targets at 200 Hz, bounds motor requests and preserves safety gates.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import errno
import fcntl
import json
import math
import select
import socket
import struct
import threading
import time

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float64MultiArray, MultiArrayDimension, String

from .can_protocol import (
    RADIANS_PER_COUNT, current_feedback_to_torque_nm, parse_a2_velocity, parse_a3_multi_turn_count,
    parse_a3_position, parse_b0_torque_constant, parse_current_feedback, q_current_payload,
)
from .contracts import (
    DRIVE_DAMPING_NM_S_PER_RAD, DRIVE_EFFORT_LIMIT_NM,
    STEERING_KD_NM_S_PER_RAD, STEERING_KP_NM_PER_RAD, motor_fault_clear_allowed,
    STEERING_ACTION_SCALE_RAD,
)
from .mit_protocol import (
    MitCommand, MitFeedback, MitRanges, f0_with_position, mit_motion_payload,
    parse_f0, parse_f1, parse_mit_motion,
)
from .telemetry import (
    MOTOR_CONTROL_FIELDS as BASE_CONTROL_FIELDS,
    MOTOR_STATE_FIELDS, MOTOR_STATE_LABEL, DisableResult, MotorControlPublishGate, MotorSendResult,
    age_ms, fresh_torque_feedback_nm, motor_disable_reasons, send_result_values, telemetry_values,
)


# Division 1: policy-originated ROS commands. This is the producer's rate,
# not a callback throttle: disable requests are never discarded/delayed.
COMMAND_RATE_HZ = 50.0

# Division 2: encoder/current query rounds and state/status/fault publication.
STATE_RATE_HZ = 100.0
STATE_DT_S = 1.0 / STATE_RATE_HZ

# Division 3: MIT steering, rear C0, safety checks and control diagnostics.
MIT_RATE_HZ = 200.0
MIT_DT_S = 1.0 / MIT_RATE_HZ
MIT_POSITION_MAX_RAD = 3.0
QUERY_TIMEOUT_S = 0.002
DISABLE_TIMEOUT_S = 0.25
DISABLE_RETRY_S = 0.02
MAX_RX_BATCH = 64
CAN_FRAME = struct.Struct("=IB3x8s")
CAN_FILTER = struct.Struct("=II")
CAN_MASK = 0xC00007FF
SIOCGIFFLAGS = 0x8913
IFF_UP = 0x1
SO_RXQ_OVFL = getattr(socket, "SO_RXQ_OVFL", 40)
# Linux asm-generic/socket.h: NEW uses a 64-bit timespec on both 32/64-bit Pi OS.
SO_TIMESTAMPNS_NEW = 64
KERNEL_TIMESPEC = struct.Struct("=qq")
MOTOR_CONTROL_FIELDS = BASE_CONTROL_FIELDS + (
    "mit_config_ready", "mit_restart_required", "mit_f1_age_ms",
    "mit_send_due", "mit_last_send_age_ms", "mit_native_target_rad",
    "mit_native_zero_rad", "mit_position_max_rad", "mit_velocity_max_rad_s", "mit_torque_max_nm",
    "mit_feedback_position_motor_rad", "mit_feedback_velocity_motor_rad_s",
    "mit_feedback_torque_motor_nm", "mit_feedback_active", "mit_feedback_fault",
    "mit_encoded_position_motor_rad", "mit_encoded_position_rad", "mit_encoded_velocity_motor_rad_s",
    "mit_encoded_kp_nm_per_motor_rad", "mit_encoded_kd_nm_s_per_motor_rad",
    "mit_encoded_feedforward_torque_motor_nm", "mit_actual_send_interval_ms",
    "steering_unwrapped_position_rad", "steering_unwrapped_position_age_ms",
    "disable_pending", "disable_confirmed", "steering_disable_confirmed", "drive_disable_confirmed",
)


@dataclass
class ChannelState:
    """MIT-owned encoder/current state; joint feedback uses the configured signs."""

    name: str
    host_id: int
    feedback_id: int
    gear_ratio: float
    encoder_sign: float
    torque_sign: float
    zero_count: int | None = None
    position_rad: float = 0.0
    velocity_rad_s: float = 0.0
    torque_constant_nm_per_a: float | None = None
    sample_monotonic: float = 0.0
    torque_nm: float = math.nan
    torque_sample_monotonic: float = 0.0
    position_sample_monotonic: float = 0.0
    velocity_sample_monotonic: float = 0.0
    position_query_ms: float = math.nan
    velocity_query_ms: float = math.nan
    position_query_failures: int = 0
    velocity_query_failures: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class CanReply:
    can_id: int
    data: bytes
    local_tx: bool
    dropped: int
    received_monotonic: float


class SocketCan:
    """Sole receiver/sender; strict standard-frame and overflow checking."""

    def __init__(self, interface: str, steering_id: int, drive_id: int):
        self.interface = interface
        self.socket = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        ids = [steering_id, drive_id, 0, 0xFF, 0x400, 0x4FF]
        for device_id in (steering_id, drive_id):
            ids.extend((0x100 | device_id, 0x400 | device_id, 0x500 | device_id))
        try:
            filters = b"".join(CAN_FILTER.pack(can_id, CAN_MASK) for can_id in ids)
            self.socket.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, filters)
            self.socket.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_RECV_OWN_MSGS, 0)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
            self.socket.setsockopt(socket.SOL_SOCKET, SO_RXQ_OVFL, 1)
            self.socket.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS_NEW, 1)
            self.socket.setblocking(False)
            self.socket.bind((interface,))
            # Binding a raw CAN socket can succeed on an administratively down
            # interface. Reject that before any motor preflight commands.
            request = struct.pack("256s", interface.encode())
            flags = fcntl.ioctl(self.socket.fileno(), SIOCGIFFLAGS, request)
            if not struct.unpack_from("=H", flags, 16)[0] & IFF_UP:
                raise OSError(
                    errno.ENETDOWN,
                    f"CAN interface {interface!r} is DOWN; configure its motor bitrate "
                    "and bring it UP before starting motor_mit_io",
                )
        except BaseException:
            self.socket.close()
            raise
        self._last_dropped = 0
        self._wall_offset_s = None
        self._clock_uncertainty_s = 0.0

    def send(self, can_id: int, payload: bytes) -> None:
        if not 0 <= can_id <= 0x7FF or not 1 <= len(payload) <= 8:
            raise ValueError("invalid standard CAN frame")
        try:
            sent = self.socket.send(CAN_FRAME.pack(can_id, len(payload), payload))
        except OSError as error:
            raise OSError(
                error.errno,
                f"CAN send on {self.interface!r}, ID {can_id:03X}, "
                f"data {payload.hex().upper()}: {error.strerror or error}",
            ) from error
        if sent != CAN_FRAME.size:
            raise OSError("short SocketCAN send")

    def receive(self, timeout_s: float) -> CanReply | None:
        if not select.select([self.socket], [], [], max(0.0, timeout_s))[0]:
            return None
        frame, ancillary, flags, _ = self.socket.recvmsg(
            CAN_FRAME.size, socket.CMSG_SPACE(4) + socket.CMSG_SPACE(KERNEL_TIMESPEC.size)
        )
        if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or len(frame) != CAN_FRAME.size:
            raise RuntimeError("truncated SocketCAN frame or metadata")
        dropped = self._last_dropped
        received_wall_s = None
        for level, kind, data in ancillary:
            if level == socket.SOL_SOCKET and kind == SO_RXQ_OVFL:
                if len(data) != 4:
                    raise RuntimeError("invalid CAN overflow metadata")
                dropped = struct.unpack("=I", data)[0]
            elif level == socket.SOL_SOCKET and kind == SO_TIMESTAMPNS_NEW:
                if len(data) != KERNEL_TIMESPEC.size or received_wall_s is not None:
                    raise RuntimeError("invalid CAN receive timestamp metadata")
                seconds, nanoseconds = KERNEL_TIMESPEC.unpack(data)
                if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
                    raise RuntimeError("invalid CAN receive timestamp value")
                received_wall_s = seconds + nanoseconds * 1e-9
        if dropped != self._last_dropped:
            raise RuntimeError(f"CAN receive queue overflow ({dropped - self._last_dropped} dropped frames)")
        self._last_dropped = dropped
        can_id, dlc, padded = CAN_FRAME.unpack(frame)
        if can_id & ~0x7FF or dlc > 8:
            raise RuntimeError("nonstandard, remote, or CAN error frame received")
        if received_wall_s is None:
            raise RuntimeError("CAN kernel receive timestamp missing")
        # Kernel SO_TIMESTAMPNS is CLOCK_REALTIME. Preserve queue age when
        # mapping to the monotonic safety clock; fail closed on clock steps.
        before = time.monotonic()
        wall_s = time.time()
        after = time.monotonic()
        offset_s = wall_s - before
        uncertainty_s = after - before
        if uncertainty_s > 0.001:
            raise RuntimeError("system clock sampling delayed; CAN feedback age is uncertain")
        if (self._wall_offset_s is not None and abs(offset_s - self._wall_offset_s)
                > 0.001 + uncertainty_s + self._clock_uncertainty_s):
            raise RuntimeError("system clock changed; CAN feedback age is uncertain")
        self._wall_offset_s, self._clock_uncertainty_s = offset_s, uncertainty_s
        received_monotonic = received_wall_s - offset_s
        if received_monotonic > after:
            raise RuntimeError("CAN receive timestamp is in the future")
        return CanReply(can_id, padded[:dlc], bool(flags & socket.MSG_DONTROUTE), dropped, received_monotonic)

    def close(self) -> None:
        self.socket.close()


class MotorMitIo(Node):
    """Independent MIT/C0 node implementing the established ROS motor contracts."""

    _parameter_defaults = {
        "can_interface": "can0", "steering_host_id": "102", "steering_feedback_id": "002",
        "drive_host_id": "101", "drive_feedback_id": "001", "steering_gear_ratio": 1.0,
        "drive_gear_ratio": 1.0, "steering_encoder_sign": -1.0, "drive_encoder_sign": -1.0,
        "steering_torque_sign": -1.0, "drive_torque_sign": -1.0,
        # MIT-specific rate keys ignore the shared YAML's legacy overrides.
        "command_rate_hz": COMMAND_RATE_HZ, "state_rate_hz": STATE_RATE_HZ,
        "actuation_rate_hz": MIT_RATE_HZ,
        "command_timeout_s": 0.08, "feedback_timeout_s": 0.05,
        "steering_limit_rad": math.radians(90.0), "steering_fault_clear_rad": math.radians(5.0),
        "dry_run": False,
        "mit_kp_nm_per_motor_rad": STEERING_KP_NM_PER_RAD,
        "mit_kd_nm_s_per_motor_rad": STEERING_KD_NM_S_PER_RAD,
    }
    _control_fields = MOTOR_CONTROL_FIELDS
    _control_label = "motor_control_mit_v1"
    _snapshot_fields = (
        "position_rad", "velocity_rad_s", "sample_monotonic", "torque_constant_nm_per_a",
        "position_sample_monotonic", "velocity_sample_monotonic", "position_query_ms",
        "velocity_query_ms", "position_query_failures", "velocity_query_failures",
    )

    def __init__(self) -> None:
        # One CAN owner prevents competing pollers from stealing replies.
        Node.__init__(self, "trackstand_motor_io")
        for name, default in self._parameter_defaults.items():
            self.declare_parameter(name, default, ParameterDescriptor(read_only=True))
        parameter = lambda name: self.get_parameter(name).value
        self._can_interface = str(parameter("can_interface"))
        self._command_rate_hz = float(parameter("command_rate_hz"))
        self._poll_rate_hz = float(parameter("state_rate_hz"))
        self._control_rate_hz = float(parameter("actuation_rate_hz"))
        self._command_timeout_s = float(parameter("command_timeout_s"))
        self._feedback_timeout_s = float(parameter("feedback_timeout_s"))
        self._steering_limit_rad = float(parameter("steering_limit_rad"))
        self._steering_fault_clear_rad = float(parameter("steering_fault_clear_rad"))
        self._dry_run = bool(parameter("dry_run"))
        self._mit_kp = float(parameter("mit_kp_nm_per_motor_rad"))
        self._mit_kd = float(parameter("mit_kd_nm_s_per_motor_rad"))
        scalars = (self._command_rate_hz, self._poll_rate_hz, self._control_rate_hz, self._command_timeout_s,
                   self._feedback_timeout_s, self._steering_limit_rad, self._steering_fault_clear_rad,
                   self._mit_kp, self._mit_kd)
        if (not all(map(math.isfinite, scalars)) or self._poll_rate_hz <= 0 or self._control_rate_hz <= 0
                or self._command_timeout_s <= 0 or self._feedback_timeout_s <= 0
                or self._steering_limit_rad <= 0 or not 0 < self._steering_fault_clear_rad < self._steering_limit_rad
                or not 0 <= self._mit_kp <= 500 or not 0 <= self._mit_kd <= 5):
            raise ValueError("invalid motor rate, timeout, limit, or MIT gain")
        if (self._command_rate_hz, self._poll_rate_hz, self._control_rate_hz) != (
                COMMAND_RATE_HZ, STATE_RATE_HZ, MIT_RATE_HZ):
            raise ValueError("MIT command/state/actuation rates must remain 50/100/200 Hz")
        if self._command_timeout_s <= 1.0 / COMMAND_RATE_HZ:
            raise ValueError("command timeout must exceed the 20 ms policy interval")
        if self._feedback_timeout_s <= STATE_DT_S:
            raise ValueError("feedback timeout must exceed the 10 ms state-query interval")

        def channel(prefix: str) -> ChannelState:
            host = int(str(parameter(f"{prefix}_host_id")), 16)
            feedback = int(str(parameter(f"{prefix}_feedback_id")), 16)
            motor = ChannelState(prefix, host, feedback, float(parameter(f"{prefix}_gear_ratio")),
                                 float(parameter(f"{prefix}_encoder_sign")),
                                 float(parameter(f"{prefix}_torque_sign")))
            if not (1 <= feedback <= 254 and host == (0x100 | feedback)):
                raise ValueError(f"{prefix} host/feedback CAN IDs do not identify the same motor")
            if not math.isfinite(motor.gear_ratio) or motor.gear_ratio <= 0 or motor.encoder_sign not in (-1.0, 1.0):
                raise ValueError(f"invalid {prefix} encoder configuration")
            if motor.torque_sign not in (-1.0, 1.0):
                raise ValueError(f"invalid {prefix} torque sign")
            return motor

        self._steering, self._drive = channel("steering"), channel("drive")
        if self._steering.feedback_id == self._drive.feedback_id:
            raise ValueError("steering and drive cannot have the same CAN ID")
        self._initialize_command_state()
        self._hard_fault = False
        self._mit_ranges: MitRanges | None = None
        self._mit_zero_rad: float | None = None
        self._mit_feedback: MitFeedback | None = None
        self._mit_encoded_command: MitCommand | None = None
        self._mit_send_interval_ms = math.nan
        self._mit_f1_at = self._mit_last_sent_at = self._mit_first_sent_at = self._next_mit_at = 0.0
        self._mit_sent_this_enable = False
        self._slew_last_at = 0.0
        self._mit_config_ready = False
        self._pending = None
        self._poll_jobs = deque()
        self._next_poll_at = self._next_b0_at = self._next_f0_at = 0.0
        self._tx_bus: SocketCan | None = None
        self._last_disable_errors = ()
        self._disable_wait: dict[str, float | None] = {}
        self._disable_f1_wait: dict[str, float] = {}
        self._disable_confirmed = False
        self._disable_deadline_at = self._next_disable_retry_at = 0.0

        self._create_interfaces()
        self._publish_timer = self.create_timer(STATE_DT_S, self._publish_feedback)
        self._worker = threading.Thread(target=self._run_io, name="trackstand_mit_can", daemon=True)
        self.get_logger().info(
            f"Policy input {COMMAND_RATE_HZ:g} Hz, state/query {STATE_RATE_HZ:g} Hz, "
            f"MIT steering/C0 drive/control {MIT_RATE_HZ:g} Hz on {self._can_interface}; "
            f"dry_run={self._dry_run}"
        )
        self._worker.start()

    def _initialize_command_state(self) -> None:
        self._running = True
        self._requested_enable = self._outputs_active = self._fault_latched = False
        self._steering_target_rad = self._requested_steering_target_rad = self._slewed_steering_target_rad = 0.0
        self._drive_target_nm = self._requested_drive_target_nm = self._command_monotonic = 0.0
        self._fault_reason: str | None = None
        self._command_lock = threading.Lock()
        self._disable_generation = self._active_disable_generation = 0
        self._last_published_sample_times = (0.0, 0.0)
        self._control_sequence = 0
        self._last_control_monotonic = 0.0
        self._last_event_state = None
        self._control_publish_gate = MotorControlPublishGate(MIT_RATE_HZ)
        self._feedback_gate_fault = False

    def _create_interfaces(self) -> None:
        for name, kind, qos in (
            ("state", Float64MultiArray, 10),
            ("control", Float64MultiArray, QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT)),
            ("event", String, 100), ("fault", Bool, 10), ("status", String, 10),
        ):
            setattr(self, f"_{name}_publisher", self.create_publisher(kind, f"/trackstand/motor_{name}", qos))
        for name, label, fields in (("state", MOTOR_STATE_LABEL, MOTOR_STATE_FIELDS),
                                    ("control", self._control_label + ":" + ",".join(self._control_fields),
                                     self._control_fields)):
            setattr(self, f"_{name}_dimension", MultiArrayDimension(label=label, size=len(fields), stride=len(fields)))
        self._command_subscription = self.create_subscription(
            Float64MultiArray, "/trackstand/motor_command", self._command_callback, 10,
        )

    def _command_callback(self, message: Float64MultiArray) -> None:
        if len(message.data) != 3:
            self.get_logger().warning("motor_command must be [enable, steering_target_rad, drive_torque_nm]")
            return
        values = tuple(map(float, message.data))
        if not all(map(math.isfinite, values)):
            self.get_logger().error("Rejected non-finite motor command")
            return
        with self._command_lock:
            self._requested_enable = bool(values[0] > 0.5)
            if not self._requested_enable:
                self._disable_generation += 1
            self._requested_steering_target_rad, self._requested_drive_target_nm = values[1:]
            limit = min(self._steering_limit_rad, STEERING_ACTION_SCALE_RAD)
            self._steering_target_rad = max(-limit, min(limit, values[1]))
            self._drive_target_nm = max(-DRIVE_EFFORT_LIMIT_NM, min(DRIVE_EFFORT_LIMIT_NM, values[2]))
            self._command_monotonic = time.monotonic()

    def _snapshot(self, motor: ChannelState) -> tuple:
        with motor.lock:
            return tuple(getattr(motor, name) for name in self._snapshot_fields)

    def _send(self, can_id: int, data: bytes) -> None:
        if self._tx_bus is None:
            raise RuntimeError("CAN socket not open")
        self._tx_bus.send(can_id, data)

    def _fault(self, reason: str, hard: bool = False) -> None:
        self._fault_latched = True
        if not self._hard_fault:
            self._fault_reason = reason
        self._hard_fault |= hard

    def _consume(self, frame: CanReply, stamp: float) -> None:
        # stamp is dequeue time; only kernel arrival time can refresh feedback.
        stamp = frame.received_monotonic
        if not math.isfinite(stamp) or stamp <= 0:
            raise RuntimeError("invalid CAN monotonic receive timestamp")
        if frame.local_tx:
            raise RuntimeError(f"another local CAN writer used ID {frame.can_id:03X}")
        motor = self._steering if frame.can_id == self._steering.feedback_id else (
            self._drive if frame.can_id == self._drive.feedback_id else None)
        if motor is None:
            # Another process is issuing selected-motor commands or queries.
            raise RuntimeError(f"another CAN writer used selected motor ID {frame.can_id:03X}")
        data = frame.data
        if not data:
            raise RuntimeError("empty motor response")
        opcode = data[0]
        if opcode == 0xF1 and len(data) == 7:
            if data[6] & ~3:
                raise RuntimeError("unknown F1 status bits")
            verify_sent_at = self._disable_f1_wait.get(motor.name)
            if verify_sent_at is not None and stamp >= verify_sent_at:
                if stamp > self._disable_deadline_at:
                    self._fault("disable_ack_timeout", hard=True)
                if data[6] & 2:
                    self._fault(f"{motor.name}_mit_system_fault", hard=True)
                    self._disable_wait[motor.name] = None
                    del self._disable_f1_wait[motor.name]
                elif data[6] & 1:
                    self._fault(f"{motor.name}_disable_not_confirmed", hard=True)
                    self._disable_wait[motor.name] = None
                    del self._disable_f1_wait[motor.name]
                else:
                    self._confirm_disable(motor)
        if opcode in (0xA3, 0xA2):
            pending = self._pending
            if (pending is None or pending[0] is not motor or pending[1] != opcode
                    or not pending[2] <= stamp <= pending[3] or len(data) != pending[4]):
                return  # An expired or unsolicited encoder reply cannot refresh a pair.
        if opcode in (0xA1, 0xC0) and len(data) == 5:
            current_a = parse_current_feedback(data)
            with motor.lock:
                constant = motor.torque_constant_nm_per_a
            if constant is not None:
                torque_nm = current_feedback_to_torque_nm(current_a, constant, motor.torque_sign)
                with motor.lock:
                    motor.torque_nm = torque_nm
                    motor.torque_sample_monotonic = stamp
        elif opcode == 0xB0 and len(data) == 7:
            try:
                value = parse_b0_torque_constant(data)
            except ValueError:
                return
            with motor.lock:
                motor.torque_constant_nm_per_a = value
        elif opcode == 0xA3 and len(data) == 7:
            count = parse_a3_multi_turn_count(data)
            with motor.lock:
                if motor.zero_count is None:
                    motor.zero_count = count
                    self.get_logger().info(f"{motor.name} A3 zero={count} counts")
                motor._unwrapped_rad = (
                    motor.encoder_sign * (count - motor.zero_count) * RADIANS_PER_COUNT / motor.gear_ratio
                )
                motor._unwrapped_at = stamp
                motor._pending_position = (parse_a3_position(data, motor.zero_count, motor.gear_ratio,
                                                               motor.encoder_sign), stamp)
                if self._pending is not None and self._pending[:2] == (motor, 0xA3):
                    motor.position_query_ms = (stamp - self._pending[2]) * 1000.0
        elif opcode == 0xA2 and len(data) == 5:
            value = parse_a2_velocity(data, motor.gear_ratio, motor.encoder_sign)
            with motor.lock:
                if self._pending is not None and self._pending[:2] == (motor, 0xA2):
                    motor.velocity_query_ms = (stamp - self._pending[2]) * 1000.0
                position = getattr(motor, "_pending_position", None)
                if position is not None and stamp - position[1] <= self._feedback_timeout_s:
                    motor.position_rad = position[0]
                    motor.velocity_rad_s = value
                    motor.sample_monotonic = position[1]  # Older receipt bounds pair freshness.
                    motor.position_sample_monotonic = position[1]
                    motor.velocity_sample_monotonic = stamp
                    motor._pending_position = None
        elif motor is self._steering and opcode == 0xF0 and len(data) == 7:
            ranges = parse_f0(data)
            if self._mit_config_ready and ranges != self._mit_ranges:
                raise RuntimeError("steering F0 encoding ranges changed")
            self._mit_ranges = ranges
        elif opcode == 0xF1 and len(data) == 7:
            if data[6] & 2:
                self._fault(f"{motor.name}_mit_system_fault", hard=True)
            if motor is self._steering and self._mit_ranges is not None:
                feedback = parse_f1(data, self._mit_ranges)
                if stamp >= self._mit_f1_at:
                    self._mit_feedback, self._mit_f1_at = feedback, stamp
                if (self._mit_sent_this_enable and stamp >= self._mit_first_sent_at
                        and not feedback.mode_active):
                    self._fault("steering_mit_unexpected_exit", hard=True)
        elif motor is self._steering and opcode == 0xB1 and self._mit_config_ready:
            raise RuntimeError("steering origin changed after startup")
        elif opcode in (0xCF, 0xAE) and len(data) == 8:
            if data[7]:
                self._fault(f"{motor.name}_driver_fault", hard=True)
            disable_sent_at = self._disable_wait.get(motor.name)
            if opcode == 0xCF and disable_sent_at is not None and stamp >= disable_sent_at:
                if stamp > self._disable_deadline_at:
                    self._fault("disable_ack_timeout", hard=True)
                if data[7]:
                    self._disable_wait[motor.name] = None
                    self._disable_f1_wait.pop(motor.name, None)
                elif data[6] == 0:
                    self._confirm_disable(motor)
                elif 1 <= data[6] <= 5:
                    # This firmware retains its selected mode after CF. Require
                    # a fresh, fault-free F1 readback confirming MIT is inactive.
                    if motor.name not in self._disable_f1_wait:
                        sent_at = time.monotonic()
                        try:
                            self._send(motor.host_id, bytes([0xF1]))
                        except (OSError, RuntimeError, ValueError):
                            self._disable_wait[motor.name] = None
                            raise
                        self._disable_f1_wait[motor.name] = sent_at
                else:
                    self._fault(f"{motor.name}_disable_not_confirmed", hard=True)
                    self._disable_wait[motor.name] = None
                    self._disable_f1_wait.pop(motor.name, None)
            elif (self._mit_sent_this_enable and stamp >= self._mit_first_sent_at and data[6] == 0):
                reason = "steering_driver_left_mit_mode" if motor is self._steering else "drive_driver_unexpected_exit"
                self._fault(reason, hard=True)
        elif opcode in (0xA1, 0xA2, 0xA3, 0xB0, 0xF0, 0xF1, 0xCF, 0xAE):
            raise RuntimeError(f"malformed {motor.name} {opcode:02X} feedback")

        pending = self._pending
        if (pending is not None and pending[0] is motor and opcode == pending[1]
                and len(data) == pending[4] and pending[2] <= stamp <= pending[3]):
            self._pending = None

    def _confirm_disable(self, motor: ChannelState) -> None:
        self._disable_wait.pop(motor.name, None)
        self._disable_f1_wait.pop(motor.name, None)
        if not self._disable_wait:
            self._disable_confirmed, self._outputs_active = True, False

    def _receive_one(self, timeout_s: float) -> CanReply | None:
        frame = self._tx_bus.receive(timeout_s)
        if frame is not None:
            self._consume(frame, frame.received_monotonic)
        return frame

    def _drain_received(self) -> None:
        """Read queued faults before any actuation or new request; bound backlog."""
        for _ in range(MAX_RX_BATCH):
            if self._receive_one(0.0) is None:
                return
        raise RuntimeError("CAN receive backlog did not clear")

    def _query(self, motor: ChannelState, opcode: int, expected_length: int, timeout_s: float,
               payload: bytes | None = None) -> bytes:
        self._drain_received()
        sent_at = time.monotonic()
        deadline = sent_at + timeout_s
        self._pending = (motor, opcode, sent_at, deadline, expected_length)
        try:
            self._send(motor.host_id, bytes([opcode]) if payload is None else payload)
            while self._running and time.monotonic() < deadline:
                frame = self._receive_one(max(0.0, deadline - time.monotonic()))
                if frame is None:
                    continue
                if (sent_at <= frame.received_monotonic <= deadline and frame.can_id == motor.feedback_id
                        and len(frame.data) == expected_length and frame.data[0] == opcode):
                    return frame.data
            raise TimeoutError(f"{motor.name} {opcode:02X} startup reply missing or invalid")
        finally:
            self._pending = None

    def _verify_startup_disable(self, motor: ChannelState) -> None:
        status = self._query(motor, 0xCF, 8, 0.25)
        if status[7] or status[6] > 5:
            raise RuntimeError(
                f"{motor.name} CF invalid/faulted: mode={status[6]}, "
                f"fault=0x{status[7]:02X}, reply={status.hex().upper()}"
            )
        if status[6] != 0:
            inactive = self._query(motor, 0xF1, 7, 0.25)
            if inactive[6] != 0:
                raise RuntimeError(
                    f"{motor.name} CF retained mode={status[6]}; F1 did not "
                    f"confirm inactive/fault-free state: {inactive.hex().upper()}"
                )
            self.get_logger().info(
                f"{motor.name} CF acknowledged with retained mode={status[6]}; "
                "F1 confirms MIT inactive and fault-free"
            )

    def _startup(self) -> None:
        steer, drive = self._steering, self._drive
        if not self._dry_run:
            # CF exits MIT; B1 sets the persistent origin once per process, not per episode.
            for motor in (steer, drive):
                self._send(motor.host_id, q_current_payload(0.0))
                self._verify_startup_disable(motor)
            with self._command_lock:
                if self._requested_enable:
                    raise RuntimeError("kickstands must remain down during MIT origin setup")
            self._disable_confirmed = True
            self._query(steer, 0xB1, 3, 0.25)
            steer.zero_count = None
            self._mit_zero_rad = None
        f0 = self._query(steer, 0xF0, 7, 0.25)
        if not self._dry_run:
            wanted = f0_with_position(f0, MIT_POSITION_MAX_RAD)
            if self._query(steer, 0xF0, 7, 0.25, wanted) != wanted:
                raise RuntimeError("steering F0 write was not confirmed")
            if self._query(steer, 0xF0, 7, 0.25) != wanted:
                raise RuntimeError("steering F0 readback differs")
        a3 = self._query(steer, 0xA3, 7, 0.25)
        single_turn_count = struct.unpack("<H", a3[1:3])[0]
        single_near_zero = abs((single_turn_count + 8192) % 16384 - 8192) <= 2
        if not self._dry_run and (not single_near_zero or single_turn_count >= 16384
                                 or abs(parse_a3_multi_turn_count(a3)) > 2):
            raise RuntimeError("steering A3 not near B1 origin")
        # The verified reply, never an earlier unsolicited frame, defines zero.
        with steer.lock:
            steer.zero_count = parse_a3_multi_turn_count(a3)
            steer._unwrapped_rad = 0.0
            steer._pending_position = None
        f1 = self._query(steer, 0xF1, 7, 0.25)
        feedback = parse_f1(f1, self._mit_ranges)
        if feedback.system_fault or feedback.mode_active:
            raise RuntimeError("steering F1 not inactive and fault-free at startup")
        if not self._dry_run and abs(feedback.position_rad) > (
            2 * RADIANS_PER_COUNT + 2 * self._mit_ranges.position_rad / 65535
        ):
            raise RuntimeError("steering F1 does not agree with B1 origin")
        if abs(feedback.velocity_rad_s) > 0.1 + 2 * self._mit_ranges.velocity_rad_s / 4095:
            raise RuntimeError("steering must be stationary at MIT startup")
        self._mit_zero_rad = feedback.position_rad
        if abs(self._mit_zero_rad) + self._steering_limit_rad * steer.gear_ratio > self._mit_ranges.position_rad:
            raise RuntimeError("startup reference plus steering target range exceeds F0 Pos_Max")
        self._mit_config_ready = True
        if self._dry_run:
            self._disable_confirmed = True
        for motor in (steer, drive):
            try:
                self._query(motor, 0xB0, 7, 0.1)
            except TimeoutError:
                self.get_logger().warning(f"{motor.name} B0 unavailable; motor remains gated pending retry")
        self.get_logger().info(
            f"MIT preflight: F0={self._mit_ranges}, F1 zero={self._mit_zero_rad:.6f} native rad"
        )

    def _schedule_polls(self, now: float) -> None:
        if self._pending is not None and now > self._pending[3]:
            motor, opcode = self._pending[:2]
            if opcode == 0xA3:
                motor.position_query_failures += 1
                motor._pending_position = None
            elif opcode == 0xA2:
                motor.velocity_query_failures += 1
                motor._pending_position = None
            self._pending = None
        if now + 1e-9 >= self._next_poll_at and not self._poll_jobs:
            self._poll_jobs.extend(((self._steering, 0xA3, 7), (self._steering, 0xA2, 5),
                                    (self._drive, 0xA3, 7), (self._drive, 0xA2, 5),
                                    (self._steering, 0xA1, 5), (self._drive, 0xA1, 5)))
            # Configuration verification/calibration retries remain at 1 s.
            if now >= self._next_b0_at:
                for motor in (self._steering, self._drive):
                    if motor.torque_constant_nm_per_a is None:
                        self._poll_jobs.append((motor, 0xB0, 7))
                self._next_b0_at = now + 1.0
            if now >= self._next_f0_at:
                self._poll_jobs.append((self._steering, 0xF0, 7))
                self._next_f0_at = now + 1.0
            if self._next_poll_at <= 0.0:
                self._next_poll_at = now + STATE_DT_S
            else:
                missed = math.floor(max(0.0, now - self._next_poll_at + 1e-9) / STATE_DT_S)
                self._next_poll_at += (missed + 1) * STATE_DT_S
        if self._pending is None and self._poll_jobs:
            motor, opcode, length = self._poll_jobs.popleft()
            sent = time.monotonic()
            self._send(motor.host_id, bytes([opcode]))
            self._pending = (motor, opcode, sent, sent + QUERY_TIMEOUT_S, length)

    def _run_io(self) -> None:
        try:
            self._tx_bus = SocketCan(self._can_interface, self._steering.feedback_id, self._drive.feedback_id)
            self._startup()
            next_control = time.monotonic()
            while self._running:
                self._drain_received()
                now = time.monotonic()
                if now >= next_control:
                    self._control_tick()
                    next_control += 1.0 / self._control_rate_hz
                    # Align both motors, respecting the actual MIT send deadline.
                    if self._outputs_active and not self._disable_wait:
                        next_control = max(next_control, self._next_mit_at)
                    if next_control <= time.monotonic():
                        next_control = time.monotonic() + 1.0 / self._control_rate_hz
                self._schedule_polls(time.monotonic())
                deadline = next_control
                if self._pending is not None:
                    deadline = min(deadline, self._pending[3])
                elif not self._poll_jobs:
                    deadline = min(deadline, self._next_poll_at)
                self._receive_one(min(MIT_DT_S, max(0.0, deadline - time.monotonic())))
        except Exception as error:
            self._fault(f"mit_io_error:{type(error).__name__}:{error}", hard=True)
            self.get_logger().error(f"MIT motor I/O stopped: {error}")
        finally:
            self._running = False
            if self._tx_bus is not None:
                try:
                    self._disable_outputs(force=True)
                    deadline = time.monotonic() + DISABLE_TIMEOUT_S
                    while self._disable_wait and time.monotonic() < deadline:
                        self._receive_one(min(STATE_DT_S, max(0.0, deadline - time.monotonic())))
                        self._disable_outputs()
                    if self._disable_wait:
                        self._fault("disable_ack_timeout", hard=True)
                        self.get_logger().error("Motor disable unconfirmed; inspect motor state physically")
                except Exception as error:
                    self._fault("shutdown_disable_unconfirmed", hard=True)
                    self.get_logger().error(f"Motor disable/shutdown failed: {error}")
                finally:
                    try:
                        self._tx_bus.close()
                    except Exception as error:
                        self._fault("shutdown_close_failed", hard=True)
                        self.get_logger().error(f"CAN close failed: {error}")
                    finally:
                        self._tx_bus = None
            # SIGINT can invalidate ROS before the worker finishes. CAN cleanup
            # above must still run, but a terminal ROS event is then unavailable.
            if self.context.ok():
                try:
                    self._publish_event(time.monotonic(), {
                        "gate_enabled": False, "ros_time_s": self.get_clock().now().nanoseconds * 1e-9,
                    }, ["mit_io_stopped"] + (["fault_latched"] if self._fault_latched else []),
                        DisableResult(errors=self._last_disable_errors))
                except Exception as error:
                    if self.context.ok():
                        self.get_logger().error(f"Terminal motor event unavailable: {error}")

    def _send_torque(self, motor: ChannelState, torque_nm: float, torque_constant: float,
                     guard=None) -> MotorSendResult:
        result = MotorSendResult()
        started = time.monotonic()
        try:
            if not math.isfinite(torque_constant) or torque_constant <= 0:
                raise ValueError("B0 torque constant unavailable")
            current_a = motor.torque_sign * torque_nm / torque_constant
            payload = q_current_payload(current_a)
            encoded = parse_current_feedback(payload)
            result.current_command_a = current_a
            result.encoded_current_a = encoded
            result.encoded_torque_nm = current_feedback_to_torque_nm(encoded, torque_constant, motor.torque_sign)
            if (guard is None or guard()) and not self._dry_run:
                result.attempted = True
                self._send(motor.host_id, payload)
                result.success = True
        except (OSError, RuntimeError, ValueError, OverflowError) as error:
            self._fault(f"{motor.name}_can_send_failed")
            result.error = str(error)
            self.get_logger().error(f"{motor.name} C0 send failed: {error}")
        result.duration_ms = (time.monotonic() - started) * 1000.0
        return result

    def _send_mit(self, target_rad: float, guard=None) -> tuple[MotorSendResult, float]:
        result = MotorSendResult()
        started = time.monotonic()
        native = math.nan
        try:
            native = self._mit_zero_rad + self._steering.encoder_sign * self._steering.gear_ratio * target_rad
            payload = mit_motion_payload(native, self._mit_ranges, self._mit_kp, self._mit_kd)
            # Disabled by request: duplicate travel clamp at wire quantization.
            # Round inside travel limits; nearest-code rounding can exceed an endpoint.
            # half_span = self._steering.gear_ratio * self._steering_limit_rad
            # pmax = self._mit_ranges.position_rad
            # low_code = math.ceil((self._mit_zero_rad - half_span + pmax) * 65535 / (2 * pmax))
            # high_code = math.floor((self._mit_zero_rad + half_span + pmax) * 65535 / (2 * pmax))
            # code = max(low_code, min(high_code, int.from_bytes(payload[:2], "big")))
            # payload = code.to_bytes(2, "big") + payload[2:]
            # mit_motion_payload still rejects positions outside the F0 encoding range.
            self._mit_encoded_command = parse_mit_motion(payload, self._mit_ranges)
            if guard is not None and not guard():
                result.duration_ms = (time.monotonic() - started) * 1000.0
                return result, native
            if not self._dry_run:
                result.attempted = True
                self._send(0x400 | self._steering.host_id, payload)
                result.success = True
                sent_at = time.monotonic()
                self._mit_send_interval_ms = age_ms(sent_at, self._mit_last_sent_at)
                self._mit_last_sent_at = sent_at
                if not self._mit_sent_this_enable:
                    self._mit_first_sent_at = self._mit_last_sent_at
                self._mit_sent_this_enable = True
            self._slew_last_at = time.monotonic()
            self._next_mit_at = self._slew_last_at + MIT_DT_S
            self._outputs_active, self._disable_confirmed = True, False
        except (OSError, RuntimeError, ValueError, OverflowError, TypeError) as error:
            self._fault("steering_mit_send_failed")
            result.error = str(error)
            self.get_logger().error(f"steering MIT send failed: {error}")
        result.duration_ms = (time.monotonic() - started) * 1000.0
        return result, native

    def _disable_outputs(self, force: bool = False) -> DisableResult:
        result = DisableResult()
        now = time.monotonic()
        if self._disable_wait:
            if now >= self._disable_deadline_at:
                self._fault("disable_ack_timeout", hard=True)
            if now < self._next_disable_retry_at:
                return result
            if now < self._disable_deadline_at and all(value is not None for value in self._disable_wait.values()):
                return result
        elif not force and not self._outputs_active:
            return result
        self._mit_sent_this_enable = False
        self._next_mit_at = self._slew_last_at = 0.0
        if self._dry_run:
            self._outputs_active = False
            self._disable_confirmed = True
            return result
        if not self._disable_wait:
            self._disable_confirmed = False
            self._disable_wait = {"steering": None, "drive": None}
            self._disable_f1_wait.clear()
            self._disable_deadline_at = now + DISABLE_TIMEOUT_S
        if self._tx_bus is not None:
            result.attempted = True
            errors = []
            operations = []
            if ("steering" in self._disable_wait and
                    (self._disable_wait["steering"] is None or now >= self._disable_deadline_at)):
                operations.append((self._steering, "disable", bytes([0xCF])))
            if ("drive" in self._disable_wait and
                    (self._disable_wait["drive"] is None or now >= self._disable_deadline_at)):
                operations.extend(((self._drive, "zero", q_current_payload(0.0)),
                                   (self._drive, "disable", bytes([0xCF]))))
            for motor, operation, payload in operations:
                try:
                    sent_at = time.monotonic()
                    self._send(motor.host_id, payload)
                    setattr(result, f"{motor.name}_{operation}_success", True)
                    if operation == "disable" and self._disable_wait[motor.name] is None:
                        self._disable_wait[motor.name] = sent_at
                except (OSError, RuntimeError, ValueError) as error:
                    setattr(result, f"{motor.name}_{operation}_success", False)
                    errors.append(f"{motor.name}_{operation}: {error}")
            result.errors = tuple(errors)
            self._last_disable_errors = result.errors
            if errors:
                self._fault("disable_send_failed")
                self.get_logger().error("; ".join(errors))
        self._next_disable_retry_at = now + DISABLE_RETRY_S
        # Socket send success is not evidence that either motor disabled.
        self._outputs_active = bool(self._disable_wait)
        return result

    def _gate_state(self, now: float, requested_enable: bool, command_stamp: float,
                    sending: bool = False) -> tuple[dict, list[str]]:
        """Shared safety decision; active watchdog failures require a safe RC-down reset."""
        steering, drive = self._snapshot(self._steering), self._snapshot(self._drive)
        with self._steering.lock:
            steering_unwrapped = getattr(self._steering, "_unwrapped_rad", steering[0])
            steering_unwrapped_at = getattr(self._steering, "_unwrapped_at", steering[4])
        if sending:
            with self._command_lock:
                requested_enable = self._requested_enable
        now = max(now, time.monotonic())  # Sample after lock waits, not before them.
        steering_fresh = 0 < steering[2] <= now and now - steering[2] <= self._feedback_timeout_s
        drive_fresh = 0 < drive[2] <= now and now - drive[2] <= self._feedback_timeout_s
        steering_ready = steering_fresh and steering[3] is not None
        drive_ready = drive_fresh and drive[3] is not None
        feedback_ready = steering_ready and drive_ready
        self._feedback_gate_fault = requested_enable and (not feedback_ready or bool(self._disable_wait))
        command_ready = 0 < command_stamp <= now and now - command_stamp <= self._command_timeout_s
        if requested_enable and (sending or (self._outputs_active and not self._disable_wait)):
            if not command_ready:
                self._fault("command_timeout")
            elif not feedback_ready:
                self._fault("active_feedback_unavailable")
        if self._disable_wait and now >= self._disable_deadline_at:
            self._fault("disable_ack_timeout", hard=True)
        if (self._mit_sent_this_enable and
                now - max(self._mit_f1_at, self._mit_first_sent_at) > self._feedback_timeout_s):
            self._fault("steering_mit_feedback_timeout", hard=True)
        if (not sending and not self._hard_fault and feedback_ready
                and self._disable_confirmed
                and not self._outputs_active and not self._disable_wait and
                motor_fault_clear_allowed(requested_enable, steering_unwrapped, self._steering_fault_clear_rad)):
            self._fault_latched, self._fault_reason = False, None
        config_ready = self._mit_config_ready and self._mit_ranges is not None and self._mit_zero_rad is not None
        disable_ready = self._disable_confirmed or self._mit_sent_this_enable or (self._dry_run and self._outputs_active)
        enabled = (requested_enable and command_ready and feedback_ready and config_ready
                   and disable_ready and not self._disable_wait and not self._fault_latched)
        reasons = motor_disable_reasons(
            requested_enable, command_stamp > 0, command_ready, steering[3] is not None,
            drive[3] is not None, steering_fresh, drive_fresh, self._fault_latched,
        )
        if not config_ready:
            reasons.append("mit_preflight_incomplete")
        if self._disable_wait or not disable_ready:
            reasons.append("motor_disable_unconfirmed")
        diag = {
            "requested_enable": requested_enable, "gate_enabled": enabled,
            "command_age_ms": age_ms(now, command_stamp), "command_ready": command_ready,
            "steering_feedback_ready": steering_ready, "drive_feedback_ready": drive_ready,
            "steering_pair_age_ms": age_ms(now, steering[2]), "drive_pair_age_ms": age_ms(now, drive[2]),
            "steering_position_rad": steering[0], "steering_velocity_rad_s": steering[1],
            "drive_position_rad": drive[0], "drive_velocity_rad_s": drive[1],
            "mit_config_ready": config_ready, "mit_restart_required": self._hard_fault,
            "mit_f1_age_ms": age_ms(now, self._mit_f1_at),
            "mit_last_send_age_ms": age_ms(now, self._mit_last_sent_at),
            "mit_native_zero_rad": self._mit_zero_rad if self._mit_zero_rad is not None else math.nan,
            "steering_unwrapped_position_rad": steering_unwrapped,
            "steering_unwrapped_position_age_ms": age_ms(now, steering_unwrapped_at),
        }
        if self._mit_ranges is not None:
            diag.update(mit_position_max_rad=self._mit_ranges.position_rad,
                        mit_velocity_max_rad_s=self._mit_ranges.velocity_rad_s,
                        mit_torque_max_nm=self._mit_ranges.torque_nm)
        if self._mit_feedback is not None:
            diag.update(mit_feedback_position_motor_rad=self._mit_feedback.position_rad,
                        mit_feedback_velocity_motor_rad_s=self._mit_feedback.velocity_rad_s,
                        mit_feedback_torque_motor_nm=self._mit_feedback.torque_nm,
                        mit_feedback_active=self._mit_feedback.mode_active,
                        mit_feedback_fault=self._mit_feedback.system_fault)
        for name, sample in (("steering", steering), ("drive", drive)):
            fields = ("position_age_ms", "velocity_age_ms", "position_query_ms", "velocity_query_ms",
                      "position_query_failures", "velocity_query_failures")
            values = (age_ms(now, sample[4]), age_ms(now, sample[5]), *sample[6:])
            diag.update(zip((f"{name}_{field}" for field in fields), values))
        return diag, reasons

    def _motion_allowed(self, token: tuple[float, int], diag: dict, reasons: list[str]) -> bool:
        """Revalidate after encoding, immediately before each motor submission; no ROS work."""
        # Keep the original packet's age: a later policy update cannot refresh it.
        checked, current_reasons = self._gate_state(time.monotonic(), True, token[0], sending=True)
        enabled, withdrawn = self._requested_enable, self._disable_generation != token[1]
        if enabled and withdrawn:
            self._fault("enable_withdrawn_during_control")
        if not enabled or not self._running or withdrawn:
            checked["gate_enabled"] = False
            current_reasons.append("enable_not_requested" if not enabled else
                                   "mit_io_stopped" if not self._running else "enable_withdrawn_during_control")
        checked["requested_enable"] = enabled
        diag.update(checked)
        reasons[:] = list(dict.fromkeys(current_reasons))
        return bool(diag["gate_enabled"])

    def _control_tick(self, send_drive: bool = True) -> None:
        now = time.monotonic()
        self._control_sequence += 1
        interval_ms = age_ms(now, self._last_control_monotonic)
        self._last_control_monotonic = now
        with self._command_lock:
            requested_enable, command_stamp = self._requested_enable, self._command_monotonic
            token = (command_stamp, self._disable_generation)
            target_rad, target_drive_nm = self._steering_target_rad, self._drive_target_nm
            requested_steering, requested_drive = self._requested_steering_target_rad, self._requested_drive_target_nm
        if self._outputs_active and requested_enable and token[1] != self._active_disable_generation:
            self._fault("enable_withdrawn_during_control")
        diag, reasons = self._gate_state(now, requested_enable, command_stamp)
        diag.update(
            sequence=self._control_sequence, ros_time_s=self.get_clock().now().nanoseconds * 1e-9,
            monotonic_time_s=now, cycle_interval_ms=interval_ms, dry_run=self._dry_run,
            requested_steering_target_rad=requested_steering, requested_drive_target_nm=requested_drive,
            bounded_steering_target_rad=target_rad, bounded_drive_target_nm=target_drive_nm,
            steering_target_clipped=requested_steering != target_rad, drive_target_clipped=requested_drive != target_drive_nm,
        )
        guard = lambda: self._motion_allowed(token, diag, reasons)
        steer_send, drive_send, disable = MotorSendResult(), MotorSendResult(), DisableResult()
        if not diag["gate_enabled"]:
            self._slewed_steering_target_rad = diag["steering_position_rad"]
            disable = self._disable_outputs()
            self._feedback_gate_fault |= requested_enable and bool(self._disable_wait)
            diag["slewed_steering_target_rad"] = self._slewed_steering_target_rad
            self._publish_control_telemetry(now, diag, reasons, steer_send, drive_send, disable)
            return
        due = now + 1e-9 >= self._next_mit_at
        diag["mit_send_due"] = due
        diag["steering_slew_limited"] = False
        self._active_disable_generation = token[1]
        if due:
            # Already clipped/slewed at the 50-Hz producer. Repeat without a
            # second limiter or updating its state on each 200-Hz transmission.
            proposed = target_rad
            steer_send, native = self._send_mit(proposed, guard=guard)
            diag["mit_native_target_rad"] = native
            if diag["gate_enabled"] and (self._dry_run or steer_send.success):
                self._slewed_steering_target_rad = proposed
        diag["slewed_steering_target_rad"] = self._slewed_steering_target_rad
        diag["steering_error_rad"] = self._slewed_steering_target_rad - diag["steering_position_rad"]
        drive_damping = -DRIVE_DAMPING_NM_S_PER_RAD * diag["drive_velocity_rad_s"]
        drive_unclipped = target_drive_nm + drive_damping
        drive_torque = max(-DRIVE_EFFORT_LIMIT_NM, min(DRIVE_EFFORT_LIMIT_NM, drive_unclipped))
        diag.update(drive_damping_torque_nm=drive_damping, drive_unclipped_torque_nm=drive_unclipped,
                    drive_final_torque_nm=drive_torque, drive_torque_saturated=drive_torque != drive_unclipped)
        if diag["gate_enabled"] and not self._fault_latched and send_drive:
            drive_send = self._send_torque(self._drive, drive_torque, float(self._drive.torque_constant_nm_per_a), guard)
        if self._fault_latched or not diag["gate_enabled"]:
            diag["gate_enabled"] = False
            if self._fault_latched:
                reasons = list(dict.fromkeys(reasons + ["fault_latched", self._fault_reason or "send_failed"]))
            disable = self._disable_outputs(force=steer_send.attempted or drive_send.attempted)
        self._publish_control_telemetry(now, diag, reasons, steer_send, drive_send, disable)

    def _publish_control_telemetry(
        self, started: float, diag: dict, reasons: list[str],
        steer_send: MotorSendResult, drive_send: MotorSendResult, disable: DisableResult,
    ) -> None:
        if self._control_publish_gate.due(started):
            if self._mit_encoded_command is not None:
                command = self._mit_encoded_command
                diag.update(mit_encoded_position_motor_rad=command.position_rad,
                            mit_encoded_position_rad=self._steering.encoder_sign
                            * (command.position_rad - self._mit_zero_rad) / self._steering.gear_ratio,
                            mit_encoded_velocity_motor_rad_s=command.velocity_rad_s,
                            mit_encoded_kp_nm_per_motor_rad=command.kp_nm_per_rad,
                            mit_encoded_kd_nm_s_per_motor_rad=command.kd_nm_s_per_rad,
                            mit_encoded_feedforward_torque_motor_nm=command.feedforward_torque_nm,
                            mit_actual_send_interval_ms=self._mit_send_interval_ms)
            diag.update(self._event_details())
            for name, result in (("steering", steer_send), ("drive", drive_send)):
                diag.update(send_result_values(name, result))
            diag.update(outputs_active=self._outputs_active, fault_latched=self._fault_latched,
                        work_duration_ms=(time.monotonic() - started) * 1000.0, disable_attempted=disable.attempted)
            for name in ("steering_zero", "steering_disable", "drive_zero", "drive_disable"):
                value = getattr(disable, f"{name}_success")
                diag[f"{name}_send_success"] = math.nan if value is None else value
            message = Float64MultiArray()
            message.layout.dim = [self._control_dimension]
            message.data = telemetry_values(self._control_fields, diag)
            self._control_publisher.publish(message)
        self._publish_event(started, diag, reasons, disable)

    def _publish_event(self, started: float, diag: dict, reasons: list[str], disable: DisableResult) -> None:
        if disable.errors:
            reasons = list(dict.fromkeys(reasons + ["disable_send_failed"]))
        # Transition events observe every control tick, not the state-topic timer.
        details = self._event_details()
        event_state = (bool(diag["gate_enabled"]), self._outputs_active, self._fault_latched,
                       tuple(reasons), self._fault_reason, disable.errors, tuple(details.values()))
        if event_state != self._last_event_state:
            event = String()
            event.data = json.dumps({
                "sequence": self._control_sequence, "ros_time_s": diag["ros_time_s"],
                "monotonic_time_s": started, "gate_enabled": bool(diag["gate_enabled"]),
                "outputs_active": self._outputs_active, "dry_run": self._dry_run,
                "fault_latched": self._fault_latched, "fault_reason": self._fault_reason,
                "disable_reasons": reasons, "disable_send_errors": list(disable.errors),
                **details,
            }, sort_keys=True)
            self._event_publisher.publish(event)
            self._last_event_state = event_state

    def _event_details(self) -> dict:
        pending = bool(self._disable_wait)
        return {"disable_pending": pending, "disable_confirmed": self._disable_confirmed,
                **{f"{name}_disable_confirmed": self._disable_confirmed or (pending and name not in self._disable_wait)
                   for name in ("steering", "drive")}}

    def _publish_feedback(self) -> None:
        """100 Hz ROS output; actuation and fault detection never wait for this timer."""
        self._fault_publisher.publish(Bool(data=self._fault_latched or self._feedback_gate_fault))
        self._publish_state()
        self._publish_status()

    def _publish_state(self) -> None:
        steering, drive = self._snapshot(self._steering), self._snapshot(self._drive)
        now = time.monotonic()
        sample_times = (steering[2], drive[2])
        if any(stamp <= 0.0 or now - stamp > self._feedback_timeout_s for stamp in sample_times):
            return
        if any(stamp <= previous for stamp, previous in zip(sample_times, self._last_published_sample_times)):
            return
        torques = []
        for motor in (self._steering, self._drive):
            with motor.lock:
                torques.append(fresh_torque_feedback_nm(
                    motor.torque_nm, motor.torque_sample_monotonic, time.monotonic(), self._feedback_timeout_s,
                ))
        message = Float64MultiArray()
        message.layout.dim = [self._state_dimension]
        message.data = [steering[0], drive[0], steering[1], drive[1], time.time(), *torques]
        self._state_publisher.publish(message)
        self._last_published_sample_times = sample_times

    def _publish_status(self) -> None:
        steering, drive = self._snapshot(self._steering), self._snapshot(self._drive)
        now = time.monotonic()
        message = String()
        ages = {"steering_torque_age_ms": self._steering.torque_sample_monotonic,
                "drive_torque_age_ms": self._drive.torque_sample_monotonic, "mit_f1_age_ms": self._mit_f1_at}
        message.data = json.dumps({
            "outputs_active": self._outputs_active, "fault_latched": self._fault_latched,
            **self._event_details(),
            "motors_awaiting_disable": sorted(self._disable_wait),
            "fault_reason": self._fault_reason, "dry_run": self._dry_run,
            "steering_torque_constant_nm_per_a": steering[3],
            "drive_torque_constant_nm_per_a": drive[3],
            **{name: age_ms(now, stamp) if stamp > 0 else None for name, stamp in ages.items()},
            "torque_feedback_source": "measured A1/C0 Q-axis current times B0 torque constant",
            "mit_config_ready": self._mit_config_ready,
            "mit_restart_required": self._hard_fault,
            "last_disable_send_errors": list(getattr(self, "_last_disable_errors", ())),
            "mit_command_rate_hz": MIT_RATE_HZ,
            "policy_command_rate_hz": COMMAND_RATE_HZ,
            "state_query_rate_hz": STATE_RATE_HZ,
            "state_status_fault_rate_hz": STATE_RATE_HZ,
            "motor_control_rate_hz": MIT_RATE_HZ,
            "drive_command_rate_hz": self._control_rate_hz,
        }, sort_keys=True, allow_nan=False)
        self._status_publisher.publish(message)

    def destroy_node(self) -> None:
        self._running = False
        if hasattr(self, "_worker"):
            self._worker.join(timeout=1.0)
            if self._worker.is_alive():
                self.get_logger().error("MIT CAN worker did not stop within 1 s; inspect motor disable physically")
        Node.destroy_node(self)


def main(args=None) -> None:
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=args)
    node = None
    try:
        node = MotorMitIo()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
