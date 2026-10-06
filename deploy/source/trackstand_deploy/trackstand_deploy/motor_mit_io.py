"""Alternative deployment motor node: 50 Hz MIT steering and 200 Hz C0 rear torque.

One SocketCAN owner performs preflight, receives feedback, polls A3/A2/A1,
checks the 200 Hz safety gate, and sends commands. The ROS interface matches
motor_io; versioned motor_control telemetry exposes native MIT command/feedback data.
Firmware PD frequency and MIT current limiting are not established by the
local protocol; F0 ranges do not implement the previous host torque clamp.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import select
import socket
import struct
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray, MultiArrayDimension, String

from .can_protocol import (
    RADIANS_PER_COUNT,
    current_feedback_to_torque_nm,
    parse_a2_velocity,
    parse_a3_multi_turn_count,
    parse_a3_position,
    parse_b0_torque_constant,
    parse_current_feedback,
    q_current_payload,
)
from .contracts import (
    DRIVE_DAMPING_NM_S_PER_RAD,
    DRIVE_EFFORT_LIMIT_NM,
    DRIVE_VELOCITY_LIMIT_RAD_S,
    STEERING_KD_NM_S_PER_RAD,
    STEERING_KP_NM_PER_RAD,
    STEERING_VELOCITY_LIMIT_RAD_S,
    motor_fault_clear_allowed,
)
from .mit_protocol import (
    MitCommand, MitFeedback, MitRanges, f0_with_position, mit_motion_payload,
    parse_f0, parse_f1, parse_mit_motion,
)
from .motor_io import ChannelState, MotorIo
from .telemetry import (
    MOTOR_CONTROL_FIELDS as BASE_CONTROL_FIELDS,
    DisableResult,
    MotorSendResult,
    age_ms,
    motor_disable_reasons,
    send_result_values,
    telemetry_values,
)


MIT_RATE_HZ = 50.0
MIT_DT_S = 1.0 / MIT_RATE_HZ
MIT_POSITION_MAX_RAD = 3.0
QUERY_TIMEOUT_S = 0.002
DISABLE_TIMEOUT_S = 0.25
DISABLE_RETRY_S = 0.02
MAX_RX_BATCH = 64
CAN_FRAME = struct.Struct("=IB3x8s")
CAN_FILTER = struct.Struct("=II")
CAN_MASK = 0xC00007FF
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
class CanReply:
    can_id: int
    data: bytes
    local_tx: bool
    dropped: int
    received_monotonic: float


class SocketCan:
    """Sole receiver/sender; strict standard-frame and overflow checking."""

    def __init__(self, interface: str, steering_id: int, drive_id: int):
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
        except BaseException:
            self.socket.close()
            raise
        self._last_dropped = 0
        self._wall_offset_s = None
        self._clock_uncertainty_s = 0.0

    def send(self, can_id: int, payload: bytes) -> None:
        if not 0 <= can_id <= 0x7FF or not 1 <= len(payload) <= 8:
            raise ValueError("invalid standard CAN frame")
        sent = self.socket.send(CAN_FRAME.pack(can_id, len(payload), payload))
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


class MotorMitIo(MotorIo):
    """Reuse the established ROS command, state and measured-current contracts."""

    def __init__(self) -> None:
        # MotorIo.__init__ starts independent python-can pollers. MIT needs one
        # receiver so replies cannot be stolen or paired with the wrong query.
        Node.__init__(self, "trackstand_motor_io")
        defaults = {
            "can_interface": "can0", "steering_host_id": "102", "steering_feedback_id": "002",
            "drive_host_id": "101", "drive_feedback_id": "001", "steering_gear_ratio": 1.0,
            "drive_gear_ratio": 1.0, "steering_encoder_sign": -1.0, "drive_encoder_sign": 1.0,
            "steering_torque_sign": -1.0, "drive_torque_sign": -1.0,
            "poll_rate_hz": 200.0, "control_rate_hz": 200.0,
            "command_timeout_s": 0.08, "feedback_timeout_s": 0.05,
            "steering_limit_rad": math.pi / 2, "steering_fault_clear_rad": math.radians(5.0),
            "dry_run": False,
            "mit_kp_nm_per_motor_rad": STEERING_KP_NM_PER_RAD,
            "mit_kd_nm_s_per_motor_rad": STEERING_KD_NM_S_PER_RAD,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        parameter = lambda name: self.get_parameter(name).value
        self._can_interface = str(parameter("can_interface"))
        self._poll_rate_hz = float(parameter("poll_rate_hz"))
        self._control_rate_hz = float(parameter("control_rate_hz"))
        self._command_timeout_s = float(parameter("command_timeout_s"))
        self._feedback_timeout_s = float(parameter("feedback_timeout_s"))
        self._steering_limit_rad = float(parameter("steering_limit_rad"))
        self._steering_fault_clear_rad = float(parameter("steering_fault_clear_rad"))
        self._dry_run = bool(parameter("dry_run"))
        self._mit_kp = float(parameter("mit_kp_nm_per_motor_rad"))
        self._mit_kd = float(parameter("mit_kd_nm_s_per_motor_rad"))
        scalars = (self._poll_rate_hz, self._control_rate_hz, self._command_timeout_s,
                   self._feedback_timeout_s, self._steering_limit_rad, self._steering_fault_clear_rad,
                   self._mit_kp, self._mit_kd)
        if (not all(map(math.isfinite, scalars)) or self._poll_rate_hz <= 0 or self._control_rate_hz <= 0
                or self._command_timeout_s <= 0 or self._feedback_timeout_s <= 0
                or self._steering_limit_rad <= 0 or not 0 < self._steering_fault_clear_rad < self._steering_limit_rad
                or not 0 <= self._mit_kp <= 500 or not 0 <= self._mit_kd <= 5):
            raise ValueError("invalid motor rate, timeout, limit, or MIT gain")
        if self._control_rate_hz < MIT_RATE_HZ or self._poll_rate_hz < MIT_RATE_HZ:
            raise ValueError("control and encoder-poll rates must be at least 50 Hz")
        if self._feedback_timeout_s <= MIT_DT_S:
            raise ValueError("feedback timeout must exceed the 20 ms MIT reply interval")

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
        self._running = True
        self._requested_enable = False
        self._steering_target_rad = self._requested_steering_target_rad = self._slewed_steering_target_rad = 0.0
        self._drive_target_nm = self._requested_drive_target_nm = self._command_monotonic = 0.0
        self._fault_latched = self._hard_fault = self._outputs_active = False
        self._fault_reason = None
        self._command_lock = threading.Lock()
        self._last_published_sample_times = (0.0, 0.0)
        self._control_sequence = 0
        self._last_control_monotonic = 0.0
        self._last_event_state = None
        self._mit_ranges: MitRanges | None = None
        self._mit_zero_rad: float | None = None
        self._mit_feedback: MitFeedback | None = None
        self._mit_encoded_command: MitCommand | None = None
        self._mit_send_interval_ms = math.nan
        self._mit_f1_at = self._mit_last_sent_at = self._mit_first_sent_at = self._next_mit_at = 0.0
        self._mit_sent_this_enable = False
        self._mit_config_ready = False
        self._pending = None
        self._poll_jobs = deque()
        self._next_poll_at = self._next_b0_at = self._next_f0_at = 0.0
        self._last_steering_current_query_at = 0.0
        self._tx_bus: SocketCan | None = None
        self._last_disable_errors = ()
        self._disable_wait: dict[str, float | None] = {}
        self._disable_confirmed = False
        self._disable_deadline_at = self._next_disable_retry_at = 0.0

        self._state_publisher = self.create_publisher(Float64MultiArray, "/trackstand/motor_state", 10)
        self._torque_publisher = self.create_publisher(
            JointState,
            "/trackstand/motor_torque", QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self._control_publisher = self.create_publisher(
            Float64MultiArray, "/trackstand/motor_control",
            QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self._event_publisher = self.create_publisher(String, "/trackstand/motor_event", 100)
        self._fault_publisher = self.create_publisher(Bool, "/trackstand/motor_fault", 10)
        self._status_publisher = self.create_publisher(String, "/trackstand/motor_status", 10)
        self._control_dimension = MultiArrayDimension(
            label="motor_control_mit_v1:" + ",".join(MOTOR_CONTROL_FIELDS),
            size=len(MOTOR_CONTROL_FIELDS), stride=len(MOTOR_CONTROL_FIELDS),
        )
        self._command_subscription = self.create_subscription(
            Float64MultiArray, "/trackstand/motor_command", self._command_callback, 10,
        )
        self._publish_timer = self.create_timer(1.0 / self._poll_rate_hz, self._publish_state)
        self._status_timer = self.create_timer(0.5, self._publish_status)
        self._worker = threading.Thread(target=self._run_io, name="trackstand_mit_can", daemon=True)
        self._worker.start()
        self.get_logger().info(
            f"MIT steering 50 Hz, C0 drive {self._control_rate_hz:g} Hz on {self._can_interface}; "
            f"dry_run={self._dry_run}"
        )

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
                sample = JointState()
                sample.header.stamp = self.get_clock().now().to_msg()  # ROS receipt time, not motor sample time.
                sample.name = ["steering_joint" if motor is self._steering else "rear_wheel_joint"]
                sample.effort = [current_feedback_to_torque_nm(current_a, constant, motor.torque_sign)]
                self._torque_publisher.publish(sample)
                with motor.lock:
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
        elif motor is self._steering and opcode == 0xF1 and len(data) == 7 and self._mit_ranges is not None:
            feedback = parse_f1(data, self._mit_ranges)
            if stamp >= self._mit_f1_at:
                self._mit_feedback, self._mit_f1_at = feedback, stamp
            if feedback.system_fault:
                self._fault("steering_mit_system_fault", hard=True)
            elif (self._mit_sent_this_enable and stamp >= self._mit_first_sent_at
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
                if data[6] == 0:
                    del self._disable_wait[motor.name]
                    if not self._disable_wait:
                        self._disable_confirmed, self._outputs_active = True, False
                else:
                    self._fault(f"{motor.name}_disable_not_confirmed", hard=True)
                    self._disable_wait[motor.name] = None
            elif (self._mit_sent_this_enable and stamp >= self._mit_first_sent_at and data[6] == 0):
                reason = "steering_driver_left_mit_mode" if motor is self._steering else "drive_driver_unexpected_exit"
                self._fault(reason, hard=True)
        elif opcode in (0xA1, 0xA2, 0xA3, 0xB0, 0xF0, 0xF1, 0xCF, 0xAE):
            raise RuntimeError(f"malformed {motor.name} {opcode:02X} feedback")

        pending = self._pending
        if (pending is not None and pending[0] is motor and opcode == pending[1]
                and len(data) == pending[4] and pending[2] <= stamp <= pending[3]):
            self._pending = None

    def _drain_received(self) -> None:
        """Read queued faults before any actuation or new request; bound backlog."""
        for _ in range(MAX_RX_BATCH):
            frame = self._tx_bus.receive(0.0)
            if frame is None:
                return
            self._consume(frame, time.monotonic())
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
                frame = self._tx_bus.receive(max(0.0, deadline - time.monotonic()))
                if frame is None:
                    continue
                self._consume(frame, time.monotonic())
                if (sent_at <= frame.received_monotonic <= deadline and frame.can_id == motor.feedback_id
                        and len(frame.data) == expected_length and frame.data[0] == opcode):
                    return frame.data
            raise TimeoutError(f"{motor.name} {opcode:02X} startup reply missing or invalid")
        finally:
            self._pending = None

    def _startup(self) -> None:
        steer, drive = self._steering, self._drive
        if not self._dry_run:
            # CF exits MIT; B1 changes the persistent native origin exactly once
            # per process start, never on later RC up/down transitions.
            for motor in (steer, drive):
                self._send(motor.host_id, q_current_payload(0.0))
                status = self._query(motor, 0xCF, 8, 0.25)
                if status[6] or status[7]:
                    raise RuntimeError(f"{motor.name} CF did not confirm disabled/fault-free mode")
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
        if now >= self._next_poll_at and not self._poll_jobs:
            self._poll_jobs.extend(((self._steering, 0xA3, 7), (self._steering, 0xA2, 5),
                                    (self._drive, 0xA3, 7), (self._drive, 0xA2, 5)))
            if now - self._last_steering_current_query_at >= MIT_DT_S:
                self._poll_jobs.append((self._steering, 0xA1, 5))
                self._last_steering_current_query_at = now
            if now >= self._next_b0_at:
                for motor in (self._steering, self._drive):
                    if motor.torque_constant_nm_per_a is None:
                        self._poll_jobs.append((motor, 0xB0, 7))
                self._next_b0_at = now + 1.0
            if now >= self._next_f0_at:
                self._poll_jobs.append((self._steering, 0xF0, 7))
                self._next_f0_at = now + 1.0
            self._next_poll_at = now + 1.0 / self._poll_rate_hz
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
                drive_due = now >= next_control
                mit_due = self._outputs_active and not self._disable_wait and now >= self._next_mit_at
                if drive_due or mit_due:
                    self._control_tick(send_drive=drive_due)
                    if drive_due:
                        next_control += 1.0 / self._control_rate_hz
                        if next_control <= time.monotonic():
                            next_control = time.monotonic() + 1.0 / self._control_rate_hz
                self._schedule_polls(time.monotonic())
                deadline = next_control
                if self._outputs_active and not self._disable_wait:
                    deadline = min(deadline, self._next_mit_at)
                if self._pending is not None:
                    deadline = min(deadline, self._pending[3])
                elif not self._poll_jobs:
                    deadline = min(deadline, self._next_poll_at)
                frame = self._tx_bus.receive(min(0.005, max(0.0, deadline - time.monotonic())))
                if frame is not None:
                    self._consume(frame, time.monotonic())
        except Exception as error:
            self._fault(f"mit_io_error:{type(error).__name__}:{error}", hard=True)
            self.get_logger().error(f"MIT motor I/O stopped: {error}")
            self._fault_publisher.publish(Bool(data=True))
        finally:
            if self._tx_bus is not None:
                try:
                    self._disable_outputs(force=True)
                    deadline = time.monotonic() + DISABLE_TIMEOUT_S
                    while self._disable_wait and time.monotonic() < deadline:
                        frame = self._tx_bus.receive(min(0.01, max(0.0, deadline - time.monotonic())))
                        if frame is not None:
                            self._consume(frame, time.monotonic())
                        self._disable_outputs()
                    if self._disable_wait:
                        self._fault("disable_ack_timeout", hard=True)
                        self.get_logger().error("Motor disable unconfirmed; inspect motor state physically")
                except Exception as error:
                    self._fault("shutdown_disable_unconfirmed", hard=True)
                    self.get_logger().error(f"Motor disable/shutdown failed: {error}")
                finally:
                    self._tx_bus.close()
                    self._tx_bus = None

    def _send_torque(self, motor: ChannelState, torque_nm: float, torque_constant: float) -> MotorSendResult:
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
            if not self._dry_run:
                result.attempted = True
                self._send(motor.host_id, payload)
                result.success = True
        except (OSError, RuntimeError, ValueError, OverflowError) as error:
            self._fault(f"{motor.name}_can_send_failed")
            result.error = str(error)
            self.get_logger().error(f"{motor.name} C0 send failed: {error}")
        result.duration_ms = (time.monotonic() - started) * 1000.0
        return result

    def _send_mit(self, target_rad: float) -> tuple[MotorSendResult, float]:
        result = MotorSendResult()
        started = time.monotonic()
        native = math.nan
        try:
            native = self._mit_zero_rad + self._steering.encoder_sign * self._steering.gear_ratio * target_rad
            payload = mit_motion_payload(native, self._mit_ranges, self._mit_kp, self._mit_kd)
            # Round INSIDE the joint travel range. Nearest-code rounding alone
            # can put an endpoint command a fraction of a code beyond +/-90 deg.
            half_span = self._steering.gear_ratio * self._steering_limit_rad
            pmax = self._mit_ranges.position_rad
            low_code = math.ceil((self._mit_zero_rad - half_span + pmax) * 65535 / (2 * pmax))
            high_code = math.floor((self._mit_zero_rad + half_span + pmax) * 65535 / (2 * pmax))
            code = max(low_code, min(high_code, int.from_bytes(payload[:2], "big")))
            payload = code.to_bytes(2, "big") + payload[2:]
            self._mit_encoded_command = parse_mit_motion(payload, self._mit_ranges)
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
                self._next_mit_at = self._mit_last_sent_at + MIT_DT_S
            else:
                self._next_mit_at = time.monotonic() + MIT_DT_S
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
        self._next_mit_at = 0.0
        if self._dry_run:
            self._outputs_active = False
            self._disable_confirmed = True
            return result
        if not self._disable_wait:
            self._disable_confirmed = False
            self._disable_wait = {"steering": None, "drive": None}
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

    def _control_tick(self, send_drive: bool = True) -> None:
        now = time.monotonic()
        self._control_sequence += 1
        interval_ms = age_ms(now, self._last_control_monotonic)
        self._last_control_monotonic = now
        steering, drive = self._snapshot(self._steering), self._snapshot(self._drive)
        with self._command_lock:
            requested_enable = self._requested_enable
            command_stamp = self._command_monotonic
            target_rad = self._steering_target_rad
            target_drive_nm = self._drive_target_nm
            requested_steering = self._requested_steering_target_rad
            requested_drive = self._requested_drive_target_nm
        steering_fresh = steering[2] > 0 and now - steering[2] <= self._feedback_timeout_s
        drive_fresh = drive[2] > 0 and now - drive[2] <= self._feedback_timeout_s
        steering_ready = steering_fresh and steering[3] is not None
        drive_ready = drive_fresh and drive[3] is not None
        feedback_ready = steering_ready and drive_ready
        command_ready = command_stamp > 0 and now - command_stamp <= self._command_timeout_s
        with self._steering.lock:
            steering_unwrapped = getattr(self._steering, "_unwrapped_rad", steering[0])
            steering_unwrapped_at = getattr(self._steering, "_unwrapped_at", steering[4])
        limit_reason = None
        if abs(steering_unwrapped) > self._steering_limit_rad:
            limit_reason = "steering_position_limit"
        elif abs(steering[1]) > STEERING_VELOCITY_LIMIT_RAD_S * 1.1:
            limit_reason = "steering_speed_limit"
        elif abs(drive[1]) > DRIVE_VELOCITY_LIMIT_RAD_S * 1.1:
            limit_reason = "drive_speed_limit"
        if limit_reason is not None:
            self._fault(limit_reason)
        if self._disable_wait and now >= self._disable_deadline_at:
            self._fault("disable_ack_timeout", hard=True)
        if (self._mit_sent_this_enable and
                now - max(self._mit_f1_at, self._mit_first_sent_at) > self._feedback_timeout_s):
            self._fault("steering_mit_feedback_timeout", hard=True)
        if (not self._hard_fault and feedback_ready and limit_reason is None
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
            "sequence": self._control_sequence, "ros_time_s": self.get_clock().now().nanoseconds * 1e-9,
            "monotonic_time_s": now, "cycle_interval_ms": interval_ms, "dry_run": self._dry_run,
            "requested_enable": requested_enable, "gate_enabled": enabled,
            "command_age_ms": age_ms(now, command_stamp), "command_ready": command_ready,
            "steering_feedback_ready": steering_ready, "drive_feedback_ready": drive_ready,
            "steering_pair_age_ms": age_ms(now, steering[2]), "drive_pair_age_ms": age_ms(now, drive[2]),
            "steering_position_rad": steering[0], "steering_velocity_rad_s": steering[1],
            "drive_position_rad": drive[0], "drive_velocity_rad_s": drive[1],
            "requested_steering_target_rad": requested_steering, "requested_drive_target_nm": requested_drive,
            "bounded_steering_target_rad": target_rad, "bounded_drive_target_nm": target_drive_nm,
            "steering_target_clipped": requested_steering != target_rad,
            "drive_target_clipped": requested_drive != target_drive_nm,
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
            diag.update({
                f"{name}_position_age_ms": age_ms(now, sample[4]),
                f"{name}_velocity_age_ms": age_ms(now, sample[5]),
                f"{name}_position_query_ms": sample[6],
                f"{name}_velocity_query_ms": sample[7],
                f"{name}_position_query_failures": sample[8],
                f"{name}_velocity_query_failures": sample[9],
            })
        steer_send, drive_send, disable = MotorSendResult(), MotorSendResult(), DisableResult()
        if not enabled:
            self._slewed_steering_target_rad = steering[0]
            disable = self._disable_outputs()
            self._fault_publisher.publish(Bool(data=self._fault_latched or
                (requested_enable and (not feedback_ready or bool(self._disable_wait)))))
            diag["slewed_steering_target_rad"] = self._slewed_steering_target_rad
            self._publish_control_telemetry(now, diag, reasons, steer_send, drive_send, disable)
            return
        due = now + 1e-9 >= self._next_mit_at
        diag["mit_send_due"] = due
        if due:
            elapsed = MIT_DT_S if not self._mit_sent_this_enable else max(0.0, now - self._mit_last_sent_at)
            max_step = STEERING_VELOCITY_LIMIT_RAD_S * min(elapsed, 2 * MIT_DT_S)
            delta = target_rad - self._slewed_steering_target_rad
            proposed = self._slewed_steering_target_rad + max(-max_step, min(max_step, delta))
            diag["steering_slew_limited"] = abs(delta) > max_step
            steer_send, native = self._send_mit(proposed)
            diag["mit_native_target_rad"] = native
            if self._dry_run or steer_send.success:
                self._slewed_steering_target_rad = proposed
        diag["slewed_steering_target_rad"] = self._slewed_steering_target_rad
        diag["steering_error_rad"] = self._slewed_steering_target_rad - steering[0]
        drive_damping = -DRIVE_DAMPING_NM_S_PER_RAD * drive[1]
        drive_unclipped = target_drive_nm + drive_damping
        drive_torque = max(-DRIVE_EFFORT_LIMIT_NM, min(DRIVE_EFFORT_LIMIT_NM, drive_unclipped))
        diag.update(drive_damping_torque_nm=drive_damping, drive_unclipped_torque_nm=drive_unclipped,
                    drive_final_torque_nm=drive_torque, drive_torque_saturated=drive_torque != drive_unclipped)
        if not self._fault_latched and send_drive:
            drive_send = self._send_torque(self._drive, drive_torque, float(drive[3]))
        if self._fault_latched:
            diag["gate_enabled"] = False
            reasons = ["fault_latched", self._fault_reason or "send_failed"]
            disable = self._disable_outputs(force=True)
            self._fault_publisher.publish(Bool(data=True))
        else:
            self._outputs_active = True
            self._disable_confirmed = False
            self._fault_publisher.publish(Bool(data=False))
        self._publish_control_telemetry(now, diag, reasons, steer_send, drive_send, disable)

    def _publish_control_telemetry(
        self, started: float, diag: dict, reasons: list[str],
        steer_send: MotorSendResult, drive_send: MotorSendResult, disable: DisableResult,
    ) -> None:
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
        diag.update(send_result_values("steering", steer_send))
        diag.update(send_result_values("drive", drive_send))
        diag.update({
            "outputs_active": self._outputs_active, "fault_latched": self._fault_latched,
            "disable_pending": bool(self._disable_wait), "disable_confirmed": self._disable_confirmed,
            "steering_disable_confirmed": self._disable_confirmed or
            (bool(self._disable_wait) and "steering" not in self._disable_wait),
            "drive_disable_confirmed": self._disable_confirmed or
            (bool(self._disable_wait) and "drive" not in self._disable_wait),
            "work_duration_ms": (time.monotonic() - started) * 1000.0,
            "disable_attempted": disable.attempted,
            "steering_zero_send_success": math.nan,
            "steering_disable_send_success": math.nan if disable.steering_disable_success is None
            else disable.steering_disable_success,
            "drive_zero_send_success": math.nan if disable.drive_zero_success is None else disable.drive_zero_success,
            "drive_disable_send_success": math.nan if disable.drive_disable_success is None
            else disable.drive_disable_success,
        })
        if disable.errors:
            reasons = list(dict.fromkeys(reasons + ["disable_send_failed"]))
        message = Float64MultiArray()
        message.layout.dim = [self._control_dimension]
        message.data = telemetry_values(MOTOR_CONTROL_FIELDS, diag)
        self._control_publisher.publish(message)
        event_state = (bool(diag["gate_enabled"]), self._outputs_active, self._fault_latched,
                       tuple(reasons), self._fault_reason, disable.errors)
        if event_state != self._last_event_state:
            event = String()
            event.data = json.dumps({
                "sequence": self._control_sequence, "ros_time_s": diag["ros_time_s"],
                "monotonic_time_s": started, "gate_enabled": bool(diag["gate_enabled"]),
                "outputs_active": self._outputs_active, "dry_run": self._dry_run,
                "disable_pending": bool(self._disable_wait), "disable_confirmed": self._disable_confirmed,
                "fault_latched": self._fault_latched, "fault_reason": self._fault_reason,
                "disable_reasons": reasons, "disable_send_errors": list(disable.errors),
            }, sort_keys=True)
            self._event_publisher.publish(event)
            self._last_event_state = event_state

    def _publish_status(self) -> None:
        steering, drive = self._snapshot(self._steering), self._snapshot(self._drive)
        now = time.monotonic()
        message = String()
        steering_torque_age = age_ms(now, self._steering.torque_sample_monotonic)
        drive_torque_age = age_ms(now, self._drive.torque_sample_monotonic)
        mit_f1_age = age_ms(now, self._mit_f1_at)
        message.data = json.dumps({
            "outputs_active": self._outputs_active, "fault_latched": self._fault_latched,
            "disable_pending": bool(self._disable_wait), "disable_confirmed": self._disable_confirmed,
            "motors_awaiting_disable": sorted(self._disable_wait),
            "fault_reason": self._fault_reason, "dry_run": self._dry_run,
            "steering_torque_constant_nm_per_a": steering[3],
            "drive_torque_constant_nm_per_a": drive[3],
            "steering_torque_age_ms": steering_torque_age if math.isfinite(steering_torque_age) else None,
            "drive_torque_age_ms": drive_torque_age if math.isfinite(drive_torque_age) else None,
            "torque_feedback_source": "measured A1/C0 Q-axis current times B0 torque constant",
            "mit_config_ready": self._mit_config_ready,
            "mit_restart_required": self._hard_fault,
            "last_disable_send_errors": list(getattr(self, "_last_disable_errors", ())),
            "mit_command_rate_hz": MIT_RATE_HZ,
            "mit_f1_age_ms": mit_f1_age if math.isfinite(mit_f1_age) else None,
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
