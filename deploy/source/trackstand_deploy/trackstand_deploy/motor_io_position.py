"""200 Hz GIM I/O with firmware C2 steering position control and C0 rear torque.

This is a separate alternative to motor_io, not a replacement for it. The ROS
steering target remains an absolute, startup-zero-relative joint angle. C2
payloads add the converted target to the latched raw A3 startup zero; current
position is used for safety and observations, not this target conversion.
Native increasing steering position turns left, as verified by the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import struct
import threading
import time

import can
import numpy as np
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
    query,
)
from .contracts import (
    DRIVE_DAMPING_NM_S_PER_RAD,
    DRIVE_EFFORT_LIMIT_NM,
    DRIVE_VELOCITY_LIMIT_RAD_S,
    STEERING_EFFORT_LIMIT_NM,
    STEERING_VELOCITY_LIMIT_RAD_S,
    motor_fault_clear_allowed,
)
from .telemetry import (
    MOTOR_CONTROL_FIELDS as MOTOR_CURRENT_CONTROL_FIELDS,
    DisableResult,
    MotorSendResult,
    age_ms,
    motor_disable_reasons,
    send_result_values,
    telemetry_values,
)


MOTOR_CONTROL_FIELDS = MOTOR_CURRENT_CONTROL_FIELDS + (
    "gate_monotonic_time_s",
    "steering_position_command_sign",
    "steering_position_limits_configured", "steering_position_speed_limit_raw",
    "steering_position_current_limit_raw", "steering_c2_zero_count",
    "steering_c2_target_count", "steering_c2_target_offset_count", "steering_c2_encoded_target_rad",
    "steering_current_query_send_attempted", "steering_current_query_send_success",
    "steering_current_query_send_duration_ms",
)


def absolute_position_payload(
    target_rad: float, zero_count: int, gear_ratio: float, command_sign: float, steering_limit_rad: float,
) -> bytes:
    """Encode C2 from a startup-relative angle, without a live-position input."""
    if not math.isfinite(target_rad) or not math.isfinite(gear_ratio) or gear_ratio <= 0.0:
        raise ValueError("target angle must be finite and gear ratio positive and finite")
    if command_sign not in (-1.0, 1.0):
        raise ValueError("C2 position-command sign must be explicitly set to +/-1")
    if not isinstance(zero_count, int) or isinstance(zero_count, bool) or not -(2**31) <= zero_count < 2**31:
        raise ValueError("startup zero must be an A3 signed int32 encoder count")
    if not math.isfinite(steering_limit_rad) or steering_limit_rad <= 0.0:
        raise ValueError("steering angle limit must be positive and finite")
    bounded_target = max(-steering_limit_rad, min(steering_limit_rad, target_rad))
    offset_float = command_sign * bounded_target * gear_ratio / RADIANS_PER_COUNT
    limit_float = steering_limit_rad * gear_ratio / RADIANS_PER_COUNT
    if not math.isfinite(offset_float) or not math.isfinite(limit_float):
        raise OverflowError("absolute position does not fit C2")
    # Round within the angle limit, not across it for noninteger gear ratios.
    max_offset = math.floor(limit_float)
    offset = max(-max_offset, min(max_offset, int(round(offset_float))))
    raw = zero_count + offset
    if not -(2**31) <= raw < 2**31:
        raise OverflowError("absolute position does not fit C2")
    return bytes([0xC2]) + struct.pack("<i", raw)


def position_mode_limits(gear_ratio: float, torque_constant: float) -> tuple[int, int]:
    """B2: unsigned 0.01 rpm; B3: unsigned 0.001 A, rounded DOWN."""
    if not math.isfinite(gear_ratio) or gear_ratio <= 0.0:
        raise ValueError("gear ratio must be positive and finite")
    if not math.isfinite(torque_constant) or torque_constant <= 0.0:
        raise ValueError("torque constant must be positive and finite")
    speed_raw = math.floor(STEERING_VELOCITY_LIMIT_RAD_S * gear_ratio * 60.0 / (2.0 * math.pi) * 100.0)
    current_raw = math.floor(STEERING_EFFORT_LIMIT_NM / torque_constant * 1000.0)
    if not all(0 < value < 2**32 for value in (speed_raw, current_raw)):
        raise OverflowError("position-mode limits must fit nonzero B2/B3 uint32 fields")
    return speed_raw, current_raw


@dataclass
class ChannelState:
    name: str
    host_id: int
    feedback_id: int
    gear_ratio: float
    encoder_sign: float
    torque_sign: float
    zero_count: int | None = None
    multi_turn_count: int | None = None
    position_rad: float = 0.0
    velocity_rad_s: float = 0.0
    torque_constant_nm_per_a: float | None = None
    sample_monotonic: float = 0.0
    torque_sample_monotonic: float = 0.0
    position_sample_monotonic: float = 0.0
    velocity_sample_monotonic: float = 0.0
    position_query_ms: float = math.nan
    velocity_query_ms: float = math.nan
    position_query_failures: int = 0
    velocity_query_failures: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class MotorIoPosition(Node):
    def __init__(self) -> None:
        super().__init__("trackstand_motor_io")
        self.declare_parameter("can_interface", "can0")
        self.declare_parameter("steering_host_id", "102")
        self.declare_parameter("steering_feedback_id", "002")
        self.declare_parameter("drive_host_id", "101")
        self.declare_parameter("drive_feedback_id", "001")
        self.declare_parameter("steering_gear_ratio", 1.0)
        self.declare_parameter("drive_gear_ratio", 1.0)
        self.declare_parameter("steering_encoder_sign", -1.0)
        self.declare_parameter("drive_encoder_sign", 1.0)
        self.declare_parameter("steering_torque_sign", -1.0)
        self.declare_parameter("drive_torque_sign", -1.0)
        self.declare_parameter("poll_rate_hz", 200.0)
        self.declare_parameter("control_rate_hz", 200.0)
        self.declare_parameter("command_timeout_s", 0.08)
        self.declare_parameter("feedback_timeout_s", 0.05)
        self.declare_parameter("steering_limit_rad", math.radians(90.0))
        self.declare_parameter("steering_fault_clear_rad", math.radians(5.0))
        self.declare_parameter("dry_run", False)

        self._can_interface = str(self.get_parameter("can_interface").value)
        self._poll_rate_hz = float(self.get_parameter("poll_rate_hz").value)
        self._control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self._command_timeout_s = float(self.get_parameter("command_timeout_s").value)
        self._feedback_timeout_s = float(self.get_parameter("feedback_timeout_s").value)
        self._steering_limit_rad = float(self.get_parameter("steering_limit_rad").value)
        self._steering_fault_clear_rad = float(self.get_parameter("steering_fault_clear_rad").value)
        self._dry_run = bool(self.get_parameter("dry_run").value)
        self._position_limits_configured = False
        self._position_speed_limit_raw = None
        self._position_current_limit_raw = None
        self._position_limits_retry_at = 0.0
        if (
            not all(math.isfinite(value) for value in (
                self._poll_rate_hz, self._control_rate_hz, self._command_timeout_s,
                self._feedback_timeout_s, self._steering_limit_rad, self._steering_fault_clear_rad,
            ))
            or min(self._poll_rate_hz, self._control_rate_hz) <= 0.0
            or min(self._command_timeout_s, self._feedback_timeout_s) <= 0.0
            or self._steering_limit_rad <= 0.0
            or not 0.0 < self._steering_fault_clear_rad < self._steering_limit_rad
        ):
            raise ValueError(
                "motor rates/timeouts must be positive; fault-clear angle must be within steering limit"
            )

        def channel(prefix: str, name: str) -> ChannelState:
            return ChannelState(
                name=name,
                host_id=int(str(self.get_parameter(f"{prefix}_host_id").value), 16),
                feedback_id=int(str(self.get_parameter(f"{prefix}_feedback_id").value), 16),
                gear_ratio=float(self.get_parameter(f"{prefix}_gear_ratio").value),
                encoder_sign=float(self.get_parameter(f"{prefix}_encoder_sign").value),
                torque_sign=float(self.get_parameter(f"{prefix}_torque_sign").value),
            )

        self._steering = channel("steering", "steering")
        self._drive = channel("drive", "drive")
        # Native increasing steering position turns left (rider's view), while
        # policy-positive steering turns right. A3 and C2 use native counts.
        self._position_command_sign = self._steering.encoder_sign
        for motor in (self._steering, self._drive):
            if not math.isfinite(motor.gear_ratio) or motor.gear_ratio <= 0.0 or motor.encoder_sign not in (-1.0, 1.0):
                raise ValueError(f"invalid encoder configuration for {motor.name}")
            if motor.torque_sign not in (-1.0, 1.0):
                raise ValueError(f"invalid torque sign for {motor.name}")

        self._running = True
        self._requested_enable = False
        self._steering_target_rad = 0.0
        self._requested_steering_target_rad = 0.0
        self._slewed_steering_target_rad = 0.0
        self._drive_target_nm = 0.0
        self._requested_drive_target_nm = 0.0
        self._command_monotonic = 0.0
        self._fault_latched = False
        self._fault_reason: str | None = None
        self._outputs_active = False
        self._command_lock = threading.Lock()
        self._last_published_sample_times = (0.0, 0.0)
        self._control_sequence = 0
        self._last_control_monotonic = 0.0
        self._last_event_state = None

        self._tx_bus = can.interface.Bus(
            channel=self._can_interface, interface="socketcan", receive_own_messages=False
        )
        self._state_publisher = self.create_publisher(
            Float64MultiArray, "/trackstand/motor_state", 10
        )
        self._torque_publisher = self.create_publisher(
            JointState, "/trackstand/motor_torque",
            QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self._control_publisher = self.create_publisher(
            Float64MultiArray, "/trackstand/motor_control",
            QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self._event_publisher = self.create_publisher(String, "/trackstand/motor_event", 100)
        self._control_dimension = MultiArrayDimension(
            label="motor_control_position_v2:" + ",".join(MOTOR_CONTROL_FIELDS),
            size=len(MOTOR_CONTROL_FIELDS), stride=len(MOTOR_CONTROL_FIELDS),
        )
        self._fault_publisher = self.create_publisher(Bool, "/trackstand/motor_fault", 10)
        self._status_publisher = self.create_publisher(String, "/trackstand/motor_status", 10)
        self._command_subscription = self.create_subscription(
            Float64MultiArray, "/trackstand/motor_command", self._command_callback, 10
        )

        # A separate receive-only socket sees feedback without consuming the
        # encoder pollers' replies or adding CAN traffic/control-loop waits.
        self._torque_bus = can.interface.Bus(
            channel=self._can_interface,
            interface="socketcan",
            receive_own_messages=False,
            can_filters=[
                {"can_id": motor.feedback_id, "can_mask": 0x7FF, "extended": False}
                for motor in (self._steering, self._drive)
            ],
        )
        self._threads = [threading.Thread(target=self._receive_torque_feedback, daemon=True)] + [
            threading.Thread(target=self._poll_motor, args=(motor,), daemon=True)
            for motor in (self._steering, self._drive)
        ]
        for thread in self._threads:
            thread.start()
        self._control_timer = self.create_timer(1.0 / self._control_rate_hz, self._control_tick)
        self._publish_timer = self.create_timer(1.0 / self._poll_rate_hz, self._publish_state)
        self._status_timer = self.create_timer(0.5, self._publish_status)
        self._disable_outputs(force=True)
        self.get_logger().info(
            f"C2 position motor I/O on {self._can_interface}: poll={self._poll_rate_hz:.0f} Hz, "
            f"control={self._control_rate_hz:.0f} Hz, dry_run={self._dry_run}"
        )
        self.get_logger().info(
            f"C2 command sign={self._position_command_sign:+.0f}, following steering_encoder_sign; "
            "startup A3 zero + converted absolute target. Run only one motor I/O node."
        )


    def _receive_torque_feedback(self) -> None:
        while self._running:
            try:
                message = self._torque_bus.recv(timeout=0.05)
            except can.CanError as error:
                self.get_logger().error(f"Torque feedback listener stopped: {error}")
                return
            if message is not None:
                self._publish_torque_feedback(message)

    def _publish_torque_feedback(self, message: can.Message) -> None:
        if message.is_extended_id or message.is_remote_frame or message.is_error_frame:
            return
        if not message.data or message.data[0] not in (0xA1, 0xC0):
            return
        if message.arbitration_id == self._steering.feedback_id:
            motor, joint_name = self._steering, "steering_joint"
        elif message.arbitration_id == self._drive.feedback_id:
            motor, joint_name = self._drive, "rear_wheel_joint"
        else:
            return
        # ROS receipt time, not a hardware acquisition timestamp. Each message
        # contains one motor so asynchronous samples are not paired or repeated.
        receipt_stamp = self.get_clock().now().to_msg()
        try:
            current_a = parse_current_feedback(bytes(message.data))
            with motor.lock:
                torque_constant = motor.torque_constant_nm_per_a
            if torque_constant is None:
                return  # No substituted constant and no fabricated zero torque.
            torque_nm = current_feedback_to_torque_nm(current_a, torque_constant, motor.torque_sign)
        except ValueError as error:
            self.get_logger().warning(f"Invalid {motor.name} torque feedback: {error}", throttle_duration_sec=2.0)
            return
        sample = JointState()
        sample.header.stamp = receipt_stamp
        sample.name = [joint_name]
        sample.effort = [torque_nm]
        self._torque_publisher.publish(sample)
        with motor.lock:
            motor.torque_sample_monotonic = time.monotonic()

    def _poll_motor(self, motor: ChannelState) -> None:
        bus = can.interface.Bus(
            channel=self._can_interface,
            interface="socketcan",
            receive_own_messages=False,
            can_filters=[{"can_id": motor.feedback_id, "can_mask": 0x7FF, "extended": False}],
        )
        request_b0 = can.Message(
            arbitration_id=motor.host_id, data=bytes([0xB0]), is_extended_id=False
        )
        request_a2 = can.Message(
            arbitration_id=motor.host_id, data=bytes([0xA2]), is_extended_id=False
        )
        request_a3 = can.Message(
            arbitration_id=motor.host_id, data=bytes([0xA3]), is_extended_id=False
        )
        period_s = 1.0 / self._poll_rate_hz
        deadline = time.monotonic()
        retry_b0_at = 0.0
        try:
            while self._running:
                now = time.monotonic()
                if motor.torque_constant_nm_per_a is None and now >= retry_b0_at:
                    payload = query(bus, request_b0, motor.feedback_id, max(0.002, period_s * 0.8))
                    if payload is not None:
                        try:
                            torque_constant = parse_b0_torque_constant(payload)
                            with motor.lock:
                                motor.torque_constant_nm_per_a = torque_constant
                            self.get_logger().info(
                                f"{motor.name} B0 torque constant={torque_constant:.6g} N m/A"
                            )
                        except ValueError as error:
                            self.get_logger().error(str(error), throttle_duration_sec=2.0)
                    retry_b0_at = now + 1.0

                position_started = time.monotonic()
                position_payload = query(
                    bus, request_a3, motor.feedback_id, max(0.001, period_s * 0.35)
                )
                position_received = time.monotonic()
                velocity_payload = query(
                    bus, request_a2, motor.feedback_id, max(0.001, period_s * 0.35)
                )
                velocity_received = time.monotonic()
                with motor.lock:
                    motor.position_query_ms = (position_received - position_started) * 1000.0
                    motor.velocity_query_ms = (velocity_received - position_received) * 1000.0
                    motor.position_query_failures += int(position_payload is None)
                    motor.velocity_query_failures += int(velocity_payload is None)
                if position_payload is not None and velocity_payload is not None:
                    try:
                        encoder_count = parse_a3_multi_turn_count(position_payload)
                        with motor.lock:
                            if motor.zero_count is None:
                                motor.zero_count = encoder_count
                                self.get_logger().info(
                                    f"{motor.name} encoder zero={encoder_count} counts"
                                )
                            motor.multi_turn_count = encoder_count
                            motor.position_rad = parse_a3_position(
                                position_payload,
                                motor.zero_count,
                                motor.gear_ratio,
                                motor.encoder_sign,
                            )
                            motor.velocity_rad_s = parse_a2_velocity(
                                velocity_payload, motor.gear_ratio, motor.encoder_sign
                            )
                            motor.sample_monotonic = time.monotonic()
                            motor.position_sample_monotonic = position_received
                            motor.velocity_sample_monotonic = velocity_received
                    except (ValueError, IndexError) as error:
                        self.get_logger().warning(
                            f"Invalid {motor.name} A2/A3 response: {error}",
                            throttle_duration_sec=2.0,
                        )

                deadline += period_s
                sleep_s = deadline - time.monotonic()
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
                else:
                    deadline = time.monotonic()
        finally:
            bus.shutdown()

    def _command_callback(self, message: Float64MultiArray) -> None:
        if len(message.data) != 3:
            self.get_logger().warning("motor_command must be [enable, steering_target_rad, drive_torque_nm]")
            return
        values = np.asarray(message.data, dtype=float)
        if not np.all(np.isfinite(values)):
            self.get_logger().error("Rejected non-finite motor command")
            return
        with self._command_lock:
            self._requested_enable = bool(values[0] > 0.5)
            self._requested_steering_target_rad = float(values[1])
            self._requested_drive_target_nm = float(values[2])
            self._steering_target_rad = float(
                np.clip(values[1], -self._steering_limit_rad, self._steering_limit_rad)
            )
            self._drive_target_nm = float(
                np.clip(values[2], -DRIVE_EFFORT_LIMIT_NM, DRIVE_EFFORT_LIMIT_NM)
            )
            self._command_monotonic = time.monotonic()

    def _snapshot(self, motor: ChannelState) -> tuple:
        with motor.lock:
            return (
                motor.position_rad,
                motor.velocity_rad_s,
                motor.sample_monotonic,
                motor.torque_constant_nm_per_a,
                motor.position_sample_monotonic,
                motor.velocity_sample_monotonic,
                motor.position_query_ms,
                motor.velocity_query_ms,
                motor.position_query_failures,
                motor.velocity_query_failures,
                motor.multi_turn_count,
                motor.zero_count,
            )

    def _control_tick(self) -> None:
        now = time.monotonic()
        self._control_sequence += 1
        interval_ms = age_ms(now, self._last_control_monotonic)
        self._last_control_monotonic = now
        steering = self._snapshot(self._steering)
        drive = self._snapshot(self._drive)
        with self._command_lock:
            requested_enable = self._requested_enable
            command_age = now - self._command_monotonic
            steering_target = self._steering_target_rad
            drive_target_nm = self._drive_target_nm
            requested_steering = self._requested_steering_target_rad
            requested_drive = self._requested_drive_target_nm

        sign_ready = self._position_command_sign in (-1.0, 1.0)
        if (
            not self._dry_run and sign_ready
            and not self._position_limits_configured and not self._fault_latched
            and steering[3] is not None and now >= self._position_limits_retry_at
        ):
            self._position_limits_retry_at = now + 1.0
            self._configure_position_limits(float(steering[3]))
            # Initial B2/B3 setup/readback can take longer than one control tick.
            # Recheck safety feedback after setup, rather than using old samples.
            steering = self._snapshot(self._steering)
            drive = self._snapshot(self._drive)

        gate_now = time.monotonic()
        command_age = gate_now - self._command_monotonic
        steering_fresh = steering[2] > 0.0 and gate_now - steering[2] <= self._feedback_timeout_s
        drive_fresh = drive[2] > 0.0 and gate_now - drive[2] <= self._feedback_timeout_s
        steering_ready = steering[3] is not None and steering_fresh
        drive_ready = drive[3] is not None and drive_fresh
        feedback_ready = (
            steering_ready and drive_ready
        )
        command_ready = self._command_monotonic > 0.0 and command_age <= self._command_timeout_s
        over_limit = (
            abs(steering[0]) > self._steering_limit_rad
            or abs(steering[1]) > STEERING_VELOCITY_LIMIT_RAD_S * 1.1
            or abs(drive[1]) > DRIVE_VELOCITY_LIMIT_RAD_S * 1.1
        )
        if over_limit:
            self._fault_latched = True
            self._fault_reason = (
                "steering_position_limit" if abs(steering[0]) > self._steering_limit_rad
                else "steering_speed_limit" if abs(steering[1]) > STEERING_VELOCITY_LIMIT_RAD_S * 1.1
                else "drive_speed_limit"
            )
        if motor_fault_clear_allowed(requested_enable, steering[0], self._steering_fault_clear_rad):
            self._fault_latched = False
            self._fault_reason = None

        position_feedback_ready = steering[10] is not None and steering[11] is not None
        position_mode_ready = (
            sign_ready and position_feedback_ready
            and (self._dry_run or self._position_limits_configured)
        )
        enabled = (
            requested_enable and command_ready and feedback_ready and not self._fault_latched and position_mode_ready
        )
        reasons = motor_disable_reasons(
            requested_enable, self._command_monotonic > 0.0, command_ready,
            steering[3] is not None, drive[3] is not None, steering_fresh, drive_fresh, self._fault_latched,
        )
        if not sign_ready:
            reasons.append("c2_position_command_sign_invalid")
        if not position_feedback_ready:
            reasons.append("c2_encoder_reference_unavailable")
        if not self._dry_run and not self._position_limits_configured:
            reasons.append("steering_position_limits_unconfirmed")
        diagnostics = {
            "sequence": self._control_sequence,
            "ros_time_s": self.get_clock().now().nanoseconds * 1e-9,
            "monotonic_time_s": now, "cycle_interval_ms": interval_ms,
            "dry_run": self._dry_run, "requested_enable": requested_enable, "gate_enabled": enabled,
            "command_age_ms": age_ms(gate_now, self._command_monotonic), "command_ready": command_ready,
            "steering_feedback_ready": steering_ready, "drive_feedback_ready": drive_ready,
            "steering_pair_age_ms": age_ms(gate_now, steering[2]), "drive_pair_age_ms": age_ms(gate_now, drive[2]),
            "steering_position_rad": steering[0], "steering_velocity_rad_s": steering[1],
            "drive_position_rad": drive[0], "drive_velocity_rad_s": drive[1],
            "requested_steering_target_rad": requested_steering, "requested_drive_target_nm": requested_drive,
            "bounded_steering_target_rad": steering_target, "bounded_drive_target_nm": drive_target_nm,
            "steering_target_clipped": requested_steering != steering_target,
            "drive_target_clipped": requested_drive != drive_target_nm,
            "gate_monotonic_time_s": gate_now,
            "steering_position_command_sign": self._position_command_sign,
            "steering_position_limits_configured": self._position_limits_configured,
            "steering_position_speed_limit_raw": (
                math.nan if self._position_speed_limit_raw is None else self._position_speed_limit_raw
            ),
            "steering_position_current_limit_raw": (
                math.nan if self._position_current_limit_raw is None else self._position_current_limit_raw
            ),
        }
        for name, snapshot in (("steering", steering), ("drive", drive)):
            diagnostics.update({
                f"{name}_position_age_ms": age_ms(gate_now, snapshot[4]),
                f"{name}_velocity_age_ms": age_ms(gate_now, snapshot[5]),
                f"{name}_position_query_ms": snapshot[6],
                f"{name}_velocity_query_ms": snapshot[7],
                f"{name}_position_query_failures": snapshot[8],
                f"{name}_velocity_query_failures": snapshot[9],
            })
        steering_send, drive_send = MotorSendResult(), MotorSendResult()
        disable_result = DisableResult()
        if not enabled:
            self._slewed_steering_target_rad = steering[0]
            disable_result = self._disable_outputs()
            fault = self._fault_latched or (requested_enable and (not feedback_ready or not position_mode_ready))
            self._fault_publisher.publish(Bool(data=bool(fault)))
            diagnostics["slewed_steering_target_rad"] = self._slewed_steering_target_rad
            self._publish_control_telemetry(now, diagnostics, reasons, steering_send, drive_send, disable_result)
            return

        max_step = STEERING_VELOCITY_LIMIT_RAD_S / self._control_rate_hz
        target_error = steering_target - self._slewed_steering_target_rad
        self._slewed_steering_target_rad += float(np.clip(target_error, -max_step, max_step))
        error_rad = self._slewed_steering_target_rad - steering[0]
        # The firmware closes the position/speed/current loops. No host steering
        # torque command or artificial PD estimate is reported in C2 mode.
        drive_damping_nm = -DRIVE_DAMPING_NM_S_PER_RAD * drive[1]
        drive_unclipped_nm = drive_target_nm + drive_damping_nm
        drive_torque_nm = drive_unclipped_nm
        drive_torque_nm = float(
            np.clip(drive_torque_nm, -DRIVE_EFFORT_LIMIT_NM, DRIVE_EFFORT_LIMIT_NM)
        )
        diagnostics.update({
            "slewed_steering_target_rad": self._slewed_steering_target_rad,
            "steering_slew_limited": abs(target_error) > max_step,
            "steering_error_rad": error_rad,
            "drive_damping_torque_nm": drive_damping_nm, "drive_unclipped_torque_nm": drive_unclipped_nm,
            "drive_final_torque_nm": drive_torque_nm,
            "drive_torque_saturated": drive_torque_nm != drive_unclipped_nm,
        })
        steering_send = self._send_position(self._slewed_steering_target_rad, steering, diagnostics)
        if self._dry_run or steering_send.success:
            drive_send = self._send_torque(self._drive, drive_torque_nm, float(drive[3]))
        if not self._fault_latched:
            self._request_steering_current(diagnostics)
        if self._fault_latched:
            reasons = ["fault_latched", self._fault_reason or "can_send_failed"]
            diagnostics["gate_enabled"] = False
            disable_result = self._disable_outputs(force=True)
            self._fault_publisher.publish(Bool(data=True))
        else:
            self._outputs_active = True
            self._fault_publisher.publish(Bool(data=False))
        self._publish_control_telemetry(now, diagnostics, reasons, steering_send, drive_send, disable_result)

    def _configure_position_limits(self, torque_constant: float) -> None:
        """Apply and independently read back B2/B3 before any powered C2 send.

        B6..B9 firmware gains are deliberately not guessed or changed. These
        limits are volatile; restart this node after resetting motor power.
        """
        try:
            speed_raw, current_raw = position_mode_limits(self._steering.gear_ratio, torque_constant)
            self._position_speed_limit_raw = speed_raw
            self._position_current_limit_raw = current_raw
            for command, raw in ((0xB2, speed_raw), (0xB3, current_raw)):
                request = can.Message(
                    arbitration_id=self._steering.host_id,
                    data=bytes([command]) + struct.pack("<I", raw), is_extended_id=False,
                )
                response = query(self._tx_bus, request, self._steering.feedback_id, 0.008)
                if (
                    response is None or len(response) != 5 or response[0] != command
                    or struct.unpack("<I", response[1:5])[0] != raw
                ):
                    raise ValueError(f"B{command - 0xB0:X} limit setup was not confirmed")
                read_request = can.Message(
                    arbitration_id=self._steering.host_id, data=bytes([command]), is_extended_id=False,
                )
                response = query(self._tx_bus, read_request, self._steering.feedback_id, 0.008)
                if (
                    response is None or len(response) != 5 or response[0] != command
                    or struct.unpack("<I", response[1:5])[0] != raw
                ):
                    raise ValueError(f"B{command - 0xB0:X} limit readback did not match")
            self._position_limits_configured = True
            self.get_logger().info(
                f"C2 steering limits confirmed: B2={speed_raw * 0.01:.2f} motor rpm, "
                f"B3={current_raw * 0.001:.3f} A ({current_raw * 0.001 * torque_constant:.4f} N m)"
            )
        except (can.CanError, ValueError, OverflowError) as error:
            self._position_limits_configured = False
            self.get_logger().error(f"C2 limits unavailable; outputs stay disabled: {error}", throttle_duration_sec=1.0)

    def _send_position(self, target_rad: float, steering: tuple, diagnostics: dict) -> MotorSendResult:
        result = MotorSendResult()  # No host steering current/torque fields in C2 mode.
        started = time.monotonic()
        try:
            motor = self._steering
            if steering[10] is None or steering[11] is None:
                raise ValueError("C2 requires an initialized A3 startup zero and safety feedback")
            if started - steering[4] > self._feedback_timeout_s:
                raise ValueError("steering safety feedback became stale before C2 submission")
            position_unwrapped_rad = (
                motor.encoder_sign * (steering[10] - steering[11]) * RADIANS_PER_COUNT / motor.gear_ratio
            )
            # Keep the unwrapped safety guard; do not use this live position in
            # target conversion or hide an extra revolution behind wrapping.
            if abs(position_unwrapped_rad) > self._steering_limit_rad:
                raise ValueError("unwrapped steering safety feedback exceeds the steering limit")
            payload = absolute_position_payload(
                target_rad, steering[11], motor.gear_ratio, self._position_command_sign, self._steering_limit_rad,
            )
            target_count = struct.unpack("<i", payload[1:5])[0]
            offset_count = target_count - steering[11]
            diagnostics.update({
                "steering_c2_zero_count": steering[11],
                "steering_c2_target_count": target_count,
                "steering_c2_target_offset_count": offset_count,
                "steering_c2_encoded_target_rad": (
                    self._position_command_sign * offset_count * RADIANS_PER_COUNT / motor.gear_ratio
                ),
            })
            if self._dry_run:
                return result
            if not self._position_limits_configured:
                raise ValueError("powered C2 requires confirmed position-mode limits")
            self._drain_tx()
            result.attempted = True
            self._tx_bus.send(can.Message(
                arbitration_id=motor.host_id, data=payload, is_extended_id=False,
            ))
            result.success = True
        except (can.CanError, ValueError, OverflowError) as error:
            self._fault_latched = True
            self._fault_reason = "steering_c2_send_failed"
            result.error = str(error)
            self.get_logger().error(f"Steering C2 send failed: {error}", throttle_duration_sec=1.0)
        finally:
            result.duration_ms = (time.monotonic() - started) * 1000.0
        return result

    def _request_steering_current(self, diagnostics: dict) -> None:
        """C2 replies contain position, so request A1 without waiting for its reply."""
        diagnostics["steering_current_query_send_attempted"] = 0.0
        diagnostics["steering_current_query_send_success"] = 0.0
        if self._dry_run:
            return
        started = time.monotonic()
        try:
            diagnostics["steering_current_query_send_attempted"] = 1.0
            self._tx_bus.send(can.Message(
                arbitration_id=self._steering.host_id, data=bytes([0xA1]), is_extended_id=False,
            ))
            diagnostics["steering_current_query_send_success"] = 1.0
        except can.CanError as error:
            self._fault_latched = True
            self._fault_reason = "steering_a1_query_send_failed"
            self.get_logger().error(f"Steering A1 query send failed: {error}", throttle_duration_sec=1.0)
        finally:
            diagnostics["steering_current_query_send_duration_ms"] = (time.monotonic() - started) * 1000.0

    def _publish_control_telemetry(
        self, started: float, diagnostics: dict, reasons: list[str],
        steering_send: MotorSendResult, drive_send: MotorSendResult, disable_result: DisableResult,
    ) -> None:
        diagnostics.update(send_result_values("steering", steering_send))
        diagnostics.update(send_result_values("drive", drive_send))
        diagnostics.update({
            "outputs_active": self._outputs_active, "fault_latched": self._fault_latched,
            "work_duration_ms": (time.monotonic() - started) * 1000.0,
            "disable_attempted": disable_result.attempted,
            "steering_zero_send_success": math.nan if disable_result.steering_zero_success is None
            else disable_result.steering_zero_success,
            "steering_disable_send_success": math.nan if disable_result.steering_disable_success is None
            else disable_result.steering_disable_success,
            "drive_zero_send_success": math.nan if disable_result.drive_zero_success is None
            else disable_result.drive_zero_success,
            "drive_disable_send_success": math.nan if disable_result.drive_disable_success is None
            else disable_result.drive_disable_success,
        })
        if disable_result.errors:
            reasons = list(dict.fromkeys(reasons + ["disable_send_failed"]))
        message = Float64MultiArray()
        message.layout.dim = [self._control_dimension]
        message.data = telemetry_values(MOTOR_CONTROL_FIELDS, diagnostics)
        self._control_publisher.publish(message)
        event_state = (bool(diagnostics["gate_enabled"]), self._outputs_active, self._fault_latched,
                       tuple(reasons), self._fault_reason, disable_result.errors)
        if event_state != self._last_event_state:
            event = String()
            event.data = json.dumps({
                "sequence": self._control_sequence, "ros_time_s": diagnostics["ros_time_s"],
                "monotonic_time_s": started, "gate_enabled": bool(diagnostics["gate_enabled"]),
                "outputs_active": self._outputs_active, "dry_run": self._dry_run,
                "fault_latched": self._fault_latched, "fault_reason": self._fault_reason,
                "disable_reasons": reasons, "disable_send_errors": list(disable_result.errors),
            }, sort_keys=True)
            self._event_publisher.publish(event)
            self._last_event_state = event_state

    def _drain_tx(self) -> None:
        for _ in range(16):
            try:
                if self._tx_bus.recv(timeout=0.0) is None:
                    break
            except can.CanError:
                break

    def _send_torque(self, motor: ChannelState, torque_nm: float, torque_constant: float) -> MotorSendResult:
        result = MotorSendResult()
        started = time.monotonic()
        try:
            current_a = motor.torque_sign * torque_nm / torque_constant
            payload = q_current_payload(current_a)
            encoded_current = parse_current_feedback(payload)  # Decoding the command, not measured feedback.
            result.current_command_a = current_a
            result.encoded_current_a = encoded_current
            result.encoded_torque_nm = current_feedback_to_torque_nm(
                encoded_current, torque_constant, motor.torque_sign
            )
            if self._dry_run:
                return result  # Calculated only; never report a simulated CAN send as success.
            self._drain_tx()
            result.attempted = True
            self._tx_bus.send(
                can.Message(
                    arbitration_id=motor.host_id,
                    data=payload,
                    is_extended_id=False,
                )
            )
            result.success = True
        except (can.CanError, ValueError, OverflowError) as error:
            self._fault_latched = True
            self._fault_reason = f"{motor.name}_can_send_failed"
            result.error = str(error)
            self.get_logger().error(f"{motor.name} C0 send failed: {error}", throttle_duration_sec=1.0)
        finally:
            result.duration_ms = (time.monotonic() - started) * 1000.0
        return result

    def _disable_outputs(self, force: bool = False) -> DisableResult:
        result = DisableResult()
        if not force and not self._outputs_active:
            return result
        if not self._dry_run:
            result.attempted = True
            errors = []
            for motor in (self._steering, self._drive):
                for operation, payload in (("zero", q_current_payload(0.0)), ("disable", bytes([0xCF]))):
                    try:
                        self._tx_bus.send(can.Message(
                            arbitration_id=motor.host_id, data=payload, is_extended_id=False,
                        ))
                        setattr(result, f"{motor.name}_{operation}_success", True)
                    except can.CanError as error:
                        setattr(result, f"{motor.name}_{operation}_success", False)
                        errors.append(f"{motor.name}_{operation}: {error}")
            result.errors = tuple(errors)
            if errors:
                self._fault_latched = True
                self._fault_reason = "disable_send_failed"
                self.get_logger().error("; ".join(errors), throttle_duration_sec=1.0)
        # Keep retrying on subsequent disabled ticks if transport failed.
        self._outputs_active = bool(result.errors)
        return result

    def _publish_state(self) -> None:
        steering = self._snapshot(self._steering)
        drive = self._snapshot(self._drive)
        if steering[2] <= 0.0 or drive[2] <= 0.0:
            return
        now = time.monotonic()
        if (
            now - steering[2] > self._feedback_timeout_s
            or now - drive[2] > self._feedback_timeout_s
        ):
            return
        sample_times = (steering[2], drive[2])
        if (
            sample_times[0] <= self._last_published_sample_times[0]
            or sample_times[1] <= self._last_published_sample_times[1]
        ):
            return
        message = Float64MultiArray()
        message.data = [steering[0], drive[0], steering[1], drive[1], time.time()]
        self._state_publisher.publish(message)
        self._last_published_sample_times = sample_times

    def _publish_status(self) -> None:
        steering = self._snapshot(self._steering)
        drive = self._snapshot(self._drive)
        now = time.monotonic()
        torque_ages_ms = {}
        for motor in (self._steering, self._drive):
            with motor.lock:
                sample_time = motor.torque_sample_monotonic
            torque_ages_ms[f"{motor.name}_torque_age_ms"] = (
                None if sample_time <= 0.0 else round(max(0.0, now - sample_time) * 1000.0, 2)
            )
        message = String()
        message.data = json.dumps(
            {
                "outputs_active": self._outputs_active,
                "fault_latched": self._fault_latched,
                "dry_run": self._dry_run,
                "steering_torque_constant_nm_per_a": steering[3],
                "drive_torque_constant_nm_per_a": drive[3],
                "steering_control_mode": "C2_absolute_position",
                "steering_position_command_sign": self._position_command_sign,
                "steering_c2_zero_count": steering[11],
                "steering_position_limits_configured": self._position_limits_configured,
                "steering_position_speed_limit_raw": self._position_speed_limit_raw,
                "steering_position_current_limit_raw": self._position_current_limit_raw,
                "torque_feedback_source": "measured A1 steering / A1 or C0 rear current times B0 torque constant",
                **torque_ages_ms,
            },
            sort_keys=True,
        )
        self._status_publisher.publish(message)

    def destroy_node(self) -> None:
        self._running = False
        self._disable_outputs(force=True)
        for thread in getattr(self, "_threads", []):
            thread.join(timeout=0.2)
        if hasattr(self, "_torque_bus"):
            self._torque_bus.shutdown()
        if hasattr(self, "_tx_bus"):
            self._tx_bus.shutdown()
        super().destroy_node()


def main(args=None) -> None:
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=args)
    node = None
    try:
        node = MotorIoPosition()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if node is not None:
                node.destroy_node()
        finally:
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
