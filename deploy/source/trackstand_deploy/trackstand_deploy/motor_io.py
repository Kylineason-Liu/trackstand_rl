"""200 Hz GIM encoder polling and actuator-law execution over SocketCAN."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import threading
import time

import can
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float64MultiArray, MultiArrayDimension, String

from .can_protocol import (
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
    STEERING_KD_NM_S_PER_RAD,
    STEERING_KP_NM_PER_RAD,
    STEERING_VELOCITY_LIMIT_RAD_S,
    motor_fault_clear_allowed,
)
from .telemetry import (
    MOTOR_CONTROL_FIELDS,
    MOTOR_STATE_FIELDS,
    MOTOR_STATE_LABEL,
    DisableResult,
    MotorControlPublishGate,
    MotorSendResult,
    age_ms,
    fresh_torque_feedback_nm,
    motor_disable_reasons,
    send_result_values,
    telemetry_values,
)


@dataclass
class ChannelState:
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


class MotorIo(Node):
    _parameter_defaults = {
        "can_interface": "can0", "steering_host_id": "102", "steering_feedback_id": "002",
        "drive_host_id": "101", "drive_feedback_id": "001", "steering_gear_ratio": 1.0,
        "drive_gear_ratio": 1.0, "steering_encoder_sign": -1.0, "drive_encoder_sign": 1.0,
        "steering_torque_sign": -1.0, "drive_torque_sign": -1.0,
        "poll_rate_hz": 200.0, "control_rate_hz": 200.0,
        "command_timeout_s": 0.08, "feedback_timeout_s": 0.05,
        "steering_limit_rad": math.radians(90.0), "steering_fault_clear_rad": math.radians(5.0),
        "dry_run": False,
    }
    _channel_type = ChannelState
    _control_fields = MOTOR_CONTROL_FIELDS
    _control_label = "motor_control_v1"
    _startup_label = "Motor I/O"
    _track_multi_turn = False
    _snapshot_fields = (
        "position_rad", "velocity_rad_s", "sample_monotonic", "torque_constant_nm_per_a",
        "position_sample_monotonic", "velocity_sample_monotonic", "position_query_ms",
        "velocity_query_ms", "position_query_failures", "velocity_query_failures",
    )

    def __init__(self) -> None:
        super().__init__("trackstand_motor_io")
        self._setup_parameters()
        self._initialize_command_state()
        self._connect_legacy_io()

    def _setup_parameters(self) -> None:
        for name, default in self._parameter_defaults.items():
            self.declare_parameter(name, default)

        self._can_interface = str(self.get_parameter("can_interface").value)
        self._poll_rate_hz = float(self.get_parameter("poll_rate_hz").value)
        self._control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self._command_timeout_s = float(self.get_parameter("command_timeout_s").value)
        self._feedback_timeout_s = float(self.get_parameter("feedback_timeout_s").value)
        self._steering_limit_rad = float(self.get_parameter("steering_limit_rad").value)
        self._steering_fault_clear_rad = float(self.get_parameter("steering_fault_clear_rad").value)
        self._dry_run = bool(self.get_parameter("dry_run").value)
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
            return self._channel_type(
                name=name,
                host_id=int(str(self.get_parameter(f"{prefix}_host_id").value), 16),
                feedback_id=int(str(self.get_parameter(f"{prefix}_feedback_id").value), 16),
                gear_ratio=float(self.get_parameter(f"{prefix}_gear_ratio").value),
                encoder_sign=float(self.get_parameter(f"{prefix}_encoder_sign").value),
                torque_sign=float(self.get_parameter(f"{prefix}_torque_sign").value),
            )

        self._steering = channel("steering", "steering")
        self._drive = channel("drive", "drive")
        for motor in (self._steering, self._drive):
            if not math.isfinite(motor.gear_ratio) or motor.gear_ratio <= 0.0 or motor.encoder_sign not in (-1.0, 1.0):
                raise ValueError(f"invalid encoder configuration for {motor.name}")
            if motor.torque_sign not in (-1.0, 1.0):
                raise ValueError(f"invalid torque sign for {motor.name}")

    def _initialize_command_state(self) -> None:
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
        self._control_publish_gate = MotorControlPublishGate()

    def _connect_legacy_io(self) -> None:
        self._tx_bus = can.interface.Bus(
            channel=self._can_interface, interface="socketcan", receive_own_messages=False
        )
        self._create_interfaces()

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
            f"{self._startup_label} on {self._can_interface}: poll={self._poll_rate_hz:.0f} Hz, "
            f"control={self._control_rate_hz:.0f} Hz, dry_run={self._dry_run}"
        )

    def _create_interfaces(self) -> None:
        self._state_publisher = self.create_publisher(
            Float64MultiArray, "/trackstand/motor_state", 10
        )
        self._state_dimension = MultiArrayDimension(
            label=MOTOR_STATE_LABEL, size=len(MOTOR_STATE_FIELDS), stride=len(MOTOR_STATE_FIELDS),
        )
        self._control_publisher = self.create_publisher(
            Float64MultiArray, "/trackstand/motor_control",
            QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self._event_publisher = self.create_publisher(String, "/trackstand/motor_event", 100)
        self._control_dimension = MultiArrayDimension(
            label=self._control_label + ":" + ",".join(self._control_fields),
            size=len(self._control_fields), stride=len(self._control_fields),
        )
        self._fault_publisher = self.create_publisher(Bool, "/trackstand/motor_fault", 10)
        self._status_publisher = self.create_publisher(String, "/trackstand/motor_status", 10)
        self._command_subscription = self.create_subscription(
            Float64MultiArray, "/trackstand/motor_command", self._command_callback, 10
        )

    def _receive_torque_feedback(self) -> None:
        while self._running:
            try:
                message = self._torque_bus.recv(timeout=0.05)
            except can.CanError as error:
                self.get_logger().error(f"Torque feedback listener stopped: {error}")
                return
            if message is not None:
                self._update_torque_feedback(message)

    def _update_torque_feedback(self, message: can.Message) -> None:
        if message.is_extended_id or message.is_remote_frame or message.is_error_frame:
            return
        if not message.data or message.data[0] not in (0xA1, 0xC0):
            return
        if message.arbitration_id == self._steering.feedback_id:
            motor = self._steering
        elif message.arbitration_id == self._drive.feedback_id:
            motor = self._drive
        else:
            return
        # Keep each asynchronous torque reply's age separate from encoder age.
        receipt_time = time.monotonic()
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
        with motor.lock:
            motor.torque_nm = torque_nm
            motor.torque_sample_monotonic = receipt_time

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
                            if self._track_multi_turn:
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
            return tuple(getattr(motor, name) for name in self._snapshot_fields)

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

        steering_fresh = steering[2] > 0.0 and now - steering[2] <= self._feedback_timeout_s
        drive_fresh = drive[2] > 0.0 and now - drive[2] <= self._feedback_timeout_s
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

        enabled = requested_enable and command_ready and feedback_ready and not self._fault_latched
        reasons = motor_disable_reasons(
            requested_enable, self._command_monotonic > 0.0, command_ready,
            steering[3] is not None, drive[3] is not None, steering_fresh, drive_fresh, self._fault_latched,
        )
        diagnostics = {
            "sequence": self._control_sequence,
            "ros_time_s": self.get_clock().now().nanoseconds * 1e-9,
            "monotonic_time_s": now, "cycle_interval_ms": interval_ms,
            "dry_run": self._dry_run, "requested_enable": requested_enable, "gate_enabled": enabled,
            "command_age_ms": age_ms(now, self._command_monotonic), "command_ready": command_ready,
            "steering_feedback_ready": steering_ready, "drive_feedback_ready": drive_ready,
            "steering_pair_age_ms": age_ms(now, steering[2]), "drive_pair_age_ms": age_ms(now, drive[2]),
            "steering_position_rad": steering[0], "steering_velocity_rad_s": steering[1],
            "drive_position_rad": drive[0], "drive_velocity_rad_s": drive[1],
            "requested_steering_target_rad": requested_steering, "requested_drive_target_nm": requested_drive,
            "bounded_steering_target_rad": steering_target, "bounded_drive_target_nm": drive_target_nm,
            "steering_target_clipped": requested_steering != steering_target,
            "drive_target_clipped": requested_drive != drive_target_nm,
        }
        for name, snapshot in (("steering", steering), ("drive", drive)):
            diagnostics.update({
                f"{name}_position_age_ms": age_ms(now, snapshot[4]),
                f"{name}_velocity_age_ms": age_ms(now, snapshot[5]),
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
            fault = self._fault_latched or (requested_enable and not feedback_ready)
            self._fault_publisher.publish(Bool(data=bool(fault)))
            diagnostics["slewed_steering_target_rad"] = self._slewed_steering_target_rad
            self._publish_control_telemetry(now, diagnostics, reasons, steering_send, drive_send, disable_result)
            return

        max_step = STEERING_VELOCITY_LIMIT_RAD_S / self._control_rate_hz
        target_error = steering_target - self._slewed_steering_target_rad
        self._slewed_steering_target_rad += float(np.clip(target_error, -max_step, max_step))
        error_rad = self._slewed_steering_target_rad - steering[0]
        p_torque_nm = STEERING_KP_NM_PER_RAD * error_rad
        d_torque_nm = -STEERING_KD_NM_S_PER_RAD * steering[1]
        steering_unclipped_nm = p_torque_nm + d_torque_nm
        steering_torque_nm = steering_unclipped_nm
        steering_torque_nm = float(
            np.clip(steering_torque_nm, -STEERING_EFFORT_LIMIT_NM, STEERING_EFFORT_LIMIT_NM)
        )
        drive_damping_nm = -DRIVE_DAMPING_NM_S_PER_RAD * drive[1]
        drive_unclipped_nm = drive_target_nm + drive_damping_nm
        drive_torque_nm = drive_unclipped_nm
        drive_torque_nm = float(
            np.clip(drive_torque_nm, -DRIVE_EFFORT_LIMIT_NM, DRIVE_EFFORT_LIMIT_NM)
        )
        diagnostics.update({
            "slewed_steering_target_rad": self._slewed_steering_target_rad,
            "steering_slew_limited": abs(target_error) > max_step,
            "steering_error_rad": error_rad, "steering_p_torque_nm": p_torque_nm,
            "steering_d_torque_nm": d_torque_nm, "steering_unclipped_torque_nm": steering_unclipped_nm,
            "steering_final_torque_nm": steering_torque_nm,
            "steering_torque_saturated": steering_torque_nm != steering_unclipped_nm,
            "drive_damping_torque_nm": drive_damping_nm, "drive_unclipped_torque_nm": drive_unclipped_nm,
            "drive_final_torque_nm": drive_torque_nm,
            "drive_torque_saturated": drive_torque_nm != drive_unclipped_nm,
        })
        steering_send = self._send_torque(self._steering, steering_torque_nm, float(steering[3]))
        if self._dry_run or steering_send.success:
            drive_send = self._send_torque(self._drive, drive_torque_nm, float(drive[3]))
        if self._fault_latched:
            reasons = ["fault_latched", self._fault_reason or "can_send_failed"]
            diagnostics["gate_enabled"] = False
            disable_result = self._disable_outputs(force=True)
            self._fault_publisher.publish(Bool(data=True))
        else:
            self._outputs_active = True
            self._fault_publisher.publish(Bool(data=False))
        self._publish_control_telemetry(now, diagnostics, reasons, steering_send, drive_send, disable_result)

    def _publish_control_telemetry(
        self, started: float, diagnostics: dict, reasons: list[str],
        steering_send: MotorSendResult, drive_send: MotorSendResult, disable_result: DisableResult,
    ) -> None:
        if self._control_publish_gate.due(started):
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
            message = Float64MultiArray()
            message.layout.dim = [self._control_dimension]
            message.data = telemetry_values(self._control_fields, diagnostics)
            self._control_publisher.publish(message)
        if disable_result.errors:
            reasons = list(dict.fromkeys(reasons + ["disable_send_failed"]))
        # Events must still observe every control tick, including unsampled ones.
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
                **self._event_details(),
            }, sort_keys=True)
            self._event_publisher.publish(event)
            self._last_event_state = event_state

    def _event_details(self) -> dict:
        return {}

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
                "torque_feedback_source": "measured A1/C0 Q-axis current times B0 torque constant",
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
    rclpy.init(args=args)
    node = MotorIo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
