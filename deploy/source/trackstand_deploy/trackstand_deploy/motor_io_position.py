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
from std_msgs.msg import Bool, String

from .can_protocol import (
    RADIANS_PER_COUNT,
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
from .motor_io import MotorIo
from .telemetry import (
    MOTOR_CONTROL_FIELDS as MOTOR_CURRENT_CONTROL_FIELDS,
    DisableResult,
    MotorSendResult,
    age_ms,
    motor_disable_reasons,
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
    torque_nm: float = math.nan
    torque_sample_monotonic: float = 0.0
    position_sample_monotonic: float = 0.0
    velocity_sample_monotonic: float = 0.0
    position_query_ms: float = math.nan
    velocity_query_ms: float = math.nan
    position_query_failures: int = 0
    velocity_query_failures: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class MotorIoPosition(MotorIo):
    _channel_type = ChannelState
    _control_fields = MOTOR_CONTROL_FIELDS
    _control_label = "motor_control_position_v2"
    _startup_label = "C2 position motor I/O"
    _track_multi_turn = True
    _snapshot_fields = MotorIo._snapshot_fields + ("multi_turn_count", "zero_count")

    def __init__(self) -> None:
        self._position_limits_configured = False
        self._position_speed_limit_raw = self._position_current_limit_raw = None
        self._position_limits_retry_at = 0.0
        super().__init__()
        self.get_logger().info(
            f"C2 command sign={self._position_command_sign:+.0f}, following steering_encoder_sign; "
            "startup A3 zero + converted absolute target. Run only one motor I/O node."
        )


    def _setup_parameters(self) -> None:
        super()._setup_parameters()
        # Native-positive A3/C2 turns left; policy-positive steering turns right.
        self._position_command_sign = self._steering.encoder_sign






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
