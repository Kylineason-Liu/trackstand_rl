"""Locomotive motor node: MIT steering (unchanged) plus rear C1 firmware speed control.

Implements the CycleRL riding-policy drive contract on this bike: the policy's
rear action is a wheel speed target (rad/s), trained against Isaac's implicit
velocity actuator tau = 5.0 * (w* - w) clipped at 6.28 N m. Hardware regulates
the same target with the GIM firmware speed PI loop (C1), whose output current
is capped by B3 at 6.28 N m / B0 torque constant. Firmware PI gains (B8/B9) and
acceleration (B5) are left untouched; they are NOT identified as equal to the
trained Kd = 5.0 law. This is a documented actuator mismatch, not a calibration.

Everything else (single CAN owner, A3/A2/A1/AE polling, freshness gates, CF
disable confirmation, MIT steering at Kp 10 / Kd 0.2) is inherited from
motor_mit_io. Commands arrive on /locomotive/motor_command so a trackstand
torque command can never be interpreted as a speed, or vice versa.
"""

from __future__ import annotations

import math
import time

import rclpy

from .can_protocol import b3_current_limit_payload, c1_speed_payload, parse_b3_current_limit, parse_c1_speed
from .contracts import DRIVE_EFFORT_LIMIT_NM
from .locomotive_contracts import DRIVE_VELOCITY_LIMIT_RAD_S
from .motor_mit_io import MOTOR_CONTROL_FIELDS, MotorMitIo
from .telemetry import MotorSendResult


B3_DOWNWARD_TOLERANCE_A = 0.001  # One protocol count; observed 1000 mA request -> 999 mA reply.
VELOCITY_CONTROL_FIELDS = MOTOR_CONTROL_FIELDS + (
    "drive_velocity_target_rad_s", "drive_c1_encoded_velocity_rad_s", "drive_c1_native_rpm",
    "drive_b3_current_limit_a",
)


class MotorMitVelocityIo(MotorMitIo):
    """MIT steering + rear C1 speed; the drive command field is joint rad/s, not N m."""

    _control_fields = VELOCITY_CONTROL_FIELDS
    # Inherited requested/bounded_drive_target_nm fields carry rad/s under this label.
    _control_label = "motor_control_mit_c1_v1"
    _command_topic = "/locomotive/motor_command"

    def _initialize_command_state(self) -> None:
        super()._initialize_command_state()
        self._drive_b3_limit_a = math.nan
        self._drive_b3_original_a = math.nan

    @staticmethod
    def _bound_drive_target(value: float) -> float:
        return max(-DRIVE_VELOCITY_LIMIT_RAD_S, min(DRIVE_VELOCITY_LIMIT_RAD_S, value))

    def _startup(self) -> None:
        super()._startup()
        drive = self._drive
        if drive.torque_constant_nm_per_a is None:
            self._query(drive, 0xB0, 7, 0.25)
        constant = drive.torque_constant_nm_per_a
        if constant is None or not math.isfinite(constant) or constant <= 0.0:
            raise RuntimeError("drive B0 torque constant is required to cap C1 current")
        original = parse_b3_current_limit(self._query(drive, 0xB3, 5, 0.25))
        self._drive_b3_original_a = original
        # Never raise the existing ceiling; only lower it to the trained torque cap.
        wanted = min(original, DRIVE_EFFORT_LIMIT_NM / constant)
        if self._dry_run:
            self._drive_b3_limit_a = original
            self.get_logger().info(f"dry_run: drive B3={original:.3f} A left unchanged (would set {wanted:.3f} A)")
            return
        payload = b3_current_limit_payload(wanted)
        requested = parse_b3_current_limit(payload)
        for reply in (self._query(drive, 0xB3, 5, 0.25, payload), self._query(drive, 0xB3, 5, 0.25)):
            reported = parse_b3_current_limit(reply)
            if not (0.0 < reported <= requested and requested - reported <= B3_DOWNWARD_TOLERANCE_A + 1e-9):
                raise RuntimeError(f"drive B3 ceiling not confirmed: requested {requested:.3f} A, got {reported:.3f} A")
        self._drive_b3_limit_a = reported
        self.get_logger().info(
            f"Rear C1 speed mode: B3 {original:.3f} -> {reported:.3f} A "
            f"(= {reported * constant:.3f} N m at B0 {constant:.4f} N m/A); volatile until power cycle"
        )

    def _actuate_drive(self, diag: dict, target_drive_rad_s: float, guard, send_drive: bool) -> MotorSendResult:
        diag.update(drive_damping_torque_nm=math.nan, drive_unclipped_torque_nm=math.nan,
                    drive_final_torque_nm=math.nan, drive_torque_saturated=math.nan,
                    drive_velocity_target_rad_s=target_drive_rad_s, drive_b3_current_limit_a=self._drive_b3_limit_a)
        if diag["gate_enabled"] and not self._fault_latched and send_drive:
            return self._send_velocity(target_drive_rad_s, diag, guard)
        return MotorSendResult()

    def _send_velocity(self, velocity_rad_s: float, diag: dict, guard=None) -> MotorSendResult:
        result = MotorSendResult()
        started = time.monotonic()
        motor = self._drive
        try:
            if not math.isfinite(self._drive_b3_limit_a) and not self._dry_run:
                raise RuntimeError("drive B3 current ceiling was not configured")
            payload = c1_speed_payload(velocity_rad_s, motor.gear_ratio, motor.encoder_sign)
            encoded = parse_c1_speed(payload, motor.gear_ratio, motor.encoder_sign)
            diag.update(drive_c1_encoded_velocity_rad_s=encoded,
                        drive_c1_native_rpm=int.from_bytes(payload[1:], "little", signed=True) * 0.01)
            if (guard is None or guard()) and not self._dry_run:
                result.attempted = True
                self._send(motor.host_id, payload)
                result.success = True
        except (OSError, RuntimeError, ValueError, OverflowError) as error:
            self._fault("drive_can_send_failed")
            result.error = str(error)
            self.get_logger().error(f"drive C1 send failed: {error}")
        result.duration_ms = (time.monotonic() - started) * 1000.0
        return result


def main(args=None) -> None:
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=args)
    node = None
    try:
        node = MotorMitVelocityIo()
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
