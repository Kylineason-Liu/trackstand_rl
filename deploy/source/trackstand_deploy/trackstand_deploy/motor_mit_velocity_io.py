"""Locomotive motor node: MIT steering (unchanged) plus a rear wheel speed target.

The policy's rear action is a wheel speed target (rad/s). Two rear backends,
chosen by the read-only ``rear_control_mode`` parameter:

``mit`` (default) reproduces Locomotive-Cylinder-Aligned-v0's trained law
tau = clip(Kd * (w* - w), +/-6.28 N m) with Kd 0.2, as rear MIT frames with
position Kp 0, velocity Kd and nominal zero feedforward (no host integral: the
policy was trained against this P-only law). Startup saves the drive's F0, widens
only its Vel_Max so +/-45 rad/s targets encode, and restores the saved F0 after a
confirmed shutdown disable. The 6.28 N m clip is enforced on the host at command
time by bounding w* to w +/- 6.28/Kd using the latest A2 speed; between commands
the motor's own loop is not clipped by the host. B3 is also lowered as a current
backstop; whether B3 bounds MIT-mode current is NOT verified on this firmware.

``c1`` keeps the earlier GIM firmware speed PI (C1) behind the B3 ceiling. Its
gains (B8/B9) are not identified as equal to any trained law.

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
from .mit_protocol import MitRanges, encode, f0_with_velocity, mit_velocity_payload, parse_f0
from .motor_mit_io import MOTOR_CONTROL_FIELDS, MotorMitIo
from .telemetry import MotorSendResult


B3_DOWNWARD_TOLERANCE_A = 0.001  # One protocol count; observed 1000 mA request -> 999 mA reply.
REAR_CONTROL_MODES = ("mit", "c1")
DRIVE_MIT_KD_NM_S_PER_RAD = 0.2  # Locomotive-Cylinder-Aligned-v0 / trackstand v2 rear damping.
DRIVE_MIT_VELOCITY_MAX_RAD_S = 50.0  # F0 Vel_Max: covers the 45 rad/s clip plus FF correction.
VELOCITY_CONTROL_FIELDS = MOTOR_CONTROL_FIELDS + (
    "drive_velocity_target_rad_s", "drive_c1_encoded_velocity_rad_s", "drive_c1_native_rpm",
    "drive_b3_current_limit_a", "drive_mit_effective_target_rad_s", "drive_mit_kd_nm_s_per_rad",
    "drive_mit_estimated_torque_nm", "drive_mit_torque_limited", "drive_mit_f1_age_ms",
)


class MotorMitVelocityIo(MotorMitIo):
    """MIT steering + rear MIT velocity or C1 speed; the drive command field is joint rad/s, not N m."""

    _parameter_defaults = {
        **MotorMitIo._parameter_defaults,
        "rear_control_mode": "mit",
        "drive_mit_kd_nm_s_per_rad": DRIVE_MIT_KD_NM_S_PER_RAD,
        "drive_mit_velocity_max_rad_s": DRIVE_MIT_VELOCITY_MAX_RAD_S,
    }
    _control_fields = VELOCITY_CONTROL_FIELDS
    # Inherited requested/bounded_drive_target_nm fields carry rad/s under this label.
    _control_label = "motor_control_mit_rear_velocity_v2"
    _command_topic = "/locomotive/motor_command"

    def _initialize_command_state(self) -> None:
        super()._initialize_command_state()
        parameter = lambda name: self.get_parameter(name).value
        self._rear_control_mode = str(parameter("rear_control_mode"))
        self._drive_mit_kd = float(parameter("drive_mit_kd_nm_s_per_rad"))
        self._drive_mit_velocity_max = float(parameter("drive_mit_velocity_max_rad_s"))
        if self._rear_control_mode not in REAR_CONTROL_MODES:
            raise ValueError(f"rear_control_mode must be one of {REAR_CONTROL_MODES}")
        if not (math.isfinite(self._drive_mit_kd) and 0 < self._drive_mit_kd <= 5):
            raise ValueError("drive MIT Kd must be in (0, 5] N m s/rad")
        if not (math.isfinite(self._drive_mit_velocity_max) and
                DRIVE_VELOCITY_LIMIT_RAD_S + 1.0 <= self._drive_mit_velocity_max <= 655.35):
            raise ValueError("drive MIT Vel_Max must cover the 45 rad/s clip plus 1 rad/s and fit F0")
        self._drive_b3_limit_a = math.nan
        self._drive_b3_original_a = math.nan
        self._drive_mit_ranges: MitRanges | None = None
        self._drive_f0_original: bytes | None = None
        self._drive_f0_written = False
        self._drive_mit_f1_at = self._drive_mit_first_sent_at = 0.0
        self._drive_mit_sent_this_enable = False

    @staticmethod
    def _bound_drive_target(value: float) -> float:
        return max(-DRIVE_VELOCITY_LIMIT_RAD_S, min(DRIVE_VELOCITY_LIMIT_RAD_S, value))

    def _startup(self) -> None:
        super()._startup()
        self._configure_b3()
        if self._rear_control_mode == "mit":
            self._configure_drive_f0()
        self.get_logger().info(
            f"Rear control: {self._rear_control_mode}"
            + (f", MIT Kp 0 / Kd {self._drive_mit_kd:g} N m s/rad, host torque bound "
               f"{DRIVE_EFFORT_LIMIT_NM:g} N m" if self._rear_control_mode == "mit" else ", firmware C1 speed PI"))

    def _configure_b3(self) -> None:
        drive = self._drive
        if drive.torque_constant_nm_per_a is None:
            self._query(drive, 0xB0, 7, 0.25)
        constant = drive.torque_constant_nm_per_a
        if constant is None or not math.isfinite(constant) or constant <= 0.0:
            raise RuntimeError("drive B0 torque constant is required to cap rear current")
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
            f"Rear B3 {original:.3f} -> {reported:.3f} A "
            f"(= {reported * constant:.3f} N m at B0 {constant:.4f} N m/A); volatile until power cycle"
        )

    def _configure_drive_f0(self) -> None:
        """Widen only the drive's F0 Vel_Max; F0 persists in firmware, so log and restore the original."""
        drive = self._drive
        original = self._query(drive, 0xF0, 7, 0.25)
        parse_f0(original)
        wanted = f0_with_velocity(original, self._drive_mit_velocity_max)
        self.get_logger().warning(
            f"Drive F0 original {original.hex().upper()} (restore by hand with this value if shutdown is not clean)")
        if self._dry_run:
            self._drive_mit_ranges = parse_f0(wanted)
            self.get_logger().info(f"dry_run: drive F0 left unchanged (would write {wanted.hex().upper()})")
            return
        self._drive_f0_original = original
        if wanted != original:
            self._drive_f0_written = True
            if self._query(drive, 0xF0, 7, 0.25, wanted) != wanted:
                raise RuntimeError("drive F0 write was not confirmed")
        if self._query(drive, 0xF0, 7, 0.25) != wanted:
            raise RuntimeError("drive F0 readback differs")
        self._drive_mit_ranges = parse_f0(wanted)
        self.get_logger().info(f"Drive F0 -> {wanted.hex().upper()} ({self._drive_mit_ranges})")

    def _after_confirmed_shutdown_disable(self) -> None:
        if not self._drive_f0_written or self._drive_f0_original is None:
            return
        original = self._drive_f0_original
        try:
            if self._query(self._drive, 0xF0, 7, 0.25, original) != original:
                raise RuntimeError("write not confirmed")
            if self._query(self._drive, 0xF0, 7, 0.25) != original:
                raise RuntimeError("readback differs")
            self._drive_f0_written = False
            self.get_logger().info(f"Drive F0 restored to {original.hex().upper()}")
        except (OSError, RuntimeError, TimeoutError, ValueError) as error:
            self.get_logger().error(f"Drive F0 restore failed ({error}); original was {original.hex().upper()}")

    def _consume(self, frame, stamp: float) -> None:
        data = frame.data
        if frame.can_id == self._drive.feedback_id and data and data[0] == 0xF0 and len(data) == 7:
            ranges = parse_f0(data)
            if self._drive_mit_ranges is not None and ranges != self._drive_mit_ranges and not (
                    self._pending is not None and self._pending[:2] == (self._drive, 0xF0)):
                raise RuntimeError("drive F0 encoding ranges changed")
            return
        super()._consume(frame, stamp)
        if (frame.can_id == self._drive.feedback_id and data and data[0] == 0xF1 and len(data) == 7
                and self._drive_mit_sent_this_enable and frame.received_monotonic >= self._drive_mit_first_sent_at):
            self._drive_mit_f1_at = max(self._drive_mit_f1_at, frame.received_monotonic)
            if not data[6] & 1:
                self._fault("drive_mit_unexpected_exit", hard=True)

    def _gate_state(self, now: float, requested_enable: bool, command_stamp: float,
                    sending: bool = False) -> tuple[dict, list[str]]:
        diag, reasons = super()._gate_state(now, requested_enable, command_stamp, sending)
        now = max(now, time.monotonic())
        if (self._drive_mit_sent_this_enable and
                now - max(self._drive_mit_f1_at, self._drive_mit_first_sent_at) > self._feedback_timeout_s):
            self._fault("drive_mit_feedback_timeout", hard=True)
            diag["gate_enabled"] = False
        if self._rear_control_mode == "mit" and self._drive_mit_ranges is None:
            diag["gate_enabled"] = False
            reasons.append("drive_mit_preflight_incomplete")
        if self._drive_mit_sent_this_enable:
            diag["drive_mit_f1_age_ms"] = (now - self._drive_mit_f1_at) * 1000.0 if self._drive_mit_f1_at else math.nan
        return diag, reasons

    def _disable_outputs(self, force: bool = False):
        self._drive_mit_sent_this_enable = False
        return super()._disable_outputs(force)

    def _actuate_drive(self, diag: dict, target_drive_rad_s: float, guard, send_drive: bool) -> MotorSendResult:
        diag.update(drive_damping_torque_nm=math.nan, drive_unclipped_torque_nm=math.nan,
                    drive_final_torque_nm=math.nan, drive_torque_saturated=math.nan,
                    drive_velocity_target_rad_s=target_drive_rad_s, drive_b3_current_limit_a=self._drive_b3_limit_a)
        if diag["gate_enabled"] and not self._fault_latched and send_drive:
            if self._rear_control_mode == "mit":
                return self._send_mit_velocity(target_drive_rad_s, diag, guard)
            return self._send_velocity(target_drive_rad_s, diag, guard)
        return MotorSendResult()

    def _send_mit_velocity(self, velocity_rad_s: float, diag: dict, guard=None) -> MotorSendResult:
        """Rear MIT frame: tau = Kd * (w* - w) with w* bounded so the host-predicted |tau| <= 6.28 N m."""
        result = MotorSendResult()
        started = time.monotonic()
        motor = self._drive
        try:
            if self._drive_mit_ranges is None:
                raise RuntimeError("drive MIT F0 ranges were not configured")
            if not self._dry_run and not math.isfinite(self._drive_b3_limit_a):
                raise RuntimeError("drive B3 current ceiling was not configured")
            measured = float(diag["drive_velocity_rad_s"])
            if not math.isfinite(measured):
                raise RuntimeError("drive speed feedback unavailable")
            kd_native = self._drive_mit_kd / motor.gear_ratio ** 2
            # Bound with the quantized wire Kd, reserving one velocity code of quantization.
            kd_bound = encode(kd_native, 0.0, 5.0, 12) * 5 / 4095 * motor.gear_ratio ** 2
            resolution = 2.0 * self._drive_mit_ranges.velocity_rad_s / 4095 / motor.gear_ratio
            band = DRIVE_EFFORT_LIMIT_NM / kd_bound - resolution
            bounded = max(measured - band, min(measured + band, velocity_rad_s))
            bounded = self._bound_drive_target(bounded)
            native = motor.encoder_sign * motor.gear_ratio * bounded
            payload, kd_wire, effective_native = mit_velocity_payload(native, self._drive_mit_ranges, kd_native)
            effective = motor.encoder_sign * effective_native / motor.gear_ratio
            kd_joint = kd_wire * motor.gear_ratio ** 2
            torque = kd_joint * (effective - measured)
            diag.update(drive_mit_effective_target_rad_s=effective, drive_mit_kd_nm_s_per_rad=kd_joint,
                        drive_mit_estimated_torque_nm=torque, drive_mit_torque_limited=bounded != velocity_rad_s)
            if abs(torque) > DRIVE_EFFORT_LIMIT_NM + 1e-9:
                raise RuntimeError(f"estimated rear MIT torque {torque:.3f} N m exceeds the trained cap")
            if (guard is None or guard()) and not self._dry_run:
                result.attempted = True
                self._send(0x400 | motor.host_id, payload)
                result.success = True
                if not self._drive_mit_sent_this_enable:
                    self._drive_mit_first_sent_at = time.monotonic()
                    self._drive_mit_f1_at = 0.0
                self._drive_mit_sent_this_enable = True
                self._outputs_active, self._disable_confirmed = True, False
        except (OSError, RuntimeError, ValueError, OverflowError, TypeError) as error:
            self._fault("drive_can_send_failed")
            result.error = str(error)
            self.get_logger().error(f"drive MIT send failed: {error}")
        result.duration_ms = (time.monotonic() - started) * 1000.0
        return result

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
