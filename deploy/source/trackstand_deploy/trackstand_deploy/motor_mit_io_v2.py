"""V2: one CAN owner, MIT steering and rear velocity P plus 200-Hz host I.

Shares the proven MIT transport/disable/state machinery, never the rear C0 law.
50-Hz versioned velocity commands; 100-Hz queries/state; 200-Hz packets and I.
F0 describes encoding, not a verified clamp on total internal MIT PD torque.
"""

from contextlib import nullcontext
import math
import struct
import threading
import time

import rclpy

from .can_protocol import RADIANS_PER_COUNT, parse_a2_velocity, parse_a3_multi_turn_count
from .contracts import STEERING_ACTION_SCALE_RAD
from .contracts_v2 import (
    COMMAND_LABEL, COMMAND_TOPIC, DRIVE_EFFORT_LIMIT_NM, DRIVE_KD_NM_S_PER_RAD,
    DRIVE_KI_NM_PER_RAD, DRIVE_VELOCITY_SCALE_RAD_S, INTEGRAL_RATE_HZ, TASK,
    VelocityIntegral, encoded_drive_kd, integral_limit, rear_f0_payload, velocity_packet,
)
from .mit_protocol import f0_with_position, parse_f0, parse_f1
from .mit_settings import SettingsJournal
from .motor_mit_io import (
    MIT_RATE_HZ, MOTOR_CONTROL_FIELDS as V1_CONTROL_FIELDS,
    MotorMitIo, MotorSendResult, DisableResult, age_ms,
)

_REAR_C0_FIELDS = {
    "requested_drive_target_nm", "bounded_drive_target_nm", "drive_damping_torque_nm",
    "drive_unclipped_torque_nm", "drive_final_torque_nm", "drive_torque_saturated",
}
MOTOR_CONTROL_FIELDS = tuple(name for name in V1_CONTROL_FIELDS if name not in _REAR_C0_FIELDS) + (
    "requested_drive_target_rad_s", "bounded_drive_target_rad_s", "drive_pi_velocity_rad_s",
    "drive_pi_velocity_age_ms", "drive_velocity_error_rad_s", "drive_velocity_p_torque_nm",
    "drive_integral_torque_nm", "drive_integral_error_rad", "drive_integral_dt_ms",
    "drive_integral_limited_updates", "drive_pi_unclipped_torque_nm",
    "drive_sim_reference_clipped_torque_nm", "drive_sim_reference_saturated",
    "drive_mit_config_ready", "drive_mit_sent", "drive_mit_f1_age_ms", "drive_mit_last_send_age_ms",
    "drive_mit_position_max_rad", "drive_mit_velocity_max_rad_s", "drive_mit_torque_max_nm",
    "drive_mit_feedback_position_motor_rad", "drive_mit_feedback_velocity_motor_rad_s",
    "drive_mit_feedback_torque_motor_nm", "drive_mit_feedback_active", "drive_mit_feedback_fault",
    "drive_mit_encoded_velocity_motor_rad_s", "drive_mit_encoded_feedforward_torque_motor_nm",
    "drive_mit_encoded_kp_nm_per_motor_rad", "drive_mit_encoded_kd_nm_s_per_motor_rad",
    "drive_mit_effective_target_rad_s", "drive_mit_estimated_total_torque_nm",
    "drive_mit_actual_send_interval_ms", "mit_total_torque_clamp_verified", "settings_restore_pending_count",
)


class MotorMitIoV2(MotorMitIo):
    _node_name = "trackstand_motor_io_v2"
    _command_topic = COMMAND_TOPIC
    _backend_description = "MIT steering/rear velocity PI/control"
    _control_fields = MOTOR_CONTROL_FIELDS
    _control_label = "motor_control_mit_velocity_v2"
    _shutdown_timeout_s = 4.0  # Both disables plus up to four write/readback restorations.
    _parameter_defaults = {
        **MotorMitIo._parameter_defaults,
        "drive_mit_kd_nm_s_per_motor_rad": DRIVE_KD_NM_S_PER_RAD,
        "drive_integral_gain_nm_per_rad": DRIVE_KI_NM_PER_RAD,
        "drive_integral_rate_hz": INTEGRAL_RATE_HZ,
        "settings_journal_path": "~/.ros/trackstand_deploy/v2_mit_settings.json",
        "recover_pending_settings": False,
    }

    def _initialize_backend(self):
        for name, expected in (("drive_mit_kd_nm_s_per_motor_rad", DRIVE_KD_NM_S_PER_RAD),
                               ("drive_integral_gain_nm_per_rad", DRIVE_KI_NM_PER_RAD),
                               ("drive_integral_rate_hz", INTEGRAL_RATE_HZ)):
            if float(self.get_parameter(name).value) != expected:
                raise ValueError(f"v2 model contract requires {name}={expected}")
        if self._mit_kp != 10. or self._mit_kd != .2:
            raise ValueError("v2 steering MIT gains must match training: 10 / 0.2")
        for motor in (self._steering, self._drive):
            if motor.gear_ratio != 1. or motor.encoder_sign != -1. or motor.torque_sign != -1.:
                raise ValueError("v2 currently supports the tested bicycle's gear=1, feedback/current signs=-1")
        self._drive.lock = threading.RLock()  # Atomic rear rollover repair around the inherited parser.
        del self._drive_target_nm, self._requested_drive_target_nm
        self._drive_target_rad_s = self._requested_drive_target_rad_s = 0.
        self._rear_integral = VelocityIntegral()
        self._rear_lock = threading.Lock()
        self._rear_ranges = self._rear_command_ranges = self._rear_feedback = self._rear_packet = None
        self._rear_config_ready = self._rear_sent_this_enable = False
        self._rear_first_sent_at = self._rear_last_sent_at = self._rear_f1_at = 0.
        self._rear_send_interval_ms = math.nan
        self._rear_a2_velocity_rad_s, self._rear_a2_at = math.nan, 0.
        self._rear_last_count, self._rear_travel_counts = None, 0
        path = str(self.get_parameter("settings_journal_path").value)
        if not path.strip():
            raise ValueError("settings_journal_path cannot be empty")
        self._settings = SettingsJournal(path, {m.name: m.feedback_id for m in (self._steering, self._drive)})
        self._recover_pending = bool(self.get_parameter("recover_pending_settings").value)
        self._planned_settings, self._expected_watchdogs = {}, {}
        self._restoring_configuration = False

    def _command_callback(self, message):
        if (len(message.data) != 3 or not message.layout.dim
                or message.layout.dim[0].label != COMMAND_LABEL):
            self._reject_command("v2 requires its versioned [enable, steering rad, rear rad/s] command")
            return
        try:
            values = tuple(map(float, message.data))
            if not all(map(math.isfinite, values)) or values[0] not in (0., 1.):
                raise ValueError("invalid enable or nonfinite target")
        except (ValueError, TypeError, OverflowError):
            self._reject_command("invalid v2 velocity command")
            return
        with self._command_lock:
            self._requested_enable = bool(values[0])
            if not self._requested_enable:
                self._disable_generation += 1
            self._requested_steering_target_rad, self._requested_drive_target_rad_s = values[1:]
            limit = min(self._steering_limit_rad, STEERING_ACTION_SCALE_RAD)
            self._steering_target_rad = max(-limit, min(limit, values[1]))
            self._drive_target_rad_s = max(-DRIVE_VELOCITY_SCALE_RAD_S,
                                          min(DRIVE_VELOCITY_SCALE_RAD_S, values[2]))
            self._command_monotonic = time.monotonic()

    def _consume(self, frame, stamp):
        data = frame.data
        rear_encoder = frame.can_id == self._drive.feedback_id and data and data[0] in (0xA2, 0xA3)
        pending = self._pending
        paired_query = (rear_encoder and pending is not None and pending[0] is self._drive
                        and pending[1] == data[0] and len(data) == pending[4]
                        and pending[2] <= frame.received_monotonic <= pending[3])
        if data and data[0] == 0xCD and len(data) != 5:
            raise RuntimeError("malformed motor CD response")
        with self._drive.lock if rear_encoder else nullcontext():
            super()._consume(frame, stamp)
            if paired_query and data[0] == 0xA3:
                count = parse_a3_multi_turn_count(data)
                if self._rear_last_count is not None:
                    self._rear_travel_counts += (count - self._rear_last_count + 2**31) % 2**32 - 2**31
                self._rear_last_count = count
                position = self._drive.encoder_sign * self._rear_travel_counts * RADIANS_PER_COUNT
                self._drive._unwrapped_rad = position
                self._drive._pending_position = (position, frame.received_monotonic)
            elif paired_query and data[0] == 0xA2:
                self._rear_a2_velocity_rad_s = parse_a2_velocity(data, 1., self._drive.encoder_sign)
                self._rear_a2_at = frame.received_monotonic
        motor = self._steering if frame.can_id == self._steering.feedback_id else self._drive
        if data and data[0] == 0xCD and self._rear_config_ready and not self._dry_run:
            expected = self._expected_watchdogs.get(motor.name)
            if expected is not None and data != expected:
                raise RuntimeError(f"{motor.name} watchdog setting changed")
        if (data and data[0] in (0xAE, 0xCF) and len(data) == 8
                and self._expects_active_mode(motor, frame.received_monotonic) and data[6] != 5):
            self._fault(f"{motor.name}_driver_left_mit_mode", hard=True)

    def _consume_ranges(self, motor, data):
        if motor is self._steering:
            return super()._consume_ranges(motor, data)
        ranges = parse_f0(data)
        if self._rear_config_ready and ranges != self._rear_ranges:
            raise RuntimeError("rear F0 encoding ranges changed")
        self._rear_ranges = ranges

    def _consume_mit_feedback(self, motor, data, stamp):
        super()._consume_mit_feedback(motor, data, stamp)
        if motor is self._drive and self._rear_ranges is not None:
            feedback = parse_f1(data, self._rear_ranges)
            if stamp >= self._rear_f1_at:
                self._rear_feedback, self._rear_f1_at = feedback, stamp
            if self._expects_active_mode(motor, stamp) and not feedback.mode_active:
                self._fault("drive_mit_unexpected_exit", hard=True)

    def _expects_active_mode(self, motor, stamp):
        if motor is self._drive:
            return self._rear_sent_this_enable and stamp >= self._rear_first_sent_at
        return super()._expects_active_mode(motor, stamp)

    def _disable_both_for_startup(self):
        for motor in (self._steering, self._drive):
            self._send(motor.host_id, b"\xcf")
        for motor in (self._steering, self._drive):
            self._verify_startup_disable(motor)
        self._disable_confirmed = True

    def _startup_disable(self):
        self._disable_both_for_startup()
        if self._settings.path.exists():
            raise RuntimeError(
                f"pending motor settings at {self._settings.path}; run with recover_pending_settings:=true"
            )
        originals, wanted = {}, {}
        for motor in (self._steering, self._drive):
            f0 = self._query(motor, 0xF0, 7, .25)
            cd = self._query(motor, 0xCD, 5, .25)
            if cd[1] not in (0, 1):
                raise RuntimeError("invalid original motor watchdog")
            originals[f"{motor.name}:F0"], originals[f"{motor.name}:CD"] = f0.hex(), cd.hex()
            wanted[f"{motor.name}:F0"] = (f0_with_position(f0, 3.) if motor is self._steering else rear_f0_payload(f0))
            wanted[f"{motor.name}:CD"] = b"\xcd\x01" + struct.pack("<H", 500) + cd[4:]
        pending = {key: original for key, original in originals.items() if bytes.fromhex(original) != wanted[key]}
        self._settings.claim(originals, pending)  # fsync before B1/F0/CD writes.
        self._planned_settings = wanted

    def _startup(self):
        if self._recover_pending:
            if self._dry_run:
                raise ValueError("settings recovery requires dry_run=false")
            self._disable_both_for_startup()
            self._settings.load()
            self._restore_backend_configuration()
            self.get_logger().info("Pending settings restored with motors disabled; restart v2 normally")
            self._running = False
            return
        super()._startup()  # Proven steering B1/A3/F1 reference; no rear B1.
        if self._dry_run:
            actual = self._query(self._drive, 0xF0, 7, .25)
            self._rear_command_ranges = parse_f0(rear_f0_payload(actual))
        else:
            for key, wanted in self._planned_settings.items():
                if key == "steering:F0":
                    continue  # The base startup already verified this write.
                name, opcode = key.split(":")
                motor = self._steering if name == "steering" else self._drive
                if (self._query(motor, int(opcode, 16), len(wanted), .25, wanted) != wanted
                        or self._query(motor, int(opcode, 16), len(wanted), .25) != wanted):
                    raise RuntimeError(f"{name} {opcode} setting/readback mismatch")
            self._rear_command_ranges = self._rear_ranges
            self._expected_watchdogs = {name: self._planned_settings[f"{name}:CD"] for name in ("steering", "drive")}
        self._rear_last_count, self._rear_travel_counts = None, 0
        with self._drive.lock:
            self._drive.zero_count = None
            self._drive._pending_position = None
        self._query(self._drive, 0xA3, 7, .25)
        self._query(self._drive, 0xA2, 5, .25)
        self._query(self._drive, 0xA1, 5, .25)
        f1 = self._query(self._drive, 0xF1, 7, .25)
        if f1[6] != 0 or abs(self._rear_a2_velocity_rad_s) > .5:
            raise RuntimeError("rear must be inactive/fault-free and stationary at startup")
        self._rear_config_ready = True
        self.get_logger().info(
            f"V2 rear MIT Kp=0 Kd={encoded_drive_kd():.9f}, host Ki=3.3 at 200 Hz; "
            f"encoding={self._rear_command_ranges}; total internal PD torque clamp is unverified"
        )

    def _restore_backend_configuration(self):
        if self._dry_run or self._settings.document is None:
            return
        if not self._disable_confirmed or self._disable_wait:
            raise RuntimeError("cannot restore settings before both motors confirm disable")
        self._mit_config_ready = self._rear_config_ready = False
        self._restoring_configuration = True  # Receive configuration replies with motion still stopped.
        try:
            for key, original in list(self._settings.pending.items()):
                name, op = key.split(":")
                motor = self._steering if name == "steering" else self._drive
                payload = bytes.fromhex(original)
                if (self._query(motor, int(op, 16), len(payload), .25, payload) != payload
                        or self._query(motor, int(op, 16), len(payload), .25) != payload):
                    raise RuntimeError(f"{key} restore not confirmed; original remains in {self._settings.path}")
                self._settings.restored(key)
            self._settings.finish()
        finally:
            self._restoring_configuration = False

    def _extra_configuration_queries(self):
        return ((self._drive, 0xF0, 7), (self._steering, 0xCD, 5), (self._drive, 0xCD, 5))

    def _reset_drive_controller(self):
        with self._rear_lock:
            self._rear_integral.reset()
        self._rear_sent_this_enable = False
        self._rear_first_sent_at = 0.

    def _disable_outputs(self, force=False):
        self._reset_drive_controller()
        return super()._disable_outputs(force)

    def _disable_operations(self, now):
        return [(motor, "disable", b"\xcf") for motor in (self._steering, self._drive)
                if motor.name in self._disable_wait and
                (self._disable_wait[motor.name] is None or now >= self._disable_deadline_at)]

    def _gate_state(self, now, requested_enable, command_stamp, sending=False):
        diag, reasons = super()._gate_state(now, requested_enable, command_stamp, sending)
        now = max(now, time.monotonic())
        with self._drive.lock:
            velocity, stamp = self._rear_a2_velocity_rad_s, self._rear_a2_at
        fresh = math.isfinite(velocity) and 0 < stamp <= now and now-stamp <= self._feedback_timeout_s
        rear_mit_age = now - max(self._rear_f1_at, self._rear_first_sent_at)
        if self._rear_sent_this_enable and rear_mit_age > self._feedback_timeout_s:
            self._fault("drive_mit_feedback_timeout", hard=True)
        if not fresh:
            reasons.append("drive_pi_velocity_missing_or_stale")
            if requested_enable and (sending or self._outputs_active):
                self._fault("drive_pi_velocity_stale")
        if not self._rear_config_ready:
            reasons.append("drive_mit_preflight_incomplete")
        self._feedback_gate_fault |= requested_enable and (not fresh or not self._rear_config_ready)
        diag["gate_enabled"] &= fresh and self._rear_config_ready and not self._fault_latched
        diag.update(drive_pi_velocity_rad_s=velocity, drive_pi_velocity_age_ms=age_ms(now, stamp),
                    drive_mit_f1_age_ms=age_ms(now, self._rear_f1_at),
                    drive_mit_last_send_age_ms=age_ms(now, self._rear_last_sent_at),
                    mit_restart_required=self._hard_fault)
        if self._fault_latched:
            reasons = list(dict.fromkeys(reasons + ["fault_latched", self._fault_reason or "motor_fault"]))
        return diag, reasons

    def _send_drive_mit(self, target, velocity, integral_nm, guard):
        result, started = MotorSendResult(), time.monotonic()
        try:
            self._rear_packet = velocity_packet(target, velocity, integral_nm, self._rear_command_ranges,
                                                self._drive.encoder_sign)
            if not guard():
                return result
            if not self._dry_run:
                result.attempted = True
                self._send(0x400 | self._drive.host_id, self._rear_packet.payload)
                result.success = True
                now = time.monotonic()
                self._rear_send_interval_ms = age_ms(now, self._rear_last_sent_at)
                self._rear_last_sent_at = now
                if not self._rear_sent_this_enable:
                    self._rear_first_sent_at = now
                self._rear_sent_this_enable = True
            self._outputs_active, self._disable_confirmed = True, False
        except (OSError, RuntimeError, ValueError, OverflowError, TypeError) as error:
            self._fault("drive_mit_send_failed")
            result.error = str(error)
            self.get_logger().error(f"rear MIT send failed: {error}")
        finally:
            result.duration_ms = (time.monotonic() - started) * 1000.
        return result

    def _next_control_deadline(self, previous):
        period = 1. / MIT_RATE_HZ
        # Stay on the anchored 200-Hz grid; skip missed ticks, never catch up in a burst.
        steps = max(1, math.floor(max(0., time.monotonic()-previous) / period) + 1)
        return previous + steps * period

    def _control_tick(self, send_drive=True):
        now = time.monotonic()
        self._control_sequence += 1
        interval_ms = age_ms(now, self._last_control_monotonic)
        self._last_control_monotonic = now
        with self._command_lock:
            requested_enable, command_stamp = self._requested_enable, self._command_monotonic
            token = (command_stamp, self._disable_generation)
            steering_target, drive_target = self._steering_target_rad, self._drive_target_rad_s
            steering_request, drive_request = self._requested_steering_target_rad, self._requested_drive_target_rad_s
        if self._outputs_active and requested_enable and token[1] != self._active_disable_generation:
            self._fault("enable_withdrawn_during_control")
        diag, reasons = self._gate_state(now, requested_enable, command_stamp)
        diag.update(sequence=self._control_sequence, ros_time_s=self.get_clock().now().nanoseconds*1e-9,
                    monotonic_time_s=now, cycle_interval_ms=interval_ms, dry_run=self._dry_run,
                    requested_steering_target_rad=steering_request, bounded_steering_target_rad=steering_target,
                    requested_drive_target_rad_s=drive_request, bounded_drive_target_rad_s=drive_target,
                    steering_target_clipped=steering_request != steering_target,
                    drive_target_clipped=drive_request != drive_target,
                    mit_send_due=True, steering_slew_limited=False)
        steer_send, drive_send, disable = MotorSendResult(), MotorSendResult(), DisableResult()
        if diag["gate_enabled"]:
            self._active_disable_generation = token[1]
            error = drive_target - diag["drive_pi_velocity_rad_s"]
            with self._rear_lock:
                self._rear_integral.update(error, now, encoded_drive_kd(), integral_limit(self._rear_command_ranges))
                integral_nm = self._rear_integral.torque_nm
                diag.update(drive_integral_torque_nm=integral_nm,
                            drive_integral_error_rad=self._rear_integral.error_rad,
                            drive_integral_dt_ms=self._rear_integral.last_dt_s*1000.,
                            drive_integral_limited_updates=self._rear_integral.limited_updates)
            total = encoded_drive_kd()*error + integral_nm
            reference = max(-DRIVE_EFFORT_LIMIT_NM, min(DRIVE_EFFORT_LIMIT_NM, total))
            diag.update(drive_velocity_error_rad_s=error, drive_velocity_p_torque_nm=encoded_drive_kd()*error,
                        drive_pi_unclipped_torque_nm=total, drive_sim_reference_clipped_torque_nm=reference,
                        drive_sim_reference_saturated=reference != total)
            guard = lambda: self._motion_allowed(token, diag, reasons)
            steer_send, native = self._send_mit(steering_target, guard)
            diag["mit_native_target_rad"] = native
            if diag["gate_enabled"] and (self._dry_run or steer_send.success):
                self._slewed_steering_target_rad = steering_target
            if diag["gate_enabled"] and not self._fault_latched and send_drive:
                drive_send = self._send_drive_mit(drive_target, diag["drive_pi_velocity_rad_s"], integral_nm, guard)
        if self._fault_latched or not diag["gate_enabled"]:
            diag["gate_enabled"] = False
            self._slewed_steering_target_rad = diag["steering_position_rad"]
            disable = self._disable_outputs(force=steer_send.attempted or drive_send.attempted)
            self._feedback_gate_fault |= requested_enable and bool(self._disable_wait)
            if self._fault_latched:
                reasons = list(dict.fromkeys(reasons + ["fault_latched", self._fault_reason or "send_failed"]))
        diag["slewed_steering_target_rad"] = self._slewed_steering_target_rad
        diag["steering_error_rad"] = self._slewed_steering_target_rad - diag["steering_position_rad"]
        self._publish_control_telemetry(now, diag, reasons, steer_send, drive_send, disable)

    def _publish_control_telemetry(self, started, diag, reasons, steer_send, drive_send, disable):
        if self._rear_ranges is not None:
            diag.update(drive_mit_position_max_rad=self._rear_ranges.position_rad,
                        drive_mit_velocity_max_rad_s=self._rear_ranges.velocity_rad_s,
                        drive_mit_torque_max_nm=self._rear_ranges.torque_nm)
        if self._rear_feedback is not None:
            value = self._rear_feedback
            diag.update(drive_mit_feedback_position_motor_rad=value.position_rad,
                        drive_mit_feedback_velocity_motor_rad_s=value.velocity_rad_s,
                        drive_mit_feedback_torque_motor_nm=value.torque_nm,
                        drive_mit_feedback_active=value.mode_active, drive_mit_feedback_fault=value.system_fault)
        if self._rear_packet is not None:
            value = self._rear_packet.command
            diag.update(drive_mit_encoded_velocity_motor_rad_s=value.velocity_rad_s,
                        drive_mit_encoded_feedforward_torque_motor_nm=value.feedforward_torque_nm,
                        drive_mit_encoded_kp_nm_per_motor_rad=value.kp_nm_per_rad,
                        drive_mit_encoded_kd_nm_s_per_motor_rad=value.kd_nm_s_per_rad,
                        drive_mit_effective_target_rad_s=self._rear_packet.effective_target_rad_s,
                        drive_mit_estimated_total_torque_nm=self._rear_packet.estimated_total_torque_nm,
                        drive_mit_actual_send_interval_ms=self._rear_send_interval_ms)
        # Disabled ticks expose the reset I rather than stale active episode state.
        if not diag["gate_enabled"]:
            with self._rear_lock:
                diag.update(drive_integral_torque_nm=self._rear_integral.torque_nm,
                            drive_integral_error_rad=self._rear_integral.error_rad,
                            drive_integral_dt_ms=self._rear_integral.last_dt_s*1000.,
                            drive_integral_limited_updates=self._rear_integral.limited_updates)
        super()._publish_control_telemetry(started, diag, reasons, steer_send, drive_send, disable)

    def _event_details(self):
        return {**super()._event_details(), "drive_mit_config_ready": self._rear_config_ready,
                "drive_mit_sent": self._rear_sent_this_enable,
                "mit_total_torque_clamp_verified": False, "settings_restore_pending_count": len(self._settings.pending)}

    def _status_additions(self):
        with self._rear_lock:
            integral_nm, dt_s = self._rear_integral.torque_nm, self._rear_integral.last_dt_s
        return dict(task=TASK, motor_command_topic=COMMAND_TOPIC, drive_action="velocity_rad_s",
                    drive_mit_config_ready=self._rear_config_ready, drive_mit_kd=encoded_drive_kd(),
                    drive_integral_gain_nm_per_rad=DRIVE_KI_NM_PER_RAD, drive_integral_rate_hz=INTEGRAL_RATE_HZ,
                    drive_integral_torque_nm=integral_nm, drive_integral_dt_ms=dt_s*1000.,
                    drive_mit_f1_age_ms=None if not self._rear_f1_at else age_ms(time.monotonic(), self._rear_f1_at),
                    drive_command_encoding_planned_only=self._dry_run,
                    torque_feedback_source="measured A1 Q-axis current times B0 torque constant",
                    mit_total_torque_clamp_verified=False, recover_pending_settings=self._recover_pending,
                    settings_journal_path=str(self._settings.path),
                    settings_pending_restore=dict(self._settings.pending))


def main(args=None):
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=args)
    node = None
    try:
        node = MotorMitIoV2()
        while rclpy.ok() and node._worker.is_alive():
            rclpy.spin_once(node, timeout_sec=.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if node is not None:
                node.destroy_node()
        finally:
            rclpy.try_shutdown()
