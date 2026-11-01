"""MIT range/packet/scheduler/restore checks with simulated CAN; never real motors."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
import csv
import io
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import drive_c1_kp_identify as c1
import drive_mit_velocity_test as experiment
from test_drive_velocity_sine import Clock, FakeCAN, MemoryLog


class MitCAN(FakeCAN):
    def __init__(self, clock):
        super().__init__(clock)
        self.config[0xF0] = b"\xf0"+struct.pack("<HHH", 955, 4500, 1800)
        self.reject_f0_once = self.fail_restore = False
        self.lose_f0_reply_once = False
        self.retain_mit_mode = self.keep_mit_active = False
        self.before_f0_write = lambda: None
        self.socket_options = {}

    def setsockopt(self, level, option, value):
        self.socket_options[level, option] = value

    def send(self, frame):
        cid, length, padded = experiment.shared.CAN_FRAME.unpack(frame)
        data = padded[:length]
        if cid == 0x101 and data[0] == 0xCF and self.retain_mit_mode:
            self.sent.append((self.clock.now, cid, data))
            self.mode = 5
            self.enqueue(self.status(0xCF))
            return len(frame)
        if cid == 0x101 and data == b"\xf1":
            self.sent.append((self.clock.now, cid, data))
            self.enqueue(bytes((0xF1, 128, 0, 128, 8, 0, int(self.keep_mit_active))))
            return len(frame)
        if cid == 0x501:
            if self.pause_once and not self.paused and len(self.sent) > 100:
                self.clock.now += .03
                self.paused = True
            self.sent.append((self.clock.now, cid, data))
            _, vcode, tcode, kd = decode_packet(data)
            _, vmax, tmax = struct.unpack("<HHH", self.config[0xF0][1:])
            self.speed = experiment.shared.decode(vcode, vmax*.01, 12)
            self.speed += experiment.shared.decode(tcode, tmax*.01, 12)/kd
            self.mode = 5
            self.enqueue(bytes((0xF1, data[0], data[1], vcode >> 4,
                                ((vcode & 15) << 4) | (tcode >> 8), tcode & 255, 1)))
            return len(frame)
        if data[0] == 0xF0 and len(data) > 1:
            self.before_f0_write()
            if self.lose_f0_reply_once:
                self.lose_f0_reply_once = False
                self.sent.append((self.clock.now, cid, data))
                self.config[0xF0] = data
                return len(frame)
            if self.reject_f0_once:
                self.reject_f0_once = False
                self.sent.append((self.clock.now, cid, data))
                self.enqueue(self.config[0xF0])
                return len(frame)
            if self.fail_restore and data == b"\xf0"+struct.pack("<HHH", 955, 4500, 1800):
                raise OSError("restore failed")
        return super().send(frame)

    def close(self):
        pass


def decode_packet(data):
    kp = ((data[3] & 15) << 8) | data[4]
    v = (data[2] << 4) | (data[3] >> 4)
    t = ((data[6] & 15) << 8) | data[7]
    kd = ((data[5] << 4) | (data[6] >> 4))*5/4095
    return kp, v, t, kd


class MitVelocityTests(unittest.TestCase):
    def make_io(self, flags=(), persist=lambda: None):
        args = experiment.parse_args(list(flags))
        clock, log = Clock(), MemoryLog()
        bus = MitCAN(clock)
        return experiment.MitVelocityTestIO(args, bus, log, clock=clock, persist=persist), bus, log, clock

    def test_default_plan_matches_c1_exactly_except_labels(self):
        plan = experiment.make_plan(experiment.parse_args([]))
        baseline = c1.make_plan(c1.parse_args([]))
        for a, b in zip(plan, baseline):
            d = asdict(a)
            d.update(name=b.name, mode=b.mode)
            self.assertEqual(d, asdict(b))
        self.assertEqual(len(plan), 342)
        self.assertAlmostEqual(sum(s.duration_s for s in plan), 1194.)
        self.assertEqual(experiment.velocity_range(plan, kd=.3, torque_max=18.), 13.)

    def test_pi_plan_preserves_waveforms_and_uses_distinct_labels(self):
        args = experiment.parse_args(["--velocity-ki", "3.3"])
        for pi, p in zip(experiment.make_plan(args), experiment.make_plan(experiment.parse_args([]))):
            self.assertEqual(pi.mode, "mit_velocity_pi")
            self.assertEqual(replace(pi, name=p.name, mode=p.mode), p)
        self.assertEqual(experiment.parse_args([]).velocity_ki, 0.)

    def test_preview_never_opens_can_or_creates_files(self):
        with patch.object(experiment.shared, "open_bus", side_effect=AssertionError), redirect_stdout(io.StringIO()):
            self.assertEqual(experiment.main([]), 0)
            self.assertEqual(experiment.main(["--velocity-ki", "3.3"]), 0)

    def test_cli_rejects_unencodable_gains_and_invalid_inputs(self):
        for flags in (["--kd", "0"], ["--kd", "0.00001"], ["--kd", "6"], ["--kd", "nan"],
                      ["--velocity-ki", "-1"], ["--velocity-ki", "nan"], ["--velocity-ki", "inf"],
                      ["--velocity-max", "0"], ["--velocity-max", "700"], ["--repeats", "0"],
                      ["--packet-rate", "50", "--frequencies", "25"]):
            with self.subTest(flags=flags), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                experiment.parse_args(flags)

    def test_small_plan_auto_range_and_too_small_explicit_range(self):
        plan = experiment.make_plan(experiment.parse_args(["--suite", "steps", "--step-bases", "0"]))
        self.assertEqual(experiment.velocity_range(plan, torque_max=18.), 3.)
        with self.assertRaises(ValueError):
            experiment.velocity_range(plan, 1., torque_max=18.)
        self.assertAlmostEqual(2*13/4095, .006349206349)

    def test_native_sign_kp_zero_no_integral_and_torque_quantization_cancelled(self):
        for gain in (.2, .3, 1., 5.):
            for target in (-.25, -.05, 0., .05, .25):
                packet, fields, effective = experiment.velocity_payload(target, 0., gain, (95.5, 3., 18.))
                kp, vcode, tcode, kd = decode_packet(packet)
                self.assertEqual(kp, 0)
                self.assertAlmostEqual(kd, fields["mit_kd"])
                native_v = experiment.shared.decode(vcode, 3., 12)
                ff = experiment.shared.decode(tcode, 18., 12)
                self.assertAlmostEqual(-native_v-ff/kd, effective)
                self.assertAlmostEqual(effective, target, delta=3/4095+1e-12)
                self.assertAlmostEqual(fields["estimated_mit_total_torque_nm"], kd*effective)
                self.assertEqual(fields["integral_error_rad"], 0.)
                self.assertEqual(fields["integral_torque_nm"], 0.)
                if target > .04:
                    self.assertLess(native_v, 0)
        _, fields, _ = experiment.velocity_payload(0., 0., .3, (95.5, 3., 18.))
        self.assertLess(abs(fields["estimated_mit_total_torque_nm"]), .3*3/4095+.00001)

    def test_full_plan_quantization_stays_in_range_and_no_gain_or_target_clipping(self):
        plan = experiment.make_plan(experiment.parse_args([]))
        for segment in plan:
            for time in (0., .61, .99, 1.01):
                target = segment.target(time)
                _, _, effective = experiment.velocity_payload(target, target, .3, (95.5, 13., 18.))
                self.assertAlmostEqual(effective, target, delta=13/4095+1e-12)
        with self.assertRaises(RuntimeError):
            experiment.velocity_payload(10., 0., 1., (95.5, 13., 18.))

    def test_integral_packet_native_sign_and_only_quantization_is_cancelled(self):
        for vmax, tmax in ((3., 18.), (13., 6.28)):
            for target in (-.4, 0., .4):
                for integral in (-.14, -.05, 0., .05, .14):
                    packet, fields, effective = experiment.velocity_payload(
                        target, .1, .3, (95.5, vmax, tmax), integral, integral/3.3)
                    kp, vcode, tcode, kd = decode_packet(packet)
                    native_v = experiment.shared.decode(vcode, vmax, 12)
                    native_ff = experiment.shared.decode(tcode, tmax, 12)
                    torque = -(kd*(native_v+.1)+native_ff)
                    self.assertEqual(kp, 0)
                    self.assertAlmostEqual(native_ff, -integral, delta=tmax/4095+1e-12)
                    self.assertAlmostEqual(effective, target, delta=vmax/4095+1e-12)
                    self.assertAlmostEqual(torque, fields["estimated_mit_total_torque_nm"])
                    self.assertAlmostEqual(torque, kd*(target-.1)+integral, delta=kd*vmax/4095+1e-12)
                    self.assertEqual(fields["integral_torque_nm"], integral)
                    self.assertAlmostEqual(fields["integral_error_rad"], integral/3.3)
        with self.assertRaises(RuntimeError):
            experiment.velocity_payload(10., 0., .3, (95.5, 13., 18.), 4.)

    def test_integral_uses_measured_speed_real_elapsed_time_and_can_unwind(self):
        motor, _, _, _ = self.make_io(["--velocity-ki", "3.3"])
        motor.command_ranges, motor.velocity = (95.5, 13., 18.), .5
        motor.update_integral(1.5, 10.)
        self.assertEqual(motor.integral_nm, 0.)
        motor.update_integral(1.5, 10.03)  # 30 ms, not one nominal 5-ms packet period.
        self.assertAlmostEqual(motor.integral.error_rad, .03)
        self.assertAlmostEqual(motor.integral_nm, .099)
        motor.update_integral(-.5, 10.04)
        motor.update_integral(-.5, 10.05)
        self.assertAlmostEqual(motor.integral.error_rad, .02)
        self.assertAlmostEqual(motor.integral_nm, .066)
        with self.assertRaisesRegex(RuntimeError, "non-monotonic"):
            motor.update_integral(0., 9.)

    def test_anti_windup_respects_total_torque_headroom_without_integrating_hidden_state(self):
        for sign in (-1, 1):
            motor, _, _, _ = self.make_io(["--velocity-ki", "3.3"])
            motor.command_ranges, motor.velocity = (95.5, 30., 18.), 0.
            values = []
            for i in range(2000):
                motor.update_integral(sign*20., 10.+i*.005)
                _, fields, _ = experiment.velocity_payload(
                    sign*20., 0., motor.kd, motor.command_ranges, motor.integral_nm, motor.integral.error_rad)
                self.assertLessEqual(abs(fields["estimated_mit_total_torque_nm"]), 6.28)
                values.append(motor.integral_nm)
            self.assertGreater(motor.integral_limited_updates, 0)
            self.assertLess(abs(motor.integral_nm), .28)
            self.assertAlmostEqual(values[-1], values[100])
            self.assertAlmostEqual(motor.integral.error_rad, motor.integral_nm/3.3)

    def test_integral_obeys_actual_feedforward_range_and_recovers_after_saturation(self):
        motor, _, _, _ = self.make_io(["--velocity-ki", "1"])
        motor.command_ranges, motor.velocity = (95.5, 3., .05), 0.
        for i in range(100):
            motor.update_integral(1., 10.+i*.01)
        self.assertEqual(motor.integral_nm, .05)
        self.assertEqual(motor.integral.error_rad, .05)
        self.assertGreater(motor.integral_limited_updates, 0)
        motor.update_integral(-1., 11.)
        motor.update_integral(-1., 11.01)
        self.assertAlmostEqual(motor.integral_nm, .04)

    def test_segment_reset_preserves_integral_within_step_and_zeros_it_on_disable(self):
        motor, bus, log, _ = self.make_io(["--velocity-ki", "3.3"])
        send = bus.send

        def fixed_speed(frame):
            result = send(frame)
            bus.speed = 0.  # Known stationary plant; A2 independent of command/FF encoding.
            return result

        bus.send = fixed_speed
        segment = replace(motor.plan[0], name="step1", kind="step", base_rad_s=0., amplitude_rad_s=.4,
                          pulse_start_s=.03, pulse_end_s=.07, duration_s=.1)
        with patch.object(experiment.shared.select, "select", bus.select), redirect_stdout(io.StringIO()):
            motor.startup()
            for name in ("step1", "step2"):
                motor.run_segment(replace(segment, name=name))
            final = motor.integral_nm
            motor.cleanup()
        for name in ("step1", "step2"):
            commands = [r for r in log.rows if r["record_type"] == "command" and r["method"] == name]
            self.assertEqual(commands[0]["integral_torque_nm"], 0.)
            # Integral survives the return from the pulse to zero within the same segment.
            self.assertGreater(commands[-1]["integral_torque_nm"], 0.)
            for row in commands:
                self.assertAlmostEqual(row["integral_torque_nm"], 3.3*row["integral_error_rad"])
                self.assertTrue(math.isfinite(row["estimated_mit_total_torque_nm"]))
                self.assertLessEqual(set(row), set(experiment.shared.FIELDS))
        self.assertGreater(final, .04)
        self.assertAlmostEqual(motor.phases[0]["final_integral_torque_nm"], final)
        self.assertAlmostEqual(motor.phases[1]["final_integral_torque_nm"], final)
        self.assertEqual(motor.integral_nm, 0.)
        self.assertIsNone(motor.integral.previous_time)

    def test_read_only_settings_and_confirmed_disable_gate(self):
        motor, bus, _, _ = self.make_io()
        for op in (0xB3, 0xB5, 0xB8, 0xB9):
            with self.assertRaisesRegex(ValueError, "READ ONLY"):
                motor.send(bytes([op])+b"\0"*4)
        for payload in (b"\xb1", b"\xc0\0\0\0\0"):
            with self.assertRaises(ValueError):
                motor.send(payload)
        with self.assertRaisesRegex(RuntimeError, "confirmed disabled"):
            motor.send(bus.config[0xF0])
        self.assertFalse(bus.sent)

    def test_actual_mit_packets_queries_zero_reference_and_restore(self):
        motor, bus, log, _ = self.make_io()
        original = dict(bus.config)
        segment = replace(motor.plan[0], name="dynamic", kind="sine", amplitude_rad_s=.25,
                          frequency_hz=5., duration_s=.2)
        with patch.object(experiment.shared.select, "select", bus.select), redirect_stdout(io.StringIO()):
            motor.startup()
            self.assertEqual(motor.command_ranges, (95.5, 13., 18.))
            motor.run_segment(segment)
            motor.cleanup()
        self.assertEqual(motor.zero_count, 123456)
        self.assertTrue(motor.disable_confirmed)
        self.assertEqual(bus.config, original)
        self.assertEqual(motor.ranges, (95.5, 45., 18.))
        self.assertEqual(motor.command_ranges, (95.5, 13., 18.))
        self.assertFalse(motor.saved)
        commands = [r for r in log.rows if r["record_type"] == "command"]
        self.assertEqual(len(commands), 40)
        self.assertTrue(all(r["can_id"] == "501" and r["opcode"] == "MIT" for r in commands))
        self.assertFalse(any(cid == 0x101 and data[0] == 0xC1 for _, cid, data in bus.sent))
        self.assertTrue(all(set(r) <= set(experiment.shared.FIELDS) for r in log.rows))
        for opcode in ("A1", "A2"):
            self.assertGreaterEqual(sum(r["method"] == "dynamic" and r["record_type"] == "tx"
                                        and r["opcode"] == opcode for r in log.rows), 75)

    def test_f0_write_failure_still_restores_original_and_disables(self):
        motor, bus, _, _ = self.make_io()
        original = dict(bus.config)
        bus.reject_f0_once = True
        with patch.object(experiment.shared.select, "select", bus.select), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "F0 readback mismatch"):
                try:
                    motor.startup()
                finally:
                    motor.cleanup()
        self.assertEqual(bus.config, original)
        self.assertEqual(bus.mode, 0)

    def test_restore_failure_is_reported_after_disable(self):
        motor, bus, _, _ = self.make_io()
        with patch.object(experiment.shared.select, "select", bus.select), redirect_stdout(io.StringIO()):
            motor.startup()
            bus.fail_restore = True
            with self.assertRaisesRegex(RuntimeError, "restoration not confirmed"):
                motor.cleanup()
        self.assertTrue(motor.disable_confirmed)
        self.assertEqual(bus.mode, 0)
        self.assertTrue(motor.saved)

    def test_changed_f0_with_lost_ack_is_still_restored(self):
        motor, bus, _, _ = self.make_io()
        original = dict(bus.config)
        bus.lose_f0_reply_once = True
        with patch.object(experiment.shared.select, "select", bus.select), redirect_stdout(io.StringIO()):
            with self.assertRaises(TimeoutError):
                try:
                    motor.startup()
                finally:
                    motor.cleanup()
        self.assertEqual(bus.config, original)
        self.assertTrue(motor.disable_confirmed)
        self.assertFalse(motor.saved)

    def test_startup_has_no_durable_json_write_after_enabling_watchdog(self):
        motor, bus, _, _ = self.make_io()
        watchdog_states = []
        motor.persist = lambda: watchdog_states.append(bus.config[0xCD][1])
        with patch.object(experiment.shared.select, "select", bus.select), redirect_stdout(io.StringIO()):
            motor.startup()
            motor.cleanup()
        self.assertTrue(watchdog_states)
        self.assertTrue(all(value == 0 for value in watchdog_states))

    def test_interrupt_log_failure_driver_fault_and_missing_mit_feedback_disable_and_restore(self):
        for failure in ("interrupt", "log", "driver", "missing_mit"):
            with self.subTest(failure=failure):
                motor, bus, log, clock = self.make_io()
                original = dict(bus.config)
                segment = replace(motor.plan[0], name="stop", duration_s=1.3)
                with patch.object(experiment.shared.select, "select", bus.select), redirect_stdout(io.StringIO()):
                    motor.startup()
                    if failure == "interrupt":
                        motor.stop_requested = lambda: True
                    elif failure == "log":
                        log.fail = True
                    elif failure == "driver":
                        bus.fault = 32
                    else:
                        bus.drop.add(0xF1)
                    with self.assertRaises((KeyboardInterrupt, OSError, RuntimeError)):
                        try:
                            motor.run_segment(segment)
                        finally:
                            motor.cleanup()
                self.assertTrue(motor.disable_confirmed)
                self.assertEqual(bus.config, original)

    def test_lateness_is_logged_and_skipped_without_abort_or_burst(self):
        motor, bus, _, _ = self.make_io()
        bus.pause_once = True
        segment = replace(motor.plan[0], name="late", duration_s=.6)
        with patch.object(experiment.shared.select, "select", bus.select), redirect_stdout(io.StringIO()):
            motor.startup()
            motor.run_segment(segment)
            motor.cleanup()
        self.assertTrue(bus.paused)
        self.assertGreater(motor.phases[0]["skipped_slots"], 0)
        self.assertLess(motor.phases[0]["actual_duration_s"], .601)
        times = [t for t, cid, _ in bus.sent if cid == 0x501]
        self.assertGreater(min(b-a for a, b in zip(times, times[1:])), .0049)

    def test_main_logs_three_files_and_saves_originals_before_persistent_write(self):
        clock = Clock()
        bus = MitCAN(clock)
        original = dict(bus.config)
        cls = experiment.MitVelocityTestIO
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"virtual_mit.csv"

            def check_snapshot():
                snapshot = json.loads(output.with_suffix(".json").read_text())
                self.assertEqual(snapshot["original_settings"]["F0"], original[0xF0].hex().upper())
                self.assertEqual(snapshot["original_settings"]["CD"], original[0xCD].hex().upper())

            bus.before_f0_write = check_snapshot
            with patch.object(experiment.shared, "open_bus", return_value=bus), \
                 patch.object(experiment.shared.select, "select", bus.select), \
                 patch.object(experiment, "MitVelocityTestIO",
                              side_effect=lambda *a, **kw: cls(*a, clock=clock, **kw)), \
                 redirect_stdout(io.StringIO()):
                status = experiment.main(["--execute", "--velocity-ki", "3.3", "--suite", "steps", "--repeats", "1",
                                          "--step-bases", "0", "--step-sizes", ".25", "--output", str(output)])
            self.assertEqual(status, 0)
            report = json.loads(output.with_suffix(".json").read_text())
            self.assertTrue(report["completed"] and report["disabled_and_restored"] and report["disable_confirmed"])
            self.assertEqual(report["mit_encoding_ranges"], [95.5, 3., 18.])
            self.assertEqual(report["final_readback_ranges"], [95.5, 45., 18.])
            self.assertFalse(report["pending_restore_payloads"])
            self.assertLessEqual(report["encoder_zero_monotonic_s"], report["phases"][0]["start_monotonic_s"])
            config = report["mit_configuration"]
            self.assertEqual(config["position_kp"], 0.)
            self.assertEqual(config["integral_gain_nm_per_rad"], 3.3)
            self.assertEqual(report["schema"], "drive_mit_velocity_test_v2")
            self.assertEqual(config["velocity_kd_code"], 246)
            self.assertAlmostEqual(config["velocity_kd_encoded_nm_per_rad_s"], 246*5/4095)
            self.assertEqual(config["velocity_range_rad_s"], 3.)
            self.assertEqual(sum(p["commands"] for p in report["phases"]), 1440)
            self.assertTrue(output.with_name(output.stem+"_plan.csv").exists())
            with output.open() as stream:
                rows = list(csv.DictReader(stream))
            commands = [r for r in rows if r["record_type"] == "command"]
            self.assertEqual(len(commands), 1440)
            self.assertTrue(all(r["can_id"] == "501" for r in commands))
            self.assertTrue(all(r["method"].startswith("mit_velocity_pi_") for r in commands))
            self.assertTrue(any(abs(float(r["integral_torque_nm"])) > 1e-5 for r in commands))
            zero_events = [r for r in rows if r["note"].startswith("Capture software zero:")]
            self.assertEqual(len(zero_events), 1)
            self.assertEqual(int(zero_events[0]["multi_turn_count"]), 123456)
            self.assertEqual(float(zero_events[0]["position_rad"]), 0.)
            self.assertEqual(bus.config, original)

    def test_retained_mode5_requires_f1_inactive_before_range_writes(self):
        motor, bus, _, _ = self.make_io()
        original = dict(bus.config)
        bus.retain_mit_mode = True
        with patch.object(experiment.shared.select, "select", bus.select), redirect_stdout(io.StringIO()):
            motor.startup()
            motor.cleanup()
        self.assertTrue(motor.disable_confirmed)
        self.assertIn("F1 inactive", motor.disable_confirmation_method)
        self.assertEqual(bus.config, original)

    def test_retained_mode5_active_never_counts_as_disabled(self):
        motor, bus, _, _ = self.make_io()
        bus.retain_mit_mode = bus.keep_mit_active = True
        with patch.object(experiment.shared.select, "select", bus.select), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "CF did not confirm"):
                motor.startup()
        self.assertFalse(motor.disable_confirmed)
        self.assertFalse(any(cid == 0x101 and len(data) > 1 for _, cid, data in bus.sent))

    def test_packet_rate_options_really_send_mit_at_50_200_400_hz(self):
        for rate in (50, 200, 400):
            for ki in (0., 3.3):
                with self.subTest(rate=rate, ki=ki):
                    motor, bus, log, _ = self.make_io(["--packet-rate", str(rate), "--velocity-ki", str(ki)])
                    segment = replace(motor.plan[0], name="rate", kind="steady", base_rad_s=.1, duration_s=.2)
                    with patch.object(experiment.shared.select, "select", bus.select), redirect_stdout(io.StringIO()):
                        motor.startup()
                        motor.run_segment(segment)
                        motor.cleanup()
                    commands = [r for r in log.rows if r["record_type"] == "command"]
                    self.assertEqual(len(commands), rate//5)
                    self.assertAlmostEqual(motor.phases[0]["achieved_rate_hz"], rate, places=6)

    def test_gain_bounds_are_checked_before_rounding_in_all_entry_points(self):
        for gain in (0., -.01, .00001, 5.00001, float("inf"), float("nan")):
            with self.subTest(gain=gain), self.assertRaises(ValueError):
                experiment.kd_encoding(gain)
            with self.subTest(gain=gain), self.assertRaises(ValueError):
                experiment.velocity_payload(0., 0., gain, (95.5, 13., 18.))

    def test_golden_zero_packet_and_native_torque_equation(self):
        packet, fields, effective = experiment.velocity_payload(0., 0., .3, (95.5, 13., 18.))
        self.assertEqual(packet.hex(), "80007fd0000f6800")
        self.assertEqual(decode_packet(packet)[0], 0)
        self.assertEqual(fields["mit_kd"], 246*5/4095)
        for target in (-2., -.25, 0., .25, 2.):
            measured = .9*target
            packet, fields, effective = experiment.velocity_payload(target, measured, .3, (95.5, 13., 18.))
            _, v, t, kd = decode_packet(packet)
            native_v = (v*2/4095-1)*13
            native_ff = (t*2/4095-1)*18
            native_measured = -measured
            self.assertAlmostEqual(-(kd*(native_v-native_measured)+native_ff),
                                   fields["estimated_mit_total_torque_nm"])
            self.assertAlmostEqual(effective, target, delta=13/4095+1e-12)

    def test_nondefault_f0_preserves_position_torque_and_c1_gains_do_not_set_mit_gain(self):
        motor, bus, _, _ = self.make_io()
        original_f0 = b"\xf0"+struct.pack("<HHH", 31, 2650, 628)
        bus.config[0xF0] = original_f0
        bus.config[0xB8] = b"\xb8"+struct.pack("<f", 123.)
        bus.config[0xB9] = b"\xb9"+struct.pack("<f", 456.)
        with patch.object(experiment.shared.select, "select", bus.select), redirect_stdout(io.StringIO()):
            motor.startup()
            self.assertEqual(motor.command_ranges, (3.1, 13., 6.28))
            self.assertEqual(motor.configuration()["velocity_kd_code"], 246)
            applied = bytes.fromhex(motor.applied_settings["F0"])
            self.assertEqual(applied[:3], original_f0[:3])
            self.assertEqual(applied[5:], original_f0[5:])
            self.assertEqual(applied[3:5], struct.pack("<H", 1300))
            motor.cleanup()
        self.assertEqual(bus.config[0xF0], original_f0)
        self.assertEqual(motor.configuration()["velocity_range_rad_s"], 13.)

    def test_capture_zero_ignores_earlier_a3_and_never_zeroes_measured_velocity(self):
        motor, bus, log, _ = self.make_io()
        bus.speed = 2.  # Native backward-positive speed; do not subtract it as bias.
        bus.enqueue(b"\xa3"+struct.pack("<Hi", 77, -9999), delay=0)
        with patch.object(experiment.shared.select, "select", bus.select), redirect_stdout(io.StringIO()):
            motor.startup()
            self.assertEqual(motor.zero_count, 123456)
            self.assertEqual(motor.position_rad, 0.)
            self.assertAlmostEqual(motor.velocity, -2., delta=.001)
            motor.run_segment(replace(motor.plan[0], name="zero", duration_s=.02))
            motor.cleanup()
        first = next(r for r in log.rows if r["record_type"] == "command")
        self.assertGreater(first["estimated_mit_total_torque_nm"], .59)
        self.assertFalse(any(cid == 0x101 and data[0] == 0xB1 for _, cid, data in bus.sent))

    def test_signed_multiturn_rollover_keeps_one_continuous_software_origin(self):
        motor, bus, log, _ = self.make_io()
        counts = (2**31-2, -2**31, 2**31-1)
        expected = (0., -2*2*3.141592653589793/16384, -2*3.141592653589793/16384)
        for count, position in zip(counts, expected):
            bus.enqueue(b"\xa3"+struct.pack("<Hi", 0, count), delay=0)
            motor.receive()
            self.assertAlmostEqual(motor.position_rad, position)
        self.assertEqual(motor.zero_count, counts[0])
        self.assertEqual(len(log.rows), 3)

    def test_partial_restore_records_only_failed_setting_and_retry_does_not_rewrite_f0(self):
        motor, bus, _, _ = self.make_io()
        original = dict(bus.config)
        with patch.object(experiment.shared.select, "select", bus.select), redirect_stdout(io.StringIO()):
            motor.startup()
            configure = motor.configure_setting

            def reject_cd(payload):
                if payload == original[0xCD]:
                    raise OSError("CD restore failed")
                return configure(payload)

            with patch.object(motor, "configure_setting", side_effect=reject_cd):
                with self.assertRaisesRegex(RuntimeError, "restoration not confirmed"):
                    motor.cleanup()
            self.assertEqual(motor.saved, [original[0xCD]])
            self.assertEqual(bus.config[0xF0], original[0xF0])
            f0_writes = sum(cid == 0x101 and data[0] == 0xF0 and len(data) > 1 for _, cid, data in bus.sent)
            motor.cleanup()
            self.assertEqual(sum(cid == 0x101 and data[0] == 0xF0 and len(data) > 1
                                 for _, cid, data in bus.sent), f0_writes)
        self.assertEqual(bus.config, original)
        self.assertFalse(motor.saved)

    def test_cleanup_never_restores_ranges_without_disable_confirmation(self):
        motor, bus, _, _ = self.make_io()
        with patch.object(experiment.shared.select, "select", bus.select), redirect_stdout(io.StringIO()):
            motor.startup()
            bus.bad_cf = True
            before = len(bus.sent)
            with self.assertRaisesRegex(RuntimeError, "CF did not confirm"):
                motor.cleanup()
        self.assertFalse(motor.disable_confirmed)
        self.assertFalse(any(cid == 0x101 and len(data) > 1 for _, cid, data in bus.sent[before:]))
        self.assertTrue(motor.saved)

    def test_filter_covers_both_legal_rear_mit_ids_and_other_writer_is_rejected(self):
        motor, bus, _, _ = self.make_io()
        with patch.object(experiment.shared, "open_bus", return_value=bus):
            self.assertIs(experiment.open_bus("can0"), bus)
        encoded = bus.socket_options[experiment.shared.socket.SOL_CAN_RAW, experiment.shared.socket.CAN_RAW_FILTER]
        ids = {cid for cid, mask in struct.iter_unpack("=II", encoded)}
        self.assertTrue({0x401, 0x501, 0x101, 0x001} <= ids)
        bus.enqueue(b"\0"*8, cid=0x401, delay=0)
        with self.assertRaisesRegex(RuntimeError, "another rear"):
            motor.receive()


if __name__ == "__main__":
    unittest.main()
