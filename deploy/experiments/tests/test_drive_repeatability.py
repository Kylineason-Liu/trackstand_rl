"""Controlled rear suite: virtual CAN/clock and temporary files only."""

import contextlib
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive_repeatability as suite
from test_drive_policy_replay import DriveCAN
from test_steering_policy_replay import stall_after_command


class ControlledCAN(DriveCAN):
    def __init__(self, fault=None, moving=False):
        super().__init__(fault)
        self.count, self.moving = 1000, moving
        self.preparation_shift = 0

    def send(self, frame):
        cid, length, data = suite.rear.mit.CAN_FRAME.unpack(frame)
        if data[0] == 0xA3:
            self.frames.append((self.now, cid, data[:length]))
            if self.enabled and self.preparation_shift:
                self.count += self.preparation_shift
                self.preparation_shift = 0
            if self.enabled and self.moving and abs(self.current_code) >= 200:
                self.count += 20 if self.current_code > 0 else -20
            self.enqueue(1, b"\xa3" + struct.pack("<Hi", self.count % 16384, self.count), self.reply_delay)
            return len(frame)
        return super().send(frame)


class RepeatabilityTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.session = Path(self.folder.name)

    def args(self, *extra):
        return suite.parse_args(["--preset", "quick", "--pre-zero", ".2", "--coast", ".2",
                                 "--rest-seconds", ".2", "--rest-timeout", "2", "--duration", ".1",
                                 "--pulse-seconds", ".04", "--ramp-dwell", ".04",
                                 "--ramp-step-nm", ".04", "--ramp-max-nm", ".12", *extra])

    def trial(self, family="replay", rate=200, sign=1):
        return dict(trial_id="0001", repeat=1, angle_deg=0., family=family, polarity=sign,
                    amplitude_nm=.12, packet_rate=rate, attempts=[])

    @contextlib.contextmanager
    def hardware(self, bus, inputs=None):
        with patch.object(suite.time, "monotonic", side_effect=lambda: bus.now), \
                patch.object(suite.time, "time", side_effect=lambda: 1700000000 + bus.now), \
                patch.object(suite.rear.mit.select, "select", side_effect=bus.select), \
                patch.object(suite.rear.shared.socket, "socket", return_value=bus), \
                patch("builtins.input", side_effect=inputs or [""]):
            yield

    def run_trial(self, family="replay", rate=200, bus=None, **kwargs):
        bus = bus or ControlledCAN()
        with self.hardware(bus, kwargs.pop("inputs", None)):
            result, error = suite.run_trial(self.args(), self.trial(family, rate), self.session, 1000, 1)
        return result, error, bus

    def test_preview_stdlib_path_has_no_can_or_output(self):
        with patch.object(suite.rear.shared.socket, "socket", side_effect=AssertionError("real CAN")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(suite.main(["--preset", "quick", "--session", str(self.session / "absent")]), 0)
        self.assertEqual(list(self.session.iterdir()), [])

    def test_full_design_balanced_deterministic_rotated_rates(self):
        args = suite.parse_args([])
        trials = suite.build_trials(args)
        self.assertEqual(len(trials), 504)
        self.assertEqual(trials, suite.build_trials(args))
        for family in args.families:
            for angle in args.angles:
                for sign in (-1, 1):
                    amplitudes = args.pulse_torques if family == "pulse" else [.24] if family == "reversal" else [0.]
                    for amplitude in amplitudes:
                        selected = [t for t in trials if (t["family"], t["angle_deg"], t["polarity"], t["amplitude_nm"])
                                    == (family, angle, sign, amplitude)]
                        first_rates = [next(t["packet_rate"] for t in selected if t["repeat"] == r) for r in (1, 2, 3)]
                        self.assertEqual(set(first_rates), {50, 100, 200})

    def test_mirrored_replay_sign_and_holds(self):
        args = self.args()
        a = suite.build_plan(args, self.trial(sign=1))
        b = suite.build_plan(args, self.trial(sign=-1))
        self.assertEqual([r["requested_torque_nm"] for r in a], [-r["requested_torque_nm"] for r in b])
        for row in a:
            if row["phase"] in ("pre_zero", "coast"):
                self.assertEqual(row["requested_torque_nm"], 0.)
        self.assertEqual(a[10]["source_time_s"], 12.5)

    def test_angles_are_physical_modulo_not_software_zero(self):
        self.assertEqual(suite.phase_error(1000 - 4096 - 16384, 1000, 90.), (90., 0.))
        self.assertAlmostEqual(suite.phase_error(1000 + 1, 1000, 0.)[1], -360 / 16384)

    def test_all_rates_current_sign_queries_cleanup_and_json_acyclic(self):
        for rate in (50, 100, 200):
            with self.subTest(rate=rate):
                trial = self.trial(rate=rate)
                trial["trial_id"] = str(rate)
                bus = ControlledCAN()
                with self.hardware(bus):
                    result, error = suite.run_trial(self.args(), trial, self.session, 1000, 1)
                self.assertIsNone(error)
                self.assertEqual(result["status"], "completed")
                self.assertTrue(result["disable_confirmed"] and result["watchdog_restored"])
                self.assertTrue(result["timing_valid"])
                self.assertEqual(result["summary"]["displacement_rad"], 0.)
                self.assertEqual(bus.cd, bus.initial_cd)
                self.assertNotIn("attempts", result)
                json.dumps(result, allow_nan=False)
                with (self.session / result["csv"]).open() as stream:
                    rows = list(csv.DictReader(stream))
                commands = [r for r in rows if r["record_type"] == "drive_target"]
                self.assertEqual(len(commands), round(.5 * rate))
                for row in commands:
                    raw = bytes.fromhex(row["payload_hex"])
                    requested = float(row["requested_torque_nm"])
                    self.assertEqual(struct.unpack("<i", raw[1:])[0], round(-requested / bus.constant * 1000))
                self.assertTrue(all(cid == 0x101 for _, cid, _ in bus.frames))
                self.assertTrue(all(len(data) == 1 or data[0] in (0xC0, 0xCD) for _, _, data in bus.frames))

    def test_breakaway_detection_cuts_ramp_and_records_zero_coast(self):
        result, error, bus = self.run_trial("breakaway", bus=ControlledCAN(moving=True))
        self.assertIsNone(error)
        self.assertIsNotNone(result["breakaway_detected_s"])
        self.assertFalse(result["breakaway_censored"])
        with (self.session / result["csv"]).open() as stream:
            commands = [r for r in csv.DictReader(stream) if r["record_type"] == "drive_target"]
        first_coast = next(i for i, r in enumerate(commands) if r["phase"] == "early_coast")
        self.assertTrue(all(float(r["requested_torque_nm"]) == 0 for r in commands[first_coast:]))
        self.assertLess(abs(result["breakaway_command_nm"]), .12)

    def test_no_motion_ramp_is_censored_not_fabricated_threshold(self):
        result, error, _ = self.run_trial("breakaway")
        self.assertIsNone(error)
        self.assertTrue(result["breakaway_censored"])
        self.assertIsNone(result["breakaway_command_nm"])

    def test_placement_mismatch_and_operator_quit_never_send_c0(self):
        bus = ControlledCAN()
        bus.count = 2000
        with self.hardware(bus, ["", "q"]), contextlib.redirect_stdout(io.StringIO()):
            result, error = suite.run_trial(self.args(), self.trial(), self.session, 1000, 1)
        self.assertIsInstance(error, KeyboardInterrupt)
        self.assertTrue(result["disable_confirmed"])
        self.assertFalse(any(data[0] == 0xC0 for _, _, data in bus.frames))

    def test_current_and_send_errors_disable_and_preserve_partial_data(self):
        for fault in ("send_failed", "motor_fault"):
            trial = self.trial()
            trial["trial_id"] = fault
            bus = ControlledCAN(fault)
            with self.hardware(bus):
                result, error = suite.run_trial(self.args(), trial, self.session, 1000, 1)
            self.assertIsNotNone(error)
            self.assertEqual(result["status"], "failed")
            self.assertTrue(result["disable_confirmed"] and result["watchdog_restored"])
            self.assertTrue((self.session / result["csv"]).exists())

    def test_15ms_stall_logged_without_stretching_signal_or_catchup_burst(self):
        bus = ControlledCAN()
        with self.hardware(bus), stall_after_command(bus, suite.ExperimentIO, .015, 0x101, 0xC0):
            result, error = suite.run_trial(self.args(), self.trial(), self.session, 1000, 1)
        self.assertIsNone(error)
        self.assertFalse(result["timing_valid"])
        self.assertGreaterEqual(result["skipped_packet_slots"], 1)
        self.assertAlmostEqual(result["actual_motion_duration_s"], .5, delta=.002)
        frames = [(t, data) for t, _, data in bus.frames if data[0] == 0xC0]
        self.assertGreaterEqual(min(b[0] - a[0] for a, b in zip(frames, frames[1:])), .004 - 1e-8)

    def test_over_20ms_stall_aborts_and_restores(self):
        bus = ControlledCAN()
        with self.hardware(bus), stall_after_command(bus, suite.ExperimentIO, .035, 0x101, 0xC0):
            result, error = suite.run_trial(self.args(), self.trial(), self.session, 1000, 1)
        self.assertIsInstance(error, RuntimeError)
        self.assertIn("deadline", str(error))
        self.assertTrue(result["disable_confirmed"] and result["watchdog_restored"])

    def test_execute_pause_resume_end_to_end_retains_first_attempt(self):
        directory = self.session / "whole_session"
        bus = ControlledCAN()
        with self.hardware(bus, ["", ""]), contextlib.redirect_stdout(io.StringIO()):
            result = suite.main(["--execute", "--families", "pulse", "--angles", "0", "--rates", "50",
                                 "--repeats", "1", "--pulse-torques", ".12", "--pulse-seconds", ".04",
                                 "--pre-zero", ".2", "--coast", ".2", "--rest-seconds", ".2",
                                 "--rest-timeout", "2", "--session", str(directory), "--max-trials", "1"])
        self.assertEqual(result, 0)
        saved = json.loads((directory / "session.json").read_text())
        self.assertEqual(sum(bool(t["attempts"]) for t in saved["trials"]), 1)
        first_trial = saved["trials"][0]
        original_csv = (directory / first_trial["attempts"][0]["csv"]).read_bytes()
        with self.hardware(bus, ["", ""]), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(suite.main(["--execute", "--resume", str(directory)]), 0)
        final = json.loads((directory / "session.json").read_text())
        self.assertTrue(all(t["attempts"][-1]["status"] == "completed" for t in final["trials"]))
        self.assertEqual((directory / first_trial["attempts"][0]["csv"]).read_bytes(), original_csv)
        with (directory / "summary.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 2)

    def test_resume_restores_config_order_and_reference(self):
        args = self.args("--session", str(self.session))
        _, _, saved = suite.load_session(args)
        saved["reference_count"] = 123456
        suite.save_json(self.session / "session.json", saved)
        resume = suite.parse_args(["--resume", str(self.session), "--max-trials", "2"])
        config, directory, loaded = suite.load_session(resume)
        self.assertEqual(config.resume, self.session)
        self.assertEqual(config.max_trials, 2)
        self.assertEqual(saved, loaded)
        self.assertEqual(directory, self.session)

    def test_resume_rejects_changed_targets_and_runtime(self):
        args = self.args("--session", str(self.session))
        _, _, saved = suite.load_session(args)
        saved["targets_sha256"] = "tampered"
        suite.save_json(self.session / "session.json", saved)
        with self.assertRaisesRegex(ValueError, "targets changed"):
            suite.load_session(suite.parse_args(["--resume", str(self.session)]))
        saved["source_sha256"] = {}
        suite.save_json(self.session / "session.json", saved)
        with self.assertRaisesRegex(ValueError, "scripts?|helpers"):
            suite.load_session(suite.parse_args(["--resume", str(self.session)]))

    def test_reference_prompt_occurs_after_confirmed_disable_and_checks_saved_phase(self):
        bus = ControlledCAN()
        bus.enabled = True
        def prompt(_):
            self.assertFalse(bus.enabled)
            return ""
        with self.hardware(bus), patch("builtins.input", side_effect=prompt):
            self.assertEqual(suite.capture_reference(self.args(), self.session), 1000)
        self.assertFalse(any(data[0] == 0xC0 for _, _, data in bus.frames))

    def test_preparation_shift_is_not_false_zero_torque_breakaway(self):
        bus = ControlledCAN()
        bus.preparation_shift = 40  # 0.015 rad: above onset threshold, within allowed 2-degree phase tolerance.
        result, error, _ = self.run_trial("breakaway", bus=bus)
        self.assertIsNone(error)
        self.assertEqual(result["core_start_count"], 1040)
        self.assertTrue(result["breakaway_censored"])
        self.assertIsNone(result["breakaway_detected_s"])

    def test_stall_crossing_end_is_not_successful_completion(self):
        bus = ControlledCAN()
        original, stalled = suite.ExperimentIO.guard, False
        def guard(controller):
            nonlocal stalled
            original(controller)
            if not stalled and bus.now - controller.motion_start >= .49:
                bus.now += .05
                stalled = True
        with self.hardware(bus), patch.object(suite.ExperimentIO, "guard", guard):
            result, error = suite.run_trial(self.args(), self.trial(), self.session, 1000, 1)
        self.assertIsInstance(error, RuntimeError)
        self.assertIn("end deadline", str(error))
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["disable_confirmed"] and result["watchdog_restored"])

    def test_cleanup_failure_does_not_hide_primary_current_failure(self):
        bus = ControlledCAN("restore_failed")
        bus.active_current_offset = 3000
        result, error, _ = self.run_trial(bus=bus)
        self.assertIn("current", str(error))
        self.assertIn("current", result["error"])
        self.assertIn("restoration", result["cleanup_error"])
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["watchdog_restored"])
        self.assertTrue(result["disable_confirmed"])

    def test_summary_error_preserves_recording_and_returns_failed_outcome(self):
        with patch.object(suite, "summarize", side_effect=ValueError("bad CSV analysis")):
            result, error, _ = self.run_trial()
        self.assertIn("bad CSV analysis", str(error))
        self.assertIn("bad CSV analysis", result["analysis_error"])
        self.assertEqual(result["status"], "failed")
        self.assertTrue((self.session / result["csv"]).exists())
        self.assertTrue((self.session / result["csv"]).with_suffix(".json").exists())

    def test_changed_motor_settings_reject_before_nonzero_or_zero_c0(self):
        bus = ControlledCAN()
        baseline = dict(Kt=bus.constant, B3=.001, B4=bus.b4_raw + 1, CD=bus.initial_cd.hex())
        with self.hardware(bus):
            result, error = suite.run_trial(self.args(), self.trial(), self.session, 1000, 1, baseline)
        self.assertIn("settings changed", str(error))
        self.assertFalse(any(data[0] == 0xC0 for _, _, data in bus.frames))
        self.assertTrue(result["disable_confirmed"] and result["watchdog_restored"])

    def test_reference_rechecks_disabled_status_after_manual_prompt(self):
        bus = ControlledCAN()
        def prompt(_):
            bus.enabled, bus.mode = True, 2  # Simulate external state change after initial CF.
            return ""
        with self.hardware(bus), patch("builtins.input", side_effect=prompt):
            with self.assertRaisesRegex(RuntimeError, "remain disabled"):
                suite.capture_reference(self.args(), self.session)
        self.assertFalse(bus.enabled)

    def test_session_lock_excludes_second_runner_and_releases(self):
        with suite.session_lock(self.session):
            with self.assertRaisesRegex(RuntimeError, "already running"):
                with suite.session_lock(self.session):
                    self.fail("lock allowed concurrent writer")
        with suite.session_lock(self.session):
            pass

    def test_resume_validates_arguments_and_trial_design(self):
        _, _, original = suite.load_session(self.args())
        for change in ("configuration", "design"):
            saved = json.loads(json.dumps(original))
            if change == "configuration":
                saved["arguments"]["command_current_limit_a"] = 10
            else:
                saved["trials"][0]["packet_rate"] = 500
            suite.save_json(self.session / "session.json", saved)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(ValueError, change):
                suite.load_session(suite.parse_args(["--resume", str(self.session)]))

    def test_resume_rejects_modified_completed_csv(self):
        args = self.args("--families", "pulse", "--angles", "0", "--rates", "50", "--repeats", "1",
                         "--pulse-torques", ".12")
        args, _, saved = suite.load_session(args)
        saved["reference_count"] = 1000
        trial = saved["trials"][0]
        with self.hardware(ControlledCAN()):
            result, error = suite.run_trial(args, trial, self.session, 1000, 1)
        self.assertIsNone(error)
        trial["attempts"].append(result)
        suite.save_json(self.session / "session.json", saved)
        suite.load_session(suite.parse_args(["--resume", str(self.session)]))
        path = self.session / result["csv"]
        with path.open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            suite.load_session(suite.parse_args(["--resume", str(self.session)]))

    def test_eof_during_placement_disables_and_records_failure(self):
        bus = ControlledCAN()
        with self.hardware(bus), patch("builtins.input", side_effect=EOFError("stdin closed")):
            result, error = suite.run_trial(self.args(), self.trial(), self.session, 1000, 1)
        self.assertIsInstance(error, EOFError)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["disable_confirmed"])

    def test_replay_plan_uses_validated_snapshot_during_long_session(self):
        args, _, _ = suite.load_session(self.args())
        with patch.object(suite.rear.shared, "load_target_rows", side_effect=AssertionError("reread mutable CSV")):
            self.assertTrue(suite.build_plan(args, self.trial()))

    def test_invalid_a3_encoding_refuses_motion(self):
        bus = ControlledCAN()
        enqueue = bus.enqueue
        def malformed(cid, payload, delay):
            if payload[0] == 0xA3:
                payload = payload[:1] + struct.pack("<H", 20000) + payload[3:]
            return enqueue(cid, payload, delay)
        with self.hardware(bus), patch.object(bus, "enqueue", side_effect=malformed):
            result, error = suite.run_trial(self.args(), self.trial(), self.session, 1000, 1)
        self.assertIn("invalid A3", str(error))
        self.assertFalse(any(data[0] == 0xC0 for _, _, data in bus.frames))
        self.assertTrue(result["disable_confirmed"])

    def test_core_torque_rms_is_not_diluted_by_zero_phases(self):
        path = self.session / "summary_input.csv"
        rows = []
        for index, (phase, command, actual) in enumerate((("pre_zero", 0, 0), ("pulse", .1, .05), ("coast", 0, 0))):
            rows.append(dict(record_type="drive_target", can_direction="local_tx", phase=phase, can_opcode=192,
                             replay_time_s=index, encoded_command_torque_nm=command, torque_nm=""))
            rows.append(dict(record_type="feedback", can_direction="bus_rx", phase="motion", can_opcode=161,
                             replay_time_s=index + .5, encoded_command_torque_nm="", torque_nm=actual))
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = suite.summarize(path, {})
        self.assertAlmostEqual(summary["A1_torque_error_rms_nm"], .05 / math.sqrt(3))
        self.assertAlmostEqual(summary["core_A1_torque_error_rms_nm"], .05)


if __name__ == "__main__":
    unittest.main()
