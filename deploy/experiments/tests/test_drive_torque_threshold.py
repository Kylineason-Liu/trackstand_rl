"""Validate constant-torque timing, raw CAN mapping and failure cleanup without hardware."""

import contextlib
from collections import Counter
import csv
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive_torque_threshold as threshold
from test_drive_policy_replay import DriveCAN
from test_steering_policy_replay import stall_after_command


class ThresholdTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.output = Path(self.folder.name) / "threshold.csv"

    def args(self, *flags):
        return threshold.parse_args(["--output", str(self.output), *flags])

    def execute(self, bus, *flags):
        args = self.args("--hold", ".06", "--rest", ".04", *flags)
        with patch.object(threshold.time, "monotonic", side_effect=lambda: bus.now), \
                patch.object(threshold.time, "time", side_effect=lambda: 1700000000 + bus.now), \
                patch.object(threshold.rear.mit.select, "select", side_effect=bus.select), \
                patch.object(threshold.rear.shared.socket, "socket", return_value=bus), \
                contextlib.redirect_stdout(io.StringIO()):
            threshold.execute(args, threshold.build_plan(args))
        return args

    def rows(self, suffix=""):
        path = self.output.with_name(self.output.stem + suffix + ".csv")
        with path.open() as stream:
            return list(csv.DictReader(stream))

    def metadata(self):
        return json.loads(self.output.with_suffix(".json").read_text())

    def test_default_plan_is_exactly_fifty_trials_and_348_seconds(self):
        args = self.args()
        plan = threshold.build_plan(args)
        counts = Counter((row["torque_level_nm"], row["direction"]) for row in plan)
        self.assertEqual(len(plan), 50)
        self.assertEqual(set(counts.values()), {5})
        self.assertEqual(len(counts), 10)
        self.assertEqual(sum(row["hold_s"] for row in plan), 250)
        self.assertEqual(sum(row["rest_after_s"] for row in plan), 98)
        self.assertEqual(plan[-1]["nominal_start_s"], 343)
        self.assertEqual(plan[-1]["rest_after_s"], 0)
        self.assertEqual([r["requested_torque_nm"] for r in plan[:4]], [.12, -.12, .13, -.13])

    def test_preview_never_opens_socket_or_files(self):
        with patch.object(threshold.rear.shared.socket, "socket", side_effect=AssertionError("CAN opened")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(threshold.main(["--output", str(self.output)]), 0)
        self.assertFalse(list(Path(self.folder.name).iterdir()))

    def test_all_fifty_trials_complete_and_restore_settings(self):
        bus = DriveCAN()
        self.execute(bus)
        meta, summaries = self.metadata(), self.rows("_summary")
        self.assertTrue(meta["completed"] and meta["disable_confirmed"] and meta["watchdog_restored"])
        self.assertEqual(meta["completed_trials"], 50)
        self.assertEqual(len(summaries), 50)
        self.assertFalse(bus.enabled)
        self.assertEqual(bus.cd, bus.initial_cd)
        self.assertEqual(bus.b3_ma, 1)
        self.assertEqual(bus.b4_raw, 100000)
        self.assertTrue(all(cid == 0x101 for _, cid, _ in bus.frames))
        self.assertFalse(any(data[0] in (0xB1, 0xC1, 0xC2, 0xC3) for _, _, data in bus.frames))
        self.assertTrue(all(float(row["hold_actual_s"]) >= .06 - 1e-8 for row in summaries))
        self.assertTrue(all(float(row["hold_actual_s"]) < .061 for row in summaries))
        self.assertTrue(all(int(row["A1_samples"]) >= 4 and int(row["A2_samples"]) >= 4
                            and int(row["A3_samples"]) >= 4 for row in summaries))

    def test_both_packet_rates_hold_constant_signed_current_and_rest_is_disabled(self):
        for rate in (50, 200):
            with self.subTest(rate=rate), tempfile.TemporaryDirectory() as folder:
                self.output = Path(folder) / "run.csv"
                bus = DriveCAN()
                self.execute(bus, "--packet-rate", str(rate), "--repeats", "1")
                rows = self.rows()
                commands = [r for r in rows if r["record_type"] == "drive_target"]
                starts = [next(r for r in commands if int(r["trial_id"]) == trial) for trial in range(1, 11)]
                for row in commands:
                    target = float(row["requested_torque_nm"])
                    payload = bytes.fromhex(row["payload_hex"])
                    self.assertEqual(struct.unpack("<i", payload[1:])[0],
                                     round(-target / float(row["torque_constant_nm_per_a"]) * 1000))
                    self.assertEqual(row["phase"], "torque")
                for first, second in zip(starts, starts[1:]):
                    lo, hi = float(first["elapsed_s"]), float(second["elapsed_s"])
                    disabled = next(r for r in rows if r["can_direction"] == "local_tx"
                                    and r["payload_hex"] == "CF" and lo < float(r["elapsed_s"]) < hi)
                    self.assertAlmostEqual(hi - float(disabled["elapsed_s"]), .04, delta=.0002)
                    self.assertFalse(any(r["can_opcode"] == "192" and r["can_direction"] == "local_tx"
                                         and float(disabled["elapsed_s"]) < float(r["elapsed_s"]) < hi
                                         for r in rows))

    def test_coasting_start_is_logged_without_extra_wait_or_velocity_command(self):
        class CoastingCAN(DriveCAN):
            moved = False

            def send(self, frame):
                _, length, padded = threshold.rear.mit.CAN_FRAME.unpack(frame)
                if padded[0] == 0xC0:
                    self.moved = True
                if padded[:length] == b"\xa2" and self.moved:
                    self.frames.append((self.now, 0x101, b"\xa2"))
                    self.enqueue(1, b"\xa2" + struct.pack("<i", -3000), self.reply_delay)
                    return len(frame)
                return super().send(frame)

        self.execute(CoastingCAN(), "--repeats", "1")
        starts = [float(row["start_velocity_rad_s"]) for row in self.rows("_summary")]
        self.assertEqual(starts[0], 0)
        self.assertTrue(all(value > 3 for value in starts[1:]))
        self.assertTrue(self.metadata()["completed"])

    def test_global_position_retains_first_startup_zero_if_later_sample_differs(self):
        class SlightMovementCAN(DriveCAN):
            position_reads = 0

            def send(self, frame):
                _, size, padded = threshold.rear.mit.CAN_FRAME.unpack(frame)
                if padded[:size] == b"\xa3":
                    self.position_reads += 1
                    count = 1000 if self.position_reads == 1 else 1002
                    self.frames.append((self.now, 0x101, b"\xa3"))
                    self.enqueue(1, b"\xa3" + struct.pack("<Hi", count, count), self.reply_delay)
                    return len(frame)
                return super().send(frame)

        self.execute(SlightMovementCAN(), "--repeats", "1")
        self.assertEqual(self.metadata()["encoder_zero_count"], 1000)
        positions = [r for r in self.rows() if r["phase"] == "torque" and r["can_opcode"] == "163"
                     and r["can_direction"] == "bus_rx"]
        for row in positions:
            self.assertAlmostEqual(float(row["position_unwrapped_rad"]), -2 * 2 * threshold.math.pi / 16384)
            self.assertEqual(float(row["trial_position_rad"]), 0.)

    def test_fault_current_and_missing_feedback_disable_and_preserve_partial_logs(self):
        for fault in ("motor_fault", "hot", "send_failed"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as folder:
                self.output = Path(folder) / "run.csv"
                bus = DriveCAN(fault)
                with self.assertRaises((RuntimeError, OSError)):
                    self.execute(bus, "--repeats", "1")
                self.assertFalse(bus.enabled)
                self.assertFalse(self.metadata()["completed"])
                self.assertEqual(bus.cd, bus.initial_cd)
        with tempfile.TemporaryDirectory() as folder:
            self.output = Path(folder) / "run.csv"
            bus = DriveCAN()
            bus.active_current_offset = 3000
            with self.assertRaisesRegex(RuntimeError, "current"):
                self.execute(bus, "--repeats", "1")
            self.assertFalse(bus.enabled)
            self.assertTrue(self.metadata()["disable_confirmed"])
        with tempfile.TemporaryDirectory() as folder:
            self.output = Path(folder) / "run.csv"
            bus = DriveCAN()
            bus.drop_aux.add(0xA1)
            with self.assertRaisesRegex(RuntimeError, "stale rear A1"):
                self.execute(bus, "--repeats", "1", "--hold", ".3")
            self.assertFalse(bus.enabled)
            self.assertTrue(self.metadata()["disable_confirmed"])

    def test_host_stall_crossing_hold_end_cannot_silently_extend_the_trial(self):
        bus = DriveCAN()
        with stall_after_command(bus, threshold.ThresholdIO, .10, 0x101, 0xC0):
            with self.assertRaisesRegex(RuntimeError, "interval-end deadline"):
                self.execute(bus, "--repeats", "1")
        self.assertFalse(bus.enabled)
        self.assertFalse(self.metadata()["completed"])
        self.assertTrue(self.metadata()["disable_confirmed"])

    def test_logging_failure_after_command_still_disables(self):
        bus = DriveCAN()
        original = threshold.CsvLog.emit

        def emit(logger, row):
            if row["record_type"] == "drive_target":
                raise RuntimeError("simulated recording failure")
            return original(logger, row)

        with patch.object(threshold.CsvLog, "emit", emit):
            with self.assertRaisesRegex(RuntimeError, "recording failure"):
                self.execute(bus, "--repeats", "1")
        self.assertFalse(bus.enabled)
        self.assertEqual(bus.cd, bus.initial_cd)
        self.assertTrue(self.metadata()["disable_confirmed"])

    def test_rollover_is_continuous_and_trial_reference_is_independent(self):
        args = self.args()
        motor = threshold.rear.mit.Motor("drive", 1, 1., -1, -1)
        motor.zero_count = 2**31 - 2
        logger = unittest.mock.Mock()
        controller = threshold.ThresholdIO(unittest.mock.Mock(), logger, motor, args)
        controller.phase = "startup"
        for count in (2**31 - 2, -(2**31) + 1):
            row = controller.row(motor, "can_rx")
            payload = b"\xa3" + struct.pack("<Hi", count % 16384, count)
            self.assertEqual(controller.decode_reply(motor, payload, 1., row), "")
        self.assertEqual(controller.total_count, 2**31 + 1)
        controller.startup_total = 2**31 - 2
        controller.trial_zero = controller.total_count
        row = controller.row(motor, "can_rx")
        controller.decode_reply(motor, b"\xa3" + struct.pack("<Hi", 3, -(2**31) + 3), 2., row)
        self.assertAlmostEqual(row["position_unwrapped_rad"], -5 * 2 * threshold.math.pi / 16384)
        self.assertAlmostEqual(row["trial_position_rad"], -2 * 2 * threshold.math.pi / 16384)

    def test_previous_phase_status_is_not_treated_as_fresh_mode_confirmation(self):
        motor = threshold.rear.mit.Motor("drive", 1, 1., -1, -1)
        controller = threshold.ThresholdIO(unittest.mock.Mock(), unittest.mock.Mock(), motor, self.args())
        controller.phase, controller.phase_start = "torque", 5.
        controller.pending = threshold.rear.mit.PendingRead(motor, 0xAE, 5.1, 5.12)
        disabled = bytes([0xAE]) + struct.pack("<HH", 2400, 0) + bytes([30, 0, 0])
        self.assertEqual(controller.decode_reply(motor, disabled, 5.05, controller.row(motor, "can_rx")), "")
        self.assertIn("mode differs", controller.decode_reply(motor, disabled, 5.11,
                                                              controller.row(motor, "can_rx")))

    def test_existing_outputs_are_never_overwritten(self):
        self.output.write_text("keep\n")
        with self.assertRaises(FileExistsError):
            self.execute(DriveCAN())
        self.assertEqual(self.output.read_text(), "keep\n")


if __name__ == "__main__":
    unittest.main()
