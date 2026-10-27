"""Actual startup-target and virtual-CAN checks; never opens real hardware."""

import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import steering_initial_transient_replay as initial
from test_steering_policy_replay import ReplayCAN


class InitialTransientTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)

    def test_default_plan_starts_at_spawn_without_artificial_preparation(self):
        args = initial.parse_args([])
        plan, _ = initial.replay.load_plan(args)
        with args.targets.open() as stream:
            source = list(csv.DictReader(stream))
        self.assertEqual((args.start, args.ramp, args.pre_hold, args.motor_id), (0., 0., 0., 2))
        self.assertEqual((args.kp, args.kd, args.current_abort_a, args.max_lateness_ms), (10., .2, 3., 3.))
        self.assertEqual(len(plan), 300)
        self.assertEqual([row["phase"] for row in plan], ["core"] * 250 + ["post_hold"] * 50)
        self.assertEqual([row["target_rad"] for row in plan[:250]],
                         [float(row["scaled_steering_rad"]) for row in source[:250]])
        self.assertEqual(plan[0]["source_time_s"], 0.)
        self.assertEqual(plan[249]["source_time_s"], 4.98)
        self.assertAlmostEqual(plan[0]["target_rad"], .3573741018772125)
        self.assertTrue(all(row["target_rad"] == plan[249]["target_rad"] for row in plan[250:]))
        self.assertIn("steering_initial_transient", args.output.name)

    def test_full_episode_and_conditioned_level_are_supported_without_second_slew(self):
        args = initial.parse_args(["--duration", "30", "--post-hold", "0", "--level", "conditioned"])
        plan, _ = initial.replay.load_plan(args)
        with args.targets.open() as stream:
            source = list(csv.DictReader(stream))
        self.assertEqual(len(plan), 1500)
        self.assertFalse(args.v1_conditioning)
        self.assertEqual([row["target_rad"] for row in plan],
                         [float(row["conditioned_steering_rad"]) for row in source])

    def test_preview_has_no_can_or_output_and_help_describes_startup(self):
        with patch.object(initial.replay.socket, "socket", side_effect=AssertionError("hardware access")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(initial.main(["--output", str(self.folder / "preview.csv")]), 0)
        self.assertIn("Source [0, 5)", output.getvalue())
        self.assertIn("No preparation ramp/pre-hold", output.getvalue())
        self.assertEqual(list(self.folder.iterdir()), [])
        with self.assertRaises(SystemExit) as exit_code, contextlib.redirect_stdout(io.StringIO()) as help_text:
            initial.parse_args(["--help"])
        self.assertEqual(exit_code.exception.code, 0)
        self.assertIn("source time zero", help_text.getvalue())

    def test_cannot_select_rear_or_skip_startup_and_invalid_times_are_rejected(self):
        for flags in (["--motor-id", "1"], ["--start", "12.5"], ["--ramp", "1"],
                      ["--duration", ".03"], ["--duration", "0"], ["--post-hold", "-1"]):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    initial.parse_args(flags)

    def test_complete_both_packet_rates_with_corrected_zero_and_steering_only_frames(self):
        payloads = {}
        for rate in (50, 200):
            with self.subTest(rate=rate):
                bus = ReplayCAN()
                bus.configs[2], bus.positions[2], bus.initial_count = (30, 4500, 1800), 65535, 8282
                path = self.folder / f"initial_{rate}.csv"
                with patch.object(initial.replay.time, "monotonic", side_effect=lambda: bus.now), \
                        patch.object(initial.replay.time, "time", side_effect=lambda: 1700000000 + bus.now), \
                        patch.object(initial.replay.mit.select, "select", side_effect=bus.select), \
                        patch.object(initial.replay.socket, "socket", return_value=bus), \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(initial.main(["--execute", "--packet-rate", str(rate),
                                                   "--output", str(path)]), 0)
                metadata = json.loads(path.with_suffix(".json").read_text())
                self.assertTrue(metadata["completed"] and metadata["disable_confirmed"])
                self.assertTrue(metadata["steering_origin_setup"]["verified"])
                self.assertEqual(metadata["encoder_zero_count"], 0)
                self.assertFalse(metadata["rear_motor_commanded"])
                self.assertEqual(metadata["arguments"]["entry_point"], Path(initial.__file__).name)
                self.assertEqual(metadata["arguments"]["entry_point_sha256"],
                                 hashlib.sha256(Path(initial.__file__).read_bytes()).hexdigest())
                with path.open() as stream:
                    targets = [row for row in csv.DictReader(stream) if row["record_type"] == "mit_target"]
                repeats = rate // 50
                self.assertEqual(len(targets), 300 * repeats)
                self.assertEqual(targets[0]["phase"], "core")
                self.assertEqual(float(targets[0]["source_time_s"]), 0.)
                grouped = [targets[i:i + repeats] for i in range(0, len(targets), repeats)]
                self.assertTrue(all(len({row["mit_payload_hex"] for row in group}) == 1 for group in grouped))
                payloads[rate] = [group[0]["mit_payload_hex"] for group in grouped]
                self.assertEqual(bus.origin_resets, 1)
                self.assertTrue(all(cid in (0x102, 0x502) for _, cid, _ in bus.frames))
                self.assertTrue(all(len(data) == 1 for _, cid, data in bus.frames if cid == 0x102))
        self.assertEqual(payloads[50], payloads[200])

    def test_failed_zero_or_overcurrent_retains_cleanup_and_partial_metadata(self):
        for fault in ("missing_b1", "current"):
            with self.subTest(fault=fault):
                bus = ReplayCAN("missing_b1" if fault == "missing_b1" else None)
                if fault == "current":
                    bus.active_current_ma = 3500
                path = self.folder / f"failure_{fault}.csv"
                with patch.object(initial.replay.time, "monotonic", side_effect=lambda: bus.now), \
                        patch.object(initial.replay.time, "time", side_effect=lambda: 1700000000 + bus.now), \
                        patch.object(initial.replay.mit.select, "select", side_effect=bus.select), \
                        patch.object(initial.replay.socket, "socket", return_value=bus), \
                        contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(initial.main(["--execute", "--duration", ".1", "--post-hold", "0",
                                                   "--output", str(path)]), 1)
                meta = json.loads(path.with_suffix(".json").read_text())
                self.assertFalse(meta["completed"])
                self.assertTrue(meta["disable_confirmed"])
                self.assertFalse(bus.enabled)
                if fault == "missing_b1":
                    self.assertFalse(any(cid == 0x502 for _, cid, _ in bus.frames))


if __name__ == "__main__":
    unittest.main()
