"""Data-boundary and numerical checks on the real replay report; no CAN access."""

import json
import math
import re
import unittest

import numpy as np

import analyze as report


class HardwareReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = [report.Run(stem) for stem in report.FILES]
        cls.statistics = json.loads((report.HERE / "statistics.json").read_text())

    def test_all_twelve_files_and_runtime_hashes_are_validated(self):
        self.assertEqual(len(self.statistics["input_sha256"]), 12)
        for name, digest in self.statistics["input_sha256"].items():
            self.assertEqual(report.digest(report.ROOT / name), digest)
        for run in self.runs:
            run.validate()

    def test_historical_runtime_snapshots_match_the_recorded_versions(self):
        snapshots = report.HERE / "runtime_snapshot"
        for run in self.runs:
            hashes = run.meta.get("source_sha256", {
                "steering_policy_replay.py": run.meta.get("script_sha256"),
                "steering_sine_mit_io.py": run.meta.get("transport_sha256"),
            })
            for name, digest in hashes.items():
                self.assertEqual(report.digest(snapshots/name), digest)

    def test_only_steering200_is_incomplete_and_no_tail_is_invented(self):
        self.assertEqual([run.meta["completed"] for run in self.runs], [True, False, True, True])
        self.assertEqual([len(run.tx) for run in self.runs], [400, 1074, 350, 1400])
        partial = self.runs[1]
        self.assertEqual(partial.tx[-1]["target_index"], "268")
        self.assertEqual(partial.tx[-1]["packet_index"], "1073")
        self.assertLess(partial.qtime.max(), partial.end)
        self.assertLess(partial.ttime.max(), partial.end)
        self.assertLess(partial.end, 5.38)

    def test_rate_pairs_are_exactly_equal_at_input_and_payload_levels(self):
        pairs = report.compare_pairs(self.runs)
        self.assertTrue(all(item["plans_exactly_equal"] for item in pairs.values()))
        self.assertEqual(pairs["steering"]["full_matched_50hz_intervals"], 268)
        self.assertEqual(pairs["rear"]["full_matched_50hz_intervals"], 350)

    def test_initial_f1_endpoint_and_a3_outside_range_are_real_raw_bytes(self):
        for run in self.runs[:2]:
            first = next(row for row in run.rows if row["can_direction"] == "bus_rx"
                         and row["can_opcode"] == "241")
            self.assertEqual(bytes.fromhex(first["mit_payload_hex"])[1:3], b"\xff\xff")
            self.assertEqual(run.meta["native_zero_rad"], 3.)
            self.assertGreater(run.meta["encoder_zero_count"] * report.RAD_PER_COUNT, 3.)
            m = self.statistics["runs"][run.stem]["zero_reference"]
            self.assertLess(abs(m["residual_after_predicted_offset_rad"]["mean"]), .0002)
            self.assertLess(m["residual_after_predicted_offset_rad"]["rms"], .001)

    def test_absolute_physical_error_is_not_replaced_by_f1_or_demeaned_error(self):
        for run in self.runs[:2]:
            m = self.statistics["runs"][run.stem]
            self.assertGreater(math.degrees(m["tracking"]["rms"]), 9.)
            self.assertLess(math.degrees(m["zero_reference"]["F1_based_tracking"]["rms"]), 1.)
            self.assertLess(math.degrees(m["tracking"]["demeaned_rms"]), 1.)

    def test_main_error_metrics_match_independent_reference_walk(self):
        for run in self.runs:
            low, high = run.metric_window
            rows = [row for row in run.rows if row["phase"] == "motion" and row["can_direction"] == "bus_rx"
                    and row["can_opcode"] == str(0xA1 if run.drive else 0xA3)
                    and low <= float(row["replay_time_s"]) < high]
            errors = []
            index = 0
            for row in rows:
                at = float(row["replay_time_s"])
                while index + 1 < len(run.tx) and float(run.tx[index+1]["replay_time_s"]) <= at:
                    index += 1
                key = "torque_nm" if run.drive else "position_unwrapped_rad"
                target = "encoded_command_torque_nm" if run.drive else "mit_target_position_rad"
                errors.append(float(row[key]) - float(run.tx[index][target]))
            expected = self.statistics["runs"][run.stem]["tracking"]
            self.assertEqual(expected["n"], len(errors))
            self.assertAlmostEqual(expected["rms"], math.sqrt(sum(e*e for e in errors)/len(errors)), places=12)
            self.assertAlmostEqual(expected["bias"], sum(errors)/len(errors), places=12)

    def test_rear_accuracy_uses_equal_independent_a1_streams(self):
        counts = []
        for run in self.runs[2:]:
            m = self.statistics["runs"][run.stem]
            counts.append(m["tracking"]["n"])
            self.assertLess(m["tracking"]["rms"], .02)
            self.assertLess(abs(m["tracking"]["bias"]), .001)
            self.assertGreater(m["near_zero_speed_core_fraction"], .7)
        self.assertEqual(counts, [495, 495])

    def test_aligned_exports_preserve_every_fresh_sample_and_time(self):
        for run in self.runs:
            name = ("rear" if run.drive else "steering") + f"_{run.meta['packet_hz']}hz_aligned.csv"
            rows = report.read_csv(report.HERE/name)
            source_time, actual = (run.ttime, run.torque) if run.drive else (run.qtime, run.q)
            np.testing.assert_array_equal([float(r["replay_time_s"]) for r in rows], source_time)
            np.testing.assert_array_equal([float(r["actual"]) for r in rows], actual)

    def test_zero_order_hold_never_uses_a_future_command(self):
        np.testing.assert_array_equal(report.held(np.array([0., 1., 2.]), np.array([3., 4., 5.]),
                                                 np.array([.99, 1., 1.01])), [3., 4., 4.])
        with self.assertRaises(ValueError):
            report.held(np.array([1., 2.]), np.array([3., 4.]), np.array([.9]))

    def test_alignment_recovers_known_synthetic_lag_despite_constant_angle_offset(self):
        run = object.__new__(report.Run)
        run.drive = False
        run.ct = np.arange(151)*.02
        run.cv = np.random.default_rng(42).normal(size=len(run.ct))
        run.qtime = np.arange(.1, 2.9, .0013)
        run.q = report.held(run.ct, run.cv, run.qtime-.02) + .2
        result = run.alignment((.3, 2.7))
        self.assertAlmostEqual(result["best_shift_ms"], 20., delta=.2)
        self.assertLess(result["minimum_rms"], 1e-12)

    def test_existing_simulator_inputs_match_replayed_plans(self):
        with np.load(report.SIM/"inputs.npz") as bank:
            for run in self.runs:
                key = "airborne_rear_nm" if run.drive else "airborne_target_rad"
                column = "requested_torque_nm" if run.drive else "target_rad"
                np.testing.assert_array_equal([float(r[column]) for r in run.plan],
                                              bank[key][:len(run.plan), 4])

    def test_all_figures_and_local_links_exist(self):
        for name in report.FIGURES:
            for extension in ("png", "pdf", "svg"):
                self.assertGreater((report.HERE/f"{name}.{extension}").stat().st_size, 5000)
        for name in ("README.md", "index.html"):
            content = (report.HERE/name).read_text()
            targets = re.findall(r'\]\(([^)]+)\)|(?:href|src)="([^"]+)"', content)
            for match in targets:
                target = next(value for value in match if value)
                if target == "verification.json":
                    continue  # Written after this suite completes; checked in final verification.
                self.assertTrue((report.HERE/target).exists(), target)


if __name__ == "__main__":
    unittest.main()
