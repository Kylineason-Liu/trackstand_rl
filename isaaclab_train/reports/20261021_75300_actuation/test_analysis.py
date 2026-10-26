"""Independent checks of the generated 75300 figure bundle and capture contract."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("analysis75300", HERE / "analyze.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class TestAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads((HERE / "summary.json").read_text())
        cls.data = {mode: analysis.load(mode) for mode in analysis.MODES}

    def test_reset_segments_have_no_empty_or_cross_reset_interval(self):
        self.assertEqual(analysis.segments(np.array([False, True, False, True])), [(0, 2), (2, 4)])
        self.assertEqual(analysis.segments(np.zeros(4, dtype=bool)), [(0, 4)])
        self.assertEqual(analysis.segments(np.ones(4, dtype=bool)), [(0, 1), (1, 2), (2, 3), (3, 4)])

    def test_spawn_times_include_initial_and_capture_boundary_reset(self):
        np.testing.assert_allclose(analysis.spawn_times(np.array([False, True, False, True])), [0, 0.04, 0.08])
        np.testing.assert_array_equal(analysis.spawn_times(np.zeros(4, dtype=bool)), [0])
        np.testing.assert_allclose(analysis.spawn_times(np.ones(4, dtype=bool)), [0, 0.02, 0.04, 0.06, 0.08])
        np.testing.assert_array_equal(analysis.spawn_times(self.data["no_dr"][1]["done"][:, 0]), [0, 30, 60])

    def test_red_spawn_artists_and_full_range_axes_match_saved_capture(self):
        with patch.object(analysis, "save") as save:
            settings = analysis.traces(self.data, "test", (0, 60), dict.fromkeys(analysis.MODES, 0), fit_y=True)
        figure = save.call_args.args[0]
        try:
            for column, mode in enumerate(analysis.MODES):
                expected = analysis.spawn_times(self.data[mode][1]["done"][:, 0])
                for row, motor in enumerate(("steering", "rear")):
                    axis = figure.axes[row * 2 + column]
                    markers = [line for line in axis.lines if line.get_color() == "#d62728"]
                    np.testing.assert_array_equal([line.get_xdata()[0] for line in markers], expected)
                    self.assertTrue(all(line.get_xdata()[0] == line.get_xdata()[1] for line in markers))
                    self.assertEqual(settings[mode]["panels"][motor]["samples_outside_y_view"], 0)
                    np.testing.assert_array_equal(axis.get_ylim(), settings[mode]["panels"][motor]["ylim"])
        finally:
            analysis.plt.close(figure)

    def test_saved_vertical_zoom_is_explicit_and_preserves_original_extrema(self):
        settings = json.loads((HERE / "demand_plot_settings.json").read_text())
        self.assertEqual(settings["input_sha256"], self.summary["input_sha256"])
        for mode in analysis.MODES:
            np.testing.assert_array_equal(settings["full_range"][mode]["spawn_times_s"],
                                          analysis.spawn_times(self.data[mode][1]["done"][:, 0]))
            for motor in ("steering", "rear"):
                full = settings["full_range"][mode]["panels"][motor]
                zoom = settings["vertical_zoom"][mode]["panels"][motor]
                self.assertEqual(full["plotted_min_max"], zoom["plotted_min_max"])
                self.assertEqual(full["samples_outside_y_view"], 0)
                if mode == "no_dr":
                    self.assertEqual(zoom["axis_fitting_excludes_first_s_each_episode"], 0.5)
                    self.assertLess(np.ptp(zoom["ylim"]), np.ptp(full["ylim"]))
                    self.assertGreater(zoom["samples_outside_y_view"], 0)
                else:
                    self.assertEqual(full, zoom)

    def test_histogram_includes_exact_horizon_and_excludes_unfinished_tail(self):
        done = np.zeros((3000, 1), dtype=bool)
        done[1499, 0] = True
        data = {"done": done, "truncated": done.copy()}
        lengths, timeouts, counts, edges = analysis.episode_statistics(data)
        np.testing.assert_array_equal(lengths, [1500])
        self.assertTrue(timeouts[0])
        self.assertEqual(counts.sum(), 1)
        self.assertEqual(counts[-1], 1)
        self.assertEqual(len(counts), 10)
        np.testing.assert_array_equal(edges, np.arange(0, 31, 3))

    def test_actual_inputs_match_native_motor_feedback_and_steering_encoding(self):
        for mode, (_, data) in self.data.items():
            with self.subTest(mode=mode):
                np.testing.assert_array_equal(data["observation"][..., 5:7], data["position_before_rad"][:, 0])
                np.testing.assert_array_equal(data["observation"][..., 7:9], data["velocity_before_rad_s"][:, 0])
                np.testing.assert_allclose(data["observation"][..., 9], np.sin(data["observation"][..., 5]), atol=1e-7)
                np.testing.assert_allclose(data["observation"][..., 10], np.cos(data["observation"][..., 5]), atol=1e-7)

    def test_done_signals_and_episode_counts_are_consistent(self):
        for mode, (meta, data) in self.data.items():
            with self.subTest(mode=mode):
                np.testing.assert_array_equal(data["done"], np.any(data["termination_flags"], axis=-1))
                column = meta["termination_names"].index("time_out")
                np.testing.assert_array_equal(data["truncated"], data["termination_flags"][..., column])
                lengths, timeouts, counts, _ = analysis.episode_statistics(data)
                self.assertEqual(len(lengths), data["done"].sum())
                self.assertEqual(counts.sum(), len(lengths))
                self.assertEqual(counts.tolist(), self.summary[mode]["episode_histogram_counts"])
                self.assertAlmostEqual(float(timeouts.mean()), self.summary[mode]["timeout_rate"])

    def test_all_completed_episodes_reset_steering_state_on_next_action(self):
        for _, data in self.data.values():
            starts = np.r_[np.ones((1, 64), dtype=bool), data["done"][:-1]]
            np.testing.assert_array_equal(data["position_before_rad"][:, 0][starts], 0)
            np.testing.assert_array_equal(data["velocity_before_rad_s"][:, 0][starts], 0)

    def test_csv_timebase_and_all_plotted_motor_values_match_full_arrays(self):
        for mode, (_, data) in self.data.items():
            with self.subTest(mode=mode):
                with (HERE / f"{mode}_env0_traces.csv").open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 24000)
                np.testing.assert_allclose([float(row["command_time_s"]) for row in rows], np.arange(24000) * 0.0025,
                                           rtol=0, atol=1e-10)
                self.assertAlmostEqual(float(rows[-1]["response_time_s"]), 60.0)
                for field, expected in (
                    ("held_steer_rad", data["held_steering_target_rad"][:, :, 0].reshape(-1)),
                    ("steer_actual_rad", data["position_after_rad"][:, :, 0, 0].reshape(-1)),
                    ("rear_submitted_nm", data["submitted_motor_torque_nm"][:, :, 0, 1].reshape(-1)),
                    ("rear_solver_effort_nm", data["projected_joint_effort_nm"][:, :, 0, 1].reshape(-1)),
                ):
                    np.testing.assert_allclose([float(row[field]) for row in rows], expected, rtol=1e-7, atol=1e-8)

    def test_input_hashes_and_checkpoint_hashes_match(self):
        for name, expected in self.summary["input_sha256"].items():
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), expected)
        for mode, (meta, _) in self.data.items():
            self.assertEqual(hashlib.sha256((HERE / mode / "model_75300.pt").read_bytes()).hexdigest(),
                             meta["checkpoint_sha256"])

    def test_all_eleven_figures_exist_in_png_pdf_svg(self):
        self.assertEqual(len(self.summary["figures"]), 11)
        page = (HERE / "index.html").read_text()
        for name in self.summary["figures"]:
            png = (HERE / f"{name}.png").read_bytes()
            self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
            self.assertGreater(min(struct.unpack(">II", png[16:24])), 600)
            self.assertTrue((HERE / f"{name}.pdf").read_bytes().startswith(b"%PDF"))
            ET.parse(HERE / f"{name}.svg")
            self.assertIn(f"{name}.png", page)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestAnalysis)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    (HERE / "verification.json").write_text(json.dumps({
        "passed": result.wasSuccessful(), "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "checks": [name for name in dir(TestAnalysis) if name.startswith("test_")],
    }, indent=2) + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
