"""Independent data/timebase, plot-marker, extrema and artifact checks."""

import csv
import importlib.util
import json
from pathlib import Path
import re
import unittest

import numpy as np
from PIL import Image

SPEC = importlib.util.spec_from_file_location("comparison_analysis", Path(__file__).with_name("analyze.py"))
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


class PlotVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datasets = {tag: a.load(tag) for tag in a.TAGS}

    def test_real_data_pass_command_observation_and_old_mujoco_checks(self):
        for tag, (meta, _) in self.datasets.items():
            self.assertTrue(meta["dr"] and meta["training_dr"] and not meta["pure_survival"])
            if tag != "isaac400":
                self.assertTrue(meta["all_13_original_mujoco_arrays_byte_identical"])

    def test_completed_episode_statistics(self):
        expected = {"isaac400": (130, 108), "mujoco400": (270, 74), "mujoco800": (149, 92)}
        for tag, pair in self.datasets.items():
            result = a.describe(*pair)
            self.assertEqual((result["completed_episodes"], result["timeouts"]), expected[tag])
            self.assertAlmostEqual(result["timeout_rate"], expected[tag][1] / expected[tag][0])

    def test_segments_and_spawns_include_initial_and_final_boundary(self):
        done = np.array([False, True, False, True])
        self.assertEqual(a.segments(done), [(0, 2), (2, 4)])
        np.testing.assert_array_equal(a.spawn_times(done), [0, .04, .08])

    def test_detail_window_is_inside_one_episode(self):
        for _, data in self.datasets.values():
            for env in range(64):
                low, high = a.detail_window(data["done"][:, env])
                first, last = round(low / .02), round(high / .02)
                self.assertGreater(last, first)
                self.assertLessEqual(last - first, 250)
                self.assertFalse(data["done"][first:last - 1, env].any())

    def test_every_environment_spawn_markers_are_exact(self):
        settings = json.loads((a.HERE / "plot_settings.json").read_text())
        for kind in ("all64_demand_vs_actuation", "all64_observation_traces"):
            for tag, (_, data) in self.datasets.items():
                for env in range(64):
                    expected = a.spawn_times(data["done"][:, env])
                    for panel in settings[kind][str(env)][tag]:
                        np.testing.assert_allclose(panel["spawn_times_s"], expected, rtol=0, atol=1e-12)

    def test_red_spawn_artists_in_actual_rendering(self):
        for tag, (meta, data) in self.datasets.items():
            for env in (0, 63):
                fig, ax = a.plt.subplots()
                a.plot_panel(ax, tag, meta, data, "observation", 5, env, (0, 60))
                lines = [line for line in ax.lines if line.get_color() == "#d62728"]
                actual = [line.get_xdata()[0] for line in lines]
                np.testing.assert_array_equal(actual, a.spawn_times(data["done"][:, env]))
                self.assertTrue(all(line.get_linestyle() == "-" for line in lines))
                a.plt.close(fig)

    def test_all_observation_extrema_match_every_environment_raw_recording(self):
        settings = json.loads((a.HERE / "plot_settings.json").read_text())["all64_observation_traces"]
        for tag, (_, data) in self.datasets.items():
            for env in range(64):
                for channel, (label, _) in enumerate(a.OBSERVATIONS):
                    saved = settings[str(env)][tag][channel]
                    values = data["observation"][:, env, channel]
                    self.assertEqual(saved["extrema"][label]["minimum"], float(values.min()))
                    self.assertEqual(saved["extrema"][label]["maximum"], float(values.max()))
                    self.assertLess(saved["ylim"][0], values.min())
                    self.assertGreater(saved["ylim"][1], values.max())

    def test_full_and_detail_axes_cover_all_saved_extrema(self):
        settings = json.loads((a.HERE / "plot_settings.json").read_text())
        for name in json.loads((a.HERE / "summary.json").read_text())["figures"]:
            for panels in settings[name].values():
                for panel in panels:
                    self.assertLess(panel["ylim"][0], panel["panel_min"])
                    self.assertGreater(panel["ylim"][1], panel["panel_max"])
                    self.assertEqual(panel["samples_outside_y_view"], 0)
                    for trace in panel["extrema"].values():
                        self.assertGreater(trace["samples"], 0)
                        self.assertLessEqual(trace["minimum"], trace["maximum"])

    def test_min_max_labels_and_timebase_of_position(self):
        for tag, (meta, data) in self.datasets.items():
            fig, ax = a.plt.subplots()
            result = a.plot_panel(ax, tag, meta, data, "demand", 0, 0, (0, 60))
            labels = ax.get_legend_handles_labels()[1]
            self.assertEqual(sum("[min " in value and ", max " in value for value in labels), 3)
            trace = result["extrema"]["Actual steering position"]
            start = data["position_before_rad"][0, 0, 0, 0]
            self.assertEqual(trace["minimum"], min(start, data["position_after_rad"][:, :, 0, 0].min()))
            self.assertEqual(trace["maximum"], max(start, data["position_after_rad"][:, :, 0, 0].max()))
            self.assertTrue(any("Panel min" in text.get_text() for text in ax.texts))
            self.assertEqual(ax.get_legend()._loc, 4)  # Matplotlib lower right.
            a.plt.close(fig)

    def test_recovered_isaac_draws_are_backed_by_exact_original_array_equality(self):
        recovery = a.dr_annotations.metadata()["isaac_recovery"]
        self.assertTrue(recovery["all_original_arrays_byte_identical"])
        self.assertEqual(len(recovery["array_checks"]), 18)
        self.assertTrue(all(recovery["array_checks"].values()))
        self.assertEqual(recovery["original_sha256"], a.sha(a.HERE / "isaac400/motor_samples.npz"))
        self.assertEqual(recovery["repeat_sha256"],
                         a.sha(a.HERE / "dr_metadata/isaac_repeat/motor_samples.npz"))

    def test_all_sampled_episodes_match_recorded_resets_and_actual_gains(self):
        for tag, (_, data) in self.datasets.items():
            run = a.dr_annotations.metadata()["runs"][tag]
            self.assertEqual(run["snapshot_sha256"], a.sha(a.HERE / run["snapshot_file"]))
            for env in range(64):
                records = a.dr_annotations.episodes_in_view(tag, env, (0, 60))
                expected = a.segments(data["done"][:, env])
                self.assertEqual(len(records), len(expected))
                for item, (lo, hi) in zip(records, expected):
                    self.assertEqual([item["start_s"], item["stop_s"]], [lo * .02, hi * .02])
                    for key in ("kp", "kd"):
                        np.testing.assert_array_equal(item[key], data[f"controller_{key}"][lo, 0, env])
                        np.testing.assert_array_equal(item[key], data[f"controller_{key}"][hi - 1, -1, env])
                    self.assertTrue(np.all((np.array(item["mass_scale"]) >= .8)
                                           & (np.array(item["mass_scale"]) <= 1.2)))
                    self.assertLessEqual(abs(item["initial_roll_rad"]), .1 + 1e-7)
                    self.assertLessEqual(abs(item["initial_roll_rate_rad_s"]), .25)
                    self.assertTrue(np.all(np.array(item["contact_dynamic_friction"])
                                           <= np.array(item["contact_static_friction"])))

    def test_annotations_preserve_physical_axes_geometry_and_data(self):
        fig, ax = a.plt.subplots(figsize=(21, 8.4))
        line, = ax.plot([0, 1], [2, 4])
        ax.set_ylim(1, 5)
        width, height = fig.get_size_inches()
        before = np.array(ax.get_position().bounds) * [width, height, width, height]
        a.dr_annotations.append_blocks(fig, [(tag, 0, (0, 60)) for tag in a.TAGS])
        new_width, new_height = fig.get_size_inches()
        after = np.array(ax.get_position().bounds) * [new_width, new_height, new_width, new_height]
        np.testing.assert_allclose(after[[0, 2, 3]], before[[0, 2, 3]], atol=1e-12)
        self.assertAlmostEqual(after[1] - before[1], new_height - height)
        np.testing.assert_array_equal(line.get_xdata(), [0, 1])
        np.testing.assert_array_equal(line.get_ydata(), [2, 4])
        self.assertEqual(ax.get_ylim(), (1, 5))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for artist in fig.texts:
            box = artist.get_window_extent(renderer)
            self.assertGreaterEqual(box.x0, 0)
            self.assertGreaterEqual(box.y0, 0)
            self.assertLessEqual(box.x1, fig.bbox.x1)
            self.assertLess(box.y1, ax.get_window_extent(renderer).y0)
        a.plt.close(fig)

    def test_all_plot_settings_identify_lower_right_legends_and_correct_DR_episodes(self):
        settings = json.loads((a.HERE / "plot_settings.json").read_text())
        for panels_by_tag in settings.values():
            if "isaac400" not in panels_by_tag:
                panels_by_tag = {str(k): v for k, v in panels_by_tag.items()}
                groups = panels_by_tag.values()
            else:
                groups = [panels_by_tag]
            for group in groups:
                for tag, panels in group.items():
                    for panel in panels:
                        self.assertEqual(panel["legend_location"], "lower right")
                        expected = a.dr_annotations.episodes_in_view(tag, panel["environment"], panel["window_s"])
                        self.assertEqual(panel["sampled_dr_episodes"], [item["episode"] for item in expected])

    def test_mujoco_policy_boundaries_and_projection_are_not_mislabelled(self):
        for tag in ("mujoco400", "mujoco800"):
            _, data = self.datasets[tag]
            np.testing.assert_array_equal(data["position_after_rad"][:, -1],
                                          data["policy_boundary_position_rad"])
            np.testing.assert_array_equal(np.clip(data["velocity_after_rad_s"][:, -1],
                                                 [-23.9, -61.6], [23.9, 61.6]),
                                          data["policy_boundary_velocity_rad_s"])
        self.assertEqual(a.response_field("isaac400"), "projected_joint_effort_nm")
        self.assertEqual(a.response_field("mujoco400"), "actual_motor_torque_nm")

    def test_csv_contains_actual_inference_inputs_and_start_end_positions(self):
        for tag, (_, data) in self.datasets.items():
            with (a.HERE / f"{tag}_env0_policy_traces.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3000)
            for step in (0, 1, 1499, 1500, 2999):
                row = rows[step]
                self.assertEqual(float(row["policy_time_s"]), step * .02)
                self.assertEqual(float(row["steer_position_at_policy_start_rad"]),
                                 data["position_before_rad"][step, 0, 0, 0])
                self.assertEqual(float(row["steer_position_after_20ms_rad"]),
                                 data["position_after_rad"][step, -1, 0, 0])
                for channel, (label, _) in enumerate(a.OBSERVATIONS):
                    self.assertEqual(float(row[label]), float(data["observation"][step, 0, channel]))

    def test_every_plot_format_and_all64_pdf_pages(self):
        names = json.loads((a.HERE / "summary.json").read_text())["figures"]
        self.assertEqual(len(names), 8)
        for name in names:
            for ext in ("png", "svg", "pdf"):
                path = a.HERE / f"{name}.{ext}"
                self.assertGreater(path.stat().st_size, 1000)
            with Image.open(a.HERE / f"{name}.png") as im:
                self.assertGreater(im.width, 2000)
                im.verify()
        for name in ("all64_demand_vs_actuation", "all64_observation_traces"):
            content = (a.HERE / f"{name}.pdf").read_bytes()
            self.assertEqual(len(re.findall(rb"/Type /Page\b", content)), 64)

    def test_gallery_links_and_recorded_input_hashes(self):
        for target in re.findall(r'(?:href|src)="([^"#]+)"', (a.HERE / "index.html").read_text()):
            self.assertTrue((a.HERE / target).is_file(), target)
        summary = json.loads((a.HERE / "summary.json").read_text())
        for tag in a.TAGS:
            for path, digest in summary[tag]["input_sha256"].items():
                self.assertEqual(a.sha(a.HERE / path), digest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
