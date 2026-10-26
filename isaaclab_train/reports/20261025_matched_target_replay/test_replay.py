"""Independent checks of recorded motor inputs, controlled conditions and plots."""

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import unittest

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("matched_analysis", HERE / "analyze.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class ReplayChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = analysis.archive(HERE / "inputs.npz")
        cls.runs = {tag: {fixture: analysis.archive(HERE / tag / f"{fixture}.npz")
                          for fixture in ("airborne", "ground")} for tag in analysis.TAGS}
        cls.stats = json.loads((HERE / "statistics.json").read_text())

    def test_all_real_captures_and_motor_command_reconstruction(self):
        for tag, fixtures in self.runs.items():
            for fixture, data in fixtures.items():
                analysis.validate(tag, fixture, data, self.bank)

    def test_exact_input_identity_and_bounds(self):
        for fixture, n in (("airborne", 10), ("ground", 2)):
            target = self.bank[f"{fixture}_target_rad"]
            self.assertEqual(target.shape, (400, n))
            self.assertEqual(target.dtype, np.dtype("float32"))
            self.assertLessEqual(abs(target).max(), 1.57)
            self.assertLessEqual(abs(np.diff(np.vstack([np.zeros((1, n)), target]), axis=0)).max(), .478 + 1e-6)
            for tag in analysis.TAGS:
                np.testing.assert_array_equal(self.runs[tag][fixture]["target_rad"], target)
                np.testing.assert_array_equal(self.runs[tag][fixture]["rear_nm"], self.bank[f"{fixture}_rear_nm"])

    def test_passive_loss_pairs_and_gains_in_actual_isaac_backend(self):
        for tag in ("isaac400", "isaac800"):
            meta = json.loads((HERE / tag / "metadata.json").read_text())
            self.assertTrue(meta["airborne"]["fixed_base"])
            self.assertFalse(meta["ground"]["fixed_base"])
            joint = meta["airborne"]["joint_names"].index("steering_joint")
            friction = np.asarray(meta["airborne"]["joint_friction_properties"])[:, joint]
            np.testing.assert_allclose(friction[::2], np.tile([.046036056104, .046036056104, .000910713492], (5, 1)))
            np.testing.assert_array_equal(friction[1::2], 0.)
            for fixture in ("airborne", "ground"):
                np.testing.assert_allclose(meta[fixture]["gains"]["steering"]["stiffness"], 10.)
                np.testing.assert_allclose(meta[fixture]["gains"]["steering"]["damping"], .2)
                np.testing.assert_array_equal(meta[fixture]["initial_joint_position"], 0.)
                np.testing.assert_array_equal(meta[fixture]["initial_joint_velocity"], 0.)
                np.testing.assert_array_equal(meta[fixture]["initial_root_velocity"], 0.)
            poses = np.asarray(meta["ground"]["initial_root_pose"])
            np.testing.assert_allclose(poses[:, :3], [[0., 4., -.016], [2., 4., -.016]], atol=1e-8)

    def test_no_contacts_and_loss_ablation_in_mujoco(self):
        for tag in ("mujoco400", "mujoco800"):
            meta = json.loads((HERE / tag / "metadata.json").read_text())
            for index, cfg in enumerate(meta["airborne"]):
                expected = 0. if index % 2 else .046036056104
                self.assertEqual(cfg["frictionloss"], [expected, .106])
                self.assertEqual(cfg["damping"], [0. if index % 2 else .000910713492, 0.])
            self.assertFalse(self.runs[tag]["airborne"]["contacts"].any())

    def test_statistics_are_computed_from_actual_records(self):
        for tag, fixtures in self.runs.items():
            for fixture, data in fixtures.items():
                self.assertEqual(analysis.metrics(data, fixture), self.stats[tag][fixture])

    def test_ground_metric_window_precedes_all_first_crossings(self):
        for tag in analysis.TAGS:
            for metric in self.stats[tag]["ground"].values():
                self.assertEqual(metric["window_s"], [0., .25])
                self.assertGreater(metric["first_30deg_time_s"], .25)

    def test_phase_fit_recovers_known_gain_and_delay(self):
        time = np.arange(1600) / 400
        reference = np.sin(2 * np.pi * time)
        response = .7 * np.sin(2 * np.pi * (time - .023)) + .14
        result = analysis.phase_measure(time, reference, response, 1.)
        self.assertAlmostEqual(result["fundamental_gain"], .7, places=12)
        self.assertAlmostEqual(result["phase_equivalent_lag_ms"], 23., places=10)

    def test_spawn_fall_markers_do_not_pollute_motor_extrema(self):
        fig, ax = analysis.plt.subplots()
        ax.plot([0., .5], [2., 3.], label="Measured")
        ax.axvline(.3, color="red")
        analysis.finish_panel(ax, (0., .5), "Nm")
        self.assertTrue(any(text.get_text() == "min 2 | max 3" for text in ax.texts))
        self.assertGreater(ax.get_ylim()[0], 1.)
        analysis.plt.close(fig)

    def test_original_files_unchanged(self):
        before = json.loads((HERE / "protected_before.json").read_text())
        for name, digest in before.items():
            self.assertEqual(hashlib.sha256(Path(name).read_bytes()).hexdigest(), digest, name)

    def test_figures_and_links(self):
        names = ("matched_position_full", "matched_position_core", "matched_position_detail",
                 "matched_position_final_hold", "matched_torque_detail", "ground_replay_first_half_second")
        for name in names:
            for extension in ("png", "svg", "pdf"):
                self.assertGreater((HERE / f"{name}.{extension}").stat().st_size, 1000)
            with Image.open(HERE / f"{name}.png") as image:
                image.verify()
        for name in ("README.md", "index.html"):
            text = (HERE / name).read_text()
            targets = re.findall(r'\]\(([^)]+)\)|(?:href|src)="([^"]+)"', text)
            for pair in targets:
                target = next(value for value in pair if value)
                if target != "verification.json" and not target.startswith("http"):
                    self.assertTrue((HERE / target).exists(), target)


if __name__ == "__main__":
    unittest.main()
