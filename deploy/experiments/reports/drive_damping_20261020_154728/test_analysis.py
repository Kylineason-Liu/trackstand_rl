"""Numerical analysis checks; synthetic inputs only, no hardware or source CSV mutations."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("drive_damping_analysis", Path(__file__).with_name("analysis.py"))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class FrictionFitTests(unittest.TestCase):
    def test_exact_viscous_coulomb_and_bias(self):
        w = np.array([-8, -4, -2, -1, 1, 2, 4, 8], dtype=float)
        torque = .002*w+.1*np.sign(w)+.003
        for constrained in (False, True):
            result = analysis.fit_loss(w, torque, constrained)
            self.assertAlmostEqual(result["B_nm_s_per_rad"], .002)
            self.assertAlmostEqual(result["coulomb_nm"], .1)
            self.assertAlmostEqual(result["offset_nm"], .003)
            self.assertLess(result["rmse_nm"], 1e-12)

    def test_negative_slope_not_installed_as_negative_damping(self):
        w = np.array([-8, -4, -2, -1, 1, 2, 4, 8], dtype=float)
        torque = -.001*w+.1*np.sign(w)
        self.assertLess(analysis.fit_loss(w, torque)["B_nm_s_per_rad"], 0)
        result = analysis.fit_loss(w, torque, True)
        self.assertEqual(result["B_nm_s_per_rad"], 0)
        self.assertGreater(result["coulomb_nm"], 0)

    def test_constant_friction_is_not_viscous_damping(self):
        w = np.array([-8, -4, -2, -1, 1, 2, 4, 8], dtype=float)
        torque = .106*np.sign(w)
        result = analysis.fit_loss(w, torque, True)
        self.assertAlmostEqual(result["B_nm_s_per_rad"], 0)
        self.assertAlmostEqual(result["coulomb_nm"], .106)
        self.assertGreater(np.dot(w, torque)/np.dot(w, w), .01)

    def test_design_must_identify_three_coefficients(self):
        for w in ([1, 2, 4, 8], [-1, 1]):
            with self.subTest(w=w), self.assertRaises(ValueError):
                analysis.fit_loss(w, np.ones(len(w)))

    def test_nonnegative_active_sets_with_free_offset(self):
        w = np.array([-8, -4, -2, -1, 1, 2, 4, 8], dtype=float)
        result = analysis.fit_loss(w, .003*w+.004, True)
        self.assertAlmostEqual(result["B_nm_s_per_rad"], .003)
        self.assertAlmostEqual(result["coulomb_nm"], 0)
        self.assertAlmostEqual(result["offset_nm"], .004)

    def test_plateau_means_exclude_cached_rows_and_end_transients(self):
        rows = []
        for t in np.linspace(0, 15, 151):
            for op in ("A1", "A2"):
                row = dict(record_type="rx", phase="hold", phase_elapsed_s=t, elapsed_s=t+10,
                           repeat=1, step=1, plateau_speed_rad_s=2, opcode_hex=op,
                           velocity_rad_s=2 if 1 <= t < 14 else 999,
                           current_a=-.3, torque_estimate_nm=.1 if 1 <= t < 14 else 999,
                           current_age_s=0, velocity_age_s=0, current_limit_a=1,
                           motor_temperature_c=40)
                rows.append(row)
                rows.append({**row, "record_type": "query", "velocity_rad_s": 999, "torque_estimate_nm": 999})
        result = analysis.plateau_means(pd.DataFrame(rows))
        self.assertEqual(len(result), 1)
        self.assertEqual(result.n_a1.iloc[0], 130)
        self.assertEqual(result.n_a2.iloc[0], 130)
        self.assertAlmostEqual(result.speed_mean_rad_s.iloc[0], 2)
        self.assertAlmostEqual(result.torque_mean_nm.iloc[0], .1)

    def test_raw_current_torque_scaling_is_independently_checked(self):
        row = dict(record_type="rx", opcode_hex="A1", payload_hex="A138FFFFFF", current_a=-.2,
                   torque_sign=-1, torque_constant_nm_per_a=.3, torque_estimate_nm=.06)
        self.assertEqual(analysis.raw_checks(pd.DataFrame([row])), {"A1": 1})
        row["torque_estimate_nm"] = .2
        with self.assertRaises(AssertionError):
            analysis.raw_checks(pd.DataFrame([row]))

    def test_bootstrap_is_reproducible_and_does_not_resample_cached_rows(self):
        w = np.array([1, 2, 4, 6, 8, -1, -2, -4, -6, -8, 1, 2, 4, 6, 8, -1, -2, -4, -6], dtype=float)
        points = pd.DataFrame(dict(speed_mean_rad_s=w, torque_mean_nm=.002*w+.1*np.sign(w), target_rad_s=w,
                                   repeat=[1]*10+[2]*9))
        first = analysis.bootstrap(points, repetitions=100)
        self.assertEqual(first, analysis.bootstrap(points, repetitions=100))
        np.testing.assert_allclose(first["B_percentile_2_5_50_97_5"], .002, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
