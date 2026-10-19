"""Synthetic helper checks; none of these samples enter the real-bag report."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location("corrected_deployment_analysis", Path(__file__).with_name("analyze_corrected.py"))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class AnalysisTests(unittest.TestCase):
    def test_upright_gravity(self):
        gravity, roll = analysis.gravity_and_roll(np.array([[0., 0., 0., 1.]]))
        np.testing.assert_allclose(gravity, [[0, -1]])
        np.testing.assert_allclose(roll, [0])

    def test_pure_roll_sign_and_normalization(self):
        angle = np.array([.3, -.3])
        q = np.column_stack((np.sin(angle / 2), np.zeros(2), np.zeros(2), np.cos(angle / 2))) * 2
        gravity, roll = analysis.gravity_and_roll(q)
        np.testing.assert_allclose(gravity[:, 0], -np.sin(angle), atol=1e-12)
        np.testing.assert_allclose(gravity[:, 1], -np.cos(angle), atol=1e-12)
        np.testing.assert_allclose(roll, angle, atol=1e-12)

    def test_zero_order_hold_not_interpolation(self):
        result = analysis.held_values(np.array([0., .02, .04]), np.array([1., -1., 2.]), np.array([.019, .02, .039, .04]))
        np.testing.assert_equal(result, [1, -1, -1, 2])

    def test_known_delay_gain_and_bias(self):
        command_t = np.arange(0, 1.5, .02)
        command = np.sin(9 * command_t) + .2 * np.cos(13 * command_t)
        feedback_t = .0013 + np.arange(348) * .0043
        feedback = .8 * analysis.held_values(command_t, command, feedback_t - .027) + .1
        result, scan = analysis.lag_profile(command_t, command, feedback_t, feedback)
        self.assertAlmostEqual(result["lag_ms"], 27, delta=1)
        self.assertGreater(result["correlation"], .999)
        self.assertAlmostEqual(result["fit_gain"], .8, places=6)
        self.assertAlmostEqual(result["fit_bias"], .1, places=6)
        self.assertEqual(scan.shape[1], 6)

    def test_encoder_wrap_with_constant_speed(self):
        state = np.zeros((240, 8))
        state[:, 0] = np.arange(240) * .01
        state[:, 1] = (2.8 + 2 * state[:, 0] + np.pi) % (2 * np.pi) - np.pi
        state[:, 3] = 2
        _, samples = analysis.encoder_consistency(state, 1, 3)
        self.assertGreater(len(samples), 200)
        np.testing.assert_allclose(samples[:, 1:], 2, atol=1e-12)

    def test_statistics_exclude_nonfinite(self):
        result = analysis.stats([1, np.nan, 3, np.inf])
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["mean"], 2)
        self.assertAlmostEqual(result["rms"], np.sqrt(5))

    def test_relationship_known_scale(self):
        x = np.linspace(-1, 1, 30)
        result = analysis.relationship(x, 2 * x - .1)
        self.assertAlmostEqual(result["slope"], 2)
        self.assertAlmostEqual(result["offset"], -.1)
        self.assertAlmostEqual(result["correlation"], 1)

    def test_quaternion_rate_matches_synthetic_gyro(self):
        t = np.arange(400) * .01
        angle = .3 * np.sin(np.pi * t)
        imu = np.zeros((len(t), 12))
        imu[:, 0] = imu[:, 1] = t
        imu[:, 2] = np.sin(angle / 2)
        imu[:, 5] = np.cos(angle / 2)
        imu[:, 6] = .3 * np.pi * np.cos(np.pi * t)
        result = analysis.quaternion_gyro_check(imu)["x"]
        self.assertGreater(result["correlation"], .999)
        self.assertAlmostEqual(result["slope"], 1, delta=.002)


if __name__ == "__main__":
    unittest.main()
