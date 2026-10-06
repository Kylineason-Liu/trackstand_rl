"""Numerical-method tests, independent of CAN, ROS, and hardware."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_c2_sine import (correlation_check, final_hold, first_order_fit, harmonic, held_values,
                             interval_stats, labels, minimize_scalar, window_metrics)


class C2SineAnalysisTests(unittest.TestCase):
    def test_filename_amplitudes_are_peak_not_peak_to_peak(self):
        for token, value in (("01", .1), ("05", .5), ("1", 1), ("2", 2)):
            parsed = labels(Path(f"motor_feedback_202610041321_2hz{token}rad.csv"))
            self.assertEqual(parsed["amplitude_rad"], value)
            self.assertEqual(parsed["frequency_hz"], 2)

    def test_reject_unknown_filename(self):
        with self.assertRaises(ValueError):
            labels(Path("unlabelled.csv"))

    def test_harmonic_known_amplitude_phase_and_bias(self):
        time = np.linspace(11, 16, 1703)
        expected_phase = -.7
        value = 2*np.sin(2*np.pi*3*(time-11)+expected_phase)+.12
        result = harmonic(time, value, 3, 11)
        self.assertAlmostEqual(result["amplitude"], 2, places=10)
        self.assertAlmostEqual(result["phase_rad"], expected_phase, places=10)
        self.assertAlmostEqual(result["mean"], .12, places=10)
        self.assertAlmostEqual(result["r_squared"], 1, places=10)

    def test_harmonic_skips_missing_measurement_not_replaces_with_zero(self):
        time = np.linspace(0, 5, 1001)
        value = np.sin(2*np.pi*2*time)
        value[20:30] = np.nan
        result = harmonic(time, value, 2, 0)
        self.assertEqual(result["samples"], 991)
        self.assertAlmostEqual(result["amplitude"], 1, places=10)

    def test_insufficient_harmonic_window_rejected(self):
        with self.assertRaises(ValueError):
            harmonic(np.arange(5)*.01, np.ones(5), 2, 0)

    def test_minimizer(self):
        result = minimize_scalar(lambda x: (x-.1234)**2, -1, 1)
        self.assertAlmostEqual(result, .1234, places=9)

    def test_held_values_do_not_interpolate(self):
        command = np.array([[0, 1], [1, 2], [2, 3]])
        np.testing.assert_array_equal(held_values(command, np.array([0, .8, 1, 1.9, 2])), [1, 1, 2, 2, 3])

    def test_held_values_refuse_pre_start_and_post_end(self):
        command = np.array([[0, 1], [1, 2]])
        for requested in ([-.1], [1.1]):
            with self.assertRaises(ValueError):
                held_values(command, np.array(requested))

    def test_interval_statistics_duplicate_clock_rejected(self):
        with self.assertRaises(ValueError):
            interval_stats(np.array([0, .005, .005]))
        result = interval_stats(np.array([0, .005, .01, .015]))
        self.assertAlmostEqual(result["mean_rate_hz"], 200)

    @staticmethod
    def synthetic_response():
        frequency, delay, gain = 2.0, .023, .4
        tc = np.arange(0, 10.001, .005)
        command = np.column_stack((tc, np.sin(2*np.pi*frequency*tc)))
        dense = np.arange(2, 8, .0001)
        fitted = harmonic(dense, held_values(command, dense), frequency, 0)
        tf = tc+.001
        phase = fitted["phase_rad"]-2*np.pi*frequency*delay
        position = fitted["amplitude"]*gain*np.sin(2*np.pi*frequency*tf+phase)
        velocity = fitted["amplitude"]*gain*2*np.pi*frequency*np.cos(2*np.pi*frequency*tf+phase)
        torque = .1*np.sin(2*np.pi*frequency*tf+.3)
        return [command, np.column_stack((tf, position)), np.column_stack((tf, velocity)),
                np.column_stack((tf, torque))], frequency, delay, gain

    def test_known_phase_lag_against_actual_held_input(self):
        traces, frequency, delay, gain = self.synthetic_response()
        result = window_metrics(traces, frequency, 2, 8)
        self.assertAlmostEqual(result["gain"], gain, places=8)
        self.assertAlmostEqual(result["phase_equivalent_lag_ms"], delay*1000, places=8)
        self.assertGreater(result["lag_against_ideal_command_ms"], delay*1000)
        mask = (traces[1][:, 0] >= 2) & (traces[1][:, 0] < 8)
        q = traces[1][mask]
        error = held_values(traces[0], q[:, 0])-q[:, 1]
        self.assertAlmostEqual(result["error_rms_rad"], np.sqrt(np.mean(error**2)), places=10)

    def test_independent_correlation_agrees_with_known_lag(self):
        traces, frequency, delay, _ = self.synthetic_response()
        result = correlation_check(traces, frequency, 2, 8)
        self.assertTrue(result["available"])
        self.assertAlmostEqual(result["lag_ms"], delay*1000, delta=1)

    def test_correlation_does_not_bridge_long_outage(self):
        traces, frequency, _, _ = self.synthetic_response()
        traces[1] = traces[1][(traces[1][:, 0] < 4) | (traces[1][:, 0] > 5)]
        self.assertFalse(correlation_check(traces, frequency, 2, 8)["available"])

    def test_correlation_ignores_outage_outside_analyzed_window(self):
        traces, frequency, delay, _ = self.synthetic_response()
        traces[1] = traces[1][(traces[1][:, 0] < .3) | (traces[1][:, 0] > .6)]
        result = correlation_check(traces, frequency, 2, 8)
        self.assertTrue(result["available"])
        self.assertAlmostEqual(result["lag_ms"], delay*1000, delta=1)

    def test_first_order_model_recovers_known_response(self):
        results = []
        for frequency in (2, 5, 10):
            for amplitude in (.5, 1, 2):
                response = .97/(1+1j*2*np.pi*frequency*.085)
                results.append(dict(labels=dict(amplitude_rad=amplitude), frequency_fit_hz=frequency,
                                    steady=dict(gain=abs(response), phase_lag_deg=-np.degrees(np.angle(response)))))
        model = first_order_fit(results)
        self.assertAlmostEqual(model["tau_s"], .085, places=7)
        self.assertAlmostEqual(model["dc_gain"], .97, places=7)
        self.assertLess(model["complex_response_rmse"], 1e-8)
        self.assertAlmostEqual(model["model_implied_corner_hz"], 1/(2*math.pi*.085), places=7)

    def test_final_hold_requires_confirmation_and_all_remaining_samples(self):
        command = np.array([[0, 0], [1, 1]])
        position = np.array([[1.001, 0], [1.05, .5], [1.1, 1], [1.2, .7], [1.3, 1], [1.7, 1]])
        result = final_hold([command, position])
        self.assertAlmostEqual(result["finite_record_settling_ms_by_band_rad"]["0.1"], 300)
        short = final_hold([command, position[:-1]])
        self.assertIsNone(short["finite_record_settling_ms_by_band_rad"]["0.1"])

    def test_final_hold_no_recorded_tail(self):
        result = final_hold([np.array([[0, 0], [1, 1]]), np.array([[0, 0], [.5, .5]])])
        self.assertFalse(result["available"])


if __name__ == "__main__":
    unittest.main()
