"""Offline synthetic numerical/decoder checks; no CAN, runtime or log mutation."""

import importlib.util
from pathlib import Path
import struct
import unittest

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("steering_friction_analysis", Path(__file__).with_name("analysis.py"))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class SteeringIdentificationTests(unittest.TestCase):
    def test_recovers_viscous_coulomb_offset(self):
        w = np.array([-12, -8, -4, -2, -1, 1, 2, 4, 8, 12], float)
        y = .0009*w+.046*np.sign(w)+.0013
        for constrained in (True, False):
            fit = analysis.fit_loss(w, y, constrained)
            np.testing.assert_allclose([fit[k] for k in ("B_nm_s_per_rad", "coulomb_nm", "offset_nm")],
                                       [.0009, .046, .0013], rtol=1e-12, atol=1e-12)

    def test_negative_slope_not_passive_antidamping(self):
        w = np.array([-8, -4, -1, 1, 4, 8], float)
        self.assertEqual(analysis.fit_loss(w, -.001*w+.1*np.sign(w), True)["B_nm_s_per_rad"], 0.)

    def test_rejects_single_direction_as_unidentifiable(self):
        with self.assertRaises(ValueError):
            analysis.fit_loss([1, 2, 4, 8], [.1, .11, .12, .13])

    def test_constant_drag_not_mistaken_for_viscous(self):
        w = np.array([-12, -8, -4, -1, 1, 4, 8, 12], float)
        fit = analysis.fit_loss(w, .046*np.sign(w), True)
        self.assertAlmostEqual(fit["B_nm_s_per_rad"], 0.)
        self.assertAlmostEqual(fit["coulomb_nm"], .046)

    @staticmethod
    def hold_rows():
        rows = []
        for t in np.arange(.025, 10., .05):
            for op in ("A1", "A2", "A3"):
                row = dict(record_type="rx", repeat=1, step=1, phase="hold", phase_elapsed_s=t,
                           elapsed_s=20+t, opcode_hex=op, plateau_speed_rad_s=4., velocity_age_s=0., current_age_s=0.,
                           velocity_rad_s=4. if 1 <= t < 9 else 100., current_a=-.06,
                           torque_estimate_nm=.05 if 1 <= t < 9 else 99., position_rad=4*t,
                           motor_temperature_c=40.)
                rows.append(row)
                rows.append({**row, "record_type": "command", "velocity_rad_s": 1000., "torque_estimate_nm": 1000.})
        return pd.DataFrame(rows)

    def test_only_fresh_trimmed_samples_and_one_point_per_hold(self):
        g = analysis.plateau_means(self.hold_rows())
        self.assertEqual(len(g), 1)
        self.assertEqual(int(g.n_a1.iloc[0]), 160)
        self.assertEqual(int(g.n_a2.iloc[0]), 160)
        self.assertAlmostEqual(g.speed_mean_rad_s.iloc[0], 4.)
        self.assertAlmostEqual(g.torque_mean_nm.iloc[0], .05)
        self.assertAlmostEqual(g.a3_speed_regression_rad_s.iloc[0], 4.)
        self.assertAlmostEqual(g.a3_displacement_over_a2_integral.iloc[0], 1.)

    def test_interrupted_hold_not_used(self):
        d = self.hold_rows()
        self.assertTrue(analysis.plateau_means(d[d.phase_elapsed_s < 8]).empty)

    def test_zero_trim_covers_first_and_last_sample_without_boundary_failure(self):
        self.assertEqual(len(analysis.plateau_means(self.hold_rows(), trim=0)), 1)

    def test_full_turn_window_ends_at_same_position_phase(self):
        g = analysis.plateau_means(self.hold_rows(), full_turns=True)
        self.assertEqual(len(g), 1)
        # 8 s at 4 rad/s includes five full revolutions after the first retained sample.
        self.assertAlmostEqual(g.window_end_s.iloc[0]-g.window_start_s.iloc[0], 5*2*np.pi/4)

    def test_nonuniform_sample_time_average(self):
        z = pd.DataFrame({"elapsed_s": [0., 1., 4.], "y": [0., 2., 8.]})
        self.assertAlmostEqual(analysis.window_mean(z, "y", .5, 3.5), 4.)

    def test_raw_current_decoding_and_sign_rejects_corruption(self):
        row = dict(record_type="rx", opcode_hex="A1", payload_hex="A138FFFFFF", can_id_hex="002", motor_id=2,
                   current_a=-.2, torque_sign=-1, torque_constant_nm_per_a=.78, torque_estimate_nm=.156, current_age_s=0.)
        self.assertEqual(analysis.validate_raw(pd.DataFrame([row])), {"A1": 1})
        with self.assertRaises(AssertionError):
            analysis.validate_raw(pd.DataFrame([{**row, "torque_estimate_nm": -.156}]))

    def test_a3_multiturn_position_and_rollover(self):
        for count, zero, delta in ((100000, -20000, 120000), (-2**31+2, 2**31-2, 4)):
            payload = b"\xa3"+struct.pack("<Hi", 321, count)
            row = dict(record_type="rx", opcode_hex="A3", payload_hex=payload.hex(), can_id_hex="002", motor_id=2,
                       encoder_single_turn_count=321, encoder_count=count, zero_count=zero, encoder_sign=-1,
                       gear_ratio=1., position_rad=-delta*2*np.pi/16384, position_age_s=0.)
            self.assertEqual(analysis.validate_raw(pd.DataFrame([row])), {"A3": 1})

    def test_c1_transmission_quantization(self):
        row = dict(record_type="command", opcode_hex="C1", payload_hex="C145FCFFFF", can_id_hex="102", motor_id=2,
                   command_native_rpm=-9.55, command_sign=-1, gear_ratio=1., encoded_command_rad_s=9.55*2*np.pi/60,
                   command_speed_rad_s=1.)
        self.assertEqual(analysis.validate_raw(pd.DataFrame([row])), {})

    def test_bootstrap_reproducible(self):
        w = np.tile([1, 2, 4, 6, 8, 10, 12, -1, -2, -4, -6, -8, -10, -12], 3)
        d = pd.DataFrame(dict(speed_mean_rad_s=w, torque_mean_nm=.0009*w+.046*np.sign(w), target_rad_s=w,
                              repeat=np.repeat([1, 2, 3], 14)))
        first = analysis.rear.bootstrap(d, repetitions=100)
        self.assertEqual(first, analysis.rear.bootstrap(d, repetitions=100))
        np.testing.assert_allclose(first["B_percentile_2_5_50_97_5"], .0009, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
