"""Synthetic helper tests; synthetic points never enter presentation output."""
import unittest

import numpy as np

from plot_policy_to_motor import break_gaps, symmetric_limit


class PresentationHelpersTest(unittest.TestCase):
    def test_gap_break_preserves_every_original_value(self):
        data = np.array([[0.0, 1.0, 2.0], [0.02, 3.0, 4.0], [0.12, 5.0, 6.0]])
        broken = break_gaps(data)
        self.assertEqual(len(broken), 4)
        np.testing.assert_array_equal(broken[[0, 1, 3]], data)
        self.assertTrue(np.isnan(broken[2, 1:]).all())

    def test_missing_feedback_is_not_replaced_with_zero(self):
        data = np.array([[0.0, 1.0, np.nan], [0.02, 2.0, 3.0]])
        np.testing.assert_array_equal(break_gaps(data), data)

    def test_bad_clock_or_shape_is_rejected(self):
        for data in (np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]),
                     np.array([[np.nan, 0.0, 0.0]]), np.zeros((0, 3)), np.zeros((2, 2))):
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    break_gaps(data)

    def test_common_axis_extent_encloses_all_finite_values(self):
        arrays = [np.array([-1.6, np.nan, 1.57]), np.array([1.8, -1.0])]
        self.assertEqual(symmetric_limit(arrays), 2.0)
        self.assertEqual(symmetric_limit([np.array([-6.7, 6.28])]), 8.0)

    def test_axis_without_measured_data_is_rejected(self):
        with self.assertRaises(ValueError):
            symmetric_limit([np.array([np.nan])])


if __name__ == "__main__":
    unittest.main()
