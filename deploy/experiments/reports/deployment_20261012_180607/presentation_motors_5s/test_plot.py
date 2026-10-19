"""Synthetic window-selection tests; synthetic data never enter plotted reports."""
import unittest

import numpy as np

from plot_motor_overlay import centred_window, crop


class WindowSelectionTest(unittest.TestCase):
    def test_full_attempt_middle_five_seconds(self):
        self.assertEqual(centred_window(10, 20), (12.5, 17.5))

    def test_short_attempt_keeps_five_second_axis_without_inventing_samples(self):
        start, end = centred_window(0, 1)
        self.assertEqual((start, end), (-2, 3))
        series = np.array([[0, 1, 2], [0.5, 3, 4], [1, 5, 6]], dtype=float)
        actual = crop(series, start, end)
        np.testing.assert_array_equal(actual[:, 0], [2, 2.5, 3])
        np.testing.assert_array_equal(actual[:, 1:], series[:, 1:])

    def test_crop_preserves_values_including_missing_feedback(self):
        series = np.array([[0, 10, 20], [1, 30, np.nan], [2, 50, 60], [3, 70, 80]])
        actual = crop(series, 1, 2)
        np.testing.assert_array_equal(actual, [[0, 30, np.nan], [1, 50, 60]])
        self.assertEqual(series[1, 0], 1)  # Never mutate original timestamps.

    def test_invalid_endpoints_rejected(self):
        for endpoints in ((0, 0), (1, 0), (np.nan, 1), (0, np.inf)):
            with self.subTest(endpoints=endpoints), self.assertRaises(ValueError):
                centred_window(*endpoints)

    def test_no_samples_rejected(self):
        with self.assertRaises(ValueError):
            crop(np.array([[0, 1, 2]]), 1, 2)

    def test_invalid_timestamp_sequence_rejected(self):
        for series in (np.array([[1, 1, 2], [0, 3, 4]]), np.array([[np.nan, 1, 2]])):
            with self.subTest(series=series), self.assertRaises(ValueError):
                crop(series, 0, 5)


if __name__ == "__main__":
    unittest.main()
