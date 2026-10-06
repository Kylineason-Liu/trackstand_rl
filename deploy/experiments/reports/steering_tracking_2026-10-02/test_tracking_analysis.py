"""Numerical-method tests; no ROS or hardware dependency."""

import numpy as np

from analyze_tracking import first_crossing, settling_time, starts_and_stops


def test_crossing_is_not_settlement():
    times = np.arange(0, .6, .01)
    position = np.zeros(len(times))
    position[4] = 1.2
    assert first_crossing(times, position, 0, 1, .9) == 40
    assert settling_time(times, position - 1, .1) is None


def test_settlement_requires_confirmed_remaining_tail():
    times = np.arange(0, .6, .01)
    error = np.ones(len(times))
    error[10:] = 0
    assert settling_time(times, error, .1) == 100
    error[45] = 1
    assert settling_time(times, error, .1) is None


def test_persistent_error_is_not_settlement():
    times = np.arange(0, .6, .01)
    error = np.full(len(times), .245)
    assert settling_time(times, error, .1) is None
    assert settling_time(times, error, .3) == 0


def test_enabled_runs_are_separate():
    result = starts_and_stops(np.array([False, True, True, False, True, False]))
    np.testing.assert_array_equal(result, [[1, 3], [4, 5]])


def test_zero_step_has_no_fractional_crossing():
    assert first_crossing(np.array([0, .01]), np.array([0, 0]), 0, 0, .9) is None
