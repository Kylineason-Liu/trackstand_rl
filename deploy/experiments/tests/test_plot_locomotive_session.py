"""Offline checks for plot_locomotive_session.py: no ROS, bag or plotting library needed."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plot_locomotive_session as plot  # noqa: E402


def quaternion(roll, pitch=0.0, yaw=0.0):
    """ZYX (yaw, pitch, roll) -> xyzw."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy]


def session_with(events, duration=10.0, **tables):
    session = plot.Session(Path("locomotive_test"), {}, 0, duration_s=duration)
    session.events = events
    session.tables.update(tables)
    return session


class RollTests(unittest.TestCase):
    def test_roll_matches_policy_gravity_y_convention_with_heading(self):
        for roll in (-0.4, -0.1, 0.0, 0.2, 0.5):
            for yaw in (0.0, 1.0, -2.5):
                q = np.asarray([quaternion(roll, 0.05, yaw)])
                got, pitch, heading = plot.quaternion_to_rpy(q)
                # Exact Euler roll/yaw, including a small pitch (gravity_y = -sin roll cos pitch).
                self.assertAlmostEqual(got[0], roll, places=9)
                self.assertAlmostEqual(pitch[0], 0.05, places=9)
                self.assertAlmostEqual(heading[0], yaw, places=9)


class DecimationTests(unittest.TestCase):
    def test_envelope_keeps_single_sample_spikes_and_order(self):
        x = np.arange(100_000) / 200.0
        y = np.zeros_like(x)
        y[54_321] = 42.0  # One-sample timing outlier.
        y[77_000] = -7.0
        dx, dy = plot._decimate(x, y, 2000)
        self.assertLessEqual(len(dx), 2000)
        self.assertIn(42.0, dy)
        self.assertIn(-7.0, dy)
        self.assertTrue(np.all(np.diff(dx) > 0))

    def test_short_series_are_untouched(self):
        x, y = np.arange(10.0), np.arange(10.0)
        dx, dy = plot._decimate(x, y, 2000)
        self.assertIs(dx, x)


class ModeTests(unittest.TestCase):
    def test_intervals_merge_events_and_status_and_mask_riding(self):
        events = {"policy_event": [(1.0, {"mode": "launch"}), (3.0, {"mode": "riding"}),
                                   (8.0, {"mode": "fault_latched", "latched_fault_reasons": ["roll_limit"]})],
                  "status": [(0.0, {"mode": "down"}), (3.5, {"mode": "riding"})]}
        session = session_with(events)
        self.assertEqual(plot.mode_intervals(session), [(0.0, 1.0, "down"), (1.0, 3.0, "launch"),
                                                        (3.0, 8.0, "riding"), (8.0, 10.0, "fault_latched")])
        mask = plot.mask_for(np.array([2.9, 3.0, 7.99, 8.0]), plot.mode_intervals(session))
        self.assertEqual(mask.tolist(), [False, True, True, False])
        self.assertEqual(plot.fault_markers(session), [(8.0, "policy: roll_limit")])


class SummaryTests(unittest.TestCase):
    def test_riding_only_lean_saturation_and_ride_segments(self):
        t = np.arange(0, 10, 0.01)
        roll = np.where((t >= 3) & (t < 8), 6.0 * np.sin(t), 40.0)  # Outside riding must not count.
        events = {"policy_event": [(0.0, {"mode": "down"}), (3.0, {"mode": "riding"}),
                                   (8.0, {"mode": "fault_latched", "latched_fault_reasons": ["roll_limit"]})]}
        cmd_t = np.arange(0, 10, 0.02)
        steer = np.where(cmd_t < 5, 0.0, math.radians(20))
        session = session_with(
            events,
            imu=plot.Table(t, {"roll_deg": roll}),
            motor_command=plot.Table(cmd_t, {"enable": np.ones_like(cmd_t), "steering_target_rad": steer,
                                             "rear_velocity_rad_s": np.full_like(cmd_t, 20.0)}))
        plot.add_derived(session)
        summary = plot.summarize(session)
        self.assertEqual(summary["rides"]["count"], 1)
        self.assertEqual(summary["rides"]["segments"][0]["ended_by"], "fault_latched")
        self.assertLess(summary["lean_riding"]["roll_deg"]["max_abs"], 6.0 + 1e-9)
        # Commands at the 20 deg clip from 5 s to 8 s of a 3-8 s ride.
        self.assertAlmostEqual(summary["saturation_riding"]["steering_at_clip"], 0.6, places=2)
        self.assertEqual(summary["saturation_riding"]["rear_at_clip"], 0.0)
        markdown = plot.summary_markdown(summary)
        self.assertIn("| Rides / total riding / longest | 1 / 5.0 s / 5.0 s |", markdown)

    def test_missing_topics_report_not_available_instead_of_failing(self):
        session = session_with({})
        plot.add_derived(session)
        markdown = plot.summary_markdown(plot.summarize(session))
        self.assertIn("Speed error RMS (v − v*) | n/a m/s", markdown)


if __name__ == "__main__":
    unittest.main()
