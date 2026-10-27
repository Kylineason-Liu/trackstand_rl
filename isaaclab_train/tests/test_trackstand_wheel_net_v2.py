"""V2 wheel-net position unwrapping, reset isolation and environment wiring."""

import importlib
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
PACKAGE = "source.trackstand_rl.trackstand_rl.tasks.manager_based.trackstand"


@unittest.skipUnless(importlib.util.find_spec("isaaclab"), "Run with Isaac Lab Python")
class TestWheelNetV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from isaaclab.envs import ManagerBasedRLEnv

        cls.torch, cls.parent = torch, ManagerBasedRLEnv
        cls.Tracker = importlib.import_module(f"{PACKAGE}.mdp.stationary.wheel_net").WheelNetAngleTracker
        cls.Env = importlib.import_module(f"{PACKAGE}.trackstand_bike.trackstand_rl_env_v2").TrackstandRLEnvV2
        cls.V1 = importlib.import_module(f"{PACKAGE}.trackstand_bike.trackstand_rl_env").TrackstandRLEnvV1

    def make_env(self):
        torch = self.torch
        env = object.__new__(self.Env)
        env._is_closed = True
        position = torch.tensor([[0., 0.], [1., 0.], [-1., 0.], [2., 0.]])
        env._drive_idx = 0
        env.scene = {"robot": SimpleNamespace(root_view=SimpleNamespace(get_dof_positions=lambda: position))}
        env._wheel_net_tracker = self.Tracker(position[:, 0])
        env.wheel_net_angle = env._wheel_net_tracker.angle_rad
        env.spawn_yaw = torch.tensor([.1, .2, .3, .4])
        env._heading_of = Mock(return_value=torch.tensor([-.1, -.2, -.3, -.4]))
        env._wheel_net_callback_handle = None
        return env, position

    def test_initial_position_is_a_reference_not_motion(self):
        torch = self.torch
        initial = torch.tensor([2., -5., 0.])
        tracker = self.Tracker(initial)
        tracker.update(initial)
        torch.testing.assert_close(tracker.angle_rad, torch.zeros(3), rtol=0, atol=0)
        initial += .3  # The initial reference must not alias the mutable input.
        tracker.update(initial)
        torch.testing.assert_close(tracker.angle_rad, torch.full((3,), .3))

    def test_varying_substep_motion_uses_angle_not_final_speed(self):
        torch = self.torch
        tracker = self.Tracker(torch.zeros(1))
        for position in (.10, .18, .12, .07, -.02, .05, .08, .11):
            tracker.update(torch.tensor([position]))
        torch.testing.assert_close(tracker.angle_rad, torch.tensor([.11]), rtol=0, atol=1e-7)

    def test_back_and_forth_cancels(self):
        torch = self.torch
        tracker = self.Tracker(torch.zeros(1))
        for position in (.3, -.4, .8, -.1, 0.):
            tracker.update(torch.tensor([position]))
        torch.testing.assert_close(tracker.angle_rad, torch.zeros(1), rtol=0, atol=1e-7)

    def test_physx_four_pi_branch_crosses_in_both_directions(self):
        torch = self.torch
        tracker = self.Tracker(torch.tensor([6.1, -6.1]))
        tracker.update(torch.tensor([-6.1, 6.1]))
        expected = torch.tensor([4*math.pi - 12.2, 12.2 - 4*math.pi])
        torch.testing.assert_close(tracker.angle_rad, expected, rtol=0, atol=1e-6)

    def test_crossing_pi_is_not_incorrectly_wrapped(self):
        torch = self.torch
        tracker = self.Tracker(torch.tensor([3., -3.]))
        tracker.update(torch.tensor([3.5, -3.5]))
        torch.testing.assert_close(tracker.angle_rad, torch.tensor([.5, -.5]), rtol=0, atol=0)

    def test_multiple_turns_accumulate_beyond_the_termination_limit(self):
        torch = self.torch
        tracker = self.Tracker(torch.zeros(2))
        for index in range(1, 601):
            angle = torch.tensor([index*.05, -index*.04])
            native = (angle + 2*math.pi) % (4*math.pi) - 2*math.pi
            tracker.update(native)
        torch.testing.assert_close(tracker.angle_rad, torch.tensor([30., -24.]), rtol=0, atol=1e-4)

    def test_partial_reset_excludes_teleports_and_preserves_other_envs(self):
        torch = self.torch
        tracker = self.Tracker(torch.zeros(3))
        tracker.update(torch.tensor([.3, .4, .5]))
        position = torch.tensor([2., .4, -1.])
        tracker.reset(position, torch.tensor([0, 2]))
        tracker.update(position)
        torch.testing.assert_close(tracker.angle_rad, torch.tensor([0., .4, 0.]))
        tracker.update(position + .1)
        torch.testing.assert_close(tracker.angle_rad, torch.tensor([.1, .5, .1]))

    def test_empty_reset_and_full_reset(self):
        torch = self.torch
        tracker = self.Tracker(torch.zeros(2))
        tracker.update(torch.tensor([1., -2.]))
        tracker.reset(torch.zeros(2), torch.empty(0, dtype=torch.long))
        torch.testing.assert_close(tracker.angle_rad, torch.tensor([1., -2.]))
        tracker.reset(torch.tensor([3., -4.]))
        tracker.update(torch.tensor([3., -4.]))
        torch.testing.assert_close(tracker.angle_rad, torch.zeros(2), rtol=0, atol=0)

    def test_environment_callback_reads_native_position(self):
        torch = self.torch
        env, position = self.make_env()
        position[:, 0] += .2
        env._update_wheel_net(.0025)
        torch.testing.assert_close(env.wheel_net_angle, torch.full((4,), .2))
        self.assertIs(env.wheel_net_angle, env._wheel_net_tracker.angle_rad)

    def test_step_skips_the_legacy_endpoint_speed_integral(self):
        env, _ = self.make_env()
        result = (object(), object(), object(), object(), object())
        action = self.torch.zeros(4, 2)
        with patch.object(self.parent, "step", return_value=result) as base_step:
            with patch.object(self.V1, "step", side_effect=AssertionError("legacy step must not run")):
                self.assertIs(env.step(action), result)
            base_step.assert_called_once_with(env, action)

    def test_selected_scene_resets_refresh_angle_reference_and_yaw(self):
        torch = self.torch
        env, position = self.make_env()
        position[:, 0] += .3
        env._update_wheel_net(.0025)
        ids = torch.tensor([0, 2])

        def scene_reset(_env, env_ids):
            position[env_ids, 0] = 0.

        with patch.object(self.parent, "_reset_idx", autospec=True, side_effect=scene_reset) as reset:
            env._reset_idx(ids)
            reset.assert_called_once_with(env, ids)
        env._update_wheel_net(.0025)
        torch.testing.assert_close(env.wheel_net_angle, torch.tensor([0., .3, 0., .3]))
        torch.testing.assert_close(env.spawn_yaw, torch.tensor([-.1, .2, -.3, .4]))

    def test_full_and_partial_reset_delegate_without_legacy_clear_all(self):
        env, _ = self.make_env()
        ids = self.torch.tensor([1, 3])
        result = (object(), object())
        with patch.object(self.parent, "reset", return_value=result) as reset:
            with patch.object(self.V1, "reset", side_effect=AssertionError("legacy reset must not run")):
                self.assertIs(env.reset(seed=42, env_ids=ids, options={"a": 1}), result)
            reset.assert_called_once_with(env, seed=42, env_ids=ids, options={"a": 1})

    def test_callback_is_deregistered_before_closing(self):
        env, _ = self.make_env()
        handle = env._wheel_net_callback_handle = Mock()
        with patch.object(self.V1, "close") as close:
            env.close()
            env.close()
        handle.deregister.assert_called_once_with()
        self.assertIsNone(env._wheel_net_callback_handle)
        self.assertEqual(close.call_count, 2)


if __name__ == "__main__":
    unittest.main()
