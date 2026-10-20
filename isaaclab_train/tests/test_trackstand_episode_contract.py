"""CPU regressions for v1 episode bookkeeping and normalized-action smoothing."""

import importlib
import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
TASK_PACKAGE = "source.trackstand_rl.trackstand_rl.tasks.manager_based.trackstand"


@unittest.skipUnless(importlib.util.find_spec("isaaclab"), "Run with Isaac Lab Python")
class TestEpisodeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gymnasium as gym
        import torch
        from isaaclab.envs import ManagerBasedRLEnv

        cls.gym, cls.torch, cls.parent = gym, torch, ManagerBasedRLEnv
        module = importlib.import_module(f"{TASK_PACKAGE}.trackstand_bike.trackstand_rl_env")
        cls.env_types = (module.TrackstandRLEnv, module.TrackstandRLEnvV1)

    def make_env(self, version, terminated, truncated):
        """Use real environment methods; mock only the simulator-facing parent step."""
        torch = self.torch
        env = object.__new__(self.env_types[version])
        env._is_closed = True  # No simulator was created; suppress destructor cleanup.
        env.cfg = SimpleNamespace(sim=SimpleNamespace(dt=0.0025), decimation=8)
        yaw = torch.tensor([-0.4, 0.3, 0.2, -0.1])
        quat = torch.stack((torch.zeros(4), torch.zeros(4), (yaw / 2).sin(), (yaw / 2).cos()), dim=-1)
        env.scene = {"robot": SimpleNamespace(data=SimpleNamespace(
            joint_vel=torch.tensor([[1.0], [-2.0], [3.0], [-4.0]]),
            root_quat_w=SimpleNamespace(torch=quat),
        ))}
        env._drive_idx = 0
        env.wheel_net_angle = torch.tensor([2.0, 3.0, 4.0, 5.0])
        env.spawn_yaw = torch.tensor([0.1, 0.2, 0.3, 0.4])
        env.reset_terminated = torch.tensor(terminated)
        env.reset_time_outs = torch.tensor(truncated)
        return env, yaw

    def step_without_physics(self, env):
        result = ({"policy": object()}, object(), env.reset_terminated, env.reset_time_outs, {"test": True})
        with patch.object(self.parent, "step", return_value=result) as parent_step:
            action = self.torch.zeros(4, 2)
            actual = env.step(action)
            parent_step.assert_called_once_with(action)
        for returned, expected in zip(actual, result, strict=True):
            self.assertIs(returned, expected)  # Preserve PPO termination/truncation signals and extras.

    def test_v1_clears_timeout_failure_and_overlap_without_touching_continuing_envs(self):
        torch = self.torch
        env, yaw = self.make_env(1, [False, True, False, True], [True, False, False, True])
        self.step_without_physics(env)
        torch.testing.assert_close(env.wheel_net_angle, torch.tensor([0.0, 0.0, 4.06, 0.0]))
        expected_yaw = yaw.clone()
        expected_yaw[2] = 0.3  # Continuing episode keeps its original baseline.
        torch.testing.assert_close(env.spawn_yaw, expected_yaw)
        self.assertEqual(env.reset_terminated.tolist(), [False, True, False, True])
        self.assertEqual(env.reset_time_outs.tolist(), [True, False, False, True])

    def test_v1_no_resets_integrates_all_wheels_and_keeps_all_baselines(self):
        torch = self.torch
        env, _ = self.make_env(1, [False] * 4, [False] * 4)
        before_yaw = env.spawn_yaw.clone()
        self.step_without_physics(env)
        torch.testing.assert_close(env.wheel_net_angle, torch.tensor([2.02, 2.96, 4.06, 4.92]))
        torch.testing.assert_close(env.spawn_yaw, before_yaw, rtol=0, atol=0)

    def test_v1_repeated_timeout_resets_do_not_carry_previous_episode_state(self):
        torch = self.torch
        env, yaw = self.make_env(1, [False] * 4, [True] * 4)
        for _ in range(3):
            env.wheel_net_angle += 2.0
            env.spawn_yaw += 0.123
            self.step_without_physics(env)
            torch.testing.assert_close(env.wheel_net_angle, torch.zeros(4))
            torch.testing.assert_close(env.spawn_yaw, yaw)

    def test_v0_keeps_historical_failure_only_reset_behavior(self):
        torch = self.torch
        env, yaw = self.make_env(0, [False, True, False, True], [True, False, False, True])
        self.step_without_physics(env)
        torch.testing.assert_close(env.wheel_net_angle, torch.tensor([2.02, 0.0, 4.06, 0.0]))
        torch.testing.assert_close(env.spawn_yaw, torch.tensor([0.1, yaw[1], 0.3, yaw[3]]))

    def test_explicit_full_reset_still_clears_all_bookkeeping_in_both_versions(self):
        torch = self.torch
        for version in (0, 1):
            with self.subTest(version=version):
                env, yaw = self.make_env(version, [False] * 4, [False] * 4)
                result = ({"policy": object()}, {"test": True})
                with patch.object(self.parent, "reset", return_value=result) as parent_reset:
                    actual = env.reset(seed=42, options={"test": True})
                    parent_reset.assert_called_once_with(seed=42, options={"test": True})
                self.assertIs(actual[0], result[0])
                self.assertIs(actual[1], result[1])
                torch.testing.assert_close(env.wheel_net_angle, torch.zeros(4))
                torch.testing.assert_close(env.spawn_yaw, yaw)

    def test_v1_rate_penalty_has_equal_normalized_weights_and_correct_sign(self):
        torch = self.torch
        term = self.gym.spec("Trackstand-Balance-v1").kwargs["env_cfg_entry_point"]().rewards.rate_penalty
        actions = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 1.0], [0.0, 0.0]])
        env = SimpleNamespace(action_manager=SimpleNamespace(action=actions, prev_action=torch.zeros_like(actions)))
        rate = term.func(env, **term.params)
        torch.testing.assert_close(rate, torch.tensor([-1.0] * 4 + [-math.sqrt(2), 0.0]))
        self.assertEqual(term.weight, 0.1)
        self.assertEqual(term.params, {})
        torch.testing.assert_close(rate[:4] * term.weight * 0.02, torch.full((4,), -0.002))
        # A full [-1,+1] reversal is delta=2 per channel, not another physical rescaling.
        env.action_manager.action = torch.ones(1, 2)
        env.action_manager.prev_action = -torch.ones(1, 2)
        torch.testing.assert_close(term.func(env, **term.params), torch.tensor([-math.sqrt(8)]))

    def test_v0_rate_penalty_retains_historical_motor_scale_division(self):
        torch = self.torch
        term = self.gym.spec("Trackstand-Balance-v0").kwargs["env_cfg_entry_point"]().rewards.rate_penalty
        actions = torch.eye(2)
        env = SimpleNamespace(
            device="cpu", action_manager=SimpleNamespace(action=actions, prev_action=torch.zeros_like(actions))
        )
        rate = term.func(env, **term.params)
        expected = -torch.sqrt(torch.tensor([1 / 1.57**2, 1 / 6.28**2]) + 1e-8)
        torch.testing.assert_close(rate, expected)
        self.assertEqual(term.params, {"action_scales": (1.57, 6.28)})


if __name__ == "__main__":
    unittest.main()
