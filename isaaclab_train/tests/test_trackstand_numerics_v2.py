"""V2 containment regressions for invalid physical/controller state and rewards."""

import importlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
PACKAGE = "source.trackstand_rl.trackstand_rl.tasks.manager_based.trackstand"


@unittest.skipUnless(importlib.util.find_spec("isaaclab"), "Run with Isaac Lab Python")
class TestNumericsV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gymnasium as gym
        import torch

        cls.gym, cls.torch = gym, torch
        cls.numeric = importlib.import_module(f"{PACKAGE}.mdp.stationary.numerics")
        cls.legacy = importlib.import_module(f"{PACKAGE}.mdp.stationary.terminations")
        cls.rewards = importlib.import_module(f"{PACKAGE}.mdp.stationary.rewards")

    def make_env(self):
        torch = self.torch
        raw = SimpleNamespace(
            root_pos_w=torch.zeros(4, 3), root_quat_w=torch.tensor([[0., 0., 0., 1.]]).repeat(4, 1),
            root_link_lin_vel_w=torch.zeros(4, 3), root_link_ang_vel_w=torch.zeros(4, 3),
            joint_pos=torch.zeros(4, 3), root_lin_vel_b=torch.zeros(4, 3),
            root_ang_vel_b=torch.zeros(4, 3), joint_vel=torch.zeros(4, 3),
        )
        integral = SimpleNamespace(error_rad=torch.zeros(4, 1), torque_nm=torch.zeros(4, 1),
                                   previous_error=torch.zeros(4, 1))
        scene = type("Scene", (dict,), {})({"robot": SimpleNamespace(data=raw)})
        scene.env_origins = torch.zeros(4, 3)
        env = SimpleNamespace(scene=scene, wheel_net_angle=torch.zeros(4),
                              action_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(integral=integral)))
        return env, raw, integral

    def test_invalid_pose_joint_and_velocity_are_detected_per_environment(self):
        torch = self.torch
        for field in ("root_pos_w", "root_quat_w", "joint_pos", "root_link_lin_vel_w", "root_link_ang_vel_w",
                      "root_lin_vel_b", "root_ang_vel_b", "joint_vel"):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(field=field, value=value):
                    env, data, _ = self.make_env()
                    getattr(data, field)[1, 0] = value
                    torch.testing.assert_close(self.numeric.nonfinite_or_extreme_state_v2(env),
                                               torch.tensor([False, True, False, False]))

    def test_zero_quaternion_is_rejected_without_rejecting_identity(self):
        env, data, _ = self.make_env()
        data.root_quat_w[2] = 0.
        self.assertEqual(self.numeric.nonfinite_or_extreme_state_v2(env).tolist(), [False, False, True, False])

    def test_invalid_integral_or_wheel_net_cannot_survive_as_hidden_state(self):
        for field in ("error_rad", "torque_nm", "previous_error", "wheel_net_angle"):
            with self.subTest(field=field):
                env, _, integral = self.make_env()
                value = env.wheel_net_angle if field == "wheel_net_angle" else getattr(integral, field)
                value[3] = float("nan")
                self.assertEqual(self.numeric.nonfinite_or_extreme_state_v2(env).tolist(),
                                 [False, False, False, True])

    def test_normal_finite_states_and_extreme_speed_limits_match_previous_task(self):
        torch = self.torch
        env, data, _ = self.make_env()
        data.root_lin_vel_b[0, 0], data.root_ang_vel_b[1, 1], data.joint_vel[2, 2] = 21., 51., 301.
        expected = self.legacy.nonfinite_or_extreme_state(env)
        torch.testing.assert_close(self.numeric.nonfinite_or_extreme_state_v2(env), expected)
        self.assertEqual(expected.tolist(), [True, True, True, False])
        # No finite observation caps or new speed/travel limits are added.
        data.root_lin_vel_b.zero_()
        data.root_ang_vel_b.zero_()
        data.joint_vel.zero_()
        data.joint_pos[:, 1] = 200.
        self.assertFalse(self.numeric.nonfinite_or_extreme_state_v2(env).any())

    def test_proxy_array_style_raw_buffers_are_supported(self):
        env, data, _ = self.make_env()
        for field in ("root_pos_w", "root_quat_w", "joint_pos", "root_link_lin_vel_w", "root_link_ang_vel_w"):
            setattr(data, field, SimpleNamespace(torch=getattr(data, field)))
        data.joint_pos.torch[0, 0] = float("nan")
        self.assertEqual(self.numeric.nonfinite_or_extreme_state_v2(env).tolist(), [True, False, False, False])

    def test_finite_drift_reward_is_unchanged_and_invalid_position_yields_zero(self):
        torch = self.torch
        env, data, _ = self.make_env()
        data.root_pos_w[:, 0] = torch.tensor([0., .1, .4, .5])
        expected = self.rewards.balance_drift_gaussian(env)
        torch.testing.assert_close(self.numeric.finite_balance_drift_gaussian(env), expected, rtol=0, atol=0)
        data.root_pos_w[1, 0], data.root_pos_w[2, 1] = float("nan"), float("inf")
        actual = self.numeric.finite_balance_drift_gaussian(env)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(actual[1:3].tolist(), [0., 0.])
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[3], expected[3])

    def test_containment_overrides_are_v2_only_and_preserve_weights_limits_and_params(self):
        cfgs = [self.gym.spec(f"Trackstand-Balance-v{i}").kwargs["env_cfg_entry_point"]() for i in (0, 1, 2)]
        for cfg in cfgs[:2]:
            self.assertIs(cfg.terminations.diverged.func, self.legacy.nonfinite_or_extreme_state)
            self.assertIs(cfg.rewards.drift_gauss.func, self.rewards.balance_drift_gaussian)
        self.assertIs(cfgs[2].terminations.diverged.func, self.numeric.nonfinite_or_extreme_state_v2)
        self.assertIs(cfgs[2].rewards.drift_gauss.func, self.numeric.finite_balance_drift_gaussian)
        self.assertEqual(cfgs[2].terminations.diverged.params, cfgs[1].terminations.diverged.params)
        self.assertEqual(cfgs[2].rewards.drift_gauss.weight, 2.)
        self.assertEqual(cfgs[2].rewards.drift_gauss.params, {"sigma": .25})


if __name__ == "__main__":
    unittest.main()
