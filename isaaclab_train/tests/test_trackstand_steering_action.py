"""Test the actual v1 action term on a mock articulation, without physics."""

import copy
import importlib
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
TASK_PACKAGE = "source.trackstand_rl.trackstand_rl.tasks.manager_based.trackstand"


@unittest.skipUnless(importlib.util.find_spec("isaaclab"), "Run with Isaac Lab Python")
class TestSteeringAction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gymnasium as gym
        import torch

        importlib.import_module(TASK_PACKAGE)
        cls.torch = torch
        cls.action_cfg = gym.spec("Trackstand-Balance-v1").kwargs["env_cfg_entry_point"]().actions.steering

    def make_action(self, cfg=None, step_dt=0.02):
        torch = self.torch
        positions = torch.zeros(3, 3)
        writes = []
        asset = SimpleNamespace(
            num_joints=3,
            data=SimpleNamespace(
                joint_pos=SimpleNamespace(torch=positions),
                default_joint_pos=SimpleNamespace(torch=torch.zeros_like(positions)),
            ),
            find_joints=lambda names, preserve_order=False: ([1], ["steering_joint"]),
            set_joint_position_target_index=lambda target, joint_ids: writes.append(target.clone()),
        )
        env = SimpleNamespace(scene={"robot": asset}, num_envs=3, device="cpu", step_dt=step_dt)
        cfg = copy.deepcopy(self.action_cfg if cfg is None else cfg)
        return cfg.class_type(cfg, env), positions, writes

    def test_scale_clip_then_slew(self):
        torch = self.torch
        action, _, _ = self.make_action()
        requests = torch.tensor([[1000.0], [-1000.0], [0.1]])
        for step, magnitude in enumerate((0.478, 0.956, 1.434, 1.57), 1):
            action.process_actions(requests)
            expected = torch.tensor([[magnitude], [-magnitude], [0.157]])
            torch.testing.assert_close(action.processed_actions, expected, atol=2e-7, rtol=1e-6)
            torch.testing.assert_close(action.raw_actions, requests, rtol=0, atol=0)
            self.assertLessEqual(action.processed_actions.abs().max().item(), 1.570001, step)

    def test_physics_applications_hold_the_target_and_do_not_advance_slew(self):
        torch = self.torch
        action, positions, writes = self.make_action()
        action.process_actions(torch.ones(3, 1))
        expected = action.processed_actions.clone()
        for _ in range(8):
            positions[:, 1] += 0.05  # Measured motion must not change the held target.
            action.apply_actions()
            torch.testing.assert_close(action.processed_actions, expected, rtol=0, atol=0)
        self.assertEqual(len(writes), 8)
        for target in writes:
            torch.testing.assert_close(target, expected, rtol=0, atol=0)
        action.process_actions(torch.ones(3, 1))
        torch.testing.assert_close(action.processed_actions, torch.full((3, 1), 0.956))

    def test_reversal_respects_rate_in_both_directions(self):
        torch = self.torch
        action, _, _ = self.make_action()
        for _ in range(4):
            action.process_actions(torch.ones(3, 1))
        previous = action.processed_actions.clone()
        for _ in range(7):
            action.process_actions(-torch.ones(3, 1))
            self.assertLessEqual((action.processed_actions - previous).abs().max().item(), 0.478001)
            previous = action.processed_actions.clone()
        torch.testing.assert_close(action.processed_actions, torch.full((3, 1), -1.57))

    def test_partial_reset_primes_only_selected_environments(self):
        torch = self.torch
        action, positions, _ = self.make_action()
        action.process_actions(torch.ones(3, 1))
        positions[:, 1] = torch.tensor([-0.6, 0.9, -0.8])
        action.reset(env_ids=torch.tensor([0, 2]))
        torch.testing.assert_close(action.processed_actions, torch.tensor([[-0.6], [0.478], [-0.8]]))
        torch.testing.assert_close(action.raw_actions, torch.tensor([[0.0], [1.0], [0.0]]))
        action.process_actions(torch.zeros(3, 1))
        torch.testing.assert_close(action.processed_actions, torch.tensor([[-0.122], [0.0], [-0.322]]))

    def test_full_reset_uses_measured_positions_clipped_to_command_bounds(self):
        torch = self.torch
        action, positions, _ = self.make_action()
        positions[:, 1] = torch.tensor([0.3, 2.0, -2.0])
        action.reset()
        torch.testing.assert_close(action.processed_actions, torch.tensor([[0.3], [1.57], [-1.57]]))
        action.process_actions(torch.zeros(3, 1))
        torch.testing.assert_close(action.processed_actions, torch.tensor([[0.0], [1.092], [-1.092]]))

    def test_empty_reset_does_not_modify_other_environments(self):
        torch = self.torch
        action, positions, _ = self.make_action()
        action.process_actions(torch.ones(3, 1))
        before = action.processed_actions.clone()
        positions[:, 1] = 1.0
        action.reset(env_ids=torch.empty(0, dtype=torch.long))
        torch.testing.assert_close(action.processed_actions, before, rtol=0, atol=0)

    def test_non_50_hz_policy_is_rejected(self):
        for step_dt in (0.0025, 0.005, 0.01, 0.04):
            with self.subTest(step_dt=step_dt), self.assertRaisesRegex(ValueError, "50-Hz"):
                self.make_action(step_dt=step_dt)

    def test_invalid_rate_or_missing_clip_is_rejected(self):
        for rate in (0.0, -1.0, float("nan"), float("inf")):
            cfg = copy.deepcopy(self.action_cfg)
            cfg.max_target_velocity_rad_s = rate
            with self.subTest(rate=rate), self.assertRaisesRegex(ValueError, "positive and finite"):
                self.make_action(cfg=cfg)
        cfg = copy.deepcopy(self.action_cfg)
        cfg.clip = None
        with self.assertRaisesRegex(ValueError, "explicit position clip"):
            self.make_action(cfg=cfg)


if __name__ == "__main__":
    unittest.main()
