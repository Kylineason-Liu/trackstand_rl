"""Test v1's real IdealPDActuator calculation and configuration on CPU."""

import importlib
import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
TASK_PACKAGE = "source.trackstand_rl.trackstand_rl.tasks.manager_based.trackstand"


@unittest.skipUnless(importlib.util.find_spec("isaaclab"), "Run with Isaac Lab Python")
class TestDriveActuator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gymnasium as gym
        import torch
        from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg, ImplicitActuatorCfg
        from isaaclab.utils.types import ArticulationActions

        importlib.import_module(TASK_PACKAGE)
        cls.gym, cls.torch = gym, torch
        cls.IdealPDActuator, cls.IdealPDActuatorCfg = IdealPDActuator, IdealPDActuatorCfg
        cls.ImplicitActuatorCfg, cls.ArticulationActions = ImplicitActuatorCfg, ArticulationActions

    def env_cfg(self, version=1, **kwargs):
        return self.gym.spec(f"Trackstand-Balance-v{version}").kwargs["env_cfg_entry_point"](**kwargs)

    def make_actuator(self, num_envs, damping=None):
        cfg = self.env_cfg().scene.robot.actuators["drive"]
        if damping is not None:
            cfg.damping = damping
        return self.IdealPDActuator(
            cfg=cfg, joint_names=["rear_wheel_joint"], joint_ids=slice(None),
            num_envs=num_envs, device="cpu",
        )

    def calculate(self, actuator, feedforward, speed, velocity_target=None):
        torch = self.torch
        if velocity_target is None:
            velocity_target = torch.zeros_like(speed)
        return actuator.compute(
            self.ArticulationActions(
                joint_positions=torch.ones_like(speed), joint_velocities=velocity_target,
                joint_efforts=feedforward.clone(),
            ),
            joint_pos=-torch.ones_like(speed), joint_vel=speed,
        )

    def test_v1_only_actuator_and_limits(self):
        for dr in (False, True):
            cfg = self.env_cfg(domain_randomization=dr)
            drive = cfg.scene.robot.actuators["drive"]
            self.assertIs(type(drive), self.IdealPDActuatorCfg)
            self.assertEqual(drive.joint_names_expr, ["rear_wheel_joint"])
            self.assertEqual((drive.stiffness, drive.damping), (0.0, 0.0))
            self.assertEqual((drive.friction, drive.dynamic_friction), (0.106, 0.106))
            self.assertIsNone(drive.viscous_friction)
            self.assertEqual((drive.effort_limit, drive.effort_limit_sim), (6.28, 6.28))
            self.assertEqual((drive.velocity_limit, drive.velocity_limit_sim), (61.6, 61.6))
            steering = cfg.scene.robot.actuators["steering"]
            self.assertIs(type(steering), self.ImplicitActuatorCfg)
            self.assertEqual((steering.effort_limit, steering.effort_limit_sim), (5.6, 5.6))
            self.assertEqual((steering.velocity_limit, steering.velocity_limit_sim), (23.9, 23.9))
            for actuator in self.env_cfg(0).scene.robot.actuators.values():
                self.assertIs(type(actuator), self.ImplicitActuatorCfg)
                self.assertIsNone(actuator.effort_limit)
                self.assertIsNone(actuator.velocity_limit)
                self.assertIsNone(actuator.friction)
                self.assertIsNone(actuator.dynamic_friction)

    def test_scaled_feedforward_plus_damping_is_clipped_before_submission(self):
        torch = self.torch
        actions = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
        speeds = torch.tensor([-61.6, -20.0, 0.0, 20.0, 61.6])
        action, speed = torch.meshgrid(actions, speeds, indexing="ij")
        action, speed = action.reshape(-1, 1), speed.reshape(-1, 1)
        actuator = self.make_actuator(len(action), damping=0.05)
        feedforward = 6.28 * action
        expected_unclipped = feedforward - 0.05 * speed
        expected = expected_unclipped.clamp(-6.28, 6.28)
        output = self.calculate(actuator, feedforward, speed)
        torch.testing.assert_close(actuator.computed_effort, expected_unclipped)
        torch.testing.assert_close(actuator.applied_effort, expected)
        torch.testing.assert_close(output.joint_efforts, expected)
        self.assertIsNone(output.joint_positions)
        self.assertIsNone(output.joint_velocities)
        self.assertLessEqual(output.joint_efforts.abs().max().item(), 6.280001)

    def test_opposing_damping_cannot_exceed_the_torque_command_limit(self):
        torch = self.torch
        actuator = self.make_actuator(2, damping=0.05)
        speed = torch.tensor([[61.6], [-61.6]])
        feedforward = torch.tensor([[-6.28], [6.28]])
        output = self.calculate(actuator, feedforward, speed)
        torch.testing.assert_close(actuator.computed_effort, torch.tensor([[-9.36], [9.36]]))
        torch.testing.assert_close(output.joint_efforts, feedforward)

    def test_zero_policy_action_still_produces_clipped_damping(self):
        torch = self.torch
        speed = torch.tensor([[-20.0], [20.0], [-200.0], [200.0]])
        output = self.calculate(self.make_actuator(4, damping=0.05), torch.zeros_like(speed), speed)
        torch.testing.assert_close(output.joint_efforts, torch.tensor([[1.0], [-1.0], [6.28], [-6.28]]))

    def test_default_zero_damping_preserves_requested_torque_at_all_speeds(self):
        torch = self.torch
        action, speed = torch.meshgrid(
            torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0]),
            torch.tensor([-61.6, -20.0, 0.0, 20.0, 61.6]), indexing="ij",
        )
        action, speed = action.reshape(-1, 1), speed.reshape(-1, 1)
        feedforward = 6.28 * action
        actuator = self.make_actuator(len(action))
        output = self.calculate(actuator, feedforward, speed)
        torch.testing.assert_close(actuator.computed_effort, feedforward)
        torch.testing.assert_close(output.joint_efforts, feedforward)

    def test_default_zero_damping_still_clips_excess_effort(self):
        torch = self.torch
        speed = torch.tensor([[-61.6], [61.6]])
        output = self.calculate(self.make_actuator(2), torch.tensor([[-10.0], [10.0]]), speed)
        torch.testing.assert_close(output.joint_efforts, torch.tensor([[-6.28], [6.28]]))

    def test_friction_is_not_subtracted_from_the_actuator_torque_command(self):
        torch = self.torch
        actuator = self.make_actuator(2)
        torch.testing.assert_close(actuator.friction, torch.full((2, 1), 0.106))
        torch.testing.assert_close(actuator.dynamic_friction, torch.full((2, 1), 0.106))
        speed = torch.tensor([[-4.0], [4.0]])
        output = self.calculate(actuator, torch.zeros_like(speed), speed)
        torch.testing.assert_close(output.joint_efforts, torch.zeros_like(speed))

    def test_per_environment_gain_updates_are_used_before_clipping(self):
        torch = self.torch
        actuator = self.make_actuator(2)
        actuator.damping[:] = torch.tensor([[0.045], [0.055]])
        speed = torch.full((2, 1), 20.0)
        output = self.calculate(actuator, torch.zeros_like(speed), speed)
        torch.testing.assert_close(output.joint_efforts, torch.tensor([[-0.9], [-1.1]]))

    def test_v1_configuration_does_not_mutate_v0_or_other_instances(self):
        first = self.env_cfg()
        first.scene.robot.actuators["drive"].damping = 0.123
        first.scene.robot.actuators["drive"].friction = 0.789
        first.scene.robot.actuators["drive"].dynamic_friction = 0.456
        first.scene.robot.actuators["steering"].stiffness = 999.0
        for version in (0, 1):
            fresh = self.env_cfg(version)
            self.assertEqual(fresh.scene.robot.actuators["drive"].damping, 0.05 if version == 0 else 0.0)
            self.assertEqual(fresh.scene.robot.actuators["drive"].friction, None if version == 0 else 0.106)
            self.assertEqual(fresh.scene.robot.actuators["drive"].dynamic_friction, None if version == 0 else 0.106)
            self.assertEqual(fresh.scene.robot.actuators["steering"].stiffness, 10.0)


if __name__ == "__main__":
    unittest.main()
