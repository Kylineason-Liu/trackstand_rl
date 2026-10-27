"""V2 rear MIT velocity-law tests using actual Isaac action/actuator classes."""

from copy import deepcopy
import importlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
TASK_PACKAGE = "source.trackstand_rl.trackstand_rl.tasks.manager_based.trackstand"


@unittest.skipUnless(importlib.util.find_spec("isaaclab"), "Run with Isaac Lab Python")
class TestRearVelocityV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gymnasium as gym
        import torch
        from isaaclab.actuators import IdealPDActuator
        from isaaclab.envs.mdp.actions import JointVelocityAction, JointVelocityActionCfg
        from isaaclab.utils.types import ArticulationActions

        importlib.import_module(TASK_PACKAGE)
        cls.gym, cls.torch = gym, torch
        cls.Actuator, cls.Action, cls.ActionCfg = IdealPDActuator, JointVelocityAction, JointVelocityActionCfg
        cls.Command = ArticulationActions

    def cfg(self, version=2, **kwargs):
        return self.gym.spec(f"Trackstand-Balance-v{version}").kwargs["env_cfg_entry_point"](**kwargs)

    def make_actuator(self, size):
        return self.Actuator(cfg=self.cfg().scene.robot.actuators["drive"], joint_names=["rear_wheel_joint"],
                             joint_ids=slice(None), num_envs=size, device="cpu")

    def calculate(self, target, speed, position=None):
        torch = self.torch
        zeros = torch.zeros_like(speed)
        actuator = self.make_actuator(len(speed))
        result = actuator.compute(
            self.Command(joint_positions=zeros, joint_velocities=target, joint_efforts=zeros),
            joint_pos=zeros if position is None else position, joint_vel=speed,
        )
        return actuator, result

    def make_action(self, size=5):
        torch = self.torch
        writes = []
        asset = SimpleNamespace(
            num_joints=3,
            # A nonzero default proves use_default_offset=False is effective.
            data=SimpleNamespace(default_joint_vel=SimpleNamespace(torch=torch.full((size, 3), 7.))),
            find_joints=lambda names, preserve_order=False: ([1], ["rear_wheel_joint"]),
            set_joint_velocity_target_index=lambda target, joint_ids: writes.append((target.clone(), joint_ids)),
        )
        env = SimpleNamespace(scene={"robot": asset}, num_envs=size, device="cpu", step_dt=.02)
        return self.Action(deepcopy(self.cfg().actions.drive), env), writes

    def test_distinct_registration_and_v2_bookkeeping_override(self):
        spec = self.gym.spec("Trackstand-Balance-v2")
        self.assertTrue(spec.entry_point.endswith("trackstand_rl_env_v2:TrackstandRLEnvV2"))
        self.assertEqual(set(spec.kwargs), {"env_cfg_entry_point", "rsl_rl_cfg_entry_point"})
        module, name = spec.entry_point.split(":")
        env_cls = getattr(importlib.import_module(module), name)
        v1_cls = importlib.import_module(f"{TASK_PACKAGE}.trackstand_bike.trackstand_rl_env").TrackstandRLEnvV1
        self.assertTrue(issubclass(env_cls, v1_cls))
        self.assertIsNot(env_cls.step, v1_cls.step)
        self.assertIsNot(env_cls.reset, v1_cls.reset)
        self.assertIs(env_cls._bookkeeping_reset_mask, v1_cls._bookkeeping_reset_mask)

    def test_only_rear_action_controller_and_gain_dr_change_from_v1(self):
        for dr in (False, True):
            v1, v2 = self.cfg(1, domain_randomization=dr), self.cfg(2, domain_randomization=dr)
            self.assertIs(type(v2.actions.drive), self.ActionCfg)
            self.assertEqual((v2.actions.drive.scale, v2.actions.drive.use_default_offset), (61.6, False))
            self.assertEqual(v2.actions.drive.clip, {"rear_wheel_joint": (-61.6, 61.6)})
            drive = v2.scene.robot.actuators["drive"]
            self.assertEqual((drive.stiffness, drive.damping), (0., .2))
            self.assertIsNone(v2.events.randomize_actuator_gains)
            # Revert exactly the intended differences, then compare
            # the complete configurations: observations, rewards, DR, resets, etc.
            v2.actions.drive = deepcopy(v1.actions.drive)
            drive.damping = v1.scene.robot.actuators["drive"].damping
            v2.events.randomize_actuator_gains = deepcopy(v1.events.randomize_actuator_gains)
            self.assertEqual(v2.to_dict(), v1.to_dict())

    def test_timing_observations_and_steering_are_unchanged(self):
        v1, v2 = self.cfg(1), self.cfg()
        self.assertEqual((v2.sim.dt, v2.decimation, v2.episode_length_s), (.0025, 8, 30.))
        self.assertEqual(len(v2.observations["policy"].imu.noise.std), 11)
        self.assertIs(v2.observations["policy"].imu.func, v1.observations["policy"].imu.func)
        self.assertEqual(v2.actions.steering.to_dict(), v1.actions.steering.to_dict())

    def test_beta_runner_matches_v1_except_separate_checkpoint_directory(self):
        cfg_path = self.gym.spec("Trackstand-Balance-v2").kwargs["rsl_rl_cfg_entry_point"]
        module, name = cfg_path.split(":")
        v2 = getattr(importlib.import_module(module), name)()
        v1 = importlib.import_module(f"{TASK_PACKAGE}.agents.rsl_rl_ppo_beta_cfg").TrackstandRLBetaPPORunnerCfg()
        self.assertEqual(v2.experiment_name, "trackstand_rl_v2_rsl")
        self.assertIn("BetaDistribution", v2.actor.distribution_cfg.class_name)
        self.assertFalse(v2.actor.obs_normalization)
        v2.experiment_name = v1.experiment_name
        self.assertEqual(v2.to_dict(), v1.to_dict())

    def test_velocity_action_units_scale_and_zero_offset(self):
        torch = self.torch
        action, writes = self.make_action()
        raw = torch.tensor([[-1.], [-.5], [0.], [.25], [1.]])
        action.process_actions(raw)
        torch.testing.assert_close(action.processed_actions, raw * 61.6)
        action.apply_actions()
        torch.testing.assert_close(writes[-1][0], raw * 61.6)
        self.assertEqual(writes[-1][1], [1])

    def test_explicit_velocity_target_clip(self):
        torch = self.torch
        action, _ = self.make_action(3)
        action.process_actions(torch.tensor([[-100.], [0.], [100.]]))
        torch.testing.assert_close(action.processed_actions, torch.tensor([[-61.6], [0.], [61.6]]))

    def test_eight_physics_applications_hold_one_velocity_target(self):
        torch = self.torch
        action, writes = self.make_action(3)
        action.process_actions(torch.tensor([[-.1], [0.], [.1]]))
        held = action.processed_actions.clone()
        for _ in range(8):
            action.apply_actions()
        self.assertEqual(len(writes), 8)
        for target, _ in writes:
            torch.testing.assert_close(target, held, rtol=0, atol=0)
        action.process_actions(torch.ones(3, 1))
        torch.testing.assert_close(action.processed_actions, torch.full((3, 1), 61.6))

    def test_velocity_error_pd_is_clipped_after_controller(self):
        torch = self.torch
        targets, speeds = torch.meshgrid(torch.tensor([-61.6, -1., 0., 1., 61.6]),
                                        torch.tensor([-61.6, -1., 0., 1., 61.6]), indexing="ij")
        target, speed = targets.reshape(-1, 1), speeds.reshape(-1, 1)
        actuator, result = self.calculate(target, speed, position=torch.full_like(speed, 999.))
        expected = .2 * (target - speed)
        torch.testing.assert_close(actuator.computed_effort, expected)
        torch.testing.assert_close(result.joint_efforts, expected.clamp(-6.28, 6.28))
        self.assertIsNone(result.joint_positions)
        self.assertIsNone(result.joint_velocities)
        self.assertTrue((expected.abs() > 6.28).any().item())

    def test_zero_velocity_action_brakes_in_both_directions(self):
        torch = self.torch
        speed = torch.tensor([[-40.], [-1.], [0.], [1.], [40.]])
        _, result = self.calculate(torch.zeros_like(speed), speed)
        torch.testing.assert_close(result.joint_efforts, torch.tensor([[6.28], [.2], [0.], [-.2], [-6.28]]))

    def test_target_reached_has_zero_motor_torque_before_passive_losses(self):
        torch = self.torch
        speed = torch.tensor([[-8.], [0.], [8.]])
        actuator, result = self.calculate(speed, speed)
        torch.testing.assert_close(result.joint_efforts, torch.zeros_like(speed))
        torch.testing.assert_close(actuator.friction, torch.full_like(speed, .106))
        torch.testing.assert_close(actuator.dynamic_friction, torch.full_like(speed, .106))
        torch.testing.assert_close(actuator.viscous_friction, torch.zeros_like(speed))

    def test_all_v1_motor_limits_and_passive_losses_are_retained(self):
        v1, v2 = self.cfg(1), self.cfg()
        for name in v1.scene.robot.actuators:
            one, two = v1.scene.robot.actuators[name].to_dict(), v2.scene.robot.actuators[name].to_dict()
            if name == "drive":
                two["damping"] = one["damping"]
            self.assertEqual(one, two)

    def test_gain_dr_stays_disabled_through_constructor_and_cli_switches(self):
        from scripts.domain_randomization import configure_domain_randomization

        for initial_dr in (False, True):
            cfg = self.cfg(domain_randomization=initial_dr)
            for enabled in (initial_dr, True, True, False, False, True):
                configure_domain_randomization(cfg, enabled)
                self.assertEqual(cfg.domain_randomization, enabled)
                self.assertIsNone(cfg.events.randomize_actuator_gains)
                for name, expected in (("steering", (10., .2)), ("drive", (0., .2))):
                    actuator = cfg.scene.robot.actuators[name]
                    self.assertEqual((actuator.stiffness, actuator.damping), expected)
                # All other event parameters must still match v1 in either mode.
                events = deepcopy(cfg.events.to_dict())
                expected = self.cfg(1, domain_randomization=enabled).events.to_dict()
                events.pop("randomize_actuator_gains")
                expected.pop("randomize_actuator_gains")
                self.assertEqual(events, expected)
                # Repeated v2 switches must not remove DR from v0/v1 instances.
                for version in (0, 1):
                    self.assertIsNotNone(self.cfg(version, domain_randomization=True).events.randomize_actuator_gains)

    def test_v2_instance_mutations_never_change_v0_v1_or_fresh_v2(self):
        one = self.cfg()
        one.actions.drive.scale = 8.
        one.actions.steering.scale = .1
        one.scene.robot.actuators["drive"].damping = 123.
        one.scene.robot.actuators["steering"].stiffness = 123.
        one.observations["policy"].imu.noise.std = (1.,)
        for version in (0, 1, 2):
            fresh = self.cfg(version)
            self.assertEqual(fresh.actions.steering.scale, 1.57)
            self.assertEqual(fresh.scene.robot.actuators["steering"].stiffness, 10.)
            self.assertEqual(fresh.actions.drive.scale, 61.6 if version == 2 else 6.28)
            self.assertEqual(fresh.scene.robot.actuators["drive"].damping, (.05, 0., .2)[version])
            self.assertEqual(len(fresh.observations["policy"].imu.noise.std), 13 if version == 0 else 11)


if __name__ == "__main__":
    unittest.main()
