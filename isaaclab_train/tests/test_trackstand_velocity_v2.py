"""V2 rear PI tests using actual Isaac action/actuator classes."""

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
        from isaaclab.utils.types import ArticulationActions

        importlib.import_module(TASK_PACKAGE)
        pi = importlib.import_module(f"{TASK_PACKAGE}.mdp.stationary.velocity_pi")
        cls.gym, cls.torch = gym, torch
        cls.Actuator = IdealPDActuator
        cls.Action, cls.ActionCfg = pi.IntegralJointVelocityAction, pi.IntegralJointVelocityActionCfg
        cls.Integral = pi.VelocityIntegral
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

    def make_action(self, size=5, physics_dt=.0025, **overrides):
        torch = self.torch
        writes = []
        asset = SimpleNamespace(
            num_joints=3,
            # A nonzero default proves use_default_offset=False is effective.
            data=SimpleNamespace(
                default_joint_vel=SimpleNamespace(torch=torch.full((size, 3), 7.)),
                joint_vel=SimpleNamespace(torch=torch.zeros((size, 3))),
            ),
            actuators={"drive": self.make_actuator(size)},
            find_joints=lambda names, preserve_order=False: ([1], ["rear_wheel_joint"]),
            set_joint_velocity_target_index=lambda target, joint_ids, env_ids=None: writes.append(
                (target.clone(), joint_ids)
            ),
            set_joint_effort_target_index=lambda target, joint_ids, env_ids=None: None,
        )
        env = SimpleNamespace(scene={"robot": asset}, num_envs=size, device="cpu", step_dt=.02, physics_dt=physics_dt)
        cfg = deepcopy(self.cfg().actions.drive)
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return self.Action(cfg, env), writes

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
            self.assertEqual((drive.stiffness, drive.damping), (0., .3))
            self.assertEqual(v2.actions.drive.integral_gain_nm_per_rad, 3.3)
            self.assertEqual(v2.actions.drive.integral_rate_hz, 200.)
            self.assertIsNone(v2.events.randomize_actuator_gains)
            # Revert exactly the intended differences, then compare
            # the complete configurations: observations, rewards, DR, resets, etc.
            v2.actions.drive = deepcopy(v1.actions.drive)
            drive.damping = v1.scene.robot.actuators["drive"].damping
            v2.events.randomize_actuator_gains = deepcopy(v1.events.randomize_actuator_gains)
            v2.sim.dt, v2.decimation = v1.sim.dt, v1.decimation
            v2.sim.render_interval = v1.sim.render_interval
            self.assertEqual(v2.to_dict(), v1.to_dict())

    def test_timing_observations_and_steering_are_unchanged(self):
        v1, v2 = self.cfg(1), self.cfg()
        self.assertEqual((v2.sim.dt, v2.decimation, v2.episode_length_s), (1 / 1200, 24, 30.))
        self.assertAlmostEqual(v2.sim.dt * v2.decimation, .02)
        self.assertEqual(v2.sim.render_interval, 24)
        self.assertEqual((v1.sim.dt, v1.decimation), (.0025, 8))
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
        expected = .3 * (target - speed)
        torch.testing.assert_close(actuator.computed_effort, expected)
        torch.testing.assert_close(result.joint_efforts, expected.clamp(-6.28, 6.28))
        self.assertIsNone(result.joint_positions)
        self.assertIsNone(result.joint_velocities)
        self.assertTrue((expected.abs() > 6.28).any().item())

    def test_zero_velocity_action_brakes_in_both_directions(self):
        torch = self.torch
        speed = torch.tensor([[-40.], [-1.], [0.], [1.], [40.]])
        _, result = self.calculate(torch.zeros_like(speed), speed)
        torch.testing.assert_close(result.joint_efforts, torch.tensor([[6.28], [.3], [0.], [-.3], [-6.28]]))

    def test_target_reached_has_zero_proportional_torque_before_external_i(self):
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
                for name, expected in (("steering", (10., .2)), ("drive", (0., .3))):
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
            self.assertEqual(fresh.scene.robot.actuators["drive"].damping, (.05, 0., .3)[version])
            self.assertEqual(len(fresh.observations["policy"].imu.noise.std), 13 if version == 0 else 11)

    def test_integral_updates_at_200_hz_with_held_50_hz_target(self):
        torch = self.torch
        for physics_hz in (400, 800, 1000, 1200, 1600):
            with self.subTest(physics_hz=physics_hz):
                action, _ = self.make_action(2, physics_dt=1 / physics_hz)
                action.process_actions(torch.tensor([[1 / 61.6], [-1 / 61.6]]))
                samples = []
                for _ in range(physics_hz // 50):
                    action.apply_actions()
                    samples.append(action.integral.torque_nm.clone())
                for index, sample in enumerate(samples):
                    expected = 3.3 * .005 * (index // (physics_hz // 200))
                    torch.testing.assert_close(sample, torch.tensor([[expected], [-expected]]))

    def test_new_targets_and_zero_error_retain_integral_memory(self):
        torch = self.torch
        action, _ = self.make_action(1)
        action.process_actions(torch.tensor([[1 / 61.6]]))
        for _ in range(8):
            action.apply_actions()
        before = action.integral.torque_nm.clone()
        action.process_actions(torch.zeros(1, 1))
        action.apply_actions()  # Trapezoid across the target change, as on the Pi.
        torch.testing.assert_close(action.integral.torque_nm, before + .5 * 3.3 * .005)
        held = action.integral.torque_nm.clone()
        for _ in range(10):
            action.apply_actions()
        torch.testing.assert_close(action.integral.torque_nm, held)

    def test_partial_and_full_reset_clear_only_selected_i_and_targets(self):
        torch = self.torch
        action, _ = self.make_action(3)
        velocities, efforts = [], []
        action._asset.set_joint_velocity_target_index = lambda **kw: velocities.append(kw)
        action._asset.set_joint_effort_target_index = lambda **kw: efforts.append(kw)
        action.process_actions(torch.full((3, 1), 1 / 61.6))
        for _ in range(8):
            action.apply_actions()
        action.reset(torch.tensor([0, 2]))
        torch.testing.assert_close(action.integral.torque_nm[:, 0], torch.tensor([0., .0495, 0.]))
        torch.testing.assert_close(action.processed_actions[:, 0], torch.tensor([0., 1., 0.]))
        self.assertEqual(action.integral.initialized[:, 0].tolist(), [False, True, False])
        self.assertEqual(velocities[-1]["target"].shape, (2, 1))
        torch.testing.assert_close(efforts[-1]["target"], torch.zeros(2, 1))
        action.apply_actions()
        torch.testing.assert_close(action.integral.torque_nm[:, 0], torch.tensor([0., .066, 0.]))
        action.reset()
        torch.testing.assert_close(action.integral.torque_nm, torch.zeros(3, 1))
        torch.testing.assert_close(action.processed_actions, torch.zeros(3, 1))
        self.assertFalse(action.integral.initialized.any())

    def test_i_feedforward_reaches_actual_ideal_pd_and_clips_combined_torque(self):
        torch = self.torch
        action, _ = self.make_action(3)
        outputs = []
        action._asset.set_joint_effort_target_index = lambda **kw: outputs.append(kw["target"].clone())
        action.integral.torque_nm[:] = torch.tensor([[.106], [-.106], [6.]])
        action.integral.error_rad[:] = action.integral.torque_nm / 3.3
        speed = torch.tensor([[1.], [-1.], [0.]])
        action._asset.data.joint_vel.torch[:, 1] = speed[:, 0]
        target = torch.tensor([[1.], [-1.], [10.]])
        action.process_actions(target / 61.6)
        action.apply_actions()
        torch.testing.assert_close(outputs[-1], action.integral.torque_nm)
        actuator = action._asset.actuators["drive"]
        result = actuator.compute(
            self.Command(joint_positions=torch.zeros_like(speed), joint_velocities=action.processed_actions,
                         joint_efforts=outputs[-1]),
            joint_pos=torch.zeros_like(speed), joint_vel=speed,
        )
        torch.testing.assert_close(actuator.computed_effort, torch.tensor([[.106], [-.106], [9.]]))
        torch.testing.assert_close(result.joint_efforts, torch.tensor([[.106], [-.106], [6.28]]))

    def test_anti_windup_freezes_outward_updates_and_allows_unwinding(self):
        torch = self.torch
        pi = self.Integral(torch.zeros(2, 1), 3.3)
        kp, limit = torch.full((2, 1), .3), torch.full((2, 1), 6.28)
        for _ in range(100):
            pi.update(torch.tensor([[100.], [-100.]]), .005, kp, limit)
        torch.testing.assert_close(pi.torque_nm, torch.zeros(2, 1))
        self.assertTrue((pi.limited_updates > 0).all())
        pi.torque_nm[:] = torch.tensor([[6.], [-6.]])
        pi.error_rad[:] = pi.torque_nm / 3.3
        pi.previous_error[:] = torch.tensor([[-1.], [1.]])
        pi.update(pi.previous_error.clone(), .005, kp, limit)
        torch.testing.assert_close(pi.torque_nm, torch.tensor([[5.9835], [-5.9835]]))
        torch.testing.assert_close(pi.error_rad * 3.3, pi.torque_nm)

    def test_integral_clamp_updates_stored_error_without_hidden_windup(self):
        torch = self.torch
        pi = self.Integral(torch.zeros(1, 1), 3.3)
        kp, limit = torch.zeros(1, 1), torch.full((1, 1), 6.28)
        for _ in range(20):
            pi.update(torch.full((1, 1), 1000.), .005, kp, limit)
        torch.testing.assert_close(pi.torque_nm, limit)
        torch.testing.assert_close(pi.error_rad * 3.3, limit)
        pi.update(torch.full((1, 1), -1000.), .005, kp, limit)  # Trapezoid crosses zero.
        pi.update(torch.full((1, 1), -1000.), .005, kp, limit)
        torch.testing.assert_close(pi.torque_nm, -limit)
        torch.testing.assert_close(pi.error_rad * 3.3, -limit)

    def test_trapezoidal_i_matches_hardware_test_in_unsaturated_regime(self):
        torch = self.torch
        experiment = Path(__file__).resolve().parents[2] / "deploy/experiments"
        sys.path.insert(0, str(experiment))
        try:
            hardware = importlib.import_module("drive_mit_velocity_test")
        finally:
            sys.path.pop(0)
        reference = object.__new__(hardware.MitVelocityTestIO)
        reference.ki, reference.kd, reference.command_ranges = 3.3, .3, (95.5, 13., 18.)
        reference.reset_integral()
        pi = self.Integral(torch.zeros(1, 1, dtype=torch.float64), 3.3)
        kp, limit = torch.tensor([[.3]]), torch.tensor([[6.28]])
        for index in range(300):
            target = .2 if index < 100 else -.1 if index < 200 else 0.
            reference.velocity = .05 if index % 3 else -.02
            reference.update_integral(target, index * .005)
            pi.update(torch.tensor([[target - reference.velocity]], dtype=torch.float64), .005, kp, limit)
            self.assertAlmostEqual(pi.torque_nm.item(), reference.integral_nm, places=12)
        pi.reset()
        torch.testing.assert_close(pi.torque_nm, torch.zeros_like(pi.torque_nm))

    def test_zero_ki_and_invalid_update_rates(self):
        torch = self.torch
        action, _ = self.make_action(1, integral_gain_nm_per_rad=0.)
        action.process_actions(torch.ones(1, 1))
        for _ in range(12):
            action.apply_actions()
        torch.testing.assert_close(action.integral.torque_nm, torch.zeros(1, 1))
        for hz in (0., -1., 300., 800., float("nan")):
            with self.subTest(hz=hz), self.assertRaises(ValueError):
                self.make_action(1, integral_rate_hz=hz)
        for ki in (-1., float("inf")):
            with self.subTest(ki=ki), self.assertRaises(ValueError):
                self.make_action(1, integral_gain_nm_per_rad=ki)


if __name__ == "__main__":
    unittest.main()
