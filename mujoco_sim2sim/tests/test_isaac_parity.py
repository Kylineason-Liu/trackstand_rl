"""Execute real Isaac task terms on matched states; no GPU simulation needed.

Run with ~/IsaacLab/_isaac_sim/python.sh -B -m unittest discover -s tests -v.
These tests skip outside the Isaac Python environment, rather than fake imports.
"""

import importlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "isaaclab_train"))
sys.path.insert(0, str(ROOT / "mujoco_sim2sim/src"))
from newbike_mujoco import contract as c
from newbike_mujoco.env import physx_revolute_angle
from newbike_mujoco.policy import DeployedPolicy
from test_multi_viewer import first_env

PACKAGE = "source.trackstand_rl.trackstand_rl.tasks.manager_based.trackstand"


@unittest.skipUnless(importlib.util.find_spec("isaaclab"), "Run with Isaac Python for live task-term parity")
class IsaacParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gymnasium as gym
        import isaaclab.utils.math as math_utils
        importlib.import_module(PACKAGE)
        cls.cfg = gym.spec(c.TASK).kwargs["env_cfg_entry_point"]()
        cls.math_utils = math_utils
        cls.terms = importlib.import_module(f"{PACKAGE}.mdp.stationary.terminations")

    def test_actual_resolved_v1_configuration(self):
        cfg = self.cfg
        self.assertEqual((cfg.sim.dt, cfg.decimation, cfg.episode_length_s), (.0025, 8, 30.))
        self.assertEqual(cfg.observations["policy"].imu.noise.std, c.OBS_NOISE_STD)
        self.assertEqual(cfg.actions.steering.scale, c.STEER_SCALE)
        self.assertEqual(cfg.actions.steering.max_target_velocity_rad_s, c.STEER_SLEW_RATE)
        self.assertEqual(cfg.actions.steering.clip, {"steering_joint": (-1.57, 1.57)})
        self.assertEqual(cfg.actions.drive.scale, c.DRIVE_SCALE)
        for name, kp, kd, effort, speed, friction, viscous in (
            ("steering", 10., .2, 5.6, 23.9, c.STEER_FRICTION, c.STEER_VISCOUS_FRICTION),
            ("drive", 0., 0., 6.28, 61.6, .106, None),
        ):
            actuator = cfg.scene.robot.actuators[name]
            self.assertEqual((actuator.stiffness, actuator.damping), (kp, kd))
            self.assertEqual((actuator.effort_limit, actuator.effort_limit_sim), (effort, effort))
            self.assertEqual((actuator.velocity_limit, actuator.velocity_limit_sim), (speed, speed))
            self.assertEqual((actuator.friction, actuator.dynamic_friction), (friction, friction))
            self.assertEqual(actuator.viscous_friction, viscous)
        self.assertEqual(cfg.scene.robot.init_state.pos, (0., 0., c.SPAWN_Z))

    @unittest.skipUnless(c.DEFAULT_CHECKPOINT.is_file(), "Local model75300 checkpoint not installed")
    def test_actual_beta_mean_mapping(self):
        BetaMean = importlib.import_module(f"{PACKAGE}.agents.beta_distribution").BetaMean
        policy = DeployedPolicy(pt_path=c.DEFAULT_CHECKPOINT)
        obs = np.random.default_rng(42).normal(size=(1024, 11)).astype(np.float32)
        with torch.no_grad():
            logits = policy._torch_net(torch.from_numpy(obs)).reshape(-1, 2, 2)
            expected = BetaMean()(logits).numpy()
        np.testing.assert_array_equal(policy.act_batch(obs), expected)

    def test_actual_dr_ranges_and_sampling_configuration(self):
        import gymnasium as gym
        cfg = gym.spec(c.TASK).kwargs["env_cfg_entry_point"](domain_randomization=True)
        events = cfg.events
        self.assertEqual(events.randomize_mass.params["mass_distribution_params"], c.DR_MASS)
        self.assertEqual(events.randomize_com.params["com_range"], {axis: (-c.DR_COM, c.DR_COM) for axis in "xyz"})
        material = events.randomize_material.params
        self.assertEqual(material["static_friction_range"], c.DR_FRICTION)
        self.assertEqual(material["dynamic_friction_range"], c.DR_FRICTION)
        self.assertEqual(material["num_buckets"], c.DR_FRICTION_BUCKETS)
        self.assertTrue(material["make_consistent"])
        gains = events.randomize_actuator_gains.params
        self.assertEqual(gains["stiffness_distribution_params"], c.DR_GAINS)
        self.assertEqual(gains["damping_distribution_params"], c.DR_GAINS)
        self.assertEqual(events.initial_state.params["pose_range"]["roll"], c.DR_INIT_ROLL)
        self.assertEqual(events.initial_state.params["velocity_range"]["roll"], c.DR_INIT_ROLL_RATE)
        residual = events.reset_imu_turnon_residuals.params
        self.assertEqual((residual["gyro_res_bias_std"], residual["accel_res_bias_std"]),
                         (c.DR_GYRO_RES, c.DR_ACCEL_RES))

    def test_actual_observation_function_on_1000_matched_states(self):
        rng = np.random.default_rng(42)
        env = first_env()
        env.OBS_NOISE_STD = (0.,)*11
        env.reset()
        data = SimpleNamespace()
        robot = SimpleNamespace(data=data, find_joints=lambda name: ([0 if name == "steering_joint" else 1], [name]))
        isaac = SimpleNamespace(scene={"robot": robot}, device="cpu", num_envs=1)
        obs_term = self.cfg.observations["policy"].imu
        for _ in range(1000):
            q = rng.normal(size=4).astype(np.float32)
            q /= np.linalg.norm(q)
            body_omega = rng.normal(size=3).astype(np.float32)
            steer, rear = rng.uniform(-1.5, 1.5), rng.uniform(-20., 20.)
            speeds = rng.normal(size=2).astype(np.float32)*150
            env.data.qpos[3:7] = q[[3, 0, 1, 2]]
            env.data.qpos[[env._steer_qadr, env._rear_qadr]] = [steer, rear]
            env.data.qvel[3:6] = body_omega
            env.data.qvel[env._dofs] = speeds
            quat = torch.from_numpy(q[None])
            data.root_quat_w = SimpleNamespace(torch=quat)
            data.root_link_ang_vel_w = SimpleNamespace(
                torch=self.math_utils.quat_apply(quat, torch.from_numpy(body_omega[None])))
            data.joint_pos = torch.tensor([[steer, physx_revolute_angle(rear)]], dtype=torch.float32)
            data.joint_vel = torch.from_numpy(speeds[None])
            expected = obs_term.func(isaac, **obs_term.params).numpy()[0]
            np.testing.assert_allclose(env.observe(), expected, atol=2e-6, rtol=2e-6)

    def test_actual_termination_functions_match_boundary_cases(self):
        env = first_env()
        env.reset()
        for roll, yaw, drift, speed, wheel, count in (
            (0., 0., 0., 0., 0., 0), (.53, 0., 0., 0., 0., 0),
            (0., .53, .51, 301., 7.21, 1500), (0., -.53, .49, 299., -7.21, 1499),
        ):
            quat = self.math_utils.quat_from_euler_xyz(torch.tensor([roll]), torch.tensor([0.]), torch.tensor([yaw]))
            gravity = self.math_utils.quat_apply_inverse(quat, torch.tensor([[0., 0., -1.]]))
            data = SimpleNamespace(projected_gravity_b=gravity, root_quat_w=SimpleNamespace(torch=quat),
                                   root_pos_w=torch.tensor([[drift, 0., -.016]]), root_lin_vel_b=torch.zeros(1, 3),
                                   root_ang_vel_b=torch.zeros(1, 3), joint_vel=torch.tensor([[0., speed, 0.]]))
            isaac = SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(1, 3)),
                                    spawn_yaw=torch.zeros(1), wheel_net_angle=torch.tensor([wheel]), device="cpu")
            # The real terms index scene["robot"], in addition to origins.
            class Scene(dict):
                env_origins = torch.zeros(1, 3)
            isaac.scene = Scene(robot=SimpleNamespace(data=data))
            env.data.qpos[:2] = [drift, 0.]
            env.data.qpos[3:7] = quat.numpy()[0][[3, 0, 1, 2]]
            env.data.qvel[env._rear_dof] = speed
            env.wheel_net_angle, env._step = wheel, count
            env._check_terminations()
            for name in c.TERMINATIONS[1:]:
                term = getattr(self.cfg.terminations, name)
                expected = bool(term.func(isaac, **term.params)[0])
                self.assertEqual(env.termination_flags[name], expected, name)

    def test_actual_reward_terms_match(self):
        env = first_env()
        env.reset()
        data = SimpleNamespace(projected_gravity_b=torch.tensor([[0., 0., -1.]]),
                               root_pos_w=torch.tensor([[.1, .2, 0.]]))
        class Scene(dict):
            env_origins = torch.zeros(1, 3)
        isaac = SimpleNamespace(scene=Scene(robot=SimpleNamespace(data=data)),
                                action_manager=SimpleNamespace(action=torch.tensor([[.3, -.4]]),
                                                               prev_action=torch.tensor([[-.3, .4]])))
        expected = 0.
        for name in ("survival", "drift_gauss", "rate_penalty"):
            term = getattr(self.cfg.rewards, name)
            expected += float(term.func(isaac, **term.params)[0])*term.weight*c.CONTROL_DT
        env.data.qpos[:2] = [.1, .2]
        env.raw_action[:], env._prev_action[:] = [.3, -.4], [-.3, .4]
        env._compute_reward()
        self.assertAlmostEqual(env.last_reward, expected, places=7)


if __name__ == "__main__":
    unittest.main()
