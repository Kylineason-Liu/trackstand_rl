"""Headless v1 runtime, checkpoint, timing, reset and DR regression tests."""

import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newbike_mujoco import contract as c
from newbike_mujoco.env import BikeBalanceEnv, physx_revolute_angle
from newbike_mujoco.model import build_env_from_mjcf
from newbike_mujoco.policy import DeployedPolicy
from newbike_mujoco.reporting import summarize
from newbike_mujoco.viewer_envs import ViewerEnvGroup
from test_multi_viewer import first_env


class V1ContractTests(unittest.TestCase):
    def test_nominal_spawn_noise_shape_no_finite_cap(self):
        env = first_env()
        env.OBS_NOISE_STD = (0.,)*11
        obs = env.reset()
        np.testing.assert_array_equal(obs, [0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 1.])
        np.testing.assert_array_equal(env.data.qpos[:3], [0., 0., -.016])
        env.data.qvel[env._rear_dof] = 120.
        self.assertEqual(env.observe()[8], 120.)
        env.data.qvel[env._rear_dof] = np.inf
        self.assertEqual(env.observe()[8], 0.)

    def test_rear_native_branch_matches_new_gpu_probe(self):
        path = ROOT / "reports/20261022_v1_alignment/isaac_angle_probe.json"
        result = json.loads(path.read_text())
        # Deliberately extreme100-rad manual state injection is not a certified
        # native-coordinate contract (the solver returned0); keep that anomaly.
        requested = np.array(result["requested_position_rad"][:-1])
        measured = np.array(result["after_step_position_rad"][:-1])
        np.testing.assert_allclose(physx_revolute_angle(requested), measured, atol=2e-6)
        self.assertEqual(physx_revolute_angle(3.5), 3.5)
        env = first_env()
        env.reset()
        env.data.qpos[env._rear_qadr] = 7.1
        self.assertAlmostEqual(env.observe()[6], 7.1-4*np.pi, places=6)

    def test_eight_explicit_pd_steps_and_post_pd_clip(self):
        env = first_env()
        env.reset()
        env.data.qpos[2] = 3.  # No ground contacts.
        env.data.qvel[env._steer_dof] = -20.
        actual_step, captured = mujoco.mj_step, []

        def check(model, data):
            expected = 10*(env._steer_cmd-data.qpos[env._steer_qadr])-.2*data.qvel[env._steer_dof]
            self.assertAlmostEqual(data.ctrl[env._dr_steer_act], np.clip(expected, -5.6, 5.6))
            self.assertAlmostEqual(data.ctrl[env._dr_drive_act], np.float32(.25)*np.float32(6.28))
            captured.append(data.ctrl[env._acts].copy())
            actual_step(model, data)

        with patch("newbike_mujoco.env.mujoco.mj_step", side_effect=check):
            env.step(np.array([1., .25]))
        self.assertEqual(len(captured), 8)
        self.assertAlmostEqual(env._steer_cmd, .478)
        np.testing.assert_allclose(env.physics_torques, captured, atol=1e-12)
        self.assertAlmostEqual(env.data.time, .02)
        self.assertTrue(np.any(np.abs(np.array(captured)[:, 0]) == 5.6))

    def test_target_clip_slew_and_reset_not_measured_position_clamp(self):
        env = first_env()
        env.reset()
        with patch("newbike_mujoco.env.mujoco.mj_step"):
            expected = [.478, .956, 1.434, 1.57, 1.57]
            for value in expected:
                env.step([2., 2.])
                self.assertAlmostEqual(env._steer_cmd, value, places=6)
                self.assertAlmostEqual(env.data.ctrl[env._dr_drive_act], 6.28)
            env.step([-2., -2.])
            self.assertAlmostEqual(env._steer_cmd, 1.57-.478, places=6)
        env.wheel_net_angle, env._yaw0 = 4., .5
        env.reset()
        self.assertEqual(env._steer_cmd, 0.)
        self.assertEqual(env.wheel_net_angle, 0.)
        self.assertEqual(env._yaw0, 0.)
        np.testing.assert_array_equal(env.raw_action, 0.)
        np.testing.assert_array_equal(env._prev_action, 0.)

    def test_passive_losses_not_software_torque_damping(self):
        env = first_env()
        env.reset()
        np.testing.assert_allclose(env.model.dof_damping[env._dofs], [c.STEER_VISCOUS_FRICTION, 0.])
        np.testing.assert_allclose(env.model.dof_frictionloss[env._dofs], [c.STEER_FRICTION, .106])
        env.data.qvel[env._rear_dof] = 40.
        with patch("newbike_mujoco.env.mujoco.mj_step"):
            env.step([0., .25])
        self.assertAlmostEqual(env.applied_torque[1], np.float32(.25)*np.float32(6.28))
        self.assertEqual(env.kd[1], 0.)

    def test_disabled_motor_telemetry_matches_actual_zero_output(self):
        env = first_env()
        env.reset()
        env.motors_enabled = False
        env.step([1., 1.])
        np.testing.assert_array_equal(env.applied_torque, [0., 0.])
        np.testing.assert_array_equal(env.physics_torques, np.zeros((8, 2)))

    def test_speed_constraint_modes_are_explicit_not_hidden_target_limits(self):
        for mode, expected in (("project", 61.6), ("none", 100.)):
            env = first_env()
            env.velocity_limit_mode = mode
            env.reset()
            env.data.qvel[env._rear_dof] = 100.
            with patch("newbike_mujoco.env.mujoco.mj_step"):
                env.step([0., 0.])
            self.assertAlmostEqual(env.data.qvel[env._rear_dof], expected)

    def test_all_six_and_overlapping_flags(self):
        env = first_env()
        env.reset()
        self.assertEqual(env.enabled_terms, set(c.TERMINATIONS))
        env._step = 1500
        env.data.qpos[0] = 1.
        env.data.qpos[3:7] = [np.cos(.4), 0., 0., np.sin(.4)]
        env.data.qvel[env._front_dof] = 301.
        env.wheel_net_angle = 8.
        self.assertTrue(env._check_terminations())
        for name in ("time_out", "diverged", "drifted", "yaw_deviation", "wheel_net"):
            self.assertTrue(env.termination_flags[name])
        self.assertFalse(env.termination_flags["fallen"])
        self.assertTrue(env.stats.timeout)

    def test_fall_and_yaw_exact_thresholds(self):
        env = first_env()
        for angle, expected in ((c.FALL_ANGLE-1e-5, False), (c.FALL_ANGLE+1e-5, True)):
            env.reset()
            env.data.qpos[3:7] = [np.cos(angle/2), np.sin(angle/2), 0., 0.]
            env._check_terminations()
            self.assertEqual(env.termination_flags["fallen"], expected)
        env.reset()
        env.data.qpos[3:7] = [0., 0., 1., 0.]
        env._check_terminations()
        self.assertTrue(env.termination_flags["fallen"])

    def test_wheel_net_lag_and_timeout_reset(self):
        env = first_env()
        env.reset()
        env.enabled_terms = {"wheel_net", "time_out"}
        env.wheel_net_angle = 7.19
        env.data.qvel[env._rear_dof] = 2.
        with patch("newbike_mujoco.env.mujoco.mj_step"):
            self.assertFalse(env.step([0., 0.])[1])
            self.assertTrue(env.step([0., 0.])[1])
        env.reset()
        env._step = env.MAX_STEPS-1
        with patch("newbike_mujoco.env.mujoco.mj_step"):
            self.assertTrue(env.step([0., 0.])[1])
        self.assertTrue(env.stats.timeout)
        env.reset()
        self.assertEqual(env.wheel_net_angle, 0.)
        self.assertEqual(env.stats.length, 0)

    def test_auto_reset_path_does_not_sample_terminal_noise(self):
        env = first_env()
        env.reset()
        env._step = env.MAX_STEPS-1
        with patch.object(env, "observe", wraps=env.observe) as observe:
            _, done = env.step([0., 0.], observe_terminal=False)
            self.assertTrue(done)
            observe.assert_not_called()
            env.reset()
            observe.assert_called_once()

    def test_reward_equal_raw_weighting_and_dt(self):
        env = first_env()
        for action in ([1., 0.], [0., 1.], [-1., 0.], [0., -1.]):
            env.reset()
            with patch("newbike_mujoco.env.mujoco.mj_step"):
                env.step(action)
            self.assertAlmostEqual(env.last_reward, (7.-.1)*.02, places=8)
            self.assertAlmostEqual(env.reward_terms["rate_penalty"], -.1)
            with patch("newbike_mujoco.env.mujoco.mj_step"):
                env.step(action)
            self.assertEqual(env.reward_terms["rate_penalty"], 0.)

    def test_shared_material_bank_independent_reset_noncompounding(self):
        group = ViewerEnvGroup(first_env(True), 4, 42)
        group.reset()
        first = group.envs[0]
        for env in group.envs:
            np.testing.assert_array_equal(env._dr_friction_buckets, first._dr_friction_buckets)
            for _ in range(5):
                env.reset()
                np.testing.assert_allclose(env.model.body_mass[env._dr_body_ids]/env._dr_mass0, env.dr["mass_scale"])
                self.assertTrue(np.all(env.dr["mass_scale"] >= .8))
                self.assertTrue(np.all(env.dr["mass_scale"] <= 1.2))
                np.testing.assert_array_equal(env.kp[1:], [0.])
                np.testing.assert_array_equal(env.kd[1:], [0.])
                self.assertTrue(np.all(env.dr["tire_dynamic_friction"] <= env.dr["tire_friction"]))

    def test_cad_com_offsets_are_expressed_in_matching_local_frames(self):
        env, _ = build_env_from_mjcf(c.DEFAULT_MJCF, 42, True)
        env.reset()
        local = env.model.body_ipos[env._dr_body_ids]-env._dr_ipos0
        # Undo the CAD-frame conversion to recover the ONE Isaac-local offset.
        recovered = env._com_isaac_to_mj.transpose(0, 2, 1) @ local[..., None]
        np.testing.assert_allclose(recovered[..., 0], np.tile(env.dr["com_offset"], (len(local), 1)), atol=1e-12)
        self.assertEqual(len(local), 4)  # No massless model_root draw.

    def test_dr_reset_zeroes_root_com_velocity_not_free_origin_velocity(self):
        from newbike_mujoco.env import _rot_matrix
        env, _ = build_env_from_mjcf(c.DEFAULT_MJCF, 42, True)
        env.reset()
        rotation = _rot_matrix(env._base_quat())
        offset = env.data.xipos[env._root_body_id]-env.data.qpos[:3]
        omega_w = rotation @ env.data.qvel[3:6]
        velocity_com = env.data.qvel[:3]+np.cross(omega_w, offset)
        np.testing.assert_allclose(velocity_com, 0., atol=1e-12)
        self.assertGreater(np.linalg.norm(env.data.qvel[:3]), 0.)

    def test_group_does_not_clone_randomized_episode_as_nominal_model(self):
        first = first_env(True)
        nominal = first._dr_mass0.copy()
        first.reset()
        group = ViewerEnvGroup(first, 2, 42)
        np.testing.assert_array_equal(group.envs[1]._dr_mass0, nominal)
        group.reset()
        second = group.envs[1]
        np.testing.assert_allclose(second.model.body_mass[second._dr_body_ids], nominal*second.dr["mass_scale"])

    def test_read_only_verified_models_and_self_collision_disabled(self):
        path = c.DEFAULT_MJCF
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        env, _ = build_env_from_mjcf(path, 42)
        env.reset()
        self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(env.model.nexclude, 10)  # Five nonworld bodies, every pair.
        self.assertEqual(env.model.opt.timestep, .0025)
        self.assertEqual(env.model.geom("floor").friction[0], 1.)
        floor = env.model.geom("floor").id
        contacts = [contact for contact in env.data.contact if floor in contact.geom]
        self.assertTrue(contacts)
        for contact in contacts:
            self.assertAlmostEqual(contact.friction[0], .65)

    def test_ground_one_tire_priority_preserves_the_intended_pair_dynamics(self):
        for dr in (False, True):
            current, _ = build_env_from_mjcf(c.DEFAULT_MJCF, 42, dr)
            previous, _ = build_env_from_mjcf(c.DEFAULT_MJCF, 42, dr)
            previous.model.geom("floor").friction[0] = 0.
            previous.model.geom_priority[:] = 0
            np.testing.assert_array_equal(current.reset(), previous.reset())
            for _ in range(100):
                one, done_one = current.step([-.4, .01], observe_terminal=False)
                two, done_two = previous.step([-.4, .01], observe_terminal=False)
                self.assertEqual(done_one, done_two)
                np.testing.assert_array_equal(current.data.qpos, previous.data.qpos)
                np.testing.assert_array_equal(current.data.qvel, previous.data.qvel)
                np.testing.assert_array_equal(one, two)
                if done_one:
                    np.testing.assert_array_equal(current.reset(), previous.reset())

    @unittest.skipUnless(c.DEFAULT_CHECKPOINT.is_file(), "Local model75300 checkpoint not installed")
    def test_real_beta_checkpoint_and_incompatible_v0(self):
        policy = DeployedPolicy(pt_path=c.DEFAULT_CHECKPOINT)
        obs = np.random.default_rng(42).normal(size=(1000, 11)).astype(np.float32)
        out = policy.act_batch(obs)
        self.assertEqual(out.shape, (1000, 2))
        self.assertTrue(np.all(np.abs(out) <= 1.))
        with torch.no_grad():
            raw = policy._torch_net(torch.from_numpy(obs)).reshape(-1, 2, 2)
            alpha = torch.nn.functional.softplus(raw[:, 0])+1.
            beta = torch.nn.functional.softplus(raw[:, 1])+1.
            expected = (2*(alpha/(alpha+beta))-1).numpy()
        np.testing.assert_array_equal(out, expected)
        with self.assertRaisesRegex(ValueError, "Gaussian/v0"):
            DeployedPolicy(pt_path=ROOT / "policies/model_500.pt")
        with self.assertRaises(ValueError):
            policy.act_batch(np.zeros((2, 13)))

    def test_exported_onnx_matches_beta_checkpoint(self):
        import importlib.util
        path = ROOT / "reports/20261022_v1_alignment/model_75300_v1.onnx"
        if not path.exists() or not c.DEFAULT_CHECKPOINT.exists() or importlib.util.find_spec("onnxruntime") is None:
            self.skipTest("ONNX export or onnxruntime unavailable")
        torch_policy = DeployedPolicy(pt_path=c.DEFAULT_CHECKPOINT)
        ort_policy = DeployedPolicy(onnx_path=path)
        obs = np.random.default_rng(42).normal(size=(1024, 11)).astype(np.float32)
        np.testing.assert_allclose(torch_policy.act_batch(obs), ort_policy.act_batch(obs), atol=5e-7, rtol=2e-6)

    def test_summary_uses_global_max_and_overlap_not_mean_maxima(self):
        first, second = first_env(), first_env()
        first.stats.length, second.stats.length = 10, 20
        first.stats.roll_max, second.stats.roll_max = .1, .5
        first.stats.terminations = dict(fallen=True, time_out=True)
        summary = summarize([(10, None, first.stats), (20, None, second.stats)], 2)
        self.assertAlmostEqual(summary["max_roll_deg"], np.degrees(.5))
        self.assertEqual(summary["termination_counts"]["fallen"], 1)
        self.assertEqual(summary["termination_counts"]["time_out"], 1)


if __name__ == "__main__":
    unittest.main()
