"""Focused v2 PI timing, state, action-contract and environment regressions."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newbike_mujoco_v2 import contract as c
from newbike_mujoco_v2.control import VelocityIntegral
from newbike_mujoco_v2.env import BikeBalanceEnv, physx_revolute_angle
from newbike_mujoco_v2.policy_contract import validate_policy_contract
from newbike_mujoco_v2.viewer_envs import ViewerEnvGroup

XML = """<mujoco><option gravity="0 0 0"/>
<worldbody><geom name="floor" type="plane" size="10 10 .1"/>
<body name="base_link"><freejoint name="base_free"/><geom type="sphere" size=".1" mass="1"/>
<body pos=".2 0 0"><joint name="steering_joint"/><geom type="sphere" size=".05" mass=".1"/></body>
<body pos="-.2 0 0"><joint name="rearwheel_joint"/><geom type="sphere" size=".05" mass=".1"/></body>
<body pos=".3 0 0"><joint name="frontwheel_joint"/><geom type="sphere" size=".05" mass=".1"/></body>
</body></worldbody><actuator>
<motor name="steering_act" joint="steering_joint"/>
<motor name="drive_act" joint="rearwheel_joint"/>
</actuator></mujoco>"""


def first_env(dr=False, physics_hz=None):
    env = BikeBalanceEnv(scene_xml=XML, seed=42, enable_dr=dr, physics_hz=physics_hz)
    env.OBS_NOISE_STD = (0.,) * 11
    env.reset()
    return env


class VelocityIntegralTests(unittest.TestCase):
    def test_first_tick_primes_without_applying_rectangular_impulse(self):
        integral = VelocityIntegral(3.3)
        self.assertEqual(integral.update(1., .005, .3, 6.28), 0.)
        self.assertEqual(integral.previous_error, 1.)
        self.assertAlmostEqual(integral.update(1., .005, .3, 6.28), .0165)

    def test_target_change_retains_memory(self):
        integral = VelocityIntegral(3.3)
        integral.update(1., .005, .3, 6.28)
        integral.update(1., .005, .3, 6.28)
        integral.update(-1., .005, .3, 6.28)
        self.assertAlmostEqual(integral.torque_nm, .0165)
        integral.update(0., .005, .3, 6.28)
        held = integral.torque_nm
        integral.update(0., .005, .3, 6.28)
        self.assertEqual(integral.torque_nm, held)

    def test_saturation_freezes_worsening_increment_but_advances_error(self):
        integral = VelocityIntegral(3.3)
        integral.update(30., .005, .3, 6.28)
        integral.update(40., .005, .3, 6.28)
        self.assertEqual(integral.torque_nm, 0.)
        self.assertEqual(integral.previous_error, 40.)
        self.assertEqual(integral.limited_updates, 1)

    def test_clamping_corrects_internal_error_and_unwinds(self):
        integral = VelocityIntegral(3.3)
        integral.update(1000., .005, 0., 6.28)
        integral.update(1000., .005, 0., 6.28)
        self.assertEqual(integral.torque_nm, 6.28)
        self.assertAlmostEqual(integral.error_rad, 6.28 / 3.3)
        integral.previous_error = -1.
        integral.update(-1., .005, .3, 6.28)
        self.assertAlmostEqual(integral.torque_nm, 6.28 - .0165)

    def test_reset_and_zero_gain(self):
        integral = VelocityIntegral(3.3)
        for _ in range(5):
            integral.update(1., .005, .3, 6.28)
        integral.reset()
        self.assertEqual((integral.error_rad, integral.torque_nm, integral.previous_error), (0., 0., 0.))
        self.assertFalse(integral.initialized)
        zero = VelocityIntegral(0.)
        for _ in range(5):
            self.assertEqual(zero.update(100., .005, .3, 6.28), 0.)
        self.assertFalse(zero.initialized)

    def test_invalid_gain(self):
        for gain in (-1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                VelocityIntegral(gain)


class V2EnvironmentTests(unittest.TestCase):
    def test_default_and_alternate_frequencies(self):
        self.assertEqual(c.physics_timing(), (1. / 1200., 24))
        for hz in (200, 400, 800, 1200, 1600):
            dt, decimation = c.physics_timing(hz)
            self.assertAlmostEqual(dt * decimation, .02)
            self.assertAlmostEqual(1. / (dt * 200), round(1. / (dt * 200)))
        for hz in (0, -200, 50, 100, 250, 450, np.nan, np.inf):
            with self.assertRaises(ValueError):
                c.physics_timing(hz)

    def test_24_pd_and_four_integral_ticks_per_target(self):
        env = first_env()
        with patch("newbike_mujoco_v2.env.mujoco.mj_step"), patch.object(
                env.integral, "update", wraps=env.integral.update) as update:
            env.step([1., 1. / c.DRIVE_SCALE])
        self.assertEqual(update.call_count, 4)
        self.assertEqual(env._physics_steps, 24)
        self.assertAlmostEqual(env._steer_cmd, .478)
        expected_i = np.repeat(np.arange(4) * .005 * env._drive_cmd * 3.3, 6)
        np.testing.assert_allclose(env.physics_drive_integral_nm, expected_i)
        np.testing.assert_allclose(env.physics_computed_torques[:, 1], .3 * env._drive_cmd + expected_i)

    def test_velocity_scale_and_post_pi_effort_clip(self):
        env = first_env()
        with patch("newbike_mujoco_v2.env.mujoco.mj_step"):
            env.step([0., 2.])
            self.assertEqual(env._drive_cmd, float(np.float32(61.6)))
            self.assertTrue(np.all(env.physics_computed_torques[:, 1] > 6.28))
            self.assertEqual(env.applied_torque[1], 6.28)
            env.step([0., -2.])
            self.assertEqual(env._drive_cmd, float(np.float32(-61.6)))
            self.assertEqual(env.applied_torque[1], -6.28)

    def test_passive_losses_are_separate_from_velocity_p(self):
        env = first_env()
        np.testing.assert_allclose(env.model.dof_damping[env._dofs], [c.STEER_VISCOUS_FRICTION, 0.])
        np.testing.assert_allclose(env.model.dof_frictionloss[env._dofs], [c.STEER_FRICTION, .106])
        np.testing.assert_array_equal(env.kp, [10., 0.])
        np.testing.assert_array_equal(env.kd, [.2, .3])
        env.data.qvel[env._rear_dof] = 1.
        with patch("newbike_mujoco_v2.env.mujoco.mj_step"):
            env.step([0., 1. / c.DRIVE_SCALE])
        self.assertLess(abs(env.applied_torque[1]), 1e-6)

    def test_reset_clears_held_commands_integral_and_rotation_reference(self):
        env = first_env()
        with patch("newbike_mujoco_v2.env.mujoco.mj_step"):
            env.step([1., .01])
        self.assertNotEqual(env.integral.torque_nm, 0.)
        env.data.qpos[env._rear_qadr] = 100.
        env.wheel_net_angle = 100.
        obs = env.reset()
        self.assertFalse(env.integral.initialized)
        self.assertEqual(env.integral.torque_nm, 0.)
        self.assertEqual((env._drive_cmd, env._steer_cmd, env.wheel_net_angle, env._previous_rear_position),
                         (0., 0., 0., 0.))
        np.testing.assert_array_equal(env.data.ctrl, 0.)
        np.testing.assert_array_equal(obs, [0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 1.])

    def test_same_step_wheel_net_crossing_from_position_not_endpoint_velocity(self):
        env = first_env()
        def advance(_model, data):
            data.qpos[env._rear_qadr] += .31
        with patch("newbike_mujoco_v2.env.mujoco.mj_step", side_effect=advance):
            _, done = env.step([0., 0.])
        self.assertAlmostEqual(env.wheel_net_angle, 24 * .31)
        self.assertTrue(done)
        self.assertTrue(env.termination_flags["wheel_net"])
        self.assertEqual(env.data.qvel[env._rear_dof], 0.)

    def test_rear_observation_branch_is_separate_from_net_rotation(self):
        env = first_env()
        env.data.qpos[env._rear_qadr] = 7.1
        self.assertAlmostEqual(env.observe()[6], physx_revolute_angle(7.1), places=6)
        self.assertEqual(env.wheel_net_angle, 0.)
        env.data.qvel[env._rear_dof] = 120.
        self.assertEqual(env.observe()[8], 120.)

    def test_dr_keeps_both_controller_gains_exact_and_resets_independently(self):
        group = ViewerEnvGroup(first_env(True), 3, 42)
        group.reset()
        for _ in range(3):
            for env in group.envs:
                np.testing.assert_array_equal(env.kp, [10., 0.])
                np.testing.assert_array_equal(env.kd, [.2, .3])
                self.assertEqual(env.dr["drive_kv_scale"], 1.)
                env.reset()
        self.assertFalse(np.array_equal(group.envs[0].dr["mass_scale"], group.envs[1].dr["mass_scale"]))
        group.envs[1].integral.torque_nm = .2
        group.envs[0].reset()
        self.assertEqual(group.envs[1].integral.torque_nm, .2)

    def test_invalid_pose_pi_and_counter_detected_and_reward_finite(self):
        for field in ("root_pos", "joint_pos", "quat", "integral", "wheel_net"):
            with self.subTest(field=field):
                env = first_env()
                if field == "root_pos":
                    env.data.qpos[0] = np.nan
                elif field == "joint_pos":
                    env.data.qpos[env._rear_qadr] = np.nan
                elif field == "quat":
                    env.data.qpos[3:7] = 0.
                elif field == "integral":
                    env.integral.error_rad = np.inf
                else:
                    env.wheel_net_angle = np.nan
                self.assertTrue(env._check_terminations())
                self.assertTrue(env.termination_flags["diverged"])
                env._compute_reward()
                self.assertTrue(np.isfinite(env.last_reward))

    def test_disabled_motors_submit_zero_despite_position_or_speed_request(self):
        env = first_env()
        env.motors_enabled = False
        with patch("newbike_mujoco_v2.env.mujoco.mj_step"):
            env.step([1., 1.])
        np.testing.assert_array_equal(env.data.ctrl, 0.)
        np.testing.assert_array_equal(env.applied_torque, 0.)
        env.reset()
        self.assertEqual(env.integral.torque_nm, 0.)


class PolicyAndSamplingTests(unittest.TestCase):
    def test_current_v2_saved_metadata_accepted(self):
        metadata = validate_policy_contract(c.DEFAULT_CHECKPOINT)
        self.assertEqual(metadata["task"], c.TASK)

    def test_v1_same_shape_metadata_rejected(self):
        v1 = c.REPO_ROOT / "isaaclab_train/logs/rsl_rl/trackstand_rl_rsl/2026-10-21_00-41-38/model_75300.pt"
        with self.assertRaisesRegex(ValueError, "not a v2"):
            validate_policy_contract(v1)

    def test_export_sidecar_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            policy = Path(temporary) / "export.onnx"
            policy.write_bytes(b"fixture")
            metadata = dict(task=c.TASK, drive_action="velocity", observations=list(c.OBSERVATIONS),
                            drive_velocity_scale_rad_s=61.6, drive_velocity_p_nm_s_per_rad=.3,
                            drive_integral_gain_nm_per_rad=3.3, drive_integral_rate_hz=200.,
                            drive_effort_limit_nm=6.28)
            sidecar = policy.with_suffix(".contract.json")
            sidecar.write_text(json.dumps(metadata))
            self.assertEqual(validate_policy_contract(policy)["drive_action"], "velocity")
            metadata["policy_sha256"] = "wrong"
            sidecar.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                validate_policy_contract(policy)
            sidecar.unlink()
            with self.assertRaisesRegex(ValueError, "metadata is required"):
                validate_policy_contract(policy)

    def test_balanced_episode_quotas(self):
        spec = importlib.util.spec_from_file_location("v2_eval", ROOT / "scripts/run_deploy_eval_v2.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for total, count in ((1, 64), (64, 64), (65, 64), (128, 64), (7, 3)):
            quotas = module.episode_quotas(total, count)
            self.assertEqual(int(quotas.sum()), total)
            self.assertLessEqual(int(np.ptp(quotas)), 1)


if __name__ == "__main__":
    unittest.main()
