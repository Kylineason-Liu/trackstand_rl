"""Physics/PD rate must not change the 50-Hz policy clock or leak across bikes."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from test_multi_viewer import XML, VIEWER
from newbike_mujoco import contract as c
from newbike_mujoco.env import BikeBalanceEnv
from newbike_mujoco.model import build_env_from_mjcf
from newbike_mujoco.viewer_envs import ViewerEnvGroup

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("frequency_eval", ROOT / "scripts/run_deploy_eval.py")
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


class PhysicsFrequencyTests(unittest.TestCase):
    def test_resolve_valid_and_invalid_rates(self):
        self.assertEqual(c.physics_timing(), (.0025, 8))
        for hz in (50, 100, 200, 400, 800, 1600):
            with self.subTest(hz=hz):
                dt, substeps = c.physics_timing(hz)
                self.assertAlmostEqual(dt, 1/hz)
                self.assertEqual(substeps, hz//50)
                self.assertAlmostEqual(dt*substeps, .02)
        for hz in (0, -400, 25, 60, 799, 800.5, float("nan"), float("inf")):
            with self.subTest(hz=hz), self.assertRaisesRegex(ValueError, "multiple of 50"):
                c.physics_timing(hz)

    def test_pd_substeps_telemetry_slew_and_policy_period(self):
        advance = mujoco.mj_step
        for hz in (400, 800, 1600):
            with self.subTest(hz=hz):
                env = BikeBalanceEnv(scene_xml=XML, physics_hz=hz, seed=42)
                env.reset()
                env.data.qpos[2] = 3.  # Isolate the actuator loop from contacts.
                torques = []

                def record(model, data):
                    torque = 10*(env._steer_cmd-data.qpos[env._steer_qadr])-.2*data.qvel[env._steer_dof]
                    self.assertAlmostEqual(data.ctrl[env._dr_steer_act], np.clip(torque, -5.6, 5.6))
                    torques.append(data.ctrl[env._acts].copy())
                    advance(model, data)

                with patch("newbike_mujoco.env.mujoco.mj_step", side_effect=record), \
                        patch.object(env, "observe", wraps=env.observe) as observe:
                    env.step([1., .25])
                    observe.assert_called_once()
                self.assertEqual(len(torques), hz//50)
                self.assertEqual(env.physics_torques.shape, (hz//50, 2))
                np.testing.assert_array_equal(env.physics_torques, torques)
                self.assertAlmostEqual(env.model.opt.timestep, 1/hz)
                self.assertAlmostEqual(env.data.time, .02)
                self.assertAlmostEqual(env._steer_cmd, .478)
                self.assertEqual(env.stats.length, 1)
                self.assertEqual(env.MAX_STEPS, 1500)
        self.assertEqual((c.PHYSICS_DT, c.DECIMATION), (.0025, 8))

    def test_groups_and_mixed_rates_are_independent_of_global_defaults(self):
        slow = BikeBalanceEnv(scene_xml=XML, physics_hz=400, seed=42)
        fast = BikeBalanceEnv(scene_xml=XML, physics_hz=800, seed=42)
        with patch.object(c, "PHYSICS_DT", .000625), patch.object(c, "DECIMATION", 32):
            group = ViewerEnvGroup(fast, 4, 42)
            group.reset()
            for env in [slow, *group.envs]:
                env.reset()
                env.step([0., 0.])
                expected = 8 if env is slow else 16
                self.assertEqual(env.decimation, expected)
                self.assertEqual(env.physics_torques.shape, (expected, 2))
                self.assertAlmostEqual(env.data.time, .02)
                self.assertAlmostEqual(env.physics_dt, .02/expected)

    def test_actual_model_loader_uses_selected_rate(self):
        env, _ = build_env_from_mjcf(c.DEFAULT_MJCF, 42, physics_hz=800)
        env.reset()
        env.step([0., 0.])
        self.assertEqual(env.model.opt.timestep, .00125)
        self.assertEqual(env.decimation, 16)
        self.assertAlmostEqual(env.data.time, .02)

    def test_both_clis_validate_rate_before_loading_policy_or_model(self):
        for module in (VIEWER, EVAL):
            with self.subTest(module=module.__name__), \
                    patch.object(sys, "argv", ["entry", "--physics_hz", "799"]), \
                    patch.object(module, "build_env_from_mjcf") as build, \
                    patch.object(module, "DeployedPolicy") as policy, \
                    contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    module.main()
                self.assertEqual(error.exception.code, 2)
                self.assertIn("multiple of 50", stderr.getvalue())
                build.assert_not_called()
                policy.assert_not_called()

    def test_both_clis_forward_selected_rate_to_model_loader(self):
        for module in (VIEWER, EVAL):
            with self.subTest(module=module.__name__), \
                    patch.object(sys, "argv", ["entry", "--physics_hz", "800"]), \
                    patch.object(module, "DeployedPolicy"), \
                    patch.object(module, "build_env_from_mjcf", side_effect=RuntimeError("parsed")) as build:
                with self.assertRaisesRegex(RuntimeError, "parsed"):
                    module.main()
                self.assertEqual(build.call_args.kwargs["physics_hz"], 800.)

    def test_headless_cli_saves_selected_timing_and_all_substep_torques(self):
        def build(*args, **kwargs):
            return BikeBalanceEnv(scene_xml=XML, physics_hz=kwargs["physics_hz"], seed=42), True

        with tempfile.TemporaryDirectory() as folder, \
                patch.object(sys, "argv", ["eval", "--physics_hz", "800", "--num_envs", "2",
                                           "--num_steps", "2", "--steps", "1", "--output_dir", folder]), \
                patch.object(EVAL, "build_env_from_mjcf", side_effect=build), \
                patch.object(EVAL, "DeployedPolicy") as policy, \
                contextlib.redirect_stdout(io.StringIO()):
            policy.return_value.act_batch.return_value = np.zeros((2, 2), dtype=np.float32)
            policy.return_value.path = str(Path(__file__))
            policy.return_value.backend = "test"
            EVAL.main()
            path = Path(folder)
            metadata = json.loads(next(path.glob("*_metadata.json")).read_text())
            summary = json.loads(next(path.glob("*_summary.json")).read_text())
            for record in (metadata, summary):
                self.assertEqual(record["physics_hz"], 800)
                self.assertEqual(record["pd_hz"], 800)
                self.assertEqual(record["policy_hz"], 50)
                self.assertEqual(record["decimation"], 16)
            self.assertEqual(metadata["arguments"]["physics_hz"], 800)
            with np.load(next(path.glob("*_evaluation.npz"))) as z:
                self.assertEqual(z["physics_motor_torque_nm"].shape, (2, 2, 16, 2))
                self.assertEqual(z["physics_dt_s"], .00125)
                self.assertEqual(z["control_dt_s"], .02)
                self.assertEqual(z["decimation"], 16)


if __name__ == "__main__":
    unittest.main()
