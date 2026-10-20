"""CPU contract/PPO/export tests; run with the installed Isaac Lab Python.

No simulator rollout is performed here. Observations use the real term with a
mock articulation, while PPO, configs, task registration and exports are real.
"""

import contextlib
import copy
import importlib
import importlib.metadata
import importlib.util
import io
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
TASK_PACKAGE = "source.trackstand_rl.trackstand_rl.tasks.manager_based.trackstand"
HAS_RSL_RL = importlib.util.find_spec("rsl_rl") is not None


@unittest.skipUnless(HAS_RSL_RL, "Run with Isaac Lab Python (RSL-RL >= 5.0)")
class TestTrackstandBeta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gymnasium as gym
        import torch
        from tensordict import TensorDict

        importlib.import_module(f"{TASK_PACKAGE}")
        cls.gym, cls.torch, cls.TensorDict = gym, torch, TensorDict
        cls.beta_module = importlib.import_module(f"{TASK_PACKAGE}.agents.beta_distribution")
        cls.BetaDistribution = cls.beta_module.BetaDistribution
        cls.v0_runner = importlib.import_module(f"{TASK_PACKAGE}.agents.rsl_rl_ppo_cfg").TrackstandRLPPORunnerCfg
        beta_cfg_module = importlib.import_module(f"{TASK_PACKAGE}.agents.rsl_rl_ppo_beta_cfg")
        cls.v1_runner = beta_cfg_module.TrackstandRLBetaPPORunnerCfg
        cls.exporter = importlib.import_module("scripts.deployment.export_rsl_rl_onnx")

    def setUp(self):
        self.torch.manual_seed(42)

    def make_algorithm(self):
        from rsl_rl.algorithms import PPO

        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

        cfg = handle_deprecated_rsl_rl_cfg(
            self.v1_runner(), importlib.metadata.version("rsl-rl-lib")
        ).to_dict()
        cfg["multi_gpu"] = None
        obs = self.TensorDict({"policy": self.torch.randn(32, 11)}, batch_size=[32])
        env = SimpleNamespace(num_envs=32, num_actions=2)
        with contextlib.redirect_stdout(io.StringIO()):
            algorithm = PPO.construct_algorithm(obs, env, cfg, device="cpu")
        return algorithm, obs

    def test_registration_resolves_separate_configs_and_beta_runner(self):
        v0, v1 = (self.gym.spec(f"Trackstand-Balance-v{version}") for version in (0, 1))
        self.assertEqual(v0.entry_point, v1.entry_point)  # Same environment dynamics by design.
        self.assertIsNot(v0.kwargs["env_cfg_entry_point"], v1.kwargs["env_cfg_entry_point"])
        module, name = v1.kwargs["rsl_rl_cfg_entry_point"].split(":")
        self.assertIs(getattr(importlib.import_module(module), name), self.v1_runner)
        self.assertIn("skrl_cfg_entry_point", v0.kwargs)
        self.assertNotIn("skrl_cfg_entry_point", v1.kwargs)  # Do not silently run a Gaussian as v1.

    def test_env_configs_differ_only_in_observations_and_requested_motor_constraints(self):
        for dr in (False, True):
            with self.subTest(dr=dr):
                configs = [
                    self.gym.spec(f"Trackstand-Balance-v{version}").kwargs["env_cfg_entry_point"](
                        domain_randomization=dr
                    )
                    for version in (0, 1)
                ]
                v0, v1 = (cfg.to_dict() for cfg in configs)
                obs0, obs1 = v0.pop("observations"), v1.pop("observations")
                steering0, steering1 = v0["actions"].pop("steering"), v1["actions"].pop("steering")
                self.assertNotEqual(steering0.pop("class_type"), steering1.pop("class_type"))
                self.assertIsNone(steering0.pop("clip"))
                self.assertEqual(steering1.pop("clip"), {"steering_joint": (-1.57, 1.57)})
                self.assertEqual(steering1.pop("max_target_velocity_rad_s"), 23.9)
                self.assertEqual(steering0, steering1)
                drive0 = v0["scene"]["robot"]["actuators"].pop("drive")
                drive1 = v1["scene"]["robot"]["actuators"].pop("drive")
                self.assertNotEqual(drive0.pop("class_type"), drive1.pop("class_type"))
                self.assertIsNone(drive0.pop("effort_limit"))
                self.assertEqual(drive1.pop("effort_limit"), 6.28)
                self.assertIsNone(drive0.pop("velocity_limit"))
                self.assertEqual(drive1.pop("velocity_limit"), 61.6)
                self.assertEqual(drive0.pop("damping"), 0.05)
                self.assertEqual(drive1.pop("damping"), 0.0)
                for field in ("friction", "dynamic_friction"):
                    self.assertIsNone(drive0.pop(field))
                    self.assertEqual(drive1.pop(field), 0.106)
                self.assertEqual(drive0, drive1)
                steering_actuator0 = v0["scene"]["robot"]["actuators"]["steering"]
                steering_actuator1 = v1["scene"]["robot"]["actuators"]["steering"]
                for name, value in (("effort_limit", 5.6), ("velocity_limit", 23.9)):
                    self.assertIsNone(steering_actuator0.pop(name))
                    self.assertEqual(steering_actuator1.pop(name), value)
                self.assertEqual(v0, v1)
                self.assertNotEqual(obs0["policy"]["imu"].pop("func"), obs1["policy"]["imu"].pop("func"))
                noise0 = obs0["policy"]["imu"]["noise"].pop("std")
                noise1 = obs1["policy"]["imu"]["noise"].pop("std")
                self.assertEqual(len(noise0), 13)
                self.assertEqual(noise1, noise0[:5] + noise0[7:])
                self.assertEqual(obs0, obs1)
                self.assertEqual(len(configs[1].observations["policy"].imu.noise.std), 11)
                self.assertEqual(configs[1].sim.dt, 0.0025)
                self.assertEqual(configs[1].decimation, 8)

    def test_v1_observation_group_does_not_inherit_the_v0_layout(self):
        v0_cfg_module = importlib.import_module(f"{TASK_PACKAGE}.trackstand_bike.trackstand_rl_env_cfg")
        v1_cfg_module = importlib.import_module(f"{TASK_PACKAGE}.trackstand_bike.trackstand_rl_env_cfg_v1")
        self.assertFalse(
            issubclass(v1_cfg_module.TrackstandRLObservationsCfgV1, v0_cfg_module.TrackstandRLObservationsCfg)
        )
        observations = importlib.import_module(f"{TASK_PACKAGE}.mdp.stationary.observations")
        with patch(f"{TASK_PACKAGE}.mdp.stationary.observations.balance_imu_state", side_effect=AssertionError):
            v0_cfg = self.gym.spec("Trackstand-Balance-v0").kwargs["env_cfg_entry_point"]()
            v0_cfg.observations["policy"].imu.noise.std = (9.0,) * 13
            v1_cfg = self.gym.spec("Trackstand-Balance-v1").kwargs["env_cfg_entry_point"]()
            term = v1_cfg.observations["policy"].imu
            self.assertIs(term.func, observations.balance_imu_state_v1)
            self.assertEqual(term.func.__module__, observations.__name__)
            self.assertEqual(len(term.noise.std), 11)
            self.assertEqual(term.noise.std[0], 0.001)

    def test_runner_changes_only_actor_distribution_and_copies_mutable_values(self):
        v0, v1 = self.v0_runner().to_dict(), self.v1_runner().to_dict()
        actor0, actor1 = v0.pop("actor"), v1.pop("actor")
        self.assertEqual(v0, v1)
        self.assertEqual(actor0.pop("distribution_cfg")["class_name"], "GaussianDistribution")
        self.assertTrue(actor1.pop("distribution_cfg")["class_name"].endswith(":BetaDistribution"))
        self.assertEqual(actor0, actor1)
        changed = self.v1_runner()
        changed.actor.hidden_dims[0] = 999
        self.assertEqual(self.v0_runner().actor.hidden_dims, [32, 32])
        self.assertEqual(self.v1_runner().actor.hidden_dims, [32, 32])

    def test_older_rsl_rl_cannot_silently_fall_back_to_gaussian(self):
        with patch(f"{TASK_PACKAGE}.agents.rsl_rl_ppo_beta_cfg.version", return_value="4.0.0"):
            with self.assertRaisesRegex(RuntimeError, "require RSL-RL >= 5.0.0"):
                self.v1_runner()

    def test_real_observation_term_has_11_correctly_ordered_channels(self):
        torch = self.torch
        joint_pos = torch.tensor([[99.0, 0.2, -2.5], [98.0, -0.4, 3.5]])
        joint_vel = torch.tensor([[97.0, 1.0, -6.0], [96.0, -2.0, 7.0]])
        data = SimpleNamespace(
            root_quat_w=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2)),
            root_link_ang_vel_w=SimpleNamespace(torch=torch.tensor([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])),
            joint_pos=joint_pos,
            joint_vel=joint_vel,
        )  # No root linear velocity attribute: reading it would fail the test.
        asset = SimpleNamespace(
            data=data,
            find_joints=lambda name: ([{"steering_joint": 1, "rear_wheel_joint": 2}[name]], [name]),
        )
        env = SimpleNamespace(scene={"robot": asset}, device="cpu", num_envs=2)
        term = self.gym.spec("Trackstand-Balance-v1").kwargs["env_cfg_entry_point"]().observations["policy"].imu
        actual = term.func(env, **term.params)
        expected = torch.cat(
            (
                torch.tensor([[0.0, -1.0]] * 2),
                data.root_link_ang_vel_w.torch,
                joint_pos[:, 1:],
                joint_vel[:, 1:],
                joint_pos[:, 1:2].sin(),
                joint_pos[:, 1:2].cos(),
            ),
            dim=-1,
        )
        self.assertEqual(actual.shape, (2, 11))
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual[1, 6].item(), 3.5)  # Native rear angle, not a new +/-pi wrap.
        env._imu_tilt_bias = torch.full((2, 2), 0.002)
        env._imu_gyro_bias = torch.full((2, 3), 0.0002)
        biased = term.func(env, **term.params)
        expected[:, :2] += 0.002
        expected[:, 2:5] += 0.0002
        torch.testing.assert_close(biased, expected)
        noise_module = importlib.import_module(f"{TASK_PACKAGE}.mdp.stationary.noise")
        noisy = noise_module.per_channel_gaussian_noise(biased, term.noise)
        torch.testing.assert_close(noisy[:, 5:], biased[:, 5:])

    def test_v0_has_13_channels_and_v1_matches_all_11_retained_channels(self):
        torch = self.torch
        quat = torch.randn(16, 4)
        quat /= quat.norm(dim=-1, keepdim=True)
        data = SimpleNamespace(
            root_quat_w=SimpleNamespace(torch=quat),
            root_link_ang_vel_w=SimpleNamespace(torch=torch.randn(16, 3)),
            root_link_lin_vel_w=SimpleNamespace(torch=torch.randn(16, 3)),
            joint_pos=torch.randn(16, 3),
            joint_vel=torch.randn(16, 3),
        )
        data.root_link_ang_vel_w.torch[0, 0] = torch.nan
        data.joint_pos[1, 1] = torch.inf
        data.joint_vel[2, 2] = -500.0
        asset = SimpleNamespace(
            data=data,
            find_joints=lambda name: ([{"steering_joint": 1, "rear_wheel_joint": 2}[name]], [name]),
        )
        terms = [
            self.gym.spec(f"Trackstand-Balance-v{version}")
            .kwargs["env_cfg_entry_point"]()
            .observations["policy"].imu
            for version in (0, 1)
        ]
        envs = [SimpleNamespace(scene={"robot": asset}, device="cpu", num_envs=16) for _ in range(2)]
        for biased in (False, True):
            if biased:
                for env in envs:
                    env._imu_tilt_bias = torch.full((16, 2), 0.002)
                    env._imu_gyro_bias = torch.full((16, 3), 0.0002)
            v0, v1 = (term.func(env, **term.params) for term, env in zip(terms, envs, strict=True))
            self.assertEqual(v0.shape, (16, 13))
            self.assertEqual(v1.shape, (16, 11))
            torch.testing.assert_close(v1, torch.cat((v0[:, :5], v0[:, 7:]), dim=-1), rtol=0, atol=0)
            self.assertTrue(torch.isfinite(v1).all())
            self.assertLessEqual(v1.abs().max().item(), 100.0)
        self.assertEqual(len(terms[0].noise.std), 13)
        self.assertIsNot(terms[0].func, terms[1].func)

    def test_sample_and_mean_bounds_including_extreme_logits(self):
        torch = self.torch
        distribution = self.BetaDistribution(2)
        logits = torch.randn(10000, 2, 2) * 20.0
        distribution.update(logits)
        for output in (distribution.sample(), distribution.mean, distribution.deterministic_output(logits)):
            self.assertEqual(output.shape, (10000, 2))
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue((output.abs() <= 1.0).all())
        self.assertTrue(torch.isfinite(distribution.log_prob(distribution.sample())).all())
        extremes = torch.tensor([[[1e6, -1e6], [-1e6, 1e6]]])
        self.assertTrue((distribution.deterministic_output(extremes).abs() <= 1.0).all())

    def test_log_probability_entropy_std_and_kl_match_transformed_beta(self):
        torch = self.torch
        from torch.distributions import AffineTransform, Beta, TransformedDistribution

        distribution = self.BetaDistribution(2)
        logits = torch.randn(32, 2, 2, requires_grad=True)
        distribution.update(logits)
        alpha, beta = distribution.params
        unit = Beta(alpha, beta)
        reference = TransformedDistribution(unit, [AffineTransform(-1.0, 2.0)])
        actions = torch.linspace(-0.9, 0.9, 64).reshape(32, 2)
        torch.testing.assert_close(distribution.log_prob(actions), reference.log_prob(actions).sum(-1))
        torch.testing.assert_close(distribution.entropy, (unit.entropy() + math.log(2.0)).sum(-1))
        torch.testing.assert_close(distribution.std, 2.0 * unit.stddev)
        old = tuple(parameter.detach().clone() for parameter in distribution.params)
        distribution.update(logits + 0.5)
        expected_kl = torch.distributions.kl_divergence(Beta(*old), Beta(*distribution.params)).sum(-1)
        torch.testing.assert_close(distribution.kl_divergence(old, distribution.params), expected_kl)
        loss = -distribution.log_prob(actions).mean() - 0.005 * distribution.entropy.mean()
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)

    def test_endpoint_densities_are_finite_and_outside_support_is_rejected(self):
        torch = self.torch
        distribution = self.BetaDistribution(2)
        distribution.update(torch.zeros(3, 2, 2))
        outputs = torch.tensor([[-1.0, 1.0], [-1.01, 0.0], [0.0, 1.01]])
        log_prob = distribution.log_prob(outputs)
        self.assertTrue(torch.isfinite(log_prob[0]))
        self.assertTrue(torch.isneginf(log_prob[1:]).all())
        with self.assertRaisesRegex(RuntimeError, "update"):
            self.BetaDistribution(2).sample()

    def test_initial_policy_is_centered_and_v0_checkpoints_are_incompatible(self):
        algorithm, obs = self.make_algorithm()
        self.torch.testing.assert_close(algorithm.actor(obs), self.torch.zeros(32, 2), rtol=0, atol=0)
        algorithm.actor(obs, stochastic_output=True)
        self.assertGreater(algorithm.actor.output_std.mean().item(), 0.4)
        checkpoint = copy.deepcopy(algorithm.actor.state_dict())
        self.assertIn("distribution._beta_policy_version", checkpoint)
        restored, _ = self.make_algorithm()
        restored.actor.load_state_dict(checkpoint, strict=True)
        self.torch.testing.assert_close(restored.actor(obs), algorithm.actor(obs), rtol=0, atol=0)
        del checkpoint["distribution._beta_policy_version"]
        with self.assertRaises(RuntimeError):
            restored.actor.load_state_dict(checkpoint, strict=True)

    def test_real_ppo_updates_are_finite_and_keep_actions_bounded(self):
        torch = self.torch
        algorithm, obs = self.make_algorithm()
        initial = copy.deepcopy(algorithm.actor.state_dict())
        for _ in range(3):
            with torch.inference_mode():
                for _ in range(algorithm.storage.num_transitions_per_env):
                    actions = algorithm.act(obs)
                    self.assertTrue((actions.abs() <= 1.0).all())
                    new_obs = self.TensorDict({"policy": torch.randn(32, 11)}, batch_size=[32])
                    algorithm.process_env_step(new_obs, -actions.square().sum(-1), torch.zeros(32), {})
                    obs = new_obs
                algorithm.compute_returns(obs)
            losses = algorithm.update()
            self.assertTrue(all(math.isfinite(value) for value in losses.values()), losses)
            self.assertTrue(all(torch.isfinite(p).all() for p in algorithm.actor.parameters()))
            self.assertTrue((algorithm.actor(obs).abs() <= 1.0).all())
        self.assertTrue(
            any(not torch.equal(initial[key], value) for key, value in algorithm.actor.state_dict().items())
        )

    def test_native_and_standalone_deterministic_export_match(self):
        torch = self.torch
        algorithm, _ = self.make_algorithm()
        # Nonzero/asymmetric heads ensure a missing mean mapping cannot pass.
        with torch.no_grad():
            algorithm.actor.mlp[-2].weight.normal_(0.0, 0.2)
            algorithm.actor.mlp[-2].bias.copy_(torch.tensor([0.3, -0.7, 0.1, 0.8]))
        inputs = torch.randn(256, 11)
        observations = self.TensorDict({"policy": inputs}, batch_size=[256])
        expected = algorithm.actor(observations)
        bare_mlp = torch.nn.Sequential(*list(copy.deepcopy(algorithm.actor.mlp))[:-1])
        standalone = self.exporter.DeploymentPolicy(bare_mlp, beta_policy=True)
        torch.testing.assert_close(standalone(inputs), expected, rtol=0, atol=0)
        torch.testing.assert_close(algorithm.actor.as_onnx(verbose=False)(inputs), expected, rtol=0, atol=0)
        scripted = torch.jit.script(algorithm.actor.as_jit())
        torch.testing.assert_close(scripted(inputs), expected, rtol=0, atol=0)
        gaussian = self.exporter.DeploymentPolicy(bare_mlp)
        torch.testing.assert_close(gaussian(inputs), bare_mlp(inputs), rtol=0, atol=0)

    @unittest.skipUnless(importlib.util.find_spec("onnx"), "ONNX is not installed")
    def test_onnx_includes_bounded_mean_with_dynamic_batch(self):
        import numpy as np
        import onnx
        from onnx.reference import ReferenceEvaluator

        torch = self.torch
        algorithm, _ = self.make_algorithm()
        with torch.no_grad():
            algorithm.actor.mlp[-2].weight.normal_(0.0, 0.2)
        inputs = torch.randn(7, 11)
        bare_mlp = torch.nn.Sequential(*list(copy.deepcopy(algorithm.actor.mlp))[:-1])
        standalone = self.exporter.DeploymentPolicy(bare_mlp, beta_policy=True).eval()
        buffer = io.BytesIO()
        torch.onnx.export(
            standalone,
            (inputs[:1],),
            buffer,
            opset_version=18,
            dynamo=False,
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        )
        model = onnx.load_model_from_string(buffer.getvalue())
        onnx.checker.check_model(model)
        actual = ReferenceEvaluator(model).run(None, {"observation": inputs.numpy()})[0]
        self.assertEqual(actual.shape, (7, 2))
        self.assertTrue((np.abs(actual) <= 1.0).all())
        with torch.no_grad():
            np.testing.assert_allclose(actual, standalone(inputs).numpy(), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
