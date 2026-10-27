"""Focused CPU probes for v2 defaults, observation math and audit findings."""

import ast
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

TRAIN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRAIN))
sys.path.insert(0, str(TRAIN / "tests"))
PACKAGE = "source.trackstand_rl.trackstand_rl.tasks.manager_based.trackstand"
importlib.import_module(PACKAGE)
import gymnasium as gym

ROOT = Path(__file__).resolve().parent
cfg_type = gym.spec("Trackstand-Balance-v2").kwargs["env_cfg_entry_point"]
cfg = cfg_type(domain_randomization=True)
term = cfg.observations["policy"].imu

# Literal xyzw pure rolls prove the signs independently of Isaac quaternion helpers.
roll = torch.tensor([0., .3, -.3, 0.])
quat = torch.zeros(4, 4)
quat[:, 0], quat[:, 3] = (roll*.5).sin(), (roll*.5).cos()
position = torch.tensor([[0., 150., 175.], [0., .2, 3.5], [0., -.2, -3.5], [0., 0., 0.]])
velocity = torch.zeros_like(position)
data = SimpleNamespace(root_quat_w=SimpleNamespace(torch=quat),
                       root_link_ang_vel_w=SimpleNamespace(torch=torch.zeros(4, 3)),
                       joint_pos=position, joint_vel=velocity)
asset = SimpleNamespace(
    data=data, find_joints=lambda name: ([{"steering_joint": 1, "rear_wheel_joint": 2}[name]], [name]))
env = SimpleNamespace(num_envs=4, device="cpu", scene={"robot": asset})
actual = term.func(env, **term.params)
expected_gravity = torch.stack((-roll.sin(), -roll.cos()), -1)
torch.testing.assert_close(actual[:, :2], expected_gravity)
assert actual[0, 5] == 150. and actual[0, 6] == 175.
assert actual[1, 6] == 3.5 and actual[2, 6] == -3.5
data.joint_pos[3, 1:] = torch.tensor([float("nan"), float("inf")])
sanitized = term.func(env, **term.params)
assert torch.isfinite(sanitized).all() and not sanitized[3, 5:7].count_nonzero()

terminations = importlib.import_module(f"{PACKAGE}.mdp.stationary.terminations")
rewards = importlib.import_module(f"{PACKAGE}.mdp.stationary.rewards")
unsafe = SimpleNamespace(
    root_pos_w=torch.tensor([[float("nan"), 0., 0.]]),
    root_quat_w=SimpleNamespace(torch=torch.tensor([[0., 0., 0., 1.]])),
    root_lin_vel_b=torch.zeros(1, 3), root_ang_vel_b=torch.zeros(1, 3),
    joint_pos=torch.zeros(1, 3), joint_vel=torch.zeros(1, 3),
)
bad_env = SimpleNamespace(device="cpu", scene={"robot": SimpleNamespace(data=unsafe)})
bad_env.scene = type("Scene", (dict,), {})(bad_env.scene)
bad_env.scene.env_origins = torch.zeros(1, 3)
diverged = bool(terminations.nonfinite_or_extreme_state(bad_env)[0])
nan_reward = bool(torch.isnan(rewards.balance_drift_gaussian(bad_env))[0])
assert not diverged and nan_reward  # A coverage gap, not an observed live simulator failure.

script = ast.parse((TRAIN / "scripts/rsl_rl/train.py").read_text())
weight_loop = next(node for node in ast.walk(script) if isinstance(node, ast.For)
                   and isinstance(node.iter, ast.Tuple)
                   and "args_cli.arp_w" in ast.unparse(node.iter))
arp_target = next(pair.elts[1].value for pair in weight_loop.iter.elts
                  if ast.unparse(pair.elts[0]) == "args_cli.arp_w")
assert arp_target == "action_rate" and not hasattr(cfg.rewards, arp_target)
assert cfg.rewards.rate_penalty.weight == .1

beta_module = importlib.import_module(f"{PACKAGE}.agents.beta_distribution")
torch.manual_seed(42)
dist = beta_module.BetaDistribution(2)
dist.update(torch.zeros(100000, 2, 2))
sample = dist.sample()
assert torch.isfinite(sample).all() and sample.abs().max() <= 1.
exploration = {
    "raw_std_analytic": float(dist.std[0, 1]),
    "rear_target_std_rad_s": float(dist.std[0, 1]*61.6),
    "fraction_rear_samples_saturating_at_zero_speed": float((.2*61.6*sample[:, 1]).abs().gt(6.28).float().mean()),
    "sample_count": len(sample),
}

# These CLI tests do not launch an application or require a GPU.
module = importlib.import_module("test_domain_randomization")
cli_result = unittest.TextTestRunner(verbosity=2).run(
    unittest.defaultTestLoader.loadTestsFromTestCase(module.TestDomainRandomizationCLI))
assert cli_result.wasSuccessful()
runner_path = gym.spec("Trackstand-Balance-v2").kwargs["rsl_rl_cfg_entry_point"]
module_name, class_name = runner_path.split(":")
runner = getattr(importlib.import_module(module_name), class_name)()
assert not runner.actor.obs_normalization and runner.clip_actions is None and not runner.resume
result = {
    "passed": True,
    "pure_roll_projected_gravity_signs_verified": True,
    "no_finite_100_clamp_and_no_added_rear_pi_wrap": True,
    "nan_inf_observations_sanitized": True,
    "default_num_envs": cfg.scene.num_envs, "default_max_iterations": runner.max_iterations,
    "default_reward_weights": {name: getattr(cfg.rewards, name).weight
                               for name in ("survival", "drift_gauss", "rate_penalty")},
    "policy_hidden_dims": runner.actor.hidden_dims, "critic_hidden_dims": runner.critic.hidden_dims,
    "clip_actions": runner.clip_actions, "obs_normalization": runner.actor.obs_normalization,
    "is_finite_horizon": cfg.is_finite_horizon, "episode_length_s": cfg.episode_length_s,
    "experiment_name": runner.experiment_name, "cli_tests_passed": cli_result.testsRun,
    "initial_exploration": exploration,
    "confirmed_findings": {
        "arp_w_targets_missing_term": arp_target,
        "nonfinite_pose_probe_diverged": diverged,
        "nonfinite_root_position_probe_drift_reward_nan": nan_reward,
    },
}
(ROOT / "cpu_summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
