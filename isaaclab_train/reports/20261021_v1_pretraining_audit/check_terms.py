"""CPU edge-case checks of the actual registered v1 reward/termination functions."""

import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gymnasium as gym
import torch

PACKAGE = "source.trackstand_rl.trackstand_rl.tasks.manager_based.trackstand"
importlib.import_module(PACKAGE)
terms = importlib.import_module(PACKAGE+".mdp.stationary.terminations")
rewards = importlib.import_module(PACKAGE+".mdp.stationary.rewards")
cfg = gym.spec("Trackstand-Balance-v1").kwargs["env_cfg_entry_point"]()

angles = torch.deg2rad(torch.tensor([0.0, 29.0, 31.0, 0.0, 0.0]))
gravity = torch.stack((torch.zeros_like(angles), -angles.sin(), -angles.cos()), dim=-1)
gravity[3] = torch.tensor([0.8660254, 0.0, -0.5])  # Pure60-degree pitch, not lateral roll.
gravity[4] = torch.tensor([0.0, 0.0, 1.0])
data = SimpleNamespace(projected_gravity_b=gravity)
asset = SimpleNamespace(data=data)
env = SimpleNamespace(scene={"robot": asset}, device="cpu")
fallen = cfg.terminations.fallen.func(env, **cfg.terminations.fallen.params)
assert fallen.tolist() == [False, False, True, False, True]

env.action_manager = SimpleNamespace(action=torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
                                    prev_action=torch.zeros((3, 2)))
rate = cfg.rewards.rate_penalty.func(env, **cfg.rewards.rate_penalty.params)
torch.testing.assert_close(rate, -torch.sqrt(torch.tensor([1/1.57**2, 1/6.28**2, 0.0])+1e-8))

n = 7
data.root_lin_vel_b = torch.zeros((n, 3))
data.root_ang_vel_b = torch.zeros((n, 3))
data.joint_vel = torch.zeros((n, 3))
data.joint_pos = torch.zeros((n, 3))
data.root_pos_w = torch.zeros((n, 3))
data.root_lin_vel_b[1, 0] = 20.1
data.root_ang_vel_b[2, 0] = 50.1
data.joint_vel[3, 0] = 300.1
data.joint_vel[4, 0] = torch.nan
data.root_pos_w[5, 0] = torch.nan
data.joint_pos[6, 0] = torch.nan
env.scene = {"robot": asset}
env.scene = type("Scene", (dict,), {})(env.scene)
env.scene.env_origins = torch.zeros((n, 3))
diverged = cfg.terminations.diverged.func(env, **cfg.terminations.diverged.params)
assert diverged.tolist() == [False, True, True, True, True, False, False]
drift_reward = cfg.rewards.drift_gauss.func(env, **cfg.rewards.drift_gauss.params)
assert not torch.isfinite(drift_reward[5])
drift_done = cfg.terminations.drifted.func(env, **cfg.terminations.drifted.params)
assert not drift_done[5]

yaw = torch.tensor([3.13, -3.13, 0.6, -0.6])
data.root_quat_w = SimpleNamespace(torch=torch.stack((torch.zeros_like(yaw), torch.zeros_like(yaw),
                                                     (yaw*0.5).sin(), (yaw*0.5).cos()), dim=-1))
env.spawn_yaw = torch.tensor([-3.13, 3.13, 0.0, 0.0])
yaw_done = cfg.terminations.yaw_deviation.func(env, **cfg.terminations.yaw_deviation.params)
assert yaw_done.tolist() == [False, False, True, True]
env.wheel_net_angle = torch.tensor([-7.3, -7.2, 7.2, 7.3])
wheel_done = cfg.terminations.wheel_net.func(env, **cfg.terminations.wheel_net.params)
assert wheel_done.tolist() == [True, False, False, True]

result = {
    "checks_completed": True,
    "fallen_for_upright_roll29_roll31_pitch60_inverted": fallen.tolist(),
    "action_rate_for_unit_steer_change_unit_drive_change_and_zero": rate.tolist(),
    "weighted_dt_rate_contributions_at_50hz": (rate*cfg.rewards.rate_penalty.weight*0.02).tolist(),
    "diverged_for_normal_linear20_1_angular50_1_joint300_1_nan_joint_speed_nan_root_position_nan_joint_position": diverged.tolist(),
    "nan_root_position_produces_nan_drift_reward_without_diverged_or_drifted_termination": True,
    "yaw_wraparound_cases": yaw_done.tolist(), "wheel_net_boundary_cases": wheel_done.tolist(),
    "scope": "Synthetic edge cases using real configured functions; not failures observed in a physical rollout",
}
Path(__file__).with_name("term_checks.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
