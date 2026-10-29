"""Copied verbatim from ~/bike_test/CycleRL mdp/dynamic/rewards.py (CycleRL-Trackstand-Cylinder-Paper-v0 deps)."""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

def survival_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def velocity_tracking_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    alpha: float = 0.25,
    squared: bool = False,
    tolerance: float = 0.25,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    actual_vel = asset.data.root_lin_vel_b[:, 0]  # Forward velocity (X)
    cmd_vel = env.command_manager.get_command("velocity")[:, 0]

    abs_error = torch.abs(actual_vel - cmd_vel)
    error_with_deadzone = torch.clamp(abs_error - tolerance, min=0.0)
    if squared:
        return torch.exp(-alpha * error_with_deadzone * error_with_deadzone)
    return torch.exp(-alpha * error_with_deadzone)


def heading_tracking_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "steering",
    beta: float = 0.1,
) -> torch.Tensor:
    """CycleRL paper (2603.15013): r_steer = exp(-β |ψ - ψ_desired|)."""
    asset = env.scene[asset_cfg.name]
    heading = asset.data.heading_w
    heading = heading.torch if hasattr(heading, "torch") else heading
    term = env.command_manager._terms[command_name]
    heading_target = term.heading_target
    error = torch.abs(torch.atan2(torch.sin(heading_target - heading), torch.cos(heading_target - heading)))
    return torch.exp(-beta * error)


def weighted_action_magnitude_penalty(
    env: ManagerBasedRLEnv, 
    weights: list[float]
) -> torch.Tensor:
    """Penalize action magnitude, but apply different multipliers to different motors."""
    w = torch.tensor(weights, device=env.device)
    # Multiply the actions by our custom weights before calculating the penalty
    return -torch.sum(w * torch.abs(env.action_manager.action), dim=-1)


def weighted_action_rate_penalty(
    env: ManagerBasedRLEnv, 
    weights: list[float]
) -> torch.Tensor:
    """Penalize rapid twitching, but apply different multipliers to different motors."""
    w = torch.tensor(weights, device=env.device)
    action_diff = env.action_manager.action - env.action_manager.prev_action
    # Multiply the twitching by our custom weights before calculating the penalty
    return -torch.norm(w * action_diff, p=2, dim=-1)
