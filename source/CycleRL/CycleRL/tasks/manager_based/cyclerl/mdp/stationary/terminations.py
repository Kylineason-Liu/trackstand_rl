import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def thubike_fallen_termination(
    env: ManagerBasedRLEnv,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the bike falls over or flips.

    Two checks:
    1. Roll tilt — |gravity_y| > sin(limit_angle) means the bike leaned too far sideways.
    2. Inversion  — gravity_z > 0 means the bike is upside-down (pitch or roll flip past 90°).
    """
    asset = env.scene[asset_cfg.name]
    gravity_y = asset.data.projected_gravity_b[:, 1]
    gravity_z = asset.data.projected_gravity_b[:, 2]

    limit = torch.sin(torch.tensor(limit_angle, device=env.device))
    too_tilted = torch.abs(gravity_y) > limit
    flipped = gravity_z > 0.0  # bike is inverted (more than 90° from upright)

    return too_tilted | flipped


def nonfinite_or_extreme_state(
    env: ManagerBasedRLEnv,
    max_lin_vel: float = 20.0,
    max_ang_vel: float = 50.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reset a diverging bike *before* it can NaN-poison the shared policy.

    A single env whose root state goes non-finite contaminates the batched
    observation, so the PPO update writes NaN into the shared network weights and
    the whole run dies. Firing this termination resets the bad env each step, so
    the blow-up is contained to one env instead of ending training.
    """
    asset = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_lin_vel_b
    ang_vel = asset.data.root_ang_vel_b
    nonfinite = ~torch.isfinite(lin_vel).all(dim=-1) | ~torch.isfinite(ang_vel).all(dim=-1)
    too_fast = torch.norm(lin_vel, dim=-1) > max_lin_vel
    spinning = torch.norm(ang_vel, dim=-1) > max_ang_vel
    return nonfinite | too_fast | spinning
