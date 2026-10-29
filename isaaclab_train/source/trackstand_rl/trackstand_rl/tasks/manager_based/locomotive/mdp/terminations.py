"""Copied verbatim from ~/bike_test/CycleRL mdp/dynamic/terminations.py (CycleRL-Trackstand-Cylinder-Paper-v0 deps)."""

import math

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

def bicycle_lean_termination(
    env: ManagerBasedRLEnv,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Terminates when the bicycle leans too far sideways.
    Because the URDF is Y-Up and rotated 90 degrees to fit Isaac Sim,
    we check the Y-axis (index 1) of the projected gravity vector.
    """
    asset = env.scene[asset_cfg.name]

    # projected_gravity_b[:, 1] is -1.0 when perfectly upright.
    # As the bike falls over, this value moves towards 0.0.
    return asset.data.projected_gravity_b[:, 2] > -math.cos(limit_angle)
