"""V2 numerical containment without changing finite observations or reward values."""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .rewards import balance_drift_gaussian
from .terminations import nonfinite_or_extreme_state


def nonfinite_or_extreme_state_v2(
    env: ManagerBasedRLEnv,
    max_lin_vel: float = 20.0,
    max_ang_vel: float = 50.0,
    max_joint_vel: float = 300.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Retain velocity limits and also reject invalid pose, joints and PI memory.

    Sanitizing the policy observation alone does not repair invalid physical
    state. These checks run before the manager resets the affected environments.
    """
    bad = nonfinite_or_extreme_state(env, max_lin_vel, max_ang_vel, max_joint_vel, asset_cfg)
    data = env.scene[asset_cfg.name].data
    integral = env.action_manager.get_term("drive").integral
    for state in (data.root_pos_w, data.root_quat_w, data.root_link_lin_vel_w, data.root_link_ang_vel_w, data.joint_pos,
                  integral.error_rad, integral.torque_nm, integral.previous_error):
        value = getattr(state, "torch", state)
        bad |= ~torch.isfinite(value).all(dim=-1)
    quat = getattr(data.root_quat_w, "torch", data.root_quat_w)
    bad |= (quat == 0).all(dim=-1)  # A zero quaternion cannot represent an orientation.
    return bad | ~torch.isfinite(env.wheel_net_angle)


def finite_balance_drift_gaussian(
    env: ManagerBasedRLEnv,
    sigma: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Keep normal drift rewards; give zero for invalid state awaiting reset.

    RewardManager computes rewards after terminations but before scene resets.
    A NaN position must not contaminate returns or episodic reward sums there.
    """
    return torch.nan_to_num(balance_drift_gaussian(env, sigma, asset_cfg), nan=0., posinf=0., neginf=0.)
