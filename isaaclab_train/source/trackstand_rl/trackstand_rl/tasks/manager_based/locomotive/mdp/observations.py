"""Copied verbatim from ~/bike_test/CycleRL mdp/dynamic/observations.py (CycleRL-Trackstand-Cylinder-Paper-v0 deps)."""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

def _as_torch(value):
    """Isaac Lab data buffers may be a ProxyArray with a ``.torch`` view."""
    return value.torch if hasattr(value, "torch") else value


def _apply_dropout(env, cache_key: str, value: torch.Tensor, prob: float) -> torch.Tensor:
    """Hold the previous sample with probability ``prob`` (sensor dropout).

    Models the paper's Table VI "intermittent dropouts" condition: a lost packet
    leaves the consumer reading the last good value, not a zero. Default 0 is a
    no-op so training and deployment are unaffected.
    """
    if prob <= 0.0:
        return value
    last = getattr(env, cache_key, None)
    if last is None or last.shape != value.shape:
        last = value.clone()
    just_reset = env.episode_length_buf <= 1
    held = torch.where(just_reset.unsqueeze(-1), value, last)
    drop = (torch.rand(value.shape[0], 1, device=value.device) < prob) & ~just_reset.unsqueeze(-1)
    out = torch.where(drop, held, value)
    setattr(env, cache_key, out.clone())
    return out


def imu_orientation_and_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    steer_joint_name: str = "Revolute_7",
    lpf_alpha: float = 0.0,
    dropout_prob: float = 0.0,
) -> torch.Tensor:
    """Extracts a 6-D IMU block: gravity_y, wx, wy, wz, forward speed, wrapped steer.

    ``lpf_alpha`` is an EMA on ``ang_vel_*`` only (deploy wx LPF). Default 0 keeps
    other bikes unfiltered. ``gravity_y`` stays raw ``projected_gravity_b[:, 1]``.
    ``dropout_prob`` holds the previous block to emulate lost sensor packets.
    """
    asset = env.scene[asset_cfg.name]

    # Use projected gravity in the body frame to represent lean orientation.
    gravity_y = _as_torch(asset.data.projected_gravity_b)[:, 1:2]

    # Root angular velocity in the local frame: roll, pitch, yaw.
    ang_vel = _as_torch(asset.data.root_ang_vel_b)

    alpha = float(lpf_alpha)
    if alpha > 0.0:
        filt = getattr(env, "_imu_ang_vel_lpf", None)
        if filt is None or filt.shape != ang_vel.shape:
            filt = ang_vel.clone()
        just_reset = env.episode_length_buf <= 1
        filt = torch.where(
            just_reset.unsqueeze(-1),
            ang_vel,
            filt + alpha * (ang_vel - filt),
        )
        env._imu_ang_vel_lpf = filt
        ang_vel = filt

    # Forward velocity in the local frame.
    forward_vel = _as_torch(asset.data.root_lin_vel_b)[:, 0:1]

    # Steering joint position, wrapped to [-π, π] for continuous 360° forks.
    steering_joint_idx = asset.find_joints(steer_joint_name)[0][0]
    joint_pos = _as_torch(asset.data.joint_pos)
    steering_pos = joint_pos[:, steering_joint_idx:steering_joint_idx + 1]
    steering_pos = torch.atan2(torch.sin(steering_pos), torch.cos(steering_pos))

    out = torch.cat([gravity_y, ang_vel, forward_vel, steering_pos], dim=-1)
    return _apply_dropout(env, "_imu_obs_dropout_cache", out, float(dropout_prob))


def command_tracking_error(
    env: ManagerBasedRLEnv,
    command_name: str = "velocity",
    heading_command_name: str = "steering",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    use_vel_error: bool = False,
) -> torch.Tensor:
    """Command features for the policy observation (2-D).

    Default (legacy): raw ``v*`` and yaw-rate command from the steering term.
    NewBike: ``v* - v`` and ``yaw_rate_cmd - wz`` (steer term index 2 minus body yaw rate).
    """
    vel_cmd = env.command_manager.get_command(command_name)
    steer_cmd = env.command_manager.get_command(heading_command_name)
    asset = env.scene[asset_cfg.name]

    if use_vel_error:
        actual_vel = asset.data.root_lin_vel_b[:, 0]
        slot0 = (vel_cmd[:, 0] - actual_vel).unsqueeze(-1)
    else:
        slot0 = vel_cmd[:, 0].unsqueeze(-1)

    # steer_cmd[:, 2] is yaw-rate command when heading_command=True
    yaw_rate_cmd = steer_cmd[:, 2]
    if use_vel_error:
        yaw_rate = asset.data.root_ang_vel_b[:, 2]
        slot1 = (yaw_rate_cmd - yaw_rate).unsqueeze(-1)
    else:
        slot1 = yaw_rate_cmd.unsqueeze(-1)

    return torch.cat([slot0, slot1], dim=-1)
