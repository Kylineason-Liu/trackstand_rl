from typing import Optional

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from ..events import randomize_ground_plane_material

__all__ = [
    "randomize_ground_plane_material",
    "randomize_initial_roll",
    "reset_imu_turnon_residuals",
]


def randomize_initial_roll(
    env: ManagerBasedRLEnv,
    env_ids: Optional[torch.Tensor],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    roll_range: tuple[float, float] = (-0.1, 0.1),
):
    """Apply a small random roll perturbation to the root state on reset."""
    import isaaclab.utils.math as math_utils

    asset = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = slice(None)
        num = env.num_envs
    else:
        num = len(env_ids)

    roll = math_utils.sample_uniform(
        roll_range[0], roll_range[1], (num,), device=env.device
    )

    # Isaac Sim uses (x, y, z, w) quaternion order.
    # Pure roll rotation around X: x = sin(roll/2), y=0, z=0, w = cos(roll/2)
    half_roll = roll * 0.5
    quat = torch.zeros(num, 4, device=env.device)
    quat[:, 0] = torch.sin(half_roll)  # x
    quat[:, 3] = torch.cos(half_roll)  # w

    asset.write_root_pose_to_sim(
        torch.cat(
            [
                asset.data.root_pos_w[env_ids, :3],  # keep position
                quat,
            ],
            dim=-1,
        ),
        env_ids=env_ids,
    )


def reset_imu_turnon_residuals(
    env: ManagerBasedRLEnv,
    env_ids: Optional[torch.Tensor],
    gyro_res_bias_std: float = 0.03,   # rad/s: per-episode gyro turn-on bias RESIDUAL (post-calibration)
    accel_res_bias_std: float = 0.02,  # per-episode ACCELEROMETER turn-on residual on the gravity channels, ~1 deg
):
    """Resample the per-episode IMU sensor model read by ``balance_imu_state``.

    Models what remains AFTER calibration: the turn-on/initial bias residual
    (the spread of per-boot mean offsets the firmware's calibration leaves
    behind). The stds should be set to the MEASURED post-calibration residual
    spread, not the raw uncalibrated bias.

    (The sensor/loop LATENCY part of this event was removed on 2026-09-08: the
    min_delay_steps/max_delay_steps params and the delay ring-buffer sampling
    are gone; observations.py keeps _apply_obs_latency dormant if the delay is
    ever re-added.)

    Writes buffers onto ``env`` (created lazily on first call):
      - ``_imu_tilt_bias`` (n, 2): constant offset on [gravity_y, gravity_z]
      - ``_imu_gyro_bias`` (n, 3): constant offset on body angular velocity

    Biases are per-episode *constants* (the observation term's GaussianNoiseCfg
    adds white noise on top).
    """
    num = env.num_envs
    device = env.device
    if not hasattr(env, "_imu_gyro_bias"):
        env._imu_gyro_bias = torch.zeros(num, 3, device=device)
        env._imu_tilt_bias = torch.zeros(num, 2, device=device)

    if env_ids is None:
        env_ids = torch.arange(num, device=device)
    n = len(env_ids)

    env._imu_gyro_bias[env_ids] = torch.randn(n, 3, device=device) * gyro_res_bias_std
    env._imu_tilt_bias[env_ids] = torch.randn(n, 2, device=device) * accel_res_bias_std
