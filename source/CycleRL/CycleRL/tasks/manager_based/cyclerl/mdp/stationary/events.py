from typing import Optional

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from ..events import randomize_ground_plane_material

__all__ = [
    "randomize_ground_plane_material",
    "randomize_initial_roll",
    "reset_imu_sensor_model",
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


def reset_imu_sensor_model(
    env: ManagerBasedRLEnv,
    env_ids: Optional[torch.Tensor],
    gyro_bias_std: float = 0.03,   # rad/s: per-episode gyro (angular-velocity) bias
    tilt_bias_std: float = 0.02,   # per-episode gravity-component (tilt) bias, ~1 deg
    min_delay_steps: int = 1,      # step_dt = 20 ms  ->  1..3 steps = 20..60 ms latency
    max_delay_steps: int = 3,
):
    """Resample the per-episode IMU sensor model read by ``balance_imu_state``.

    Writes buffers onto ``env`` (created lazily on first call):
      - ``_imu_tilt_bias`` (n, 2): constant offset on [gravity_y, gravity_z]
      - ``_imu_gyro_bias`` (n, 3): constant offset on body angular velocity
      - ``_obs_delay``     (n,):   integer observation delay in steps
      - ``_obs_max_delay``:        ring-buffer depth (constant across the run)
      - ``_obs_needs_prime`` (n,): flag so a reset env doesn't read stale history

    Biases are per-episode *constants* (the observation term's GaussianNoiseCfg
    adds white noise on top). Latency models the real bike's measured 20-60 ms
    sensor/control-loop delay.
    """
    num = env.num_envs
    device = env.device
    if not hasattr(env, "_imu_gyro_bias"):
        env._imu_gyro_bias = torch.zeros(num, 3, device=device)
        env._imu_tilt_bias = torch.zeros(num, 2, device=device)
        env._obs_delay = torch.full((num,), min_delay_steps, dtype=torch.long, device=device)
        env._obs_max_delay = int(max_delay_steps)
        env._obs_needs_prime = torch.zeros(num, dtype=torch.bool, device=device)
        env._obs_buffer = None  # lazily sized by balance_imu_state once obs_dim is known

    # Track the largest delay ever requested, so a curriculum that ramps
    # max_delay_steps up later isn't capped by an early small value that would
    # otherwise have frozen the ring-buffer depth.
    env._obs_max_delay = max(env._obs_max_delay, int(max_delay_steps))

    if env_ids is None:
        env_ids = torch.arange(num, device=device)
    n = len(env_ids)

    env._imu_gyro_bias[env_ids] = torch.randn(n, 3, device=device) * gyro_bias_std
    env._imu_tilt_bias[env_ids] = torch.randn(n, 2, device=device) * tilt_bias_std
    env._obs_delay[env_ids] = torch.randint(
        min_delay_steps, max_delay_steps + 1, (n,), device=device
    )
    env._obs_needs_prime[env_ids] = True
