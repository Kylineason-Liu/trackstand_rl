import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def _apply_obs_latency(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Return a per-env delayed copy of ``obs`` via a ring buffer on ``env``.

    Delay (steps) per env is ``env._obs_delay`` (set by ``reset_imu_turnon_residuals``);
    index 0 of the buffer is the newest sample, so buffer[d] is ``d`` steps old.
    Freshly reset envs are primed with the current obs so they never read stale
    history from the previous episode.
    """
    n_env, obs_dim = obs.shape
    buf_len = env._obs_max_delay + 1
    if getattr(env, "_obs_buffer", None) is None or env._obs_buffer.shape != (buf_len, n_env, obs_dim):
        env._obs_buffer = obs.unsqueeze(0).expand(buf_len, n_env, obs_dim).clone()

    env._obs_buffer = torch.roll(env._obs_buffer, shifts=1, dims=0)
    env._obs_buffer[0] = obs

    # prime just-reset envs so buffer[d] is a valid (not stale) sample
    prime = getattr(env, "_obs_needs_prime", None)
    if prime is not None and bool(prime.any()):
        env._obs_buffer[:, prime] = obs[prime].unsqueeze(0)
        env._obs_needs_prime[prime] = False

    delay = env._obs_delay.clamp(0, buf_len - 1)
    idx = delay.view(1, n_env, 1).expand(1, n_env, obs_dim)
    return torch.gather(env._obs_buffer, 0, idx).squeeze(0)


def _accumulate_motor_speeds(env: ManagerBasedRLEnv, steer_v: torch.Tensor, drive_v: torch.Tensor) -> None:
    """Sum / max the |motor speeds| per env for later terminal logging.

    Called every step from balance_imu_state; consumed and reset per episode by
    ``mdp.stationary.curriculums.log_motor_speeds``. Pure diagnostics — no effect
    on the policy, reward, or dynamics.
    """
    if not hasattr(env, "_motor_speed_count"):
        z = torch.zeros(env.num_envs, device=env.device)
        env._drive_speed_sum = z.clone()
        env._drive_speed_max = z.clone()
        env._steer_speed_sum = z.clone()
        env._steer_speed_max = z.clone()
        env._motor_speed_count = z.clone()
    dv, sv = drive_v.abs(), steer_v.abs()
    env._drive_speed_sum += dv
    env._steer_speed_sum += sv
    env._drive_speed_max = torch.maximum(env._drive_speed_max, dv)
    env._steer_speed_max = torch.maximum(env._steer_speed_max, sv)
    env._motor_speed_count += 1.0


def balance_imu_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    steer_joint_name: str = "Revolute_9",
    drive_joint_name: str = "Revolute_8",
    obs_clip: float = 100.0,
) -> torch.Tensor:
    """IMU-rich observation for stationary balancing.

    Returns (num_envs, 13):
        [gravity_y, gravity_z,           # projected gravity (roll-lean, upright)
         ang_vel_x, ang_vel_y, ang_vel_z, # body-frame angular velocity
         lin_vel_x, lin_vel_y,           # body-frame linear velocity (drift)
         steering_pos, drive_pos,        # joint positions
         steering_vel, drive_vel,        # joint velocities
         sin_steer, cos_steer]           # encoded steering angle
    (The four joint channels are currently UN-commented for the 13-dim x noise
     2x2 experiment (09090950). Re-comment them to restore the 9-dim obs that
     2242/2246/2248/09090944 trained with — remember to shorten the noise tuple
     to 9 entries at the same time.)
    """
    asset = env.scene[asset_cfg.name]

    # Derive gravity/velocities from the RAW root quaternion instead of the
    # kernel-cached derived fields. The raw pose buffer is updated synchronously
    # by the reset-write kernels, but _reset_pose() does NOT invalidate the
    # projected_gravity_b / *_b caches (audit finding) — so in the post-reset
    # window those return the dying pose's values, and the FIRST observation of
    # a new episode would feed the policy stale gravity for one step.
    q = asset.data.root_quat_w.torch  # (N, 4) xyzw, raw buffer — fresh in all windows
    g_vec = torch.zeros(q.shape[0], 3, device=env.device)
    g_vec[:, 2] = -1.0  # normalized gravity direction in world frame
    g_b = math_utils.quat_apply_inverse(q, g_vec)
    gravity_y = g_b[:, 1:2]  # roll lean
    gravity_z = g_b[:, 2:3]  # upright

    ang_vel = math_utils.quat_apply_inverse(q, asset.data.root_link_ang_vel_w.torch)
    lin_vel_xy = math_utils.quat_apply_inverse(q, asset.data.root_link_lin_vel_w.torch)[:, :2]

    steer_idx = asset.find_joints(steer_joint_name)[0][0]
    drive_idx = asset.find_joints(drive_joint_name)[0][0]

    steer_pos = asset.data.joint_pos[:, steer_idx : steer_idx + 1]
    drive_pos = asset.data.joint_pos[:, drive_idx : drive_idx + 1]
    steer_vel = asset.data.joint_vel[:, steer_idx : steer_idx + 1]
    drive_vel = asset.data.joint_vel[:, drive_idx : drive_idx + 1]

    # diagnostics: accumulate TRUE motor speeds each step for per-episode
    # terminal logging (read + reset at reset by curriculums.log_motor_speeds)
    _accumulate_motor_speeds(env, steer_vel[:, 0], drive_vel[:, 0])

    sin_steer = torch.sin(steer_pos)
    cos_steer = torch.cos(steer_pos)

    obs = torch.cat(
        [
            gravity_y, gravity_z,
            ang_vel,
            lin_vel_xy,
            steer_pos, drive_pos,
            steer_vel, drive_vel,
            sin_steer, cos_steer,
        ],
        dim=-1,
    )

    # --- IMU sensor-model domain randomization (per-episode, set by
    #     mdp.stationary.events.reset_imu_turnon_residuals) ---
    if hasattr(env, "_imu_tilt_bias"):
        obs[:, 0:2] = obs[:, 0:2] + env._imu_tilt_bias   # [gravity_y, gravity_z] accel/mount bias
        obs[:, 2:5] = obs[:, 2:5] + env._imu_gyro_bias   # body angular velocity gyro bias
    # if hasattr(env, "_obs_delay"):                      # latency REMOVED 2026-09-08:
    #     obs = _apply_obs_latency(env, obs)              # the event no longer creates _obs_delay;
    #                                                     # _apply_obs_latency is dormant below

    # Last line of defence: a stray non-finite value from one diverging env must
    # never reach the shared network, or the PPO update NaNs every policy weight.
    # The clamp covers the *finite but huge* case, which killed the torque runs: an
    # unbounded rear wheel put |drive_vel| = 2420 rad/s in this vector, and with
    # obs_normalization=False that single feature drove the critic's value loss to
    # 1e4 -> NaN -> NaN policy weights. Every real signal here is well inside +-obs_clip
    # (gravity <= 1, motor speeds <= ~60 rad/s), so this only ever bites an env that is
    # already diverging (and which the `diverged` termination resets the same step).
    obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(obs, -obs_clip, obs_clip)


def balance_command_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Provide the current velocity command to the policy.

    Returns (num_envs, 1): [target_vel_x]
    """
    vel_cmd = env.command_manager.get_command("velocity")
    return vel_cmd[:, 0].unsqueeze(-1)
