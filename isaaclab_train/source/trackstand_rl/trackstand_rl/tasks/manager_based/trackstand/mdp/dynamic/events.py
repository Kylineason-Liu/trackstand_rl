import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedEnv

from ..events import randomize_ground_plane_material


def _as_torch(value):
    """Isaac Lab data buffers may be a ProxyArray with a ``.torch`` view."""
    return value.torch if hasattr(value, "torch") else value


def wrap_root_at_terrain_border(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    inset: float = 0.4,
):
    """Teleport the root across the flat apron before it hits the perimeter wall.

    Generator terrains add a flat z=0 apron around the curriculum grid, then a
    thin collision wall on the apron's outer edge. This term wraps world XY
    ``inset`` metres inside that wall so PhysX never applies a contact impulse.
    Orientation and spatial velocity are written back unchanged.

    Plane terrains (no generator) are a no-op. Env origins are left unchanged
    so reset still returns to the original cell.
    """
    terrain = env.scene["terrain"]
    gen = getattr(terrain.cfg, "terrain_generator", None)
    if gen is None:
        return

    apron = float(getattr(gen, "apron_width", 0.0))
    half_x = 0.5 * float(gen.num_rows) * float(gen.size[0]) + apron - float(inset)
    half_y = 0.5 * float(gen.num_cols) * float(gen.size[1]) + apron - float(inset)
    if half_x <= 0.0 or half_y <= 0.0:
        return

    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    else:
        env_ids = env_ids.to(device=asset.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    pos = _as_torch(asset.data.root_pos_w)[env_ids].clone()
    needs_wrap = (pos[:, 0].abs() > half_x) | (pos[:, 1].abs() > half_y)
    if not torch.any(needs_wrap):
        return

    ids = env_ids[needs_wrap]
    pos_w = pos[needs_wrap]
    pos_w[:, 0] = torch.remainder(pos_w[:, 0] + half_x, 2.0 * half_x) - half_x
    pos_w[:, 1] = torch.remainder(pos_w[:, 1] + half_y, 2.0 * half_y) - half_y

    quat = _as_torch(asset.data.root_quat_w)[ids].clone()
    vel = _as_torch(asset.data.root_vel_w)[ids].clone()
    asset.write_root_pose_to_sim_index(
        root_pose=torch.cat([pos_w, quat], dim=-1),
        env_ids=ids,
    )
    # Re-apply the current spatial velocity so PhysX does not treat the
    # teleport as a collision or zero the root twist.
    asset.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=ids)


def _kickstand_body_ids(asset_cfg: SceneEntityCfg):
    """Single body for the kickstand wrench (the first resolved name, base_link).

    Extra names in ``body_names`` are ignored so the disturbance is never
    silently applied to the fork or wheels.
    """
    ids = asset_cfg.body_ids
    if isinstance(ids, list) and len(ids) > 0:
        return [ids[0]]
    if torch.is_tensor(ids) and ids.numel() > 0:
        return ids.reshape(-1)[:1]
    raise ValueError(
        "Kickstand wrench needs a resolved body id. "
        "Set body_names=['base_link'] on the event SceneEntityCfg."
    )


def _kickstand_buffers(env: ManagerBasedEnv, num_envs: int, device) -> None:
    """Per-env kickstand lift state (not part of the policy I/O)."""
    if getattr(env, "_ks_state", None) is not None and env._ks_state.shape[0] == num_envs:
        return
    env._ks_state = torch.zeros(num_envs, dtype=torch.int32, device=device)  # 0 idle, 1 lift, 2 done
    env._ks_t0 = torch.zeros(num_envs, device=device)
    env._ks_sign = torch.ones(num_envs, device=device)
    env._ks_mag = torch.zeros(num_envs, device=device)
    env._ks_delay = torch.zeros(num_envs, device=device)
    env._ks_was_slow = torch.ones(num_envs, dtype=torch.bool, device=device)
    env._ks_duration = torch.full((num_envs,), 0.3, device=device)


def reset_kickstand_lift_schedule(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["base_link"]),
    deploy_speed: float = 1.2,
    retract_speed: float = 1.5,
    delay_range_s: tuple[float, float] = (0.2, 1.0),
    torque_range: tuple[float, float] = (0.05, 0.12),
):
    """Reset kickstand lift buffers. Spawned-fast envs get a delayed pulse."""
    del deploy_speed  # hysteresis lower bound is used in the interval term
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    env_ids = env_ids.to(device=asset.device, dtype=torch.long)
    _kickstand_buffers(env, env.scene.num_envs, asset.device)

    n = len(env_ids)
    env._ks_state[env_ids] = 0
    env._ks_t0[env_ids] = 0.0
    env._ks_sign[env_ids] = torch.where(
        torch.rand(n, device=asset.device) > 0.5,
        torch.ones(n, device=asset.device),
        -torch.ones(n, device=asset.device),
    )
    env._ks_mag[env_ids] = math_utils.sample_uniform(
        torque_range[0], torque_range[1], (n,), asset.device
    )
    env._ks_duration[env_ids] = math_utils.sample_uniform(0.2, 0.4, (n,), asset.device)

    speed = asset.data.root_lin_vel_b[env_ids, 0]
    already_fast = speed >= retract_speed
    env._ks_was_slow[env_ids] = True
    env._ks_delay[env_ids] = 0.0
    if already_fast.any():
        fast_ids = env_ids[already_fast]
        env._ks_delay[fast_ids] = math_utils.sample_uniform(
            delay_range_s[0], delay_range_s[1], (len(fast_ids),), asset.device
        )

    body_ids = _kickstand_body_ids(asset_cfg)
    zeros = torch.zeros((n, 1, 3), device=asset.device)
    # Local base_link frame (is_global=False): fy lateral, fz up, τx roll.
    asset.permanent_wrench_composer.set_forces_and_torques_index(
        forces=zeros,
        torques=zeros,
        body_ids=body_ids,
        env_ids=env_ids.to(dtype=torch.int32),
        is_global=False,
    )


def apply_kickstand_lift_wrench(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["base_link"]),
    deploy_speed: float = 1.2,
    retract_speed: float = 1.5,
    lift_duration_s: float = 0.3,
    torque_range: tuple[float, float] = (0.05, 0.12),
    swing_rad: float = 0.4363,
):
    """Apply a short local-frame wrench on ``base_link`` when speed crosses retract.

    Forces/torques are expressed in the body frame (``is_global=False``): ``fy``
    lateral, ``fz`` up, ``τx`` roll. Only ``base_link`` is loaded; extra names
    on ``asset_cfg`` are ignored. Not exposed in observations or actions.
    """
    del torque_range, swing_rad  # magnitude / sign sampled at reset
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    env_ids = env_ids.to(device=asset.device, dtype=torch.long)
    _kickstand_buffers(env, env.scene.num_envs, asset.device)

    speed = asset.data.root_lin_vel_b[env_ids, 0]
    # step_dt * count; float32 integers are exact well past 64 s / 0.02 s episodes.
    episode_t = env.episode_length_buf[env_ids].to(dtype=torch.float32) * float(env.step_dt)

    was_slow = env._ks_was_slow[env_ids]
    state = env._ks_state[env_ids]
    delay = env._ks_delay[env_ids]

    # Re-arm after slowing below the deploy threshold (same hysteresis as ROS).
    slow_now = speed <= deploy_speed
    env._ks_was_slow[env_ids] = was_slow | slow_now
    state = torch.where(slow_now, torch.zeros_like(state), state)

    start = (state == 0) & (speed >= retract_speed) & env._ks_was_slow[env_ids] & (
        episode_t >= delay
    )
    state = torch.where(start, torch.ones_like(state), state)
    env._ks_t0[env_ids] = torch.where(start, episode_t, env._ks_t0[env_ids])
    env._ks_was_slow[env_ids] = torch.where(start, torch.zeros_like(was_slow), env._ks_was_slow[env_ids])

    duration = env._ks_duration[env_ids]
    duration = torch.where(duration <= 1e-6, torch.full_like(duration, lift_duration_s), duration)
    progress = torch.clamp((episode_t - env._ks_t0[env_ids]) / duration, 0.0, 1.0)
    lifting = state == 1
    done = lifting & (progress >= 1.0 - 1e-6)
    state = torch.where(done, torch.full_like(state, 2), state)
    env._ks_state[env_ids] = state

    n = len(env_ids)
    body_ids = _kickstand_body_ids(asset_cfg)
    forces = torch.zeros((n, 1, 3), device=asset.device)
    torques = torch.zeros((n, 1, 3), device=asset.device)

    # Half-sine envelope; extra α spike in the first ~30% of the swing.
    envelope = torch.sin(torch.pi * progress)
    spike = torch.clamp(1.0 - progress / 0.3, min=0.0)
    active = lifting & ~done
    roll_tau = env._ks_sign[env_ids] * env._ks_mag[env_ids] * (envelope + 0.5 * spike)
    fz = 0.15 * 9.81 * 0.05 * envelope  # ~0.15 kg CoM lift, ~5 cm
    fy = 0.02 * env._ks_sign[env_ids] * envelope
    torques[:, 0, 0] = torch.where(active, roll_tau, torch.zeros_like(roll_tau))
    forces[:, 0, 1] = torch.where(active, fy, torch.zeros_like(fy))
    forces[:, 0, 2] = torch.where(active, fz, torch.zeros_like(fz))

    # Local base_link frame: buffers are (N, 1, 3) for that one body only.
    asset.permanent_wrench_composer.set_forces_and_torques_index(
        forces=forces,
        torques=torques,
        body_ids=body_ids,
        env_ids=env_ids.to(dtype=torch.int32),
        is_global=False,
    )


def _eval_impulse_buffers(env: ManagerBasedEnv, num_envs: int, device) -> None:
    """Per-env lateral impulse state used only during evaluation."""
    if getattr(env, "_eval_imp_sign", None) is not None and env._eval_imp_sign.shape[0] == num_envs:
        return
    env._eval_imp_sign = torch.ones(num_envs, device=device)
    env._eval_imp_delay = torch.zeros(num_envs, device=device)
    env._eval_imp_force = torch.zeros(num_envs, device=device)
    env._eval_imp_duration = torch.full((num_envs,), 0.1, device=device)
    env._eval_imp_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._eval_imp_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
    env._eval_imp_fired = torch.zeros(num_envs, dtype=torch.bool, device=device)


def reset_eval_impulse_schedule(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["base_link"]),
    delay_s: float = 2.0,
    force_n: float = 10.0,
    duration_s: float = 0.1,
):
    """Arm a one-shot lateral impulse on ``base_link`` after ``delay_s``."""
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    env_ids = env_ids.to(device=asset.device, dtype=torch.long)
    _eval_impulse_buffers(env, env.scene.num_envs, asset.device)

    n = len(env_ids)
    env._eval_imp_sign[env_ids] = torch.where(
        torch.rand(n, device=asset.device) > 0.5,
        torch.ones(n, device=asset.device),
        -torch.ones(n, device=asset.device),
    )
    env._eval_imp_delay[env_ids] = delay_s
    env._eval_imp_force[env_ids] = force_n
    env._eval_imp_duration[env_ids] = duration_s
    env._eval_imp_active[env_ids] = False
    env._eval_imp_done[env_ids] = False
    env._eval_imp_fired[env_ids] = False

    body_ids = _kickstand_body_ids(asset_cfg)
    zeros = torch.zeros((n, 1, 3), device=asset.device)
    asset.permanent_wrench_composer.set_forces_and_torques_index(
        forces=zeros,
        torques=zeros,
        body_ids=body_ids,
        env_ids=env_ids.to(dtype=torch.int32),
        is_global=False,
    )


def apply_eval_lateral_impulse(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["base_link"]),
    delay_s: float = 2.0,
    force_n: float = 10.0,
    duration_s: float = 0.1,
):
    """Apply a body-frame lateral force pulse: default 10 N for 0.1 s.

    Used for the evaluation kick / impulse-rejection test. The wrench is
    expressed in the local ``base_link`` frame (``fy`` = lateral).
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    env_ids = env_ids.to(device=asset.device, dtype=torch.long)
    _eval_impulse_buffers(env, env.scene.num_envs, asset.device)

    episode_t = env.episode_length_buf[env_ids].to(dtype=torch.float32) * float(env.step_dt)
    delay = torch.where(env._eval_imp_delay[env_ids] > 0, env._eval_imp_delay[env_ids], torch.full_like(episode_t, delay_s))
    duration = torch.where(
        env._eval_imp_duration[env_ids] > 0,
        env._eval_imp_duration[env_ids],
        torch.full_like(episode_t, duration_s),
    )
    force = torch.where(
        env._eval_imp_force[env_ids] > 0,
        env._eval_imp_force[env_ids],
        torch.full_like(episode_t, force_n),
    )

    done = env._eval_imp_done[env_ids]
    active = (~done) & (episode_t >= delay) & (episode_t < delay + duration)
    just_finished = (~done) & (episode_t >= delay + duration) & env._eval_imp_fired[env_ids]
    env._eval_imp_fired[env_ids] = env._eval_imp_fired[env_ids] | active
    env._eval_imp_active[env_ids] = active
    env._eval_imp_done[env_ids] = done | just_finished | ((episode_t >= delay + duration) & env._eval_imp_fired[env_ids])

    n = len(env_ids)
    body_ids = _kickstand_body_ids(asset_cfg)
    forces = torch.zeros((n, 1, 3), device=asset.device)
    torques = torch.zeros((n, 1, 3), device=asset.device)
    fy = env._eval_imp_sign[env_ids] * force
    forces[:, 0, 1] = torch.where(active, fy, torch.zeros_like(fy))

    asset.permanent_wrench_composer.set_forces_and_torques_index(
        forces=forces,
        torques=torques,
        body_ids=body_ids,
        env_ids=env_ids.to(dtype=torch.int32),
        is_global=False,
    )


def _body_mass_tensor(asset) -> torch.Tensor:
    """``(num_envs, num_bodies)`` masses, including after reset randomization."""
    mass = asset.data.body_mass
    return mass.torch if hasattr(mass, "torch") else mass


def _random_force_buffers(env: ManagerBasedEnv, num_envs: int, device) -> None:
    """Per-env random external-force pulse state."""
    if getattr(env, "_extf_wait", None) is not None and env._extf_wait.shape[0] == num_envs:
        return
    env._extf_wait = torch.zeros(num_envs, device=device)
    env._extf_t0 = torch.zeros(num_envs, device=device)
    env._extf_dur = torch.zeros(num_envs, device=device)
    env._extf_force = torch.zeros(num_envs, 3, device=device)
    env._extf_torque = torch.zeros(num_envs, 3, device=device)
    env._extf_active = torch.zeros(num_envs, dtype=torch.bool, device=device)


def _sample_weight_scaled_wrench(
    total_mass: torch.Tensor,
    force_weight_frac_range: tuple[float, float],
    torque_weight_frac_range: tuple[float, float],
    elevation_range: tuple[float, float],
    torque_lever_m: float,
    gravity: float,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a force (and optional torque) as a fraction of each env's weight ``m g``.

    Direction is a random azimuth with elevation from the horizontal plane.
    """
    n = total_mass.shape[0]
    weight = total_mass * gravity
    force_frac = math_utils.sample_uniform(
        force_weight_frac_range[0], force_weight_frac_range[1], (n,), device
    )
    mag = force_frac * weight
    azimuth = math_utils.sample_uniform(0.0, 2.0 * torch.pi, (n,), device)
    elevation = math_utils.sample_uniform(elevation_range[0], elevation_range[1], (n,), device)
    ce, se = torch.cos(elevation), torch.sin(elevation)
    ca, sa = torch.cos(azimuth), torch.sin(azimuth)
    direction = torch.stack((ce * ca, ce * sa, se), dim=-1)
    forces = mag.unsqueeze(-1) * direction

    torque_frac = math_utils.sample_uniform(
        torque_weight_frac_range[0], torque_weight_frac_range[1], (n,), device
    )
    tau_mag = torque_frac * weight * torque_lever_m
    tau_az = math_utils.sample_uniform(0.0, 2.0 * torch.pi, (n,), device)
    # Horizontal-axis torque (roll/pitch disturbance), random yaw of the axis.
    torques = tau_mag.unsqueeze(-1) * torch.stack(
        (torch.cos(tau_az), torch.sin(tau_az), torch.zeros(n, device=device)), dim=-1
    )
    return forces, torques


def reset_random_external_force(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["base_link"]),
    force_weight_frac_range: tuple[float, float] = (0.10, 0.35),
    torque_weight_frac_range: tuple[float, float] = (0.0, 0.0),
    duration_range_s: tuple[float, float] = (0.05, 0.20),
    interval_range_s: tuple[float, float] = (3.0, 8.0),
    elevation_range: tuple[float, float] = (-0.25, 0.35),
    torque_lever_m: float = 0.10,
    gravity: float = 9.81,
    is_global: bool = True,
    enabled: bool = True,
):
    """Arm the first random force pulse after a sampled delay. Magnitude uses live mass."""
    del (
        force_weight_frac_range,
        torque_weight_frac_range,
        duration_range_s,
        elevation_range,
        torque_lever_m,
        gravity,
        is_global,
    )
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    env_ids = env_ids.to(device=asset.device, dtype=torch.long)
    _random_force_buffers(env, env.scene.num_envs, asset.device)

    n = len(env_ids)
    env._extf_active[env_ids] = False
    env._extf_force[env_ids] = 0.0
    env._extf_torque[env_ids] = 0.0
    env._extf_t0[env_ids] = 0.0
    env._extf_dur[env_ids] = 0.0
    if enabled:
        env._extf_wait[env_ids] = math_utils.sample_uniform(
            interval_range_s[0], interval_range_s[1], (n,), asset.device
        )
    else:
        env._extf_wait[env_ids] = 1.0e9


def apply_random_external_force(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["base_link"]),
    force_weight_frac_range: tuple[float, float] = (0.10, 0.35),
    torque_weight_frac_range: tuple[float, float] = (0.0, 0.0),
    duration_range_s: tuple[float, float] = (0.05, 0.20),
    interval_range_s: tuple[float, float] = (3.0, 8.0),
    elevation_range: tuple[float, float] = (-0.25, 0.35),
    torque_lever_m: float = 0.10,
    gravity: float = 9.81,
    is_global: bool = True,
    enabled: bool = True,
):
    """Apply short force pulses in random directions, scaled to each env's weight ``m g``.

    Uses the instantaneous wrench composer so the pulse does not overwrite the
    kickstand (permanent) wrench. ``force_weight_frac_range`` is the force as a
    fraction of the bike's current weight (after mass randomization).
    """
    if not enabled:
        return
    if force_weight_frac_range[0] == 0.0 and force_weight_frac_range[1] == 0.0:
        if torque_weight_frac_range[0] == 0.0 and torque_weight_frac_range[1] == 0.0:
            return

    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    env_ids = env_ids.to(device=asset.device, dtype=torch.long)
    _random_force_buffers(env, env.scene.num_envs, asset.device)

    n = len(env_ids)
    episode_t = env.episode_length_buf[env_ids].to(dtype=torch.float32) * float(env.step_dt)
    active = env._extf_active[env_ids]
    wait = env._extf_wait[env_ids]

    start = (~active) & (episode_t >= wait)
    if start.any():
        start_ids = env_ids[start]
        total_mass = _body_mass_tensor(asset)[start_ids].sum(dim=-1)
        forces, torques = _sample_weight_scaled_wrench(
            total_mass,
            force_weight_frac_range,
            torque_weight_frac_range,
            elevation_range,
            torque_lever_m,
            gravity,
            asset.device,
        )
        env._extf_force[start_ids] = forces
        env._extf_torque[start_ids] = torques
        env._extf_t0[start_ids] = episode_t[start]
        env._extf_dur[start_ids] = math_utils.sample_uniform(
            duration_range_s[0], duration_range_s[1], (int(start.sum().item()),), asset.device
        )
        env._extf_active[start_ids] = True

    active = env._extf_active[env_ids]
    t0 = env._extf_t0[env_ids]
    dur = env._extf_dur[env_ids]
    finished = active & (episode_t >= t0 + dur)
    if finished.any():
        fin_ids = env_ids[finished]
        env._extf_active[fin_ids] = False
        env._extf_wait[fin_ids] = episode_t[finished] + math_utils.sample_uniform(
            interval_range_s[0], interval_range_s[1], (int(finished.sum().item()),), asset.device
        )

    pulsing = env._extf_active[env_ids]
    if not pulsing.any():
        return

    body_ids = _kickstand_body_ids(asset_cfg)
    forces = torch.zeros((n, 1, 3), device=asset.device)
    torques = torch.zeros((n, 1, 3), device=asset.device)
    ext_f = env._extf_force[env_ids]
    ext_t = env._extf_torque[env_ids]
    forces[:, 0, :] = torch.where(pulsing.unsqueeze(-1), ext_f, torch.zeros_like(ext_f))
    torques[:, 0, :] = torch.where(pulsing.unsqueeze(-1), ext_t, torch.zeros_like(ext_t))

    asset.instantaneous_wrench_composer.set_forces_and_torques_index(
        forces=forces,
        torques=torques,
        body_ids=body_ids,
        env_ids=env_ids.to(dtype=torch.int32),
        is_global=is_global,
    )
