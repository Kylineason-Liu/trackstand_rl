"""Eval-only event terms copied verbatim from ~/bike_test/CycleRL mdp/dynamic/events.py.

Used by locomotive/eval/eval_scenarios.py; training terms stay in events.py, which
this module reuses instead of duplicating."""

import math

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from .events import _as_torch, randomize_imu_noise_fraction  # noqa: F401  (re-exported for eval_scenarios)


def assign_eval_speed_ladder(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    velocity_command_name: str = "velocity",
    heading_command_name: str = "steering",
    v_min: float = 0.3,
    v_max: float = 2.5,
):
    """Pin each env to a fixed vx* on the first step after reset (speed-sweep eval).

    Runs as an interval term after ``command_manager.compute`` so it overrides the
    uniform resample with a deterministic ``linspace(v_min, v_max)`` ladder.
    """
    device = env.device
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=device, dtype=torch.long)

    fresh = env.episode_length_buf[env_ids] == 0
    if not torch.any(fresh):
        return

    ids = env_ids[fresh]
    n = env.scene.num_envs
    ladder = torch.linspace(float(v_min), float(v_max), n, device=device)
    vel_term = env.command_manager._terms[velocity_command_name]
    vel_term.vel_command_b[ids, 0] = ladder[ids]
    vel_term.vel_command_b[ids, 1] = 0.0
    vel_term.vel_command_b[ids, 2] = 0.0
    if hasattr(vel_term, "is_standing_env"):
        vel_term.is_standing_env[ids] = False

    if heading_command_name in env.command_manager._terms:
        steer_term = env.command_manager._terms[heading_command_name]
        steer_term.vel_command_b[ids, 0] = 0.0
        steer_term.vel_command_b[ids, 1] = 0.0
        steer_term.vel_command_b[ids, 2] = 0.0
        if hasattr(steer_term, "heading_target"):
            steer_term.heading_target[ids] = 0.0
        if hasattr(steer_term, "is_heading_env"):
            steer_term.is_heading_env[ids] = True
        if hasattr(steer_term, "is_standing_env"):
            steer_term.is_standing_env[ids] = False


def _eval_heading_ladder(h_min: float, h_max: float, n: int, device) -> torch.Tensor:
    """Linspace heading* ladder that always includes exact 0 when the range straddles it.

    Plain ``linspace(lo, hi, n)`` skips 0 for even ``n`` (e.g. ±10° with 64 envs),
    so the sweep would have no true straight (θ*=0) bin.
    """
    lo = float(h_min)
    hi = float(h_max)
    if n <= 0:
        return torch.empty(0, device=device)
    if n == 1:
        mid = 0.0 if lo <= 0.0 <= hi else 0.5 * (lo + hi)
        return torch.tensor([mid], device=device, dtype=torch.float32)
    if lo < -1e-12 and hi > 1e-12:
        n_neg = (n - 1) // 2
        n_pos = n - 1 - n_neg
        neg = (
            torch.linspace(lo, 0.0, n_neg + 1, device=device)[:-1]
            if n_neg > 0
            else torch.empty(0, device=device)
        )
        pos = (
            torch.linspace(0.0, hi, n_pos + 1, device=device)[1:]
            if n_pos > 0
            else torch.empty(0, device=device)
        )
        zero = torch.zeros(1, device=device)
        return torch.cat([neg, zero, pos])
    return torch.linspace(lo, hi, n, device=device)


def assign_eval_heading_ladder(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    velocity_command_name: str = "velocity",
    heading_command_name: str = "steering",
    heading_min_rad: float = -0.17453292519943295,
    heading_max_rad: float = 0.17453292519943295,
    vx_ms: float = 1.8,
):
    """Pin each env to a fixed heading* on the first step after reset.

    Holds vx* at ``vx_ms`` and sets ``heading_target`` on a ladder that always
    includes θ*=0 (straight) when ``heading_min < 0 < heading_max``.
    Runs as an interval term after ``command_manager.compute``.
    """
    device = env.device
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=device, dtype=torch.long)

    fresh = env.episode_length_buf[env_ids] == 0
    if not torch.any(fresh):
        return

    ids = env_ids[fresh]
    n = env.scene.num_envs
    ladder = _eval_heading_ladder(
        float(heading_min_rad), float(heading_max_rad), n, device
    )

    if velocity_command_name in env.command_manager._terms:
        vel_term = env.command_manager._terms[velocity_command_name]
        vel_term.vel_command_b[ids, 0] = float(vx_ms)
        vel_term.vel_command_b[ids, 1] = 0.0
        vel_term.vel_command_b[ids, 2] = 0.0
        if hasattr(vel_term, "is_standing_env"):
            vel_term.is_standing_env[ids] = False

    if heading_command_name in env.command_manager._terms:
        steer_term = env.command_manager._terms[heading_command_name]
        steer_term.vel_command_b[ids, 0] = 0.0
        steer_term.vel_command_b[ids, 1] = 0.0
        # Heading P-controller writes ang_vel_z from heading_target; clear seed wz*.
        steer_term.vel_command_b[ids, 2] = 0.0
        if hasattr(steer_term, "heading_target"):
            steer_term.heading_target[ids] = ladder[ids]
        if hasattr(steer_term, "is_heading_env"):
            steer_term.is_heading_env[ids] = True
        if hasattr(steer_term, "is_standing_env"):
            steer_term.is_standing_env[ids] = False


def assign_eval_drive_accel_ladder(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    drive_action_name: str = "drive",
    a_min: float = 0.25,
    a_max: float = 4.0,
):
    """Pin each env to a fixed hub accel (m/s²) on the first step after reset.

    Policy a1 is unused: RC command drive owns ω*. Interval term so it runs after
    the action manager exists. Writes ``env._eval_drive_accel_ms2`` for the logger.
    """
    device = env.device
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=device, dtype=torch.long)

    n = env.scene.num_envs
    ladder = torch.linspace(float(a_min), float(a_max), n, device=device)
    if not hasattr(env, "_eval_drive_accel_ms2") or env._eval_drive_accel_ms2.shape[0] != n:
        env._eval_drive_accel_ms2 = ladder.clone()

    fresh = env.episode_length_buf[env_ids] == 0
    if not torch.any(fresh):
        return

    ids = env_ids[fresh]
    env._eval_drive_accel_ms2[ids] = ladder[ids]

    terms = getattr(getattr(env, "action_manager", None), "_terms", None) or {}
    term = terms.get(drive_action_name)
    if term is not None and hasattr(term, "set_accel_ms2"):
        term.set_accel_ms2(ids, ladder[ids])


def assign_eval_mass_scale_ladder(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale_min: float = 1.0,
    scale_max: float = 1.5,
    recompute_inertia: bool = True,
    min_mass: float = 1e-6,
):
    """Pin each env to a fixed mass scale on a linspace ladder (CoM unchanged).

    Scales body mass (and optionally inertia) from cached defaults so repeated
    resets do not compound. Writes ``env._eval_mass_scale`` for the logger.
    """
    asset = env.scene[asset_cfg.name]
    device = asset.device
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=device, dtype=torch.long)

    n = env.scene.num_envs
    default_mass = getattr(env, "_eval_mass_default", None)
    if default_mass is None or default_mass.shape[0] != n:
        env._eval_mass_default = _as_torch(asset.data.body_mass).clone()
        env._eval_inertia_default = _as_torch(asset.data.body_inertia).clone()
        default_mass = env._eval_mass_default

    ladder = torch.linspace(float(scale_min), float(scale_max), n, device=device)
    if not hasattr(env, "_eval_mass_scale") or env._eval_mass_scale.shape[0] != n:
        env._eval_mass_scale = ladder.clone()
    env._eval_mass_scale[:] = ladder

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int32, device=device)
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int32, device=device)

    masses = _as_torch(asset.data.body_mass).clone()
    scales = ladder[env_ids].unsqueeze(-1)  # (E, 1)
    masses[env_ids[:, None], body_ids] = (
        default_mass[env_ids[:, None], body_ids] * scales
    ).clamp(min=min_mass)
    asset.set_masses_index(
        masses=masses[env_ids[:, None], body_ids],
        body_ids=body_ids,
        env_ids=env_ids,
    )

    if recompute_inertia:
        ratios = masses[env_ids[:, None], body_ids] / default_mass[env_ids[:, None], body_ids]
        inertias = _as_torch(asset.data.body_inertia).clone()
        inertias[env_ids[:, None], body_ids] = (
            env._eval_inertia_default[env_ids[:, None], body_ids] * ratios[..., None]
        )
        asset.set_inertias_index(
            inertias=inertias[env_ids[:, None], body_ids],
            body_ids=body_ids,
            env_ids=env_ids,
        )


def assign_eval_com_offset_ladder(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    offset_min_m: float = -0.05,
    offset_max_m: float = 0.05,
    axis: str = "y",
):
    """Pin each env to a fixed body-frame CoM offset on a linspace ladder.

    Mass stays nominal. Default axis is lateral ``y`` (balance-sensitive).
    Writes ``env._eval_com_offset_m`` (signed metres along ``axis``) for logging.
    """
    asset = env.scene[asset_cfg.name]
    device = asset.device
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=device, dtype=torch.long)

    axis_key = str(axis).lower().strip()
    if axis_key not in ("x", "y", "z"):
        raise ValueError(f"com_sweep axis must be x/y/z, got {axis!r}")
    axis_i = {"x": 0, "y": 1, "z": 2}[axis_key]

    n = env.scene.num_envs
    # Prefer full pose buffer (PhysX); fall back to position-only (Newton).
    pose = getattr(asset.data, "body_com_pose_b", None)
    if pose is not None:
        default = getattr(env, "_eval_com_default", None)
        if default is None or default.shape[0] != n:
            env._eval_com_default = _as_torch(pose).clone()
            env._eval_com_is_pose = True
    else:
        pos = asset.data.body_com_pos_b
        default = getattr(env, "_eval_com_default", None)
        if default is None or default.shape[0] != n:
            env._eval_com_default = _as_torch(pos).clone()
            env._eval_com_is_pose = False

    ladder = torch.linspace(float(offset_min_m), float(offset_max_m), n, device=device)
    if not hasattr(env, "_eval_com_offset_m") or env._eval_com_offset_m.shape[0] != n:
        env._eval_com_offset_m = ladder.clone()
    env._eval_com_offset_m[:] = ladder
    env._eval_com_axis = axis_key

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int32, device=device)
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int32, device=device)

    coms = env._eval_com_default.clone()
    offsets = ladder[env_ids].unsqueeze(-1)  # (E, 1)
    coms[env_ids[:, None], body_ids, axis_i] = (
        env._eval_com_default[env_ids[:, None], body_ids, axis_i] + offsets
    )

    if getattr(env, "_eval_com_is_pose", True) and coms.shape[-1] >= 7:
        asset.set_coms_index(
            coms=coms[env_ids[:, None], body_ids],
            body_ids=body_ids,
            env_ids=env_ids,
        )
    else:
        asset.set_coms_index(
            coms=coms[env_ids[:, None], body_ids, :3],
            body_ids=body_ids,
            env_ids=env_ids,
        )


def _eval_mass_com_grid(
    n: int,
    scale_min: float,
    scale_max: float,
    offset_min_m: float,
    offset_max_m: float,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor ``n`` envs into a mass-scale × CoM-offset grid (√n × n/√n)."""
    if n <= 0:
        empty = torch.empty(0, device=device)
        return empty, empty
    n_mass = max(1, int(round(math.sqrt(float(n)))))
    n_mass = min(n_mass, n)
    n_com = max(1, n // n_mass)
    mass_levels = torch.linspace(float(scale_min), float(scale_max), n_mass, device=device)
    com_levels = torch.linspace(float(offset_min_m), float(offset_max_m), n_com, device=device)
    idx = torch.arange(n, device=device)
    mass = mass_levels[(idx // n_com) % n_mass]
    com = com_levels[idx % n_com]
    return mass, com


def assign_eval_mass_com_grid(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale_min: float = 1.0,
    scale_max: float = 1.25,
    offset_min_m: float = -0.03,
    offset_max_m: float = 0.03,
    axis: str = "y",
    recompute_inertia: bool = True,
    min_mass: float = 1e-6,
):
    """Pin each env to a (mass scale, lateral CoM) cell on a 2-D eval grid.

    Writes ``env._eval_mass_scale`` and ``env._eval_com_offset_m`` for logging.
    """
    asset = env.scene[asset_cfg.name]
    device = asset.device
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=device, dtype=torch.long)

    axis_key = str(axis).lower().strip()
    if axis_key not in ("x", "y", "z"):
        raise ValueError(f"mass_com_sweep axis must be x/y/z, got {axis!r}")
    axis_i = {"x": 0, "y": 1, "z": 2}[axis_key]

    n = env.scene.num_envs
    ladder_mass, ladder_com = _eval_mass_com_grid(
        n, scale_min, scale_max, offset_min_m, offset_max_m, device
    )
    if not hasattr(env, "_eval_mass_scale") or env._eval_mass_scale.shape[0] != n:
        env._eval_mass_scale = ladder_mass.clone()
    if not hasattr(env, "_eval_com_offset_m") or env._eval_com_offset_m.shape[0] != n:
        env._eval_com_offset_m = ladder_com.clone()
    env._eval_mass_scale[:] = ladder_mass
    env._eval_com_offset_m[:] = ladder_com
    env._eval_com_axis = axis_key

    # --- mass / inertia ---
    default_mass = getattr(env, "_eval_mass_default", None)
    if default_mass is None or default_mass.shape[0] != n:
        env._eval_mass_default = _as_torch(asset.data.body_mass).clone()
        env._eval_inertia_default = _as_torch(asset.data.body_inertia).clone()
        default_mass = env._eval_mass_default

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int32, device=device)
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int32, device=device)

    masses = _as_torch(asset.data.body_mass).clone()
    scales = ladder_mass[env_ids].unsqueeze(-1)
    masses[env_ids[:, None], body_ids] = (
        default_mass[env_ids[:, None], body_ids] * scales
    ).clamp(min=min_mass)
    asset.set_masses_index(
        masses=masses[env_ids[:, None], body_ids],
        body_ids=body_ids,
        env_ids=env_ids,
    )
    if recompute_inertia:
        ratios = masses[env_ids[:, None], body_ids] / default_mass[env_ids[:, None], body_ids]
        inertias = _as_torch(asset.data.body_inertia).clone()
        inertias[env_ids[:, None], body_ids] = (
            env._eval_inertia_default[env_ids[:, None], body_ids] * ratios[..., None]
        )
        asset.set_inertias_index(
            inertias=inertias[env_ids[:, None], body_ids],
            body_ids=body_ids,
            env_ids=env_ids,
        )

    # --- CoM ---
    pose = getattr(asset.data, "body_com_pose_b", None)
    if pose is not None:
        default_com = getattr(env, "_eval_com_default", None)
        if default_com is None or default_com.shape[0] != n:
            env._eval_com_default = _as_torch(pose).clone()
            env._eval_com_is_pose = True
    else:
        default_com = getattr(env, "_eval_com_default", None)
        if default_com is None or default_com.shape[0] != n:
            env._eval_com_default = _as_torch(asset.data.body_com_pos_b).clone()
            env._eval_com_is_pose = False

    coms = env._eval_com_default.clone()
    offsets = ladder_com[env_ids].unsqueeze(-1)
    coms[env_ids[:, None], body_ids, axis_i] = (
        env._eval_com_default[env_ids[:, None], body_ids, axis_i] + offsets
    )
    if getattr(env, "_eval_com_is_pose", True) and coms.shape[-1] >= 7:
        asset.set_coms_index(
            coms=coms[env_ids[:, None], body_ids],
            body_ids=body_ids,
            env_ids=env_ids,
        )
    else:
        asset.set_coms_index(
            coms=coms[env_ids[:, None], body_ids, :3],
            body_ids=body_ids,
            env_ids=env_ids,
        )


def assign_eval_friction_ladder(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mu_min: float = 0.2,
    mu_max: float = 1.0,
):
    """Pin each env to a fixed tire μ on a linspace ladder.

    Ground plane μ is shared (set at scenario config to ``mu_min``); this term
    only varies articulationshape materials. Writes ``env._eval_friction_mu``.
    """
    import warp as wp

    asset = env.scene[asset_cfg.name]
    device = asset.device
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=device, dtype=torch.long)

    n = env.scene.num_envs
    ladder = torch.linspace(float(mu_min), float(mu_max), n, device=device)
    if not hasattr(env, "_eval_friction_mu") or env._eval_friction_mu.shape[0] != n:
        env._eval_friction_mu = ladder.clone()
    env._eval_friction_mu[:] = ladder

    materials = wp.to_torch(asset.root_view.get_material_properties())
    # (num_envs, num_shapes, 3) — static, dynamic, restitution
    ids_cpu = env_ids.to(device="cpu", dtype=torch.long)
    mu = ladder[env_ids].detach().cpu().unsqueeze(-1)
    materials[ids_cpu, :, 0] = mu
    materials[ids_cpu, :, 1] = mu
    asset.root_view.set_material_properties(
        wp.from_torch(materials, dtype=wp.float32),
        wp.from_torch(ids_cpu.to(torch.int32), dtype=wp.int32),
    )


def assign_eval_roll_ladder(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    roll_min_deg: float = -40.0,
    roll_max_deg: float = 40.0,
):
    """Spawn each env at a fixed initial roll on a linspace ladder.

    Runs after ``initial_state`` and rewrites only the roll component of the root
    orientation, so the recoverable-lean frontier (CAT) can be measured directly.
    """
    asset = env.scene[asset_cfg.name]
    device = asset.device
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=device, dtype=torch.long)
    if env_ids.numel() == 0:
        return

    n = env.scene.num_envs
    ladder = torch.deg2rad(
        torch.linspace(float(roll_min_deg), float(roll_max_deg), n, device=device)
    )

    pos = _as_torch(asset.data.root_pos_w)[env_ids].clone()
    quat = _as_torch(asset.data.root_quat_w)[env_ids].clone()
    _, pitch, yaw = math_utils.euler_xyz_from_quat(quat)
    new_quat = math_utils.quat_from_euler_xyz(ladder[env_ids], pitch, yaw)

    asset.write_root_pose_to_sim_index(
        root_pose=torch.cat([pos, new_quat], dim=-1), env_ids=env_ids
    )


def _eval_yaw_bias_vx_bias_grid(
    n: int,
    vx_min: float,
    vx_max: float,
    bias_min_deg: float,
    bias_max_deg: float,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a (vx*, bias°) grid covering ``n`` envs.

    When ``vx_min ≈ vx_max``, every env shares that speed and bias is a dense
    linspace (legacy single-speed sweep). Otherwise env count is factored into
    roughly ``√n`` speed levels × ``n // n_vx`` bias levels (tiled if needed).
    """
    v0 = float(vx_min)
    v1 = float(vx_max)
    b0 = float(bias_min_deg)
    b1 = float(bias_max_deg)
    if n <= 0:
        empty = torch.empty(0, device=device)
        return empty, empty
    if abs(v1 - v0) < 1e-9:
        bias = torch.linspace(b0, b1, n, device=device)
        vx = torch.full((n,), v0, device=device)
        return vx, bias

    n_vx = max(1, int(round(math.sqrt(float(n)))))
    n_vx = min(n_vx, n)
    n_bias = max(1, n // n_vx)
    vx_levels = torch.linspace(v0, v1, n_vx, device=device)
    bias_levels = torch.linspace(b0, b1, n_bias, device=device)
    idx = torch.arange(n, device=device)
    vx = vx_levels[(idx // n_bias) % n_vx]
    bias = bias_levels[idx % n_bias]
    return vx, bias


def assign_eval_yaw_bias_ladder(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    velocity_command_name: str = "velocity",
    heading_command_name: str = "steering",
    bias_min_deg: float = -5.0,
    bias_max_deg: float = 5.0,
    yaw_rate_scale: float = 0.45,
    stick_full_scale_deg: float = 10.0,
    vx_ms: float = 1.8,
    vx_min: float | None = None,
    vx_max: float | None = None,
):
    """Pin each env to a fixed (vx*, ωz*) on a yaw-bias × speed grid.

    Bias ladder is in stick-equivalent degrees (same mapping as ``play_teleop``):
    ``ωz* = yaw_rate_scale * (bias_deg / stick_full_scale_deg)``.
    Speed defaults to ``vx_ms`` when ``vx_min``/``vx_max`` are unset; otherwise
    envs cover a 2-D grid of commanded speeds and biases.
    Sets ``is_heading_env=False`` so heading P-control does not overwrite ωz*.
    Re-applies every interval step after ``command_manager.compute``.
    Writes ``env._eval_yaw_bias_deg`` / ``env._eval_yaw_bias_vx_ms`` for the logger.
    """
    device = env.device
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=device, dtype=torch.long)

    n = env.scene.num_envs
    v_lo = float(vx_ms if vx_min is None else vx_min)
    v_hi = float(vx_ms if vx_max is None else vx_max)
    ladder_vx, ladder_deg = _eval_yaw_bias_vx_bias_grid(
        n, v_lo, v_hi, float(bias_min_deg), float(bias_max_deg), device
    )
    if not hasattr(env, "_eval_yaw_bias_deg") or env._eval_yaw_bias_deg.shape[0] != n:
        env._eval_yaw_bias_deg = ladder_deg.clone()
    if not hasattr(env, "_eval_yaw_bias_vx_ms") or env._eval_yaw_bias_vx_ms.shape[0] != n:
        env._eval_yaw_bias_vx_ms = ladder_vx.clone()
    env._eval_yaw_bias_deg[:] = ladder_deg
    env._eval_yaw_bias_vx_ms[:] = ladder_vx

    scale = float(yaw_rate_scale) / max(float(stick_full_scale_deg), 1e-6)
    wz = ladder_deg * scale
    ids = env_ids

    if velocity_command_name in env.command_manager._terms:
        vel_term = env.command_manager._terms[velocity_command_name]
        vel_term.vel_command_b[ids, 0] = ladder_vx[ids]
        vel_term.vel_command_b[ids, 1] = 0.0
        vel_term.vel_command_b[ids, 2] = 0.0
        if hasattr(vel_term, "is_standing_env"):
            vel_term.is_standing_env[ids] = False

    if heading_command_name in env.command_manager._terms:
        steer_term = env.command_manager._terms[heading_command_name]
        steer_term.vel_command_b[ids, 0] = 0.0
        steer_term.vel_command_b[ids, 1] = 0.0
        steer_term.vel_command_b[ids, 2] = wz[ids]
        if hasattr(steer_term, "heading_target"):
            steer_term.heading_target[ids] = 0.0
        if hasattr(steer_term, "is_heading_env"):
            # Direct ωz* (teleop path); do not let heading P overwrite.
            steer_term.is_heading_env[ids] = False
        if hasattr(steer_term, "is_standing_env"):
            steer_term.is_standing_env[ids] = False


def match_spawn_to_velocity_command(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "velocity",
    drive_joint_name: str = "rearwheel_joint",
    wheel_radius_m: float = 0.0725,
    drive_joint_sign: float = 1.0,
    velocity_noise: float = 0.1,
):
    """On the first step after reset, set root ``vx`` and rear-wheel spin to ``v*``.

    Runs as an interval term (after ``command_manager.compute``) because reset
    events fire before commands are resampled.
    """
    asset = env.scene[asset_cfg.name]
    device = asset.device
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=device, dtype=torch.long)

    fresh = env.episode_length_buf[env_ids] == 0
    if not torch.any(fresh):
        return

    ids = env_ids[fresh]
    cmd_vx = env.command_manager.get_command(command_name)[ids, 0]
    if velocity_noise > 0.0:
        cmd_vx = cmd_vx + (torch.rand_like(cmd_vx) * 2.0 - 1.0) * velocity_noise

    quat_w = _as_torch(asset.data.root_quat_w)[ids]
    root_vel = _as_torch(asset.data.root_vel_w)[ids].clone()
    lin_vel_b = torch.zeros(len(ids), 3, device=device)
    lin_vel_b[:, 0] = cmd_vx
    root_vel[:, :3] = math_utils.quat_apply(quat_w, lin_vel_b)
    asset.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=ids)

    joint_idx = asset.find_joints(drive_joint_name)[0][0]
    wheel_omega = drive_joint_sign * cmd_vx / wheel_radius_m
    asset.write_joint_velocity_to_sim_index(
        velocity=wheel_omega.unsqueeze(-1),
        joint_ids=[joint_idx],
        env_ids=ids,
    )


def _kickstand_body_ids(asset_cfg: SceneEntityCfg, env: ManagerBasedEnv | None = None):
    """Single body for the kickstand wrench (the first resolved name, base_link).

    Extra names in ``body_names`` are ignored so the disturbance is never
    silently applied to the fork or wheels.

    When called from the event manager, ``body_ids`` are already resolved. When
    called from a play script, pass ``env`` so names can be looked up on the asset.
    """
    ids = asset_cfg.body_ids
    if isinstance(ids, list) and len(ids) > 0:
        return [ids[0]]
    if torch.is_tensor(ids) and ids.numel() > 0:
        return ids.reshape(-1)[:1]
    if env is not None:
        names = asset_cfg.body_names or ["base_link"]
        if isinstance(names, str):
            names = [names]
        asset = env.scene[asset_cfg.name]
        found_ids, _ = asset.find_bodies(names[0])
        if len(found_ids) > 0:
            return [int(found_ids[0])]
    raise ValueError(
        "Kickstand wrench needs a resolved body id. "
        "Set body_names=['base_link'] on the event SceneEntityCfg."
    )


def _eval_impulse_buffers(env: ManagerBasedEnv, num_envs: int, device) -> None:
    """Per-env lateral impulse state used only during evaluation."""
    if getattr(env, "_eval_imp_sign", None) is not None and env._eval_imp_sign.shape[0] == num_envs:
        return
    env._eval_imp_sign = torch.ones(num_envs, device=device)
    env._eval_imp_delay = torch.zeros(num_envs, device=device)
    # -1 = unset (fall back to apply ``force_n``); 0 is a valid kick_sweep rung.
    env._eval_imp_force = torch.full((num_envs,), -1.0, device=device)
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
    force_min: float | None = None,
    force_max: float | None = None,
):
    """Arm a one-shot lateral impulse on ``base_link`` after ``delay_s``.

    When ``force_min``/``force_max`` are set, each env gets a fixed force from a
    linspace ladder (kick_sweep). Writes ``env._eval_kick_force_n`` for logging.
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    env_ids = env_ids.to(device=asset.device, dtype=torch.long)
    _eval_impulse_buffers(env, env.scene.num_envs, asset.device)

    n_all = env.scene.num_envs
    if force_min is not None and force_max is not None:
        ladder = torch.linspace(float(force_min), float(force_max), n_all, device=asset.device)
        if not hasattr(env, "_eval_kick_force_n") or env._eval_kick_force_n.shape[0] != n_all:
            env._eval_kick_force_n = ladder.clone()
        env._eval_kick_force_n[:] = ladder
        env._eval_imp_force[env_ids] = ladder[env_ids]
    else:
        env._eval_imp_force[env_ids] = float(force_n)
        if not hasattr(env, "_eval_kick_force_n") or env._eval_kick_force_n.shape[0] != n_all:
            env._eval_kick_force_n = torch.full((n_all,), float(force_n), device=asset.device)
        else:
            env._eval_kick_force_n[env_ids] = float(force_n)

    n = len(env_ids)
    env._eval_imp_sign[env_ids] = torch.where(
        torch.rand(n, device=asset.device) > 0.5,
        torch.ones(n, device=asset.device),
        -torch.ones(n, device=asset.device),
    )
    env._eval_imp_delay[env_ids] = delay_s
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
    # Scheduled force uses >= 0 so kick_sweep can include a 0 N rung.
    force = torch.where(
        env._eval_imp_force[env_ids] >= 0,
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
