"""Copied verbatim from ~/bike_test/CycleRL mdp/events.py, mdp/dynamic/events.py and mdp/dynamic/curriculums.py
(CycleRL-Trackstand-Cylinder-Paper-v0 deps). The noise helper is local instead of a lazy curriculums import."""

import math
from collections.abc import Sequence

import torch
from pxr import UsdPhysics

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

_IMU_OBS_DIM = 6
_CMD_OBS_DIM = 2

def randomize_ground_plane_material(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("terrain"),
    static_friction_range: tuple[float, float] = (0.5, 1.1),
    dynamic_friction_range: tuple[float, float] = (0.5, 1.1),
    restitution_range: tuple[float, float] = (0.0, 0.0),
    num_buckets: int = 1,
    make_consistent: bool = True,
):
    """Randomize the physics material on the shared ground plane.

    The terrain importer uses a shared ground plane prim for all environments.
    This helper updates the material prim bound to that plane instead of trying to
    randomize a non-rigid TerrainImporter asset directly.
    """
    # The terrain importer is not a rigid body asset, so we use its configured prim.
    terrain = env.scene[asset_cfg.name]
    if not hasattr(terrain, "terrain_prim_paths") or len(terrain.terrain_prim_paths) == 0:
        raise ValueError("Expected terrain importer asset with imported terrain prim paths.")

    terrain_prim_path = terrain.terrain_prim_paths[0]
    material_path = f"{terrain_prim_path}/physicsMaterial"
    material_prim = env.sim.stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        raise ValueError(
            f"Could not find ground plane material prim at '{material_path}'. "
            "Ensure the terrain importer is configured with a physics material."
        )

    # Sample material values once for the shared ground plane.
    static_friction = float(
        math_utils.sample_uniform(static_friction_range[0], static_friction_range[1], (1,), device="cpu")[0]
    )
    dynamic_friction = float(
        math_utils.sample_uniform(dynamic_friction_range[0], dynamic_friction_range[1], (1,), device="cpu")[0]
    )
    restitution = float(
        math_utils.sample_uniform(restitution_range[0], restitution_range[1], (1,), device="cpu")[0]
    )

    if make_consistent:
        dynamic_friction = min(static_friction, dynamic_friction)

    material_api = UsdPhysics.MaterialAPI(material_prim)
    material_api.CreateStaticFrictionAttr().Set(static_friction)
    material_api.CreateDynamicFrictionAttr().Set(dynamic_friction)
    material_api.CreateRestitutionAttr().Set(restitution)


def _as_torch(value):
    """Isaac Lab data buffers may be a ProxyArray with a ``.torch`` view."""
    return value.torch if hasattr(value, "torch") else value


def _iter_obs_groups(observations):
    """Yield observation groups from a dict or a configclass."""
    if observations is None:
        return
    if isinstance(observations, dict):
        yield from observations.values()
        return
    group = getattr(observations, "policy", observations)
    if group is not None:
        yield group


def set_obs_term_noise_std(env: ManagerBasedRLEnv, term_name: str, std) -> bool:
    """Write ``std`` onto the live and cfg copies of one observation term's noise.

    Unlike :func:`_set_imu_noise_std` this accepts a tensor (per-env and/or
    per-channel), so Table I's ``ρ · s_max`` can be scaled per signal.
    """
    applied = False
    obs_man = getattr(env, "observation_manager", None)
    if obs_man is not None:
        names_by_group = getattr(obs_man, "_group_obs_term_names", {})
        cfgs_by_group = getattr(obs_man, "_group_obs_term_cfgs", {})
        for group_name, names in names_by_group.items():
            if term_name not in names:
                continue
            term_cfg = cfgs_by_group[group_name][names.index(term_name)]
            noise = getattr(term_cfg, "noise", None)
            if noise is not None and hasattr(noise, "std"):
                noise.std = std
                applied = True
    for group in _iter_obs_groups(getattr(env.cfg, "observations", None)):
        term = getattr(group, term_name, None)
        noise = getattr(term, "noise", None) if term is not None else None
        if noise is not None and hasattr(noise, "std"):
            noise.std = std
            applied = True
    return applied


def randomize_imu_noise_fraction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    rho_range: tuple[float, float] = (0.01, 0.20),
    imu_smax: float | Sequence[float] = 0.02,
    rho_ladder: tuple[float, float] | None = None,
):
    """CycleRL paper Table I: ρ ~ U([0.01, 0.20]), obs noise std = ρ · s_max.

    ``rho_ladder=(lo, hi)`` pins each env to a fixed rung of a linspace ladder (MNT
    sweep). It is built here rather than by a separate event so the first episode
    is already on its rung regardless of reset-event order.

    Table I scales the noise by *each signal's* maximum magnitude, so ``imu_smax``
    accepts a per-channel sequence ordered ``[gravity_y, wx, wy, wz, speed, steer]``
    (6 long, IMU block only) or ``[..., v*, ωz*]`` (8 long, also the command block).
    A scalar keeps the legacy behaviour of one s_max for every IMU channel.

    ρ is drawn per environment so a reset batch spans the noise range rather than
    sharing one draw, and is published as ``env._eval_noise_rho`` for the MNT metric.
    """
    del env_ids
    device = env.device
    n = env.scene.num_envs
    rho_lo, rho_hi = float(rho_range[0]), float(rho_range[1])

    if rho_ladder is not None:
        env._eval_noise_rho_ladder = torch.linspace(
            float(rho_ladder[0]), float(rho_ladder[1]), n, device=device
        )
    ladder = getattr(env, "_eval_noise_rho_ladder", None)
    if ladder is not None and ladder.shape[0] == n:
        rho = ladder.to(device=device, dtype=torch.float32)
    else:
        rho = torch.empty(n, device=device).uniform_(rho_lo, rho_hi)
    env._eval_noise_rho = rho

    smax = torch.as_tensor(imu_smax, dtype=torch.float32, device=device).reshape(-1)
    if smax.numel() == 1:
        smax = smax.repeat(_IMU_OBS_DIM)
    if smax.numel() not in (_IMU_OBS_DIM, _IMU_OBS_DIM + _CMD_OBS_DIM):
        raise ValueError(
            f"imu_smax must be a scalar or have {_IMU_OBS_DIM} or "
            f"{_IMU_OBS_DIM + _CMD_OBS_DIM} entries, got {smax.numel()}"
        )

    rho_col = rho.unsqueeze(-1)  # (N, 1) broadcasts over channels
    set_obs_term_noise_std(env, "imu_state", rho_col * smax[:_IMU_OBS_DIM])
    if smax.numel() == _IMU_OBS_DIM + _CMD_OBS_DIM:
        set_obs_term_noise_std(env, "commands", rho_col * smax[_IMU_OBS_DIM:])


def reset_paper_joint_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    steer_joint_name: str = "steering_joint",
    drive_joint_name: str = "rearwheel_joint",
    front_wheel_joint_name: str = "frontwheel_joint",
    steer_deg_range: tuple[float, float] = (-20.0, 20.0),
    drive_vel_range: tuple[float, float] = (0.0, 3.0),
    couple_drive_to_root_vx: bool = False,
    wheel_radius_m: float = 0.0725,
    contact_radius_m: float | None = None,
    drive_joint_sign: float = 1.0,
    couple_front_wheel: bool = True,
):
    """CycleRL paper Table I: θ_m,init ∈ ±20°, ω_m,init ∈ [0, 3] rad/s.

    When ``couple_drive_to_root_vx=True``, hub ω is set from the root forward
    speed written by the preceding ``initial_state`` reset event
    (ω = sign · vx / wheel_radius_m). Use on Newton to avoid MuJoCo
    redistributing inconsistent root/joint state on the first integrate step.
    """
    asset = env.scene[asset_cfg.name]
    device = asset.device
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=device, dtype=torch.long)

    steer_idx = asset.find_joints(steer_joint_name)[0][0]
    drive_idx = asset.find_joints(drive_joint_name)[0][0]
    radius = float(contact_radius_m if contact_radius_m is not None else wheel_radius_m)

    joint_pos = _as_torch(asset.data.default_joint_pos)[env_ids].clone()
    joint_vel = _as_torch(asset.data.default_joint_vel)[env_ids].clone()

    n = len(env_ids)
    steer_lo, steer_hi = (math.radians(steer_deg_range[0]), math.radians(steer_deg_range[1]))
    joint_pos[:, steer_idx] = torch.empty(n, device=device).uniform_(steer_lo, steer_hi)
    if couple_drive_to_root_vx:
        # root_lin_vel_b may lag until the next sim update; read world COM velocity
        # written by the preceding initial_state event and transform to body frame.
        root_vel_w = _as_torch(asset.data.root_com_vel_w)[env_ids, :3]
        quat_w = _as_torch(asset.data.root_quat_w)[env_ids]
        lin_vel_b = math_utils.quat_apply_inverse(quat_w, root_vel_w)
        hub_omega = float(drive_joint_sign) * lin_vel_b[:, 0] / radius
        joint_vel[:, drive_idx] = hub_omega
        if couple_front_wheel:
            try:
                front_idx = asset.find_joints(front_wheel_joint_name)[0][0]
                joint_vel[:, front_idx] = hub_omega
            except (IndexError, ValueError):
                pass
    else:
        drive_lo, drive_hi = float(drive_vel_range[0]), float(drive_vel_range[1])
        joint_vel[:, drive_idx] = torch.empty(n, device=device).uniform_(drive_lo, drive_hi)

    pos_limits = _as_torch(asset.data.soft_joint_pos_limits)[env_ids, steer_idx]
    joint_pos[:, steer_idx] = joint_pos[:, steer_idx].clamp(
        pos_limits[..., 0], pos_limits[..., 1]
    )
    vel_limits = _as_torch(asset.data.soft_joint_vel_limits)[env_ids, drive_idx]
    joint_vel[:, drive_idx] = joint_vel[:, drive_idx].clamp(-vel_limits, vel_limits)
    if couple_drive_to_root_vx and couple_front_wheel:
        try:
            front_idx = asset.find_joints(front_wheel_joint_name)[0][0]
            front_limits = _as_torch(asset.data.soft_joint_vel_limits)[env_ids, front_idx]
            joint_vel[:, front_idx] = joint_vel[:, front_idx].clamp(-front_limits, front_limits)
        except (IndexError, ValueError):
            pass

    asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
