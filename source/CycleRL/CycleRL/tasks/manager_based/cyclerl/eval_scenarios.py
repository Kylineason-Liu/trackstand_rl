"""Evaluation-scenario helpers for CycleRL locomotion checkpoints.

This policy family is dynamic balancing / velocity tracking only — no trackstand.
Scenarios:

  nominal     Flat plane, no domain rand, fixed ~1.8 m/s straight. Deployment baseline.
  training    Complex terrain + training-like commands and randomizations (no kickstand).
  kick        Nominal plane + 10 N × 0.1 s lateral impulse at t = 2 s.
  mass        Plane, mass ×1.10 and CoM ±2 cm.
  friction    Plane, ground μ = 0.4.

Terrain extremity stages pin every env to one curriculum *row* while still
spreading across columns (terrain types). Stairs and complex stay as separate
grids (4 m vs 10 m cells):

  complex_L0 … complex_L4   easiest → hardest height-field
  stairs_L0  … stairs_L2    shallow → steep (steps and waves)

Suite aliases: ``terrain`` (all 8 stages), ``complex`` (L0–L4), ``stairs`` (L0–L2).
"""

from __future__ import annotations

import math
from typing import Any

from isaaclab.managers import EventTermCfg, SceneEntityCfg

from .config.terrain_presets import (
    COMPLEX_NUM_ROWS,
    STAIRS_NUM_ROWS,
    TERRAIN_PRESET_META,
    apply_terrain_preset,
    pin_terrain_level,
)
from .mdp.dynamic.events import apply_eval_lateral_impulse, reset_eval_impulse_schedule

EVAL_BASE_MODES = ("nominal", "training", "kick", "mass", "friction")
COMPLEX_STAGE_MODES = tuple(f"complex_L{i}" for i in range(COMPLEX_NUM_ROWS))
STAIRS_STAGE_MODES = tuple(f"stairs_L{i}" for i in range(STAIRS_NUM_ROWS))
TERRAIN_STAGE_MODES = COMPLEX_STAGE_MODES + STAIRS_STAGE_MODES
EVAL_MODES = EVAL_BASE_MODES + TERRAIN_STAGE_MODES

EVAL_SUITE_ALIASES: dict[str, tuple[str, ...]] = {
    "terrain": TERRAIN_STAGE_MODES,
    "complex": COMPLEX_STAGE_MODES,
    "stairs": STAIRS_STAGE_MODES,
}

# Match the 2026-08-13_15-24-02 nice2 training snapshot (params/env.yaml).
_TRAIN_VX = (1.0, 3.0)
_TRAIN_HEADING = (-0.174, 0.174)
_TRAIN_SPAWN_VX = (0.5, 1.5)
_NOMINAL_VX = 1.8
_NOMINAL_SPAWN_VX = (1.5, 1.8)


def parse_terrain_stage(mode: str) -> tuple[str, int] | None:
    """Return ``(preset, row)`` for ``complex_L2`` / ``stairs_L0``, else None."""
    for preset, n_rows in (("complex", COMPLEX_NUM_ROWS), ("stairs", STAIRS_NUM_ROWS)):
        prefix = f"{preset}_L"
        if mode.startswith(prefix):
            try:
                level = int(mode[len(prefix) :])
            except ValueError:
                return None
            if 0 <= level < n_rows:
                return preset, level
    return None


def is_terrain_stage(mode: str) -> bool:
    return parse_terrain_stage(mode) is not None


def expand_eval_modes(modes: list[str]) -> list[str]:
    """Expand suite aliases (``terrain``, ``complex``, ``stairs``) into stage names."""
    expanded: list[str] = []
    for mode in modes:
        if mode in EVAL_SUITE_ALIASES:
            expanded.extend(EVAL_SUITE_ALIASES[mode])
            continue
        if mode not in EVAL_MODES:
            raise ValueError(
                f"Unknown eval mode {mode!r}. Choose from {EVAL_MODES} "
                f"or aliases {tuple(EVAL_SUITE_ALIASES)}"
            )
        expanded.append(mode)
    return expanded


def pin_eval_terrain_level(env: Any, level: int) -> dict[str, Any]:
    """Pin every env to one difficulty row after ``gym.make``, before reset."""
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    return pin_terrain_level(unwrapped, level)


def _row_difficulty(num_rows: int, level: int) -> float:
    if num_rows <= 1:
        return 0.0
    return float(level) / float(num_rows - 1)


def _lerp(lo: float, hi: float, t: float) -> float:
    return lo + (hi - lo) * t


def terrain_stage_description(preset: str, level: int) -> str:
    """Human-readable extremity at a pinned curriculum row."""
    if preset == "complex":
        t = _row_difficulty(COMPLEX_NUM_ROWS, level)
        bump_mm = _lerp(5.0, 10.0, t)
        slope_deg = math.degrees(math.atan(_lerp(0.05, 0.35, t)))
        block_cm = _lerp(2.0, 12.0, t)
        wave_cm = _lerp(2.0, 12.0, t)
        return (
            f"Pinned complex row L{level}/{COMPLEX_NUM_ROWS - 1} "
            f"(difficulty {t:.2f}): columns flat | rough ±{bump_mm:.0f} mm | "
            f"slopes ~{slope_deg:.0f}° | blocks {block_cm:.0f} cm | waves {wave_cm:.0f} cm. "
            "Nominal cmds vx*=1.8 m/s, heading*=0, no domain rand."
        )
    if preset == "stairs":
        t = _row_difficulty(STAIRS_NUM_ROWS, level)
        step_cm = _lerp(1.0, 4.0, t)
        stair_deg = math.degrees(math.atan((step_cm / 100.0) / 0.20))
        wave_cm = _lerp(1.5, 5.0, t)
        # Peak slope of A(sin kx + cos ky), λ ≈ 1.8 m, k = 2π/λ.
        wave_deg = math.degrees(math.atan((wave_cm / 100.0) * (2.0 * math.pi / 1.8) * math.sqrt(2.0)))
        return (
            f"Pinned stairs row L{level}/{STAIRS_NUM_ROWS - 1} "
            f"(difficulty {t:.2f}): 2×flat | 2×up | 2×down | 2×waves, "
            f"{step_cm:.1f} cm steps (~{stair_deg:.0f}°), "
            f"waves ±{wave_cm:.1f} cm (~{wave_deg:.0f}°). "
            "Nominal cmds vx*=1.8 m/s, heading*=0, no domain rand."
        )
    raise ValueError(f"Unknown terrain preset {preset!r}.")


def _set_event(events: Any, name: str, value: Any) -> None:
    if hasattr(events, name):
        setattr(events, name, value)


def _disable_domain_rand(env_cfg: Any) -> None:
    events = getattr(env_cfg, "events", None)
    if events is None:
        return
    for name in (
        "randomize_ground_friction",
        "randomize_mass",
        "randomize_com",
        "randomize_material",
        "randomize_actuator_gains",
        "reset_kickstand_lift",
        "kickstand_lift_wrench",
        "reset_random_external_force",
        "random_external_force",
        "eval_impulse_reset",
        "eval_impulse",
    ):
        _set_event(events, name, None)


def _disable_debug_vis(env_cfg: Any) -> None:
    commands = getattr(env_cfg, "commands", None)
    if commands is not None:
        for name in ("velocity", "steering"):
            term = getattr(commands, name, None)
            if term is not None and hasattr(term, "debug_vis"):
                term.debug_vis = False
    terrain = getattr(getattr(env_cfg, "scene", None), "terrain", None)
    if terrain is not None and hasattr(terrain, "debug_vis"):
        terrain.debug_vis = False
    _set_event(getattr(env_cfg, "events", None), "setup_heading_vis", None)


def _set_plane_terrain(env_cfg: Any, friction: float = 1.0) -> None:
    terrain = env_cfg.scene.terrain
    terrain.terrain_type = "plane"
    terrain.terrain_generator = None
    terrain.use_terrain_origins = False
    if getattr(terrain, "env_spacing", None) is None:
        terrain.env_spacing = env_cfg.scene.env_spacing
    mat = terrain.physics_material
    mat.static_friction = friction
    mat.dynamic_friction = friction
    mat.restitution = 0.0
    env_cfg.sim.physics_material = mat


def _disable_obs_noise(env_cfg: Any) -> None:
    observations = getattr(env_cfg, "observations", None)
    if observations is None:
        return
    groups = observations.items() if isinstance(observations, dict) else getattr(observations, "__dict__", {}).items()
    for _, group in groups:
        if group is None or isinstance(group, type):
            continue
        imu = getattr(group, "imu_state", None)
        if imu is not None and hasattr(imu, "noise"):
            imu.noise = None


def _set_velocity_command(env_cfg: Any, vx: tuple[float, float]) -> None:
    env_cfg.commands.velocity.ranges.lin_vel_x = vx
    env_cfg.commands.velocity.rel_standing_envs = 0.0
    env_cfg.commands.velocity.heading_command = False


def _set_heading_command(
    env_cfg: Any,
    heading: tuple[float, float],
    wz_clip: tuple[float, float] = (-0.174, 0.174),
) -> None:
    env_cfg.commands.steering.heading_command = True
    env_cfg.commands.steering.ranges.heading = heading
    # Keep a non-zero yaw-rate clip so the heading P-controller can correct drift.
    env_cfg.commands.steering.ranges.ang_vel_z = wz_clip
    env_cfg.commands.steering.rel_standing_envs = 0.0


def _set_spawn_velocity(env_cfg: Any, vx: tuple[float, float]) -> None:
    params = env_cfg.events.initial_state.params
    params["velocity_range"]["x"] = vx
    params["pose_range"]["yaw"] = (0.0, 0.0)
    params["pose_range"]["roll"] = (-math.radians(5.0), math.radians(5.0))


def _configure_terrain_stage(env_cfg: Any, preset: str, level: int, meta: dict[str, Any]) -> dict[str, Any]:
    _disable_domain_rand(env_cfg)
    _disable_obs_noise(env_cfg)
    _set_heading_command(env_cfg, (0.0, 0.0))
    _set_velocity_command(env_cfg, (_NOMINAL_VX, _NOMINAL_VX))
    _set_spawn_velocity(env_cfg, _NOMINAL_SPAWN_VX)
    apply_terrain_preset(env_cfg, preset, init_level=level)
    eye = TERRAIN_PRESET_META[preset].get("camera_eye")
    viewer = getattr(env_cfg, "viewer", None)
    if viewer is not None and eye is not None:
        viewer.eye = eye
    meta["description"] = terrain_stage_description(preset, level)
    meta["terrain"] = preset
    meta["terrain_level"] = level
    meta["pin_terrain_level"] = level
    return meta


def configure_eval_scenario(env_cfg: Any, mode: str, *, disable_vis: bool = True) -> dict[str, Any]:
    """Mutate ``env_cfg`` in place for one evaluation mode. Returns metadata."""
    stage = parse_terrain_stage(mode)
    if stage is None and mode not in EVAL_BASE_MODES:
        raise ValueError(
            f"Unknown eval mode {mode!r}. Choose one of: {', '.join(EVAL_MODES)} "
            f"or aliases {', '.join(EVAL_SUITE_ALIASES)}"
        )

    env_cfg.curriculum = None
    if disable_vis:
        _disable_debug_vis(env_cfg)

    meta: dict[str, Any] = {
        "mode": mode,
        "trackstand": False,
        "description": "",
    }

    if stage is not None:
        return _configure_terrain_stage(env_cfg, stage[0], stage[1], meta)

    if mode == "training":
        apply_terrain_preset(env_cfg, "complex")
        _set_velocity_command(env_cfg, _TRAIN_VX)
        _set_heading_command(env_cfg, _TRAIN_HEADING)
        _set_spawn_velocity(env_cfg, _TRAIN_SPAWN_VX)
        _set_event(env_cfg.events, "reset_kickstand_lift", None)
        _set_event(env_cfg.events, "kickstand_lift_wrench", None)
        meta["description"] = (
            "Training-like: 5×5 complex terrain, vx∈[1,3], heading ±10°, "
            "mass/CoM/friction/actuator rand on, kickstand off."
        )
        meta["terrain"] = "complex"
        return meta

    # All other modes: clean plane.
    _disable_domain_rand(env_cfg)
    _disable_obs_noise(env_cfg)
    _set_heading_command(env_cfg, (0.0, 0.0))
    _set_velocity_command(env_cfg, (_NOMINAL_VX, _NOMINAL_VX))
    _set_spawn_velocity(env_cfg, _NOMINAL_SPAWN_VX)

    if mode == "nominal":
        _set_plane_terrain(env_cfg, friction=1.0)
        meta["description"] = "Nominal locomotion: flat μ=1.0, no rand, vx*=1.8 m/s, heading*=0."
        meta["terrain"] = "plane"
        return meta

    if mode == "kick":
        _set_plane_terrain(env_cfg, friction=1.0)
        env_cfg.events.eval_impulse_reset = EventTermCfg(
            func=reset_eval_impulse_schedule,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
                "delay_s": 2.0,
                "force_n": 10.0,
                "duration_s": 0.1,
            },
        )
        env_cfg.events.eval_impulse = EventTermCfg(
            func=apply_eval_lateral_impulse,
            mode="interval",
            interval_range_s=(0.02, 0.02),
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
                "delay_s": 2.0,
                "force_n": 10.0,
                "duration_s": 0.1,
            },
        )
        meta["description"] = "Impulse rejection: 10 N lateral on base_link for 0.1 s at t=2 s."
        meta["terrain"] = "plane"
        meta["impulse"] = {"delay_s": 2.0, "force_n": 10.0, "duration_s": 0.1}
        return meta

    if mode == "mass":
        _set_plane_terrain(env_cfg, friction=1.0)
        import isaaclab.envs.mdp as mdp

        env_cfg.events.randomize_mass = EventTermCfg(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "mass_distribution_params": (1.10, 1.10),
            },
        )
        env_cfg.events.randomize_com = EventTermCfg(
            func=mdp.randomize_rigid_body_com,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
            },
        )
        meta["description"] = "Mass/inertia: mass ×1.10 and CoM ±2 cm."
        meta["terrain"] = "plane"
        return meta

    # friction
    _set_plane_terrain(env_cfg, friction=0.4)
    import isaaclab.envs.mdp as mdp

    env_cfg.events.randomize_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.4, 0.4),
            "dynamic_friction_range": (0.4, 0.4),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "make_consistent": True,
        },
    )
    meta["description"] = "Friction drop: ground and tires μ=0.4."
    meta["terrain"] = "plane"
    meta["friction"] = 0.4
    return meta
