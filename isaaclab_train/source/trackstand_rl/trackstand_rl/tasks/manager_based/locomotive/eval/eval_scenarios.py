# Ported from ~/bike_test/CycleRL/source/CycleRL/CycleRL/tasks/manager_based/cyclerl/eval/eval_scenarios.py (imports/paths rewritten for trackstand_rl locomotive).
"""Evaluation-scenario helpers for CycleRL locomotion checkpoints.

This policy family is dynamic balancing / velocity tracking only — no trackstand.
Scenarios:

  nominal     Flat plane, no domain rand, fixed ~1.8 m/s straight. Deployment baseline.
  speed_sweep Flat plane; each env holds a fixed vx* from a linspace ladder. Estimates
              the minimum balancing speed and the smooth-drive (no bang-bang) threshold.
  heading_sweep Flat plane; fixed vx*=1.8; each env holds a fixed heading* on a ladder.
  yaw_bias_sweep Flat plane; each env holds a fixed (vx*, ωz*) on a speed × stick-deg
              grid (vx* from speed_sweep range). Per-speed zero-drift trim; report mean.
  drive_accel_sweep  Flat plane; RC owns hub ω* (policy steers only). Each env holds a
              fixed drive accel (m/s²) while ramping spawn vx → vx*=1.8.
  training    Complex terrain + training-like commands and randomizations (no kickstand).
  kick        Nominal plane + 10 N × 0.1 s lateral impulse at t = 2 s.
  kick_sweep  Flat plane; each env holds a fixed kick force (N) on a linspace ladder.
  mass        Plane, mass ×1.10 and CoM ±2 cm.
  mass_sweep  Flat plane; each env holds a fixed mass scale (CoM fixed) on a ladder.
  com_sweep   Flat plane; mass nominal; each env holds a fixed lateral CoM offset (y).
  mass_com_sweep Flat plane; each env holds a (mass scale, lateral CoM) grid cell.
  friction    Plane, ground μ = 0.4.
  friction_sweep Flat plane; tire μ ladder (plane at μ_min); headline = min balancing μ.

Terrain extremity stages pin every env to one curriculum *row* while still
spreading across columns (terrain types). Stairs and complex stay as separate
grids (4 m vs 10 m cells):

  complex_L0 … complex_L4   easiest → hardest height-field
  stairs_L0  … stairs_L2    shallow → steep (steps and waves)

Suite aliases: ``terrain`` (all 8 stages), ``complex`` (L0–L4), ``stairs`` (L0–L2),
``envelope`` / ``ladder`` / ``end`` (plane ladder modes only — no fixed probes,
no staged terrain). ``full`` = ladder + terrain stages.

Fixed point gates (``nominal``, ``kick``, ``mass``, ``friction``, ``training``)
remain registered for ``--eval_mode`` / ``--modes`` but are **not** in ``end``.
"""

from __future__ import annotations

import math
from typing import Any

from isaaclab.managers import EventTermCfg, SceneEntityCfg

from ..config.terrain_presets import (
    COMPLEX_NUM_ROWS,
    STAIRS_NUM_ROWS,
    TERRAIN_PRESET_META,
    apply_terrain_preset,
    pin_terrain_level,
)
from ..mdp.eval_events import apply_eval_lateral_impulse, reset_eval_impulse_schedule

EVAL_BASE_MODES = (
    "nominal",
    "speed_sweep",
    "heading_sweep",
    "yaw_bias_sweep",
    "drive_accel_sweep",
    "training",
    "kick",
    "kick_sweep",
    "mass",
    "mass_sweep",
    "com_sweep",
    "mass_com_sweep",
    "friction",
    "friction_sweep",
    # Paper Table VI robustness conditions (headline suite).
    "paper_low_speed",
    "paper_normal_speed",
    "paper_no_noise",
    "paper_max_noise",
    "paper_dropout",
    "paper_flat",
    "paper_rough",
    "paper_gravel",
    "paper_slope",
    "paper_steps",
    # Frontier ladders backing CAT / MNT (MSS comes from speed_sweep).
    "paper_roll_sweep",
    "paper_noise_sweep",
)
COMPLEX_STAGE_MODES = tuple(f"complex_L{i}" for i in range(COMPLEX_NUM_ROWS))
STAIRS_STAGE_MODES = tuple(f"stairs_L{i}" for i in range(STAIRS_NUM_ROWS))
TERRAIN_STAGE_MODES = COMPLEX_STAGE_MODES + STAIRS_STAGE_MODES
EVAL_MODES = EVAL_BASE_MODES + TERRAIN_STAGE_MODES

# Plane envelope ladders only (no fixed probes, no terrain).
_LADDER_MODES = (
    "speed_sweep",
    "heading_sweep",
    "yaw_bias_sweep",
    "drive_accel_sweep",
    "kick_sweep",
    "mass_sweep",
    "com_sweep",
    "mass_com_sweep",
    "friction_sweep",
)
_ENVELOPE_MODES = (
    "mass_sweep",
    "com_sweep",
    "mass_com_sweep",
    "friction_sweep",
    "kick_sweep",
)

# Paper Table VI: the ten reported robustness conditions, in table order.
_PAPER_TABLE_VI_MODES = (
    "paper_low_speed",
    "paper_normal_speed",
    "paper_no_noise",
    "paper_max_noise",
    "paper_dropout",
    "paper_flat",
    "paper_rough",
    "paper_gravel",
    "paper_slope",
    "paper_steps",
)
# Ladders that turn Table II's frontier metrics (CAT / MNT / MSS) into numbers.
_PAPER_FRONTIER_MODES = ("paper_roll_sweep", "paper_noise_sweep", "speed_sweep")

EVAL_SUITE_ALIASES: dict[str, tuple[str, ...]] = {
    "terrain": TERRAIN_STAGE_MODES,
    "complex": COMPLEX_STAGE_MODES,
    "stairs": STAIRS_STAGE_MODES,
    "envelope": _ENVELOPE_MODES,
    "ladder": _LADDER_MODES,
    # Paper reproduction suites.
    "paper": _PAPER_TABLE_VI_MODES,
    "paper_frontier": _PAPER_FRONTIER_MODES,
    "paper_full": _PAPER_TABLE_VI_MODES + _PAPER_FRONTIER_MODES,
    # Headline suite: Table VI conditions + the CAT/MNT/MSS ladders.
    "end": _PAPER_TABLE_VI_MODES + _PAPER_FRONTIER_MODES,
    # Legacy plane ladders — diagnostics only, no longer the ranking suite.
    "diagnostics": _LADDER_MODES,
    "full": _PAPER_TABLE_VI_MODES + _PAPER_FRONTIER_MODES + _LADDER_MODES,
}

# Match the 2026-08-13_15-24-02 nice2 training snapshot (params/env.yaml).
_TRAIN_VX = (1.0, 3.0)
_TRAIN_HEADING = (-0.174, 0.174)
_TRAIN_SPAWN_VX = (0.5, 1.5)
_NOMINAL_VX = 1.8
_NOMINAL_SPAWN_VX = (1.5, 1.8)
_PAPER_WHEEL_RADIUS_M = 0.0725
_PAPER_ROLL_INIT_RAD = (-math.radians(10.0), math.radians(10.0))
_PAPER_NOMINAL_ROLL_INIT_RAD = (-math.radians(5.0), math.radians(5.0))
_DRIVE_ACCEL_MIN = 0.25
_DRIVE_ACCEL_MAX = 8.0
_DRIVE_ACCEL_SPAWN_VX = 0.5
_YAW_BIAS_MIN_DEG = -5.0
_YAW_BIAS_MAX_DEG = 5.0
_YAW_RATE_SCALE = 0.45
_YAW_STICK_FULL_SCALE_DEG = 10.0
_MASS_SCALE_MIN = 1.0
_MASS_SCALE_MAX = 2.5
# CoM / mass×CoM ladders tightened around the known tip (~1.2–1.7 cm / ×1.10).
_COM_OFFSET_MIN_M = -0.02
_COM_OFFSET_MAX_M = 0.02
_MASS_COM_SCALE_MIN = 1.0
_MASS_COM_SCALE_MAX = 1.15
_MASS_COM_OFFSET_MIN_M = -0.02
_MASS_COM_OFFSET_MAX_M = 0.02
# Joint-grid headline probe (was 2 cm — past tip for every candidate → always 0%).
_MASS_COM_PROBE_COM_ABS_M = 0.005
_MASS_COM_PROBE_MASS_SCALE = 1.10
_FRICTION_MU_MIN = 0.05
_FRICTION_MU_MAX = 1.0
_KICK_FORCE_MIN = 0.0
_KICK_FORCE_MAX = 40.0
_KICK_DELAY_S = 2.0
_KICK_DURATION_S = 0.1
_KICK_FORCE_N = 10.0

# --- Paper Table VI conditions, Froude-scaled to new_bike -------------------
# new_bike wheelbase 0.283 m vs the paper platform's 1.100 m (Table VIII), so
# speeds scale by sqrt(0.283 / 1.100) = 0.507. The paper's [1, 3] and [3, 5] m/s
# bands therefore map to [0.51, 1.52] and [1.52, 2.54] m/s here.
PAPER_FROUDE_SPEED_SCALE = 0.507
_PAPER_LOW_SPEED = (0.51, 1.52)
_PAPER_NORMAL_SPEED = (1.52, 2.54)
# Table I observation noise: ρ ~ U([0.01, 0.20]); Table VI's "maximum noise
# level" row is the 20% amplitude end of that range.
_PAPER_NOISE_MAX_RHO = 0.20
_PAPER_DROPOUT_PROB = 0.10
_PAPER_NOISE_SWEEP_RHO = (0.0, 0.5)
# CAT ladder spans past the 45 deg crash termination so the frontier is bracketed.
_PAPER_ROLL_SWEEP_DEG = (-40.0, 40.0)
# "Gravel surface — reduced traction" (Table VI). Between the μ=0.4 friction
# probe and the μ=1.0 nominal plane.
_PAPER_GRAVEL_MU = 0.45
# Table VI terrain rows mapped onto the repo's generators: rough/slope are
# low/mid rows of the 5x5 complex grid, gentle steps is the shallowest stairs row.
_PAPER_ROUGH_LEVEL = 1
_PAPER_SLOPE_LEVEL = 2
_PAPER_STEPS_LEVEL = 0
_PAPER_HEADING = (-math.radians(10.0), math.radians(10.0))
# Mid of the Froude-scaled [1, 5] m/s band — used by the non-speed Table VI rows.
_PAPER_EVAL_VX = 1.4
# Table I per-signal s_max: [gravity_y, wx, wy, wz, speed, steer, v*, ωz*].
_PAPER_OBS_SMAX = (0.7, 5.0, 5.0, 5.0, 2.5, 0.35, 2.5, 0.5)


def _is_paper_env(env_cfg: Any) -> bool:
    """True for CycleRL-NewBike-Paper-v0 (RC command drive / paper joint reset)."""
    events = getattr(env_cfg, "events", None)
    reset_js = getattr(events, "reset_joint_state", None)
    if reset_js is not None:
        func = getattr(reset_js, "func", None)
        name = getattr(func, "__name__", "") or str(func)
        if "reset_paper_joint_state" in name:
            return True
    drive = getattr(getattr(env_cfg, "actions", None), "drive", None)
    if drive is not None:
        cls = getattr(drive, "class_type", type(drive))
        name = getattr(cls, "__name__", str(cls))
        if "CommandDrive" in name or "SlewLimitedCommandDrive" in name:
            return True
    return False


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
        "randomize_imu_noise",
        "reset_kickstand_lift",
        "kickstand_lift_wrench",
        "reset_random_external_force",
        "random_external_force",
        "eval_impulse_reset",
        "eval_impulse",
        "ramp_velocity_command",
        "invalidate_velocity_ramp",
        "randomize_drive_accel",
    ):
        _set_event(events, name, None)


def _hold_commands_for_episode(env_cfg: Any) -> None:
    """Stop 3–5 s command resampling so nominal vx*/heading* stay fixed."""
    commands = getattr(env_cfg, "commands", None)
    if commands is None:
        return
    for name in ("velocity", "steering"):
        term = getattr(commands, name, None)
        if term is not None and hasattr(term, "resampling_time_range"):
            term.resampling_time_range = (1.0e6, 1.0e6)


def _paper_drive_joint_name(env_cfg: Any) -> str:
    """Drive joint from the env (cylinder CAD uses ``rear_wheel_joint``)."""
    events = getattr(env_cfg, "events", None)
    reset_js = getattr(events, "reset_joint_state", None) if events is not None else None
    params = getattr(reset_js, "params", None) if reset_js is not None else None
    if isinstance(params, dict) and params.get("drive_joint_name"):
        return str(params["drive_joint_name"])
    drive = getattr(getattr(env_cfg, "actions", None), "drive", None)
    names = getattr(drive, "joint_names", None) or []
    if names:
        return str(names[0])
    return "rearwheel_joint"


def _paper_wheel_radius(env_cfg: Any) -> float:
    """Hub radius implied by Paper drive scale ``a1=±1 → ±3 m/s``."""
    drive = getattr(getattr(env_cfg, "actions", None), "drive", None)
    explicit = getattr(drive, "wheel_radius_m", None)
    if isinstance(explicit, (int, float)) and explicit > 0.0:
        return float(explicit)
    scale = getattr(drive, "scale", None)
    if isinstance(scale, (int, float)) and float(scale) > 1.0:
        return 3.0 / float(scale)
    return _PAPER_WHEEL_RADIUS_M


def _align_paper_eval_spawn(env_cfg: Any, vx: float) -> None:
    """Make Paper resets match commanded speed (avoids instant tip-overs).

    Training used to randomize hub ω independently of body vx. Paper-faithful
    training now couples hub ω to root vx, so eval only needs to pin body vx
    and keep the couple flag on.
    """
    events = getattr(env_cfg, "events", None)
    if events is None:
        return

    reset_js = getattr(events, "reset_joint_state", None)
    if reset_js is not None and getattr(reset_js, "params", None) is not None:
        reset_js.params["couple_drive_to_root_vx"] = True
        reset_js.params["steer_deg_range"] = (-2.0, 2.0)

    params = getattr(getattr(events, "initial_state", None), "params", None)
    if params is not None:
        params["velocity_range"]["x"] = _NOMINAL_SPAWN_VX
        params["pose_range"]["yaw"] = (0.0, 0.0)
        # Softer than training ±10° — matches the 15:50 nominal eval that reached ~84%.
        params["pose_range"]["roll"] = _PAPER_NOMINAL_ROLL_INIT_RAD

    if getattr(events, "match_spawn_velocity", None) is None:
        from ..mdp.eval_events import match_spawn_to_velocity_command

        step_dt = float(getattr(env_cfg, "step_dt", 0.02) or 0.02)
        events.match_spawn_velocity = EventTermCfg(
            func=match_spawn_to_velocity_command,
            mode="interval",
            interval_range_s=(step_dt, step_dt),
            is_global_time=True,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "velocity",
                "drive_joint_name": _paper_drive_joint_name(env_cfg),
                "wheel_radius_m": _paper_wheel_radius(env_cfg),
                "drive_joint_sign": 1.0,
                "velocity_noise": 0.0,
            },
        )
    else:
        events.match_spawn_velocity.params["velocity_noise"] = 0.0


def _ensure_rc_command_drive(
    env_cfg: Any,
    *,
    accel_ms2: float = 1.0,
    accel_ms2_range: tuple[float, float] | None = None,
) -> None:
    """Force hub ω* from vx* (policy a1 unused). Used by drive_accel_sweep."""
    from ..mdp.actions import SlewLimitedCommandDriveActionCfg

    current = getattr(getattr(env_cfg, "actions", None), "drive", None)
    joint_names = list(getattr(current, "joint_names", None) or ["rearwheel_joint"])
    asset_name = str(getattr(current, "asset_name", None) or "robot")
    wheel_r = float(getattr(current, "wheel_radius_m", _PAPER_WHEEL_RADIUS_M) or _PAPER_WHEEL_RADIUS_M)
    sign = float(getattr(current, "drive_joint_sign", 1.0) or 1.0)
    clip = float(getattr(current, "velocity_clip_rad_s", 45.0) or 45.0)
    env_cfg.actions.drive = SlewLimitedCommandDriveActionCfg(
        asset_name=asset_name,
        joint_names=joint_names,
        scale=1.0,
        command_name=str(getattr(current, "command_name", None) or "velocity"),
        wheel_radius_m=wheel_r,
        drive_joint_sign=sign,
        velocity_clip_rad_s=clip,
        accel_ms2=accel_ms2,
        accel_ms2_range=accel_ms2_range,
    )


def _set_paper_spawn_speed(env_cfg: Any, vx: float) -> None:
    """Body vx and hub ω at spawn (no match-to-command overlay)."""
    events = getattr(env_cfg, "events", None)
    if events is None:
        return
    omega = float(vx) / _PAPER_WHEEL_RADIUS_M
    reset_js = getattr(events, "reset_joint_state", None)
    if reset_js is not None and getattr(reset_js, "params", None) is not None:
        reset_js.params["steer_deg_range"] = (-2.0, 2.0)
        reset_js.params["drive_vel_range"] = (omega, omega)
    params = getattr(getattr(events, "initial_state", None), "params", None)
    if params is not None:
        params["velocity_range"]["x"] = (vx, vx)
        params["pose_range"]["yaw"] = (0.0, 0.0)
        params["pose_range"]["roll"] = _PAPER_NOMINAL_ROLL_INIT_RAD
    _set_event(events, "match_spawn_velocity", None)


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
        for term_name in ("imu_state", "commands"):
            term = getattr(group, term_name, None)
            if term is not None and hasattr(term, "noise"):
                term.noise = None
            if term is not None and getattr(term, "params", None) is not None:
                term.params.pop("dropout_prob", None)


def _read_obs_smax(env_cfg: Any) -> Any:
    """Capture the env's per-signal noise scale before domain rand is stripped."""
    events = getattr(env_cfg, "events", None)
    term = getattr(events, "randomize_imu_noise", None)
    params = getattr(term, "params", None) if term is not None else None
    if not params:
        return None
    return params.get("imu_smax")


def _set_obs_noise_rho(
    env_cfg: Any, rho: float, smax: Any, rho_ladder: tuple[float, float] | None = None
) -> bool:
    """Re-arm Table I observation noise at a fixed ρ (0 leaves noise off).

    ``rho_ladder`` instead pins each env to a rung of a linspace ladder (MNT sweep).
    """
    events = getattr(env_cfg, "events", None)
    if events is None or smax is None:
        return False
    if rho <= 0.0:
        _set_event(events, "randomize_imu_noise", None)
        return False

    from isaaclab.utils.noise import GaussianNoiseCfg

    from ..mdp.eval_events import randomize_imu_noise_fraction

    # _disable_obs_noise cleared the term-level cfgs; the event needs them back.
    observations = getattr(env_cfg, "observations", None)
    groups = (
        observations.values()
        if isinstance(observations, dict)
        else getattr(observations, "__dict__", {}).values()
    )
    for group in groups:
        if group is None or isinstance(group, type):
            continue
        for term_name in ("imu_state", "commands"):
            obs_term = getattr(group, term_name, None)
            if obs_term is not None and getattr(obs_term, "noise", None) is None:
                obs_term.noise = GaussianNoiseCfg(mean=0.0, std=0.0)

    events.randomize_imu_noise = EventTermCfg(
        func=randomize_imu_noise_fraction,
        mode="reset",
        params={"rho_range": (rho, rho), "imu_smax": smax, "rho_ladder": rho_ladder},
    )
    return True


def _set_obs_dropout(env_cfg: Any, prob: float) -> bool:
    """Hold stale observations with probability ``prob`` (Table VI dropouts)."""
    observations = getattr(env_cfg, "observations", None)
    if observations is None:
        return False
    groups = (
        observations.values()
        if isinstance(observations, dict)
        else getattr(observations, "__dict__", {}).values()
    )
    applied = False
    for group in groups:
        if group is None or isinstance(group, type):
            continue
        imu = getattr(group, "imu_state", None)
        if imu is not None and getattr(imu, "params", None) is not None:
            imu.params["dropout_prob"] = float(prob)
            applied = True
    return applied


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
    _hold_commands_for_episode(env_cfg)
    if _is_paper_env(env_cfg):
        _align_paper_eval_spawn(env_cfg, _NOMINAL_VX)
    else:
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


def _couple_paper_hub(env_cfg: Any) -> None:
    """Pin hub ω to the body vx written by ``initial_state`` (no Table I slip)."""
    reset_js = getattr(getattr(env_cfg, "events", None), "reset_joint_state", None)
    params = getattr(reset_js, "params", None) if reset_js is not None else None
    if params is not None:
        params["couple_drive_to_root_vx"] = True


def _paper_spawn_vx(env_cfg: Any, vx: tuple[float, float]) -> None:
    params = env_cfg.events.initial_state.params
    params["velocity_range"]["x"] = (float(vx[0]), float(vx[1]))
    params["pose_range"]["yaw"] = (0.0, 0.0)
    params["pose_range"]["roll"] = _PAPER_ROLL_INIT_RAD


def _configure_paper_table_vi(
    env_cfg: Any,
    mode: str,
    meta: dict[str, Any],
    obs_smax: Any,
) -> dict[str, Any]:
    """Table VI robustness conditions + CAT/MNT ladders (paper reproduction)."""
    paper = _is_paper_env(env_cfg)
    _disable_domain_rand(env_cfg)
    _disable_obs_noise(env_cfg)
    _couple_paper_hub(env_cfg)
    meta["paper_env"] = paper
    meta["suite"] = "paper"
    smax = obs_smax if obs_smax is not None else _PAPER_OBS_SMAX

    def _speed_band(vx: tuple[float, float], label: str) -> dict[str, Any]:
        _set_plane_terrain(env_cfg, friction=1.0)
        _set_velocity_command(env_cfg, vx)
        _set_heading_command(env_cfg, _PAPER_HEADING)
        env_cfg.commands.velocity.resampling_time_range = (3.0, 5.0)
        env_cfg.commands.steering.resampling_time_range = (3.0, 5.0)
        _paper_spawn_vx(env_cfg, vx)
        meta["description"] = (
            f"Table VI {label}: vx* ∈ [{vx[0]:.2f}, {vx[1]:.2f}] m/s "
            "(Froude-scaled paper band), heading ±10°, flat μ=1, no DR, hub coupled."
        )
        meta["terrain"] = "plane"
        meta["vx_band"] = vx
        return meta

    if mode == "paper_low_speed":
        return _speed_band(_PAPER_LOW_SPEED, "low speed")
    if mode == "paper_normal_speed":
        return _speed_band(_PAPER_NORMAL_SPEED, "normal speed")

    _hold_commands_for_episode(env_cfg)
    _set_heading_command(env_cfg, (0.0, 0.0))
    _set_velocity_command(env_cfg, (_PAPER_EVAL_VX, _PAPER_EVAL_VX))
    _paper_spawn_vx(env_cfg, (_PAPER_EVAL_VX * 0.9, _PAPER_EVAL_VX))

    if mode in ("paper_no_noise", "paper_flat"):
        _set_plane_terrain(env_cfg, friction=1.0)
        kind = "no noise" if mode == "paper_no_noise" else "flat plane"
        meta["description"] = (
            f"Table VI {kind}: vx*={_PAPER_EVAL_VX:g} m/s, heading*=0, μ=1, "
            "no DR, hub coupled."
        )
        meta["terrain"] = "plane"
        return meta

    if mode == "paper_max_noise":
        _set_plane_terrain(env_cfg, friction=1.0)
        _set_obs_noise_rho(env_cfg, _PAPER_NOISE_MAX_RHO, smax)
        meta["description"] = (
            f"Table VI max noise: ρ={_PAPER_NOISE_MAX_RHO:.2f} × per-signal s_max, "
            f"vx*={_PAPER_EVAL_VX:g} m/s, flat μ=1."
        )
        meta["terrain"] = "plane"
        meta["noise_rho"] = _PAPER_NOISE_MAX_RHO
        return meta

    if mode == "paper_dropout":
        _set_plane_terrain(env_cfg, friction=1.0)
        _set_obs_dropout(env_cfg, _PAPER_DROPOUT_PROB)
        meta["description"] = (
            f"Table VI intermittent dropouts: hold last IMU packet with "
            f"p={_PAPER_DROPOUT_PROB:.0%}, vx*={_PAPER_EVAL_VX:g} m/s, flat μ=1."
        )
        meta["terrain"] = "plane"
        meta["dropout_prob"] = _PAPER_DROPOUT_PROB
        return meta

    if mode == "paper_gravel":
        import isaaclab.envs.mdp as mdp

        _set_plane_terrain(env_cfg, friction=_PAPER_GRAVEL_MU)
        env_cfg.events.randomize_material = EventTermCfg(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "static_friction_range": (_PAPER_GRAVEL_MU, _PAPER_GRAVEL_MU),
                "dynamic_friction_range": (_PAPER_GRAVEL_MU, _PAPER_GRAVEL_MU),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 1,
                "make_consistent": True,
            },
        )
        meta["description"] = (
            f"Table VI gravel: ground and tires μ={_PAPER_GRAVEL_MU:g}, "
            f"vx*={_PAPER_EVAL_VX:g} m/s."
        )
        meta["terrain"] = "plane"
        meta["friction"] = _PAPER_GRAVEL_MU
        return meta

    if mode == "paper_rough":
        apply_terrain_preset(env_cfg, "complex", init_level=_PAPER_ROUGH_LEVEL)
        meta["description"] = (
            f"Table VI rough pavement: complex row L{_PAPER_ROUGH_LEVEL}, "
            f"vx*={_PAPER_EVAL_VX:g} m/s, no DR."
        )
        meta["terrain"] = "complex"
        meta["pin_terrain_level"] = _PAPER_ROUGH_LEVEL
        return meta

    if mode == "paper_slope":
        apply_terrain_preset(env_cfg, "complex", init_level=_PAPER_SLOPE_LEVEL)
        meta["description"] = (
            f"Table VI slope terrain: complex row L{_PAPER_SLOPE_LEVEL}, "
            f"vx*={_PAPER_EVAL_VX:g} m/s, no DR."
        )
        meta["terrain"] = "complex"
        meta["pin_terrain_level"] = _PAPER_SLOPE_LEVEL
        return meta

    if mode == "paper_steps":
        apply_terrain_preset(env_cfg, "stairs", init_level=_PAPER_STEPS_LEVEL)
        meta["description"] = (
            f"Table VI gentle steps: stairs row L{_PAPER_STEPS_LEVEL}, "
            f"vx*={_PAPER_EVAL_VX:g} m/s, no DR."
        )
        meta["terrain"] = "stairs"
        meta["pin_terrain_level"] = _PAPER_STEPS_LEVEL
        return meta

    if mode == "paper_roll_sweep":
        from ..mdp.eval_events import assign_eval_roll_ladder

        _set_plane_terrain(env_cfg, friction=1.0)
        lo, hi = _PAPER_ROLL_SWEEP_DEG
        env_cfg.events.assign_eval_roll_ladder = EventTermCfg(
            func=assign_eval_roll_ladder,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "roll_min_deg": lo,
                "roll_max_deg": hi,
            },
        )
        meta["description"] = (
            f"CAT ladder: spawn roll [{lo:g}, {hi:g}]°, vx*={_PAPER_EVAL_VX:g} m/s, "
            "flat μ=1, no DR."
        )
        meta["terrain"] = "plane"
        meta["roll_sweep_min_deg"] = lo
        meta["roll_sweep_max_deg"] = hi
        return meta

    if mode == "paper_noise_sweep":
        _set_plane_terrain(env_cfg, friction=1.0)
        rho_lo, rho_hi = _PAPER_NOISE_SWEEP_RHO
        # The noise event builds the ladder itself. A separate ladder event was
        # appended after randomize_imu_noise in reset order, so every first episode
        # ran at ρ≈0 while being labelled with its ladder rung (MNT read 0.5).
        _set_obs_noise_rho(env_cfg, max(rho_lo, 1e-6), smax, rho_ladder=(rho_lo, rho_hi))
        meta["description"] = (
            f"MNT ladder: ρ ∈ [{rho_lo:g}, {rho_hi:g}] × per-signal s_max, "
            f"vx*={_PAPER_EVAL_VX:g} m/s, flat μ=1."
        )
        meta["terrain"] = "plane"
        meta["noise_rho_min"] = rho_lo
        meta["noise_rho_max"] = rho_hi
        return meta

    raise ValueError(f"Unhandled paper eval mode {mode!r}")


def configure_eval_scenario(
    env_cfg: Any,
    mode: str,
    *,
    disable_vis: bool = True,
    speed_sweep_min: float = 0.3,
    speed_sweep_max: float = 2.5,
    heading_sweep_min_deg: float = -30.0,
    heading_sweep_max_deg: float = 30.0,
    yaw_bias_min_deg: float = _YAW_BIAS_MIN_DEG,
    yaw_bias_max_deg: float = _YAW_BIAS_MAX_DEG,
    yaw_rate_scale: float = _YAW_RATE_SCALE,
    yaw_stick_full_scale_deg: float = _YAW_STICK_FULL_SCALE_DEG,
    drive_accel_min: float = _DRIVE_ACCEL_MIN,
    drive_accel_max: float = _DRIVE_ACCEL_MAX,
    drive_accel_spawn_vx: float = _DRIVE_ACCEL_SPAWN_VX,
    mass_scale_min: float = _MASS_SCALE_MIN,
    mass_scale_max: float = _MASS_SCALE_MAX,
    com_offset_min_m: float = _COM_OFFSET_MIN_M,
    com_offset_max_m: float = _COM_OFFSET_MAX_M,
    mass_com_scale_min: float = _MASS_COM_SCALE_MIN,
    mass_com_scale_max: float = _MASS_COM_SCALE_MAX,
    mass_com_offset_min_m: float = _MASS_COM_OFFSET_MIN_M,
    mass_com_offset_max_m: float = _MASS_COM_OFFSET_MAX_M,
    friction_mu_min: float = _FRICTION_MU_MIN,
    friction_mu_max: float = _FRICTION_MU_MAX,
    kick_force_min: float = _KICK_FORCE_MIN,
    kick_force_max: float = _KICK_FORCE_MAX,
) -> dict[str, Any]:
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
    # Read before _disable_domain_rand strips the event the paper noise modes re-arm.
    obs_smax = _read_obs_smax(env_cfg)

    if stage is not None:
        return _configure_terrain_stage(env_cfg, stage[0], stage[1], meta)

    if mode.startswith("paper_"):
        return _configure_paper_table_vi(env_cfg, mode, meta, obs_smax)

    if mode == "training":
        apply_terrain_preset(env_cfg, "complex")
        _set_velocity_command(env_cfg, _TRAIN_VX)
        _set_heading_command(env_cfg, _TRAIN_HEADING)
        if _is_paper_env(env_cfg):
            # Keep Paper spawn/joint randomization; only drop kickstand if present.
            params = env_cfg.events.initial_state.params
            params["velocity_range"]["x"] = _TRAIN_SPAWN_VX
            params["pose_range"]["yaw"] = (0.0, 0.0)
            params["pose_range"]["roll"] = _PAPER_ROLL_INIT_RAD
            meta["description"] = (
                "Training-like Paper: 5×5 complex terrain, vx∈[1,3], heading ±10°, "
                "Paper domain rand on (mass/CoM/friction/IMU), kickstand off."
            )
        else:
            _set_spawn_velocity(env_cfg, _TRAIN_SPAWN_VX)
            meta["description"] = (
                "Training-like: 5×5 complex terrain, vx∈[1,3], heading ±10°, "
                "mass/CoM/friction/actuator rand on, kickstand off."
            )
        _set_event(env_cfg.events, "reset_kickstand_lift", None)
        _set_event(env_cfg.events, "kickstand_lift_wrench", None)
        meta["terrain"] = "complex"
        meta["paper_env"] = _is_paper_env(env_cfg)
        return meta

    # All other modes: clean plane.
    paper = _is_paper_env(env_cfg)
    _disable_domain_rand(env_cfg)
    _disable_obs_noise(env_cfg)
    _set_heading_command(env_cfg, (0.0, 0.0))
    _set_velocity_command(env_cfg, (_NOMINAL_VX, _NOMINAL_VX))
    _hold_commands_for_episode(env_cfg)
    if paper:
        _align_paper_eval_spawn(env_cfg, _NOMINAL_VX)
    else:
        _set_spawn_velocity(env_cfg, _NOMINAL_SPAWN_VX)

    if mode == "nominal":
        _set_plane_terrain(env_cfg, friction=1.0)
        if paper:
            meta["description"] = (
                "Nominal Paper: flat μ=1.0, no rand, vx*=1.8 m/s, heading*=0, "
                "spawn hub matched to vx*, body vx∈[1.5,1.8], steer init ±2°, roll ±5°."
            )
        else:
            meta["description"] = "Nominal locomotion: flat μ=1.0, no rand, vx*=1.8 m/s, heading*=0."
        meta["terrain"] = "plane"
        meta["paper_env"] = paper
        return meta

    if mode == "speed_sweep":
        from ..mdp.eval_events import assign_eval_speed_ladder

        _set_plane_terrain(env_cfg, friction=1.0)
        _set_velocity_command(env_cfg, (speed_sweep_min, speed_sweep_max))
        _set_heading_command(env_cfg, (0.0, 0.0))
        # Hold vx* for the full episode; ladder event overrides the uniform draw.
        env_cfg.commands.velocity.resampling_time_range = (1.0e6, 1.0e6)
        env_cfg.commands.steering.resampling_time_range = (1.0e6, 1.0e6)
        if hasattr(env_cfg.terminations, "velocity_goal_timeout"):
            env_cfg.terminations.velocity_goal_timeout = None
        params = env_cfg.events.initial_state.params
        params["velocity_range"]["x"] = (speed_sweep_min, speed_sweep_max)
        if paper:
            params["pose_range"]["roll"] = _PAPER_ROLL_INIT_RAD
            reset_js = getattr(env_cfg.events, "reset_joint_state", None)
            if reset_js is not None and getattr(reset_js, "params", None) is not None:
                # Mid-ladder ω; per-env ladder then corrected by match_spawn.
                mid = 0.5 * (speed_sweep_min + speed_sweep_max)
                omega = mid / _paper_wheel_radius(env_cfg)
                reset_js.params["steer_deg_range"] = (-2.0, 2.0)
                reset_js.params["drive_vel_range"] = (omega * 0.5, omega * 1.5)
            if getattr(env_cfg.events, "match_spawn_velocity", None) is None:
                _align_paper_eval_spawn(env_cfg, mid)
            else:
                env_cfg.events.match_spawn_velocity.params["velocity_noise"] = 0.0
        else:
            params["pose_range"]["roll"] = (-math.radians(5.0), math.radians(5.0))
        if hasattr(env_cfg.events, "match_spawn_velocity") and env_cfg.events.match_spawn_velocity is not None:
            env_cfg.events.match_spawn_velocity.params["velocity_noise"] = 0.0
        env_cfg.events.assign_eval_speed_ladder = EventTermCfg(
            func=assign_eval_speed_ladder,
            mode="interval",
            interval_range_s=(0.02, 0.02),
            params={
                "velocity_command_name": "velocity",
                "heading_command_name": "steering",
                "v_min": speed_sweep_min,
                "v_max": speed_sweep_max,
            },
        )
        meta["description"] = (
            f"Speed sweep: flat μ=1.0, vx* ladder [{speed_sweep_min}, {speed_sweep_max}] m/s "
            "(one fixed speed per env), heading*=0, no domain rand."
        )
        meta["terrain"] = "plane"
        meta["speed_sweep_min"] = speed_sweep_min
        meta["speed_sweep_max"] = speed_sweep_max
        meta["paper_env"] = paper
        return meta
    if mode == "heading_sweep":
        from ..mdp.eval_events import assign_eval_heading_ladder

        _set_plane_terrain(env_cfg, friction=1.0)
        h_min = math.radians(float(heading_sweep_min_deg))
        h_max = math.radians(float(heading_sweep_max_deg))
        _set_velocity_command(env_cfg, (_NOMINAL_VX, _NOMINAL_VX))
        _set_heading_command(env_cfg, (h_min, h_max))
        env_cfg.commands.velocity.resampling_time_range = (1.0e6, 1.0e6)
        env_cfg.commands.steering.resampling_time_range = (1.0e6, 1.0e6)
        if paper:
            _align_paper_eval_spawn(env_cfg, _NOMINAL_VX)
        else:
            _set_spawn_velocity(env_cfg, _NOMINAL_SPAWN_VX)
        if hasattr(env_cfg.events, "match_spawn_velocity") and env_cfg.events.match_spawn_velocity is not None:
            env_cfg.events.match_spawn_velocity.params["velocity_noise"] = 0.0
        env_cfg.events.assign_eval_heading_ladder = EventTermCfg(
            func=assign_eval_heading_ladder,
            mode="interval",
            interval_range_s=(0.02, 0.02),
            params={
                "velocity_command_name": "velocity",
                "heading_command_name": "steering",
                "heading_min_rad": h_min,
                "heading_max_rad": h_max,
                "vx_ms": _NOMINAL_VX,
            },
        )
        meta["description"] = (
            f"Heading sweep: flat μ=1.0, vx*={_NOMINAL_VX:g} m/s, "
            f"heading* ladder [{heading_sweep_min_deg:g}, {heading_sweep_max_deg:g}]° "
            "(always includes θ*=0 straight), balance+heading-RMSE gates, no domain rand."
        )
        meta["terrain"] = "plane"
        meta["heading_sweep_min_deg"] = float(heading_sweep_min_deg)
        meta["heading_sweep_max_deg"] = float(heading_sweep_max_deg)
        meta["paper_env"] = paper
        return meta
    if mode == "yaw_bias_sweep":
        from ..mdp.eval_events import assign_eval_yaw_bias_ladder

        _set_plane_terrain(env_cfg, friction=1.0)
        b_min = float(yaw_bias_min_deg)
        b_max = float(yaw_bias_max_deg)
        v_min = float(speed_sweep_min)
        v_max = float(speed_sweep_max)
        rate_scale = float(yaw_rate_scale)
        stick_full = float(yaw_stick_full_scale_deg)
        _set_velocity_command(env_cfg, (v_min, v_max))
        # Direct ωz* path (is_heading_env=False); widen ang_vel_z clip to the teleop scale.
        _set_heading_command(
            env_cfg,
            (0.0, 0.0),
            wz_clip=(-rate_scale, rate_scale),
        )
        env_cfg.commands.velocity.resampling_time_range = (1.0e6, 1.0e6)
        env_cfg.commands.steering.resampling_time_range = (1.0e6, 1.0e6)
        params = env_cfg.events.initial_state.params
        params["velocity_range"]["x"] = (v_min, v_max)
        if paper:
            mid = 0.5 * (v_min + v_max)
            _align_paper_eval_spawn(env_cfg, mid)
            params["velocity_range"]["x"] = (v_min, v_max)
            params["pose_range"]["roll"] = _PAPER_ROLL_INIT_RAD
            reset_js = getattr(env_cfg.events, "reset_joint_state", None)
            if reset_js is not None and getattr(reset_js, "params", None) is not None:
                reset_js.params["steer_deg_range"] = (-2.0, 2.0)
                reset_js.params["drive_vel_range"] = (
                    v_min / _PAPER_WHEEL_RADIUS_M,
                    v_max / _PAPER_WHEEL_RADIUS_M,
                )
        else:
            _set_spawn_velocity(env_cfg, (v_min, v_max))
            params["pose_range"]["roll"] = (-math.radians(5.0), math.radians(5.0))
        if hasattr(env_cfg.events, "match_spawn_velocity") and env_cfg.events.match_spawn_velocity is not None:
            env_cfg.events.match_spawn_velocity.params["velocity_noise"] = 0.0
        env_cfg.events.assign_eval_yaw_bias_ladder = EventTermCfg(
            func=assign_eval_yaw_bias_ladder,
            mode="interval",
            interval_range_s=(0.02, 0.02),
            params={
                "velocity_command_name": "velocity",
                "heading_command_name": "steering",
                "bias_min_deg": b_min,
                "bias_max_deg": b_max,
                "yaw_rate_scale": rate_scale,
                "stick_full_scale_deg": stick_full,
                "vx_ms": _NOMINAL_VX,
                "vx_min": v_min,
                "vx_max": v_max,
            },
        )
        meta["description"] = (
            f"Yaw bias sweep: flat μ=1.0, vx*×ωz* grid "
            f"vx*∈[{v_min:g}, {v_max:g}] m/s × stick-deg [{b_min:g}, {b_max:g}]° "
            f"(ωz*=±{rate_scale:g} at ±{stick_full:g}°); mean trim across speeds; "
            "is_heading_env=False, no domain rand."
        )
        meta["terrain"] = "plane"
        meta["yaw_bias_min_deg"] = b_min
        meta["yaw_bias_max_deg"] = b_max
        meta["yaw_bias_vx_min"] = v_min
        meta["yaw_bias_vx_max"] = v_max
        meta["yaw_rate_scale"] = rate_scale
        meta["yaw_stick_full_scale_deg"] = stick_full
        meta["paper_env"] = paper
        return meta
    if mode == "drive_accel_sweep":
        from ..mdp.eval_events import assign_eval_drive_accel_ladder

        _set_plane_terrain(env_cfg, friction=1.0)
        a_min = float(drive_accel_min)
        a_max = float(drive_accel_max)
        spawn_vx = float(drive_accel_spawn_vx)
        _ensure_rc_command_drive(
            env_cfg,
            accel_ms2=0.5 * (a_min + a_max),
            accel_ms2_range=(a_min, a_max),
        )
        _set_paper_spawn_speed(env_cfg, spawn_vx) if paper else _set_spawn_velocity(
            env_cfg, (spawn_vx, spawn_vx)
        )
        if not paper:
            params = env_cfg.events.initial_state.params
            params["velocity_range"]["x"] = (spawn_vx, spawn_vx)
            _set_event(env_cfg.events, "match_spawn_velocity", None)
        env_cfg.events.assign_eval_drive_accel_ladder = EventTermCfg(
            func=assign_eval_drive_accel_ladder,
            mode="interval",
            interval_range_s=(0.02, 0.02),
            params={
                "drive_action_name": "drive",
                "a_min": a_min,
                "a_max": a_max,
            },
        )
        meta["description"] = (
            f"Drive accel sweep: RC hub (policy steers only), spawn {spawn_vx:g}→vx*={_NOMINAL_VX:g} m/s, "
            f"accel ladder [{a_min:g}, {a_max:g}] m/s², heading*=0, flat μ=1.0."
        )
        meta["terrain"] = "plane"
        meta["paper_env"] = paper
        meta["force_rc_drive"] = True
        meta["drive_accel_min"] = a_min
        meta["drive_accel_max"] = a_max
        meta["drive_accel_spawn_vx"] = spawn_vx
        return meta
    if mode == "kick":
        _set_plane_terrain(env_cfg, friction=1.0)
        env_cfg.events.eval_impulse_reset = EventTermCfg(
            func=reset_eval_impulse_schedule,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
                "delay_s": _KICK_DELAY_S,
                "force_n": _KICK_FORCE_N,
                "duration_s": _KICK_DURATION_S,
            },
        )
        env_cfg.events.eval_impulse = EventTermCfg(
            func=apply_eval_lateral_impulse,
            mode="interval",
            interval_range_s=(0.02, 0.02),
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
                "delay_s": _KICK_DELAY_S,
                "force_n": _KICK_FORCE_N,
                "duration_s": _KICK_DURATION_S,
            },
        )
        meta["description"] = "Impulse rejection: 10 N lateral on base_link for 0.1 s at t=2 s."
        meta["terrain"] = "plane"
        meta["impulse"] = {"delay_s": _KICK_DELAY_S, "force_n": _KICK_FORCE_N, "duration_s": _KICK_DURATION_S}
        return meta

    if mode == "kick_sweep":
        _set_plane_terrain(env_cfg, friction=1.0)
        f_min = float(kick_force_min)
        f_max = float(kick_force_max)
        env_cfg.events.eval_impulse_reset = EventTermCfg(
            func=reset_eval_impulse_schedule,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
                "delay_s": _KICK_DELAY_S,
                "force_n": _KICK_FORCE_N,
                "duration_s": _KICK_DURATION_S,
                "force_min": f_min,
                "force_max": f_max,
            },
        )
        env_cfg.events.eval_impulse = EventTermCfg(
            func=apply_eval_lateral_impulse,
            mode="interval",
            interval_range_s=(0.02, 0.02),
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
                "delay_s": _KICK_DELAY_S,
                "force_n": _KICK_FORCE_N,
                "duration_s": _KICK_DURATION_S,
            },
        )
        meta["description"] = (
            f"Kick force sweep: lateral impulse on base_link at t={_KICK_DELAY_S:g} s "
            f"for {_KICK_DURATION_S:g} s; force ladder [{f_min:g}, {f_max:g}] N."
        )
        meta["terrain"] = "plane"
        meta["kick_force_min"] = f_min
        meta["kick_force_max"] = f_max
        meta["impulse"] = {
            "delay_s": _KICK_DELAY_S,
            "force_min": f_min,
            "force_max": f_max,
            "duration_s": _KICK_DURATION_S,
        }
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

    if mode == "mass_sweep":
        from ..mdp.eval_events import assign_eval_mass_scale_ladder

        _set_plane_terrain(env_cfg, friction=1.0)
        s_min = float(mass_scale_min)
        s_max = float(mass_scale_max)
        env_cfg.events.assign_eval_mass_scale_ladder = EventTermCfg(
            func=assign_eval_mass_scale_ladder,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_min": s_min,
                "scale_max": s_max,
                "recompute_inertia": True,
            },
        )
        meta["description"] = (
            f"Mass scale sweep: CoM fixed at nominal; mass×inertia ladder "
            f"[{s_min:g}, {s_max:g}]; flat μ=1.0."
        )
        meta["terrain"] = "plane"
        meta["mass_scale_min"] = s_min
        meta["mass_scale_max"] = s_max
        return meta

    if mode == "com_sweep":
        from ..mdp.eval_events import assign_eval_com_offset_ladder

        _set_plane_terrain(env_cfg, friction=1.0)
        o_min = float(com_offset_min_m)
        o_max = float(com_offset_max_m)
        env_cfg.events.assign_eval_com_offset_ladder = EventTermCfg(
            func=assign_eval_com_offset_ladder,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "offset_min_m": o_min,
                "offset_max_m": o_max,
                "axis": "y",
            },
        )
        meta["description"] = (
            f"CoM offset sweep: mass nominal; lateral (y) CoM ladder "
            f"[{o_min * 100:g}, {o_max * 100:g}] cm; flat μ=1.0."
        )
        meta["terrain"] = "plane"
        meta["com_offset_min_m"] = o_min
        meta["com_offset_max_m"] = o_max
        meta["com_axis"] = "y"
        return meta

    if mode == "mass_com_sweep":
        from ..mdp.eval_events import assign_eval_mass_com_grid

        _set_plane_terrain(env_cfg, friction=1.0)
        s_min = float(mass_com_scale_min)
        s_max = float(mass_com_scale_max)
        o_min = float(mass_com_offset_min_m)
        o_max = float(mass_com_offset_max_m)
        env_cfg.events.assign_eval_mass_com_grid = EventTermCfg(
            func=assign_eval_mass_com_grid,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_min": s_min,
                "scale_max": s_max,
                "offset_min_m": o_min,
                "offset_max_m": o_max,
                "axis": "y",
                "recompute_inertia": True,
            },
        )
        meta["description"] = (
            f"Mass×CoM grid: mass [{s_min:g}, {s_max:g}] × lateral CoM "
            f"[{o_min * 100:g}, {o_max * 100:g}] cm; flat μ=1.0."
        )
        meta["terrain"] = "plane"
        meta["mass_com_scale_min"] = s_min
        meta["mass_com_scale_max"] = s_max
        meta["mass_com_offset_min_m"] = o_min
        meta["mass_com_offset_max_m"] = o_max
        meta["com_axis"] = "y"
        return meta

    if mode == "friction":
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

    if mode == "friction_sweep":
        from ..mdp.eval_events import assign_eval_friction_ladder

        mu_lo = float(friction_mu_min)
        mu_hi = float(friction_mu_max)
        # Shared plane μ at the low end so tire ladder is the distinguishing factor.
        _set_plane_terrain(env_cfg, friction=mu_lo)
        env_cfg.events.assign_eval_friction_ladder = EventTermCfg(
            func=assign_eval_friction_ladder,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "mu_min": mu_lo,
                "mu_max": mu_hi,
            },
        )
        meta["description"] = (
            f"Friction sweep: plane μ={mu_lo:g}; tire μ ladder [{mu_lo:g}, {mu_hi:g}]."
        )
        meta["terrain"] = "plane"
        meta["friction_mu_min"] = mu_lo
        meta["friction_mu_max"] = mu_hi
        meta["friction_plane"] = mu_lo
        return meta

    raise ValueError(f"Unhandled eval mode {mode!r}")


def apply_drive_accel_sweep_plant(env_cfg: Any, meta: dict[str, Any]) -> None:
    """Re-apply RC command drive after checkpoint overlay (policy a1 unused)."""
    if not meta.get("force_rc_drive"):
        return
    a_min = float(meta.get("drive_accel_min", _DRIVE_ACCEL_MIN))
    a_max = float(meta.get("drive_accel_max", _DRIVE_ACCEL_MAX))
    _ensure_rc_command_drive(
        env_cfg,
        accel_ms2=0.5 * (a_min + a_max),
        accel_ms2_range=(a_min, a_max),
    )
