# Ported from ~/bike_test/CycleRL/source/CycleRL/CycleRL/tasks/manager_based/cyclerl/eval/eval_logger.py (imports/paths rewritten for trackstand_rl locomotive).
"""
Production-ready GPU-vectorized evaluation logger for 2-wheeled bicycle balancing
in NVIDIA IsaacLab (ManagerBasedRLEnv).

Tracks, computes, logs, and visualises:
  - Position & spatial deviation (station-keeping drift, lateral deviation)
  - Tilt & posture stability (roll/pitch statistics, yaw drift)
  - Velocity & dynamic tracking (RMSE, overshoot, settling time)
  - Actuation smoothness & energy efficiency (action jitter, jerk, power, saturation)
  - Survival & overall task performance
  - Graphical plots (2D trajectory, velocity profile, action/torque, posture)

All metric calculations operate directly on PyTorch GPU tensors across N parallel
environments — NO per-environment Python loops.

Usage (inside an IsaacLab evaluation script):
    from source.trackstand_rl.trackstand_rl.tasks.manager_based.locomotive.eval.eval_logger import (
        BicycleMetricsLogger, BicycleEvalPlotter, export_eval_report,
    )

    logger = BicycleMetricsLogger.from_env(env)
    # ... in eval loop ...
    logger.step(env, actions)
    # ... at episode end / eval end ...
    summary = logger.finalize()
    plotter = BicycleEvalPlotter(logger, save_dir="./eval_plots")
    plotter.generate_all_plots()
    export_eval_report(summary, logger, save_dir="./eval_results")
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from .eval_failure import analyze_survival_thresholds, format_failure_report

# ---------------------------------------------------------------------------
# IsaacLab imports — available at runtime inside the Isaac Sim container.
# ---------------------------------------------------------------------------
try:
    from isaaclab.envs import ManagerBasedRLEnv
    import isaaclab.utils.math as math_utils
except ImportError:  # pragma: no cover
    ManagerBasedRLEnv = None  # type: ignore[assignment]
    math_utils = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Plotting imports — optional; degraded gracefully if unavailable.
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for headless clusters
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False

try:
    from tabulate import tabulate as _tabulate_fn
    _HAS_TABULATE = True
except ImportError:  # pragma: no cover
    _HAS_TABULATE = False


# CycleRL paper (2603.15013) Table II definitions live in ``paper_metrics`` (torch-free)
# so the MuJoCo paper suite scores episodes with the same code.
from .paper_metrics import (  # noqa: E402
    PAPER_BRT_DISTURB_RAD,
    PAPER_BSR_GATE,
    PAPER_BSR_ROLL_LIMIT_RAD,
    PAPER_MSS_ANCHOR_MS,
    PAPER_SRL_TOLERANCE,
    analyze_bsr_frontier,
    analyze_critical_angle,
    analyze_min_sustaining_speed,
    analyze_noise_tolerance,
    balance_success_rate,
)


# ===========================================================================
# 1. Utility helpers
# ===========================================================================


def _quat_to_euler_isaac(
    quat_xyzw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert Isaac Sim quaternion (x, y, z, w) to Euler roll, pitch, yaw.

    Uses the intrinsic ZYX (yaw-pitch-roll) convention consistent with
    Isaac Sim's coordinate frame: X forward, Y left, Z up.

    Args:
        quat_xyzw: (..., 4) quaternion in (x, y, z, w) order.

    Returns:
        (roll, pitch, yaw) — each (...,) in radians.
    """
    if math_utils is not None:
        # Prefer Isaac Lab's own conversion when available.
        # The canonical function name in isaaclab.utils.math is
        # ``euler_xyz_from_quat``.  We try several known aliases.
        for fn_name in ("euler_xyz_from_quat", "quat_to_euler_xyz", "matrix_from_quat"):
            fn = getattr(math_utils, fn_name, None)
            if fn is not None:
                try:
                    result = fn(quat_xyzw)
                    if isinstance(result, tuple) and len(result) == 3:
                        return result  # type: ignore[return-value]
                    # Some functions return a single tensor; unpack if needed.
                    if isinstance(result, torch.Tensor) and result.shape[-1] == 3:
                        return result[..., 0], result[..., 1], result[..., 2]
                except Exception:
                    continue

    # Fallback implementation — standard quaternion-to-Euler (intrinsic ZYX).
    x, y, z, w = quat_xyzw[..., 0], quat_xyzw[..., 1], quat_xyzw[..., 2], quat_xyzw[..., 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def _quat_to_euler_xyzw_tensor(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convenience wrapper that expects (num_envs, 4) quaternion tensor."""
    return _quat_to_euler_isaac(quat)


def _safe_div(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Element-wise safe division returning 0 where denominator ≈ 0."""
    mask = b.abs() > eps
    result = torch.zeros_like(a)
    result[mask] = a[mask] / b[mask]
    return result


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle tensor to (−π, π]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


# ===========================================================================
# 2. Dataclass for episode-level metrics
# ===========================================================================


@dataclass
class EpisodeMetrics:
    """Per-episode aggregated metrics for one environment index."""

    env_id: int
    survived: bool  # True if the episode ran the full horizon without early termination.
    episode_length: int  # number of steps actually executed.
    episode_time_s: float  # wall-clock duration of the episode.

    # -- Position & spatial deviation ----------------------------------------
    min_drift_m: float  # minimum XY displacement from spawn (m).
    max_drift_m: float  # max Euclidean XY displacement from spawn (m).
    mean_drift_m: float  # mean XY displacement (m).
    final_drift_m: float  # final XY displacement at episode end (m).
    lateral_deviation_rms_m: float  # RMS cross-track error (m).

    # -- Roll posture ---------------------------------------------------------
    min_roll_deg: float  # minimum roll angle (degrees) — negative = lean left.
    max_roll_deg: float  # maximum roll angle (degrees) — positive = lean right.
    mean_roll_deg: float  # mean roll angle over episode (degrees).
    std_roll_deg: float  # std dev of roll angle (degrees).
    max_lean_deg: float  # maximum absolute roll before recovery/fall (degrees).

    # -- Pitch posture --------------------------------------------------------
    min_pitch_deg: float
    max_pitch_deg: float
    mean_pitch_deg: float  # mean pitch angle (degrees).
    std_pitch_deg: float  # std dev of pitch (degrees).
    yaw_drift_rate_deg_per_s: float  # average yaw drift rate (deg / s).

    # -- Velocity & dynamic tracking -----------------------------------------
    min_velocity_ms: float  # minimum linear velocity (m/s).
    max_velocity_ms: float  # peak linear velocity magnitude (m/s).
    mean_velocity_ms: float  # mean velocity over episode (m/s).
    velocity_rmse_ms: float  # RMSE vs target (0 for station-keeping).
    velocity_overshoot_pct: float  # peak overshoot % (max / target * 100).

    # -- Actuation & energy --------------------------------------------------
    mean_action_jitter: float  # mean L2 action difference between consecutive steps.
    max_action_jitter: float  # worst-case single-step action change.
    mean_joint_acc_mag: float  # mean joint acceleration magnitude (rad/s²).
    max_joint_acc_mag: float
    total_power_j: float  # cumulative mechanical work (J).
    steering_usage_ratio: float  # fraction of steps where |steering_action| > 0.01.
    drive_usage_ratio: float  # fraction of steps where |drive_action| > 0.01.
    torque_saturation_ratio: float  # fraction of time any torque ≥ 90 % effort limit.

    # -- Stability & recovery -------------------------------------------------
    stable_balance_ratio: float  # fraction of steps where |roll| < 3°.
    recovery_count: int  # number of lean→recover events (crossed above threshold & returned).
    cumulative_roll_excursion_deg_s: float  # integral of |roll| beyond stable window (°·s).
    # Fraction of episode spent above each roll-angle threshold
    survival_roll_thresholds: dict[str, float] = field(default_factory=dict)

    # -- Drive motor performance ----------------------------------------------
    mean_drive_speed_rad_s: float = 0.0  # mean drive joint velocity (rad/s).
    max_drive_speed_rad_s: float = 0.0

    # -- Locomotion tracking (vs command; trackstand is out of scope) ---------
    complete: bool = True  # False if the eval horizon ended mid-episode.
    roll_rmse_deg: float = 0.0  # RMSE of roll from 0°.
    roll_rmse_straight_deg: float = 0.0  # RMSE of roll while |ωz*| < 0.05 rad/s.
    vx_rmse_ms: float = 0.0
    vx_mae_ms: float = 0.0
    vx_ss_mae_ms: float = 0.0  # MAE after the first 1 s of the episode.
    wz_rmse_rad_s: float = 0.0
    heading_rmse_deg: float = 0.0
    mean_cmd_vx_ms: float = 0.0
    steer_action_jitter: float = 0.0
    drive_action_jitter: float = 0.0
    drive_bang_bang_ratio: float = 0.0  # fraction of steps with |Δa_drive| above threshold
    action_clip_sat_ratio: float = 0.0  # |a| ≥ 0.99 on either channel.
    steer_clip_sat_ratio: float = 0.0
    drive_clip_sat_ratio: float = 0.0  # |ω*| hits the 45 rad/s drive clip.
    control_effort: float = 0.0  # Σ (τ_steer² + τ_drive²) Δt
    mean_tau_sq: float = 0.0
    kick_seen: bool = False
    kick_settled: bool = False
    kick_settle_time_s: float = float("nan")
    kick_peak_roll_deg: float = 0.0
    kick_oscillations: int = 0

    # -- Paper Table II (per-episode contributions) ---------------------------
    bsr_balanced: bool = True  # |φ| stayed under PAPER_BSR_ROLL_LIMIT_RAD all episode.
    brt_s: float = float("nan")  # mean recovery time from |φ| > 0.3 rad excursions.
    brt_events: int = 0  # number of completed recoveries (0 ⇒ brt_s is NaN).
    max_balance_duration_s: float = 0.0  # longest unbroken |φ| < 0.5 rad run.
    srl_s: float = float("nan")  # mean time to enter the ±10% vx* band.
    srl_events: int = 0
    noise_rho: float = float("nan")  # observation-noise ladder rung (MNT).

    # -- Spawn / command / fail context (for survival-threshold analysis) ----
    term_reason: str = ""
    spawn_vx_ms: float = float("nan")
    spawn_yaw_deg: float = float("nan")
    spawn_roll_deg: float = float("nan")
    spawn_cmd_vx_ms: float = float("nan")
    spawn_cmd_wz_rad_s: float = float("nan")
    spawn_cmd_heading_deg: float = float("nan")
    drive_accel_ms2: float = float("nan")  # RC hub slew limit (drive_accel_sweep)
    yaw_bias_deg: float = float("nan")  # teleop stick-deg ωz* ladder (yaw_bias_sweep)
    mass_scale: float = float("nan")  # mass_sweep ladder (× nominal)
    com_offset_m: float = float("nan")  # com_sweep lateral CoM offset (m)
    friction_mu: float = float("nan")  # friction_sweep tire μ ladder
    kick_force_n: float = float("nan")  # kick / kick_sweep lateral force (N)
    spawn_terrain_level: int = -1
    spawn_terrain_type: int = -1
    end_vx_ms: float = float("nan")
    end_yaw_deg: float = float("nan")
    end_roll_deg: float = float("nan")
    end_cmd_vx_ms: float = float("nan")
    end_cmd_wz_rad_s: float = float("nan")
    end_cmd_heading_deg: float = float("nan")
    end_pos_x_m: float = float("nan")
    end_pos_y_m: float = float("nan")
    end_terrain_level: int = -1
    end_terrain_type: int = -1


# ===========================================================================
# 3. BicycleMetricsLogger — GPU-vectorized step-by-step accumulator
# ===========================================================================


class BicycleMetricsLogger:
    """Fully vectorised per-step metrics accumulator for N parallel environments.

    All internal buffers are PyTorch tensors on the specified device.
    Per-environment Python loops are *never* used.  When episodes terminate
    (detected via ``env.reset_buf``), the affected rows are extracted,
    finalised into ``EpisodeMetrics`` dataclass instances, and the
    corresponding buffer rows are reset.

    Typical usage::

        logger = BicycleMetricsLogger.from_env(env)

        for _ in range(num_steps):
            obs, rew, done, extras = env.step(actions)
            logger.step(env, actions)

        summary = logger.finalize()
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        num_envs: int,
        device: str = "cuda:0",
        step_dt: float = 0.02,
        stable_roll_threshold_deg: float = 3.0,
        effort_limit_steering: float | None = None,
        effort_limit_drive: float | None = None,
        steer_joint_name: str | None = None,
        drive_joint_name: str | None = None,
        record_history: bool = True,
        history_downsample: int = 1,
        steer_action_scale: float | None = None,
        drive_action_scale: float | None = None,
        drive_vel_clip: float | None = None,
        action_sat_threshold: float = 0.99,
        drive_bang_threshold: float = 0.12,
        straight_wz_threshold: float = 0.05,
        ss_delay_s: float = 1.0,
        kick_settle_roll_deg: float = 2.0,
        kick_settle_hold_s: float = 0.25,
        robot_name: str = "robot",
        steer_action_index: int = 0,
        drive_action_index: int = 1,
        velocity_command_name: str = "velocity",
        heading_command_name: str = "steering",
        contract: Any | None = None,
    ) -> None:
        """
        Args:
            num_envs: Number of parallel environments.
            device: Torch device for tensor buffers.
            step_dt: Simulation time-step duration (seconds) — ``env.step_dt``.
            stable_roll_threshold_deg: Roll angle below which the bike is
                considered "stable" (degrees).
            effort_limit_steering: Steering motor effort limit (N·m). Resolved
                from the env if omitted.
            effort_limit_drive: Drive motor effort limit (N·m). Resolved from
                the env if omitted.
            steer_joint_name: Steering joint name/pattern. Resolved from the
                env action cfg if omitted — do not hardcode USD names here.
            drive_joint_name: Drive joint name/pattern. Resolved from the env
                action cfg if omitted.
            record_history: If True, keep down-sampled CPU history for the
                first environment (env_0) to enable plotting.  Turn off for
                pure-metrics runs to minimise RAM.
            history_downsample: Store every Nth step of history (default 1 =
                every step).  Increase for long episodes.
            steer_action_scale: Policy scale for steering (rad per unit action).
            drive_action_scale: Policy scale for drive (rad/s per unit action).
            drive_vel_clip: Physical drive velocity clip (rad/s).
            action_sat_threshold: |a| at/above this counts as action saturation.
            straight_wz_threshold: |ωz*| below this is a "straightaway".
            ss_delay_s: Seconds ignored at episode start for steady-state error.
            kick_settle_roll_deg: |roll| below this counts as recovered after a kick.
            kick_settle_hold_s: How long |roll| must stay below the settle threshold.
            robot_name: Scene entity name for the bike articulation.
            steer_action_index: Column of the policy action tensor for steering.
            drive_action_index: Column of the policy action tensor for drive.
            velocity_command_name: Command-manager term that holds vx*.
            heading_command_name: Command-manager term that holds ωz*/heading*.
            contract: Optional :class:`BikeEvalContract`. Prefer
                :meth:`from_env` over passing joint names by task.
        """
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.step_dt = step_dt
        self.stable_roll_threshold_rad = math.radians(stable_roll_threshold_deg)
        self.stable_roll_threshold_deg = stable_roll_threshold_deg
        self.record_history = record_history
        self.history_downsample = history_downsample
        self.action_sat_threshold = action_sat_threshold
        self.drive_bang_threshold = drive_bang_threshold
        self.straight_wz_threshold = straight_wz_threshold
        self.ss_delay_steps = max(1, int(round(ss_delay_s / max(step_dt, 1e-6))))
        self.kick_settle_roll_rad = math.radians(kick_settle_roll_deg)
        self.kick_settle_hold_steps = max(1, int(round(kick_settle_hold_s / max(step_dt, 1e-6))))
        self._max_episode_length: int = 0

        self.robot_name = robot_name
        self.steer_joint_name = steer_joint_name
        self.drive_joint_name = drive_joint_name
        self.steer_action_index = steer_action_index
        self.drive_action_index = drive_action_index
        self.steer_action_scale = 1.0 if steer_action_scale is None else float(steer_action_scale)
        self.drive_action_scale = 1.0 if drive_action_scale is None else float(drive_action_scale)
        self.drive_vel_clip = math.inf if drive_vel_clip is None else float(drive_vel_clip)
        self.effort_limit_steering = math.inf if effort_limit_steering is None else float(effort_limit_steering)
        self.effort_limit_drive = math.inf if effort_limit_drive is None else float(effort_limit_drive)
        self.velocity_command_name = velocity_command_name
        self.heading_command_name = heading_command_name
        self.contract = contract
        if contract is not None:
            self._apply_contract(contract, overwrite=True)

        # ---- Per-step scalar accumulators (num_envs,) ----------------------
        # These are updated each step and finalised at episode termination.
        self._step_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # Position & spatial
        self._spawn_pos: Optional[torch.Tensor] = None  # (N, 2) — lazy-init from env_origins
        self._spawn_pos_initialized: bool = False
        self._min_drift_sq = torch.full((num_envs,), float("inf"), device=self.device)
        self._max_drift_sq = torch.zeros(num_envs, device=self.device)
        self._sum_drift = torch.zeros(num_envs, device=self.device)
        self._sum_lateral_sq = torch.zeros(num_envs, device=self.device)

        # Roll — track min/max in addition to sum/sum_sq
        self._min_roll = torch.full((num_envs,), float("inf"), device=self.device)
        self._max_roll = torch.full((num_envs,), float("-inf"), device=self.device)
        self._sum_roll = torch.zeros(num_envs, device=self.device)
        self._sum_roll_sq = torch.zeros(num_envs, device=self.device)
        self._max_lean_abs = torch.zeros(num_envs, device=self.device)

        # Roll survival thresholds: 5°, 10°, 15°, 20°, 30°, 45°
        self._roll_thresholds_deg = [5.0, 10.0, 15.0, 20.0, 30.0, 45.0]
        self._roll_threshold_counts = torch.zeros(
            num_envs, len(self._roll_thresholds_deg), dtype=torch.long, device=self.device
        )
        # Recovery tracking — flag per env: are we currently "leaned" beyond threshold?
        self._is_leaned = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._recovery_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._cumulative_excursion = torch.zeros(num_envs, device=self.device)  # °·s integrated

        # Pitch
        self._min_pitch = torch.full((num_envs,), float("inf"), device=self.device)
        self._max_pitch = torch.full((num_envs,), float("-inf"), device=self.device)
        self._sum_pitch = torch.zeros(num_envs, device=self.device)
        self._sum_pitch_sq = torch.zeros(num_envs, device=self.device)
        self._yaw_initial = torch.zeros(num_envs, device=self.device)
        self._yaw_initialized = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

        # Velocity
        self._min_vel = torch.full((num_envs,), float("inf"), device=self.device)
        self._max_vel = torch.zeros(num_envs, device=self.device)
        self._sum_vel = torch.zeros(num_envs, device=self.device)
        self._sum_vel_err_sq = torch.zeros(num_envs, device=self.device)

        # Actuation & energy
        self._prev_action: Optional[torch.Tensor] = None
        self._sum_action_jitter = torch.zeros(num_envs, device=self.device)
        self._max_action_jitter = torch.zeros(num_envs, device=self.device)
        self._sum_joint_acc_mag = torch.zeros(num_envs, device=self.device)
        self._max_joint_acc_mag = torch.zeros(num_envs, device=self.device)
        self._sum_power = torch.zeros(num_envs, device=self.device)
        self._steering_active_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._drive_active_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._sum_drive_vel = torch.zeros(num_envs, device=self.device)
        self._max_drive_vel = torch.zeros(num_envs, device=self.device)
        self._torque_sat_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # Stability
        self._stable_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # Paper Table II — BSR / BRT / MBD / SRL
        self._bsr_balanced = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        self._brt_active = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._brt_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._brt_sum_s = torch.zeros(num_envs, device=self.device)
        self._brt_events = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._mbd_run_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._mbd_max_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._srl_pending = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._srl_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._srl_sum_s = torch.zeros(num_envs, device=self.device)
        self._srl_events = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._srl_prev_cmd = torch.full((num_envs,), float("nan"), device=self.device)

        # Locomotion tracking vs command
        self._sum_roll_sq_rad = torch.zeros(num_envs, device=self.device)
        self._sum_roll_sq_straight = torch.zeros(num_envs, device=self.device)
        self._straight_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._sum_vx_err_sq = torch.zeros(num_envs, device=self.device)
        self._sum_vx_abs_err = torch.zeros(num_envs, device=self.device)
        self._sum_vx_abs_err_ss = torch.zeros(num_envs, device=self.device)
        self._ss_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._sum_wz_err_sq = torch.zeros(num_envs, device=self.device)
        self._sum_heading_err_sq = torch.zeros(num_envs, device=self.device)
        self._sum_cmd_vx = torch.zeros(num_envs, device=self.device)
        self._sum_steer_jitter = torch.zeros(num_envs, device=self.device)
        self._sum_drive_jitter = torch.zeros(num_envs, device=self.device)
        self._drive_bang_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._jitter_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._action_sat_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._steer_sat_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._drive_clip_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._sum_tau_sq = torch.zeros(num_envs, device=self.device)

        # Kick / impulse rejection
        self._kick_seen = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._kick_peak_roll = torch.zeros(num_envs, device=self.device)
        self._kick_hold = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._kick_settled = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._kick_settle_time = torch.full((num_envs,), float("nan"), device=self.device)
        self._kick_prev_sign = torch.zeros(num_envs, device=self.device)
        self._kick_oscillations = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # ---- Episode archive -----------------------------------------------
        self._completed_episodes: list[EpisodeMetrics] = []

        # ---- History buffers (CPU, env_0 only, for plotting) ---------------
        self._history: dict[str, list[float]] = defaultdict(list)
        self._history_step: int = 0  # global evaluation step counter

        # ---- Joint indices (resolved lazily on first step) -----------------
        self._steer_idx: Optional[int] = None
        self._drive_idx: Optional[int] = None

        nan = float("nan")
        self._spawn_valid = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._spawn_vx = torch.full((num_envs,), nan, device=self.device)
        self._spawn_yaw = torch.full((num_envs,), nan, device=self.device)
        self._spawn_roll = torch.full((num_envs,), nan, device=self.device)
        self._spawn_cmd_vx = torch.full((num_envs,), nan, device=self.device)
        self._spawn_cmd_wz = torch.full((num_envs,), nan, device=self.device)
        self._spawn_cmd_heading = torch.full((num_envs,), nan, device=self.device)
        self._spawn_drive_accel = torch.full((num_envs,), nan, device=self.device)
        self._spawn_yaw_bias = torch.full((num_envs,), nan, device=self.device)
        self._spawn_mass_scale = torch.full((num_envs,), nan, device=self.device)
        self._spawn_com_offset = torch.full((num_envs,), nan, device=self.device)
        self._spawn_noise_rho = torch.full((num_envs,), nan, device=self.device)
        self._spawn_friction_mu = torch.full((num_envs,), nan, device=self.device)
        self._spawn_kick_force_n = torch.full((num_envs,), nan, device=self.device)
        self._spawn_terrain_level = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self._spawn_terrain_type = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self._last_vx = torch.full((num_envs,), nan, device=self.device)
        self._last_yaw = torch.full((num_envs,), nan, device=self.device)
        self._last_roll = torch.full((num_envs,), nan, device=self.device)
        self._last_cmd_vx = torch.full((num_envs,), nan, device=self.device)
        self._last_cmd_wz = torch.full((num_envs,), nan, device=self.device)
        self._last_cmd_heading = torch.full((num_envs,), nan, device=self.device)
        self._last_pos_x = torch.full((num_envs,), nan, device=self.device)
        self._last_pos_y = torch.full((num_envs,), nan, device=self.device)
        self._last_terrain_level = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self._last_terrain_type = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Internal — resolve joint indices / env contract
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, env: Any, **kwargs: Any) -> "BicycleMetricsLogger":
        """Construct a logger whose joints/limits come from the live env cfg."""
        from .eval_contract import contract_field_names, resolve_eval_contract

        env_config_path = kwargs.pop("env_config_path", None)
        log_dir = kwargs.pop("log_dir", None)
        contract = kwargs.pop("contract", None) or resolve_eval_contract(
            env,
            env_config_path=env_config_path,
            log_dir=log_dir,
        )
        overrides = {k: kwargs.pop(k) for k in list(kwargs) if k in contract_field_names()}
        if overrides:
            contract = contract.replace(**overrides)
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
        num_envs = kwargs.pop("num_envs", getattr(unwrapped, "num_envs"))
        device = kwargs.pop("device", str(getattr(unwrapped, "device", "cuda:0")))
        try:
            step_dt = kwargs.pop("step_dt", float(unwrapped.step_dt))
        except AttributeError:
            step_dt = kwargs.pop("step_dt", 0.02)
        return cls(
            num_envs=int(num_envs),
            device=device,
            step_dt=step_dt,
            contract=contract,
            **kwargs,
        )

    def _apply_contract(self, contract: Any, *, overwrite: bool = False) -> None:
        """Copy resolved joint/action/limit fields from an eval contract."""
        self.contract = contract
        mapping = {
            "robot_name": "robot_name",
            "steer_joint_name": "steer_joint_name",
            "drive_joint_name": "drive_joint_name",
            "steer_action_index": "steer_action_index",
            "drive_action_index": "drive_action_index",
            "steer_action_scale": "steer_action_scale",
            "drive_action_scale": "drive_action_scale",
            "drive_vel_clip": "drive_vel_clip",
            "effort_limit_steering": "effort_limit_steering",
            "effort_limit_drive": "effort_limit_drive",
            "velocity_command_name": "velocity_command_name",
            "heading_command_name": "heading_command_name",
        }
        for src, dst in mapping.items():
            value = getattr(contract, src, None)
            if value is None:
                continue
            current = getattr(self, dst, None)
            if overwrite or current is None or current == "" or (
                isinstance(current, float) and not math.isfinite(current)
            ):
                setattr(self, dst, value)

    def _resolve_joint_indices(self, env: Any, asset: Any) -> None:
        """Resolve steering and drive joint indices from the articulation."""
        if self._steer_idx is not None:
            return
        if self.steer_joint_name is None or self.drive_joint_name is None:
            from .eval_contract import resolve_eval_contract

            self._apply_contract(resolve_eval_contract(env), overwrite=True)
        if not self.steer_joint_name or not self.drive_joint_name:
            raise ValueError(
                "BicycleMetricsLogger needs steer/drive joint names. "
                "Pass them explicitly or construct with BicycleMetricsLogger.from_env(env)."
            )
        available = list(getattr(asset, "joint_names", []) or [])
        for role, pattern in (("steer", self.steer_joint_name), ("drive", self.drive_joint_name)):
            joints, names = asset.find_joints(pattern)
            if joints is None or len(joints) == 0:
                raise ValueError(
                    f"No {role} joint matching {pattern!r} on '{self.robot_name}'. "
                    f"Available joints: {available}"
                )
            idx = joints[0].item() if hasattr(joints[0], "item") else int(joints[0])
            if role == "steer":
                self._steer_idx = idx
            else:
                self._drive_idx = idx

    def _action_channel(self, actions: torch.Tensor, index: int) -> torch.Tensor:
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        if index < actions.shape[-1]:
            return actions[:, index]
        return torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)

    def _read_commands(
        self, env: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read vx*, ωz*, and heading* from the command manager.

        Returns:
            (vx_cmd, wz_cmd, heading_cmd) each shape (num_envs,).
        """
        device = self.device
        n = self.num_envs
        vx_cmd = torch.zeros(n, device=device)
        wz_cmd = torch.zeros(n, device=device)
        heading_cmd = torch.zeros(n, device=device)
        mgr = getattr(env, "command_manager", None)
        if mgr is None:
            return vx_cmd, wz_cmd, heading_cmd
        vel_name = self.velocity_command_name
        heading_name = self.heading_command_name
        try:
            vel_cmd = mgr.get_command(vel_name)
            vx_cmd = vel_cmd[:, 0].to(device)
        except Exception:
            pass
        try:
            steer_cmd = mgr.get_command(heading_name)
            wz_cmd = steer_cmd[:, 2].to(device)
        except Exception:
            pass
        try:
            term = mgr.get_term(heading_name) if hasattr(mgr, "get_term") else mgr._terms.get(heading_name)
            if term is not None and hasattr(term, "heading_target"):
                heading_cmd = term.heading_target.to(device)
        except Exception:
            pass
        return vx_cmd, wz_cmd, heading_cmd

    def _read_terrain(self, env: Any) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.num_envs
        level = torch.full((n,), -1, dtype=torch.long, device=self.device)
        typ = torch.full((n,), -1, dtype=torch.long, device=self.device)
        try:
            terrain = env.scene["terrain"]
        except Exception:
            return level, typ
        if hasattr(terrain, "terrain_levels"):
            level = terrain.terrain_levels.to(device=self.device, dtype=torch.long)
        if hasattr(terrain, "terrain_types"):
            typ = terrain.terrain_types.to(device=self.device, dtype=torch.long)
        return level, typ

    def _read_drive_accel(self, env: Any) -> torch.Tensor:
        n = self.num_envs
        nan = torch.full((n,), float("nan"), device=self.device)
        stored = getattr(env, "_eval_drive_accel_ms2", None)
        if stored is not None:
            return stored.to(device=self.device, dtype=nan.dtype).reshape(-1)[:n]
        terms = getattr(getattr(env, "action_manager", None), "_terms", None) or {}
        term = terms.get("drive")
        accel = getattr(term, "_accel_ms2", None)
        if accel is not None:
            return accel.to(device=self.device, dtype=nan.dtype).reshape(-1)[:n]
        cfg_accel = getattr(getattr(term, "cfg", None), "accel_ms2", None)
        if cfg_accel is not None:
            return torch.full((n,), float(cfg_accel), device=self.device)
        return nan

    def _read_yaw_bias(self, env: Any) -> torch.Tensor:
        n = self.num_envs
        nan = torch.full((n,), float("nan"), device=self.device)
        stored = getattr(env, "_eval_yaw_bias_deg", None)
        if stored is not None:
            return stored.to(device=self.device, dtype=nan.dtype).reshape(-1)[:n]
        return nan

    def _read_mass_scale(self, env: Any) -> torch.Tensor:
        n = self.num_envs
        nan = torch.full((n,), float("nan"), device=self.device)
        stored = getattr(env, "_eval_mass_scale", None)
        if stored is not None:
            return stored.to(device=self.device, dtype=nan.dtype).reshape(-1)[:n]
        return nan

    def _read_com_offset(self, env: Any) -> torch.Tensor:
        n = self.num_envs
        nan = torch.full((n,), float("nan"), device=self.device)
        stored = getattr(env, "_eval_com_offset_m", None)
        if stored is not None:
            return stored.to(device=self.device, dtype=nan.dtype).reshape(-1)[:n]
        return nan

    def _read_noise_rho(self, env: Any) -> torch.Tensor:
        n = self.num_envs
        nan = torch.full((n,), float("nan"), device=self.device)
        stored = getattr(env, "_eval_noise_rho", None)
        if stored is not None:
            return stored.to(device=self.device, dtype=nan.dtype).reshape(-1)[:n]
        return nan

    def _read_friction_mu(self, env: Any) -> torch.Tensor:
        n = self.num_envs
        nan = torch.full((n,), float("nan"), device=self.device)
        stored = getattr(env, "_eval_friction_mu", None)
        if stored is not None:
            return stored.to(device=self.device, dtype=nan.dtype).reshape(-1)[:n]
        return nan

    def _read_kick_force(self, env: Any) -> torch.Tensor:
        n = self.num_envs
        nan = torch.full((n,), float("nan"), device=self.device)
        stored = getattr(env, "_eval_kick_force_n", None)
        if stored is not None:
            return stored.to(device=self.device, dtype=nan.dtype).reshape(-1)[:n]
        imp = getattr(env, "_eval_imp_force", None)
        if imp is not None:
            out = imp.to(device=self.device, dtype=nan.dtype).reshape(-1)[:n].clone()
            out[out < 0] = float("nan")
            return out
        return nan

    def _read_yaw_bias_vx(self, env: Any) -> torch.Tensor | None:
        """Optional per-env vx* from yaw_bias grid (None if not a yaw-bias sweep)."""
        stored = getattr(env, "_eval_yaw_bias_vx_ms", None)
        if stored is None:
            return None
        n = self.num_envs
        return stored.to(device=self.device, dtype=torch.float32).reshape(-1)[:n]

    def _read_episode_context(self, env: Any) -> dict[str, torch.Tensor]:
        asset = env.scene[self.robot_name]
        roll, _, yaw = _quat_to_euler_xyzw_tensor(asset.data.root_quat_w)
        vx_cmd, wz_cmd, heading_cmd = self._read_commands(env)
        yaw_vx = self._read_yaw_bias_vx(env)
        if yaw_vx is not None:
            vx_cmd = yaw_vx
        accel = self._read_drive_accel(env)
        yaw_bias = self._read_yaw_bias(env)
        mass_scale = self._read_mass_scale(env)
        com_offset = self._read_com_offset(env)
        noise_rho = self._read_noise_rho(env)
        friction_mu = self._read_friction_mu(env)
        kick_force = self._read_kick_force(env)
        level, typ = self._read_terrain(env)
        pos = asset.data.root_pos_w
        return {
            "vx": asset.data.root_lin_vel_b[:, 0].to(self.device),
            "yaw": yaw.to(self.device),
            "roll": roll.to(self.device),
            "cmd_vx": vx_cmd,
            "cmd_wz": wz_cmd,
            "cmd_heading": heading_cmd,
            "drive_accel": accel,
            "yaw_bias": yaw_bias,
            "mass_scale": mass_scale,
            "com_offset_m": com_offset,
            "noise_rho": noise_rho,
            "friction_mu": friction_mu,
            "kick_force_n": kick_force,
            "pos_x": pos[:, 0].to(self.device),
            "pos_y": pos[:, 1].to(self.device),
            "terrain_level": level,
            "terrain_type": typ,
        }

    def _write_context(self, ids: torch.Tensor, ctx: dict[str, torch.Tensor], dest: str) -> None:
        if ids.dtype != torch.bool:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[ids.to(self.device).reshape(-1)] = True
        else:
            mask = ids.to(device=self.device)
        ctx = {k: v.to(device=self.device) for k, v in ctx.items()}
        if dest == "spawn":
            self._spawn_vx[mask] = ctx["vx"][mask]
            self._spawn_yaw[mask] = ctx["yaw"][mask]
            self._spawn_roll[mask] = ctx["roll"][mask]
            self._spawn_cmd_vx[mask] = ctx["cmd_vx"][mask]
            self._spawn_cmd_wz[mask] = ctx["cmd_wz"][mask]
            self._spawn_cmd_heading[mask] = ctx["cmd_heading"][mask]
            if "drive_accel" in ctx:
                self._spawn_drive_accel[mask] = ctx["drive_accel"][mask]
            if "yaw_bias" in ctx:
                self._spawn_yaw_bias[mask] = ctx["yaw_bias"][mask]
            if "mass_scale" in ctx:
                self._spawn_mass_scale[mask] = ctx["mass_scale"][mask]
            if "com_offset_m" in ctx:
                self._spawn_com_offset[mask] = ctx["com_offset_m"][mask]
            if "noise_rho" in ctx:
                self._spawn_noise_rho[mask] = ctx["noise_rho"][mask]
            if "friction_mu" in ctx:
                self._spawn_friction_mu[mask] = ctx["friction_mu"][mask]
            if "kick_force_n" in ctx:
                self._spawn_kick_force_n[mask] = ctx["kick_force_n"][mask]
            self._spawn_terrain_level[mask] = ctx["terrain_level"][mask]
            self._spawn_terrain_type[mask] = ctx["terrain_type"][mask]
            self._spawn_valid[mask] = True
            return
        self._last_vx[mask] = ctx["vx"][mask]
        self._last_yaw[mask] = ctx["yaw"][mask]
        self._last_roll[mask] = ctx["roll"][mask]
        self._last_cmd_vx[mask] = ctx["cmd_vx"][mask]
        self._last_cmd_wz[mask] = ctx["cmd_wz"][mask]
        self._last_cmd_heading[mask] = ctx["cmd_heading"][mask]
        self._last_pos_x[mask] = ctx["pos_x"][mask]
        self._last_pos_y[mask] = ctx["pos_y"][mask]
        self._last_terrain_level[mask] = ctx["terrain_level"][mask]
        self._last_terrain_type[mask] = ctx["terrain_type"][mask]

    def sync_episode_start(self, env: Any) -> None:
        """Record spawn vx/yaw/commands after ``env.reset()``, before the eval loop."""
        env = env.unwrapped if hasattr(env, "unwrapped") else env
        ctx = self._read_episode_context(env)
        all_ids = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._write_context(all_ids, ctx, "spawn")
        self._write_context(all_ids, ctx, "last")
        if self._spawn_pos is None:
            self._spawn_pos = env.scene.env_origins[:, :2].clone()
            self._spawn_pos_initialized = True
        else:
            self._spawn_pos[:] = env.scene.env_origins[:, :2]

    def _termination_reason(self, env: Any, eid: int, survived: bool) -> str:
        if survived:
            return "time_out"
        tm = getattr(env, "termination_manager", None) if env is not None else None
        names = list(getattr(tm, "active_terms", []) or []) if tm is not None else []
        last = getattr(tm, "_last_episode_dones", None) if tm is not None else None
        if last is not None and names:
            row = last[eid]
            crash_name = None
            timeout_name = None
            for i, name in enumerate(names):
                if i >= row.numel() or not bool(row[i].item()):
                    continue
                cfg = tm.get_term_cfg(name) if hasattr(tm, "get_term_cfg") else None
                if cfg is not None and bool(getattr(cfg, "time_out", False)):
                    timeout_name = name
                else:
                    crash_name = name
            if crash_name:
                return crash_name
            if timeout_name:
                return timeout_name
        try:
            if env is not None and bool(env.reset_time_outs[eid]):
                return "time_out"
            if env is not None and bool(env.reset_terminated[eid]):
                return "terminated"
        except Exception:
            pass
        return "unknown"

    def _reset_row_buffers(self, done_ids: torch.Tensor, env: Any | None = None) -> None:
        """Clear per-env accumulators for environments that just reset."""
        self._step_count[done_ids] = 0
        if env is not None and self._spawn_pos is not None:
            self._spawn_pos[done_ids] = env.scene.env_origins[done_ids, :2]
        self._min_drift_sq[done_ids] = float("inf")
        self._max_drift_sq[done_ids] = 0.0
        self._sum_drift[done_ids] = 0.0
        self._sum_lateral_sq[done_ids] = 0.0
        self._min_roll[done_ids] = float("inf")
        self._max_roll[done_ids] = float("-inf")
        self._sum_roll[done_ids] = 0.0
        self._sum_roll_sq[done_ids] = 0.0
        self._max_lean_abs[done_ids] = 0.0
        self._roll_threshold_counts[done_ids] = 0
        self._is_leaned[done_ids] = False
        self._recovery_count[done_ids] = 0
        self._cumulative_excursion[done_ids] = 0.0
        self._min_pitch[done_ids] = float("inf")
        self._max_pitch[done_ids] = float("-inf")
        self._sum_pitch[done_ids] = 0.0
        self._sum_pitch_sq[done_ids] = 0.0
        self._min_vel[done_ids] = float("inf")
        self._max_vel[done_ids] = 0.0
        self._sum_vel[done_ids] = 0.0
        self._sum_vel_err_sq[done_ids] = 0.0
        self._sum_action_jitter[done_ids] = 0.0
        self._max_action_jitter[done_ids] = 0.0
        self._sum_joint_acc_mag[done_ids] = 0.0
        self._max_joint_acc_mag[done_ids] = 0.0
        self._sum_power[done_ids] = 0.0
        self._steering_active_steps[done_ids] = 0
        self._drive_active_steps[done_ids] = 0
        self._sum_drive_vel[done_ids] = 0.0
        self._max_drive_vel[done_ids] = 0.0
        self._torque_sat_steps[done_ids] = 0
        self._stable_steps[done_ids] = 0
        self._bsr_balanced[done_ids] = True
        self._brt_active[done_ids] = False
        self._brt_steps[done_ids] = 0
        self._brt_sum_s[done_ids] = 0.0
        self._brt_events[done_ids] = 0
        self._mbd_run_steps[done_ids] = 0
        self._mbd_max_steps[done_ids] = 0
        self._srl_pending[done_ids] = False
        self._srl_steps[done_ids] = 0
        self._srl_sum_s[done_ids] = 0.0
        self._srl_events[done_ids] = 0
        self._srl_prev_cmd[done_ids] = float("nan")
        self._yaw_initialized[done_ids] = False
        self._sum_roll_sq_rad[done_ids] = 0.0
        self._sum_roll_sq_straight[done_ids] = 0.0
        self._straight_steps[done_ids] = 0
        self._sum_vx_err_sq[done_ids] = 0.0
        self._sum_vx_abs_err[done_ids] = 0.0
        self._sum_vx_abs_err_ss[done_ids] = 0.0
        self._ss_steps[done_ids] = 0
        self._sum_wz_err_sq[done_ids] = 0.0
        self._sum_heading_err_sq[done_ids] = 0.0
        self._sum_cmd_vx[done_ids] = 0.0
        self._sum_steer_jitter[done_ids] = 0.0
        self._sum_drive_jitter[done_ids] = 0.0
        self._drive_bang_steps[done_ids] = 0
        self._jitter_steps[done_ids] = 0
        self._action_sat_steps[done_ids] = 0
        self._steer_sat_steps[done_ids] = 0
        self._drive_clip_steps[done_ids] = 0
        self._sum_tau_sq[done_ids] = 0.0
        self._kick_seen[done_ids] = False
        self._kick_peak_roll[done_ids] = 0.0
        self._kick_hold[done_ids] = 0
        self._kick_settled[done_ids] = False
        self._kick_settle_time[done_ids] = float("nan")
        self._kick_prev_sign[done_ids] = 0.0
        self._kick_oscillations[done_ids] = 0
        if self._prev_action is not None:
            self._prev_action[done_ids] = 0.0
        self._spawn_valid[done_ids] = False

    def _resolve_yaw_bias_deg(self, eid: int, env: Any | None = None) -> float:
        """Spawn-captured stick deg, or live ladder buffer (per-env index is fixed)."""
        val = float(self._spawn_yaw_bias[eid].item())
        if math.isfinite(val):
            return val
        if env is not None:
            stored = getattr(env, "_eval_yaw_bias_deg", None)
            if stored is not None and eid < stored.shape[0]:
                return float(stored[eid].item())
        return float("nan")

    def _snapshot_episode(
        self,
        eid: int,
        n: int,
        survived: bool,
        complete: bool,
        env: Any | None = None,
    ) -> EpisodeMetrics:
        """Build ``EpisodeMetrics`` for one environment index from GPU buffers."""
        n = max(int(n), 1)
        t_ep = n * self.step_dt

        min_drift = math.sqrt(max(self._min_drift_sq[eid].item(), 0.0))
        max_drift = math.sqrt(max(self._max_drift_sq[eid].item(), 0.0))
        mean_drift = self._sum_drift[eid].item() / n
        if env is not None and self._spawn_pos is not None:
            robot = env.scene[self.robot_name]
            final_drift = torch.norm(
                robot.data.root_pos_w[eid, :2] - self._spawn_pos[eid], dim=-1
            ).item()
        else:
            final_drift = max_drift
        lateral_rms = math.sqrt(self._sum_lateral_sq[eid].item() / n)

        min_roll = self._min_roll[eid].item()
        max_roll = self._max_roll[eid].item()
        mean_roll = self._sum_roll[eid].item() / n
        std_roll = math.sqrt(max(self._sum_roll_sq[eid].item() / n - mean_roll**2, 0.0))
        max_lean = self._max_lean_abs[eid].item()
        if max_lean < math.pi + 1e-6:
            max_lean = math.degrees(max_lean)

        threshold_pcts = {}
        for i, thr in enumerate(self._roll_thresholds_deg):
            threshold_pcts[f">{thr:.0f}°"] = self._roll_threshold_counts[eid, i].item() / n

        min_pitch = self._min_pitch[eid].item()
        max_pitch = self._max_pitch[eid].item()
        mean_pitch = self._sum_pitch[eid].item() / n
        std_pitch = math.sqrt(max(self._sum_pitch_sq[eid].item() / n - mean_pitch**2, 0.0))

        yaw_drift_rate = 0.0
        if env is not None:
            yaw_initial = self._yaw_initial[eid].item()
            _, _, yaw_f = _quat_to_euler_xyzw_tensor(
                env.scene[self.robot_name].data.root_quat_w[eid : eid + 1]
            )
            yaw_drift_rate = math.degrees(yaw_f.item() - yaw_initial) / max(t_ep, 1e-6)

        min_vel = self._min_vel[eid].item()
        max_vel = self._max_vel[eid].item()
        mean_vel = self._sum_vel[eid].item() / n
        vel_rmse = math.sqrt(self._sum_vel_err_sq[eid].item() / n)
        mean_cmd_vx = self._sum_cmd_vx[eid].item() / n
        overshoot_pct = 0.0
        if abs(mean_cmd_vx) > 1e-3:
            overshoot_pct = max(0.0, (max_vel - abs(mean_cmd_vx)) / abs(mean_cmd_vx) * 100.0)

        jitter_n = max(self._jitter_steps[eid].item(), 1)
        mean_jitter = self._sum_action_jitter[eid].item() / jitter_n
        max_jitter = self._max_action_jitter[eid].item()
        mean_joint_acc = self._sum_joint_acc_mag[eid].item() / n
        max_joint_acc = self._max_joint_acc_mag[eid].item()
        total_power = self._sum_power[eid].item()
        steer_usage = self._steering_active_steps[eid].item() / n
        drive_usage = self._drive_active_steps[eid].item() / n
        mean_drive_speed = self._sum_drive_vel[eid].item() / n
        max_drive_speed = self._max_drive_vel[eid].item()
        sat_ratio = self._torque_sat_steps[eid].item() / n
        stable_ratio = self._stable_steps[eid].item() / n

        straight_n = max(self._straight_steps[eid].item(), 1)
        ss_n = max(self._ss_steps[eid].item(), 1)
        roll_rmse = math.degrees(math.sqrt(self._sum_roll_sq_rad[eid].item() / n))
        roll_rmse_straight = math.degrees(math.sqrt(self._sum_roll_sq_straight[eid].item() / straight_n))
        kick_settle = self._kick_settle_time[eid].item()

        brt_events = int(self._brt_events[eid].item())
        brt_s = (self._brt_sum_s[eid].item() / brt_events) if brt_events else float("nan")
        srl_events = int(self._srl_events[eid].item())
        srl_s = (self._srl_sum_s[eid].item() / srl_events) if srl_events else float("nan")

        return EpisodeMetrics(
            env_id=eid,
            survived=survived,
            episode_length=n,
            episode_time_s=t_ep,
            min_drift_m=min_drift,
            max_drift_m=max_drift,
            mean_drift_m=mean_drift,
            final_drift_m=final_drift,
            lateral_deviation_rms_m=lateral_rms,
            min_roll_deg=min_roll,
            max_roll_deg=max_roll,
            mean_roll_deg=mean_roll,
            std_roll_deg=std_roll,
            max_lean_deg=max_lean,
            min_pitch_deg=min_pitch,
            max_pitch_deg=max_pitch,
            mean_pitch_deg=mean_pitch,
            std_pitch_deg=std_pitch,
            yaw_drift_rate_deg_per_s=yaw_drift_rate,
            min_velocity_ms=min_vel,
            max_velocity_ms=max_vel,
            mean_velocity_ms=mean_vel,
            velocity_rmse_ms=vel_rmse,
            velocity_overshoot_pct=overshoot_pct,
            mean_action_jitter=mean_jitter,
            max_action_jitter=max_jitter,
            mean_joint_acc_mag=mean_joint_acc,
            max_joint_acc_mag=max_joint_acc,
            total_power_j=total_power,
            steering_usage_ratio=steer_usage,
            drive_usage_ratio=drive_usage,
            torque_saturation_ratio=sat_ratio,
            stable_balance_ratio=stable_ratio,
            recovery_count=self._recovery_count[eid].item(),
            cumulative_roll_excursion_deg_s=self._cumulative_excursion[eid].item(),
            survival_roll_thresholds=threshold_pcts,
            mean_drive_speed_rad_s=mean_drive_speed,
            max_drive_speed_rad_s=max_drive_speed,
            complete=complete,
            roll_rmse_deg=roll_rmse,
            roll_rmse_straight_deg=roll_rmse_straight,
            vx_rmse_ms=math.sqrt(self._sum_vx_err_sq[eid].item() / n),
            vx_mae_ms=self._sum_vx_abs_err[eid].item() / n,
            vx_ss_mae_ms=self._sum_vx_abs_err_ss[eid].item() / ss_n,
            wz_rmse_rad_s=math.sqrt(self._sum_wz_err_sq[eid].item() / n),
            heading_rmse_deg=math.degrees(math.sqrt(self._sum_heading_err_sq[eid].item() / n)),
            mean_cmd_vx_ms=mean_cmd_vx,
            steer_action_jitter=self._sum_steer_jitter[eid].item() / jitter_n,
            drive_action_jitter=self._sum_drive_jitter[eid].item() / jitter_n,
            drive_bang_bang_ratio=self._drive_bang_steps[eid].item() / jitter_n,
            action_clip_sat_ratio=self._action_sat_steps[eid].item() / n,
            steer_clip_sat_ratio=self._steer_sat_steps[eid].item() / n,
            drive_clip_sat_ratio=self._drive_clip_steps[eid].item() / n,
            control_effort=self._sum_tau_sq[eid].item() * self.step_dt,
            mean_tau_sq=self._sum_tau_sq[eid].item() / n,
            kick_seen=bool(self._kick_seen[eid].item()),
            kick_settled=bool(self._kick_settled[eid].item()),
            kick_settle_time_s=float(kick_settle) if math.isfinite(float(kick_settle)) else float("nan"),
            kick_peak_roll_deg=math.degrees(self._kick_peak_roll[eid].item()),
            kick_oscillations=int(self._kick_oscillations[eid].item()),
            bsr_balanced=bool(self._bsr_balanced[eid].item()),
            brt_s=brt_s,
            brt_events=brt_events,
            max_balance_duration_s=self._mbd_max_steps[eid].item() * self.step_dt,
            srl_s=srl_s,
            srl_events=srl_events,
            noise_rho=self._spawn_noise_rho[eid].item(),
            term_reason=self._termination_reason(env, eid, survived),
            spawn_vx_ms=self._spawn_vx[eid].item(),
            spawn_yaw_deg=math.degrees(self._spawn_yaw[eid].item()),
            spawn_roll_deg=math.degrees(self._spawn_roll[eid].item()),
            spawn_cmd_vx_ms=self._spawn_cmd_vx[eid].item(),
            spawn_cmd_wz_rad_s=self._spawn_cmd_wz[eid].item(),
            spawn_cmd_heading_deg=math.degrees(self._spawn_cmd_heading[eid].item()),
            drive_accel_ms2=self._spawn_drive_accel[eid].item(),
            yaw_bias_deg=self._resolve_yaw_bias_deg(eid, env),
            mass_scale=self._spawn_mass_scale[eid].item(),
            com_offset_m=self._spawn_com_offset[eid].item(),
            friction_mu=self._spawn_friction_mu[eid].item(),
            kick_force_n=self._spawn_kick_force_n[eid].item(),
            spawn_terrain_level=int(self._spawn_terrain_level[eid].item()),
            spawn_terrain_type=int(self._spawn_terrain_type[eid].item()),
            end_vx_ms=self._last_vx[eid].item(),
            end_yaw_deg=math.degrees(self._last_yaw[eid].item()),
            end_roll_deg=math.degrees(self._last_roll[eid].item()),
            end_cmd_vx_ms=self._last_cmd_vx[eid].item(),
            end_cmd_wz_rad_s=self._last_cmd_wz[eid].item(),
            end_cmd_heading_deg=math.degrees(self._last_cmd_heading[eid].item()),
            end_pos_x_m=self._last_pos_x[eid].item(),
            end_pos_y_m=self._last_pos_y[eid].item(),
            end_terrain_level=int(self._last_terrain_level[eid].item()),
            end_terrain_type=int(self._last_terrain_type[eid].item()),
        )

    # ------------------------------------------------------------------
    # Step — called every environment step
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def step(
        self,
        env: ManagerBasedRLEnv,
        actions: torch.Tensor,
        target_velocity: Optional[torch.Tensor] = None,
        extra_info: Optional[dict[str, Any]] = None,
    ) -> None:
        """Accumulate per-step metrics across all environments.

        Args:
            env: The IsaacLab ``ManagerBasedRLEnv`` instance.
            actions: The action tensor of shape ``(num_envs, action_dim)``
                that was passed to ``env.step()``.
            target_velocity: Optional target linear velocity (m/s) for each
                environment, shape ``(num_envs,)``.  Defaults to zeros
                (pure station-keeping).
            extra_info: Optional extra dict for future extensions.
        """
        if isinstance(actions, dict):
            actions = next(iter(actions.values()))
        actions = actions.to(self.device)

        asset = env.scene[self.robot_name]
        self._resolve_joint_indices(env, asset)
        assert self._steer_idx is not None and self._drive_idx is not None

        device = self.device
        num_envs = self.num_envs
        try:
            self._max_episode_length = int(env.max_episode_length)
        except Exception:
            pass

        # ---- Lazy-init spawn positions from env_origins on first call ------
        if not self._spawn_pos_initialized:
            self._spawn_pos = env.scene.env_origins[:, :2].clone()
            self._spawn_pos_initialized = True

        # ---- Fetch all batched data from the asset -------------------------
        root_pos_w = asset.data.root_pos_w  # (N, 3)  world position
        root_quat_w = asset.data.root_quat_w  # (N, 4)  (x, y, z, w)
        root_lin_vel_b = asset.data.root_lin_vel_b  # (N, 3)  body-frame linear vel
        root_ang_vel_b = asset.data.root_ang_vel_b  # (N, 3)  body-frame angular vel
        joint_pos = asset.data.joint_pos  # (N, num_joints)
        joint_vel = asset.data.joint_vel  # (N, num_joints)
        joint_acc = asset.data.joint_acc  # (N, num_joints)

        # Applied torque: either from applied_torque or computed via action.
        try:
            applied_torque = asset.data.applied_torque  # (N, num_joints)
        except AttributeError:
            applied_torque = asset.data.computed_torque

        steer_idx = self._steer_idx
        drive_idx = self._drive_idx

        # ---- A. Position & spatial deviation -------------------------------
        current_pos_xy = root_pos_w[:, :2]  # (N, 2)
        drift = torch.norm(current_pos_xy - self._spawn_pos, dim=-1)  # (N,) Euclidean
        drift_sq = drift ** 2
        self._min_drift_sq = torch.minimum(self._min_drift_sq, drift_sq)
        self._max_drift_sq = torch.maximum(self._max_drift_sq, drift_sq)
        self._sum_drift += drift

        # Heading direction for lateral deviation: commanded heading if available.
        roll, pitch, yaw = _quat_to_euler_xyzw_tensor(root_quat_w)
        vx_cmd, wz_cmd, heading_cmd = self._read_commands(env)
        if target_velocity is None:
            target_velocity = vx_cmd

        # Initialise yaw reference on first step.
        if not self._yaw_initialized.all():
            newly_active = ~self._yaw_initialized
            self._yaw_initial[newly_active] = yaw[newly_active]
            self._yaw_initialized[newly_active] = True

        heading_ref = heading_cmd

        # Cross-track error vs commanded heading.
        displacement = current_pos_xy - self._spawn_pos  # (N, 2)
        heading_dir = torch.stack(
            [torch.cos(heading_ref), torch.sin(heading_ref)], dim=-1
        )  # (N, 2)
        # Projection of displacement onto heading direction.
        along_track = (displacement * heading_dir).sum(dim=-1, keepdim=True) * heading_dir
        cross_track = displacement - along_track  # (N, 2)
        lateral_sq = (cross_track**2).sum(dim=-1)  # (N,)
        self._sum_lateral_sq += lateral_sq

        # ---- B. Tilt & posture stability -----------------------------------
        roll_deg = torch.rad2deg(roll)
        pitch_deg = torch.rad2deg(pitch)
        roll_abs = roll.abs()

        self._min_roll = torch.minimum(self._min_roll, roll_deg)
        self._max_roll = torch.maximum(self._max_roll, roll_deg)
        self._sum_roll += roll_deg
        self._sum_roll_sq += roll_deg ** 2
        self._max_lean_abs = torch.maximum(self._max_lean_abs, roll_abs)

        # Roll survival thresholds
        roll_abs_deg = torch.rad2deg(roll_abs)
        for i, thr in enumerate(self._roll_thresholds_deg):
            self._roll_threshold_counts[:, i] += (roll_abs_deg >= thr).long()

        # Recovery tracking: detect crossing above 15° then returning below 5°
        lean_threshold_deg = 15.0
        recovery_threshold_deg = 5.0
        was_leaned = self._is_leaned.clone()
        self._is_leaned = self._is_leaned | (roll_abs_deg >= lean_threshold_deg)
        newly_recovered = was_leaned & (roll_abs_deg < recovery_threshold_deg)
        self._recovery_count += newly_recovered.long()
        # Reset "leaned" flag for those that recovered
        self._is_leaned[newly_recovered] = False

        # Cumulative roll excursion beyond stable window
        beyond_stable = torch.clamp(roll_abs_deg - self.stable_roll_threshold_deg, min=0.0)
        self._cumulative_excursion += beyond_stable * self.step_dt

        # ---- B2. Paper Table II: BSR / BRT / MBD ---------------------------
        balanced = roll_abs < PAPER_BSR_ROLL_LIMIT_RAD
        self._bsr_balanced &= balanced
        self._mbd_run_steps = torch.where(
            balanced, self._mbd_run_steps + 1, torch.zeros_like(self._mbd_run_steps)
        )
        self._mbd_max_steps = torch.maximum(self._mbd_max_steps, self._mbd_run_steps)

        # BRT clocks from |φ| > 0.3 rad until roll returns inside the stable window.
        self._brt_active |= roll_abs >= PAPER_BRT_DISTURB_RAD
        self._brt_steps = torch.where(
            self._brt_active, self._brt_steps + 1, torch.zeros_like(self._brt_steps)
        )
        brt_recovered = self._brt_active & (roll_abs < self.stable_roll_threshold_rad)
        self._brt_sum_s += torch.where(
            brt_recovered,
            self._brt_steps.to(self._brt_sum_s.dtype) * self.step_dt,
            torch.zeros_like(self._brt_sum_s),
        )
        self._brt_events += brt_recovered.long()
        self._brt_active &= ~brt_recovered
        self._brt_steps = torch.where(
            brt_recovered, torch.zeros_like(self._brt_steps), self._brt_steps
        )

        self._min_pitch = torch.minimum(self._min_pitch, pitch_deg)
        self._max_pitch = torch.maximum(self._max_pitch, pitch_deg)
        self._sum_pitch += pitch_deg
        self._sum_pitch_sq += pitch_deg ** 2

        # ---- C. Velocity & dynamic tracking --------------------------------
        lin_vel_mag = torch.norm(root_lin_vel_b[:, :2], dim=-1)  # (N,)  XY speed
        vx = root_lin_vel_b[:, 0]
        wz = root_ang_vel_b[:, 2]
        self._min_vel = torch.minimum(self._min_vel, lin_vel_mag)
        self._max_vel = torch.maximum(self._max_vel, lin_vel_mag)
        self._sum_vel += lin_vel_mag

        vx_err = vx - target_velocity
        self._sum_vel_err_sq += vx_err**2
        self._sum_vx_err_sq += vx_err**2
        self._sum_vx_abs_err += vx_err.abs()
        self._sum_cmd_vx += target_velocity

        # SRL: time from each vx* change until vx first enters the ±10% band.
        cmd_changed = ((target_velocity - self._srl_prev_cmd).abs() > 1e-4) | torch.isnan(
            self._srl_prev_cmd
        )
        self._srl_prev_cmd = target_velocity.clone()
        self._srl_pending |= cmd_changed
        self._srl_steps = torch.where(
            cmd_changed, torch.zeros_like(self._srl_steps), self._srl_steps
        )
        self._srl_steps = torch.where(
            self._srl_pending, self._srl_steps + 1, self._srl_steps
        )
        srl_settled = self._srl_pending & (
            vx_err.abs() <= PAPER_SRL_TOLERANCE * target_velocity.abs().clamp(min=1e-3)
        )
        self._srl_sum_s += torch.where(
            srl_settled,
            self._srl_steps.to(self._srl_sum_s.dtype) * self.step_dt,
            torch.zeros_like(self._srl_sum_s),
        )
        self._srl_events += srl_settled.long()
        self._srl_pending &= ~srl_settled
        self._sum_wz_err_sq += (wz - wz_cmd) ** 2
        heading_err = _wrap_to_pi(heading_cmd - yaw)
        self._sum_heading_err_sq += heading_err**2
        self._sum_roll_sq_rad += roll ** 2
        is_straight = wz_cmd.abs() < self.straight_wz_threshold
        self._sum_roll_sq_straight += torch.where(is_straight, roll ** 2, torch.zeros_like(roll))
        self._straight_steps += is_straight.long()
        is_ss = self._step_count >= self.ss_delay_steps
        self._sum_vx_abs_err_ss += torch.where(is_ss, vx_err.abs(), torch.zeros_like(vx_err))
        self._ss_steps += is_ss.long()

        # ---- D. Actuation, smoothness & energy ----------------------------
        zeros_n = torch.zeros(num_envs, device=device)
        steer_raw = self._action_channel(actions, self.steer_action_index)
        drive_raw = self._action_channel(actions, self.drive_action_index)
        steer_act = steer_raw.abs()
        drive_act = drive_raw.abs()
        self._steering_active_steps += (steer_act > 0.01).long()
        self._drive_active_steps += (drive_act > 0.01).long()

        step_jitter = zeros_n
        steer_jit = zeros_n
        drive_jit = zeros_n
        if self._prev_action is not None:
            step_jitter = torch.norm(actions - self._prev_action, p=2, dim=-1)
            self._sum_action_jitter += step_jitter
            self._max_action_jitter = torch.maximum(self._max_action_jitter, step_jitter)
            steer_jit = (steer_raw - self._action_channel(self._prev_action, self.steer_action_index)).abs()
            drive_jit = (drive_raw - self._action_channel(self._prev_action, self.drive_action_index)).abs()
            self._sum_steer_jitter += steer_jit
            self._sum_drive_jitter += drive_jit
            self._drive_bang_steps += (drive_jit > self.drive_bang_threshold).long()
            self._jitter_steps += 1
        prev_for_hist = self._prev_action
        self._prev_action = actions.clone()

        steer_sat_act = steer_act >= self.action_sat_threshold
        drive_cmd_vel = drive_act * self.drive_action_scale
        drive_clip_sat = drive_cmd_vel >= (self.action_sat_threshold * self.drive_vel_clip)
        self._steer_sat_steps += steer_sat_act.long()
        self._drive_clip_steps += drive_clip_sat.long()
        # Drive |a| can exceed 1 when scale << required ω*. Saturation is:
        # steer |a| ≥ threshold, or |ω*| hitting the resolved drive velocity clip.
        self._action_sat_steps += (steer_sat_act | drive_clip_sat).long()

        steer_acc = joint_acc[:, steer_idx].abs()
        drive_acc = joint_acc[:, drive_idx].abs()
        joint_acc_mag = steer_acc + drive_acc
        self._sum_joint_acc_mag += joint_acc_mag
        self._max_joint_acc_mag = torch.maximum(self._max_joint_acc_mag, joint_acc_mag)

        # Drive motor speed tracking
        drive_vel_abs = joint_vel[:, drive_idx].abs()
        self._sum_drive_vel += drive_vel_abs
        self._max_drive_vel = torch.maximum(self._max_drive_vel, drive_vel_abs)

        # Mechanical power: sum(|torque_i * velocity_i|) for the two joints.
        steer_torque = applied_torque[:, steer_idx].abs()
        drive_torque = applied_torque[:, drive_idx].abs()
        steer_power = steer_torque * joint_vel[:, steer_idx].abs()
        drive_power = drive_torque * joint_vel[:, drive_idx].abs()
        self._sum_power += (steer_power + drive_power) * self.step_dt  # integrate energy
        self._sum_tau_sq += applied_torque[:, steer_idx] ** 2 + applied_torque[:, drive_idx] ** 2

        # Torque saturation check.
        steer_sat = steer_torque >= 0.9 * self.effort_limit_steering
        drive_sat = drive_torque >= 0.9 * self.effort_limit_drive
        self._torque_sat_steps += (steer_sat | drive_sat).long()

        # ---- E. Stability window -------------------------------------------
        in_stable_window = roll_abs < self.stable_roll_threshold_rad
        self._stable_steps += in_stable_window.long()

        # ---- E2. Impulse / kick rejection ----------------------------------
        kick_active = getattr(env, "_eval_imp_active", None)
        kick_done = getattr(env, "_eval_imp_done", None)
        if kick_active is not None and kick_done is not None:
            post = kick_done | kick_active
            self._kick_seen = self._kick_seen | post
            self._kick_peak_roll = torch.where(
                post, torch.maximum(self._kick_peak_roll, roll_abs), self._kick_peak_roll
            )
            after = kick_done & ~self._kick_settled
            hold = torch.where(roll_abs < self.kick_settle_roll_rad, self._kick_hold + 1, torch.zeros_like(self._kick_hold))
            self._kick_hold = torch.where(after, hold, self._kick_hold)
            newly_settled = after & (self._kick_hold >= self.kick_settle_hold_steps)
            ep_t = (self._step_count + 1).to(dtype=torch.float32) * self.step_dt
            impulse_end = ep_t
            if hasattr(env, "_eval_imp_delay") and hasattr(env, "_eval_imp_duration"):
                impulse_end = env._eval_imp_delay + env._eval_imp_duration
            self._kick_settle_time = torch.where(newly_settled, ep_t - impulse_end, self._kick_settle_time)
            self._kick_settled = self._kick_settled | newly_settled
            sign = torch.sign(roll)
            crossed = after & (self._kick_prev_sign != 0) & (sign != 0) & (sign != self._kick_prev_sign)
            self._kick_oscillations += crossed.long()
            self._kick_prev_sign = torch.where(after, sign, self._kick_prev_sign)

        # ---- F. Increment step counter -------------------------------------
        self._step_count += 1

        # ---- G. Record history (env_0 only, down-sampled) ------------------
        if self.record_history and self._history_step % self.history_downsample == 0:
            env0 = 0
            self._history["time_s"].append(self._history_step * self.step_dt)
            self._history["pos_x"].append(current_pos_xy[env0, 0].item())
            self._history["pos_y"].append(current_pos_xy[env0, 1].item())
            self._history["roll_deg"].append(math.degrees(roll[env0].item()))
            self._history["pitch_deg"].append(math.degrees(pitch[env0].item()))
            self._history["yaw_deg"].append(math.degrees(yaw[env0].item()))
            self._history["velocity_ms"].append(vx[env0].item())
            self._history["target_velocity_ms"].append(target_velocity[env0].item())
            self._history["wz"].append(wz[env0].item())
            self._history["cmd_wz"].append(wz_cmd[env0].item())
            self._history["heading_err_deg"].append(math.degrees(heading_err[env0].item()))
            self._history["steer_torque"].append(applied_torque[env0, steer_idx].item())
            self._history["drive_torque"].append(applied_torque[env0, drive_idx].item())
            self._history["steer_action"].append(steer_raw[env0].item())
            self._history["drive_action"].append(drive_raw[env0].item())
            if prev_for_hist is not None:
                self._history["action_jitter"].append(step_jitter[env0].item())
            else:
                self._history["action_jitter"].append(0.0)
        self._history_step += 1

        # ---- H. Handle episode terminations (env.reset_buf) ----------------
        reset_buf = env.reset_buf  # True where the episode just ended (already reset in-sim).
        # Crash pose/commands are `_last_*` from the previous logger.step.
        if reset_buf.any():
            self._finalize_episodes(reset_buf, env)
            ctx = self._read_episode_context(env)
            self._write_context(reset_buf, ctx, "spawn")
            self._write_context(reset_buf, ctx, "last")
        alive = ~reset_buf
        if alive.any():
            terrain_level, terrain_type = self._read_terrain(env)
            self._write_context(alive, {
                "vx": vx,
                "yaw": yaw,
                "roll": roll,
                "cmd_vx": vx_cmd,
                "cmd_wz": wz_cmd,
                "cmd_heading": heading_cmd,
                "pos_x": current_pos_xy[:, 0],
                "pos_y": current_pos_xy[:, 1],
                "terrain_level": terrain_level,
                "terrain_type": terrain_type,
            }, "last")

    # ------------------------------------------------------------------
    # Finalise completed episodes
    # ------------------------------------------------------------------

    def _finalize_episodes(
        self, reset_buf: torch.Tensor, env: ManagerBasedRLEnv
    ) -> None:
        """Extract and archive metrics for environments whose episode just ended."""
        done_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)  # (K,)

        if done_ids.numel() == 0:
            return

        episode_lengths = self._step_count[done_ids]  # (K,)
        max_length = env.max_episode_length
        self._max_episode_length = int(max_length)

        for i_local, env_id in enumerate(done_ids):
            eid = int(env_id.item())
            n = max(int(episode_lengths[i_local].item()), 1)
            survived = bool(n >= max_length - 1)
            self._completed_episodes.append(
                self._snapshot_episode(eid, n, survived=survived, complete=True, env=env)
            )

        self._reset_row_buffers(done_ids, env)

    # ------------------------------------------------------------------
    # Finalize — flush remaining episodes and produce summary
    # ------------------------------------------------------------------

    def finalize(self, env: Any | None = None) -> "EvalSummary":
        """Flush still-running episodes as *censored* (not counted as survived)."""
        if env is not None:
            ctx = self._read_episode_context(env)
            running = self._step_count > 0
            if running.any():
                self._write_context(running, ctx, "last")
        max_length = self._max_episode_length
        for eid in range(self.num_envs):
            n = int(self._step_count[eid].item())
            if n == 0:
                continue
            timed_out = bool(max_length > 0 and n >= max_length - 1)
            self._completed_episodes.append(
                self._snapshot_episode(
                    eid,
                    n,
                    survived=timed_out,
                    complete=timed_out,
                    env=env,
                )
            )
        return self.compute_summary()

    # ------------------------------------------------------------------
    # Compute aggregate summary from all completed episodes
    # ------------------------------------------------------------------

    def compute_summary(self) -> "EvalSummary":
        """Aggregate all completed episodes into an ``EvalSummary``."""
        eps = self._completed_episodes
        if not eps:
            return EvalSummary(num_episodes=0)

        complete = [e for e in eps if e.complete]
        survived_eps = [e for e in complete if e.survived]
        n_complete = len(complete)
        n_survived = len(survived_eps)
        n_total = len(eps)
        n_censored = n_total - n_complete

        def _mean(vals: list[float]) -> float:
            return float(np.mean(vals)) if vals else 0.0

        def _std(vals: list[float]) -> float:
            return float(np.std(vals)) if len(vals) > 1 else 0.0

        def _max(vals: list[float]) -> float:
            return float(np.max(vals)) if vals else 0.0

        def _min(vals: list[float]) -> float:
            return float(np.min(vals)) if vals else 0.0

        def _thr_mean(key: str) -> float:
            vals = [e.survival_roll_thresholds.get(key, 0.0) for e in eps]
            return float(np.mean(vals)) if vals else 0.0

        kick_eps = [e for e in eps if e.kick_seen]
        kick_settle_vals = [
            e.kick_settle_time_s for e in kick_eps
            if e.kick_settled and math.isfinite(e.kick_settle_time_s)
        ]

        brt_vals = [e.brt_s for e in eps if e.brt_events and math.isfinite(e.brt_s)]
        srl_vals = [e.srl_s for e in eps if e.srl_events and math.isfinite(e.srl_s)]

        return EvalSummary(
            num_episodes=n_total,
            num_complete_episodes=n_complete,
            num_censored_episodes=n_censored,
            survival_rate=(n_survived / n_complete) if n_complete else 0.0,
            mean_episode_length_s=_mean([e.episode_time_s for e in eps]),
            std_episode_length_s=_std([e.episode_time_s for e in eps]),
            # Roll
            min_roll_deg=_min([e.min_roll_deg for e in eps]),
            mean_roll_deg=_mean([e.mean_roll_deg for e in eps]),
            max_roll_deg=_max([e.max_roll_deg for e in eps]),
            std_roll_deg_across=_std([e.mean_roll_deg for e in eps]),
            mean_std_roll_deg=_mean([e.std_roll_deg for e in eps]),
            max_lean_deg=_max([e.max_lean_deg for e in eps]),
            max_lean_survived_deg=_max([e.max_lean_deg for e in survived_eps]),
            mean_roll_rmse_deg=_mean([e.roll_rmse_deg for e in eps]),
            mean_roll_rmse_straight_deg=_mean([e.roll_rmse_straight_deg for e in eps]),
            roll_time_above_5deg=_thr_mean(">5°"),
            roll_time_above_10deg=_thr_mean(">10°"),
            roll_time_above_15deg=_thr_mean(">15°"),
            roll_time_above_20deg=_thr_mean(">20°"),
            roll_time_above_30deg=_thr_mean(">30°"),
            roll_time_above_45deg=_thr_mean(">45°"),
            mean_recovery_count=_mean([float(e.recovery_count) for e in eps]),
            max_recovery_count=int(_max([float(e.recovery_count) for e in eps])),
            mean_roll_excursion_deg_s=_mean([e.cumulative_roll_excursion_deg_s for e in eps]),
            # Pitch
            min_pitch_deg=_min([e.min_pitch_deg for e in eps]),
            mean_pitch_deg=_mean([e.mean_pitch_deg for e in eps]),
            max_pitch_deg=_max([e.max_pitch_deg for e in eps]),
            mean_std_pitch_deg=_mean([e.std_pitch_deg for e in eps]),
            mean_yaw_drift_rate_deg_per_s=_mean([e.yaw_drift_rate_deg_per_s for e in eps]),
            # Position & spatial
            min_drift_m=_min([e.min_drift_m for e in eps]),
            mean_max_drift_m=_mean([e.max_drift_m for e in eps]),
            std_max_drift_m=_std([e.max_drift_m for e in eps]),
            max_drift_m=_max([e.max_drift_m for e in eps]),
            mean_final_drift_m=_mean([e.final_drift_m for e in eps]),
            mean_lateral_rms_m=_mean([e.lateral_deviation_rms_m for e in eps]),
            # Velocity
            min_velocity_ms=_min([e.min_velocity_ms for e in eps]),
            mean_velocity_ms=_mean([e.mean_velocity_ms for e in eps]),
            max_velocity_ms=_max([e.max_velocity_ms for e in eps]),
            mean_velocity_rmse_ms=_mean([e.velocity_rmse_ms for e in eps]),
            mean_velocity_overshoot_pct=_mean([e.velocity_overshoot_pct for e in eps]),
            mean_vx_rmse_ms=_mean([e.vx_rmse_ms for e in eps]),
            mean_vx_mae_ms=_mean([e.vx_mae_ms for e in eps]),
            mean_vx_ss_mae_ms=_mean([e.vx_ss_mae_ms for e in eps]),
            mean_wz_rmse_rad_s=_mean([e.wz_rmse_rad_s for e in eps]),
            mean_heading_rmse_deg=_mean([e.heading_rmse_deg for e in eps]),
            mean_cmd_vx_ms=_mean([e.mean_cmd_vx_ms for e in eps]),
            # Actuation & energy
            mean_action_jitter=_mean([e.mean_action_jitter for e in eps]),
            max_action_jitter=_max([e.max_action_jitter for e in eps]),
            mean_steer_action_jitter=_mean([e.steer_action_jitter for e in eps]),
            mean_drive_action_jitter=_mean([e.drive_action_jitter for e in eps]),
            mean_joint_acc_mag=_mean([e.mean_joint_acc_mag for e in eps]),
            max_joint_acc_mag=_max([e.max_joint_acc_mag for e in eps]),
            mean_total_power_j=_mean([e.total_power_j for e in eps]),
            mean_control_effort=_mean([e.control_effort for e in eps]),
            mean_tau_sq=_mean([e.mean_tau_sq for e in eps]),
            mean_steering_usage_ratio=_mean([e.steering_usage_ratio for e in eps]),
            mean_drive_usage_ratio=_mean([e.drive_usage_ratio for e in eps]),
            mean_drive_speed_rad_s=_mean([e.mean_drive_speed_rad_s for e in eps]),
            max_drive_speed_rad_s=_max([e.max_drive_speed_rad_s for e in eps]),
            mean_torque_saturation_ratio=_mean([e.torque_saturation_ratio for e in eps]),
            max_torque_saturation_ratio=_max([e.torque_saturation_ratio for e in eps]),
            mean_action_clip_sat_ratio=_mean([e.action_clip_sat_ratio for e in eps]),
            mean_steer_clip_sat_ratio=_mean([e.steer_clip_sat_ratio for e in eps]),
            mean_drive_clip_sat_ratio=_mean([e.drive_clip_sat_ratio for e in eps]),
            # Stability
            mean_stable_balance_ratio=_mean([e.stable_balance_ratio for e in eps]),
            min_stable_balance_ratio=_min([e.stable_balance_ratio for e in eps]) if eps else 0.0,
            # Kick
            kick_episodes=len(kick_eps),
            kick_settle_rate=(
                len([e for e in kick_eps if e.kick_settled]) / len(kick_eps) if kick_eps else 0.0
            ),
            mean_kick_settle_time_s=_mean(kick_settle_vals),
            mean_kick_peak_roll_deg=_mean([e.kick_peak_roll_deg for e in kick_eps]),
            mean_kick_oscillations=_mean([float(e.kick_oscillations) for e in kick_eps]),
            # Paper Table II
            bsr=balance_success_rate(eps),
            brt_s=_mean(brt_vals) if brt_vals else float("nan"),
            mbd_s=_max([e.max_balance_duration_s for e in eps]),
            cat_deg=analyze_critical_angle(eps),
            ste_deg=_mean([e.heading_rmse_deg for e in eps]),
            vte_ms=_mean([e.vx_mae_ms for e in eps]),
            srl_s=_mean(srl_vals) if srl_vals else float("nan"),
            mnt=analyze_noise_tolerance(eps),
            mss_ms=analyze_min_sustaining_speed(eps),
            failure_analysis=analyze_survival_thresholds(complete),
        )

    # ------------------------------------------------------------------
    # Property — access episode archive
    # ------------------------------------------------------------------

    @property
    def episodes(self) -> list[EpisodeMetrics]:
        """Return the list of all completed episode metrics so far."""
        return self._completed_episodes

    @property
    def history(self) -> dict[str, list[float]]:
        """Return the down-sampled history dict for env_0 (plotting data)."""
        return dict(self._history)


# ===========================================================================
# 3b. Paper Table II frontier analyzers (CAT / MNT / MSS)
# ===========================================================================


# CAT / MNT / MSS / BSR helpers are imported from ``paper_metrics`` above.


# ===========================================================================
# 4. EvalSummary — aggregate metrics across all episodes
# ===========================================================================


@dataclass
class EvalSummary:
    """Aggregate evaluation summary with max/mean/min/std across all episodes."""

    num_episodes: int
    num_complete_episodes: int = 0
    num_censored_episodes: int = 0
    survival_rate: float = 0.0
    mean_episode_length_s: float = 0.0
    std_episode_length_s: float = 0.0

    # -- Paper Table II -------------------------------------------------------
    bsr: float = 0.0  # Balance Success Rate — |φ| < 0.5 rad for the whole episode.
    brt_s: float = float("nan")  # Balance Recovery Time from |φ| > 0.3 rad.
    mbd_s: float = 0.0  # Maximum Balance Duration.
    cat_deg: float = float("nan")  # Critical Angle Tolerance (needs a roll ladder).
    ste_deg: float = 0.0  # Steering Tracking Error (RMS heading error).
    vte_ms: float = 0.0  # Velocity Tracking Error (MAE longitudinal velocity).
    srl_s: float = float("nan")  # System Response Latency to the ±10% vx* band.
    mnt: float = float("nan")  # Max Noise Tolerance (needs a noise ladder).
    mss_ms: float = float("nan")  # Min Sustaining Speed (needs a speed ladder).

    # -- Roll ---------------------------------------------------------------
    min_roll_deg: float = 0.0      # min across episodes
    mean_roll_deg: float = 0.0     # mean across episodes
    max_roll_deg: float = 0.0      # max across episodes
    std_roll_deg_across: float = 0.0
    mean_std_roll_deg: float = 0.0  # mean of within-episode roll std
    max_lean_deg: float = 0.0       # single worst lean angle ever seen
    max_lean_survived_deg: float = 0.0  # worst lean among timeout survivors
    mean_roll_rmse_deg: float = 0.0
    mean_roll_rmse_straight_deg: float = 0.0
    # Roll survival thresholds (mean across episodes)
    roll_time_above_5deg: float = 0.0
    roll_time_above_10deg: float = 0.0
    roll_time_above_15deg: float = 0.0
    roll_time_above_20deg: float = 0.0
    roll_time_above_30deg: float = 0.0
    roll_time_above_45deg: float = 0.0
    mean_recovery_count: float = 0.0
    max_recovery_count: int = 0
    mean_roll_excursion_deg_s: float = 0.0

    # -- Pitch --------------------------------------------------------------
    min_pitch_deg: float = 0.0
    mean_pitch_deg: float = 0.0
    max_pitch_deg: float = 0.0
    mean_std_pitch_deg: float = 0.0
    mean_yaw_drift_rate_deg_per_s: float = 0.0

    # -- Position -----------------------------------------------------------
    min_drift_m: float = 0.0
    mean_max_drift_m: float = 0.0
    std_max_drift_m: float = 0.0
    max_drift_m: float = 0.0
    mean_final_drift_m: float = 0.0
    mean_lateral_rms_m: float = 0.0

    # -- Velocity -----------------------------------------------------------
    min_velocity_ms: float = 0.0
    mean_velocity_ms: float = 0.0
    max_velocity_ms: float = 0.0
    mean_velocity_rmse_ms: float = 0.0
    mean_velocity_overshoot_pct: float = 0.0
    mean_vx_rmse_ms: float = 0.0
    mean_vx_mae_ms: float = 0.0
    mean_vx_ss_mae_ms: float = 0.0
    mean_wz_rmse_rad_s: float = 0.0
    mean_heading_rmse_deg: float = 0.0
    mean_cmd_vx_ms: float = 0.0

    # -- Actuation & energy -------------------------------------------------
    mean_action_jitter: float = 0.0
    max_action_jitter: float = 0.0
    mean_steer_action_jitter: float = 0.0
    mean_drive_action_jitter: float = 0.0
    mean_joint_acc_mag: float = 0.0
    max_joint_acc_mag: float = 0.0
    mean_total_power_j: float = 0.0
    mean_control_effort: float = 0.0
    mean_tau_sq: float = 0.0
    mean_steering_usage_ratio: float = 0.0
    mean_drive_usage_ratio: float = 0.0
    mean_drive_speed_rad_s: float = 0.0
    max_drive_speed_rad_s: float = 0.0
    mean_torque_saturation_ratio: float = 0.0
    max_torque_saturation_ratio: float = 0.0
    mean_action_clip_sat_ratio: float = 0.0
    mean_steer_clip_sat_ratio: float = 0.0
    mean_drive_clip_sat_ratio: float = 0.0

    # -- Stability ----------------------------------------------------------
    mean_stable_balance_ratio: float = 0.0
    min_stable_balance_ratio: float = 0.0

    # -- Kick / impulse -----------------------------------------------------
    kick_episodes: int = 0
    kick_settle_rate: float = 0.0
    mean_kick_settle_time_s: float = 0.0
    mean_kick_peak_roll_deg: float = 0.0
    mean_kick_oscillations: float = 0.0
    failure_analysis: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        """Return summary as a flat dictionary for serialisation."""
        return {f.name: getattr(self, f.name)
                for f in dataclasses.fields(self)
                if isinstance(getattr(self, f.name), (bool, int, float))}

    def to_isaaclab_log_dict(self) -> dict[str, float]:
        """Return metrics for TensorBoard / WandB."""
        d = self.to_dict()
        d.pop("num_episodes", None)
        return {f"eval/{k}": v for k, v in d.items()}

    @staticmethod
    def _fmt(val: float, unit: str = "", precision: int = 3) -> str:
        if abs(val) < 1e-9:
            return f"{0:{precision}.{precision}f}{unit}"
        return f"{val:{precision}.{precision}f}{unit}"

    def pretty_print(self) -> str:
        """Rich sectioned table with max / mean / min for every property."""
        s = self
        sep = "=" * 74

        lines = [sep, "  BICYCLE EVALUATION SUMMARY", sep, ""]

        # -- Header ----------------------------------------------------------
        lines.append(f"  Episodes: {s.num_episodes} ({s.num_complete_episodes} complete, "
                     f"{s.num_censored_episodes} censored)  |  "
                     f"Survival: {s.survival_rate:.1%}  |  "
                     f"Length: {s.mean_episode_length_s:.1f} ± {s.std_episode_length_s:.1f} s")
        lines.append(f"  Task: dynamic locomotion (trackstand N/A)  |  "
                     f"cmd vx: {s.mean_cmd_vx_ms:.2f} m/s")
        lines.append("")

        # Helper for 4-column rows
        def _row(label, lo, mid, hi, unit="", prec=3):
            return f"  {label:<28s} {lo:>{prec+len(unit)+1}.{prec}f}{unit}  {mid:>{prec+len(unit)+1}.{prec}f}{unit}  {hi:>{prec+len(unit)+1}.{prec}f}{unit}"

        def _row2(label, v1, v2):
            s1 = v1 if isinstance(v1, str) else f"{v1:.3f}"
            s2 = v2 if isinstance(v2, str) else f"{v2:.3f}"
            return f"  {label:<28s} {s1}   {s2}"

        # -- PAPER TABLE II ---------------------------------------------------
        def _opt(value, fmt, suffix=""):
            return f"{value:{fmt}}{suffix}" if math.isfinite(value) else "n/a"

        lines.append(f"{'':─^74}")
        lines.append(f"{'PAPER TABLE II METRICS':^74}")
        lines.append(f"{'':─^74}")
        lines.append(_row2("BSR (|φ| < 0.5 rad)", f"{s.bsr:.2%}", f"BRT: {_opt(s.brt_s, '.2f', ' s')}"))
        lines.append(_row2("MBD (longest balance)", f"{s.mbd_s:.1f} s", f"CAT: {_opt(s.cat_deg, '.1f', '°')}"))
        lines.append(_row2("STE (steering RMS)", f"{s.ste_deg:.2f}°", f"VTE: {s.vte_ms:.3f} m/s"))
        lines.append(_row2("SRL (to ±10% of vx*)", _opt(s.srl_s, '.2f', ' s'), f"MNT: {_opt(s.mnt, '.2f')}"))
        lines.append(_row2("MSS (min sustaining spd)", _opt(s.mss_ms, '.2f', ' m/s'), ""))
        lines.append("")

        # -- ROLL STABILITY --------------------------------------------------
        lines.append(f"{'':─^74}")
        lines.append(f"{'ROLL STABILITY':^74}")
        lines.append(f"{'':─^74}")
        lines.append(f"  {'':28s}  {'min':>8s}  {'mean':>8s}  {'max':>8s}")
        lines.append(_row("Roll Angle (°)", s.min_roll_deg, s.mean_roll_deg, s.max_roll_deg, prec=2))
        lines.append(_row2("Within-Ep Roll Std (°)", f"{s.mean_std_roll_deg:.2f}", f"max lean: {s.max_lean_deg:.1f}°"))
        lines.append(_row2("Roll RMSE from 0°", f"{s.mean_roll_rmse_deg:.2f}°", f"straight: {s.mean_roll_rmse_straight_deg:.2f}°"))
        lines.append("")

        # -- ROLL SURVIVAL THRESHOLDS ----------------------------------------
        lines.append(f"{'':─^74}")
        lines.append(f"{'ROLL SURVIVAL — time spent above threshold':^74}")
        lines.append(f"{'':─^74}")
        for thr in [5, 10, 15, 20, 30, 45]:
            key = f"roll_time_above_{thr}deg"
            pct = getattr(s, key, 0.0)
            bar = "█" * int(pct * 40) if pct > 0 else ""
            lines.append(f"  > {thr:2d}°   {pct:6.1%}  {bar}")
        lines.append("")

        # -- RECOVERY --------------------------------------------------------
        lines.append(f"{'':─^74}")
        lines.append(f"{'RECOVERY':^74}")
        lines.append(f"{'':─^74}")
        lines.append(_row2("Recovery Events (mean/max)", f"{s.mean_recovery_count:.1f}", f"max: {s.max_recovery_count}"))
        lines.append(f"  Roll Excursion (°·s)             {s.mean_roll_excursion_deg_s:.2f}")
        lines.append("")

        # -- PITCH -----------------------------------------------------------
        lines.append(f"{'':─^74}")
        lines.append(f"{'PITCH STABILITY':^74}")
        lines.append(f"{'':─^74}")
        lines.append(_row("Pitch Angle (°)", s.min_pitch_deg, s.mean_pitch_deg, s.max_pitch_deg, prec=2))
        lines.append(_row2("Within-Ep Pitch Std (°)", f"{s.mean_std_pitch_deg:.2f}", f"yaw drift: {s.mean_yaw_drift_rate_deg_per_s:.3f} °/s"))
        lines.append("")

        # -- POSITION --------------------------------------------------------
        lines.append(f"{'':─^74}")
        lines.append(f"{'POSITION & DRIFT':^74}")
        lines.append(f"{'':─^74}")
        lines.append(_row("Drift from Spawn (m)", s.min_drift_m, s.mean_max_drift_m, s.max_drift_m, prec=3))
        lines.append(_row2("Final Drift (m)", f"{s.mean_final_drift_m:.3f}", f"lateral RMS: {s.mean_lateral_rms_m:.3f}"))
        lines.append("")

        # -- VELOCITY --------------------------------------------------------
        lines.append(f"{'':─^74}")
        lines.append(f"{'VELOCITY & TRACKING':^74}")
        lines.append(f"{'':─^74}")
        lines.append(_row("Velocity (m/s)", s.min_velocity_ms, s.mean_velocity_ms, s.max_velocity_ms, prec=3))
        lines.append(_row2("vx RMSE / MAE (m/s)", f"{s.mean_vx_rmse_ms:.3f}", f"MAE: {s.mean_vx_mae_ms:.3f}"))
        lines.append(_row2("vx steady-state MAE", f"{s.mean_vx_ss_mae_ms:.3f} m/s", f"gate: < 0.10 m/s"))
        lines.append(_row2("ωz RMSE (rad/s)", f"{s.mean_wz_rmse_rad_s:.3f}", f"heading RMSE: {s.mean_heading_rmse_deg:.2f}°"))
        lines.append(_row2("Lateral RMS (m)", f"{s.mean_lateral_rms_m:.3f}", f"overshoot: {s.mean_velocity_overshoot_pct:.1f}%"))
        lines.append("")

        # -- ACTUATION -------------------------------------------------------
        lines.append(f"{'':─^74}")
        lines.append(f"{'ACTUATION & ENERGY':^74}")
        lines.append(f"{'':─^74}")
        lines.append(_row2("Action Jitter (mean/max)", f"{s.mean_action_jitter:.4f}", f"max: {s.max_action_jitter:.4f}"))
        lines.append(_row2("Steer / drive |Δa|", f"{s.mean_steer_action_jitter:.4f}", f"drive: {s.mean_drive_action_jitter:.4f}"))
        lines.append(_row2("Joint Acc (mean/max rad/s²)", f"{s.mean_joint_acc_mag:.3f}", f"max: {s.max_joint_acc_mag:.3f}"))
        lines.append(_row2("Steering Usage", f"{s.mean_steering_usage_ratio:.1%}", f"drive: {s.mean_drive_usage_ratio:.1%}"))
        lines.append(_row2("Drive Speed (mean/max rad/s)", f"{s.mean_drive_speed_rad_s:.2f}", f"max: {s.max_drive_speed_rad_s:.2f}"))
        lines.append(_row2("Total Power (J)", f"{s.mean_total_power_j:.2f}", f"Στ²Δt: {s.mean_control_effort:.2f}"))
        lines.append(_row2("Action clip sat (steer/drive)", f"{s.mean_steer_clip_sat_ratio:.1%}", f"drive clip: {s.mean_drive_clip_sat_ratio:.1%}"))
        lines.append(_row2("Torque sat (mean/max)", f"{s.mean_torque_saturation_ratio:.1%}", f"max: {s.max_torque_saturation_ratio:.1%}"))
        lines.append("")

        # -- STABILITY -------------------------------------------------------
        lines.append(f"{'':─^74}")
        lines.append(f"{'STABILITY':^74}")
        lines.append(f"{'':─^74}")
        lines.append(_row2("Stable Balance Ratio (mean/min)", f"{s.mean_stable_balance_ratio:.1%}", f"min: {s.min_stable_balance_ratio:.1%}"))
        if s.kick_episodes:
            lines.append(_row2("Kick settle rate / time", f"{s.kick_settle_rate:.1%}", f"{s.mean_kick_settle_time_s:.2f} s"))
            lines.append(_row2("Kick peak roll / osc.", f"{s.mean_kick_peak_roll_deg:.1f}°", f"{s.mean_kick_oscillations:.1f}"))
        lines.append("")

        # -- FAILURE THRESHOLDS ----------------------------------------------
        lines.append(f"{'':─^74}")
        lines.append(f"{'FAILURE THRESHOLDS — spawn / command / time-to-crash':^74}")
        lines.append(f"{'':─^74}")
        lines.append(format_failure_report(s.failure_analysis))
        lines.append("")
        lines.append(sep)

        return "\n".join(lines)


def _terrain_stage_thresholds(mode: str) -> dict[str, float] | None:
    """Relaxed locomotion gates that get looser as the pinned row gets harder."""
    if mode.startswith("complex_L"):
        try:
            level = int(mode.split("_L", 1)[1])
        except ValueError:
            return None
        surv = (0.90, 0.80, 0.70, 0.55, 0.40)
        roll = (5.0, 6.0, 8.0, 10.0, 12.0)
        vx = (0.15, 0.20, 0.25, 0.30, 0.40)
        heading = (12.0, 15.0, 18.0, 22.0, 25.0)
        steer_sat = (0.08, 0.10, 0.12, 0.15, 0.20)
        if not 0 <= level < len(surv):
            return None
        return {
            "surv": surv[level],
            "roll": roll[level],
            "vx": vx[level],
            "heading": heading[level],
            "steer_sat": steer_sat[level],
        }
    if mode.startswith("stairs_L"):
        try:
            level = int(mode.split("_L", 1)[1])
        except ValueError:
            return None
        surv = (0.90, 0.75, 0.55)
        roll = (6.0, 8.0, 12.0)
        vx = (0.15, 0.25, 0.35)
        heading = (12.0, 18.0, 25.0)
        steer_sat = (0.08, 0.12, 0.18)
        if not 0 <= level < len(surv):
            return None
        return {
            "surv": surv[level],
            "roll": roll[level],
            "vx": vx[level],
            "heading": heading[level],
            "steer_sat": steer_sat[level],
        }
    return None


def evaluate_gates(summary: "EvalSummary", mode: str = "nominal") -> dict[str, Any]:
    """Score a locomotion checkpoint against deployment gates.

    Trackstand / stationary-balance gates are omitted: this policy family
    only covers dynamic balancing while moving.
    """
    gates: list[dict[str, Any]] = []

    def _add(name: str, value: float, passed: bool, threshold: str, unit: str = "") -> None:
        gates.append({
            "name": name,
            "value": value,
            "passed": passed,
            "threshold": threshold,
            "unit": unit,
            "na": False,
        })

    def _na(name: str, reason: str) -> None:
        gates.append({
            "name": name,
            "value": None,
            "passed": None,
            "threshold": reason,
            "unit": "",
            "na": True,
        })

    _na("Trackstand roll RMSE < 2°", "N/A — policy is locomotion-only")

    stage_gates = _terrain_stage_thresholds(mode)
    if stage_gates is not None:
        _add(
            "Survival rate",
            summary.survival_rate,
            summary.survival_rate >= stage_gates["surv"],
            f">= {stage_gates['surv']:.0%}",
        )
        _add(
            "Roll RMSE (straightaways)",
            summary.mean_roll_rmse_straight_deg,
            summary.mean_roll_rmse_straight_deg < stage_gates["roll"],
            f"< {stage_gates['roll']:.0f}°",
            "°",
        )
        _add(
            "vx steady-state MAE",
            summary.mean_vx_ss_mae_ms,
            summary.mean_vx_ss_mae_ms < stage_gates["vx"],
            f"< {stage_gates['vx']:.2f} m/s",
            "m/s",
        )
        _add(
            "Heading RMSE",
            summary.mean_heading_rmse_deg,
            summary.mean_heading_rmse_deg < stage_gates["heading"],
            f"< {stage_gates['heading']:.0f}°",
            "°",
        )
        _add(
            "Steer action saturation",
            summary.mean_steer_clip_sat_ratio,
            summary.mean_steer_clip_sat_ratio < stage_gates["steer_sat"],
            f"< {stage_gates['steer_sat']:.0%}",
        )
    elif mode.startswith("paper_"):
        # Table VI / Table II: rank on BSR, not the 45° crash survival used by ladders.
        bsr_min = 0.95 if mode in (
            "paper_no_noise",
            "paper_flat",
            "paper_low_speed",
            "paper_normal_speed",
        ) else 0.75
        _add("BSR (|φ| < 0.5 rad)", summary.bsr, summary.bsr >= bsr_min, f">= {bsr_min:.0%}")
        _add("STE (heading RMS)", summary.ste_deg, summary.ste_deg <= 3.0, "<= 3.0°", "°")
        _add("VTE (vx MAE)", summary.vte_ms, summary.vte_ms <= 0.20, "<= 0.20 m/s", "m/s")
    elif mode in ("nominal", "training"):
        surv_min = 0.98 if mode == "nominal" else 0.90
        _add("Survival rate", summary.survival_rate, summary.survival_rate >= surv_min,
             f">= {surv_min:.0%}")
        _add("Roll RMSE (straightaways)", summary.mean_roll_rmse_straight_deg,
             summary.mean_roll_rmse_straight_deg < 5.0, "< 5°", "°")
        _add("Max transient roll (all)", summary.max_lean_deg, summary.max_lean_deg < 15.0, "< 15°", "°")
        _add("Max transient roll (survived)", getattr(summary, "max_lean_survived_deg", summary.max_lean_deg),
             getattr(summary, "max_lean_survived_deg", summary.max_lean_deg) < 15.0, "< 15°", "°")
        _add("vx steady-state MAE", summary.mean_vx_ss_mae_ms, summary.mean_vx_ss_mae_ms < 0.10,
             "< 0.10 m/s", "m/s")
        _add("Heading RMSE", summary.mean_heading_rmse_deg, summary.mean_heading_rmse_deg < 10.0,
             "< 10°", "°")
        _add("Steer action saturation", summary.mean_steer_clip_sat_ratio,
             summary.mean_steer_clip_sat_ratio < 0.05, "< 5%")
        _add("Drive velocity-clip saturation", summary.mean_drive_clip_sat_ratio,
             summary.mean_drive_clip_sat_ratio < 0.05, "< 5%")
        _add("Torque saturation", summary.mean_torque_saturation_ratio,
             summary.mean_torque_saturation_ratio < 0.05, "< 5%")
    elif mode == "kick":
        _add("Survival rate", summary.survival_rate, summary.survival_rate >= 0.90, ">= 90%")
        _add("Kick settle rate", summary.kick_settle_rate, summary.kick_settle_rate >= 0.90, ">= 90%")
        settle_ok = (
            summary.kick_episodes == 0
            or (summary.mean_kick_settle_time_s > 0 and summary.mean_kick_settle_time_s <= 2.0)
        )
        _add("Kick settle time", summary.mean_kick_settle_time_s, settle_ok, "<= 2.0 s", "s")
        _add("Kick oscillations", summary.mean_kick_oscillations,
             summary.mean_kick_oscillations <= 2.0, "<= 2")
        _add("Kick peak roll", summary.mean_kick_peak_roll_deg,
             summary.mean_kick_peak_roll_deg < 15.0, "< 15°", "°")
    elif mode == "mass":
        _add("Survival rate (mass+CoM)", summary.survival_rate, summary.survival_rate >= 0.90, ">= 90%")
        _add("Roll RMSE (straightaways)", summary.mean_roll_rmse_straight_deg,
             summary.mean_roll_rmse_straight_deg < 8.0, "< 8° (relaxed)", "°")
        _add("vx steady-state MAE", summary.mean_vx_ss_mae_ms, summary.mean_vx_ss_mae_ms < 0.20,
             "< 0.20 m/s", "m/s")
    elif mode == "friction":
        _add("Survival rate (μ=0.4)", summary.survival_rate, summary.survival_rate >= 0.90, ">= 90%")
        _add("vx steady-state MAE", summary.mean_vx_ss_mae_ms, summary.mean_vx_ss_mae_ms < 0.20,
             "< 0.20 m/s", "m/s")
        _add("Drive clip saturation", summary.mean_drive_clip_sat_ratio,
             summary.mean_drive_clip_sat_ratio < 0.10, "< 10%")
    elif mode == "speed_sweep":
        _na("Standard locomotion gates", "N/A — see speed_limits.json / console report")
    elif mode == "heading_sweep":
        _na("Standard locomotion gates", "N/A — see heading_limits.json / console report")
    elif mode == "yaw_bias_sweep":
        _na("Standard locomotion gates", "N/A — see yaw_bias.json / console report")
    elif mode == "drive_accel_sweep":
        _na("Standard locomotion gates", "N/A — see drive_accel.json / console report")
    elif mode == "mass_sweep":
        _na("Standard locomotion gates", "N/A — see mass_limits.json / console report")
    elif mode == "com_sweep":
        _na("Standard locomotion gates", "N/A — see com_limits.json / console report")
    elif mode == "mass_com_sweep":
        _na("Standard locomotion gates", "N/A — see mass_com_limits.json / console report")
    elif mode == "friction_sweep":
        _na("Standard locomotion gates", "N/A — see friction_limits.json / console report")
    elif mode == "kick_sweep":
        _na("Standard locomotion gates", "N/A — see kick_limits.json / console report")
    elif mode == "teleop":
        _na("Standard locomotion gates", "N/A — free teleop; see summary / plots")

    scored = [g for g in gates if not g["na"]]
    n_pass = sum(1 for g in scored if g["passed"])
    report = {
        "mode": mode,
        "trackstand": False,
        "n_gates": len(scored),
        "n_pass": n_pass,
        # Sweep/envelope modes only emit N/A placeholders — treat as pass.
        "all_passed": (len(scored) == 0) or (n_pass == len(scored)),
        "gates": gates,
    }
    return report


def format_gate_report(report: dict[str, Any]) -> str:
    """Pretty-print ``evaluate_gates`` output."""
    sep = "=" * 74
    lines = [sep, f"  DEPLOYMENT GATES — {report['mode']}  (locomotion; trackstand N/A)", sep]
    for g in report["gates"]:
        if g["na"]:
            mark = "N/A "
            val = "—"
        else:
            mark = "PASS" if g["passed"] else "FAIL"
            val = f"{g['value']:.4g}{g['unit']}"
        lines.append(f"  [{mark}]  {g['name']:<32s}  {val:>12s}   ({g['threshold']})")
    lines.append("")
    lines.append(f"  Result: {report['n_pass']}/{report['n_gates']} gates passed"
                 + ("  — READY" if report["all_passed"] else "  — NOT READY"))
    lines.append(sep)
    return "\n".join(lines)


# ===========================================================================
# 5. BicycleEvalPlotter — matplotlib visualisations
# ===========================================================================


class BicycleEvalPlotter:
    """Generates high-quality matplotlib plots from logged history data.

    All plots are saved as high-resolution PNG and PDF files.

    Typical usage::

        plotter = BicycleEvalPlotter(logger, save_dir="./eval_plots")
        plotter.generate_all_plots()
    """

    # seaborn-style colour palette — colourblind-friendly.
    COLORS = {
        "blue": "#4C72B0",
        "orange": "#DD8452",
        "green": "#55A868",
        "red": "#C44E52",
        "purple": "#8172B3",
        "grey": "#8C8C8C",
        "brown": "#937860",
        "pink": "#CCB974",
    }

    def __init__(
        self,
        logger: BicycleMetricsLogger,
        save_dir: str = "./eval_plots",
        dpi: int = 150,
        figsize: tuple[int, int] = (10, 6),
        style: str = "seaborn-v0_8-whitegrid",
        interactive: bool = True,
    ) -> None:
        """
        Args:
            logger: A ``BicycleMetricsLogger`` that has accumulated history data.
            save_dir: Directory to save plots into (created if missing).
            dpi: Resolution for saved figures.
            figsize: Default figure size in inches (width, height).
            style: Matplotlib style name.
            interactive: If True, also write zoomable Plotly HTML under ``plotly/``.
        """
        if not _HAS_MPL:
            raise ImportError(
                "matplotlib is required for plotting.  Install with: "
                "pip install matplotlib"
            )
        self.logger = logger
        self.save_dir = save_dir
        self.dpi = dpi
        self.figsize = figsize
        self.interactive = interactive
        os.makedirs(save_dir, exist_ok=True)
        plt.style.use(style)

    # ------------------------------------------------------------------
    # 5a. 2D Trajectory / Drift Map
    # ------------------------------------------------------------------

    def plot_2d_trajectory(self, filename: str = "trajectory_2d") -> str:
        """Top-down XY path showing the bicycle's wander radius.

        For station-keeping tasks, this visualises the drift pattern around
        the spawn origin.  The "wander radius" (maximum drift) is drawn as
        a dashed circle.

        Args:
            filename: Base filename (without extension).

        Returns:
            Absolute path to the saved PNG file.
        """
        hist = self.logger.history
        xs = np.array(hist.get("pos_x", []))
        ys = np.array(hist.get("pos_y", []))

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Trajectory line with colour gradient (time progression).
        if len(xs) > 1:
            t_norm = np.linspace(0, 1, len(xs))
            points = np.array([xs, ys]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            from matplotlib.collections import LineCollection
            lc = LineCollection(segments, cmap="viridis", array=t_norm, linewidth=1.2, alpha=0.85)
            ax.add_collection(lc)
            fig.colorbar(lc, ax=ax, label="Time progression", shrink=0.85)
            # Autoscale after adding collection.
            ax.autoscale()
        elif len(xs) == 1:
            ax.scatter(xs, ys, c=self.COLORS["blue"], s=60, zorder=5, label="Position")

        # Spawn origin marker.
        ax.scatter(0, 0, c=self.COLORS["red"], marker="x", s=120, linewidths=2,
                   zorder=6, label="Spawn Origin")

        # Start marker.
        if len(xs) > 0:
            ax.scatter(xs[0], ys[0], c=self.COLORS["green"], marker="o", s=60,
                       zorder=6, label="Start")
        # End marker.
        if len(xs) > 0:
            ax.scatter(xs[-1], ys[-1], c=self.COLORS["orange"], marker="s", s=60,
                       zorder=6, label="End")

        # Wander-radius circle.
        max_drift = float(np.max(np.sqrt(xs**2 + ys**2))) if len(xs) > 0 else 0.0
        if max_drift > 1e-6:
            circle = plt.Circle((0, 0), max_drift, fill=False, linestyle="--",
                                color=self.COLORS["grey"], linewidth=1.2,
                                label=f"Max drift = {max_drift:.3f} m")
            ax.add_patch(circle)

        ax.set_xlabel("X position (m)")
        ax.set_ylabel("Y position (m)")
        ax.set_title("2D Trajectory — Top-Down Drift Map")
        ax.legend(loc="best", framealpha=0.9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        path = os.path.join(self.save_dir, filename)
        fig.savefig(f"{path}.png", dpi=self.dpi, bbox_inches="tight")
        fig.savefig(f"{path}.pdf", bbox_inches="tight")
        plt.close(fig)
        return f"{path}.png"

    # ------------------------------------------------------------------
    # 5b. Velocity Tracking Profile
    # ------------------------------------------------------------------

    def plot_velocity_profile(self, filename: str = "velocity_profile") -> str:
        """Time-series overlay of target velocity and actual velocity.

        Shows how well the policy tracks the commanded velocity over the
        course of an episode, including any overshoot.

        Args:
            filename: Base filename (without extension).

        Returns:
            Absolute path to the saved PNG file.
        """
        hist = self.logger.history
        times = np.array(hist.get("time_s", []))
        vel = np.array(hist.get("velocity_ms", []))
        target = np.array(hist.get("target_velocity_ms", []))

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        ax.plot(times, vel, color=self.COLORS["blue"], linewidth=1.5, label="Actual $v$")
        ax.plot(times, target, color=self.COLORS["orange"], linewidth=1.5,
                linestyle="--", label="Target $v_{\\rm target}$")

        # Shade the error region.
        if len(vel) > 0 and len(target) > 0:
            ax.fill_between(times, vel, target, alpha=0.12, color=self.COLORS["red"],
                            label="Tracking error")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Forward velocity $v_x$ (m/s)")
        ax.set_title("Velocity Tracking Profile (body-frame $v_x$ vs command)")
        ax.legend(loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

        path = os.path.join(self.save_dir, filename)
        fig.savefig(f"{path}.png", dpi=self.dpi, bbox_inches="tight")
        fig.savefig(f"{path}.pdf", bbox_inches="tight")
        plt.close(fig)
        return f"{path}.png"

    # ------------------------------------------------------------------
    # 5c. Action Jitter & Torque Plot
    # ------------------------------------------------------------------

    def plot_action_torque(self, filename: str = "action_torque") -> str:
        """Multi-panel time-series of motor torques, actions, and action jitter.

        Visually exposes high-frequency twitching, torque saturation clipping,
        and the smoothness of the learned control policy.

        Args:
            filename: Base filename (without extension).

        Returns:
            Absolute path to the saved PNG file.
        """
        hist = self.logger.history
        times = np.array(hist.get("time_s", []))

        steer_torque = np.array(hist.get("steer_torque", []))
        drive_torque = np.array(hist.get("drive_torque", []))
        steer_action = np.array(hist.get("steer_action", []))
        drive_action = np.array(hist.get("drive_action", []))
        jitter = np.array(hist.get("action_jitter", []))

        fig, axes = plt.subplots(3, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5),
                                 dpi=self.dpi, sharex=True)

        # Panel 1: Torques.
        ax = axes[0]
        ax.plot(times, steer_torque, color=self.COLORS["blue"], linewidth=1.0,
                label=f"Steering torque ({self.logger.steer_joint_name})")
        ax.plot(times, drive_torque, color=self.COLORS["orange"], linewidth=1.0,
                label=f"Drive torque ({self.logger.drive_joint_name})")
        # Saturation thresholds (hard-coded from logger config).
        steer_limit = self.logger.effort_limit_steering
        drive_limit = self.logger.effort_limit_drive
        ax.axhline(y=steer_limit, color=self.COLORS["blue"], linestyle=":", alpha=0.5,
                   linewidth=0.8)
        ax.axhline(y=-steer_limit, color=self.COLORS["blue"], linestyle=":", alpha=0.5,
                   linewidth=0.8)
        ax.axhline(y=drive_limit, color=self.COLORS["orange"], linestyle=":", alpha=0.5,
                   linewidth=0.8)
        ax.axhline(y=-drive_limit, color=self.COLORS["orange"], linestyle=":", alpha=0.5,
                   linewidth=0.8)
        # 90 % saturation zone shading.
        ax.axhspan(0.9 * steer_limit, steer_limit, color=self.COLORS["red"], alpha=0.06)
        ax.axhspan(-steer_limit, -0.9 * steer_limit, color=self.COLORS["red"], alpha=0.06)
        ax.set_ylabel("Torque (N·m)")
        ax.set_title("Motor Torques & Saturation Limits")
        ax.legend(loc="best", framealpha=0.9, fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel 2: Actions.
        ax = axes[1]
        ax.plot(times, steer_action, color=self.COLORS["blue"], linewidth=1.0,
                label="Steering action")
        ax.plot(times, drive_action, color=self.COLORS["orange"], linewidth=1.0,
                label="Drive action")
        ax.set_ylabel("Action value")
        ax.set_title("Control Actions")
        ax.legend(loc="best", framealpha=0.9, fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel 3: Action jitter (L2-norm of action difference).
        ax = axes[2]
        ax.plot(times, jitter, color=self.COLORS["red"], linewidth=1.0, label="|a(t) - a(t-1)|")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Jitter (L2 norm)")
        ax.set_title("Action Rate Jitter — High-Frequency Twitching Indicator")
        ax.legend(loc="best", framealpha=0.9, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

        fig.tight_layout()
        path = os.path.join(self.save_dir, filename)
        fig.savefig(f"{path}.png", dpi=self.dpi, bbox_inches="tight")
        fig.savefig(f"{path}.pdf", bbox_inches="tight")
        plt.close(fig)
        return f"{path}.png"

    # ------------------------------------------------------------------
    # 5d. Posture Stability Map
    # ------------------------------------------------------------------

    def plot_posture_stability(self, filename: str = "posture_stability") -> str:
        """Time-series of Roll and Pitch angles showing balancing wobbles.

        Includes horizontal bands for the "stable window" (±3° by default)
        and the termination threshold (±45°).

        Args:
            filename: Base filename (without extension).

        Returns:
            Absolute path to the saved PNG file.
        """
        hist = self.logger.history
        times = np.array(hist.get("time_s", []))
        roll = np.array(hist.get("roll_deg", []))
        pitch = np.array(hist.get("pitch_deg", []))

        fig, axes = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2),
                                 dpi=self.dpi, sharex=True)

        stable_deg = self.logger.stable_roll_threshold_deg

        # Panel 1: Roll.
        ax = axes[0]
        ax.plot(times, roll, color=self.COLORS["blue"], linewidth=1.0, label="Roll φ")
        ax.axhline(y=0, color="black", linewidth=0.5)
        # Stable window.
        ax.axhspan(-stable_deg, stable_deg, color=self.COLORS["green"], alpha=0.08,
                   label=f"Stable window (±{stable_deg:.0f}°)")
        # Termination threshold.
        ax.axhline(y=45, color=self.COLORS["red"], linestyle="--", alpha=0.6,
                   linewidth=1.0, label="Termination limit (±45°)")
        ax.axhline(y=-45, color=self.COLORS["red"], linestyle="--", alpha=0.6, linewidth=1.0)
        ax.axhline(y=15, color=self.COLORS["orange"], linestyle=":", alpha=0.8,
                   linewidth=1.0, label="Recoverable envelope (±15°)")
        ax.axhline(y=-15, color=self.COLORS["orange"], linestyle=":", alpha=0.8, linewidth=1.0)
        ax.set_ylabel("Roll angle (degrees)")
        ax.set_title("Posture Stability — Roll Angle Over Time")
        ax.legend(loc="best", framealpha=0.9, fontsize=8)
        ax.grid(True, alpha=0.3)

        # Panel 2: Pitch.
        ax = axes[1]
        ax.plot(times, pitch, color=self.COLORS["orange"], linewidth=1.0, label="Pitch θ")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pitch angle (degrees)")
        ax.set_title("Posture Stability — Pitch Angle Over Time")
        ax.legend(loc="best", framealpha=0.9, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

        fig.tight_layout()
        path = os.path.join(self.save_dir, filename)
        fig.savefig(f"{path}.png", dpi=self.dpi, bbox_inches="tight")
        fig.savefig(f"{path}.pdf", bbox_inches="tight")
        plt.close(fig)
        return f"{path}.png"

    def plot_command_tracking(self, filename: str = "command_tracking") -> str:
        """vx, yaw-rate, and heading-error time series for locomotion eval."""
        hist = self.logger.history
        times = np.array(hist.get("time_s", []))
        vx = np.array(hist.get("velocity_ms", []))
        vx_cmd = np.array(hist.get("target_velocity_ms", []))
        wz = np.array(hist.get("wz", []))
        wz_cmd = np.array(hist.get("cmd_wz", []))
        heading_err = np.array(hist.get("heading_err_deg", []))

        fig, axes = plt.subplots(3, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5),
                                 dpi=self.dpi, sharex=True)
        ax = axes[0]
        ax.plot(times, vx, color=self.COLORS["blue"], linewidth=1.2, label="actual $v_x$")
        ax.plot(times, vx_cmd, color=self.COLORS["orange"], linewidth=1.2, linestyle="--",
                label="command $v_x^*$")
        if len(vx_cmd):
            mu = float(np.nanmean(vx_cmd))
            ax.axhspan(mu - 0.1, mu + 0.1, color=self.COLORS["green"], alpha=0.08,
                       label="±0.1 m/s gate")
        ax.set_ylabel("$v_x$ (m/s)")
        ax.set_title("Command Tracking")
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(times, wz, color=self.COLORS["blue"], linewidth=1.2, label=r"actual $\omega_z$")
        ax.plot(times, wz_cmd, color=self.COLORS["orange"], linewidth=1.2, linestyle="--",
                label=r"command $\omega_z^*$")
        ax.set_ylabel(r"$\omega_z$ (rad/s)")
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.plot(times, heading_err, color=self.COLORS["purple"], linewidth=1.2, label="heading error")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Heading error (deg)")
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

        fig.tight_layout()
        path = os.path.join(self.save_dir, filename)
        fig.savefig(f"{path}.png", dpi=self.dpi, bbox_inches="tight")
        fig.savefig(f"{path}.pdf", bbox_inches="tight")
        plt.close(fig)
        return f"{path}.png"

    def plot_survival_thresholds(self, filename: str = "survival_thresholds") -> str:
        """Binned survival vs spawn/command covariates, plus time-to-crash."""
        from .eval_failure import COVARIATES, analyze_survival_thresholds

        analysis = analyze_survival_thresholds(self.logger.episodes)
        keys = (
            "spawn_vx_ms",
            "spawn_yaw_abs_deg",
            "spawn_cmd_vx_ms",
            "spawn_cmd_heading_abs_deg",
            "spawn_roll_deg",
            "spawn_terrain_level",
        )
        fig, axes = plt.subplots(3, 2, figsize=(self.figsize[0] * 1.15, self.figsize[1] * 1.85), dpi=self.dpi)
        axes = axes.ravel()
        for ax, key in zip(axes, keys):
            label, unit = next(((lab, u) for k, lab, u in COVARIATES if k == key), (key, ""))
            cov = (analysis.get("covariates") or {}).get(key) or {}
            bins = cov.get("bins") or []
            if not bins:
                ax.set_title(label)
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                continue
            centers = [0.5 * (b["lo"] + b["hi"]) for b in bins]
            surv = [b["survival"] for b in bins]
            ns = [b["n"] for b in bins]
            ax.bar(range(len(bins)), surv, color=self.COLORS["blue"], alpha=0.85)
            ax.set_ylim(0.0, 1.05)
            ax.set_xticks(range(len(bins)))
            tick_unit = f" {unit}" if unit else ""
            ax.set_xticklabels(
                [f"{c:.2g}{tick_unit}\nn={n}" for c, n in zip(centers, ns)],
                fontsize=7,
            )
            ax.axhline(0.5, color=self.COLORS["red"], linestyle=":", linewidth=0.9)
            thr = cov.get("threshold_50")
            title = label
            if isinstance(thr, (int, float)) and math.isfinite(thr):
                title = f"{label}  —  50% @ {thr:.2g}{tick_unit}"
            ax.set_title(title, fontsize=10)
            ax.set_ylabel("Survival")
            ax.grid(True, axis="y", alpha=0.3)
        fig.suptitle("Where the bike fails — survival vs spawn / command", fontsize=12)
        fig.tight_layout()
        path = os.path.join(self.save_dir, filename)
        fig.savefig(f"{path}.png", dpi=self.dpi, bbox_inches="tight")
        fig.savefig(f"{path}.pdf", bbox_inches="tight")
        plt.close(fig)

        crashed = [e for e in self.logger.episodes if e.complete and not e.survived]
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        times = [e.episode_time_s for e in crashed]
        if times:
            ax.hist(times, bins=min(20, max(5, len(times) // 3)), color=self.COLORS["orange"], edgecolor="white")
        ft = analysis.get("fail_time_s") or {}
        if math.isfinite(ft.get("median", float("nan"))):
            ax.axvline(ft["median"], color=self.COLORS["red"], linestyle="--", label=f"median {ft['median']:.2f}s")
            ax.legend(framealpha=0.9)
        ax.set_xlabel("Time to crash (s)")
        ax.set_ylabel("Episodes")
        ax.set_title("When the bike fails")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        tpath = os.path.join(self.save_dir, "fail_times")
        fig.savefig(f"{tpath}.png", dpi=self.dpi, bbox_inches="tight")
        fig.savefig(f"{tpath}.pdf", bbox_inches="tight")
        plt.close(fig)
        self._fail_times_path = f"{tpath}.png"
        return f"{path}.png"

    def generate_all_plots(self) -> dict[str, str]:
        """Generate evaluation plots and return a mapping of plot name → file path."""
        plots: dict[str, str] = {}
        hist = self.logger.history

        if self.logger.episodes:
            print("[EvalLogger] Generating survival-threshold plots ...")
            plots["survival_thresholds"] = self.plot_survival_thresholds()
            fail_times = os.path.join(self.save_dir, "fail_times.png")
            if os.path.isfile(fail_times):
                plots["fail_times"] = fail_times

        if len(hist.get("time_s", [])) == 0:
            if not plots:
                print("[EvalLogger] WARNING: No history data recorded — skipping plots.")
            else:
                print("[EvalLogger] No env_0 history — skipped trajectory/action plots.")
            if plots:
                print(f"[EvalLogger] Plots saved to: {self.save_dir}/")
            return plots

        print("[EvalLogger] Generating 2D trajectory plot ...")
        plots["trajectory"] = self.plot_2d_trajectory()

        print("[EvalLogger] Generating velocity profile plot ...")
        plots["velocity"] = self.plot_velocity_profile()

        print("[EvalLogger] Generating action / torque plot ...")
        plots["action_torque"] = self.plot_action_torque()

        print("[EvalLogger] Generating posture stability plot ...")
        plots["posture"] = self.plot_posture_stability()

        print("[EvalLogger] Generating command tracking plot ...")
        plots["command_tracking"] = self.plot_command_tracking()

        # Interactive Plotly HTML (zoom / pan / rescale) — always attempted.
        if self.interactive:
            try:
                from .eval_plotly import generate_plotly_plots

                plots.update(generate_plotly_plots(self.logger, self.save_dir))
            except Exception as exc:  # pragma: no cover
                print(f"[EvalLogger] WARN: interactive Plotly plots skipped: {exc}")

        print(f"[EvalLogger] All plots saved to: {self.save_dir}/")
        return plots


# ===========================================================================
# 6. Environment config capture — save the exact config that was evaluated
# ===========================================================================


def capture_env_config(env_cfg: Any) -> dict[str, Any]:
    """Extract the relevant evaluation-time environment configuration into a
    plain, JSON-serialisable dictionary.

    Handles Isaac Lab ``configclass`` (frozen dataclass-like) objects by
    recursively walking their ``__dict__``, falling back to ``vars()``, and
    ultimately ``repr()`` for opaque objects.

    Args:
        env_cfg: An Isaac Lab ``ManagerBasedRLEnvCfg`` instance (or similar).

    Returns:
        Nested dict with sections: ``rewards``, ``terminations``,
        ``actions``, ``observations``, ``events``, ``scene``, ``sim``,
        ``episode``.
    """
    def _safe_repr(obj: Any) -> str:
        """Best-effort string representation."""
        try:
            return str(obj)
        except Exception:
            return "<unrepresentable>"

    def _to_dict(obj: Any, depth: int = 0) -> Any:
        """Recursively convert a config object to a plain dict / list / scalar."""
        if depth > 6:  # safety limit
            return _safe_repr(obj)
        if obj is None:
            return None
        if isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, (tuple, list)):
            return [_to_dict(v, depth + 1) for v in obj]
        if isinstance(obj, dict):
            return {str(k): _to_dict(v, depth + 1) for k, v in obj.items()}
        # Dataclass / configclass / arbitrary object
        if hasattr(obj, "__dataclass_fields__") or hasattr(obj, "__dict__"):
            result: dict[str, Any] = {}
            for key in sorted(dir(obj)):
                if key.startswith("_"):
                    continue
                try:
                    val = getattr(obj, key)
                except Exception:
                    continue
                # Skip methods and modules
                if callable(val) or isinstance(val, type):
                    continue
                result[key] = _to_dict(val, depth + 1)
            return result
        return _safe_repr(obj)

    raw = _to_dict(env_cfg)

    # ---- Extract the most relevant sections into a flat, readable layout ---
    rewards_raw = raw.get("rewards", {})
    terminations_raw = raw.get("terminations", {})
    actions_raw = raw.get("actions", {})
    observations_raw = raw.get("observations", {})
    events_raw = raw.get("events", {})
    scene_raw = raw.get("scene", {})
    sim_raw = raw.get("sim", {})

    # Rewards — extract weight and function name for each term
    rewards: dict[str, dict[str, Any]] = {}
    for name, term in rewards_raw.items():
        if isinstance(term, dict):
            rewards[name] = {
                "weight": term.get("weight", "?"),
                "func": term.get("func", "?"),
            }
            # Pull out scalar params (skip asset_cfg noise)
            for pk, pv in term.items():
                if pk in ("weight", "func"):
                    continue
                if isinstance(pv, (bool, int, float, str)):
                    rewards[name][pk] = pv

    # Terminations
    terminations: dict[str, dict[str, Any]] = {}
    for name, term in terminations_raw.items():
        if isinstance(term, dict):
            terminations[name] = {
                "time_out": term.get("time_out", "?"),
                "func": term.get("func", "?"),
            }
            for pk, pv in term.items():
                if pk in ("time_out", "func"):
                    continue
                if isinstance(pv, (bool, int, float, str)):
                    terminations[name][pk] = pv

    # Actions
    actions: dict[str, dict[str, Any]] = {}
    for name, term in actions_raw.items():
        if isinstance(term, dict):
            actions[name] = {}
            for pk, pv in term.items():
                if isinstance(pv, (bool, int, float, str, list, tuple, type(None))):
                    actions[name][pk] = pv
                elif isinstance(pv, dict):
                    # e.g. joint_names
                    actions[name][pk] = {str(k): _safe_repr(v) for k, v in pv.items()}
                else:
                    actions[name][pk] = _safe_repr(pv)

    # Observations — noise config
    observations: dict[str, Any] = {}
    for group_name, group in observations_raw.items():
        if isinstance(group, dict):
            observations[group_name] = {}
            for term_name, term in group.items():
                if isinstance(term, dict):
                    observations[group_name][term_name] = {}
                    for pk, pv in term.items():
                        if isinstance(pv, (bool, int, float, str, type(None))):
                            observations[group_name][term_name][pk] = pv
                        elif isinstance(pv, dict):
                            observations[group_name][term_name][pk] = pv
                        else:
                            observations[group_name][term_name][pk] = _safe_repr(pv)

    # Events — mode and key params
    events: dict[str, dict[str, Any]] = {}
    for name, term in events_raw.items():
        if isinstance(term, dict):
            events[name] = {"mode": term.get("mode", "?")}
            for pk, pv in term.items():
                if pk == "mode":
                    continue
                if isinstance(pv, (bool, int, float, str)):
                    events[name][pk] = pv
                elif isinstance(pv, dict):
                    # e.g. mass_distribution_params, pose_range, …
                    events[name][pk] = pv
                else:
                    events[name][pk] = _safe_repr(pv)

    # Scene
    scene: dict[str, Any] = {
        "num_envs": scene_raw.get("num_envs", "?"),
        "env_spacing": scene_raw.get("env_spacing", "?"),
    }

    # Simulation
    sim: dict[str, Any] = {
        "device": sim_raw.get("device", "?"),
        "dt": sim_raw.get("dt", "?"),
        "physx": {
            k: v for k, v in sim_raw.items()
            if isinstance(v, (bool, int, float, str))
        } if isinstance(sim_raw, dict) else {},
    }

    # Episode / control
    episode: dict[str, Any] = {
        "decimation": raw.get("decimation", "?"),
        "step_dt": raw.get("step_dt", "?"),
        "episode_length_s": raw.get("episode_length_s", "?"),
    }

    return {
        "episode": episode,
        "scene": scene,
        "sim": sim,
        "rewards": rewards,
        "terminations": terminations,
        "actions": actions,
        "observations": observations,
        "events": events,
    }


# ===========================================================================
# 7. Data Export — JSON, CSV, WandB, TensorBoard
# ===========================================================================


def export_eval_report(
    summary: EvalSummary,
    logger: BicycleMetricsLogger,
    save_dir: str = "./eval_results",
    timestamp: Optional[str] = None,
) -> dict[str, str]:
    """Export the evaluation summary as JSON, CSV, and graphs for that CSV.

    Args:
        summary: Aggregated ``EvalSummary`` from ``logger.finalize()``.
        logger: The ``BicycleMetricsLogger`` instance (for per-episode data).
        save_dir: Directory to write report files.
        timestamp: Optional ISO-8601 timestamp string.  Auto-generated if None.

    Returns:
        Dict mapping file type → file path, e.g.
        ``{"json": ".../eval_report.json", "csv": ".../eval_report.csv",
        "plot_survival_thresholds": "...png", ...}``.
    """
    os.makedirs(save_dir, exist_ok=True)
    if timestamp is None:
        timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")

    exported: dict[str, str] = {}

    # --- JSON report (compact aggregate + per-episode detail) ----------------
    json_path = os.path.join(save_dir, f"eval_report_{timestamp}.json")

    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    report = {
        "timestamp": timestamp,
        "contract": _sanitize(logger.contract.to_dict()) if getattr(logger, "contract", None) is not None else {
            "steer_joint_name": logger.steer_joint_name,
            "drive_joint_name": logger.drive_joint_name,
            "robot_name": getattr(logger, "robot_name", "robot"),
        },
        "summary": _sanitize(summary.to_dict()),
        "failure_analysis": _sanitize(getattr(summary, "failure_analysis", {})),
        "episodes": [_sanitize(dataclasses.asdict(e)) for e in logger.episodes],
    }
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    exported["json"] = json_path
    print(f"[EvalLogger] JSON report saved → {json_path}")

    # --- CSV report (flat rows, one per episode) -----------------------------
    csv_path = os.path.join(save_dir, f"eval_report_{timestamp}.csv")
    fieldnames = [
        "env_id", "survived", "episode_length_s",
        "min_drift_m", "max_drift_m", "mean_drift_m", "final_drift_m", "lateral_deviation_rms_m",
        "min_roll_deg", "max_roll_deg", "mean_roll_deg", "std_roll_deg", "max_lean_deg",
        "min_pitch_deg", "max_pitch_deg", "mean_pitch_deg", "std_pitch_deg", "yaw_drift_rate_deg_per_s",
        "min_velocity_ms", "max_velocity_ms", "mean_velocity_ms", "velocity_rmse_ms", "velocity_overshoot_pct",
        "mean_action_jitter", "max_action_jitter", "mean_joint_acc_mag", "max_joint_acc_mag",
        "total_power_j", "steering_usage_ratio", "drive_usage_ratio",
        "torque_saturation_ratio", "stable_balance_ratio",
        "recovery_count", "cumulative_roll_excursion_deg_s",
        "mean_drive_speed_rad_s", "max_drive_speed_rad_s",
        "complete", "roll_rmse_deg", "roll_rmse_straight_deg",
        "vx_rmse_ms", "vx_mae_ms", "vx_ss_mae_ms", "wz_rmse_rad_s", "heading_rmse_deg",
        "mean_cmd_vx_ms", "steer_action_jitter", "drive_action_jitter",
        "action_clip_sat_ratio", "steer_clip_sat_ratio", "drive_clip_sat_ratio",
        "control_effort", "mean_tau_sq",
        "kick_seen", "kick_settled", "kick_settle_time_s", "kick_peak_roll_deg", "kick_oscillations",
        "bsr_balanced", "brt_s", "brt_events", "max_balance_duration_s",
        "srl_s", "srl_events", "noise_rho",
        "term_reason",
        "spawn_vx_ms", "spawn_yaw_deg", "spawn_roll_deg",
        "spawn_cmd_vx_ms", "spawn_cmd_wz_rad_s", "spawn_cmd_heading_deg",
        "drive_accel_ms2", "yaw_bias_deg",
        "mass_scale", "com_offset_m", "friction_mu", "kick_force_n",
        "spawn_terrain_level", "spawn_terrain_type",
        "end_vx_ms", "end_yaw_deg", "end_roll_deg",
        "end_cmd_vx_ms", "end_cmd_wz_rad_s", "end_cmd_heading_deg",
        "end_pos_x_m", "end_pos_y_m", "end_terrain_level", "end_terrain_type",
    ]
    with open(csv_path, "w") as f:
        f.write(",".join(fieldnames) + "\n")
        for e in logger.episodes:
            row = [
                str(e.env_id), str(int(e.survived)), f"{e.episode_time_s:.4f}",
                f"{e.min_drift_m:.6f}", f"{e.max_drift_m:.6f}", f"{e.mean_drift_m:.6f}",
                f"{e.final_drift_m:.6f}", f"{e.lateral_deviation_rms_m:.6f}",
                f"{e.min_roll_deg:.6f}", f"{e.max_roll_deg:.6f}", f"{e.mean_roll_deg:.6f}",
                f"{e.std_roll_deg:.6f}", f"{e.max_lean_deg:.6f}",
                f"{e.min_pitch_deg:.6f}", f"{e.max_pitch_deg:.6f}", f"{e.mean_pitch_deg:.6f}",
                f"{e.std_pitch_deg:.6f}", f"{e.yaw_drift_rate_deg_per_s:.6f}",
                f"{e.min_velocity_ms:.6f}", f"{e.max_velocity_ms:.6f}", f"{e.mean_velocity_ms:.6f}",
                f"{e.velocity_rmse_ms:.6f}", f"{e.velocity_overshoot_pct:.6f}",
                f"{e.mean_action_jitter:.6f}", f"{e.max_action_jitter:.6f}",
                f"{e.mean_joint_acc_mag:.6f}", f"{e.max_joint_acc_mag:.6f}",
                f"{e.total_power_j:.6f}", f"{e.steering_usage_ratio:.6f}", f"{e.drive_usage_ratio:.6f}",
                f"{e.torque_saturation_ratio:.6f}", f"{e.stable_balance_ratio:.6f}",
                str(e.recovery_count), f"{e.cumulative_roll_excursion_deg_s:.6f}",
                f"{e.mean_drive_speed_rad_s:.6f}", f"{e.max_drive_speed_rad_s:.6f}",
                str(int(e.complete)), f"{e.roll_rmse_deg:.6f}", f"{e.roll_rmse_straight_deg:.6f}",
                f"{e.vx_rmse_ms:.6f}", f"{e.vx_mae_ms:.6f}", f"{e.vx_ss_mae_ms:.6f}",
                f"{e.wz_rmse_rad_s:.6f}", f"{e.heading_rmse_deg:.6f}",
                f"{e.mean_cmd_vx_ms:.6f}", f"{e.steer_action_jitter:.6f}", f"{e.drive_action_jitter:.6f}",
                f"{e.action_clip_sat_ratio:.6f}", f"{e.steer_clip_sat_ratio:.6f}", f"{e.drive_clip_sat_ratio:.6f}",
                f"{e.control_effort:.6f}", f"{e.mean_tau_sq:.6f}",
                str(int(e.kick_seen)), str(int(e.kick_settled)),
                f"{e.kick_settle_time_s:.6f}" if math.isfinite(e.kick_settle_time_s) else "",
                f"{e.kick_peak_roll_deg:.6f}", str(e.kick_oscillations),
                str(int(e.bsr_balanced)),
                f"{e.brt_s:.6f}" if math.isfinite(e.brt_s) else "",
                str(e.brt_events), f"{e.max_balance_duration_s:.6f}",
                f"{e.srl_s:.6f}" if math.isfinite(e.srl_s) else "",
                str(e.srl_events),
                f"{e.noise_rho:.6f}" if math.isfinite(e.noise_rho) else "",
                e.term_reason,
                f"{e.spawn_vx_ms:.6f}" if math.isfinite(e.spawn_vx_ms) else "",
                f"{e.spawn_yaw_deg:.6f}" if math.isfinite(e.spawn_yaw_deg) else "",
                f"{e.spawn_roll_deg:.6f}" if math.isfinite(e.spawn_roll_deg) else "",
                f"{e.spawn_cmd_vx_ms:.6f}" if math.isfinite(e.spawn_cmd_vx_ms) else "",
                f"{e.spawn_cmd_wz_rad_s:.6f}" if math.isfinite(e.spawn_cmd_wz_rad_s) else "",
                f"{e.spawn_cmd_heading_deg:.6f}" if math.isfinite(e.spawn_cmd_heading_deg) else "",
                f"{e.drive_accel_ms2:.6f}" if math.isfinite(e.drive_accel_ms2) else "",
                f"{e.yaw_bias_deg:.6f}" if math.isfinite(e.yaw_bias_deg) else "",
                f"{e.mass_scale:.6f}" if math.isfinite(e.mass_scale) else "",
                f"{e.com_offset_m:.6f}" if math.isfinite(e.com_offset_m) else "",
                f"{e.friction_mu:.6f}" if math.isfinite(e.friction_mu) else "",
                f"{e.kick_force_n:.6f}" if math.isfinite(e.kick_force_n) else "",
                str(e.spawn_terrain_level), str(e.spawn_terrain_type),
                f"{e.end_vx_ms:.6f}" if math.isfinite(e.end_vx_ms) else "",
                f"{e.end_yaw_deg:.6f}" if math.isfinite(e.end_yaw_deg) else "",
                f"{e.end_roll_deg:.6f}" if math.isfinite(e.end_roll_deg) else "",
                f"{e.end_cmd_vx_ms:.6f}" if math.isfinite(e.end_cmd_vx_ms) else "",
                f"{e.end_cmd_wz_rad_s:.6f}" if math.isfinite(e.end_cmd_wz_rad_s) else "",
                f"{e.end_cmd_heading_deg:.6f}" if math.isfinite(e.end_cmd_heading_deg) else "",
                f"{e.end_pos_x_m:.6f}" if math.isfinite(e.end_pos_x_m) else "",
                f"{e.end_pos_y_m:.6f}" if math.isfinite(e.end_pos_y_m) else "",
                str(e.end_terrain_level), str(e.end_terrain_type),
            ]
            f.write(",".join(row) + "\n")
    exported["csv"] = csv_path
    print(f"[EvalLogger] CSV report saved → {csv_path}")

    try:
        from .eval_csv_plots import plot_eval_csv

        for name, path in plot_eval_csv(csv_path).items():
            exported[f"plot_{name}" if not name.startswith("plot_") else name] = path
    except Exception as exc:
        print(f"[EvalLogger] CSV graphs skipped: {exc}")

    # Interactive Plotly HTML from the same CSV (zoom / pan / rescale).
    try:
        from .eval_csv_plotly import plot_eval_csv_plotly

        plotly_dir = os.path.join(os.path.dirname(save_dir), "plots", "plotly")
        for name, path in plot_eval_csv_plotly(csv_path, out_dir=plotly_dir).items():
            exported[f"plotly_{name}"] = path
    except Exception as exc:
        print(f"[EvalLogger] Plotly CSV graphs skipped: {exc}")

    # Persist env_0 time-series so Plotly trajectory/velocity plots can be
    # regenerated offline after the run.
    hist = getattr(logger, "history", None)
    if isinstance(hist, dict) and hist.get("time_s"):
        hist_path = os.path.join(save_dir, f"eval_history_{timestamp}.json")
        try:
            with open(hist_path, "w") as f:
                json.dump(_sanitize(hist), f)
            exported["history"] = hist_path
            print(f"[EvalLogger] env_0 history saved → {hist_path}")
        except Exception as exc:
            print(f"[EvalLogger] history save skipped: {exc}")

    return exported


# ===========================================================================
# 8. WandB & TensorBoard image logging helpers
# ===========================================================================


def log_plots_to_wandb(
    plot_paths: dict[str, str],
    step: int = 0,
    prefix: str = "eval/plots",
) -> None:
    """Log generated plot images to Weights & Biases.

    Args:
        plot_paths: Dict of plot_name → file_path (as returned by
            ``BicycleEvalPlotter.generate_all_plots()``).
        step: Global step for WandB logging.
        prefix: WandB key prefix for grouping.

    Requires ``wandb`` to be installed and ``wandb.init()`` to have been called.
    """
    try:
        import wandb
    except ImportError:
        print("[EvalLogger] wandb not installed — skipping WandB image logging.")
        return

    for name, path in plot_paths.items():
        if os.path.isfile(path):
            wandb.log(
                {f"{prefix}/{name}": wandb.Image(path, caption=name)},
                step=step,
            )
    print(f"[EvalLogger] Logged {len(plot_paths)} plots to WandB at step {step}.")


def log_plots_to_tensorboard(
    plot_paths: dict[str, str],
    writer: Any,
    step: int = 0,
    prefix: str = "eval/plots",
) -> None:
    """Log generated plot images to TensorBoard via a ``SummaryWriter``.

    Args:
        plot_paths: Dict of plot_name → file_path.
        writer: A ``torch.utils.tensorboard.SummaryWriter`` instance.
        step: Global step for TensorBoard logging.
        prefix: Tag prefix for grouping.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("[EvalLogger] TensorBoard not installed — skipping TB image logging.")
        return

    for name, path in plot_paths.items():
        if not os.path.isfile(path):
            continue
        # Read image as numpy array for add_image.
        img = plt.imread(path)  # (H, W, C) or (H, W, 4)
        # TensorBoard expects (C, H, W).
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            img = np.transpose(img, (2, 0, 1))
        writer.add_image(f"{prefix}/{name}", img, step)
    print(f"[EvalLogger] Logged {len(plot_paths)} plots to TensorBoard at step {step}.")


# ===========================================================================
# 9. Integration helper — full evaluation loop with logging
# ===========================================================================


def run_evaluation_with_logging(
    env: ManagerBasedRLEnv,
    agent: Any,
    num_steps: int,
    device: str = "cuda:0",
    step_dt: float = 0.02,
    save_dir: str = "./eval_results",
    log_to_wandb: bool = False,
    log_to_tensorboard: bool = False,
    tb_writer: Any = None,
    **logger_kwargs: Any,
) -> EvalSummary:
    """Run a full evaluation loop with integrated metrics logging.

    This is a drop-in replacement for a manual eval loop.  It:
      1. Creates a ``BicycleMetricsLogger``.
      2. Runs the agent for ``num_steps``, calling ``logger.step()``.
      3. Calls ``logger.finalize()`` to produce an ``EvalSummary``.
      4. Generates plots via ``BicycleEvalPlotter``.
      5. Exports JSON/CSV reports.
      6. Optionally logs plots to WandB / TensorBoard.

    Args:
        env: The IsaacLab ``ManagerBasedRLEnv``.
        agent: A loaded skrl agent or any object with an ``act(obs)`` method
            that returns ``(actions, ...)``.
        num_steps: Number of environment steps to run.
        device: Torch device string.
        step_dt: Simulation step duration (seconds).
        save_dir: Base directory for plots and reports.
        log_to_wandb: If True, log plot images to WandB.
        log_to_tensorboard: If True, log plot images to TensorBoard.
        tb_writer: Optional existing ``SummaryWriter`` instance.
        **logger_kwargs: Optional overrides for :meth:`BicycleMetricsLogger.from_env`
            (joint names are resolved from the env unless overridden).

    Returns:
        ``EvalSummary`` with aggregated metrics.
    """
    num_envs = env.num_envs
    logger = BicycleMetricsLogger.from_env(
        env, device=device, step_dt=step_dt, **logger_kwargs
    )

    obs, _ = env.reset()
    states = env.state() if hasattr(env, "state") else None

    print(f"[EvalLoop] Starting evaluation — {num_envs} envs × {num_steps} steps ...")
    t_start = time.time()

    for step_idx in range(num_steps):
        with torch.inference_mode():
            if hasattr(agent, "act"):
                outputs = agent.act(obs, states, timestep=0, timesteps=0)
                if isinstance(outputs, tuple):
                    actions = outputs[-1].get("mean_actions", outputs[0]) if isinstance(outputs[-1], dict) else outputs[-1]
                else:
                    actions = outputs
            else:
                # Generic callable: agent(obs) → actions.
                actions = agent(obs)

            obs, _, _, _, extras = env.step(actions)
            if hasattr(env, "state"):
                states = env.state()

            logger.step(env, actions, target_velocity=None)

        # Progress print.
        if (step_idx + 1) % max(1, num_steps // 10) == 0:
            elapsed = time.time() - t_start
            fps = (step_idx + 1) * num_envs / max(elapsed, 1e-6)
            print(f"  Step {step_idx + 1}/{num_steps}  |  {fps:.0f} env-steps/s  |  "
                  f"{len(logger.episodes)} episodes completed")

    elapsed_total = time.time() - t_start
    print(f"[EvalLoop] Finished in {elapsed_total:.1f} s  "
          f"({num_steps * num_envs / elapsed_total:.0f} env-steps/s)")

    # Finalize.
    summary = logger.finalize()

    # Pretty-print.
    print("\n" + summary.pretty_print())

    # Plots.
    plot_dir = os.path.join(save_dir, "plots")
    plotter = BicycleEvalPlotter(logger, save_dir=plot_dir, interactive=True)
    plot_paths = plotter.generate_all_plots()

    # Data export.
    report_dir = os.path.join(save_dir, "reports")
    export_eval_report(summary, logger, save_dir=report_dir)

    # Optional WandB / TensorBoard logging.
    if log_to_wandb and plot_paths:
        log_plots_to_wandb(plot_paths)
    if log_to_tensorboard and plot_paths and tb_writer is not None:
        log_plots_to_tensorboard(plot_paths, tb_writer)

    return summary
