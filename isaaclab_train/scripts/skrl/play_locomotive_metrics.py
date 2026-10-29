#!/usr/bin/env python3
# Ported from ~/bike_test/CycleRL/scripts/skrl/play_with_metrics.py (imports/paths rewritten for trackstand_rl locomotive).
"""Play a checkpoint with full evaluation metrics (BicycleMetricsLogger).

Locomotion-only gates (no trackstand). Example:

    cd ~/IsaacLab
    ./isaaclab.sh -p ~/bike_test/CycleRL/scripts/skrl/play_with_metrics.py \\
        --task CycleRL-NewBike-Paper-v0 \\
        --eval_suite nominal,kick,mass,friction,training \\
        --num_envs 64 --headless

Omit ``--checkpoint`` to load the newest ``.pt`` under
``logs/skrl/locomotive/<task>/``.

Track one env spawn in the viewport (omit ``--headless``; use ``--viz``)::

    ./isaaclab.sh -p ~/bike_test/CycleRL/scripts/skrl/play_with_metrics.py \\
        --task CycleRL-NewBike-Paper-v0 \\
        --checkpoint /path/to/best_agent.pt \\
        --num_envs 64 --viz

The **CycleRL Camera** dock panel lets you pick env index and follow mode at runtime.
CLI override: ``--camera_env 5``  Disable panel: ``--no_camera_ui``

Staged terrain extremity (pins every env to one difficulty row):

    --eval_suite end                  # paper Table VI + CAT/MNT/MSS ladders
    --eval_suite paper                # Table VI conditions only
    --eval_suite diagnostics          # legacy plane ladders (not the ranking suite)
    --eval_suite full                 # paper suite + legacy ladders
    --eval_suite envelope             # mass/com/friction/kick ladders only
    --eval_suite terrain              # complex_L0..L4 then stairs_L0..L2
    --eval_suite complex              # complex_L0..L4
    --eval_suite stairs               # stairs_L0..L2
    --eval_mode mass                  # optional fixed probe (not in end)
    --eval_mode speed_sweep           # min balance speed + smooth-drive threshold
    --eval_mode heading_sweep         # fixed vx*=1.8, heading* ladder (±30° default)
    --eval_mode yaw_bias_sweep        # fixed vx*=1.8, ωz* stick-deg ladder → model yaw trim
    --eval_mode drive_accel_sweep     # RC hub accel ladder; policy steers only
    --eval_mode complex_L2

After training, prefer ``train_end_metrics.py`` (or ``train.py --post_eval``).
"""

import argparse
import json
import os
import sys

# =========================================================================
# CRITICAL FIX: Force cache isaaclab and start AppLauncher BEFORE heavy
# ML imports (skrl, torch).  This prevents Kit's sys.path manipulation
# from breaking internal IsaacLab imports and the libusd_tf.so crash.
# =========================================================================
import isaaclab  # noqa: F401
from isaaclab.app import AppLauncher, add_launcher_args
from isaaclab_tasks.utils import setup_preset_cli

# -- argparse (ALL args must be defined before AppLauncher) ----------------
parser = argparse.ArgumentParser(description="Play a checkpoint with full evaluation metrics.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="Locomotive-Cylinder-Aligned-v0")
parser.add_argument("--agent", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default=None,
                    help=(
                        "Checkpoint .pt, run dir, or checkpoints/ folder. "
                        "If omitted, loads the newest .pt under "
                        "logs/skrl/locomotive/<task>/."
                    ))
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax"])
parser.add_argument("--algorithm", type=str, default="PPO")
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument(
    "--terrain",
    type=str,
    default=None,
    choices=["flat", "complex", "stairs"],
    help=(
        "Bicycle terrain: 'flat' is a regular plane (default when omitted); "
        "'complex' is 5×5 (flat/rough/slopes/blocks/waves); "
        "'stairs' is 8×3 planes, up/down pyramid stairs, and waves."
    ),
)
# --- eval metrics flags ---------------------------------------------------
parser.add_argument(
    "--eval_steps",
    type=int,
    default=3200,
    help="Env steps per scenario. 3200 = one 64 s episode at dt=0.02.",
)
parser.add_argument("--save_dir", type=str, default=None, help="Override output root. Default: <run>/eval/")
parser.add_argument(
    "--ablate_steer",
    type=str,
    default="policy",
    choices=["policy", "lock"],
    help=(
        "Plant-stability ablation. 'lock' replaces the policy's steer action with 0 "
        "(joint held at its default, straight ahead); the policy still drives the hub. "
        "A bike that still balances with the fork locked is passively stable, so its BSR "
        "is not evidence of steering control."
    ),
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    default=False,
    help=(
        "Replace <mode>/summary.json already in the save directory. "
        "Default: write this invocation to a dated sibling so the earlier suite stays."
    ),
)
parser.add_argument("--log_wandb", action="store_true", default=False)
parser.add_argument("--log_tensorboard", action="store_true", default=False)
parser.add_argument("--no_plots", action="store_true", default=False)
parser.add_argument(
    "--no_plotly",
    action="store_true",
    default=False,
    help="Skip interactive Plotly HTML (zoomable) plots; keep matplotlib PNGs.",
)
parser.add_argument("--history_downsample", type=int, default=1)
parser.add_argument(
    "--eval_mode",
    type=str,
    default="nominal",
    help=(
        "Single scenario: paper_* Table VI modes, paper_roll_sweep, paper_noise_sweep, "
        "nominal | training | kick | kick_sweep | mass | mass_sweep | "
        "com_sweep | mass_com_sweep | friction | friction_sweep | speed_sweep | heading_sweep | "
        "yaw_bias_sweep | drive_accel_sweep | complex_L0..L4 | stairs_L0..L2."
    ),
)
parser.add_argument(
    "--eval_suite",
    type=str,
    default=None,
    help=(
        "Comma-separated scenarios (overrides --eval_mode). Aliases: "
        "end (post-train suite), full (end+terrain), envelope (mass/friction/kick sweeps), "
        "terrain (all 8 staged rows), complex (L0-L4), stairs (L0-L2). "
        "Example: nominal,kick,terrain"
    ),
)
parser.add_argument(
    "--speed_sweep_min",
    type=float,
    default=0.3,
    help="Speed-sweep mode: lowest commanded vx* (m/s) on the per-env ladder.",
)
parser.add_argument(
    "--speed_sweep_max",
    type=float,
    default=2.5,
    help="Speed-sweep mode: highest commanded vx* (m/s) on the per-env ladder (default 2.5).",
)
parser.add_argument(
    "--heading_sweep_min",
    type=float,
    default=-30.0,
    help="Heading-sweep mode: lowest commanded heading* (deg) on the per-env ladder.",
)
parser.add_argument(
    "--heading_sweep_max",
    type=float,
    default=30.0,
    help="Heading-sweep mode: highest commanded heading* (deg) on the per-env ladder.",
)
parser.add_argument(
    "--yaw_bias_min",
    type=float,
    default=-5.0,
    help="Yaw-bias sweep: lowest commanded stick-equivalent bias (deg) on the per-env ladder.",
)
parser.add_argument(
    "--yaw_bias_max",
    type=float,
    default=5.0,
    help="Yaw-bias sweep: highest commanded stick-equivalent bias (deg) on the per-env ladder.",
)
parser.add_argument(
    "--heading_rmse_max",
    type=float,
    default=15.0,
    help="Heading-sweep: max mean heading RMSE (deg) to count as still tracking.",
)
parser.add_argument(
    "--speed_balance_survival",
    type=float,
    default=0.80,
    help="Speed-sweep: minimum per-bin survival to count as 'still balancing'.",
)
parser.add_argument(
    "--speed_drive_jitter_max",
    type=float,
    default=0.08,
    help="Speed-sweep: max mean |Δa_drive| for 'smooth' drive (no bang-bang).",
)
parser.add_argument(
    "--speed_drive_bang_max",
    type=float,
    default=0.05,
    help="Speed-sweep: max drive bang-bang step ratio for smooth drive.",
)
parser.add_argument(
    "--drive_accel_min",
    type=float,
    default=0.25,
    help="Drive-accel sweep: lowest RC hub accel (m/s²) on the per-env ladder.",
)
parser.add_argument(
    "--drive_accel_max",
    type=float,
    default=8.0,
    help="Drive-accel sweep: highest RC hub accel (m/s²) on the per-env ladder.",
)
parser.add_argument(
    "--drive_accel_spawn_vx",
    type=float,
    default=0.5,
    help="Drive-accel sweep: spawn / hub speed (m/s) before ramping to vx*=1.8.",
)
parser.add_argument(
    "--mass_scale_min",
    type=float,
    default=1.0,
    help="Mass sweep: lowest mass scale (× nominal) on the per-env ladder.",
)
parser.add_argument(
    "--mass_scale_max",
    type=float,
    default=2.5,
    help="Mass sweep: highest mass scale (× nominal) on the per-env ladder.",
)
parser.add_argument(
    "--com_offset_min",
    type=float,
    default=-0.02,
    help="CoM sweep: lowest lateral (y) CoM offset (m) on the per-env ladder.",
)
parser.add_argument(
    "--com_offset_max",
    type=float,
    default=0.02,
    help="CoM sweep: highest lateral (y) CoM offset (m) on the per-env ladder.",
)
parser.add_argument(
    "--mass_com_scale_min",
    type=float,
    default=1.0,
    help="Mass×CoM grid: lowest mass scale (× nominal).",
)
parser.add_argument(
    "--mass_com_scale_max",
    type=float,
    default=1.15,
    help="Mass×CoM grid: highest mass scale (× nominal).",
)
parser.add_argument(
    "--mass_com_offset_min",
    type=float,
    default=-0.02,
    help="Mass×CoM grid: lowest lateral CoM offset (m).",
)
parser.add_argument(
    "--mass_com_offset_max",
    type=float,
    default=0.02,
    help="Mass×CoM grid: highest lateral CoM offset (m).",
)
parser.add_argument(
    "--friction_mu_min",
    type=float,
    default=0.05,
    help="Friction sweep: lowest tire μ on the per-env ladder (plane also at this μ).",
)
parser.add_argument(
    "--friction_mu_max",
    type=float,
    default=1.0,
    help="Friction sweep: highest tire μ on the per-env ladder.",
)
parser.add_argument(
    "--kick_force_min",
    type=float,
    default=0.0,
    help="Kick sweep: lowest lateral force (N) on the per-env ladder.",
)
parser.add_argument(
    "--kick_force_max",
    type=float,
    default=40.0,
    help="Kick sweep: highest lateral force (N) on the per-env ladder.",
)
parser.add_argument(
    "--keep_vis",
    action="store_true",
    default=False,
    help=(
        "Keep command / terrain debug markers (velocity goal arrows, heading frames). "
        "By default eval scenarios disable them."
    ),
)
from play_camera import add_play_camera_args, apply_play_camera_cfg, attach_play_camera_ui

add_play_camera_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# -- Start Kit globally BEFORE heavy imports -------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =========================================================================
# NOW safe to import heavy ML libraries
# =========================================================================
import contextlib
import time

import gymnasium as gym
import torch
from packaging import version

import skrl

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_tasks.utils import resolve_task_config

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# ----- CycleRL imports ----------------------------------------------------
import source.trackstand_rl.trackstand_rl.tasks  # noqa: F401
from source.trackstand_rl.trackstand_rl.tasks.manager_based.locomotive.eval.eval_contract import apply_plant_cfg_from_checkpoint
from source.trackstand_rl.trackstand_rl.tasks.manager_based.locomotive.eval.eval_logger import (
    BicycleEvalPlotter,
    BicycleMetricsLogger,
    evaluate_gates,
    export_eval_report,
    format_gate_report,
    log_plots_to_tensorboard,
    log_plots_to_wandb,
)
from source.trackstand_rl.trackstand_rl.tasks.manager_based.locomotive.eval.eval_scenarios import (
    apply_drive_accel_sweep_plant,
    configure_eval_scenario,
    expand_eval_modes,
    is_terrain_stage,
    pin_eval_terrain_level,
)
from source.trackstand_rl.trackstand_rl.tasks.manager_based.locomotive.eval.eval_drive_accel import (
    analyze_drive_accel,
    export_drive_accel,
    format_drive_accel_report,
    plot_drive_accel,
)
from source.trackstand_rl.trackstand_rl.tasks.manager_based.locomotive.eval.eval_probe_limits import (
    analyze_com_limits,
    analyze_friction_limits,
    analyze_kick_limits,
    analyze_mass_com_limits,
    analyze_mass_limits,
    export_com_limits,
    export_friction_limits,
    export_kick_limits,
    export_mass_com_limits,
    export_mass_limits,
    format_com_limits_report,
    format_friction_limits_report,
    format_kick_limits_report,
    format_mass_com_limits_report,
    format_mass_limits_report,
    plot_com_limits,
    plot_friction_limits,
    plot_kick_limits,
    plot_mass_com_limits,
    plot_mass_limits,
)
from source.trackstand_rl.trackstand_rl.tasks.manager_based.locomotive.eval.eval_heading_limits import (
    analyze_heading_limits,
    export_heading_limits,
    format_heading_limit_report,
    plot_heading_limits,
)
from source.trackstand_rl.trackstand_rl.tasks.manager_based.locomotive.eval.eval_yaw_bias import (
    analyze_yaw_bias,
    export_yaw_bias,
    format_yaw_bias_report,
    plot_yaw_bias,
)
from source.trackstand_rl.trackstand_rl.tasks.manager_based.locomotive.eval.eval_speed_limits import (
    analyze_speed_limits,
    export_speed_limits,
    format_speed_limit_report,
    plot_speed_limits,
)

SKRL_VERSION = "2.1.0"
_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CYCLERL_LOGS = os.path.join(_PROJECT_DIR, "logs", "skrl", "locomotive")

if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(f"Unsupported skrl version: {skrl.__version__}. Install skrl>={SKRL_VERSION}")
    exit()

if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()

# -- optionally initialize WandB -------------------------------------------
if args_cli.log_wandb:
    try:
        import wandb
        wandb.init(project="cyclerl-eval", name=f"eval_{args_cli.task.split(':')[-1]}")
    except ImportError:
        print("[WARNING] wandb not installed — disabling WandB logging.")
        args_cli.log_wandb = False


def _parse_modes() -> list[str]:
    if args_cli.eval_suite:
        modes = [m.strip() for m in args_cli.eval_suite.split(",") if m.strip()]
    else:
        modes = [args_cli.eval_mode]
    return expand_eval_modes(modes)


def _log_root_for_task(task_name: str) -> str:
    """``logs/skrl/locomotive/<task>`` — same layout as ``train.py`` / ``play.py``."""
    return os.path.join(_CYCLERL_LOGS, task_name)


def _latest_checkpoint_for_task(task_name: str) -> str:
    """Return the newest ``.pt`` under ``logs/skrl/locomotive/<task-name>``."""
    log_root = _log_root_for_task(task_name)
    if not os.path.isdir(log_root):
        raise FileNotFoundError(f"No log directory for task {task_name!r}. Expected {log_root}")
    pts: list[str] = []
    for run in os.scandir(log_root):
        if not run.is_dir():
            continue
        ckpt_dir = os.path.join(run.path, "checkpoints")
        search_dir = ckpt_dir if os.path.isdir(ckpt_dir) else run.path
        for entry in os.scandir(search_dir):
            if entry.is_file() and entry.name.endswith(".pt"):
                pts.append(entry.path)
    if not pts:
        raise FileNotFoundError(f"No .pt checkpoints under {log_root}")
    pts.sort(key=os.path.getmtime)
    return pts[-1]


def _resolve_checkpoint(train_task_name: str) -> str:
    log_root_path = os.path.abspath(_log_root_for_task(train_task_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            raise RuntimeError("[INFO] No pre-trained checkpoint available for this task.")
        return resume_path
    if args_cli.checkpoint:
        path = os.path.abspath(args_cli.checkpoint)
        if os.path.isdir(path):
            ckpt_dir = os.path.join(path, "checkpoints") if os.path.isdir(os.path.join(path, "checkpoints")) else path
            best = os.path.join(ckpt_dir, "best_agent.pt")
            latest = os.path.join(ckpt_dir, "agent_300000.pt")
            if os.path.isfile(best):
                return best
            if os.path.isfile(latest):
                return latest
            pts = sorted(
                (os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pt")),
                key=os.path.getmtime,
            )
            if pts:
                return pts[-1]
            raise FileNotFoundError(f"No .pt checkpoint in {ckpt_dir}")
        return path
    return _latest_checkpoint_for_task(train_task_name)


def _write_suite_markdown(path: str, results: dict, resume_path: str) -> None:
    lines = [
        "# CycleRL locomotion evaluation",
        "",
        f"- Checkpoint: `{resume_path}`",
        "- Task class: dynamic balancing / velocity tracking (**no trackstand**)",
        "",
        "## Suite",
        "",
        "| Mode | Survival | Roll RMSE (straight) | vx SS MAE | Clip sat | Gates |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for mode, payload in results.items():
        s = payload["summary"]
        g = payload["gates"]
        sl = payload.get("speed_limits")
        da = payload.get("drive_accel")
        hl = payload.get("heading_limits")
        yb = payload.get("yaw_bias")
        ms = payload.get("mass_limits")
        cm = payload.get("com_limits")
        mc = payload.get("mass_com_limits")
        fr = payload.get("friction_limits")
        kf = payload.get("kick_limits")
        if mode == "speed_sweep" and sl is not None:
            mb = sl.get("min_balance_speed_ms")
            sd = sl.get("smooth_drive_speed_ms")
            mark = "—"
            gate_col = (
                f"min_bal={mb:.2f} m/s, smooth≥{sd:.2f} m/s"
                if mb is not None and sd is not None
                else f"min_bal={mb:.2f} m/s" if mb is not None
                else f"smooth≥{sd:.2f} m/s" if sd is not None
                else "see speed_limits.json"
            )
        elif mode == "heading_sweep" and hl is not None:
            mx = hl.get("max_balance_heading_abs_deg")
            mark = "—"
            gate_col = (
                f"max_bal_|θ*|={mx:.1f}°" if mx is not None else "see heading_limits.json"
            )
        elif mode == "yaw_bias_sweep" and yb is not None:
            trim = yb.get("recommended_yaw_trim_deg")
            mark = "—"
            gate_col = (
                f"yaw_trim={trim:+.2f}°" if trim is not None else "see yaw_bias.json"
            )
        elif mode == "drive_accel_sweep" and da is not None:
            mx = da.get("max_balance_accel_ms2")
            mark = "—"
            gate_col = (
                f"max_bal_accel={mx:.2f} m/s²" if mx is not None else "see drive_accel.json"
            )
        elif mode == "mass_sweep" and ms is not None:
            mx = ms.get("max_balance_mass_scale")
            mark = "—"
            gate_col = (
                f"max_bal_mass=×{mx:.3f}" if mx is not None else "see mass_limits.json"
            )
        elif mode == "com_sweep" and cm is not None:
            mx = cm.get("max_balance_com_abs_m")
            mark = "—"
            gate_col = (
                f"max_|CoM|={mx * 100:.2f} cm" if mx is not None else "see com_limits.json"
            )
        elif mode == "mass_com_sweep" and mc is not None:
            m_at = mc.get("max_balance_mass_at_com_2cm")
            c_at = mc.get("max_balance_com_abs_at_mass_1_10")
            mark = "—"
            parts = []
            if m_at is not None:
                parts.append(f"mass@1.5cm=×{m_at:.3f}")
            if c_at is not None:
                parts.append(f"|CoM|@1.10={c_at * 100:.2f}cm")
            gate_col = ", ".join(parts) if parts else "see mass_com_limits.json"
        elif mode == "friction_sweep" and fr is not None:
            mn = fr.get("min_friction_mu")
            mark = "—"
            gate_col = (
                f"min_bal_μ={mn:.3f}" if mn is not None else "see friction_limits.json"
            )
        elif mode == "kick_sweep" and kf is not None:
            mx = kf.get("max_kick_force_n")
            mark = "—"
            gate_col = (
                f"max_kick={mx:.1f} N" if mx is not None else "see kick_limits.json"
            )
        else:
            mark = "PASS" if g["all_passed"] else "FAIL"
            gate_col = f"{mark} {g['n_pass']}/{g['n_gates']}"
        lines.append(
            f"| {mode} | {s['survival_rate']:.1%} | {s['mean_roll_rmse_straight_deg']:.2f}° | "
            f"{s['mean_vx_ss_mae_ms']:.3f} m/s | {s['mean_action_clip_sat_ratio']:.1%} | "
            f"{gate_col} |"
        )
    lines += ["", "## Notes", ""]
    lines.append("- Nominal: flat plane, μ=1.0, no domain rand, vx*=1.8 m/s, heading*=0.")
    lines.append(
        "- Speed sweep: flat plane; each env holds a fixed vx* ladder [0.3, 2.5] m/s. "
        "Reports minimum balancing speed and smooth-drive (no bang-bang) threshold."
    )
    lines.append(
        "- Heading sweep: flat plane; vx*=1.8 m/s fixed; each env holds a fixed heading* "
        "ladder (default ±30°, **always includes θ*=0 straight**). Headline max |θ*| requires "
        "survival + stable balance **and** heading RMSE (default ≤15°)."
    )
    lines.append(
        "- Yaw bias sweep: flat plane; vx*=1.8 m/s fixed; each env holds a fixed teleop-style "
        "ωz* (stick-deg ladder, default ±5°). Reports recommended `--yaw_trim_deg` that zeros "
        "mean yaw drift."
    )
    lines.append(
        "- Drive accel sweep: RC owns hub speed (policy steers only). Each env ramps "
        "spawn vx → 1.8 m/s at a fixed accel (m/s²) on a linspace ladder."
    )
    lines.append("- Kick: 10 N lateral on `base_link` for 0.1 s at t=2 s.")
    lines.append(
        "- Kick sweep: same impulse timing; each env holds a fixed force (N) on a ladder "
        "(default 0–40 N). Headline = max balancing kick force."
    )
    lines.append("- Mass: mass ×1.10 and CoM ±2 cm.")
    lines.append(
        "- Mass sweep: CoM fixed; each env holds a fixed mass scale (default ×1.00–×2.50). "
        "Headline = max balancing mass scale."
    )
    lines.append(
        "- CoM sweep: mass nominal; each env holds a fixed lateral (y) CoM offset "
        "(default ±2 cm). Headline = max balancing |CoM|."
    )
    lines.append(
        "- Mass×CoM grid: each env holds a (mass scale, lateral CoM) cell "
        "(default mass ×1.00–×1.15 × CoM ±2 cm). Headlines at the ×1.10 / 0.5 cm probe."
    )
    lines.append("- Friction: ground and tires μ=0.4.")
    lines.append(
        "- Friction sweep: plane at μ_min; tire μ ladder (default 0.05–1.00). "
        "Headline = min balancing μ."
    )
    lines.append("- Training: 5×5 complex terrain with the nice2 command / rand snapshot.")
    lines.append(
        "- Terrain stages: `complex_L0`–`L4` and `stairs_L0`–`L2` pin every env to one "
        "difficulty row (columns still spread terrain types). Curriculum promotion is off."
    )
    lines.append("- Trackstand gates are marked N/A for this checkpoint.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _run_one_mode(
    mode: str,
    resume_path: str,
    eval_root: str,
    tb_writer,
) -> dict:
    env_cfg, experiment_cfg = resolve_task_config(args_cli.task, agent_cfg_entry_point)

    if args_cli.ml_framework.startswith("torch"):
        from skrl.utils.runner.torch import Runner
    elif args_cli.ml_framework.startswith("jax"):
        from skrl.utils.runner.jax import Runner

    from source.trackstand_rl.trackstand_rl.tasks.manager_based.locomotive.config.terrain_presets import apply_terrain_preset

    if is_terrain_stage(mode):
        if args_cli.terrain:
            print(
                f"[WARN] --terrain {args_cli.terrain} ignored for staged mode {mode} "
                "(the stage selects the generator and pins the row)."
            )
    else:
        apply_terrain_preset(env_cfg, args_cli.terrain)
    meta = configure_eval_scenario(
        env_cfg,
        mode,
        disable_vis=not args_cli.keep_vis,
        speed_sweep_min=args_cli.speed_sweep_min,
        speed_sweep_max=args_cli.speed_sweep_max,
        heading_sweep_min_deg=args_cli.heading_sweep_min,
        heading_sweep_max_deg=args_cli.heading_sweep_max,
        yaw_bias_min_deg=args_cli.yaw_bias_min,
        yaw_bias_max_deg=args_cli.yaw_bias_max,
        drive_accel_min=args_cli.drive_accel_min,
        drive_accel_max=args_cli.drive_accel_max,
        drive_accel_spawn_vx=args_cli.drive_accel_spawn_vx,
        mass_scale_min=args_cli.mass_scale_min,
        mass_scale_max=args_cli.mass_scale_max,
        com_offset_min_m=args_cli.com_offset_min,
        com_offset_max_m=args_cli.com_offset_max,
        mass_com_scale_min=args_cli.mass_com_scale_min,
        mass_com_scale_max=args_cli.mass_com_scale_max,
        mass_com_offset_min_m=args_cli.mass_com_offset_min,
        mass_com_offset_max_m=args_cli.mass_com_offset_max,
        friction_mu_min=args_cli.friction_mu_min,
        friction_mu_max=args_cli.friction_mu_max,
        kick_force_min=args_cli.kick_force_min,
        kick_force_max=args_cli.kick_force_max,
    )
    print(f"\n[Eval] mode={mode}: {meta['description']}")

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    apply_play_camera_cfg(env_cfg, args_cli)

    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    experiment_cfg["seed"] = args_cli.seed
    env_cfg.seed = experiment_cfg["seed"]
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir
    apply_plant_cfg_from_checkpoint(env_cfg, log_dir)
    apply_drive_accel_sweep_plant(env_cfg, meta)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg) and algorithm in ["ppo"]:
        from isaaclab.envs import multi_agent_to_single_agent
        env = multi_agent_to_single_agent(env)

    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "play", mode),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    _camera_ui = attach_play_camera_ui(env, args_cli)

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    scheduler = experiment_cfg.get("agent", {}).get("learning_rate_scheduler")
    if isinstance(scheduler, str) and scheduler == "CosineAnnealingLR":
        from torch.optim.lr_scheduler import CosineAnnealingLR as _Sched
        experiment_cfg["agent"]["learning_rate_scheduler"] = _Sched

    runner = Runner(env, experiment_cfg)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.enable_training_mode(False, apply_to_models=True)

    num_envs = getattr(env, "num_envs", args_cli.num_envs)
    device = str(env_cfg.sim.device)
    logger = BicycleMetricsLogger.from_env(
        env,
        device=device,
        step_dt=dt,
        stable_roll_threshold_deg=5.0,
        record_history=not args_cli.no_plots,
        history_downsample=args_cli.history_downsample,
        log_dir=log_dir,
    )
    if logger.contract is not None:
        meta["eval_contract"] = logger.contract.to_dict()
    steer_idx = int(getattr(logger.contract, "steer_action_index", 0) or 0)
    meta["ablate_steer"] = args_cli.ablate_steer
    if args_cli.ablate_steer != "policy":
        meta["description"] = f"[ablate_steer={args_cli.ablate_steer}] " + str(meta.get("description", ""))

    pin_level = meta.get("pin_terrain_level")
    if pin_level is not None:
        pin_info = pin_eval_terrain_level(env, int(pin_level))
        meta["pin"] = pin_info

    obs, _ = env.reset()
    logger.sync_episode_start(env.unwrapped)
    states = env.state()
    timestep = 0
    print(f"[EvalLoop] {mode}: {args_cli.eval_steps} steps × {num_envs} envs")
    t_start = time.time()

    try:
        while timestep < args_cli.eval_steps:
            with torch.inference_mode():
                outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
                if hasattr(env, "possible_agents"):
                    actions = {
                        a: outputs[-1][a].get("mean_actions", outputs[0][a])
                        for a in env.possible_agents
                    }
                else:
                    actions = outputs[-1].get("mean_actions", outputs[0])
                    if args_cli.ablate_steer == "lock":
                        actions = actions.clone()
                        actions[:, steer_idx] = 0.0
                obs, _, _, _, _ = env.step(actions)
                states = env.state()
                logger.step(env.unwrapped, actions)

            timestep += 1
            if args_cli.video and timestep == args_cli.video_length:
                break
            if timestep % max(1, args_cli.eval_steps // 10) == 0:
                elapsed = time.time() - t_start
                fps = timestep * num_envs / max(elapsed, 1e-6)
                print(
                    f"  [{mode}] Step {timestep}/{args_cli.eval_steps}  |  "
                    f"{fps:,.0f} env-steps/s  |  {len(logger.episodes)} episodes"
                )
    except KeyboardInterrupt:
        print(f"\n[EvalLoop] Interrupted during mode={mode} — finalising early ...")

    elapsed_total = time.time() - t_start
    print(
        f"[EvalLoop] {mode} finished in {elapsed_total:.1f} s  "
        f"({timestep * num_envs / max(elapsed_total, 1e-6):,.0f} env-steps/s)"
    )

    summary = logger.finalize(env.unwrapped)
    gates = evaluate_gates(summary, mode=mode)
    print("\n" + summary.pretty_print())
    print("\n" + format_gate_report(gates))

    mode_dir = os.path.join(eval_root, mode)
    os.makedirs(mode_dir, exist_ok=True)
    plot_paths: dict[str, str] = {}
    if not args_cli.no_plots:
        plotter = BicycleEvalPlotter(
            logger,
            save_dir=os.path.join(mode_dir, "plots"),
            interactive=not args_cli.no_plotly,
        )
        plot_paths = plotter.generate_all_plots()

    exported_files = export_eval_report(summary, logger, save_dir=os.path.join(mode_dir, "reports"))
    with open(os.path.join(mode_dir, "gates.json"), "w") as f:
        json.dump(gates, f, indent=2)
    with open(os.path.join(mode_dir, "scenario.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(mode_dir, "summary.json"), "w") as f:
        json.dump(summary.to_dict(), f, indent=2)

    speed_limits_payload = None
    if mode == "speed_sweep":
        speed_analysis = analyze_speed_limits(
            logger.episodes,
            survival_min=args_cli.speed_balance_survival,
            drive_jitter_max=args_cli.speed_drive_jitter_max,
            drive_bang_ratio_max=args_cli.speed_drive_bang_max,
            v_min=meta.get("speed_sweep_min"),
            v_max=meta.get("speed_sweep_max"),
        )
        print(format_speed_limit_report(speed_analysis))
        speed_limits_path = export_speed_limits(
            speed_analysis, os.path.join(mode_dir, "speed_limits.json")
        )
        speed_limits_payload = speed_analysis.to_dict()
        if not args_cli.no_plots:
            plot_path = plot_speed_limits(
                speed_analysis, os.path.join(mode_dir, "plots", "speed_limits.png")
            )
            if plot_path:
                plot_paths["speed_limits"] = plot_path
        print(f"[Eval] Speed limits saved to: {speed_limits_path}")

    heading_limits_payload = None
    if mode == "heading_sweep":
        heading_analysis = analyze_heading_limits(
            logger.episodes,
            survival_min=args_cli.speed_balance_survival,
            heading_rmse_max_deg=args_cli.heading_rmse_max,
            h_min_deg=meta.get("heading_sweep_min_deg"),
            h_max_deg=meta.get("heading_sweep_max_deg"),
        )
        print(format_heading_limit_report(heading_analysis))
        heading_limits_path = export_heading_limits(
            heading_analysis, os.path.join(mode_dir, "heading_limits.json")
        )
        heading_limits_payload = heading_analysis.to_dict()
        if not args_cli.no_plots:
            plot_path = plot_heading_limits(
                heading_analysis, os.path.join(mode_dir, "plots", "heading_limits.png")
            )
            if plot_path:
                plot_paths["heading_limits"] = plot_path
        print(f"[Eval] Heading limits saved to: {heading_limits_path}")

    yaw_bias_payload = None
    if mode == "yaw_bias_sweep":
        yaw_analysis = analyze_yaw_bias(
            logger.episodes,
            survival_min=args_cli.speed_balance_survival,
            bias_min_deg=meta.get("yaw_bias_min_deg"),
            bias_max_deg=meta.get("yaw_bias_max_deg"),
            yaw_rate_scale=float(meta.get("yaw_rate_scale", 0.45)),
            stick_full_scale_deg=float(meta.get("yaw_stick_full_scale_deg", 10.0)),
        )
        print(format_yaw_bias_report(yaw_analysis))
        yaw_bias_path = export_yaw_bias(
            yaw_analysis, os.path.join(mode_dir, "yaw_bias.json")
        )
        yaw_bias_payload = yaw_analysis.to_dict()
        if not args_cli.no_plots:
            plot_path = plot_yaw_bias(
                yaw_analysis, os.path.join(mode_dir, "plots", "yaw_bias.png")
            )
            if plot_path:
                plot_paths["yaw_bias"] = plot_path
        print(f"[Eval] Yaw bias sweep saved to: {yaw_bias_path}")

    drive_accel_payload = None
    if mode == "drive_accel_sweep":
        drive_analysis = analyze_drive_accel(
            logger.episodes,
            survival_min=args_cli.speed_balance_survival,
            a_min=meta.get("drive_accel_min"),
            a_max=meta.get("drive_accel_max"),
        )
        print(format_drive_accel_report(drive_analysis))
        drive_accel_path = export_drive_accel(
            drive_analysis, os.path.join(mode_dir, "drive_accel.json")
        )
        drive_accel_payload = drive_analysis.to_dict()
        if not args_cli.no_plots:
            plot_path = plot_drive_accel(
                drive_analysis, os.path.join(mode_dir, "plots", "drive_accel.png")
            )
            if plot_path:
                plot_paths["drive_accel"] = plot_path
        print(f"[Eval] Drive accel sweep saved to: {drive_accel_path}")

    mass_limits_payload = None
    if mode == "mass_sweep":
        mass_analysis = analyze_mass_limits(
            logger.episodes,
            survival_min=args_cli.speed_balance_survival,
            scale_min=meta.get("mass_scale_min"),
            scale_max=meta.get("mass_scale_max"),
        )
        print(format_mass_limits_report(mass_analysis))
        mass_limits_path = export_mass_limits(
            mass_analysis, os.path.join(mode_dir, "mass_limits.json")
        )
        mass_limits_payload = mass_analysis.to_dict()
        if not args_cli.no_plots:
            plot_path = plot_mass_limits(
                mass_analysis, os.path.join(mode_dir, "plots", "mass_limits.png")
            )
            if plot_path:
                plot_paths["mass_limits"] = plot_path
        print(f"[Eval] Mass sweep saved to: {mass_limits_path}")

    com_limits_payload = None
    if mode == "com_sweep":
        com_analysis = analyze_com_limits(
            logger.episodes,
            survival_min=args_cli.speed_balance_survival,
            offset_min_m=meta.get("com_offset_min_m"),
            offset_max_m=meta.get("com_offset_max_m"),
        )
        print(format_com_limits_report(com_analysis))
        com_limits_path = export_com_limits(
            com_analysis, os.path.join(mode_dir, "com_limits.json")
        )
        com_limits_payload = com_analysis.to_dict()
        if not args_cli.no_plots:
            plot_path = plot_com_limits(
                com_analysis, os.path.join(mode_dir, "plots", "com_limits.png")
            )
            if plot_path:
                plot_paths["com_limits"] = plot_path
        print(f"[Eval] CoM sweep saved to: {com_limits_path}")

    mass_com_limits_payload = None
    if mode == "mass_com_sweep":
        mass_com_analysis = analyze_mass_com_limits(
            logger.episodes,
            survival_min=args_cli.speed_balance_survival,
            scale_min=meta.get("mass_com_scale_min"),
            scale_max=meta.get("mass_com_scale_max"),
            offset_min_m=meta.get("mass_com_offset_min_m"),
            offset_max_m=meta.get("mass_com_offset_max_m"),
        )
        print(format_mass_com_limits_report(mass_com_analysis))
        mass_com_limits_path = export_mass_com_limits(
            mass_com_analysis, os.path.join(mode_dir, "mass_com_limits.json")
        )
        mass_com_limits_payload = mass_com_analysis.to_dict()
        if not args_cli.no_plots:
            plot_path = plot_mass_com_limits(
                mass_com_analysis, os.path.join(mode_dir, "plots", "mass_com_limits.png")
            )
            if plot_path:
                plot_paths["mass_com_limits"] = plot_path
        print(f"[Eval] Mass×CoM grid saved to: {mass_com_limits_path}")

    friction_limits_payload = None
    if mode == "friction_sweep":
        friction_analysis = analyze_friction_limits(
            logger.episodes,
            survival_min=args_cli.speed_balance_survival,
            mu_min=meta.get("friction_mu_min"),
            mu_max=meta.get("friction_mu_max"),
        )
        print(format_friction_limits_report(friction_analysis))
        friction_limits_path = export_friction_limits(
            friction_analysis, os.path.join(mode_dir, "friction_limits.json")
        )
        friction_limits_payload = friction_analysis.to_dict()
        if not args_cli.no_plots:
            plot_path = plot_friction_limits(
                friction_analysis, os.path.join(mode_dir, "plots", "friction_limits.png")
            )
            if plot_path:
                plot_paths["friction_limits"] = plot_path
        print(f"[Eval] Friction sweep saved to: {friction_limits_path}")

    kick_limits_payload = None
    if mode == "kick_sweep":
        kick_analysis = analyze_kick_limits(
            logger.episodes,
            survival_min=args_cli.speed_balance_survival,
            force_min=meta.get("kick_force_min"),
            force_max=meta.get("kick_force_max"),
        )
        print(format_kick_limits_report(kick_analysis))
        kick_limits_path = export_kick_limits(
            kick_analysis, os.path.join(mode_dir, "kick_limits.json")
        )
        kick_limits_payload = kick_analysis.to_dict()
        if not args_cli.no_plots:
            plot_path = plot_kick_limits(
                kick_analysis, os.path.join(mode_dir, "plots", "kick_limits.png")
            )
            if plot_path:
                plot_paths["kick_limits"] = plot_path
        print(f"[Eval] Kick sweep saved to: {kick_limits_path}")

    if args_cli.log_wandb and plot_paths:
        log_plots_to_wandb(plot_paths)
        try:
            import wandb
            wandb.log({f"{mode}/{k}": v for k, v in summary.to_isaaclab_log_dict().items()})
        except ImportError:
            pass
    if args_cli.log_tensorboard and plot_paths and tb_writer is not None:
        log_plots_to_tensorboard(plot_paths, tb_writer)
        for k, v in summary.to_isaaclab_log_dict().items():
            tb_writer.add_scalar(f"{mode}/{k}", v, 0)

    env.close()
    return {
        "summary": summary.to_dict(),
        "gates": gates,
        "meta": meta,
        "files": exported_files,
        "mode_dir": mode_dir,
        "speed_limits": speed_limits_payload,
        "heading_limits": heading_limits_payload,
        "yaw_bias": yaw_bias_payload,
        "drive_accel": drive_accel_payload,
        "mass_limits": mass_limits_payload,
        "com_limits": com_limits_payload,
        "mass_com_limits": mass_com_limits_payload,
        "friction_limits": friction_limits_payload,
        "kick_limits": kick_limits_payload,
    }


def _load_eval_paths():
    """Import ``scripts/eval/eval_paths.py`` without depending on the process cwd."""
    eval_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eval_locomotive"))
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    import eval_paths

    return eval_paths


def main():
    """Play with skrl agent + full evaluation metrics."""
    env_cfg, experiment_cfg = resolve_task_config(args_cli.task, agent_cfg_entry_point)

    from isaaclab.app import launch_simulation
    with launch_simulation(env_cfg, args_cli):
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")
        resume_path = _resolve_checkpoint(train_task_name)
        log_dir = os.path.dirname(os.path.dirname(resume_path))
        requested = os.path.abspath(args_cli.save_dir or os.path.join(log_dir, "eval"))
        modes = _parse_modes()
        eval_root = requested
        paths = _load_eval_paths()
        redirected = False
        if not args_cli.overwrite and paths.modes_would_overwrite(requested, modes):
            allocated, redirected = paths.allocate_eval_dir(requested, overwrite=False)
            eval_root = str(allocated)
            if redirected:
                paths.note_eval_allocation(
                    allocated,
                    requested=requested,
                    redirected=True,
                    checkpoint=os.path.abspath(resume_path),
                )
                print(
                    f"[Eval] {requested} already has summaries for {', '.join(modes)}; "
                    f"this re-run writes to {eval_root}"
                )
        else:
            os.makedirs(eval_root, exist_ok=True)

        tb_writer = None
        if args_cli.log_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=os.path.join(eval_root, "tensorboard"))

        print(f"[Eval] suite={modes}")
        print(f"[Eval] checkpoint={resume_path}")
        print("[Eval] Trackstand gates skipped — this policy is dynamic locomotion only.")

        try:
            results = {}
            for mode in modes:
                results[mode] = _run_one_mode(mode, resume_path, eval_root, tb_writer)

            suite_path = os.path.join(eval_root, "suite.json")
            with open(suite_path, "w") as f:
                json.dump(results, f, indent=2)
            md_path = os.path.join(eval_root, "SUMMARY.md")
            _write_suite_markdown(md_path, results, resume_path)

            print("\n========== SUITE ==========")
            for mode, payload in results.items():
                g = payload["gates"]
                mark = "PASS" if g["all_passed"] else "FAIL"
                line = (
                    f"  {mode:<12s}  {mark}  {g['n_pass']}/{g['n_gates']}  "
                    f"survival={payload['summary']['survival_rate']:.1%}"
                )
                sl = payload.get("speed_limits")
                if sl is not None:
                    mb = sl.get("min_balance_speed_ms")
                    sd = sl.get("smooth_drive_speed_ms")
                    if mb is not None:
                        line += f"  min_balance={mb:.3f} m/s"
                    if sd is not None:
                        line += f"  smooth_drive≥{sd:.3f} m/s"
                print(line)
            print(f"\n[EvalLoop] All outputs saved to: {eval_root}/")
        finally:
            if redirected:
                paths.publish_eval_dir(eval_root)
        if tb_writer is not None:
            tb_writer.close()


if __name__ == "__main__":
    main()
