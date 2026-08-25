#!/usr/bin/env python3
"""
Evaluation script with comprehensive metrics logging for bicycle balancing.

Loads a trained checkpoint, runs the policy, and produces:
  - Terminal summary table (survival rate, drift, roll/pitch stats, power, …)
  - Per-episode CSV + JSON report
  - Four diagnostic plots (trajectory, velocity, action/torque, posture)

Usage (inside the Isaac Sim container)::

    python scripts/skrl/play_log.py \\
        --task CycleRL:THUBike-Balance-v0 \\
        --checkpoint /path/to/agent.pt \\
        --num_envs 256 \\
        --eval_steps 1500

Or auto-resolve the latest checkpoint::

    python scripts/skrl/play_log.py \\
        --task CycleRL:THUBike-Balance-v0 \\
        --num_envs 64

Output lands in ``./eval_results/<task_name>/<timestamp>/``:
    reports/eval_report_*.json   — per-episode detail + aggregate summary
    reports/eval_report_*.csv    — flat table (one row per episode)
    plots/trajectory_2d.png      — top-down drift map
    plots/velocity_profile.png   — target vs actual velocity
    plots/action_torque.png      — torques + actions + jitter
    plots/posture_stability.png  — roll & pitch over time
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


SKRL_VERSION = "2.1.0"

# ===========================================================================
# CLI
# ===========================================================================

parser = argparse.ArgumentParser(
    description="Play a checkpoint with full evaluation metrics logging."
)

# --- Standard play_skrl arguments ------------------------------------------
parser.add_argument("--task", type=str, default="CycleRL:THUBike-Balance-v0",
                    help="Gym task id (e.g. CycleRL:THUBike-Balance-v0).")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to model checkpoint.  Auto-resolves latest if omitted.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true",
                    help="Use a published pre-trained checkpoint from Nucleus.")
parser.add_argument("--num_envs", type=int, default=256,
                    help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--agent", type=str, default=None,
                    help="skrl agent config entry point (auto-detected from --algorithm).")
parser.add_argument("--algorithm", type=str, default="PPO",
                    choices=["PPO", "IPPO", "MAPPO"])
parser.add_argument("--ml_framework", type=str, default="torch",
                    choices=["torch", "jax"])
parser.add_argument("--real_time", action="store_true", default=False,
                    help="Throttle to real-time if possible.")
parser.add_argument("--video", action="store_true", default=False,
                    help="Record a video of env_0 during play.")
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
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

# --- Evaluation-specific arguments -----------------------------------------
parser.add_argument("--eval_steps", type=int, default=1500,
                    help="Number of env steps to run (1500 = one 30-s episode at dt=0.02).")
parser.add_argument("--output_dir", type=str, default="./eval_results",
                    help="Root directory for all evaluation outputs.")
parser.add_argument("--no_plots", action="store_true", default=False,
                    help="Skip plot generation (metrics only).")
parser.add_argument("--history_downsample", type=int, default=1,
                    help="Record every Nth step in the history buffer (1 = all steps).")
parser.add_argument("--stable_roll_deg", type=float, default=3.0,
                    help="Roll angle below which balance is considered 'stable' (degrees).")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# parse_known_args() silently swallows unknown flags (e.g. a typo'd --Task), which
# then surface as a confusing crash minutes later. Reject them here instead.
_unknown = [a for a in hydra_args if a.startswith("-")]
if _unknown:
    parser.error(f"unrecognized arguments: {' '.join(_unknown)}")
if args_cli.task is None:
    parser.error("--task is required (flags are case-sensitive: --task, not --Task)")

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# Start Kit BEFORE importing torch / gym / isaaclab task modules
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from packaging import version

import skrl

# ---------------------------------------------------------------------------
# Isaac Lab imports
# ---------------------------------------------------------------------------
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import isaaclab_tasks  # noqa: F401

# ---------------------------------------------------------------------------
# CycleRL imports — registers the custom gym environments
# ---------------------------------------------------------------------------
import CycleRL.tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# ---------------------------------------------------------------------------
# Our evaluation logger
# ---------------------------------------------------------------------------
from CycleRL.tasks.manager_based.cyclerl.eval_logger import (
    BicycleEvalPlotter,
    BicycleMetricsLogger,
    EvalSummary,
    capture_env_config,
    export_eval_report,
)

# ---------------------------------------------------------------------------
# skrl version guard
# ---------------------------------------------------------------------------
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Resolve agent entry point
# ---------------------------------------------------------------------------
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = (
        "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
    )
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
task_short = args_cli.task.split(":")[-1]
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = Path(args_cli.output_dir) / task_short / timestamp
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "eval.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    """Run a full evaluation with metrics logging."""
    log.info("=" * 60)
    log.info("  CycleRL Evaluation — %s", task_short)
    log.info("  Output: %s", OUTPUT_DIR.resolve())
    log.info("=" * 60)

    # ---- Resolve task config -----------------------------------------------
    # ponytail: plain registry load, no hydra composition (see train.py)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    experiment_cfg = load_cfg_from_registry(args_cli.task, agent_cfg_entry_point)

    with contextlib.nullcontext():  # ponytail: keeps indentation; Kit is closed after main()
        # ---- skrl Runner (framework-specific) ------------------------------
        if args_cli.ml_framework.startswith("torch"):
            from skrl.utils.runner.torch import Runner
        elif args_cli.ml_framework.startswith("jax"):
            from skrl.utils.runner.jax import Runner

        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        from CycleRL.tasks.manager_based.cyclerl.config.terrain_presets import apply_terrain_preset

        apply_terrain_preset(env_cfg, args_cli.terrain)

        # ---- Override config from CLI ------------------------------------
        env_cfg.scene.num_envs = (
            args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        )
        env_cfg.sim.device = (
            args_cli.device if args_cli.device is not None else env_cfg.sim.device
        )

        if args_cli.ml_framework.startswith("jax"):
            skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        experiment_cfg["seed"] = args_cli.seed
        env_cfg.seed = experiment_cfg["seed"]

        # ---- Checkpoint ---------------------------------------------------
        log_root_path = os.path.join(
            "logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"]
        )
        log_root_path = os.path.abspath(log_root_path)
        log.info("Experiment directory: %s", log_root_path)

        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
            if not resume_path:
                log.error("No pre-trained checkpoint available for %s.", train_task_name)
                return
        elif args_cli.checkpoint:
            resume_path = os.path.abspath(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(
                log_root_path,
                run_dir=f".*_{algorithm}_{args_cli.ml_framework}",
                other_dirs=["checkpoints"],
            )
        log.info("Checkpoint: %s", resume_path)

        log_dir = os.path.dirname(os.path.dirname(resume_path))
        env_cfg.log_dir = log_dir

        # ---- Create environment -------------------------------------------
        render_mode = "rgb_array" if args_cli.video else None
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

        # Convert multi-agent → single-agent if needed.
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg) and algorithm in ["ppo"]:
            from isaaclab.envs import multi_agent_to_single_agent
            env = multi_agent_to_single_agent(env)

        try:
            dt = env.step_dt
        except AttributeError:
            dt = env.unwrapped.step_dt
        log.info("Step dt: %.3f s  |  Episode length: %.0f s", dt, env_cfg.episode_length_s)

        # ---- Video recording (optional) -----------------------------------
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            env = gym.wrappers.RecordVideo(env, **video_kwargs)
            log.info("Video recording enabled (%d steps).", args_cli.video_length)

        # ---- Wrap for skrl ------------------------------------------------
        env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

        # ---- Configure & load agent ---------------------------------------
        experiment_cfg["trainer"]["close_environment_at_exit"] = False
        experiment_cfg["agent"]["experiment"]["write_interval"] = 0
        experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        runner = Runner(env, experiment_cfg)
        runner.agent.load(resume_path)
        runner.agent.enable_training_mode(False, apply_to_models=True)
        log.info("Agent loaded successfully.")

        # ═════════════════════════════════════════════════════════════════
        #  Set up the evaluation metrics logger
        # ═════════════════════════════════════════════════════════════════
        num_envs = getattr(env, "num_envs", args_cli.num_envs)
        device = str(env_cfg.sim.device)

        logger = BicycleMetricsLogger.from_env(
            env,
            device=device,
            step_dt=dt,
            stable_roll_threshold_deg=args_cli.stable_roll_deg,
            record_history=not args_cli.no_plots,
            history_downsample=args_cli.history_downsample,
        )
        log.info(
            "Metrics logger ready — %d envs, steer=%s, drive=%s",
            num_envs, logger.steer_joint_name, logger.drive_joint_name,
        )

        # ═════════════════════════════════════════════════════════════════
        #  Evaluation loop
        # ═════════════════════════════════════════════════════════════════
        obs, _ = env.reset()
        logger.sync_episode_start(env.unwrapped)
        states = env.state()
        step = 0

        log.info(
            "Running %d steps across %d environments (≈%.0f episodes per env) ...",
            args_cli.eval_steps, num_envs,
            args_cli.eval_steps * dt / env_cfg.episode_length_s,
        )
        t_start = time.time()

        try:
            while step < args_cli.eval_steps:
                start_time = time.time()

                with torch.inference_mode():
                    # --- Agent inference (same pattern as play_skrl.py) ---
                    outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
                    if hasattr(env, "possible_agents"):
                        actions = {
                            a: outputs[-1][a].get("mean_actions", outputs[0][a])
                            for a in env.possible_agents
                        }
                    else:
                        actions = outputs[-1].get("mean_actions", outputs[0])

                    # --- Environment step ---------------------------------
                    obs, _, _, _, _ = env.step(actions)
                    states = env.state()

                    # --- Log metrics (GPU-vectorized) ----------------------
                    logger.step(env.unwrapped, actions)

                step += 1

                # --- Real-time throttling ---------------------------------
                if args_cli.real_time:
                    sleep_time = dt - (time.time() - start_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                # --- Progress report --------------------------------------
                if step % max(1, args_cli.eval_steps // 10) == 0:
                    elapsed = time.time() - t_start
                    fps = step * num_envs / max(elapsed, 1e-6)
                    n_eps = len(logger.episodes)
                    log.info(
                        "  Step %d/%d  |  %,.0f env-steps/s  |  %d episodes completed",
                        step, args_cli.eval_steps, fps, n_eps,
                    )

        except KeyboardInterrupt:
            log.info("Interrupted by user — finalising early (step %d / %d).",
                     step, args_cli.eval_steps)

        elapsed_total = time.time() - t_start
        total_env_steps = step * num_envs
        log.info(
            "Evaluation finished in %.1f s  (%,.0f env-steps/s, %d total steps).",
            elapsed_total, total_env_steps / max(elapsed_total, 1e-6), step,
        )

        # ═════════════════════════════════════════════════════════════════
        #  Finalize & report
        # ═════════════════════════════════════════════════════════════════
        summary: EvalSummary = logger.finalize()

        # --- Capture the environment config that was evaluated ------------
        log.info("Capturing environment configuration …")
        env_config_dict = capture_env_config(env.unwrapped.cfg)

        # --- Terminal table -----------------------------------------------
        print("\n" + summary.pretty_print())

        # --- Save config JSON ---------------------------------------------
        config_path = OUTPUT_DIR / "config.json"
        config_path.write_text(json.dumps(env_config_dict, indent=2))
        log.info("Config saved → %s", config_path)

        # --- Save summary JSON -------------------------------------------
        summary_path = OUTPUT_DIR / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "task": args_cli.task,
                    "checkpoint": resume_path,
                    "timestamp": timestamp,
                    "num_envs": num_envs,
                    "eval_steps_requested": args_cli.eval_steps,
                    "eval_steps_executed": step,
                    "step_dt": dt,
                    "elapsed_s": elapsed_total,
                    "config": env_config_dict,
                    "summary": summary.to_dict(),
                },
                indent=2,
            )
        )
        log.info("Summary saved → %s", summary_path)

        # --- Per-episode reports ------------------------------------------
        report_dir = OUTPUT_DIR / "reports"
        export_eval_report(summary, logger, save_dir=str(report_dir))

        # --- Plots --------------------------------------------------------
        if not args_cli.no_plots:
            plot_dir = OUTPUT_DIR / "plots"
            plotter = BicycleEvalPlotter(logger, save_dir=str(plot_dir))
            plot_paths = plotter.generate_all_plots()
            for name, path in plot_paths.items():
                log.info("Plot [%s] → %s", name, path)

        # --- Done ---------------------------------------------------------
        log.info("All outputs saved to: %s", OUTPUT_DIR.resolve())

        env.close()


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    main()
    simulation_app.close()
