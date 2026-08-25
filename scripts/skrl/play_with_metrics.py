#!/usr/bin/env python3
"""Play a checkpoint with full evaluation metrics (BicycleMetricsLogger).

Locomotion-only gates (no trackstand). Example:

    cd ~/IsaacLab
    ./isaaclab.sh -p ~/bike_test/CycleRL/scripts/skrl/play_with_metrics.py \\
        --task CycleRL-NewBike-v0 \\
        --checkpoint /path/to/best_agent.pt \\
        --eval_suite nominal,kick,mass,friction,training \\
        --num_envs 64 --headless

Staged terrain extremity (pins every env to one difficulty row):

    --eval_suite terrain              # complex_L0..L4 then stairs_L0..L2
    --eval_suite complex              # complex_L0..L4
    --eval_suite stairs               # stairs_L0..L2
    --eval_mode complex_L2
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
from isaaclab.app import AppLauncher

# -- argparse (ALL args must be defined before AppLauncher) ----------------
parser = argparse.ArgumentParser(description="Play a checkpoint with full evaluation metrics.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="CycleRL-NewBike-v0")
parser.add_argument("--agent", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
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
    default=3600,
    help="Env steps per scenario. 3600 ≈ one 64 s episode (dt=0.02) plus a short buffer.",
)
parser.add_argument("--save_dir", type=str, default=None, help="Override output root. Default: <run>/eval/")
parser.add_argument("--log_wandb", action="store_true", default=False)
parser.add_argument("--log_tensorboard", action="store_true", default=False)
parser.add_argument("--no_plots", action="store_true", default=False)
parser.add_argument("--history_downsample", type=int, default=1)
parser.add_argument(
    "--eval_mode",
    type=str,
    default="nominal",
    help=(
        "Single scenario: nominal | training | kick | mass | friction | "
        "complex_L0..L4 | stairs_L0..L2."
    ),
)
parser.add_argument(
    "--eval_suite",
    type=str,
    default=None,
    help=(
        "Comma-separated scenarios (overrides --eval_mode). Aliases: "
        "terrain (all 8 staged rows), complex (L0-L4), stairs (L0-L2). "
        "Example: nominal,kick,terrain"
    ),
)
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
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import isaaclab_tasks  # noqa: F401


def resolve_task_config(task_name: str, agent_cfg_entry_point: str):
    """IsaacLab-main stand-in for develop's ``resolve_task_config``.

    ponytail: plain registry load — no hydra composition, so `key=value` CLI
    overrides are unsupported. Swap in ``hydra_task_config`` if you need them.
    """
    return (
        parse_env_cfg(task_name, device=args_cli.device, num_envs=args_cli.num_envs),
        load_cfg_from_registry(task_name, agent_cfg_entry_point),
    )

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# ----- CycleRL imports ----------------------------------------------------
import CycleRL.tasks  # noqa: F401
from CycleRL.tasks.manager_based.cyclerl.eval_logger import (
    BicycleEvalPlotter,
    BicycleMetricsLogger,
    evaluate_gates,
    export_eval_report,
    format_gate_report,
    log_plots_to_tensorboard,
    log_plots_to_wandb,
)
from CycleRL.tasks.manager_based.cyclerl.eval_scenarios import (
    configure_eval_scenario,
    expand_eval_modes,
    is_terrain_stage,
    pin_eval_terrain_level,
)

SKRL_VERSION = "2.1.0"

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


def _resolve_checkpoint(experiment_cfg: dict, train_task_name: str) -> str:
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
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
    return get_checkpoint_path(
        log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
    )


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
        mark = "PASS" if g["all_passed"] else "FAIL"
        lines.append(
            f"| {mode} | {s['survival_rate']:.1%} | {s['mean_roll_rmse_straight_deg']:.2f}° | "
            f"{s['mean_vx_ss_mae_ms']:.3f} m/s | {s['mean_action_clip_sat_ratio']:.1%} | "
            f"{mark} {g['n_pass']}/{g['n_gates']} |"
        )
    lines += ["", "## Notes", ""]
    lines.append("- Nominal: flat plane, μ=1.0, no domain rand, vx*=1.8 m/s, heading*=0.")
    lines.append("- Kick: 10 N lateral on `base_link` for 0.1 s at t=2 s.")
    lines.append("- Mass: mass ×1.10 and CoM ±2 cm.")
    lines.append("- Friction: ground and tires μ=0.4.")
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

    from CycleRL.tasks.manager_based.cyclerl.config.terrain_presets import apply_terrain_preset

    if is_terrain_stage(mode):
        if args_cli.terrain:
            print(
                f"[WARN] --terrain {args_cli.terrain} ignored for staged mode {mode} "
                "(the stage selects the generator and pins the row)."
            )
    else:
        apply_terrain_preset(env_cfg, args_cli.terrain)
    meta = configure_eval_scenario(env_cfg, mode)
    print(f"\n[Eval] mode={mode}: {meta['description']}")

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    experiment_cfg["seed"] = args_cli.seed
    env_cfg.seed = experiment_cfg["seed"]
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

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
    )
    if logger.contract is not None:
        meta["eval_contract"] = logger.contract.to_dict()

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
        plotter = BicycleEvalPlotter(logger, save_dir=os.path.join(mode_dir, "plots"))
        plot_paths = plotter.generate_all_plots()

    exported_files = export_eval_report(summary, logger, save_dir=os.path.join(mode_dir, "reports"))
    with open(os.path.join(mode_dir, "gates.json"), "w") as f:
        json.dump(gates, f, indent=2)
    with open(os.path.join(mode_dir, "scenario.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(mode_dir, "summary.json"), "w") as f:
        json.dump(summary.to_dict(), f, indent=2)

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
    }


def main():
    """Play with skrl agent + full evaluation metrics."""
    env_cfg, experiment_cfg = resolve_task_config(args_cli.task, agent_cfg_entry_point)

    with contextlib.nullcontext():  # ponytail: keeps indentation; Kit is closed after main()
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")
        resume_path = _resolve_checkpoint(experiment_cfg, train_task_name)
        log_dir = os.path.dirname(os.path.dirname(resume_path))
        eval_root = args_cli.save_dir or os.path.join(log_dir, "eval")
        eval_root = os.path.abspath(eval_root)
        os.makedirs(eval_root, exist_ok=True)

        tb_writer = None
        if args_cli.log_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=os.path.join(eval_root, "tensorboard"))

        modes = _parse_modes()
        print(f"[Eval] suite={modes}")
        print(f"[Eval] checkpoint={resume_path}")
        print("[Eval] Trackstand gates skipped — this policy is dynamic locomotion only.")

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
            print(f"  {mode:<12s}  {mark}  {g['n_pass']}/{g['n_gates']}  survival={payload['summary']['survival_rate']:.1%}")
        print(f"\n[EvalLoop] All outputs saved to: {eval_root}/")
        if tb_writer is not None:
            tb_writer.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
