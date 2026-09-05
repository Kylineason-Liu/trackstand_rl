# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a skrl checkpoint and print summary metrics (headless-friendly).

Runs the trained policy for a fixed number of steps across parallel envs and
summarises, over completed episodes:

    mean_reward, mean/min/max/std episode_length, mean/max roll angle

plus high-value extras: fall/timeout rate, roll RMS, per-episode worst lean,
drive/steer motor speeds, and position drift from spawn.

Loading mirrors scripts/skrl/play_skrl.py; the metric intent mirrors
scripts/rsl_rl/play_eval.py (but focused, and skrl-native).

Example:
    ~/IsaacLab/isaaclab.sh -p scripts/skrl/play_eval.py --task NewBike-Balance-v0 \\
        --checkpoint logs/skrl/cyclerl_logs/NewBike-Balance-v0/<run>/checkpoints/best_agent.pt \\
        --num_envs 64 --num_steps 3000 --headless

Note: domain randomization is PINNED for eval, not ramped — by default the
`dr_ramp` curriculum is set to FULL DR from step 0 (start_step = end_step = 0);
pass --no_dr to disable DR entirely (alpha pinned to 0). Unlike training, there
is no 50k-step ramp: eval should see the regime the policy must survive. Also:
mean_reward is NOT comparable across reward-weight changes (the scale moves with
the weights) — rank configs by episode length / roll metrics instead.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import os
import sys

# =========================================================================
# CRITICAL FIX: Force cache the top-level isaaclab module in memory.
# This prevents Omniverse Kit's sys.path manipulation during AppLauncher
# startup from breaking internal IsaacLab imports (unknown location error).
# =========================================================================
import isaaclab  # noqa: F401
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a skrl checkpoint with summary metrics.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a skrl checkpoint (.pt) or its run directory.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the published pretrained checkpoint.")
parser.add_argument(
    "--num_steps", type=int, default=3000,
    help="Env steps to run (~2 episodes/env at the 1500-step / 30s cap).",
)
parser.add_argument(
    "--layers", type=str, default=None,
    help="Comma-separated hidden-layer sizes overriding BOTH policy and value networks "
    "(must match the trained checkpoint), e.g. '512,256,128,64'.",
)
parser.add_argument(
    "--no_dr", action="store_true", default=False,
    help="Disable domain randomization entirely (dr_ramp alpha pinned to 0). "
    "Without this flag, eval runs at FULL DR from step 0 (no ramp).",
)
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
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# parse_known_args() silently swallows unknown flags (e.g. a typo'd --Task), which
# then surface as a confusing crash minutes later. Reject them here instead.
_unknown = [a for a in hydra_args if a.startswith("-")]
if _unknown:
    parser.error(f"unrecognized arguments: {' '.join(_unknown)}")
if args_cli.task is None:
    parser.error("--task is required (flags are case-sensitive: --task, not --Task)")
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless: render to file only, never to a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from packaging import version  # noqa: E402

import skrl  # noqa: E402

from isaaclab.envs import DirectMARLEnvCfg  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402

from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg  # noqa: E402

import CycleRL.tasks  # noqa: F401, E402

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

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


def _roll_deg(robot, device):
    """|Roll| in degrees from the RAW root quaternion (xyzw -> euler).

    Raw buffer is fresh in every window; the kernel-cached projected_gravity_b
    is stale for reset envs until the next physics step (audit finding)."""
    roll = math_utils.euler_xyz_from_quat(robot.data.root_quat_w.torch)[0].abs()
    return torch.rad2deg(roll).to(device)


def _plot_torque_traces(resume_path, steps, envs, drive_trace, steer_trace, drive_lim, steer_lim):
    """6 stacked subplots (3 envs x drive/steer) of per-step applied motor torque.

    Rows 1-3: drive torque of the 3 traced envs; rows 4-6: steer torque. Each env
    keeps one colour across both blocks. Dashed lines mark the effort limits.
    Saved next to the checkpoint as ``torque_profile.png`` (plus raw ``.npz``).
    """
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, axes = plt.subplots(6, 1, figsize=(10, 15), sharex=True)
    steps_arr = steps.cpu().numpy()
    for i, (e, c) in enumerate(zip(envs, colors)):
        for block, trace, lim, name in ((0, drive_trace, drive_lim, "drive"), (3, steer_trace, steer_lim, "steer")):
            if trace is None:
                continue
            ax = axes[block + i]
            ax.plot(steps_arr, trace[:, i].cpu().numpy(), color=c, lw=0.7, label=f"env {e}")
            ax.axhline(lim, color="k", ls="--", lw=0.8)
            ax.axhline(-lim, color="k", ls="--", lw=0.8)
            ax.set_ylim(-1.15 * lim, 1.15 * lim)
            ax.set_ylabel(f"env {e}\n{name} τ (N·m)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(loc="upper right", fontsize=7)
    axes[0].set_title(f"applied motor torque vs step — {os.path.basename(resume_path)}")
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    out_dir = os.path.dirname(resume_path)
    fig.savefig(os.path.join(out_dir, "torque_profile.png"), dpi=150)
    plt.close(fig)
    np.savez(
        os.path.join(out_dir, "torque_profile.npz"),
        steps=steps_arr,
        envs=np.array(envs),
        drive=drive_trace.cpu().numpy() if drive_trace is not None else None,
        steer=steer_trace.cpu().numpy() if steer_trace is not None else None,
    )
    print(f"[INFO] torque traces saved next to checkpoint: torque_profile.png / .npz (envs {envs})")


def _resolve_checkpoint(experiment_cfg, train_task_name):
    """Resolve the skrl checkpoint path (same precedence as play_with_metrics.py)."""
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
            if os.path.isfile(best):
                return best
            pts = sorted(
                (os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pt")),
                key=os.path.getmtime,
            )
            if pts:
                return pts[-1]
            raise FileNotFoundError(f"No .pt checkpoint in {ckpt_dir}")
        return path
    return get_checkpoint_path(
        log_root_path,
        run_dir=f".*_{algorithm}_{args_cli.ml_framework}",
        other_dirs=["checkpoints"],
        preferred_checkpoint="best_agent",
    )


def main():
    """Evaluate a skrl agent and print a metric summary."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    experiment_cfg = load_cfg_from_registry(args_cli.task, agent_cfg_entry_point)

    # --layers: must match the trained checkpoint (see train_deterministic.py's flag).
    if args_cli.layers:
        layers = [int(x) for x in args_cli.layers.split(",")]
        experiment_cfg["models"]["policy"]["network"][0]["layers"] = layers
        experiment_cfg["models"]["value"]["network"][0]["layers"] = layers
        print(f"[INFO] network layers overridden to {layers} (policy and value).")

    from CycleRL.tasks.manager_based.cyclerl.config.terrain_presets import apply_terrain_preset

    apply_terrain_preset(env_cfg, args_cli.terrain)

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Pin DR for eval (play-only convention — training keeps its 50k-step ramp):
    # default = FULL DR from step 0 (end_step <= start_step -> alpha = 1.0 on the
    # ramp term's first call, before any reset events fire); --no_dr = alpha pinned
    # to 0 (end_step = 10**9, same mechanism as train_deterministic.py's flag).
    dr_ramp = getattr(getattr(env_cfg, "curriculum", None), "dr_ramp", None)
    if dr_ramp is not None:
        if args_cli.no_dr:
            dr_ramp.params["end_step"] = 10**9
            print("[INFO] --no_dr: domain randomization disabled (dr_ramp alpha pinned to 0).")
        else:
            dr_ramp.params["start_step"] = 0
            dr_ramp.params["end_step"] = 0
            print("[INFO] DR pinned to FULL from step 0 (no ramp); use --no_dr to disable.")
    else:
        print("[WARN] Task has no 'dr_ramp' curriculum term; --no_dr has nothing to disable.")

    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    resume_path = _resolve_checkpoint(experiment_cfg, train_task_name)

    # skrl play convention: the env log dir is the run folder (2 levels above checkpoints/)
    env_cfg.log_dir = os.path.dirname(os.path.dirname(resume_path))

    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # create env + wrap for skrl; disable any checkpoint/TB writes during eval
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg) and algorithm in ["ppo"]:
        from isaaclab.envs import multi_agent_to_single_agent

        env = multi_agent_to_single_agent(env)
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    scheduler = experiment_cfg.get("agent", {}).get("learning_rate_scheduler")
    if isinstance(scheduler, str) and scheduler == "CosineAnnealingLR":
        from torch.optim.lr_scheduler import CosineAnnealingLR as _Sched

        experiment_cfg["agent"]["learning_rate_scheduler"] = _Sched

    if args_cli.ml_framework.startswith("torch"):
        from skrl.utils.runner.torch import Runner
    elif args_cli.ml_framework.startswith("jax"):
        from skrl.utils.runner.jax import Runner

    runner = Runner(env, experiment_cfg)
    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.enable_training_mode(False, apply_to_models=True)

    device = env.unwrapped.device
    num_envs = getattr(env, "num_envs", args_cli.num_envs)
    step_dt = env.unwrapped.step_dt
    robot = env.unwrapped.scene["robot"]
    origins_xy = env.unwrapped.scene.env_origins[:, :2]

    # optional motor joints (skip those metrics if the task lacks them)
    def _joint_idx(name):
        try:
            return robot.find_joints(name)[0][0]
        except Exception:
            return None

    steer_idx = _joint_idx("steering_joint")
    drive_idx = _joint_idx("rearwheel_joint")

    # --- torque tracing: 3 random envs, per-step drive & steer applied torque ---
    _torque_envs = random.sample(range(num_envs), min(3, num_envs))
    _torque_envs.sort()
    n_trace = len(_torque_envs)
    drive_trace = torch.zeros(args_cli.num_steps, n_trace, device=device) if drive_idx is not None else None
    steer_trace = torch.zeros(args_cli.num_steps, n_trace, device=device) if steer_idx is not None else None
    drive_lim = float(robot.data.joint_effort_limits[_torque_envs, drive_idx].max()) if drive_idx is not None else 6.28
    steer_lim = float(robot.data.joint_effort_limits[_torque_envs, steer_idx].max()) if steer_idx is not None else 5.6

    # per-env running accumulators (reset at each episode boundary)
    cur_return = torch.zeros(num_envs, device=device)
    cur_len = torch.zeros(num_envs, dtype=torch.long, device=device)
    ep_max_roll = torch.zeros(num_envs, device=device)
    # completed-episode collections
    ep_returns, ep_lengths, ep_timeouts, ep_maxrolls = [], [], [], []
    # global step-wise stats
    roll_sum = roll_sqsum = roll_max = 0.0
    drive_abs_sum = drive_max = steer_abs_sum = steer_max = drift_sum = drift_max = 0.0
    n_steps_counted = 0

    obs, _ = env.reset()
    states = env.state()
    for t in range(args_cli.num_steps):
        with torch.inference_mode():
            outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            obs, rewards, terminated, truncated, info = env.step(actions)

        rewards = rewards.to(device).view(-1)
        dones_b = (terminated.to(device) | truncated.to(device)).view(-1).bool()
        time_outs = info.get("time_outs") if isinstance(info, dict) else None
        time_outs = (
            time_outs.to(device).view(-1).bool()
            if time_outs is not None
            else torch.zeros(num_envs, dtype=torch.bool, device=device)
        )

        cur_return += rewards
        cur_len += 1

        roll = _roll_deg(robot, device)
        roll_sum += float(roll.sum())
        roll_sqsum += float((roll * roll).sum())
        roll_max = max(roll_max, float(roll.max()))
        ep_max_roll = torch.maximum(ep_max_roll, roll)
        n_steps_counted += num_envs

        if drive_idx is not None:
            dv = robot.data.joint_vel[:, drive_idx].abs()
            drive_abs_sum += float(dv.sum())
            drive_max = max(drive_max, float(dv.max()))
        if steer_idx is not None:
            sv = robot.data.joint_vel[:, steer_idx].abs()
            steer_abs_sum += float(sv.sum())
            steer_max = max(steer_max, float(sv.max()))

        # torque traces for the 3 picked envs (applied torque, post effort-limit clip)
        if drive_idx is not None:
            drive_trace[t] = robot.data.applied_torque[_torque_envs, drive_idx]
        if steer_idx is not None:
            steer_trace[t] = robot.data.applied_torque[_torque_envs, steer_idx]
        drift = torch.norm(robot.data.root_pos_w[:, :2] - origins_xy, dim=-1)
        drift_sum += float(drift.sum())
        drift_max = max(drift_max, float(drift.max()))

        if dones_b.any():
            ep_returns.append(cur_return[dones_b].clone())
            ep_lengths.append(cur_len[dones_b].clone())
            ep_timeouts.append(time_outs[dones_b].clone())
            ep_maxrolls.append(ep_max_roll[dones_b].clone())
            cur_return[dones_b] = 0.0
            cur_len[dones_b] = 0
            ep_max_roll[dones_b] = 0.0

    # --- save torque traces + plot (before closing, data is on-device) ---
    _plot_torque_traces(
        resume_path, torch.arange(args_cli.num_steps, device="cpu"),
        _torque_envs, drive_trace, steer_trace, drive_lim, steer_lim,
    )

    env.close()

    # ---- summarise ----
    def _cat(lst):
        return torch.cat(lst) if lst else torch.empty(0, device=device)

    returns = _cat(ep_returns)
    lengths = _cat(ep_lengths).float()
    timeouts = _cat(ep_timeouts).float()
    maxrolls = _cat(ep_maxrolls)
    n_ep = returns.numel()
    denom = max(n_steps_counted, 1)

    print("\n" + "=" * 64)
    print(f"  EVAL SUMMARY  —  {os.path.basename(resume_path)}")
    print(f"  {num_envs} envs x {args_cli.num_steps} steps  |  {n_ep} completed episodes")
    print("=" * 64)
    if n_ep == 0:
        print("  No episode completed within --num_steps. Increase it, or the")
        print(f"  policy is timing out. In-progress mean length: {float(cur_len.float().mean()):.1f} steps.")
    else:
        surv = float(timeouts.mean())
        print(f"  mean_reward            : {float(returns.mean()):10.3f}   (episodic return; not comparable across reward-weight changes)")
        print(f"  mean_episode_length    : {float(lengths.mean()):10.1f} steps  ({float(lengths.mean()) * step_dt:6.2f} s)")
        print(f"  min_episode_length     : {float(lengths.min()):10.1f} steps  ({float(lengths.min()) * step_dt:6.2f} s)")
        print(f"  max_episode_length     : {float(lengths.max()):10.1f} steps  ({float(lengths.max()) * step_dt:6.2f} s)")
        print(f"  std_episode_length     : {float(lengths.std(unbiased=False)):10.1f} steps")
        print(f"  timeout_rate (survived): {surv:10.3f}   (reached the horizon vs fell)")
        print(f"  fall_rate              : {1.0 - surv:10.3f}")
        print("  " + "-" * 60)
    print(f"  mean_roll_angle        : {roll_sum / denom:10.2f} deg   (mean over all steps)")
    print(f"  max_roll_angle         : {roll_max:10.2f} deg")
    print(f"  roll_RMS               : {math.sqrt(roll_sqsum / denom):10.2f} deg   (tightness of balance; lower = steadier)")
    if n_ep > 0:
        print(f"  mean_worst_lean/episode: {float(maxrolls.mean()):10.2f} deg")
    print("  " + "-" * 60)
    if drive_idx is not None:
        print(f"  drive_speed mean/max   : {drive_abs_sum / denom:8.2f} / {drive_max:7.2f} rad/s")
    if steer_idx is not None:
        print(f"  steer_speed mean/max   : {steer_abs_sum / denom:8.2f} / {steer_max:7.2f} rad/s")
    print(f"  drift mean/max         : {drift_sum / denom:8.3f} / {drift_max:7.3f} m   (position drift from spawn)")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
