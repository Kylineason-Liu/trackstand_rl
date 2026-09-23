# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an RSL-RL checkpoint and print summary metrics (headless-friendly).

Runs the trained policy for a fixed number of steps across parallel envs and
summarises, over completed episodes:

    mean_reward, mean/min/max/std episode_length, mean/max roll angle

plus high-value extras: fall/timeout rate, roll RMS, per-episode worst lean,
drive/steer motor speeds, and position drift from spawn.

Loading mirrors scripts/rsl_rl/play.py; the metric intent mirrors
scripts/skrl/play_with_metrics.py (but focused, and rsl_rl-native).

Example:
    ~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play_eval.py --task NewBike-Balance-v0 \\
        --checkpoint logs/rsl_rl/newbike_balance_rsl/<run>/model_14999.pt \\
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
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate an RSL-RL checkpoint with summary metrics.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the published pretrained checkpoint.")
parser.add_argument(
    "--num_steps", type=int, default=3000,
    help="Env steps to run (~2 episodes/env at the 1500-step / 30s cap).",
)
parser.add_argument(
    "--no_dr", action="store_true", default=False,
    help="Disable domain randomization entirely (dr_ramp alpha pinned to 0). "
    "Without this flag, eval runs at FULL DR from step 0 (no ramp).",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata
import math
import os

import random

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")  # headless: render to file only, never to a display
import matplotlib.pyplot as plt
import numpy as np
import torch
import isaaclab.utils.math as math_utils
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_checkpoint
except ImportError:
    # Isaac Lab 7.2.0 ships no checkpoint migrator (only the cfg one above).
    # Checkpoints written by the installed rsl-rl are already in its current
    # format, so passing the path through unchanged is correct here.
    def handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version):
        return resume_path
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import source.trackstand_rl.trackstand_rl.tasks  # noqa: F401  # registers Trackstand-Balance-v0 (CycleRL removed 2026-09-23)

installed_version = metadata.version("rsl-rl-lib")


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
    n = len(envs)
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


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    """Evaluate an RSL-RL agent and print a metric summary."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

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

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # resolve the checkpoint path (same precedence as play.py)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] No pre-trained checkpoint available for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    # create env + wrap for rsl_rl
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading checkpoint: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)  # deterministic (mean) action

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    step_dt = env.unwrapped.step_dt
    robot = env.unwrapped.scene["robot"]
    origins_xy = env.unwrapped.scene.env_origins[:, :2]

    # optional motor joints (skip those metrics if the task lacks them)
    def _joint_idx(name):
        try:
            return robot.find_joints(name)[0][0]
        except Exception:
            return None

    # joint names from the live env cfg: the drive joint is rear_wheel_joint
    # (underscore) on Trackstand-Balance-v0 but rearwheel_joint on the NewBike
    # tasks — hardcoding only worked for NewBike.
    try:
        _steer_name = env_cfg.actions.steering.joint_names[0]
        _drive_name = env_cfg.actions.drive.joint_names[0]
    except Exception:
        _steer_name, _drive_name = "steering_joint", "rearwheel_joint"
    steer_idx = _joint_idx(_steer_name)
    drive_idx = _joint_idx(_drive_name)

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
    ep_returns, ep_lengths, ep_timeouts, ep_maxrolls, ep_turns = [], [], [], [], []
    # global step-wise stats
    roll_sum = roll_sqsum = roll_max = 0.0
    drive_abs_sum = drive_max = steer_abs_sum = steer_max = drift_sum = drift_max = 0.0
    yaw_abs_sum = yaw_max = 0.0
    # per-episode accumulated |yaw change| — orbits a circle, 2*pi rad per lap
    ep_turn = torch.zeros(num_envs, device=device)
    prev_yaw = None
    # per-env spawn-heading baseline: the bike's USD root spawns facing ~+-pi in
    # WORLD yaw, so heading metrics must be relative to the post-reset yaw
    base_yaw = None
    # termination-cause tracking: which term(s) fired for each completed episode.
    # _last_episode_dones (num_envs, num_terms) records exactly the terms that
    # fired when an episode ended, and is NOT cleared by the reset (the reset
    # only reads it to build the extras the runner logs to TensorBoard).
    try:
        _term_mgr = env.unwrapped.termination_manager
        term_names = [name for name, _ in _term_mgr.get_active_iterable_terms(0)]
        ep_term_counts = torch.zeros(len(term_names), device=device)
    except Exception:
        term_names, ep_term_counts = [], None
    n_steps_counted = 0

    obs = env.get_observations()
    for t in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)
            if hasattr(policy, "reset"):
                policy.reset(dones)

        rewards = rewards.to(device).view(-1)
        dones_b = dones.to(device).view(-1).bool()
        time_outs = extras.get("time_outs") if isinstance(extras, dict) else None
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

        # yaw (heading) vs spawn: mean/max |yaw| RELATIVE to each env's post-reset
        # heading. Uses the canonical heading_w (base-frame +x rotated to world)
        # — root_quat_w is stored (x, y, z, w) in this stack, and hand-rolled
        # wxyz math on it silently returned pi for the identity spawn quaternion.
        # Plus per-episode accumulated turn (detects orbiting policies that
        # radial drift metrics are blind to).
        yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w.torch)[2]  # raw quat -> yaw (rad)
        if base_yaw is None:
            base_yaw = yaw.clone()
        rel = torch.atan2(torch.sin(yaw - base_yaw), torch.cos(yaw - base_yaw))
        yaw_abs_sum += float(rel.abs().sum())
        yaw_max = max(yaw_max, float(rel.abs().max()))
        if prev_yaw is not None:
            # wrap-aware angular difference, then accumulate per-episode turn.
            # Envs that terminated this step are already auto-reset, so their
            # `yaw` is the fresh spawn heading — the difference to `prev_yaw`
            # is a reset teleport, not physical turning; exclude it.
            d = torch.atan2(torch.sin(yaw - prev_yaw), torch.cos(yaw - prev_yaw))
            ep_turn += torch.where(dones_b, torch.zeros_like(d), d.abs())
        prev_yaw = yaw.clone()

        if dones_b.any():
            ep_returns.append(cur_return[dones_b].clone())
            ep_lengths.append(cur_len[dones_b].clone())
            ep_timeouts.append(time_outs[dones_b].clone())
            ep_maxrolls.append(ep_max_roll[dones_b].clone())
            ep_turns.append(ep_turn[dones_b].clone())
            if ep_term_counts is not None:
                # per-term firing record for the episodes that just ended
                ep_term_counts += _term_mgr._last_episode_dones[dones_b].float().sum(dim=0)
            cur_return[dones_b] = 0.0
            cur_len[dones_b] = 0
            ep_max_roll[dones_b] = 0.0
            ep_turn[dones_b] = 0.0
            prev_yaw[dones_b] = yaw[dones_b]  # reset the diff baseline on episode reset
            base_yaw[dones_b] = yaw[dones_b]  # new episode's spawn-heading baseline

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

    # ---- build the summary block (printed AND saved to eval.txt next to the checkpoint) ----
    lines: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)
        print(text)

    emit("=" * 64)
    emit(f"  EVAL SUMMARY  —  {os.path.basename(resume_path)}")
    emit(f"  {num_envs} envs x {args_cli.num_steps} steps  |  {n_ep} completed episodes")
    emit("=" * 64)
    if n_ep == 0:
        emit("  No episode completed within --num_steps. Increase it, or the")
        emit(f"  policy is timing out. In-progress mean length: {float(cur_len.float().mean()):.1f} steps.")
    else:
        surv = float(timeouts.mean())
        emit(f"  mean_reward            : {float(returns.mean()):10.3f}   (episodic return; not comparable across reward-weight changes)")
        emit(f"  mean_episode_length    : {float(lengths.mean()):10.1f} steps  ({float(lengths.mean()) * step_dt:6.2f} s)")
        emit(f"  min_episode_length     : {float(lengths.min()):10.1f} steps  ({float(lengths.min()) * step_dt:6.2f} s)")
        emit(f"  max_episode_length     : {float(lengths.max()):10.1f} steps  ({float(lengths.max()) * step_dt:6.2f} s)")
        emit(f"  std_episode_length     : {float(lengths.std(unbiased=False)):10.1f} steps")
        emit(f"  timeout_rate (survived): {surv:10.3f}   (reached the horizon vs fell)")
        emit(f"  fall_rate              : {1.0 - surv:10.3f}")
        emit("  " + "-" * 60)
        if ep_term_counts is not None:
            emit("  death breakdown (share of completed episodes; a step may fire >1 term):")
            for i, name in enumerate(term_names):
                pct = 100.0 * float(ep_term_counts[i].item()) / n_ep
                emit(f"    {name:<22}: {pct:6.1f}%")
            emit("  " + "-" * 60)
    emit(f"  mean_roll_angle        : {roll_sum / denom:10.2f} deg   (mean over all steps)")
    emit(f"  max_roll_angle         : {roll_max:10.2f} deg")
    emit(f"  roll_RMS               : {math.sqrt(roll_sqsum / denom):10.2f} deg   (tightness of balance; lower = steadier)")
    if n_ep > 0:
        emit(f"  mean_worst_lean/episode: {float(maxrolls.mean()):10.2f} deg")
    emit("  " + "-" * 60)
    if drive_idx is not None:
        emit(f"  drive_speed mean/max   : {drive_abs_sum / denom:8.2f} / {drive_max:7.2f} rad/s")
    if steer_idx is not None:
        emit(f"  steer_speed mean/max   : {steer_abs_sum / denom:8.2f} / {steer_max:7.2f} rad/s")
    emit(f"  drift mean/max         : {drift_sum / denom:8.3f} / {drift_max:7.3f} m   (position drift from spawn)")
    emit(f"  yaw mean/max           : {math.degrees(yaw_abs_sum / denom):8.2f} / {math.degrees(yaw_max):7.2f} deg (heading vs spawn)")
    if n_ep > 0:
        turns = _cat(ep_turns).float()
        # mean accumulated |yaw change| per completed episode; 360 deg = one full circle
        emit(f"  turn/episode mean      : {float(turns.mean()) * 180.0 / math.pi:8.1f} deg (orbiting detection: 360 deg = full circle)")
    emit("=" * 64)

    # save the summary into the checkpoint's run folder (overwrites a previous eval.txt)
    out_dir = os.path.dirname(resume_path)
    try:
        with open(os.path.join(out_dir, "eval.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[INFO] summary saved to {os.path.join(out_dir, 'eval.txt')}\n")
    except OSError as e:
        print(f"[WARN] could not write eval.txt: {e}\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
