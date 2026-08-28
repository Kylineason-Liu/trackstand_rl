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

import gymnasium as gym
import torch
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

import CycleRL.tasks  # noqa: F401

installed_version = metadata.version("rsl-rl-lib")


def _roll_deg(robot, device):
    """Lean angle in degrees from projected gravity (|sin(roll)| -> deg), like the reward code."""
    lean = robot.data.projected_gravity_b[:, 1].abs().clamp(max=1.0)
    return torch.rad2deg(torch.asin(lean)).to(device)


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

    steer_idx = _joint_idx("steering_joint")
    drive_idx = _joint_idx("rearwheel_joint")

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

    obs = env.get_observations()
    for _ in range(args_cli.num_steps):
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
