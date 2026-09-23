# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
"""Print each joint's default and post-reset angle (rad + deg) for a task.

Use it to verify spawn joint states — e.g. whether the steering/front wheel is
turned at reset. Runs a zero-action loop afterwards so you can visually compare
the printed angle with the rendered pose.

    python scripts/inspect_joints.py --task NewBike-Balance-v0 --num_envs 1
"""

import argparse
import contextlib
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Print joint angles at reset.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.task is None:
    parser.error("--task is required")
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_tasks  # noqa: F401
import CycleRL.tasks  # noqa: F401


def _print_joints(robot):
    names = robot.joint_names
    default = robot.data.default_joint_pos[0]
    actual = robot.data.joint_pos[0]
    print("\n=== joint angles (env 0) ===")
    print(f"{'joint':22s} {'default rad':>12s} {'default deg':>12s} {'actual rad':>12s} {'actual deg':>12s}")
    for i, n in enumerate(names):
        print(
            f"{n:22s} {default[i].item():12.4f} {math.degrees(default[i].item()):12.1f}"
            f" {actual[i].item():12.4f} {math.degrees(actual[i].item()):12.1f}"
        )
    print("============================\n")


def main():
    torch.manual_seed(42)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    _print_joints(robot)

    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)  # type: ignore
    with contextlib.suppress(KeyboardInterrupt):
        while simulation_app.is_running():
            with torch.inference_mode():
                env.step(actions)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
