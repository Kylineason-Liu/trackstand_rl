# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

import argparse
import contextlib
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
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
# append AppLauncher cli args
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

# Start Kit BEFORE importing torch / gym / isaaclab task modules
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg

# add argparse arguments

import CycleRL.tasks  # noqa: F401


def main():
    """Random actions agent with Isaac Lab environment."""

    torch.manual_seed(42)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    with contextlib.nullcontext():  # ponytail: keeps indentation; Kit is closed after main()
        from CycleRL.tasks.manager_based.cyclerl.config.terrain_presets import apply_terrain_preset

        apply_terrain_preset(env_cfg, args_cli.terrain)

        # override with CLI arguments
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if args_cli.disable_fabric:
            env_cfg.sim.use_fabric = False

        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg)

        # print info (this is vectorized environment)
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        # reset environment
        env.reset()
        # simulate environment
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                # sample actions from -1 to 1
                actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                # apply actions
                env.step(actions)

        # close the simulator
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    simulation_app.close()
