# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings

warnings.warn(
    "scripts/reinforcement_learning/skrl/play.py is deprecated. Use "
    "`./isaaclab.sh play --rl_library skrl --task <TASK>` instead.",
    DeprecationWarning,
    stacklevel=1,
)

import argparse
import sys
import os

# =========================================================================
# CRITICAL FIX: Match train.py structure
# Force cache isaaclab and start AppLauncher BEFORE heavy ML imports
# =========================================================================
import isaaclab
from isaaclab.app import AppLauncher

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default=None, help="Name of the RL agent configuration.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help=(
        "Path to a model checkpoint. If omitted, loads the most recently modified .pt under "
        "logs/skrl/cyclerl_logs/<task-name>."
    ),
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use pretrained checkpoint.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax"])
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"])
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time.")
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
parser.add_argument("--deterministic", action="store_true", default=False, help="Enable torch deterministic mode.")

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

if args_cli.video:
    args_cli.enable_cameras = True

# Start AppLauncher globally (This prevents the libusd_tf.so crash)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =========================================================================
# NOW IT IS SAFE TO IMPORT ML LIBRARIES AND CUSTOM TASKS
# =========================================================================
import contextlib
import logging
import random
import time
import torch
import skrl
import gymnasium as gym
from packaging import version

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.seed import configure_seed

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
import CycleRL.tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

logger = logging.getLogger(__name__)
SKRL_VERSION = "2.1.0"
_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CYCLERL_LOGS = os.path.join(_PROJECT_DIR, "logs", "skrl", "cyclerl_logs")


def _latest_checkpoint_for_task(task_name: str) -> str:
    """Return the newest ``.pt`` under ``logs/skrl/cyclerl_logs/<task-name>``."""
    log_root = os.path.join(_CYCLERL_LOGS, task_name)
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

# -- check skrl version ------------------------------------------------------
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(f"Unsupported skrl version: {skrl.__version__}. Install >={SKRL_VERSION}")
    exit()

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


def main():
    """Play with skrl agent."""
    # ponytail: plain registry load, no hydra composition (see train.py)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    experiment_cfg = load_cfg_from_registry(args_cli.task, agent_cfg_entry_point)
    
    # Use the existing simulation app context
    with contextlib.nullcontext():  # ponytail: keeps indentation; Kit is closed after main()
        if args_cli.ml_framework.startswith("torch"):
            from skrl.utils.runner.torch import Runner
        elif args_cli.ml_framework.startswith("jax"):
            from skrl.utils.runner.jax import Runner

        from isaaclab_rl.skrl import SkrlVecEnvWrapper

        from CycleRL.tasks.manager_based.cyclerl.config.terrain_presets import apply_terrain_preset

        apply_terrain_preset(env_cfg, args_cli.terrain)

        # Overrides
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)

        experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
        env_cfg.seed = experiment_cfg["seed"]

        # Checkpoint: --checkpoint, else newest .pt in logs/skrl/cyclerl_logs/<task>
        log_root_path = os.path.abspath(os.path.join(_CYCLERL_LOGS, args_cli.task))
        print(f"[INFO] Loading experiment from directory: {log_root_path}")

        if args_cli.checkpoint:
            resume_path = os.path.abspath(args_cli.checkpoint)
        else:
            resume_path = _latest_checkpoint_for_task(args_cli.task)
        
        log_dir = os.path.dirname(os.path.dirname(resume_path))
        env_cfg.log_dir = log_dir

        # Create environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg) and algorithm in ["ppo"]:
            from isaaclab.envs import multi_agent_to_single_agent
            env = multi_agent_to_single_agent(env)

        try:
            dt = env.step_dt
        except AttributeError:
            dt = env.unwrapped.step_dt

        env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

        # Runner setup
        experiment_cfg["trainer"]["close_environment_at_exit"] = False
        experiment_cfg["agent"]["experiment"]["write_interval"] = 0
        experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

        # =========================================================================
        # FIX: Resolve scheduler names before Runner initialization
        # This prevents the NameError: name 'CosineAnnealingLR' is not defined
        # =========================================================================
        scheduler = experiment_cfg.get("agent", {}).get("learning_rate_scheduler")
        if isinstance(scheduler, str):
            if scheduler == "CosineAnnealingLR":
                from torch.optim.lr_scheduler import CosineAnnealingLR
                experiment_cfg["agent"]["learning_rate_scheduler"] = CosineAnnealingLR
            else:
                logger.warning(
                    "Requested learning_rate_scheduler '%s' is not a recognized string and may fail during SKRL processing.",
                    scheduler,
                )

        # Now instantiate the runner
        runner = Runner(env, experiment_cfg)
        
        if args_cli.deterministic:
            configure_seed(env_cfg.seed, True)

        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)
        runner.agent.enable_training_mode(False, apply_to_models=True)

        obs, _ = env.reset()
        states = env.state()
        
        try:
            while True:
                start_time = time.time()
                with torch.inference_mode():
                    outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
                    if hasattr(env, "possible_agents"):
                        actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
                    else:
                        actions = outputs[-1].get("mean_actions", outputs[0])
                    obs, _, _, _, _ = env.step(actions)
                    states = env.state()
                
                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            pass
        finally:
            env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
