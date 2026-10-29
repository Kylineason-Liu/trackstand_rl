"""Train the locomotive riding task with skrl PPO (CycleRL's trainer, trimmed).

Same agent flow as ~/bike_test/CycleRL/scripts/skrl/train.py (skrl Runner, CosineAnnealingLR
T_max synced to --max_iterations, final checkpoint write) without CycleRL's terrain presets,
post-eval hooks or CycleRL imports. Hydra overrides use the ``env.`` / ``agent.`` nodes.

From isaaclab_train/:
    export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
    ~/IsaacLab/isaaclab.sh -p scripts/skrl/train_locomotive.py --headless
    # other top-5 recipes, e.g. act_refine_lo:
    #   --num_envs 2048 env.rewards.act_mag.params.weights=[0.4,0.2] \
    #   env.rewards.act_rate.params.weights=[0.4,1.0] env.commands.velocity.ranges.lin_vel_x=[0.3,2.5]
Logs: logs/skrl/locomotive/<task>/<timestamp>_ppo_torch_locomotive_ppo/
"""

import argparse
import sys

import isaaclab  # noqa: F401  # Cache before Kit edits sys.path (CycleRL import fix).
from isaaclab.app import AppLauncher, add_launcher_args
from isaaclab_tasks.utils import setup_preset_cli

SKRL_VERSION = "2.1.0"

parser = argparse.ArgumentParser(description="Train the locomotive riding task with skrl PPO.")
parser.add_argument("--task", type=str, default="Locomotive-Cylinder-Aligned-v0")
parser.add_argument("--num_envs", type=int, default=None, help="Default: the env cfg (4096).")
parser.add_argument("--seed", type=int, default=42, help="-1 samples a random seed.")
parser.add_argument("--max_iterations", type=int, default=None, help="PPO updates; default: agent yaml.")
parser.add_argument("--checkpoint", type=str, default=None, help="Resume from a skrl checkpoint.")
add_launcher_args(parser)  # Also provides --deterministic, --device, --headless.
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import skrl  # noqa: E402
from packaging import version  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab.utils.seed import configure_seed  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

import source.trackstand_rl.trackstand_rl.tasks  # noqa: E402,F401

if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    raise SystemExit(f"skrl>={SKRL_VERSION} required, found {skrl.__version__}")


def main():
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "skrl_cfg_entry_point")
    with launch_simulation(env_cfg, args_cli):
        from skrl.utils.runner.torch import Runner

        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        if args_cli.max_iterations:
            agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
        # Keep CosineAnnealingLR in sync with the real number of PPO updates.
        updates = agent_cfg["trainer"]["timesteps"] // agent_cfg["agent"]["rollouts"]
        lr_kwargs = agent_cfg["agent"].get("learning_rate_scheduler_kwargs")
        if isinstance(lr_kwargs, dict):
            lr_kwargs["T_max"] = int(updates)
        agent_cfg["trainer"]["close_environment_at_exit"] = False
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg["seed"] = args_cli.seed
        env_cfg.seed = args_cli.seed

        project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        log_root = os.path.join(project_dir, "logs", "skrl", "locomotive", args_cli.task)
        run_name = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_ppo_torch_{agent_cfg['agent']['experiment']['experiment_name']}"
        log_dir = os.path.join(log_root, run_name)
        agent_cfg["agent"]["experiment"]["directory"] = log_root
        agent_cfg["agent"]["experiment"]["experiment_name"] = run_name
        os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        print(f"[INFO] Logging experiment in directory: {log_dir}")

        resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None
        env_cfg.log_dir = log_dir
        env = SkrlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), ml_framework="torch")

        if agent_cfg["agent"].get("learning_rate_scheduler") == "CosineAnnealingLR":
            from torch.optim.lr_scheduler import CosineAnnealingLR

            agent_cfg["agent"]["learning_rate_scheduler"] = CosineAnnealingLR
        runner = Runner(env, agent_cfg)
        if args_cli.deterministic:
            configure_seed(env_cfg.seed, True)
        if resume_path:
            print(f"[INFO] Loading model checkpoint from: {resume_path}")
            runner.agent.load(resume_path)

        start = time.time()
        total = agent_cfg["trainer"]["timesteps"]
        try:
            runner.run()
            runner.agent.write_checkpoint(timestep=total, timesteps=total)
            print(f"Training time: {round(time.time() - start, 2)} seconds")
        except KeyboardInterrupt:
            step = getattr(runner.agent, "timestep", 0)
            runner.agent.write_checkpoint(timestep=step, timesteps=total)
            print(f"[INFO] Interrupted; saved checkpoint at step {step}.")
        finally:
            env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
