# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic variant of train.py for skrl.

Same CLI surface as scripts/skrl/train.py, plus the sweep flags mirrored from
scripts/rsl_rl/train_deterministic.py (--no_dr, --survival_w/--drive_w/
--steer_w/--arp_w, --run_name). Unlike train.py, determinism is ALWAYS on:
TF32 off, cuDNN deterministic, torch.use_deterministic_algorithms(True) and
configure_seed(seed, torch_deterministic=True), so with a fixed --seed two runs
of the same config give (near-)identical results and a small change between
runs — e.g. --drive_w 12 vs 13 — is attributable to the change, not to CUDA
non-determinism. Trade-off: a modest slowdown from deterministic kernels.

Note (resume semantics): --checkpoint fine-tunes from a saved agent (skrl
checkpoints hold model weights, not the trainer state), so the timestep counter
and the dr_ramp curriculum restart from 0 — this is the warm-start pattern, not
an exact continuation.
"""

import warnings

warnings.warn(
    "scripts/reinforcement_learning/skrl/train_deterministic.py is deprecated. Use "
    "`./isaaclab.sh train --rl_library skrl --task <TASK>` instead. ",
    DeprecationWarning,
    stacklevel=1,
)

import argparse
import os
import sys

# =========================================================================
# CRITICAL FIX: Force cache the top-level isaaclab module in memory.
# This prevents Omniverse Kit's sys.path manipulation during AppLauncher
# startup from breaking internal IsaacLab imports (unknown location error).
# =========================================================================
import isaaclab
from isaaclab.app import AppLauncher

# Required for deterministic cuBLAS (must be set before torch is imported below).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SKRL_VERSION = "2.1.0"

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train an RL agent with skrl (deterministic).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
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
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to fine-tune from.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# -- sweep flags (mirror scripts/rsl_rl/train_deterministic.py) ---------------
parser.add_argument(
    "--no_dr", action="store_true", default=False,
    help="Disable domain randomization by pinning the dr_ramp curriculum to alpha=0.",
)
parser.add_argument("--survival_w", type=float, default=None,
    help="Override the survival reward weight (kept at the env-cfg value if omitted). 0 disables the term.")
parser.add_argument("--drive_w", type=float, default=None,
    help="Override the drive_reward weight (kept at the env-cfg value if omitted). 0 disables the term.")
parser.add_argument("--steer_w", type=float, default=None,
    help="Override the steer_reward weight (kept at the env-cfg value if omitted). 0 disables the term.")
parser.add_argument("--arp_w", type=float, default=None,
    help="Override the action_rate penalty weight (kept at the env-cfg value if omitted; cfg default 0=off). Small, e.g. 0.1.")
parser.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")

# =========================================================================
# NEW ARGUMENTS FOR TERMINAL METRICS TRACKING
# =========================================================================
parser.add_argument("--print_metrics", action="store_true", default=False, help="Print detailed metrics to the terminal during training.")
parser.add_argument("--disable_progressbar", action="store_true", default=False, help="Disable standard tqdm progress bar (useful when tracking metrics in console).")
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
# =========================================================================

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

# Start AppLauncher globally BEFORE importing heavy ML libraries
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =========================================================================
# NOW IT IS SAFE TO IMPORT ML LIBRARIES (PyTorch, SKRL, Gym)
# =========================================================================
import contextlib
import logging
import random
import time
from datetime import datetime
from packaging import version

import gymnasium as gym
import skrl
import torch

# --- DETERMINISM (the ONLY substantive difference from train.py) ---
# train.py is non-deterministic (fast); this script forces reproducibility.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
print("[INFO] DETERMINISTic training: TF32 off, cuDNN deterministic, deterministic algorithms on.")

from isaaclab.envs import DirectMARLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.seed import configure_seed

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import CycleRL.tasks  # noqa: F401

# Dynamically pull in BIKE_USD_PATH
try:
    from CycleRL.tasks.manager_based.cyclerl.bicycle_cfg import BIKE_USD_PATH
except ImportError:
    BIKE_USD_PATH = "NOT FOUND (Could not import from CycleRL.tasks.manager_based.cyclerl.bicycle_cfg)"

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

logger = logging.getLogger(__name__)

# -- check skrl version ------------------------------------------------------
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


# =========================================================================
# TERMINAL METRICS TRACKING WRAPPER
# =========================================================================
class TerminalConsoleWrapper(gym.Wrapper):
    """A wrapper that intercepts IsaacLab log dictionaries and prints tracked values."""
    def __init__(self, env, print_interval=1000):
        super().__init__(env)
        self.print_interval = print_interval
        self._step_count = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1

        if self._step_count % self.print_interval == 0:
            print(f"\n[INFO] === Step {self._step_count} Environment Metrics ===")
            if "log" in info and "episode" in info["log"]:
                for k, v in info["log"]["episode"].items():
                    try:
                        # Extract single float values out of multi-env tensors
                        val = v.float().mean().item() if hasattr(v, "float") else v
                        print(f"  {k}: {val:.4f}")
                    except Exception:
                        print(f"  {k}: {v}")
            print("===============================================")

        return obs, reward, terminated, truncated, info

    @property
    def num_envs(self):
        return self.env.unwrapped.num_envs

    @property
    def device(self):
        return getattr(self.env.unwrapped, "device", "cuda:0")
# =========================================================================


def main():
    """Train with skrl agent (deterministic)."""
    # ponytail: plain registry load, no hydra composition — `key=value` CLI overrides
    # are not supported here. Use hydra_task_config if you ever need them.
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, agent_cfg_entry_point)
    from CycleRL.tasks.manager_based.cyclerl.config.terrain_presets import apply_terrain_preset

    apply_terrain_preset(env_cfg, args_cli.terrain)

    # --no_dr: pin the DR ramp to alpha=0 (all randomized terms stay nominal).
    # Same mechanism as editing dr_ramp's end_step, but per-run. No-op if the
    # task has no dr_ramp curriculum term.
    if args_cli.no_dr:
        dr_ramp = getattr(getattr(env_cfg, "curriculum", None), "dr_ramp", None)
        if dr_ramp is not None:
            dr_ramp.params["end_step"] = 10**9
            print("[INFO] --no_dr: domain randomization disabled (dr_ramp alpha pinned to 0).")
        else:
            print("[WARN] --no_dr: task has no 'dr_ramp' curriculum term; nothing to disable.")

    # --survival_w / --drive_w / --steer_w / --arp_w: per-run overrides of reward-term
    # weights. A weight of 0 disables the term entirely (set to None so its func isn't
    # called); not passing a flag keeps the env-cfg weight. No-op if the term is absent.
    def _set_reward_weight(term_name, w):
        rewards = getattr(env_cfg, "rewards", None)
        term = getattr(rewards, term_name, None) if rewards is not None else None
        if term is None:
            print(f"[WARN] reward term '{term_name}' not found (or already off); skipping.")
            return
        if w == 0.0:
            setattr(rewards, term_name, None)
            print(f"[INFO] reward term '{term_name}' disabled (weight 0).")
        else:
            term.weight = w
            print(f"[INFO] reward term '{term_name}' weight -> {w}.")

    for w, term_name in (
        (args_cli.survival_w, "survival"),
        (args_cli.drive_w, "drive_reward"),
        (args_cli.steer_w, "steer_reward"),
        (args_cli.arp_w, "action_rate"),
    ):
        if w is not None:
            _set_reward_weight(term_name, w)

    # =========================================================================
    # PRINT SPECIFIED METADATA & CONFIGURATION
    # =========================================================================
    print("\n" + "="*80)
    print(f"[INFO] BIKE_USD_PATH: {BIKE_USD_PATH}")
    print("="*80)
    print(f"[INFO] FULL ENVIRONMENT CONFIGURATION ({args_cli.task}):")
    print("="*80)
    if hasattr(env_cfg, "to_dict"):
        print_dict(env_cfg.to_dict())
    else:
        print(env_cfg)
    print("="*80 + "\n")
    # =========================================================================

    with contextlib.nullcontext():  # ponytail: keeps indentation; Kit is closed after main()
        if args_cli.ml_framework.startswith("torch"):
            from skrl.utils.runner.torch import Runner
        elif args_cli.ml_framework.startswith("jax"):
            from skrl.utils.runner.jax import Runner

        # override configurations with non-hydra CLI arguments
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

        if not args_cli.distributed:
            env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
            raise ValueError(
                "Distributed training is not supported when using CPU device. "
                "Please use GPU device (e.g., --device cuda) for distributed training."
            )

        # multi-gpu training config
        if args_cli.distributed:
            global_rank = int(os.getenv("RANK", "0"))

        # max iterations for training
        if args_cli.max_iterations:
            agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
        agent_cfg["trainer"]["close_environment_at_exit"] = False

        # Disable progress bar to keep terminal clean if requested
        if args_cli.disable_progressbar:
            agent_cfg["trainer"]["disable_progressbar"] = True

        # configure the ML framework into the global skrl variable
        if args_cli.ml_framework.startswith("jax"):
            skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)

        # set the agent and environment seed from command line
        agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
        if args_cli.distributed:
            agent_cfg["seed"] = agent_cfg["seed"] + global_rank
        env_cfg.seed = agent_cfg["seed"]
        print(f"[INFO] deterministic run seed = {env_cfg.seed} (pass --seed to vary).")

        # log dir: logs/skrl/cyclerl_logs/<task-name>
        project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

        # Build the new path
        log_root_path = os.path.join(project_dir, "logs", "skrl", "cyclerl_logs", args_cli.task)
        log_root_path = os.path.abspath(log_root_path)

        print(f"[INFO] Logging experiment in directory: {log_root_path}")

        # Create a unique directory name based on timestamp and algorithm
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{timestamp}_{algorithm}_{args_cli.ml_framework}"
        if agent_cfg["agent"]["experiment"]["experiment_name"]:
            run_name += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
        if args_cli.run_name:
            run_name += f"_{args_cli.run_name}"

        log_dir = os.path.join(log_root_path, run_name)

        # Update agent config with new paths
        agent_cfg["agent"]["experiment"]["directory"] = log_root_path
        agent_cfg["agent"]["experiment"]["experiment_name"] = run_name

        # dump the configuration into log-directory
        # Ensure the directory exists before dumping
        os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

        # =================================================================
        # NEW: Write full config and metadata to env_config.md in log folder
        # =================================================================
        config_md_path = os.path.join(log_dir, "env_config.md")
        with open(config_md_path, "w") as f:
            f.write(f"# Experiment Configuration\n\n")
            f.write(f"- **Task**: {args_cli.task}\n")
            f.write(f"- **Bike USD Path**: `{BIKE_USD_PATH}`\n")
            f.write(f"- **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Environment Configuration Details\n\n```text\n")
            # Capture the dict string representation
            if hasattr(env_cfg, "to_dict"):
                f.write(str(env_cfg.to_dict()))
            else:
                f.write(str(env_cfg))
            f.write(f"\n```\n")
        print(f"[INFO] Full configuration saved to: {config_md_path}")
        # =================================================================

        # get checkpoint path (to fine-tune from)
        resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

        # set the IO descriptors export flag if requested
        if isinstance(env_cfg, ManagerBasedRLEnvCfg):
            env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        else:
            logger.warning(
                "IO descriptors are only supported for manager based RL environments."
                " No IO descriptors will be exported."
            )

        # set the log directory for the environment
        env_cfg.log_dir = log_dir

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # =================================================================
        # Wrap environment to intercept and print tracked values
        # =================================================================
        if args_cli.print_metrics:
            # Sync the print interval to skrl's write interval, fallback to 1000
            print_interval = agent_cfg["agent"]["experiment"].get("write_interval", 1000)
            env = TerminalConsoleWrapper(env, print_interval=print_interval)
        # =================================================================

        start_time = time.time()

        # wrap around environment for skrl
        env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

        # resolve any scheduler names that require actual class objects
        scheduler = agent_cfg.get("agent", {}).get("learning_rate_scheduler")
        if isinstance(scheduler, str):
            if scheduler == "CosineAnnealingLR":
                from torch.optim.lr_scheduler import CosineAnnealingLR
                agent_cfg["agent"]["learning_rate_scheduler"] = CosineAnnealingLR
            else:
                logger.warning(
                    "Requested learning_rate_scheduler '%s' is not a recognized string and may fail during SKRL config processing.",
                    scheduler,
                )

        # configure and instantiate the skrl runner
        runner = Runner(env, agent_cfg)

        # deterministic seeding of torch/numpy/random (skrl also seeds from agent_cfg["seed"])
        configure_seed(env_cfg.seed, True)

        # load checkpoint (if specified) — fine-tune, not exact continuation:
        # skrl checkpoints hold model weights, so the trainer timestep and the
        # dr_ramp curriculum both restart from 0.
        if resume_path:
            print(f"[INFO] Loading model checkpoint from: {resume_path}")
            runner.agent.load(resume_path)

        # run training
        try:
            runner.run()
            print(f"Training time: {round(time.time() - start_time, 2)} seconds")

            total_timesteps = agent_cfg["trainer"]["timesteps"]
            os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
            runner.agent.write_checkpoint(timestep=total_timesteps, timesteps=total_timesteps)
            print(f"[INFO] Saved final agent checkpoint to: {log_dir}/checkpoints")
            env.close()
        except KeyboardInterrupt:
            print("\n[INFO] KeyboardInterrupt (Ctrl+C) detected! Saving current progress...")
            try:
                # Grab the current timestep from the agent to label the checkpoint properly
                current_step = getattr(runner.agent, "timestep", 0)
                total_timesteps = agent_cfg["trainer"]["timesteps"]

                os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
                runner.agent.write_checkpoint(timestep=current_step, timesteps=total_timesteps)

                print(f"[INFO] Successfully saved interrupted checkpoint at step {current_step}.")
            except Exception as e:
                print(f"[ERROR] Failed to save checkpoint during interrupt: {e}")
            finally:
                # Ensure the Isaac Lab environment closes cleanly so it doesn't hang your terminal
                print("[INFO] Closing environment...")
                env.close()
                sys.exit(0)

if __name__ == "__main__":
    main()
    simulation_app.close()
