"""Free-fall test: spawn the perfectly upright NewBike-Balance-v1 and measure
the time to fall (the zero actions; the fallen termination |gy|>sin30 or
gz>0; the roll pinned to 0 for the upright spawn)."""
import argparse
import sys

sys.path.insert(0, "/home/lmbike/lmbike_cyclerl-mod/scripts/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
cli_args.add_rsl_rl_args(parser)
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks  # noqa
import CycleRL.tasks  # noqa
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config("NewBike-Balance-v1", "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    # pin the events to the 0950 set (the reset_scene + the ground friction) and
    # the initial_state's roll to 0 (the perfectly upright spawn)
    for name in list(env_cfg.events.__dict__.keys()):
        if name.startswith("_") or name in ("reset_scene", "randomize_ground_friction", "initial_state"):
            continue
        delattr(env_cfg.events, name)
    env_cfg.events.initial_state.params["pose_range"]["roll"] = (0.0, 0.0)
    env_cfg.scene.num_envs = 1
    env = gym.make("NewBike-Balance-v1", cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    obs, _ = env.reset()

    action = torch.zeros(1, 2, device=env.unwrapped.device)
    print(f"{'step':>4} {'roll':>8} {'z':>8} {'term':>5}")
    fall_step = None
    for i in range(400):
        obs, _, terminated, _, _ = env.step(action)
        q = robot.data.root_quat_w.torch
        # roll from the xyzw quat (the scalar-last convention): roll = atan2(2(wx+yz), 1-2(x^2+y^2))
        x, y, z, w = q[0, 0].item(), q[0, 1].item(), q[0, 2].item(), q[0, 3].item()
        roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        bz = robot.data.root_pos_w[0, 2].item()
        t = bool(terminated[0].item()) if hasattr(terminated, "__len__") else bool(terminated)
        if i % 5 == 4 or i < 5 or t:
            print(f"{i + 1:>4} {roll:>+8.2f} {bz:>+8.3f} {str(t):>5}")
        if t and fall_step is None:
            fall_step = i + 1
    print(f"\nFALL TIME (sim): {fall_step * 0.02:.2f} s  ({fall_step} steps)")
    env.close()


main()
