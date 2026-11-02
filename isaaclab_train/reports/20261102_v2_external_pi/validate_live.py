"""Headless PI contract, reset and unloaded timestep probe; no trained policy required."""

import argparse
import importlib
import json
from pathlib import Path
import sys
import traceback

TRAIN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRAIN))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--physics-hz", type=int, default=400)
parser.add_argument("--dr", action="store_true")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import warp as wp

importlib.import_module("source.trackstand_rl.trackstand_rl.tasks.manager_based.trackstand")


def tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    return value.torch if hasattr(value, "torch") else wp.to_torch(value)


env = None
try:
    if args.physics_hz % 200:
        raise ValueError("physics-hz must be a multiple of the 200-Hz integrator")
    cfg = gym.spec("Trackstand-Balance-v2").kwargs["env_cfg_entry_point"](domain_randomization=args.dr)
    cfg.scene.num_envs, cfg.seed, cfg.sim.device = 64, 42, args.device
    cfg.sim.dt, cfg.decimation = 1 / args.physics_hz, args.physics_hz // 50
    cfg.sim.render_interval = cfg.decimation
    env = gym.make("Trackstand-Balance-v2", cfg=cfg).unwrapped
    env.reset(seed=42)
    robot = env.scene["robot"]
    drive = robot.actuators["drive"]
    action = env.action_manager.get_term("drive")
    joint = robot.find_joints("rear_wheel_joint")[0][0]
    ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int32)

    # Airborne only for the unloaded response probe; no model/gain/friction edits.
    pose = torch.cat((tensor(robot.data.root_pos_w), tensor(robot.data.root_quat_w)), -1).clone()
    pose[:, 2], pose[:, 3:6], pose[:, 6] = 3.0, 0.0, 1.0
    robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((64, 6), device=env.device), env_ids=ids)
    zero = torch.zeros_like(tensor(robot.data.joint_pos))
    robot.write_joint_state_to_sim_index(position=zero, velocity=zero, env_ids=ids)
    targets = torch.zeros((64, 2), device=env.device)
    targets[:, 1] = 1 / 61.6
    records = []
    for index in range(int(args.physics_hz * .4)):
        if index % cfg.decimation == 0:
            env.action_manager.process_action(targets)
        speed = tensor(robot.data.joint_vel)[:, joint].clone()
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        integral = action.integral.torque_nm[:, 0].clone()
        expected = drive.damping[:, 0] * (1 - speed) + integral
        torch.testing.assert_close(drive.computed_effort[:, 0], expected, atol=2e-6, rtol=1e-6)
        torch.testing.assert_close(drive.applied_effort[:, 0], expected.clamp(-6.28, 6.28))
        records.append(torch.stack((speed, integral, drive.applied_effort[:, 0]), -1).cpu().numpy())
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
    capture = np.stack(records)
    assert tensor(robot.data.root_pos_w)[:, 2].min() > 2.0
    late = capture[len(capture) // 2 :, :, 0]
    summary = {
        "task": "Trackstand-Balance-v2", "seed": 42, "domain_randomization": args.dr,
        "num_envs": 64, "physics_hz": args.physics_hz, "policy_hz": 50, "integral_hz": 200,
        "velocity_p_nm_s_per_rad": .3, "velocity_i_nm_per_rad": 3.3,
        "unloaded_target_rad_s": 1.0, "unloaded_duration_s": .4,
        "unloaded_late_speed_peak_to_peak_max_rad_s": float(np.ptp(late, axis=0).max()),
        "unloaded_late_speed_peak_to_peak_median_rad_s": float(np.median(np.ptp(late, axis=0))),
        "unloaded_late_mean_abs_speed_error_rad_s": float(np.abs(late - 1).mean()),
        "unloaded_max_applied_torque_nm": float(np.abs(capture[:, :, 2]).max()),
        "unloaded_max_integral_torque_nm": float(np.abs(capture[:, :, 1]).max()),
    }
    np.savez_compressed(args.output / "unloaded.npz", speed_integral_applied=capture)

    # Ordinary ManagerBasedRLEnv.step exercises action/reset ordering and DR.
    env.reset(seed=42)
    generator = torch.Generator(device=env.device).manual_seed(42)
    resets = 0
    for _ in range(100):
        targets = 2 * torch.rand((64, 2), device=env.device, generator=generator) - 1
        obs, reward, terminated, truncated, _ = env.step(targets)
        done = terminated | truncated
        resets += int(done.sum())
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
        assert obs["policy"].shape == (64, 11)
        torch.testing.assert_close(action.integral.torque_nm[done], torch.zeros_like(action.integral.torque_nm[done]))
        assert float(drive.applied_effort.abs().max()) <= 6.28001

    # Give both halves known memory, reset only one half, verify backend buffers.
    action.integral.torque_nm.fill_(.1)
    action.integral.error_rad.fill_(.1 / 3.3)
    chosen = torch.arange(0, 64, 2, device=env.device, dtype=torch.int32)
    env.reset(env_ids=chosen)
    torch.testing.assert_close(action.integral.torque_nm[chosen], torch.zeros((32, 1), device=env.device))
    torch.testing.assert_close(action.integral.torque_nm[1::2], torch.full((32, 1), .1, device=env.device))
    torch.testing.assert_close(
        tensor(robot.data.joint_effort_target)[chosen, joint], torch.zeros(32, device=env.device)
    )
    torch.testing.assert_close(tensor(robot.data.joint_vel_target)[chosen, joint], torch.zeros(32, device=env.device))
    env.reset()
    # Force a timeout and ensure it follows the same reset path as a failure.
    env.episode_length_buf[:] = env.max_episode_length - 1
    _, _, _, truncated, _ = env.step(torch.zeros_like(targets))
    assert truncated.all()
    torch.testing.assert_close(action.integral.torque_nm, torch.zeros_like(action.integral.torque_nm))
    summary.update(
        contract_passed=True, grounded_random_steps=100, grounded_automatic_resets=resets,
        partial_reset_passed=True, timeout_reset_passed=True, observed_channels=11,
        note="Random-action/constant-target contract evaluation, not learned balance or hardware validation.",
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("PASS " + json.dumps(summary), flush=True)
except BaseException:
    (args.output / "failure.json").write_text(json.dumps({"traceback": traceback.format_exc()}, indent=2) + "\n")
    traceback.print_exc()
    raise
finally:
    if env is not None:
        env.close()
    app.close()
