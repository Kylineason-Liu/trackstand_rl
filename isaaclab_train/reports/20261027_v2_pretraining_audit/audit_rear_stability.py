"""Unloaded constant-target v2 rear-loop probe; runtime fixture, no model edits."""

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
parser.add_argument("--physics_hz", type=int, choices=(400, 800, 1000), default=400)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import warp as wp

PACKAGE = "source.trackstand_rl.trackstand_rl.tasks.manager_based.trackstand"
importlib.import_module(PACKAGE)


def tensor(value):
    return value if isinstance(value, torch.Tensor) else value.torch if hasattr(value, "torch") else wp.to_torch(value)


env = None
try:
    cfg = gym.spec("Trackstand-Balance-v2").kwargs["env_cfg_entry_point"](domain_randomization=True)
    cfg.scene.num_envs, cfg.seed, cfg.sim.device = 64, 42, args.device
    cfg.sim.dt, cfg.decimation = 1. / args.physics_hz, args.physics_hz // 50
    assert cfg.sim.dt * cfg.decimation == .02
    cfg.sim.render_interval = cfg.decimation
    env = gym.make("Trackstand-Balance-v2", cfg=cfg).unwrapped
    env.reset(seed=42)
    robot = env.scene["robot"]
    rear = robot.actuators["drive"]
    rj = robot.find_joints("rear_wheel_joint")[0][0]
    n, device = 64, env.device
    ids = torch.arange(n, device=device, dtype=torch.int32)
    pose = torch.cat((tensor(robot.data.root_pos_w), tensor(robot.data.root_quat_w)), -1).clone()
    pose[:, 2], pose[:, 3:6], pose[:, 6] = 3., 0., 1.
    robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((n, 6), device=device), env_ids=ids)
    zero_joints = torch.zeros_like(tensor(robot.data.joint_pos))
    robot.write_joint_state_to_sim_index(position=zero_joints, velocity=zero_joints, env_ids=ids)
    # Body mass/inertia/COM DR remains sampled; only remove contact/initial pose as a fixture.
    action = torch.zeros((n, 2), device=device)
    action[:, 1] = 1. / 61.6
    env.action_manager.process_action(action)
    rows = []
    for index in range(args.physics_hz // 5):  # 0.2 s, too short to contact ground from 3 m.
        if index and index % cfg.decimation == 0:
            env.action_manager.process_action(action)
        speed = tensor(robot.data.joint_vel)[:, rj].clone()
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        expected = .2 * (1. - speed)
        torch.testing.assert_close(rear.computed_effort[:, 0], expected, rtol=1e-6, atol=2e-7)
        rows.append(torch.stack((speed, rear.computed_effort[:, 0], rear.applied_effort[:, 0]), -1).cpu().numpy())
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
    samples = np.stack(rows)
    assert tensor(robot.data.root_pos_w)[:, 2].min() > 2.
    assert torch.isfinite(tensor(robot.data.joint_vel)).all()
    # Sample the native tensor after stepping; a pre-step derived-data cache can
    # still describe the constructor's plant rather than the reset DR values.
    mass_matrix = tensor(robot.root_view.get_generalized_mass_matrices()).clone()
    offset = mass_matrix.shape[1] - robot.num_joints
    effective_j = 1. / torch.linalg.inv(mass_matrix)[:, offset+rj, offset+rj]
    pole = 1. - cfg.sim.dt * .2 / effective_j
    # Late response metrics expose oscillation under a held target, not command changes.
    late = samples[len(samples)//2:, :, 0]
    result = {
        "passed": True, "task": "Trackstand-Balance-v2", "physics_hz": args.physics_hz,
        "policy_target_hz": 50, "seed": 42, "num_envs": n, "target_speed_rad_s": 1.,
        "rear_kp": 0., "rear_kd": .2, "duration_s": .2,
        "fixture": "All bodies lifted to root z=3m at runtime; no contact; unchanged geometry and actuator parameters.",
        "effective_free_inertia_range_kg_m2": [float(effective_j.min()), float(effective_j.max())],
        "scalar_forward_euler_pole_range": [float(pole.min()), float(pole.max())],
        "scalar_forward_euler_abs_pole_over_one_envs": int((pole.abs() > 1).sum()),
        "speed_abs_max_rad_s": float(np.abs(samples[:, :, 0]).max()),
        "late_peak_to_peak_speed_max_rad_s": float(np.ptp(late, axis=0).max()),
        "late_peak_to_peak_speed_median_rad_s": float(np.median(np.ptp(late, axis=0))),
        "late_mean_abs_target_error_rad_s": float(np.abs(late - 1.).mean()),
        "computed_torque_abs_max_nm": float(np.abs(samples[:, :, 1]).max()),
        "applied_torque_abs_max_nm": float(np.abs(samples[:, :, 2]).max()),
        "command": sys.argv,
        "scope": "Unloaded discrete-controller diagnostic, not an on-ground balance evaluation or hardware test.",
    }
    np.savez_compressed(args.output / "capture.npz", speed_computed_applied=samples,
                        effective_free_inertia_kg_m2=effective_j.cpu().numpy(), mass_matrix=mass_matrix.cpu().numpy())
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PASS " + json.dumps(result), flush=True)
except BaseException:
    (args.output / "failure.json").write_text(json.dumps({"traceback": traceback.format_exc()}, indent=2))
    traceback.print_exc()
    raise
finally:
    if env is not None:
        env.close()
    app.close()
