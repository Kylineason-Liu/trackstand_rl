"""Verify v1's rear torque submission, native gains and DR in the real environment.

Synthetic actions/velocity writes stress the controller contract, not balance
performance. No USD, training checkpoint or vendor source is changed.
"""

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--dr", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    raise ValueError("Use a fresh output directory")
args.output.mkdir(parents=True)
invocation = sys.argv.copy()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from isaaclab.actuators import IdealPDActuator, ImplicitActuator

import source.trackstand_rl.trackstand_rl.tasks  # noqa: F401


def tensor(value):
    if hasattr(value, "torch"):
        return value.torch
    return wp.to_torch(value) if isinstance(value, wp.array) else torch.as_tensor(value)


env = None
try:
    cfg = gym.spec("Trackstand-Balance-v1").kwargs["env_cfg_entry_point"](domain_randomization=args.dr)
    cfg.scene.num_envs = 4
    cfg.sim.device = args.device
    cfg.seed = 42
    env = gym.make("Trackstand-Balance-v1", cfg=cfg).unwrapped
    env.reset(seed=42)
    robot = env.scene["robot"]
    drive = robot.actuators["drive"]
    joint = robot.find_joints("rear_wheel_joint")[0][0]
    assert type(drive) is IdealPDActuator
    assert isinstance(robot.actuators["steering"], ImplicitActuator)
    assert env.step_dt == 0.02 and env.physics_dt == 0.0025
    compute = drive.compute
    write = robot.write_data_to_sim
    rows = []
    submitted = []
    native_gains = []

    def check_native_gains():
        stiffness = tensor(robot.root_view.get_dof_stiffnesses())[:, joint].cpu()
        damping = tensor(robot.root_view.get_dof_dampings())[:, joint].cpu()
        assert torch.count_nonzero(stiffness) == 0
        assert torch.count_nonzero(damping) == 0
        limits = tensor(robot.root_view.get_dof_max_velocities())[:, joint].cpu()
        torch.testing.assert_close(limits, torch.full_like(limits, 61.6))
        native_gains.append(torch.stack((stiffness, damping), dim=-1).numpy())

    def checked_compute(control_action, joint_pos, joint_vel):
        assert torch.count_nonzero(control_action.joint_velocities) == 0
        feedforward = control_action.joint_efforts.clone()
        expected_unclipped = feedforward - drive.damping * joint_vel
        expected = expected_unclipped.clamp(-6.28, 6.28)
        result = compute(control_action, joint_pos, joint_vel)
        torch.testing.assert_close(drive.computed_effort, expected_unclipped)
        torch.testing.assert_close(result.joint_efforts, expected)
        assert result.joint_positions is None and result.joint_velocities is None
        rows.append(torch.stack((
            feedforward[:, 0], joint_vel[:, 0], drive.damping[:, 0],
            drive.computed_effort[:, 0], result.joint_efforts[:, 0],
        ), dim=-1).cpu().numpy())
        return result

    def checked_write():
        old_count = len(rows)
        write()
        assert len(rows) == old_count + 1
        output = tensor(robot._joint_effort_target_sim)[:, joint].clone()
        torch.testing.assert_close(output, drive.applied_effort[:, 0], rtol=0, atol=0)
        submitted.append(output.cpu().numpy())

    drive.compute = checked_compute
    robot.write_data_to_sim = checked_write
    check_native_gains()
    resets = 0
    with torch.inference_mode():
        for step in range(64):
            if step % 8 == 0:
                # Re-exercise saturated and unsaturated cases after real resets,
                # including the existing per-environment actuator-gain DR.
                env.reset(seed=42 + step)
                check_native_gains()
                resets += 1
                velocity = torch.tensor([[-20.0], [20.0], [20.0], [-20.0]], device=env.device)
                robot.write_joint_velocity_to_sim_index(velocity=velocity, joint_ids=[joint])
            action = torch.zeros((4, 2), device=env.device)
            action[:, 0] = 1.0 if step % 16 < 8 else -1.0
            action[:, 1] = torch.tensor([1.0, -1.0, 0.2, -0.2], device=env.device)
            start = len(rows)
            env.step(action)
            assert len(rows) - start == 8
    samples, outputs = np.stack(rows), np.stack(submitted)
    assert np.max(np.abs(outputs)) <= 6.280001
    saturated = np.abs(samples[:, :, 3]) > 6.280001
    assert np.any(saturated) and np.any(~saturated)
    assert np.array_equal(outputs, samples[:, :, 4])
    kd = samples[:, :, 2]
    if args.dr:
        assert np.min(kd) >= 0.045 - 1e-7 and np.max(kd) <= 0.055 + 1e-7
        assert np.max(kd) - np.min(kd) > 1e-4
    else:
        assert np.all(np.abs(kd - 0.05) < 1e-7)
    np.savez_compressed(
        args.output / "drive_samples.npz", controller_samples=samples, submitted_torque_nm=outputs,
        native_pd_gains=np.stack(native_gains),
        columns=np.array(["feedforward_nm", "velocity_rad_s", "kd_nm_s_per_rad", "unclipped_nm", "clipped_nm"]),
    )
    summary = {
        "command": invocation, "passed": True, "task": "Trackstand-Balance-v1", "seed": 42,
        "domain_randomization": args.dr, "num_envs": 4, "policy_steps": 64,
        "policy_rate_hz": 50.0, "drive_calculation_rate_hz": 400.0,
        "physics_calls_per_policy_step": 8, "physics_samples_per_motor": 64 * 8 * 4,
        "total_submission_samples_per_motor": len(rows) * 4,
        "additional_reset_submission_calls": len(rows) - 64 * 8,
        "explicit_resets": resets, "native_rear_stiffness_and_damping_are_zero": True,
        "rear_velocity_limit_rad_s": 61.6, "max_feedforward_abs_nm": float(np.max(np.abs(samples[:, :, 0]))),
        "max_unclipped_abs_nm": float(np.max(np.abs(samples[:, :, 3]))),
        "max_submitted_abs_nm": float(np.max(np.abs(outputs))),
        "samples_requiring_post_damping_clipping": int(np.sum(saturated)),
        "kd_min_max_nm_s_per_rad": [float(np.min(kd)), float(np.max(kd))],
        "submitted_equals_clipped_torque_exactly": True,
        "scope": "synthetic controller integration test; not policy balance or passive-friction identification",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print("PASS: post-damping rear torque clamp, native gains and reset/DR handling", flush=True)
finally:
    if env is not None:
        env.close()
    app.close()
