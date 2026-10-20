"""Headless v1 steering torque/submission/DR verification, not a balance benchmark."""

import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--dr", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from isaaclab.actuators import IdealPDActuator

import source.trackstand_rl.trackstand_rl.tasks  # noqa: F401


def tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    return value.torch if hasattr(value, "torch") else wp.to_torch(value)


env = None
try:
    cfg = gym.spec("Trackstand-Balance-v1").kwargs["env_cfg_entry_point"](domain_randomization=args.dr)
    cfg.scene.num_envs, cfg.seed, cfg.sim.device = 4, 42, args.device
    env = gym.make("Trackstand-Balance-v1", cfg=cfg).unwrapped
    env.reset(seed=42)
    robot = env.scene["robot"]
    steering, drive = (robot.actuators[name] for name in ("steering", "drive"))
    steer_joint = robot.find_joints("steering_joint")[0][0]
    rear_joint = robot.find_joints("rear_wheel_joint")[0][0]
    assert type(steering) is IdealPDActuator and type(drive) is IdealPDActuator
    assert env.step_dt == 0.02 and env.physics_dt == 0.0025
    compute, write = steering.compute, robot.write_data_to_sim
    process = env.action_manager.get_term("steering").process_actions
    rows, submissions, native_checks = [], [], []
    process_calls = 0

    def check_properties():
        props = tensor(robot.root_view.get_dof_friction_properties()).cpu()
        for joint, expected in (
            (steer_joint, [0.046036056104, 0.046036056104, 0.000910713492]),
            (rear_joint, [0.106, 0.106, 0.0]),
        ):
            torch.testing.assert_close(props[:, joint], torch.tensor(expected).expand(4, 3), rtol=0, atol=1e-7)
        stiffness = tensor(robot.root_view.get_dof_stiffnesses()).cpu()
        damping = tensor(robot.root_view.get_dof_dampings()).cpu()
        assert torch.count_nonzero(stiffness[:, [steer_joint, rear_joint]]) == 0
        assert torch.count_nonzero(damping[:, [steer_joint, rear_joint]]) == 0
        speed_limits = tensor(robot.root_view.get_dof_max_velocities()).cpu()
        torch.testing.assert_close(speed_limits[:, steer_joint], torch.full((4,), 23.9))
        torch.testing.assert_close(speed_limits[:, rear_joint], torch.full((4,), 61.6))
        assert torch.count_nonzero(drive.stiffness) == 0 and torch.count_nonzero(drive.damping) == 0
        kp, kd = steering.stiffness[:, 0].cpu(), steering.damping[:, 0].cpu()
        if args.dr:
            assert torch.all((kp >= 9.0-1e-6) & (kp <= 11.0+1e-6))
            assert torch.all((kd >= 0.18-1e-7) & (kd <= 0.22+1e-7))
        else:
            torch.testing.assert_close(kp, torch.full((4,), 10.0))
            torch.testing.assert_close(kd, torch.full((4,), 0.2))
        native_checks.append(torch.stack((stiffness[:, steer_joint], damping[:, steer_joint]), dim=-1).numpy())

    def checked_process(actions):
        global process_calls
        term = env.action_manager.get_term("steering")
        previous = term.processed_actions.clone()
        process(actions)
        assert torch.all(term.processed_actions.abs() <= 1.570001)
        assert torch.all((term.processed_actions - previous).abs() <= 0.478001)
        process_calls += 1

    def checked_compute(control_action, joint_pos, joint_vel):
        assert torch.count_nonzero(control_action.joint_velocities) == 0
        assert torch.count_nonzero(control_action.joint_efforts) == 0
        target = control_action.joint_positions.clone()
        unclipped = steering.stiffness * (target - joint_pos) - steering.damping * joint_vel
        expected = unclipped.clamp(-5.6, 5.6)
        result = compute(control_action, joint_pos, joint_vel)
        torch.testing.assert_close(steering.computed_effort, unclipped)
        torch.testing.assert_close(result.joint_efforts, expected)
        assert result.joint_positions is None and result.joint_velocities is None
        rows.append(torch.stack((
            target[:, 0], joint_pos[:, 0], joint_vel[:, 0], steering.stiffness[:, 0],
            steering.damping[:, 0], unclipped[:, 0], result.joint_efforts[:, 0],
        ), dim=-1).cpu().numpy())
        return result

    def checked_write():
        old_count = len(rows)
        write()
        assert len(rows) == old_count + 1
        submitted = tensor(robot._joint_effort_target_sim)
        torch.testing.assert_close(submitted[:, steer_joint], steering.applied_effort[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(submitted[:, rear_joint], drive.applied_effort[:, 0], rtol=0, atol=0)
        assert torch.all(submitted[:, rear_joint].abs() <= 6.280001)
        submissions.append(submitted[:, [steer_joint, rear_joint]].cpu().numpy())

    env.action_manager.get_term("steering").process_actions = checked_process
    steering.compute, robot.write_data_to_sim = checked_compute, checked_write
    check_properties()
    resets, done_episodes = 1, 0
    with torch.no_grad():
        for step in range(32):
            if step and step % 8 == 0:
                env.reset(seed=42+step)
                resets += 1
                check_properties()
            if step % 8 == 0:
                # Opposing speed forces post-Kd saturation in the first two environments.
                velocity = torch.tensor([[-20.0], [20.0], [0.0], [0.0]], device=env.device)
                robot.write_joint_velocity_to_sim_index(velocity=velocity, joint_ids=[steer_joint])
            action = torch.zeros((4, 2), device=env.device)
            action[:, 0] = torch.tensor([1.0, -1.0, 0.1, -0.1], device=env.device)
            action[:, 1] = torch.tensor([0.2, -0.2, 0.1, -0.1], device=env.device)
            start, processed_before = len(rows), process_calls
            obs, reward, terminated, truncated, _ = env.step(action)
            assert len(rows) - start == 8 and process_calls - processed_before == 1
            assert tuple(obs["policy"].shape) == (4, 11)
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
            done_episodes += int((terminated | truncated).sum())
            check_properties()
    samples, outputs = np.stack(rows), np.stack(submissions)
    assert np.array_equal(samples[:, :, 6], outputs[:, :, 0])
    saturated = np.abs(samples[:, :, 5]) > 5.600001
    assert np.any(saturated) and np.any(~saturated)
    assert np.max(np.abs(outputs[:, :, 0])) <= 5.600001
    if args.dr:
        assert np.ptp(samples[:, :, 3]) > 0.01 and np.ptp(samples[:, :, 4]) > 0.001
    np.savez_compressed(
        args.output / "steering_samples.npz", controller_samples=samples,
        submitted_steer_rear_torque_nm=outputs, native_steering_pd=np.stack(native_checks),
        columns=np.array(["target_rad", "position_rad", "speed_rad_s", "kp", "kd", "unclipped_nm", "clipped_nm"]),
    )
    summary = {
        "passed": True, "command": sys.argv, "task": "Trackstand-Balance-v1", "seed": 42,
        "domain_randomization": args.dr, "num_envs": 4, "policy_steps": 32,
        "policy_rate_hz": 50, "physics_rate_hz": 400, "calculations_per_policy_step": 8,
        "physics_samples_per_motor": 32*8*4, "submission_samples_per_motor_including_resets": len(rows)*4,
        "explicit_resets_including_initial": resets, "completed_episodes": done_episodes,
        "steering_process_calls": process_calls, "steering_clip_and_50hz_slew_verified": True,
        "kp_range": [float(samples[:, :, 3].min()), float(samples[:, :, 3].max())],
        "kd_range": [float(samples[:, :, 4].min()), float(samples[:, :, 4].max())],
        "max_unclipped_abs_nm": float(np.max(np.abs(samples[:, :, 5]))),
        "max_submitted_abs_nm": float(np.max(np.abs(outputs[:, :, 0]))),
        "samples_requiring_post_pd_clipping_including_resets": int(np.sum(saturated)),
        "submitted_equals_clipped_torque_exactly": True, "native_pd_both_motors_zero": True,
        "friction_and_speed_limits_preserved_through_steps_and_resets": True,
        "steering_static_dynamic_viscous": [0.046036056104, 0.046036056104, 0.000910713492],
        "rear_static_dynamic_viscous": [0.106, 0.106, 0.0],
        "speed_limits_steer_rear_rad_s": [23.9, 61.6],
        "scope": (
            "synthetic saturation/reset/DR integration verification, "
            "not learned-policy balance or hardware performance"
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print("PASS " + json.dumps(summary), flush=True)
except BaseException:
    # Kit shutdown can suppress a pending traceback; retain explicit failure evidence.
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    if env is not None:
        env.close()
    app.close()
