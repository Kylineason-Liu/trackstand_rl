"""Headless v2 velocity/torque/reset/DR/PPO audit, not a balance benchmark."""

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback

TRAIN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRAIN))

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
from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner

PACKAGE = "source.trackstand_rl.trackstand_rl.tasks.manager_based.trackstand"
importlib.import_module(PACKAGE)


def tensor(value):
    return value if isinstance(value, torch.Tensor) else value.torch if hasattr(value, "torch") else wp.to_torch(value)


env = None
try:
    cfg = gym.spec("Trackstand-Balance-v2").kwargs["env_cfg_entry_point"](domain_randomization=args.dr)
    assert cfg.events.randomize_actuator_gains is None
    cfg.scene.num_envs, cfg.seed, cfg.sim.device = 16, 42, args.device
    state = {"active": False}
    counts = {key: 0 for key in ("observations", "drive_process", "steering_process", "drive_compute", "writes")}
    original_obs = cfg.observations["policy"].imu.func

    def checked_obs(env_arg, asset_cfg=SceneEntityCfg("robot"),
                    steer_joint_name="steering_joint", drive_joint_name="rear_wheel_joint"):
        result = original_obs(env_arg, asset_cfg, steer_joint_name, drive_joint_name)
        assert result.shape == (16, 11) and torch.isfinite(result).all()
        counts["observations"] += int(state["active"])
        return result

    cfg.observations["policy"].imu.func = checked_obs
    env = gym.make("Trackstand-Balance-v2", cfg=cfg).unwrapped
    env.reset(seed=42)
    assert env.step_dt == .02 and env.physics_dt == .0025
    robot = env.scene["robot"]
    assert "randomize_actuator_gains" not in env.event_manager.active_terms.get("reset", [])
    rear_joint = robot.find_joints("rear_wheel_joint")[0][0]
    steer_joint = robot.find_joints("steering_joint")[0][0]
    drive = robot.actuators["drive"]
    drive_term = env.action_manager.get_term("drive")
    steer_term = env.action_manager.get_term("steering")
    compute, write = drive.compute, robot.write_data_to_sim
    process_drive, process_steer = drive_term.process_actions, steer_term.process_actions
    rows, submissions, observations, actions_saved = [], [], [], []
    completed = 0

    def properties():
        stiffness = tensor(robot.root_view.get_dof_stiffnesses())
        damping = tensor(robot.root_view.get_dof_dampings())
        assert torch.count_nonzero(stiffness[:, [steer_joint, rear_joint]]) == 0
        assert torch.count_nonzero(damping[:, [steer_joint, rear_joint]]) == 0
        friction = tensor(robot.root_view.get_dof_friction_properties())
        expected = torch.tensor([.106, .106, 0.], device=friction.device).expand(16, 3)
        torch.testing.assert_close(friction[:, rear_joint], expected, rtol=0, atol=1e-7)
        speed = tensor(robot.root_view.get_dof_max_velocities())
        torch.testing.assert_close(speed[:, rear_joint], torch.full_like(speed[:, rear_joint], 61.6))
        assert torch.count_nonzero(drive.stiffness) == 0
        torch.testing.assert_close(drive.damping, torch.full_like(drive.damping, .2), rtol=0, atol=0)
        steering = robot.actuators["steering"]
        torch.testing.assert_close(steering.stiffness, torch.full_like(steering.stiffness, 10.), rtol=0, atol=0)
        torch.testing.assert_close(steering.damping, torch.full_like(steering.damping, .2), rtol=0, atol=0)

    def checked_drive_process(raw):
        process_drive(raw)
        torch.testing.assert_close(drive_term.processed_actions, (raw * 61.6).clamp(-61.6, 61.6))
        counts["drive_process"] += int(state["active"])

    def checked_steer_process(raw):
        previous = steer_term.processed_actions.clone()
        process_steer(raw)
        assert torch.all(steer_term.processed_actions.abs() <= 1.570001)
        assert torch.all((steer_term.processed_actions - previous).abs() <= .478001)
        counts["steering_process"] += int(state["active"])

    def checked_compute(command, joint_pos, joint_vel):
        target = command.joint_velocities.clone()
        assert torch.count_nonzero(command.joint_efforts) == 0
        torch.testing.assert_close(target, drive_term.processed_actions)
        expected = drive.damping * (target - joint_vel)
        result = compute(command, joint_pos, joint_vel)
        torch.testing.assert_close(drive.computed_effort, expected)
        torch.testing.assert_close(result.joint_efforts, expected.clamp(-6.28, 6.28))
        if state["active"]:
            counts["drive_compute"] += 1
            rows.append(torch.stack((target[:, 0], joint_vel[:, 0], drive.damping[:, 0],
                                     expected[:, 0], result.joint_efforts[:, 0]), -1).cpu().numpy())
        return result

    def checked_write():
        write()
        effort = tensor(robot._joint_effort_target_sim)[:, rear_joint]
        torch.testing.assert_close(effort, drive.applied_effort[:, 0], rtol=0, atol=0)
        if state["active"]:
            counts["writes"] += 1
            submissions.append(effort.cpu().numpy().copy())

    drive_term.process_actions, steer_term.process_actions = checked_drive_process, checked_steer_process
    drive.compute, robot.write_data_to_sim = checked_compute, checked_write
    properties()

    def step_checked(action, wrapped=None):
        global completed
        before = counts.copy()
        state["active"] = True
        try:
            result = env.step(action) if wrapped is None else wrapped.step(action)
        finally:
            state["active"] = False
        assert {key: counts[key] - before[key] for key in counts} == {
            "observations": 1, "drive_process": 1, "steering_process": 1, "drive_compute": 8, "writes": 8,
        }
        obs, reward = result[:2]
        done = result[2] | result[3] if wrapped is None else result[2].bool()
        completed += int(done.sum())
        assert obs["policy"].shape == (16, 11)
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
        observations.append(obs["policy"].cpu().numpy().copy())
        actions_saved.append(action.cpu().numpy().copy())
        properties()
        return result

    with torch.no_grad():
        for step in range(64):
            action = torch.empty((16, 2), device=env.device)
            action[:, 0] = torch.tensor([1., -1., .1, -.1] * 4, device=env.device)
            rear_raw = torch.tensor([1., -1., .05, -.05] * 4, device=env.device)
            action[:, 1] = rear_raw * (-1 if step % 16 >= 8 else 1)
            step_checked(action)
    manual_counts = counts.copy()

    # Two diagnostic PPO updates; no checkpoint/logging or full training campaign.
    env.reset(seed=42)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    runner_cfg = importlib.import_module(
        f"{PACKAGE}.agents.rsl_rl_ppo_beta_v2_cfg"
    ).TrackstandRLVelocityBetaPPORunnerCfg()
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, importlib.metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device=args.device)
    algorithm = runner.alg
    assert type(algorithm.actor.distribution).__name__ == "BetaDistribution"
    obs = wrapped.get_observations()
    algorithm.train_mode()
    losses = []
    for _ in range(2):
        with torch.no_grad():
            for _ in range(runner_cfg.num_steps_per_env):
                action = algorithm.act(obs)
                assert action.shape == (16, 2) and torch.isfinite(action).all() and torch.all(action.abs() <= 1)
                obs, reward, done, extras = step_checked(action, wrapped)
                algorithm.process_env_step(obs, reward, done, extras)
            algorithm.compute_returns(obs)
        values = algorithm.update()
        assert all(np.isfinite(value) for value in values.values())
        assert all(torch.isfinite(p).all() for p in algorithm.actor.parameters())
        losses.append({key: float(value) for key, value in values.items()})

    env.reset(seed=42)
    env.wheel_net_angle[:2], env.spawn_yaw[:2] = 2., .123
    env.episode_length_buf[0] = env.max_episode_length - 1
    pose = torch.cat((robot.data.root_pos_w.torch[1:2], robot.data.root_quat_w.torch[1:2]), -1).clone()
    pose[:, 0] += 1.
    robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=torch.tensor([1], device=env.device, dtype=torch.int32))
    with torch.no_grad():
        _, _, terminated, truncated, _ = step_checked(torch.zeros((16, 2), device=env.device))
    assert bool(truncated[0]) and not bool(terminated[0]) and bool(terminated[1])
    assert torch.count_nonzero(env.wheel_net_angle[:2]) == 0
    assert torch.count_nonzero(robot.data.joint_pos.torch[:2]) == 0
    assert torch.count_nonzero(robot.data.joint_vel.torch[:2]) == 0
    assert torch.count_nonzero(steer_term.processed_actions[:2]) == 0

    samples, sent = np.stack(rows), np.stack(submissions)
    np.testing.assert_array_equal(samples[:, :, 4], sent)
    assert np.any(np.abs(samples[:, :, 3]) > 6.280001)
    assert np.max(np.abs(sent)) <= 6.280001
    np.savez_compressed(args.output / "capture.npz", controller=samples, submitted_rear_torque_nm=sent,
                        observations=np.stack(observations), raw_actions=np.stack(actions_saved),
                        columns=np.array(["target_velocity_rad_s", "measured_velocity_rad_s", "kd_nm_s_per_rad",
                                          "computed_torque_nm", "applied_torque_nm"]))
    summary = {
        "passed": True, "task": "Trackstand-Balance-v2", "command": sys.argv, "seed": 42,
        "domain_randomization": args.dr, "num_envs": 16, "physics_hz": 400, "policy_hz": 50,
        "controller_gain_randomization": False,
        "fixed_controller_gains_verified_every_step": {"steering": {"kp": 10., "kd": .2},
                                                      "drive": {"kp": 0., "kd": .2}},
        "manual_policy_steps": 64, "ppo_updates": 2, "ppo_policy_steps": 32, "forced_reset_policy_steps": 1,
        "total_call_counts": counts, "manual_call_counts": manual_counts,
        "scalar_actuator_samples": int(sent.size), "completed_episodes_including_forced_resets": completed,
        "kd_range": [float(samples[:, :, 2].min()), float(samples[:, :, 2].max())],
        "max_abs_velocity_target_rad_s": float(np.max(np.abs(samples[:, :, 0]))),
        "max_abs_measured_velocity_rad_s": float(np.max(np.abs(samples[:, :, 1]))),
        "max_abs_computed_torque_nm": float(np.max(np.abs(samples[:, :, 3]))),
        "max_abs_applied_torque_nm": float(np.max(np.abs(sent))),
        "samples_needing_effort_clip": int(np.sum(np.abs(samples[:, :, 3]) > 6.280001)),
        "feedforward_always_zero": True, "native_pd_zero_both_motors": True,
        "native_friction_and_speed_limits_verified": True, "actual_submission_equals_clipped_pd_exactly": True,
        "timeout_and_failure_reset_verified": True, "finite_ppo_losses": losses,
        "scope": (
            "Untrained/synthetic functional evaluation and two PPO diagnostic updates; "
            "not balance/hardware evidence"
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("PASS " + json.dumps(summary), flush=True)
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
    failure = {"passed": False, "traceback": traceback.format_exc()}
    (args.output / "failure.json").write_text(json.dumps(failure, indent=2))
    raise
finally:
    if env is not None:
        env.close()
    app.close()
