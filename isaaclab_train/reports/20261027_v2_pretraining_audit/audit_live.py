"""Independent live v2 observation, actuator, reward, DR and episode audit."""

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
parser.add_argument("--dr", action="store_true")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=256)
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
    if isinstance(value, torch.Tensor):
        return value
    return value.torch if hasattr(value, "torch") else wp.to_torch(value)


def rotation_matrix(q):
    """Body-to-world matrix directly from xyzw; no Isaac quaternion helpers."""
    x, y, z, w = q.unbind(-1)
    return torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
    ), -1).reshape(-1, 3, 3)


env = None
try:
    cfg = gym.spec("Trackstand-Balance-v2").kwargs["env_cfg_entry_point"](domain_randomization=args.dr)
    cfg.scene.num_envs, cfg.seed, cfg.sim.device = args.num_envs, 42, args.device
    assert args.num_envs >= 8
    assert cfg.events.randomize_actuator_gains is None
    obs_cfg = cfg.observations["policy"].imu
    assert obs_cfg.clip is None and obs_cfg.scale is None and obs_cfg.history_length == 0
    (args.output / "resolved_env.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str) + "\n")
    env = gym.make("Trackstand-Balance-v2", cfg=cfg).unwrapped
    env.reset(seed=42)
    robot, device, n = env.scene["robot"], env.device, args.num_envs
    sj, rj = [robot.find_joints(name)[0][0] for name in ("steering_joint", "rear_wheel_joint")]
    steer, rear = [robot.actuators[name] for name in ("steering", "drive")]
    steer_term, rear_term = [env.action_manager.get_term(name) for name in ("steering", "drive")]
    assert (env.physics_dt, env.step_dt, env.max_episode_length) == (.0025, .02, 1500)
    events = env.event_manager.active_terms
    assert "randomize_actuator_gains" not in events.get("reset", [])
    assert "interval" not in events
    counts = {"obs": 0, "steer_compute": 0, "rear_compute": 0, "reward": 0}
    maxima = {"obs_error": 0., "steer_speed": 0., "rear_speed": 0., "steer_effort": 0., "rear_effort": 0.}
    noise, obs_rows, raw_rows, effort_rows, net_errors = [], [], [], [], []
    reset_count = 0
    phase = {"stepping": False}

    def physical_properties():
        for actuator, kp, kd in ((steer, 10., .2), (rear, 0., .2)):
            torch.testing.assert_close(actuator.stiffness, torch.full_like(actuator.stiffness, kp), rtol=0, atol=0)
            torch.testing.assert_close(actuator.damping, torch.full_like(actuator.damping, kd), rtol=0, atol=0)
        assert not tensor(robot.root_view.get_dof_stiffnesses())[:, [sj, rj]].count_nonzero()
        assert not tensor(robot.root_view.get_dof_dampings())[:, [sj, rj]].count_nonzero()
        friction = tensor(robot.root_view.get_dof_friction_properties())
        for joint, expected in ((sj, [.046036056104, .046036056104, .000910713492]), (rj, [.106, .106, 0.])):
            value = torch.tensor(expected, device=friction.device).expand(n, 3)
            torch.testing.assert_close(friction[:, joint], value, rtol=0, atol=1e-7)

    def expected_obs():
        rotation = rotation_matrix(tensor(robot.data.root_quat_w))
        gravity = -rotation[:, 2, :]
        angular = torch.bmm(rotation.transpose(1, 2), tensor(robot.data.root_link_ang_vel_w).unsqueeze(-1))[:, :, 0]
        q, v = tensor(robot.data.joint_pos), tensor(robot.data.joint_vel)
        result = torch.cat((gravity[:, 1:], angular, q[:, [sj, rj]], v[:, [sj, rj]],
                            q[:, sj:sj+1].sin(), q[:, sj:sj+1].cos()), -1)
        if args.dr:
            result[:, :2] += env._imu_tilt_bias
            result[:, 2:5] += env._imu_gyro_bias
        return torch.nan_to_num(result, nan=0., posinf=0., neginf=0.)

    # This installed ObservationManager has no public get_term_cfg accessor.
    live_obs_cfg = env.observation_manager._group_obs_term_cfgs["policy"][0]
    original_obs = live_obs_cfg.func

    def observed(env_arg, **kwargs):
        actual = original_obs(env_arg, **kwargs)
        expected = expected_obs()
        maxima["obs_error"] = max(maxima["obs_error"], float((actual - expected).abs().max()))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
        counts["obs"] += 1
        return actual

    live_obs_cfg.func = observed

    def controller_hook(name, original, limit):
        def compute(command, joint_pos, joint_vel):
            assert not command.joint_efforts.count_nonzero()
            if name == "steer":
                if phase["stepping"]:
                    torch.testing.assert_close(command.joint_positions, steer_term.processed_actions)
                assert not command.joint_velocities.count_nonzero()
                expected = 10. * (command.joint_positions - joint_pos) - .2 * joint_vel
            else:
                if phase["stepping"]:
                    torch.testing.assert_close(command.joint_velocities, rear_term.processed_actions)
                expected = .2 * (command.joint_velocities - joint_vel)
            result = original(command, joint_pos, joint_vel)
            torch.testing.assert_close(result.joint_efforts, expected.clamp(-limit, limit), rtol=0, atol=0)
            assert result.joint_positions is None and result.joint_velocities is None
            counts[name + "_compute"] += 1
            maxima[name + "_speed"] = max(maxima[name + "_speed"], float(joint_vel.abs().max()))
            maxima[name + "_effort"] = max(maxima[name + "_effort"], float(result.joint_efforts.abs().max()))
            return result
        return compute

    steer.compute = controller_hook("steer", steer.compute, 5.6)
    rear.compute = controller_hook("rear", rear.compute, 6.28)
    reward_compute = env.reward_manager.compute

    def reward(dt):
        gy = tensor(robot.data.projected_gravity_b)[:, 1].abs().clamp(max=1.)
        survival = (torch.rad2deg(torch.asin(gy)) < 30.).float()
        distance = (tensor(robot.data.root_pos_w)[:, :2] - env.scene.env_origins[:, :2]).norm(dim=-1)
        delta = env.action_manager.action - env.action_manager.prev_action
        expected = dt * (5. * survival + 2. * torch.exp(-distance.square() / (2*.25**2)) - .1*delta.norm(dim=-1))
        actual = reward_compute(dt)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=2e-7)
        counts["reward"] += 1
        return actual

    env.reward_manager.compute = reward

    def step(action, record=True):
        nonlocal_before = counts.copy()
        previous_target = steer_term.processed_actions.clone()
        previous_q = tensor(robot.data.joint_pos)[:, rj].clone()
        previous_net = env.wheel_net_angle.clone()
        phase["stepping"] = True
        try:
            result = env.step(action)
        finally:
            phase["stepping"] = False
        obs, rew, term, trunc, _ = result
        done = term | trunc
        assert counts["obs"] - nonlocal_before["obs"] == 1
        assert counts["reward"] - nonlocal_before["reward"] == 1
        for key in ("steer_compute", "rear_compute"):
            assert counts[key] - nonlocal_before[key] == 8
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(rew).all()
        expected_steer = previous_target + (action[:, :1]*1.57 - previous_target).clamp(-.478, .478)
        torch.testing.assert_close(steer_term.processed_actions[~done], expected_steer[~done], atol=2e-7, rtol=1e-6)
        torch.testing.assert_close(rear_term.processed_actions, action[:, 1:]*61.6)
        torch.testing.assert_close(env.wheel_net_angle[done], torch.zeros_like(env.wheel_net_angle[done]))
        if done.any():
            for value in (tensor(robot.data.joint_pos), tensor(robot.data.joint_vel), steer_term.processed_actions):
                torch.testing.assert_close(value[done], torch.zeros_like(value[done]))
        physical_properties()
        if record:
            error = obs["policy"] - expected_obs()
            torch.testing.assert_close(error[:, 5:], torch.zeros_like(error[:, 5:]), atol=2e-6, rtol=0)
            noise.append(error[:, :5].cpu().numpy())
            obs_rows.append(obs["policy"].cpu().numpy().copy())
            raw_rows.append(action.cpu().numpy().copy())
            effort_rows.append(torch.cat((steer.applied_effort, rear.applied_effort), -1).cpu().numpy().copy())
            actual_delta = (tensor(robot.data.joint_pos)[:, rj] - previous_q + 2*torch.pi) % (4*torch.pi) - 2*torch.pi
            net_error = env.wheel_net_angle - previous_net - actual_delta
            net_errors.extend(net_error[~done].cpu().tolist())
        return result

    physical_properties()
    initial = expected_obs()
    assert not tensor(robot.data.joint_pos).count_nonzero()
    assert not tensor(robot.data.joint_vel).count_nonzero()
    if not args.dr:
        torch.testing.assert_close(initial[:, :5], torch.tensor([0., -1., 0., 0., 0.], device=device).expand(n, 5))
    generator = torch.Generator(device=device).manual_seed(43)
    for index in range(args.steps):
        # Bounded synthetic targets; no claim this is a trained balance policy.
        action = torch.rand((n, 2), device=device, generator=generator)*2 - 1
        with torch.no_grad():
            result = step(action)
        reset_count += int((result[2] | result[3]).sum())
    synthetic_maxima = maxima.copy()

    # All six configured termination causes, followed by partial auto-resets.
    env.reset(seed=44)
    pose = torch.cat((tensor(robot.data.root_pos_w)[:4], tensor(robot.data.root_quat_w)[:4]), -1).clone()
    pose[1, 3:] = torch.tensor([np.sin(.4), 0., 0., np.cos(.4)], device=device)
    pose[2, 3:] = torch.tensor([0., 0., np.sin(.5), np.cos(.5)], device=device)
    pose[3, 0] += .7
    ids = torch.arange(4, device=device, dtype=torch.int32)
    robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    velocity = torch.zeros((1, 6), device=device)
    velocity[:, 0] = 30.
    robot.write_root_velocity_to_sim_index(
        root_velocity=velocity, env_ids=torch.tensor([5], device=device, dtype=torch.int32))
    env.episode_length_buf[0] = env.max_episode_length - 1
    env.wheel_net_angle[4] = 7.3
    with torch.no_grad():
        obs, _, terminated, truncated, _ = step(torch.zeros((n, 2), device=device), record=False)
    causes = {name: bool(env.termination_manager.get_term(name)[idx]) for idx, name in enumerate(
        ("time_out", "fallen", "yaw_deviation", "drifted", "wheel_net", "diverged"))}
    assert all(causes.values()), causes
    assert truncated[0] and not terminated[0]
    assert terminated[1:6].all()
    assert env.episode_length_buf[:6].count_nonzero() == 0

    # Quantify the inherited one-policy-step delay in wheel-net termination.
    env.reset(seed=45)
    position, speed = tensor(robot.data.joint_pos)[6:7].clone(), tensor(robot.data.joint_vel)[6:7].clone()
    speed[:, rj] = 20.
    robot.write_joint_state_to_sim_index(position=position, velocity=speed,
                                        env_ids=torch.tensor([6], device=device, dtype=torch.int32))
    env.wheel_net_angle[6] = 7.19
    action = torch.zeros((n, 2), device=device)
    action[6, 1] = 20. / 61.6
    with torch.no_grad():
        step(action, record=False)
    crossing = float(env.wheel_net_angle[6])
    boundary_probe = {"after_first_step_net_rad": crossing,
                      "after_first_step_speed_rad_s": float(tensor(robot.data.joint_vel)[6, rj]),
                      "first_step_wheel_net_termination": bool(env.termination_manager.get_term("wheel_net")[6])}
    (args.output / "wheel_net_boundary_probe.json").write_text(json.dumps(boundary_probe, indent=2) + "\n")
    assert crossing > 7.2 and not env.termination_manager.get_term("wheel_net")[6], boundary_probe
    with torch.no_grad():
        step(action, record=False)
    assert env.termination_manager.get_term("wheel_net")[6]
    assert env.wheel_net_angle[6] == 0.

    residual = np.concatenate(noise)
    errors = np.asarray(net_errors)
    np.savez_compressed(args.output / "capture.npz", observations=np.stack(obs_rows), raw_actions=np.stack(raw_rows),
                        applied_torque_nm=np.stack(effort_rows), white_noise=residual,
                        wheel_net_increment_error_rad=errors)
    result = {
        "passed": True, "task": "Trackstand-Balance-v2", "domain_randomization": args.dr,
        "command": sys.argv, "seed": 42, "num_envs": n, "synthetic_steps": args.steps,
        "frequency_hz": {"policy_observation": 50, "physics_both_controllers": 400},
        "counts": counts, "maxima": synthetic_maxima, "synthetic_resets": reset_count,
        "maxima_including_forced_termination_stimuli": maxima,
        "all_termination_causes_verified": causes, "timeout_and_failure_resets_verified": True,
        "fixed_gains_and_passive_friction_verified_after_each_step": True, "active_events": events,
        "white_noise_mean": residual.mean(axis=0).tolist(), "white_noise_std": residual.std(axis=0).tolist(),
        "configured_white_noise_std": [.001, .001, .0012, .0012, .0012],
        "wheel_net_delay_s": .02, "wheel_net_at_undetected_first_crossing_rad": crossing,
        "policy_end_speed_integral_minus_native_angle_increment_abs_max_rad": float(np.abs(errors).max()),
        "policy_end_speed_integral_minus_native_angle_increment_rms_rad": float(np.sqrt(np.mean(errors**2))),
        "scope": "Synthetic functional stress audit; no trained-policy survival or hardware fidelity conclusion.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PASS " + json.dumps(result), flush=True)
except BaseException:
    failure = {"passed": False, "traceback": traceback.format_exc()}
    (args.output / "failure.json").write_text(json.dumps(failure, indent=2))
    traceback.print_exc()
    raise
finally:
    if env is not None:
        env.close()
    app.close()
