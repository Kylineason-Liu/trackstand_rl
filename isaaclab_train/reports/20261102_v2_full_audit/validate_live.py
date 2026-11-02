"""Current-default v2 audit: observations, PI, solver submissions, reset recovery and PPO."""

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
parser.add_argument("--steps", type=int, default=256)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner

PACKAGE = "source.trackstand_rl.trackstand_rl.tasks.manager_based.trackstand"
importlib.import_module(PACKAGE)


def tensor(v):
    return v if isinstance(v, torch.Tensor) else v.torch if hasattr(v, "torch") else wp.to_torch(v)


def independent_observation(env, robot, sj, rj):
    q = tensor(robot.data.root_quat_w)
    x, y, z, w = q.unbind(-1)
    rotation = torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
    ), -1).reshape(-1, 3, 3)
    gravity = -rotation[:, 2, :]
    angular = torch.bmm(rotation.transpose(1, 2), tensor(robot.data.root_link_ang_vel_w).unsqueeze(-1))[:, :, 0]
    p, v = tensor(robot.data.joint_pos), tensor(robot.data.joint_vel)
    result = torch.cat((gravity[:, 1:], angular, p[:, [sj, rj]], v[:, [sj, rj]],
                        p[:, sj:sj+1].sin(), p[:, sj:sj+1].cos()), -1)
    if args.dr:
        result[:, :2] += env._imu_tilt_bias
        result[:, 2:5] += env._imu_gyro_bias
    return torch.nan_to_num(result, nan=0., posinf=0., neginf=0.)


env = None
try:
    cfg = gym.spec("Trackstand-Balance-v2").kwargs["env_cfg_entry_point"](domain_randomization=args.dr)
    assert cfg.sim.dt == 1 / 1200 and cfg.decimation == 24
    assert cfg.actions.drive.integral_rate_hz == 200
    cfg.scene.num_envs, cfg.seed, cfg.sim.device = 64, 42, args.device
    (args.output / "resolved_env.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str) + "\n")
    env = gym.make("Trackstand-Balance-v2", cfg=cfg).unwrapped
    env.reset(seed=42)
    assert env.step_dt == .02 and env.max_episode_length == 1500
    robot, device, n = env.scene["robot"], env.device, env.num_envs
    sj, rj = [robot.find_joints(name)[0][0] for name in ("steering_joint", "rear_wheel_joint")]
    steer, rear = [robot.actuators[name] for name in ("steering", "drive")]
    st, dt = [env.action_manager.get_term(name) for name in ("steering", "drive")]
    counts = dict(obs=0, steer_process=0, rear_process=0, steer_compute=0, rear_compute=0, integral=0, reward=0)
    maxima = dict(obs_error=0., steer_torque_nm=0., rear_torque_nm=0., rear_preclip_nm=0., integral_nm=0.)
    observations, actions, resets, noise = [], [], 0, []
    active = False

    def properties():
        assert cfg.events.randomize_actuator_gains is None
        for name, actuator, kp, kd, limit in (("steer", steer, 10., .2, 5.6), ("rear", rear, 0., .3, 6.28)):
            torch.testing.assert_close(actuator.stiffness, torch.full_like(actuator.stiffness, kp))
            torch.testing.assert_close(actuator.damping, torch.full_like(actuator.damping, kd))
            torch.testing.assert_close(actuator.effort_limit, torch.full_like(actuator.effort_limit, limit))
        assert not tensor(robot.root_view.get_dof_stiffnesses())[:, [sj, rj]].count_nonzero()
        assert not tensor(robot.root_view.get_dof_dampings())[:, [sj, rj]].count_nonzero()
        friction = tensor(robot.root_view.get_dof_friction_properties())
        for joint, values in ((sj, [.046036056104, .046036056104, .000910713492]), (rj, [.106, .106, 0.])):
            torch.testing.assert_close(friction[:, joint], torch.tensor(values, device=friction.device).expand(n, 3),
                                       rtol=0, atol=1e-7)

    obs_cfg = env.observation_manager._group_obs_term_cfgs["policy"][0]
    original_obs = obs_cfg.func

    def checked_obs(env_arg, **kwargs):
        actual = original_obs(env_arg, **kwargs)
        expected = independent_observation(env_arg, robot, sj, rj)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
        if active:
            counts["obs"] += 1
            maxima["obs_error"] = max(maxima["obs_error"], float((actual - expected).abs().max()))
        return actual

    obs_cfg.func = checked_obs

    def wrap_process(term, label):
        original = term.process_actions

        def process(raw):
            previous = term.processed_actions.clone()
            original(raw)
            if label == "steer":
                expected = previous + ((raw * 1.57).clamp(-1.57, 1.57) - previous).clamp(-.478, .478)
                torch.testing.assert_close(term.processed_actions, expected, atol=2e-7, rtol=1e-6)
            else:
                torch.testing.assert_close(term.processed_actions, (raw * 61.6).clamp(-61.6, 61.6))
            if active:
                counts[label + "_process"] += 1
        term.process_actions = process

    wrap_process(st, "steer")
    wrap_process(dt, "rear")
    original_i = dt.integral.update

    def integral_update(error, period, p_gain, effort_limit):
        original_i(error, period, p_gain, effort_limit)
        assert period == .005
        assert dt.integral.torque_nm.abs().max() <= 6.280001
        if active:
            counts["integral"] += 1
            maxima["integral_nm"] = max(maxima["integral_nm"], float(dt.integral.torque_nm.abs().max()))

    dt.integral.update = integral_update

    def wrap_actuator(actuator, label, limit):
        original = actuator.compute

        def compute(command, joint_pos, joint_vel):
            if label == "steer":
                expected = 10. * (command.joint_positions - joint_pos) - .2 * joint_vel
                assert not command.joint_velocities.count_nonzero()
            else:
                torch.testing.assert_close(command.joint_efforts, dt.integral.torque_nm)
                expected = .3 * (command.joint_velocities - joint_vel) + command.joint_efforts
            result = original(command, joint_pos, joint_vel)
            torch.testing.assert_close(actuator.computed_effort, expected)
            torch.testing.assert_close(result.joint_efforts, expected.clamp(-limit, limit))
            if active:
                counts[label + "_compute"] += 1
                maxima[label + "_torque_nm"] = max(
                    maxima[label + "_torque_nm"], float(result.joint_efforts.abs().max()))
                if label == "rear":
                    maxima["rear_preclip_nm"] = max(maxima["rear_preclip_nm"], float(expected.abs().max()))
            return result
        actuator.compute = compute

    wrap_actuator(steer, "steer", 5.6)
    wrap_actuator(rear, "rear", 6.28)
    original_reward = env.reward_manager.compute

    def checked_reward(dt):
        period = dt
        g = tensor(robot.data.projected_gravity_b)[:, 1].abs().clamp(max=1.)
        alive = (torch.rad2deg(torch.asin(g)) < 30.).float()
        pos = tensor(robot.data.root_pos_w)[:, :2] - env.scene.env_origins[:, :2]
        drift = torch.nan_to_num(torch.exp(-pos.square().sum(-1) / (2*.25**2)), nan=0.)
        rate = (env.action_manager.action - env.action_manager.prev_action).norm(dim=-1)
        expected = period * (5. * alive + 2. * drift - .1 * rate)
        value = original_reward(period)
        torch.testing.assert_close(value, expected, rtol=1e-6, atol=2e-7)
        assert torch.isfinite(value).all()
        if active:
            counts["reward"] += 1
        return value

    env.reward_manager.compute = checked_reward

    def step(action, wrapped=None):
        global active, resets
        before = counts.copy()
        active = True
        try:
            result = env.step(action) if wrapped is None else wrapped.step(action)
        finally:
            active = False
        expected = dict(obs=1, steer_process=1, rear_process=1, steer_compute=24,
                        rear_compute=24, integral=4, reward=1)
        assert {k: counts[k] - before[k] for k in counts} == expected
        obs, reward = result[:2]
        done = result[2] | result[3] if wrapped is None else result[2].bool()
        assert obs["policy"].shape == (n, 11) and torch.isfinite(obs["policy"]).all()
        assert torch.isfinite(reward).all()
        torch.testing.assert_close(dt.integral.torque_nm[done], torch.zeros_like(dt.integral.torque_nm[done]))
        torch.testing.assert_close(env.wheel_net_angle[done], torch.zeros_like(env.wheel_net_angle[done]))
        if done.any():
            assert not tensor(robot.data.joint_pos)[done].count_nonzero()
            assert not tensor(robot.data.joint_vel)[done].count_nonzero()
            assert not st.processed_actions[done].count_nonzero()
        properties()
        resets += int(done.sum())
        observations.append(obs["policy"].cpu().numpy().copy())
        actions.append(action.cpu().numpy().copy())
        noise.append((obs["policy"] - independent_observation(env, robot, sj, rj))[:, :5].cpu().numpy())
        return result

    properties()
    rng = torch.Generator(device=device).manual_seed(42)
    with torch.no_grad():
        for _ in range(args.steps):
            step(2 * torch.rand((n, 2), device=device, generator=rng) - 1)
    stress_resets, stress_maxima = resets, maxima.copy()

    # Six deliberately independent terminal stimuli; no native NaN writes.
    env.reset(seed=43)
    ids = torch.arange(4, device=device, dtype=torch.int32)
    pose = torch.cat((tensor(robot.data.root_pos_w)[:4], tensor(robot.data.root_quat_w)[:4]), -1).clone()
    pose[1, 3:] = torch.tensor([np.sin(.4), 0., 0., np.cos(.4)], device=device)
    pose[2, 3:] = torch.tensor([0., 0., np.sin(.5), np.cos(.5)], device=device)
    pose[3, 0] += .7
    robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    env.episode_length_buf[0] = env.max_episode_length - 1
    env.wheel_net_angle[4] = 7.3
    velocity = torch.zeros((1, 6), device=device)
    velocity[:, 0] = 30.
    robot.write_root_velocity_to_sim_index(root_velocity=velocity,
                                          env_ids=torch.tensor([5], device=device, dtype=torch.int32))
    _, _, terminated, truncated, _ = step(torch.zeros((n, 2), device=device))
    causes = {name: bool(env.termination_manager.get_term(name)[i]) for i, name in enumerate(
        ("time_out", "fallen", "yaw_deviation", "drifted", "wheel_net", "diverged"))}
    assert all(causes.values()) and truncated[0] and terminated[1:6].all(), causes

    # Contaminate cached buffers AFTER physics, BEFORE terminations/reward/reset.
    # This exercises actual manager recovery without sending NaNs into PhysX.
    env.reset(seed=44)
    original_termination = env.termination_manager.compute
    injected = False

    def inject_before_termination():
        global injected
        if not injected:
            tensor(robot.data.root_pos_w)[0, 0] = float("nan")
            tensor(robot.data.joint_pos)[1, rj] = float("nan")
            dt.integral.torque_nm[2] = float("nan")
            env.wheel_net_angle[3] = float("nan")
            injected = True
        return original_termination()

    env.termination_manager.compute = inject_before_termination
    try:
        _, reward, terminated, _, extras = step(torch.zeros((n, 2), device=device))
    finally:
        env.termination_manager.compute = original_termination
    assert terminated[:4].all() and torch.isfinite(reward).all()
    assert all(torch.isfinite(v).all() for v in env.reward_manager._episode_sums.values())
    assert all(np.isfinite(float(v)) for v in extras["log"].values())

    # Actual branch crossing must terminate on this same policy step.
    env.reset(seed=45)
    index = torch.tensor([6], device=device, dtype=torch.int32)
    position, velocity = tensor(robot.data.joint_pos)[6:7].clone(), tensor(robot.data.joint_vel)[6:7].clone()
    velocity[:, rj] = 20.
    robot.write_joint_state_to_sim_index(position=position, velocity=velocity, env_ids=index)
    env.wheel_net_angle[6] = 7.19
    action = torch.zeros((n, 2), device=device)
    action[6, 1] = 20 / 61.6
    step(action)
    assert env.termination_manager.get_term("wheel_net")[6] and env.wheel_net_angle[6] == 0.

    # Partial manual reset clears only selected integral memory and targets.
    dt.integral.torque_nm[:] = .1
    dt.integral.error_rad[:] = .1 / 3.3
    robot.set_joint_effort_target_index(target=dt.integral.torque_nm, joint_ids=[rj])
    ids = torch.arange(0, n, 2, device=device, dtype=torch.int32)
    env.reset(env_ids=ids)
    assert not dt.integral.torque_nm[ids].count_nonzero()
    torch.testing.assert_close(dt.integral.torque_nm[1::2], torch.full((n // 2, 1), .1, device=device))
    assert not tensor(robot.data.joint_effort_target)[ids, rj].count_nonzero()

    # Two real PPO updates with an initialized Beta actor; no checkpoint saved.
    env.reset(seed=46)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    agent_module = importlib.import_module(f"{PACKAGE}.agents.rsl_rl_ppo_beta_v2_cfg")
    runner_cfg = agent_module.TrackstandRLVelocityBetaPPORunnerCfg()
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, importlib.metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device=device)
    alg, obs, losses = runner.alg, wrapped.get_observations(), []
    alg.train_mode()
    assert type(alg.actor.distribution).__name__ == "BetaDistribution"
    for _ in range(2):
        with torch.no_grad():
            for _ in range(runner_cfg.num_steps_per_env):
                action = alg.act(obs)
                assert torch.isfinite(action).all() and (action.abs() <= 1).all()
                obs, reward, done, extras = step(action, wrapped)
                alg.process_env_step(obs, reward, done, extras)
            alg.compute_returns(obs)
        loss = alg.update()
        assert all(np.isfinite(v) for v in loss.values())
        assert all(torch.isfinite(p).all() for p in alg.actor.parameters())
        losses.append({k: float(v) for k, v in loss.items()})

    np.savez_compressed(args.output / "capture.npz", observations=np.stack(observations),
                        raw_actions=np.stack(actions), noise=np.concatenate(noise))
    summary = dict(passed=True, seed=42, domain_randomization=args.dr, num_envs=n, synthetic_steps=args.steps,
                   physics_hz=1200, policy_observation_hz=50, integral_hz=200, call_counts=counts,
                   stress_resets=stress_resets, stress_maxima=stress_maxima,
                   all_termination_causes=causes, partial_reset_passed=True, wheel_net_same_step_passed=True,
                   invalid_cached_state_recovery_passed=True, invalid_state_native_physics_not_tested=True,
                   ppo_updates=2, ppo_losses=losses, white_noise_std=np.concatenate(noise).std(axis=0).tolist(),
                   note="Functional and numerical audit; not a trained-policy survival benchmark or motor calibration.")
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
