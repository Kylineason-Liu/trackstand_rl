"""Read-only v1 integration audit: math, rates, PPO and failure/timeout reset paths.

Only in-memory instances are instrumented. Synthetic state/counter writes test
reset boundaries; two short PPO updates are diagnostic, not a training campaign.
No trained checkpoint or runtime source is written.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback

TRAIN = Path(__file__).resolve().parents[2]
ROOT = TRAIN.parent
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAIN))


def manifest():
    paths = list((TRAIN / "source").rglob("*.py"))
    paths += list((TRAIN / "scripts").rglob("*.py")) + list((TRAIN / "tests").rglob("*.py"))
    paths += [ROOT / "progress_record.md", ROOT / "progress_record.txt", TRAIN / "README.md"]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


before = manifest()
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--audit_num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import source.trackstand_rl.trackstand_rl.tasks  # noqa: F401
from source.trackstand_rl.trackstand_rl.tasks.manager_based.trackstand.agents.beta_distribution import BetaDistribution
from source.trackstand_rl.trackstand_rl.tasks.manager_based.trackstand.agents.rsl_rl_ppo_beta_cfg import (
    TrackstandRLBetaPPORunnerCfg,
)


def independent_observation(env):
    robot = env.scene["robot"]
    x, y, z, w = robot.data.root_quat_w.torch.unbind(-1)
    rotation = torch.stack((
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y),
    ), dim=-1).reshape(-1, 3, 3)
    gravity = -rotation[:, 2, :]
    angular = torch.bmm(rotation.transpose(1, 2), robot.data.root_link_ang_vel_w.torch.unsqueeze(-1)).squeeze(-1)
    steer = robot.find_joints("steering_joint")[0][0]
    rear = robot.find_joints("rear_wheel_joint")[0][0]
    position, velocity = robot.data.joint_pos.torch, robot.data.joint_vel.torch
    result = torch.cat((gravity[:, 1:3], angular, position[:, [steer, rear]], velocity[:, [steer, rear]],
                        position[:, steer:steer+1].sin(), position[:, steer:steer+1].cos()), dim=-1)
    if hasattr(env, "_imu_tilt_bias"):
        result[:, :2] += env._imu_tilt_bias
        result[:, 2:5] += env._imu_gyro_bias
    return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


env = None
results = []
try:
    for dr in (False, True):
        counters = dict(observation=0, action_processing=0, steering_compute=0, drive_compute=0,
                        reward=0, termination=0)
        state = dict(active=False, max_observation_error=0.0, max_reward_error=0.0)
        cfg = gym.spec("Trackstand-Balance-v1").kwargs["env_cfg_entry_point"](domain_randomization=dr)
        cfg.scene.num_envs, cfg.seed, cfg.sim.device = args.audit_num_envs, 42, args.device
        original_observation = cfg.observations["policy"].imu.func

        def checked_observation(
            env_arg, asset_cfg=SceneEntityCfg("robot"),
            steer_joint_name="steering_joint", drive_joint_name="rear_wheel_joint",
        ):
            value = original_observation(
                env_arg, asset_cfg=asset_cfg,
                steer_joint_name=steer_joint_name, drive_joint_name=drive_joint_name,
            )
            expected = independent_observation(env_arg)
            torch.testing.assert_close(value, expected, rtol=1e-5, atol=1e-5)
            state["max_observation_error"] = max(state["max_observation_error"], float((value-expected).abs().max()))
            if state["active"]:
                counters["observation"] += 1
                state["last_raw_observation"] = value.clone()
            return value

        cfg.observations["policy"].imu.func = checked_observation
        env = gym.make("Trackstand-Balance-v1", cfg=cfg).unwrapped
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        robot = env.scene["robot"]
        steering = env.action_manager.get_term("steering")
        assert env.step_dt == 0.02 and env.physics_dt == 0.0025
        assert env.max_episode_length == 1500

        def counted(function, key):
            def call(*args, **kwargs):
                if state["active"]:
                    counters[key] += 1
                return function(*args, **kwargs)
            return call

        env.action_manager.process_action = counted(env.action_manager.process_action, "action_processing")
        for name in ("steering", "drive"):
            actuator = robot.actuators[name]
            actuator.compute = counted(actuator.compute, name+"_compute")
        env.termination_manager.compute = counted(env.termination_manager.compute, "termination")
        original_reward = env.reward_manager.compute

        def checked_reward(dt):
            gravity_y = robot.data.projected_gravity_b.torch[:, 1]
            alive = (torch.rad2deg(torch.asin(gravity_y.abs().clamp(max=1))) < 30).float()
            distance_squared = ((robot.data.root_pos_w.torch[:, :2]-env.scene.env_origins[:, :2])**2).sum(-1)
            drift = torch.exp(-distance_squared/(2*0.25**2))
            change = env.action_manager.action-env.action_manager.prev_action
            rate = -torch.sqrt(((change/torch.tensor([1.57, 6.28], device=env.device))**2).sum(-1)+1e-8)
            expected = dt*(5*alive+2*drift+0.1*rate)
            actual = original_reward(dt)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            state["max_reward_error"] = max(state["max_reward_error"], float((actual-expected).abs().max()))
            if state["active"]:
                counters["reward"] += 1
            return actual

        env.reward_manager.compute = checked_reward
        runner_cfg = handle_deprecated_rsl_rl_cfg(
            TrackstandRLBetaPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib")
        )
        runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device=args.device)
        algorithm = runner.alg
        assert type(algorithm.actor.distribution) is BetaDistribution
        assert not algorithm.actor.obs_normalization and not algorithm.critic.obs_normalization
        assert not algorithm.actor.is_recurrent
        observation = wrapped.get_observations()
        actions_saved, observations_saved, noises, losses_saved = [], [], [], []
        termination_counts = torch.zeros(len(env.termination_manager.active_terms), device=env.device)
        algorithm.train_mode()
        for iteration in range(2):
            with torch.no_grad():
                for step in range(runner_cfg.num_steps_per_env):
                    actions = algorithm.act(observation)
                    assert actions.shape == (args.audit_num_envs, 2)
                    assert torch.isfinite(actions).all() and torch.all(actions.abs() <= 1)
                    previous = counters.copy()
                    state["active"] = True
                    observation, reward, done, extras = wrapped.step(actions)
                    state["active"] = False
                    expected_calls = dict(observation=1, action_processing=1, steering_compute=8,
                                          drive_compute=8, reward=1, termination=1)
                    assert {key: counters[key]-previous[key] for key in counters} == expected_calls
                    assert observation["policy"].shape == (args.audit_num_envs, 11)
                    assert torch.isfinite(observation["policy"]).all() and torch.isfinite(reward).all()
                    delta = observation["policy"]-state["last_raw_observation"]
                    assert torch.count_nonzero(delta[:, 5:]) == 0
                    noises.append(delta[:, :5].clone())
                    algorithm.process_env_step(observation, reward, done, extras)
                    actions_saved.append(actions.clone())
                    observations_saved.append(observation["policy"].clone())
                    for column, name in enumerate(env.termination_manager.active_terms):
                        termination_counts[column] += env.termination_manager.get_term(name).sum()
                algorithm.compute_returns(observation)
            losses = algorithm.update()
            assert all(np.isfinite(value) for value in losses.values())
            assert all(torch.isfinite(parameter).all() for parameter in algorithm.actor.parameters())
            losses_saved.append(losses)

        # Force one timeout-only reset and one failure reset, with distinct old episode bookkeeping.
        env.reset(seed=42)
        env.wheel_net_angle[0:2] = 2.0
        env.spawn_yaw[0:2] = 0.123
        env.episode_length_buf[0] = env.max_episode_length-1
        pose = torch.cat((robot.data.root_pos_w.torch[1:2], robot.data.root_quat_w.torch[1:2]), dim=-1).clone()
        pose[:, 0] += 1.0  # Env1 fails drift containment; env0 should only time out.
        ids = torch.tensor([1], device=env.device, dtype=torch.int32)
        robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        with torch.no_grad():
            observation, _, terminated, truncated, _ = env.step(torch.zeros((args.audit_num_envs, 2), device=env.device))
        assert bool(truncated[0]) and not bool(terminated[0])
        assert bool(terminated[1]) and not bool(truncated[1])
        reset_record = {
            "seeded_old_wheel_net_angle_rad": 2.0, "seeded_old_spawn_yaw_rad": 0.123,
            "terminated_first_two": terminated[:2].cpu().tolist(),
            "truncated_first_two": truncated[:2].cpu().tolist(),
            "post_reset_wheel_net_angle_first_two_rad": env.wheel_net_angle[:2].cpu().tolist(),
            "post_reset_spawn_yaw_first_two_rad": env.spawn_yaw[:2].cpu().tolist(),
            "post_reset_joint_positions_first_two_rad": robot.data.joint_pos.torch[:2].cpu().tolist(),
            "post_reset_joint_velocities_first_two_rad_s": robot.data.joint_vel.torch[:2].cpu().tolist(),
            "post_reset_slew_target_first_two_rad": steering.processed_actions[:2].cpu().tolist(),
            "post_reset_observation_joint_channels_first_two": observation["policy"][:2, 5:9].cpu().tolist(),
        }
        torch.testing.assert_close(robot.data.joint_pos.torch[:2], torch.zeros_like(robot.data.joint_pos.torch[:2]))
        torch.testing.assert_close(robot.data.joint_vel.torch[:2], torch.zeros_like(robot.data.joint_vel.torch[:2]))
        torch.testing.assert_close(steering.processed_actions[:2], torch.zeros_like(steering.processed_actions[:2]))
        assert env.wheel_net_angle[1] == 0
        reset_record["timeout_bookkeeping_leak_reproduced"] = bool(env.wheel_net_angle[0] != 0)
        env.reset(seed=42)
        assert torch.count_nonzero(env.wheel_net_angle) == 0
        reset_record["explicit_full_reset_clears_bookkeeping"] = True

        action_array = torch.stack(actions_saved).cpu().numpy()
        observation_array = torch.stack(observations_saved).cpu().numpy()
        np.savez_compressed(OUT / f"{'dr' if dr else 'nominal'}_rollout.npz",
                            actions=action_array, observations=observation_array)
        result = {
            "domain_randomization": dr, "seed": 42, "num_envs": args.audit_num_envs,
            "ppo_updates": 2, "policy_steps": 32, "observations_shape_per_step": [args.audit_num_envs, 11],
            "physics_hz": 400, "observation_hz": 50, "policy_action_hz": 50,
            "call_counts_during_rollouts": counters,
            "action_min_max": [float(action_array.min()), float(action_array.max())],
            "observation_max_abs_by_channel": np.max(np.abs(observation_array), axis=(0, 1)).tolist(),
            "max_observation_vs_independent_rotation_matrix_error": state["max_observation_error"],
            "max_reward_vs_independent_formula_error": state["max_reward_error"],
            "white_noise_std_first_five_channels": torch.cat(noises).std(dim=0).cpu().tolist(),
            "finite_ppo_update_losses": losses_saved,
            "observed_termination_counts": dict(zip(env.termination_manager.active_terms, termination_counts.cpu().tolist())),
            "forced_reset_diagnostic": reset_record,
        }
        results.append(result)
        print("AUDIT " + json.dumps(result), flush=True)
        wrapped.close()
        env = None
    summary = {
        "execution_checks_passed": True,
        "training_ready_without_reset_fix": not any(r["forced_reset_diagnostic"]["timeout_bookkeeping_leak_reproduced"] for r in results),
        "results": results,
        "scope": "Two diagnostic PPO updates per mode, not a long training/performance evaluation; no runtime fixes applied",
    }
    (OUT / "integration_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    if env is not None:
        env.close()
    after = manifest()
    changed = {path: [digest, after.get(path)] for path, digest in before.items() if after.get(path) != digest}
    (OUT / "source_integrity.json").write_text(json.dumps({"checked_files": len(before), "changed_files": changed}, indent=2)+"\n")
    app.close()
