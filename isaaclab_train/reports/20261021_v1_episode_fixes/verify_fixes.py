"""Headless regressions for v1 timeout bookkeeping and symmetric action smoothing.

Uses controlled reset sentinels, live RewardManager checks, and two short PPO
updates per nominal/DR mode. No policy checkpoint or runtime source is written.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback

TRAIN = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAIN))


def manifest():
    paths = list((TRAIN / "source").rglob("*.py"))
    paths += list((TRAIN / "scripts").rglob("*.py")) + list((TRAIN / "tests").rglob("*.py"))
    paths += [TRAIN / "README.md", TRAIN.parent / "progress_record.md", TRAIN.parent / "progress_record.txt"]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


before = manifest()
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 5:
    parser.error("--num_envs must be at least 5 for the mixed reset/symmetry diagnostics")
app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import source.trackstand_rl.trackstand_rl.tasks  # noqa: F401
from source.trackstand_rl.trackstand_rl.tasks.manager_based.trackstand.agents.rsl_rl_ppo_beta_cfg import (
    TrackstandRLBetaPPORunnerCfg,
)
from source.trackstand_rl.trackstand_rl.tasks.manager_based.trackstand.trackstand_bike.trackstand_rl_env import (
    TrackstandRLEnvV1,
)

env = None
results = []
try:
    for dr in (False, True):
        cfg = gym.spec("Trackstand-Balance-v1").kwargs["env_cfg_entry_point"](domain_randomization=dr)
        cfg.scene.num_envs, cfg.seed, cfg.sim.device = args.num_envs, 42, args.device
        env = gym.make("Trackstand-Balance-v1", cfg=cfg).unwrapped
        assert isinstance(env, TrackstandRLEnvV1)
        assert env.step_dt == 0.02 and env.physics_dt == 0.0025
        robot = env.scene["robot"]
        steering = env.action_manager.get_term("steering")
        original_reward = env.reward_manager.compute
        state = {"reward_checks": 0, "max_reward_error": 0.0}

        def checked_reward(dt):
            change = env.action_manager.action - env.action_manager.prev_action
            rate = -torch.linalg.vector_norm(change, dim=-1)
            gravity_y = robot.data.projected_gravity_b.torch[:, 1]
            alive = (torch.rad2deg(torch.asin(gravity_y.abs().clamp(max=1))) < 30).float()
            distance_squared = ((robot.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]) ** 2).sum(-1)
            expected = dt * (5 * alive + 2 * torch.exp(-distance_squared / (2 * 0.25**2)) + 0.1 * rate)
            actual = original_reward(dt)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            state["reward_checks"] += 1
            state["max_reward_error"] = max(state["max_reward_error"], float((actual - expected).abs().max()))
            return actual

        env.reward_manager.compute = checked_reward
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        runner_cfg = handle_deprecated_rsl_rl_cfg(
            TrackstandRLBetaPPORunnerCfg(), importlib.metadata.version("rsl-rl-lib")
        )
        runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=None, device=args.device)
        algorithm = runner.alg
        observation = wrapped.get_observations()
        losses_saved, actions_saved = [], []
        algorithm.train_mode()
        for _ in range(2):
            with torch.no_grad():
                for _ in range(runner_cfg.num_steps_per_env):
                    actions = algorithm.act(observation)
                    assert torch.isfinite(actions).all() and torch.all(actions.abs() <= 1)
                    observation, reward, done, extras = wrapped.step(actions)
                    assert observation["policy"].shape == (args.num_envs, 11)
                    assert torch.isfinite(observation["policy"]).all() and torch.isfinite(reward).all()
                    algorithm.process_env_step(observation, reward, done, extras)
                    actions_saved.append(actions.clone())
                algorithm.compute_returns(observation)
            losses = algorithm.update()
            assert all(torch.isfinite(torch.tensor(value)).all() for value in losses.values())
            assert all(torch.isfinite(parameter).all() for parameter in algorithm.actor.parameters())
            losses_saved.append(losses)

        # Inspect the actual manager contribution, not just a direct function call.
        env.reset(seed=42)
        action = torch.zeros((args.num_envs, 2), device=env.device)
        action[3, 0], action[4, 1] = 1.0, 1.0
        with torch.no_grad():
            env.step(action)
        contributions = [dict(env.reward_manager.get_active_iterable_terms(index))["rate_penalty"][0]
                         * env.step_dt for index in (3, 4)]
        torch.testing.assert_close(torch.tensor(contributions), torch.full((2,), -0.002))

        # Three automatic resets in the same step: timeout only, failure only, both.
        # A fourth environment continues with its existing baseline and wheel state.
        env.reset(seed=42)
        env.wheel_net_angle[:4] = 2.0
        env.spawn_yaw[:4] = 0.123
        env.episode_length_buf[[0, 2]] = env.max_episode_length - 1
        pose = torch.cat((robot.data.root_pos_w.torch[1:3], robot.data.root_quat_w.torch[1:3]), dim=-1).clone()
        pose[:, 0] += 1.0
        robot.write_root_pose_to_sim_index(
            root_pose=pose, env_ids=torch.tensor([1, 2], device=env.device, dtype=torch.int32)
        )
        with torch.no_grad():
            observation, _, terminated, truncated, _ = env.step(torch.zeros_like(action))
        assert terminated[:4].tolist() == [False, True, True, False]
        assert truncated[:4].tolist() == [True, False, True, False]
        torch.testing.assert_close(env.wheel_net_angle[:3], torch.zeros_like(env.wheel_net_angle[:3]))
        torch.testing.assert_close(env.spawn_yaw[:3], env._heading_of()[:3])
        torch.testing.assert_close(env.spawn_yaw[3], torch.tensor(0.123, device=env.device))
        torch.testing.assert_close(env.wheel_net_angle[3], 2.0 + robot.data.joint_vel[3, env._drive_idx] * env.step_dt)
        for value in (robot.data.joint_pos.torch[:3], robot.data.joint_vel.torch[:3],
                      steering.processed_actions[:3], observation["policy"][:3, 5:9]):
            torch.testing.assert_close(value, torch.zeros_like(value))
        reset_record = {
            "seeded_old_wheel_angle_rad": 2.0, "seeded_old_spawn_yaw_rad": 0.123,
            "terminated_first_four": terminated[:4].tolist(),
            "truncated_first_four": truncated[:4].tolist(),
            "post_reset_wheel_angle_first_four_rad": env.wheel_net_angle[:4].tolist(),
            "post_reset_spawn_yaw_first_four_rad": env.spawn_yaw[:4].tolist(),
            "post_reset_actual_heading_first_three_rad": env._heading_of()[:3].tolist(),
            "physical_joint_state_slew_target_and_encoder_observations_reset_to_zero": True,
            "continuing_environment_preserved": True,
        }
        env.reset(seed=42)
        torch.testing.assert_close(env.wheel_net_angle, torch.zeros_like(env.wheel_net_angle))
        torch.testing.assert_close(env.spawn_yaw, env._heading_of())
        reset_record["explicit_full_reset_passed"] = True
        actions = torch.stack(actions_saved)
        result = {
            "domain_randomization": dr, "seed": 42, "num_envs": args.num_envs,
            "ppo_updates": 2, "ppo_policy_steps": 2 * runner_cfg.num_steps_per_env,
            "sampled_action_min_max": [actions.min().item(), actions.max().item()],
            "finite_ppo_losses": losses_saved, "live_reward_formula_checks": state,
            "unit_steer_and_unit_drive_change_weighted_reward": contributions,
            "forced_reset_diagnostic": reset_record,
        }
        results.append(result)
        print("VERIFIED " + json.dumps(result), flush=True)
        wrapped.close()
        env = None
    summary = {"all_checks_passed": True, "results": results,
               "scope": "Controlled reset regressions and short PPO smoke tests, not learned-policy performance"}
    (OUT / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    if env is not None:
        env.close()
    after = manifest()
    changed = {path: [digest, after.get(path)] for path, digest in before.items() if after.get(path) != digest}
    (OUT / "source_integrity.json").write_text(json.dumps(
        {"checked_files": len(before), "changed_files": changed}, indent=2
    ) + "\n")
    app.close()
