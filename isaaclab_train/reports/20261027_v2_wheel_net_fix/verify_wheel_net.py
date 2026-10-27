"""Headless v2 wheel-net accuracy, threshold, timing, wrap and reset regression."""

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
    cfg = gym.spec("Trackstand-Balance-v2").kwargs["env_cfg_entry_point"](domain_randomization=args.dr)
    cfg.scene.num_envs, cfg.seed, cfg.sim.device = 64, 42, args.device
    env = gym.make("Trackstand-Balance-v2", cfg=cfg).unwrapped
    env.reset(seed=42)
    robot, n, device = env.scene["robot"], env.num_envs, env.device
    rj = robot.find_joints("rear_wheel_joint")[0][0]
    tracker = env._wheel_net_tracker
    assert env.wheel_net_angle is tracker.angle_rad
    assert (env.step_dt, env.physics_dt) == (.02, .0025)
    counts = {"physics_updates": 0, "observations": 0, "termination_checks": 0, "steps": 0}
    state, rows, errors, legacy_errors = {}, [], [], []
    update, termination = tracker.update, env.termination_manager.compute
    obs = env.observation_manager._group_obs_term_cfgs["policy"][0]
    original_obs = obs.func

    def observed(*positional, **kwargs):
        counts["observations"] += 1
        return original_obs(*positional, **kwargs)

    def updated(position):
        counts["physics_updates"] += 1
        update(position)

    def checked_termination():
        current = tensor(robot.root_view.get_dof_positions())[:, rj]
        # Independent policy-end angle difference; every recorded block has less
        # than 2pi net motion, including the signed threshold/wrap fixtures.
        delta = torch.remainder(current - state["position"] + 2*torch.pi, 4*torch.pi) - 2*torch.pi
        expected = state["angle"] + delta
        actual = env.wheel_net_angle.clone()
        error = actual - expected
        torch.testing.assert_close(actual, expected, rtol=0, atol=6e-6)
        result = termination()
        expected_done = actual.abs() > 7.2
        torch.testing.assert_close(env.termination_manager.get_term("wheel_net"), expected_done)
        state["pre_reset"] = actual
        state["native_end"] = current.clone()
        counts["termination_checks"] += 1
        if state["record"]:
            speed = tensor(robot.root_view.get_dof_velocities())[:, rj]
            errors.append(error.cpu().numpy())
            legacy_errors.append((speed*.02 - delta).cpu().numpy())
            rows.append(torch.stack((state["angle"], delta, actual), -1).cpu().numpy())
        return result

    tracker.update, obs.func = updated, observed
    env.termination_manager.compute = checked_termination

    def step(action, record=False):
        before = counts.copy()
        state.update(position=tensor(robot.root_view.get_dof_positions())[:, rj].clone(),
                     angle=env.wheel_net_angle.clone(), record=record)
        result = env.step(action)
        counts["steps"] += 1
        assert counts["physics_updates"] - before["physics_updates"] == 8
        assert counts["observations"] - before["observations"] == 1
        assert counts["termination_checks"] - before["termination_checks"] == 1
        done = result[2] | result[3]
        assert torch.isfinite(result[0]["policy"]).all() and torch.isfinite(result[1]).all()
        torch.testing.assert_close(env.wheel_net_angle[done], torch.zeros_like(env.wheel_net_angle[done]))
        torch.testing.assert_close(tracker._previous_position_rad,
                                   tensor(robot.root_view.get_dof_positions())[:, rj], rtol=0, atol=0)
        return result

    # Same bounded synthetic action generator as the pre-fix audit. Earlier
    # wheel-net resets can change later trajectories, so this is not a paired
    # old/new survival benchmark.
    generator = torch.Generator(device=device).manual_seed(43)
    reset_count = 0
    with torch.no_grad():
        for _ in range(256):
            action = torch.rand((n, 2), device=device, generator=generator)*2 - 1
            result = step(action, record=True)
            reset_count += int((result[2] | result[3]).sum())

    # Positive/negative wheel-net crossings and a timeout in the same policy step.
    env.reset(seed=44)
    ids = torch.tensor([0, 1], device=device, dtype=torch.int32)
    position = tensor(robot.data.joint_pos)[:2].clone()
    speed = torch.zeros_like(position)
    speed[:, rj] = torch.tensor([20., -20.], device=device)
    robot.write_joint_state_to_sim_index(position=position, velocity=speed, env_ids=ids)
    env.wheel_net_angle[:3] = torch.tensor([7.19, -7.19, 3.3], device=device)
    env.episode_length_buf[2] = env.max_episode_length - 1
    action = torch.zeros((n, 2), device=device)
    action[:2, 1] = torch.tensor([20., -20.], device=device)/61.6
    with torch.no_grad():
        _, _, terminated, truncated, _ = step(action)
    boundary = state["pre_reset"][:3].cpu().tolist()
    assert boundary[0] > 7.2 and boundary[1] < -7.2, boundary
    assert env.termination_manager.get_term("wheel_net")[:2].all()
    assert terminated[:2].all() and truncated[2] and not terminated[2]
    assert not env.wheel_net_angle[:3].count_nonzero()
    assert not tensor(robot.data.joint_pos)[:3].count_nonzero()
    assert not tensor(robot.data.joint_vel)[:3].count_nonzero()

    # Explicit partial reset must preserve all continuing episode references.
    ids = torch.tensor([1, 3], device=device, dtype=torch.int32)
    before_angle = env.wheel_net_angle.clone()
    before_position = tracker._previous_position_rad.clone()
    before_yaw = env.spawn_yaw.clone()
    counts_before_reset = counts.copy()
    env.reset(seed=45, env_ids=ids)
    assert counts["physics_updates"] == counts_before_reset["physics_updates"]
    keep = torch.ones(n, device=device, dtype=torch.bool)
    keep[ids] = False
    torch.testing.assert_close(env.wheel_net_angle[keep], before_angle[keep], rtol=0, atol=0)
    torch.testing.assert_close(tracker._previous_position_rad[keep], before_position[keep], rtol=0, atol=0)
    torch.testing.assert_close(env.spawn_yaw[keep], before_yaw[keep], rtol=0, atol=0)
    assert not env.wheel_net_angle[ids].count_nonzero()
    assert not tracker._previous_position_rad[ids].count_nonzero()
    with torch.no_grad():
        step(torch.zeros((n, 2), device=device))

    # Actual PhysX native branch crossings, away from contacts, in both directions.
    env.reset(seed=46)
    pose = torch.cat((tensor(robot.data.root_pos_w)[:2], tensor(robot.data.root_quat_w)[:2]), -1).clone()
    pose[:, 2] = 3.
    ids = torch.tensor([0, 1], device=device, dtype=torch.int32)
    robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    position = tensor(robot.data.joint_pos)[:2].clone()
    position[:, rj] = torch.tensor([6.2, -6.2], device=device)
    speed = torch.zeros_like(position)
    speed[:, rj] = torch.tensor([20., -20.], device=device)
    robot.write_joint_state_to_sim_index(position=position, velocity=speed, env_ids=ids)
    tracker.reset(tensor(robot.root_view.get_dof_positions())[:, rj], ids)
    start = tensor(robot.root_view.get_dof_positions())[:2, rj].clone()
    with torch.no_grad():
        step(action)
    end, net = state["native_end"][:2].clone(), state["pre_reset"][:2].clone()
    assert (end - start).abs().gt(6.).all(), (start, end)
    assert net[0] > 0 and net[1] < 0 and net.abs().max() < 2., net

    env.reset(seed=47)
    assert not env.wheel_net_angle.count_nonzero()
    assert not tracker._previous_position_rad.count_nonzero()
    with torch.no_grad():
        step(torch.zeros((n, 2), device=device))

    error, legacy = np.stack(errors), np.stack(legacy_errors)
    np.savez_compressed(args.output / "capture.npz", angle_comparison=np.stack(rows),
                        new_error_rad=error, legacy_endpoint_increment_error_rad=legacy,
                        columns=np.array(["previous_net_rad", "actual_native_increment_rad", "current_net_rad"]))
    handle_id, manager = env._wheel_net_callback_handle.id, env.sim.physics_manager
    env.close()
    assert handle_id not in manager._callbacks
    env = None
    summary = {
        "passed": True, "command": sys.argv, "seed": 42, "num_envs": n,
        "domain_randomization": args.dr, "physics_hz": 400, "policy_observation_hz": 50,
        "synthetic_steps": 256, "synthetic_auto_resets": reset_count,
        "counts_including_explicit_reset_observations": counts,
        "new_angle_error_abs_max_rad": float(np.abs(error).max()),
        "new_angle_error_rms_rad": float(np.sqrt(np.mean(error**2))),
        "legacy_endpoint_increment_error_abs_max_rad": float(np.abs(legacy).max()),
        "legacy_endpoint_increment_error_rms_rad": float(np.sqrt(np.mean(legacy**2))),
        "first_step_pre_reset_net_angle_rad": boundary,
        "both_signed_thresholds_trigger_same_policy_step": True, "timeout_reset_verified": True,
        "explicit_partial_reset_preserves_continuing_refs": True,
        "native_wrap_start_rad": start.cpu().tolist(), "native_wrap_end_rad": end.cpu().tolist(),
        "native_wrap_net_rad": net.cpu().tolist(), "full_reset_has_no_teleport_increment": True,
        "callback_deregistered_on_close": True,
        "scope": "Wheel-net functional regression only; 400 Hz actuator stability and other audit findings not fixed.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("PASS " + json.dumps(summary), flush=True)
except BaseException:
    failure = {"passed": False, "traceback": traceback.format_exc()}
    (args.output / "failure.json").write_text(json.dumps(failure, indent=2))
    traceback.print_exc()
    raise
finally:
    if env is not None:
        env.close()
    app.close()
