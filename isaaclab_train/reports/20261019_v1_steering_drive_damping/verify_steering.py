"""Exercise the v1 steering contract in the real headless environment."""

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
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

import source.trackstand_rl.trackstand_rl.tasks  # noqa: F401

env = None
try:
    cfg = gym.spec("Trackstand-Balance-v1").kwargs["env_cfg_entry_point"]()
    cfg.scene.num_envs = 4
    cfg.sim.device = args.device
    cfg.seed = 42
    env = gym.make("Trackstand-Balance-v1", cfg=cfg).unwrapped
    env.reset(seed=42)
    assert abs(env.step_dt - 0.02) < 1e-12
    assert abs(env.physics_dt - 0.0025) < 1e-12
    assert env.cfg.decimation == 8
    term = env.action_manager.get_term("steering")
    robot = env.scene["robot"]
    joint = robot.find_joints("steering_joint")[0][0]
    process = term.process_actions
    apply = term.apply_actions
    processed = []
    writes = []
    before = []
    raw = []
    all_writes = []
    measurement = []

    def checked_process(actions):
        previous = term.processed_actions.clone()
        process(actions)
        desired = torch.clamp(actions * 1.57, -1.57, 1.57)
        expected = previous + (desired - previous).clamp(-0.478, 0.478)
        torch.testing.assert_close(term.processed_actions, expected, rtol=0, atol=2e-7)
        torch.testing.assert_close(term.raw_actions, actions, rtol=0, atol=0)
        before.append(previous[:, 0].cpu().numpy().copy())
        raw.append(actions[:, 0].cpu().numpy().copy())
        processed.append(term.processed_actions[:, 0].cpu().numpy().copy())

    def checked_apply():
        apply()
        target = robot.data.joint_pos_target.torch[:, joint]
        torch.testing.assert_close(target, term.processed_actions[:, 0], rtol=0, atol=0)
        writes.append(target.cpu().numpy().copy())

    term.process_actions = checked_process
    term.apply_actions = checked_apply
    with torch.inference_mode():
        for step in range(32):
            # Deliberately include out-of-range requests to exercise the clip;
            # these are synthetic inputs, not samples from the bounded Beta.
            request = 2.0 if step < 8 else -2.0 if step < 16 else (-1.0) ** step
            action = torch.zeros((4, 2), device=env.device)
            action[:, 0] = torch.tensor([request, -request, request / 2, -request / 2], device=env.device)
            writes.clear()
            env.step(action)
            assert len(writes) == 8, f"Expected 8 physics applications, got {len(writes)}"
            written = np.stack(writes)
            assert np.array_equal(written, np.broadcast_to(processed[-1], written.shape))
            all_writes.append(written)
            measurement.append(robot.data.joint_pos.torch[:, joint].cpu().numpy().copy())
    previous, targets = np.stack(before), np.stack(processed)
    delta = np.abs(targets - previous)
    assert np.max(np.abs(targets)) <= 1.570001
    assert np.max(delta) <= 0.478001
    np.savez_compressed(
        args.output / "steering_samples.npz", raw_actions=np.stack(raw), previous_target_rad=previous,
        processed_target_rad=targets, written_target_rad=np.stack(all_writes),
        measured_after_step_rad=np.stack(measurement),
    )
    summary = {
        "command": invocation, "passed": True, "task": "Trackstand-Balance-v1", "seed": 42,
        "domain_randomization": False, "num_envs": 4, "policy_steps": len(targets),
        "policy_rate_hz": 1 / env.step_dt, "physics_rate_hz": 1 / env.physics_dt,
        "physics_writes_per_policy_step": 8, "max_target_abs_rad": float(np.max(np.abs(targets))),
        "max_target_step_rad": float(np.max(delta)), "identical_target_across_all_eight_substeps": True,
        "scope": "synthetic action integration/contract test, not policy balance evaluation",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print("PASS: 50-Hz steering processing and held 400-Hz joint targets", flush=True)
finally:
    if env is not None:
        env.close()
    app.close()
