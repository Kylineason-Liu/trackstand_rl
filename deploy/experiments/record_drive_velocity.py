#!/usr/bin/env python3
"""PC only: record one policy episode's ACTUAL rear velocity for C1 replay.

Run with Isaac Lab's Python. Does not change the task, actuator or checkpoint.
The 400-Hz simulation state is saved separately from the 50-Hz held reference.
V1 model_2500 outputs TORQUE; its measured velocity becomes the C1 target.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHECKPOINT = ROOT / "isaaclab_train/logs/rsl_rl/trackstand_rl_rsl/2026-10-21_11-18-29/model_2500.pt"
FIELDS = ("time_s", "velocity_rad_s", "position_rad")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_capture(output, samples, policy_dt, metadata):
    """Keep native samples and exact policy-boundary states; never interpolate."""
    if len(samples) < 2 or policy_dt <= 0:
        raise ValueError("at least two samples and a positive policy period required")
    for previous, current in zip(samples, samples[1:]):
        if current[0] <= previous[0]:
            raise ValueError("simulation sample times must increase strictly")
    if abs(samples[0][0]) > 1e-9 or not all(math.isfinite(v) for row in samples for v in row):
        raise ValueError("capture must start at zero and contain only finite states")
    duration = samples[-1][0]
    boundary_count = round(duration / policy_dt)
    if boundary_count < 1 or abs(boundary_count * policy_dt - duration) > 1e-8:
        raise ValueError("capture must end on a policy boundary")
    reference = [row for row in samples if abs(row[0] / policy_dt - round(row[0] / policy_dt)) < 1e-7]
    if len(reference) != boundary_count + 1:
        raise ValueError("missing policy-boundary states; refusing to interpolate")
    output = Path(output)
    native, manifest = output.with_name(output.stem + "_simulation.csv"), output.with_suffix(".json")
    if any(p.exists() for p in (output, native, manifest)):
        raise FileExistsError("refusing to overwrite an existing capture")
    output.parent.mkdir(parents=True, exist_ok=True)
    for path, rows in ((output, reference), (native, samples)):
        with path.open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(FIELDS)
            writer.writerows(rows)
    metadata.update(capture_completed=True, duration_s=duration, reference_hz=1 / policy_dt,
                    reference_samples=len(reference),
                    native_samples=len(samples), reference_sha256=sha(output), native_sha256=sha(native),
                    native_file=native.name, positive_direction="rear wheel forward",
                    reference_meaning="actual simulated rear velocity sampled at policy boundaries, NOT raw action",
                    position_meaning="joint displacement unwrapped at the physics rate from the initial position",
                    timing="initial state t=0; physics responses at step END, before any automatic reset",
                    maximum_absolute_velocity_rad_s=max(abs(row[1]) for row in samples))
    manifest.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--from-capture", type=Path,
                        help="Export an existing audited motor_samples.npz instead of launching Isaac Sim")
    parser.add_argument("--duration", type=float, default=10., help="Seconds from spawn; stop sooner if episode ends")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--env-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dr", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=HERE / "replay_targets/2500_isaac_velocity_startup.csv")
    args = parser.parse_args(argv)
    if (not math.isfinite(args.duration) or args.duration <= 0
            or abs(args.duration * 50 - round(args.duration * 50)) > 1e-7):
        parser.error("--duration must be positive and a multiple of 0.02 s")
    if not 0 <= args.env_index < args.num_envs:
        parser.error("--env-index must select one of --num-envs")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    args.output = args.output.resolve()
    if args.output.suffix != ".csv":
        parser.error("--output must end in .csv")
    if any(p.exists() for p in (args.output, args.output.with_suffix(".json"),
                               args.output.with_name(args.output.stem + "_simulation.csv"),
                               args.output.with_name(args.output.stem + "_env.yaml"))):
        parser.error("output already exists; choose a new --output")
    return args


def export_existing(args):
    """Export measured pre-reset physics states from the existing evaluation archive."""
    import numpy as np
    capture = args.from_capture.resolve()
    metadata = json.loads(capture.with_name("experiment.json").read_text())
    if not metadata.get("capture_completed") or metadata["checkpoint_sha256"] != sha(args.checkpoint):
        raise ValueError("capture is incomplete or belongs to another checkpoint")
    if metadata["motor_order"] != ["steering", "rear"]:
        raise ValueError("unexpected captured motor ordering")
    index = args.env_index
    policy_dt, physics_dt = metadata["policy_dt_s"], metadata["physics_dt_s"]
    if abs(policy_dt - .02) > 1e-9 or not 0 <= index < metadata["num_envs"]:
        raise ValueError("expected a 50-Hz capture and a valid environment index")
    requested = round(args.duration / policy_dt)
    with np.load(capture, allow_pickle=False) as data:
        done = data["done"][:, index]
        steps = min(requested, len(done), next((i + 1 for i, value in enumerate(done) if value), len(done)))
        # The first BEFORE sample is the initial state, not the first response at 2.5 ms.
        initial_q = float(data["position_before_rad"][0, 0, index, 1])
        samples = [(0., float(data["velocity_before_rad_s"][0, 0, index, 1]), 0.)]
        positions = data["position_after_rad"][:steps, :, index, 1].reshape(-1)
        positions = np.unwrap(np.r_[initial_q, positions])[1:]
        velocities = data["velocity_after_rad_s"][:steps, :, index, 1].reshape(-1)
        samples.extend(((i + 1) * physics_dt, float(v), float(q) - initial_q)
                       for i, (q, v) in enumerate(zip(positions, velocities)))
        terminated = bool(done[steps - 1])
    result = write_capture(args.output, samples, policy_dt, dict(
        task="Trackstand-Balance-v1", model=args.checkpoint.name, checkpoint_sha256=sha(args.checkpoint),
        seed=metadata["seed"], dr=metadata["dr"], num_envs=metadata["num_envs"], environment=index, episode=1,
        physics_hz=1 / physics_dt, source_capture=str(capture), source_capture_sha256=sha(capture),
        source_experiment=metadata, recorder_sha256=sha(__file__), exported_at=datetime.now().isoformat(),
        requested_duration_s=args.duration, episode_terminated=terminated,
        policy_drive_action_meaning="torque: raw Beta drive action * 6.28 Nm",
    ))
    print(f"Exported actual recorded simulation t=0..{result['duration_s']:g} s: {args.output}")
    print(f"Speed peak {result['maximum_absolute_velocity_rad_s']:.6g} rad/s; no interpolation/scaling/clipping")
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.from_capture:
        return export_existing(args)
    train = ROOT / "isaaclab_train"
    sys.path.insert(0, str(train))
    # Isaac Sim must start before importing Torch, Gym or the task extension.
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device=args.device).app
    env = None
    try:
        from importlib.metadata import version
        import gymnasium as gym
        import torch
        import warp as wp
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
        from isaaclab_tasks.utils import load_cfg_from_registry
        from isaaclab.utils.io import dump_yaml
        __import__("source.trackstand_rl.trackstand_rl.tasks")  # Register project tasks.
        from scripts.domain_randomization import configure_domain_randomization

        def tensor(value):
            if isinstance(value, torch.Tensor):
                return value
            return value.torch if hasattr(value, "torch") else wp.to_torch(value)

        task = "Trackstand-Balance-v1"
        cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
        agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        configure_domain_randomization(cfg, args.dr)
        cfg.scene.num_envs, cfg.seed, cfg.sim.device = args.num_envs, args.seed, args.device
        cfg.log_dir = str(args.output.parent)
        agent.seed, agent.device = args.seed, args.device
        agent = handle_deprecated_rsl_rl_cfg(agent, version("rsl-rl-lib"))
        env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=agent.clip_actions)
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=args.device)
        runner.load(str(args.checkpoint.resolve()))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        base, index = env.unwrapped, args.env_index
        robot = base.scene["robot"]
        joint = robot.find_joints("rear_wheel_joint")[0][0]
        previous_q = float(tensor(robot.data.joint_pos)[index, joint])
        displacement = 0.
        policy_dt, physics_dt = base.step_dt, base.physics_dt
        if abs(policy_dt - .02) > 1e-9:
            raise RuntimeError("expected the v1 50-Hz policy contract")
        samples, active, tick = [], False, 0

        def sample(t):
            nonlocal previous_q, displacement
            q, v = tensor(robot.data.joint_pos), tensor(robot.data.joint_vel)
            current_q = float(q[index, joint])
            displacement += math.atan2(math.sin(current_q - previous_q), math.cos(current_q - previous_q))
            previous_q = current_q
            return (t, float(v[index, joint]), displacement)

        original_update = base.scene.update

        def update(*a, **kw):
            nonlocal tick
            original_update(*a, **kw)
            if active:
                tick += 1
                samples.append(sample(tick * physics_dt))

        base.scene.update = update
        samples.append(sample(0.))
        observations = env.get_observations()
        terminated = False
        with torch.inference_mode():
            for step in range(round(args.duration / policy_dt)):
                actions = policy(observations)
                previous_tick = tick
                active = True
                try:
                    observations, _, dones, _ = env.step(actions)
                finally:
                    active = False
                if tick - previous_tick != cfg.decimation:
                    raise RuntimeError("unexpected extra/missing physics updates; refusing a mistimed capture")
                if (step + 1) % 100 == 0:
                    print(f"Recorded {samples[-1][0]:g}/{args.duration:g} simulated seconds", flush=True)
                if bool(dones[index]):
                    terminated = True
                    break  # The update hook already retained the PRE-reset terminal state.
        base.scene.update = original_update
        args.output.parent.mkdir(parents=True, exist_ok=True)
        dump_yaml(str(args.output.with_name(args.output.stem + "_env.yaml")), cfg)
        result = write_capture(args.output, samples, policy_dt, dict(
            task=task, model=args.checkpoint.name, checkpoint=str(args.checkpoint.resolve()),
            checkpoint_sha256=sha(args.checkpoint), recorder_sha256=sha(__file__),
            recorded_at=datetime.now().isoformat(), seed=args.seed, dr=args.dr,
            num_envs=args.num_envs, environment=index, episode=1, physics_hz=1 / physics_dt,
            requested_duration_s=args.duration, episode_terminated=terminated,
            termination_terms=[name for name in base.termination_manager.active_terms
                               if terminated and bool(base.termination_manager.get_term(name)[index])],
            policy_drive_action_meaning="torque: raw Beta drive action * 6.28 Nm",
        ))
        print(f"Recorded {result['duration_s']:g} s from spawn: {args.output}")
        print(f"Actual speed peak {result['maximum_absolute_velocity_rad_s']:.6g} rad/s; no scaling/clipping")
        return 0
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    sys.exit(main())
