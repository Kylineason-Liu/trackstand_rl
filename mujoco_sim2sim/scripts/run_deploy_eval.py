#!/usr/bin/env python3
"""Headless v1 evaluation using exactly the viewer's environment and Beta loader.

Default64env x3000policy steps, seed42, nominal/no-DR-trained75300. Explicit
--episodes selects an episode-count budget instead. No original models,
checkpoints or previous reports are modified.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newbike_mujoco import DeployedPolicy, build_env_from_mjcf  # noqa: E402
from newbike_mujoco import contract as c  # noqa: E402
from newbike_mujoco.reporting import (  # noqa: E402
    _evaluation_output_base, _plot_episode_length_distribution, _plot_multi_torque_traces, write_metrics_log,
)
from newbike_mujoco.viewer_envs import ViewerEnvGroup  # noqa: E402


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be at least1")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", "--scene", dest="mjcf", default=str(c.DEFAULT_MJCF))
    parser.add_argument("--pt", default=None)
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--num_envs", type=positive_int, default=64)
    parser.add_argument("--num_steps", type=positive_int, default=3000,
                        help="Policy steps per env; ignored with --episodes")
    parser.add_argument("--episodes", type=positive_int, default=None, help="Record exactly N completed episodes")
    parser.add_argument("--steps", type=positive_int, default=c.MAX_STEPS,
                        help="Episode horizon; default1500 matches v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--physics_hz", type=float, default=None,
                        help="Physics and PD rate in Hz (default: 400); multiple of 50; policy stays at 50 Hz.")
    parser.add_argument("--dr", action="store_true")
    parser.add_argument("--pure_survival", action="store_true")
    parser.add_argument("--velocity_limit_mode", choices=("project", "none"), default="project")
    parser.add_argument("--base_inertia", type=float, default=1.,
                        help="Non-v1 diagnostic override; keep1 for comparison")
    parser.add_argument("--output_dir", default=str(ROOT / "logs"))
    args = parser.parse_args()
    try:
        physics_dt, substeps = c.physics_timing(args.physics_hz)
    except ValueError as error:
        parser.error(str(error))
    args.physics_hz = 1.0 / physics_dt
    if args.steps > c.MAX_STEPS:
        parser.error("--steps cannot exceed the v1 1500-step/30-s horizon")
    if not np.isfinite(args.base_inertia) or args.base_inertia <= 0:
        parser.error("--base_inertia must be positive and finite")
    policy = DeployedPolicy(onnx_path=args.onnx,
                            pt_path=args.pt or (None if args.onnx else c.DEFAULT_CHECKPOINT))
    first, _ = build_env_from_mjcf(args.mjcf, args.seed, args.dr, velocity_limit_mode=args.velocity_limit_mode,
                                  physics_hz=args.physics_hz)
    group = ViewerEnvGroup(first, args.num_envs, args.seed)
    for env in group.envs:
        env.MAX_STEPS = args.steps
        if args.pure_survival:
            env.enabled_terms = {"time_out", "fallen"}
        if args.base_inertia != 1.:
            env.model.body_inertia[env.model.body("base_link").id] *= args.base_inertia
            env.refresh_dr_nominal()
    print(f"[INFO] {c.TASK}, {policy.backend}: {policy.path}", flush=True)
    print(f"[INFO] policy={1.0/c.CONTROL_DT:g} Hz, physics/PD={args.physics_hz:g} Hz "
          f"({substeps} substeps/policy step)", flush=True)
    if not np.isclose(physics_dt, .0025, rtol=0., atol=1e-12):
        print("[WARN] non-default physics/PD rate; Isaac v1 reference remains 400 Hz", flush=True)
    print(f"[INFO] {args.num_envs}env, seed={args.seed}, DR={args.dr}; terms={sorted(first.enabled_terms)}", flush=True)
    print(f"[WARN] speed mode={args.velocity_limit_mode}: NOT a native PhysX-equivalent soft constraint", flush=True)
    if args.steps != c.MAX_STEPS or args.base_inertia != 1.:
        print("[WARN] non-v1 diagnostic horizon/inertia override", flush=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base = _evaluation_output_base(str(output), policy.path, args.dr)
    observation = group.reset()
    records, resets, episode_ids = [], [], np.zeros(args.num_envs, dtype=int)
    traces = {name: [] for name in ("observation", "raw_action", "held_steering_target_rad", "position_rad",
                                   "velocity_rad_s", "physics_motor_torque_nm", "reward", "termination_flags")}
    started, step = time.monotonic(), 0
    while True:
        if (args.episodes is not None and len(records) >= args.episodes) or (
                args.episodes is None and step >= args.num_steps):
            break
        actions = policy.act_batch(observation)
        traces["observation"].append(observation.copy())
        traces["raw_action"].append(actions.copy())
        observation, done, _ = group.step(actions)
        traces["held_steering_target_rad"].append([env._steer_cmd for env in group.envs])
        traces["position_rad"].append([env.data.qpos[[env._steer_qadr, env._rear_qadr]].copy() for env in group.envs])
        traces["velocity_rad_s"].append([env.data.qvel[env._dofs].copy() for env in group.envs])
        traces["physics_motor_torque_nm"].append(np.stack([env.physics_torques.copy() for env in group.envs]))
        traces["reward"].append([env.last_reward for env in group.envs])
        traces["termination_flags"].append([[env.termination_flags[name] for name in c.TERMINATIONS]
                                             for env in group.envs])
        step += 1
        for index in np.flatnonzero(done):
            env, st = group.envs[index], group.envs[index].stats
            # Episode budget chooses the first N completions in batch order.
            if args.episodes is None or len(records) < args.episodes:
                records.append((st.length, st.death, st))
                result = "+".join(name for name, value in st.terminations.items() if value)
                print(f"env {index:3d}: episode {episode_ids[index]+1:3d}: {st.length:5d} steps "
                      f"({st.length*c.CONTROL_DT:5.2f} s)  {result}", flush=True)
            resets.append((step, int(index)))
            episode_ids[index] += 1
            observation[index] = env.reset()
    arrays = {key: np.asarray(value) for key, value in traces.items()}
    np.savez_compressed(base+"_evaluation.npz", **arrays, motor_order=np.array(["steering", "rear"]),
                        termination_names=np.array(c.TERMINATIONS), reset_events=np.asarray(resets).reshape(-1, 2),
                        control_dt_s=c.CONTROL_DT, physics_dt_s=first.physics_dt, decimation=first.decimation)
    write_metrics_log(records, base+"_summary.log", args.num_envs,
                      timing=dict(policy_hz=1.0/c.CONTROL_DT, physics_hz=args.physics_hz,
                                  pd_hz=args.physics_hz, decimation=substeps))
    _plot_episode_length_distribution(records, base, policy.path, args.dr)
    # Existing torque graph samples the last physics torque at each policy boundary.
    # The archive above retains ALL actual substep actuator outputs per interval.
    tau = arrays["physics_motor_torque_nm"][:, :, -1, ::-1]
    _plot_multi_torque_traces(tau, resets, c.DRIVE_EFFORT_LIMIT, c.STEER_EFFORT_LIMIT, base, args.seed)
    metadata = dict(task=c.TASK, policy=policy.path, policy_backend=policy.backend,
                    policy_sha256=hashlib.sha256(Path(policy.path).read_bytes()).hexdigest(),
                    arguments=vars(args), completed_policy_steps=step, wall_time_s=time.monotonic()-started,
                    policy_hz=1.0/c.CONTROL_DT, physics_hz=1.0/first.physics_dt, pd_hz=1.0/first.physics_dt,
                    decimation=first.decimation,
                    termination_names=list(c.TERMINATIONS), episodes=[asdict(record[2]) for record in records],
                    limitations=["MuJoCo/PhysX contact and solver dynamics differ",
                                 "project hard velocity cap is stricter than PhysX soft constraint; none omits it",
                                 "MuJoCo has one sliding coefficient; dynamic draw is in runtime env.dr, not applied"],
                    completed=True)
    Path(base+"_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    print(f"[INFO] evaluation complete: {base}", flush=True)


if __name__ == "__main__":
    main()
