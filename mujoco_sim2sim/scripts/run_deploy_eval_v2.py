#!/usr/bin/env python3
"""Independent headless Trackstand-Balance-v2 evaluation (rear velocity PI)."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from newbike_mujoco_v2 import DeployedPolicy, build_env_from_mjcf  # noqa: E402
from newbike_mujoco_v2 import contract as c  # noqa: E402
from newbike_mujoco_v2.reporting import (  # noqa: E402
    _evaluation_output_base, _plot_episode_length_distribution, _plot_multi_torque_traces, write_metrics_log,
)
from newbike_mujoco_v2.viewer_envs import ViewerEnvGroup  # noqa: E402
from newbike_mujoco_v2.runtime import preserve_working_directory  # noqa: E402


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return result


def episode_quotas(total, num_envs):
    """Distribute a total budget fairly; no stream exceeds another by >1 episode."""
    quotient, remainder = divmod(total, num_envs)
    return np.full(num_envs, quotient, dtype=int) + (np.arange(num_envs) < remainder)


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    return value


@preserve_working_directory
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", "--scene", dest="mjcf", default=str(c.DEFAULT_MJCF))
    parser.add_argument("--pt", default=None)
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--policy_config", default=None,
                        help="Saved v2 env.yaml or matching JSON sidecar; discovered beside policy by default")
    parser.add_argument("--num_envs", type=positive_int, default=64)
    parser.add_argument("--num_steps", type=positive_int, default=3000,
                        help="Policy steps per env in fixed-duration mode; ignored with --episodes")
    parser.add_argument("--episodes", type=positive_int, default=None,
                        help="Exact total completions with fixed per-env quotas, rather than first-N-completions bias")
    parser.add_argument("--max_policy_steps", type=positive_int, default=None,
                        help="Finite global loop budget; defaults to num_steps or max episode quota * horizon")
    parser.add_argument("--steps", type=positive_int, default=c.MAX_STEPS,
                        help="Episode horizon; default 1500 = 30 s, lower values are diagnostic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--physics_hz", type=float, default=None,
                        help="Physics/PD Hz (default 1200); positive multiple of 200; policy=50, I=200 Hz")
    parser.add_argument("--dr", action="store_true")
    parser.add_argument("--pure_survival", action="store_true")
    parser.add_argument("--velocity_limit_mode", choices=("project", "none"), default="project")
    parser.add_argument("--base_inertia", type=float, default=1., help="Diagnostic override; keep 1 for reference")
    parser.add_argument("--output_dir", default=str(ROOT / "logs/v2"))
    args = parser.parse_args()
    try:
        physics_dt, substeps = c.physics_timing(args.physics_hz)
    except ValueError as error:
        parser.error(str(error))
    args.physics_hz = 1.0 / physics_dt
    if args.steps > c.MAX_STEPS:
        parser.error("--steps cannot exceed the v2 1500-step/30-s horizon")
    if not np.isfinite(args.base_inertia) or args.base_inertia <= 0:
        parser.error("--base_inertia must be positive and finite")
    quotas = episode_quotas(args.episodes, args.num_envs) if args.episodes is not None else None
    default_budget = int(quotas.max()) * args.steps if quotas is not None else args.num_steps
    budget = args.max_policy_steps if args.max_policy_steps is not None else default_budget
    if quotas is None:
        budget = min(budget, args.num_steps)
    # Resolve caller-relative inputs before isolating MuJoCo's native log CWD.
    args.mjcf = str(Path(args.mjcf).expanduser().resolve())
    for name in ("pt", "onnx", "policy_config"):
        if getattr(args, name) is not None:
            setattr(args, name, str(Path(getattr(args, name)).expanduser().resolve()))
    caller_cwd = os.getcwd()
    output = Path(args.output_dir).expanduser().resolve()
    args.output_dir = str(output)
    output.mkdir(parents=True, exist_ok=True)
    policy = DeployedPolicy(onnx_path=args.onnx,
                            pt_path=args.pt or (None if args.onnx else c.DEFAULT_CHECKPOINT),
                            config_path=args.policy_config)
    base = _evaluation_output_base(str(output), policy.path, args.dr)
    runtime_dir = Path(base + "_runtime")
    runtime_dir.mkdir()
    os.chdir(runtime_dir)  # Native MUJOCO_LOG.TXT never overwrites a v1 log.
    first, _ = build_env_from_mjcf(args.mjcf, args.seed, args.dr,
                                  velocity_limit_mode=args.velocity_limit_mode, physics_hz=args.physics_hz)
    group = ViewerEnvGroup(first, args.num_envs, args.seed)
    for env in group.envs:
        env.MAX_STEPS = args.steps
        if args.pure_survival:
            env.enabled_terms = {"time_out", "fallen"}
        if args.base_inertia != 1.:
            env.model.body_inertia[env.model.body("base_link").id] *= args.base_inertia
            env.refresh_dr_nominal()
    print(f"[INFO] {c.TASK}, {policy.backend}: {policy.path}", flush=True)
    print(f"[INFO] policy=50 Hz, physics/PD={args.physics_hz:g} Hz, rear I=200 Hz; "
          f"{substeps} physics / 4 integral updates per policy step", flush=True)
    print(f"[INFO] {args.num_envs} envs, seed={args.seed}, DR={args.dr}; terms={sorted(first.enabled_terms)}", flush=True)
    print(f"[WARN] speed mode={args.velocity_limit_mode}: not a native PhysX-equivalent soft constraint", flush=True)
    if args.steps != c.MAX_STEPS or args.base_inertia != 1. or args.physics_hz != 1. / c.PHYSICS_DT:
        print("[WARN] diagnostic horizon, inertia or physics-rate override differs from reference v2", flush=True)
    observations = group.reset()
    completed = np.zeros(args.num_envs, dtype=int)
    records, episode_records, resets = [], [], []
    trace_names = ("observation", "raw_action", "held_steering_target_rad", "held_drive_target_rad_s",
                   "position_rad", "velocity_rad_s", "physics_motor_torque_nm", "physics_computed_torque_nm",
                   "physics_drive_integral_nm", "physics_drive_velocity_rad_s", "physics_drive_error_rad_s",
                   "drive_integral_nm", "reward", "termination_flags", "active_mask")
    traces = {name: [] for name in trace_names}
    started, step = time.monotonic(), 0
    while step < budget:
        active = completed < quotas if quotas is not None else np.ones(args.num_envs, dtype=bool)
        if not active.any():
            break
        actions = np.zeros((args.num_envs, 2), dtype=np.float32)
        actions[active] = policy.act_batch(observations[active])
        traces["observation"].append(np.where(active[:, None], observations, np.nan))
        traces["raw_action"].append(np.where(active[:, None], actions, np.nan))
        traces["active_mask"].append(active.copy())
        done = np.zeros(args.num_envs, dtype=bool)
        for i in np.flatnonzero(active):
            observations[i], done[i] = group.envs[i].step(actions[i], observe_terminal=False)
        values = dict(
            held_steering_target_rad=[env._steer_cmd for env in group.envs],
            held_drive_target_rad_s=[env._drive_cmd for env in group.envs],
            position_rad=[env.data.qpos[[env._steer_qadr, env._rear_qadr]].copy() for env in group.envs],
            velocity_rad_s=[env.data.qvel[env._dofs].copy() for env in group.envs],
            physics_motor_torque_nm=[env.physics_torques.copy() for env in group.envs],
            physics_computed_torque_nm=[env.physics_computed_torques.copy() for env in group.envs],
            physics_drive_integral_nm=[env.physics_drive_integral_nm.copy() for env in group.envs],
            physics_drive_velocity_rad_s=[env.physics_drive_velocity_rad_s.copy() for env in group.envs],
            physics_drive_error_rad_s=[env.physics_drive_error_rad_s.copy() for env in group.envs],
            drive_integral_nm=[env.integral.torque_nm for env in group.envs],
            reward=[env.last_reward for env in group.envs],
            termination_flags=[[env.termination_flags[name] for name in c.TERMINATIONS] for env in group.envs],
        )
        for key, value in values.items():
            array = np.asarray(value, dtype=float)
            mask = active.reshape((-1,) + (1,) * (array.ndim - 1))
            traces[key].append(np.where(mask, array, np.nan))
        step += 1
        for i in np.flatnonzero(done):
            env, stats = group.envs[i], group.envs[i].stats
            records.append((stats.length, stats.death, stats))
            episode_records.append(dict(env_id=int(i), episode_index=int(completed[i]), end_policy_step=step,
                                        sampled_dr=json_value(env.dr), **asdict(stats)))
            result = "+".join(name for name, value in stats.terminations.items() if value)
            print(f"env {i:3d}: episode {completed[i]+1:3d}: {stats.length:5d} steps "
                  f"({stats.length*c.CONTROL_DT:5.2f} s)  {result}", flush=True)
            completed[i] += 1
            resets.append((step, int(i)))
            observations[i] = env.reset()
    arrays = {key: np.asarray(value) for key, value in traces.items()}
    np.savez_compressed(base+"_evaluation.npz", **arrays, motor_order=np.array(["steering", "rear"]),
                        termination_names=np.array(c.TERMINATIONS), reset_events=np.asarray(resets).reshape(-1, 2),
                        control_dt_s=c.CONTROL_DT, physics_dt_s=first.physics_dt, decimation=first.decimation,
                        drive_integral_rate_hz=c.DRIVE_INTEGRAL_RATE_HZ)
    timing = dict(policy_hz=50., physics_hz=args.physics_hz, pd_hz=args.physics_hz,
                  integral_hz=c.DRIVE_INTEGRAL_RATE_HZ, decimation=substeps)
    write_metrics_log(records, base+"_summary.log", args.num_envs, timing=timing)
    _plot_episode_length_distribution(records, base, policy.path, args.dr)
    tau = arrays["physics_motor_torque_nm"][:, :, -1, ::-1]
    _plot_multi_torque_traces(tau, resets, c.DRIVE_EFFORT_LIMIT, c.STEER_EFFORT_LIMIT, base, args.seed)
    fulfilled = bool(np.all(completed >= quotas)) if quotas is not None else step == args.num_steps
    metadata = dict(task=c.TASK, policy=policy.path, policy_backend=policy.backend,
                    policy_contract=policy.contract_metadata,
                    policy_sha256=hashlib.sha256(Path(policy.path).read_bytes()).hexdigest(),
                    arguments=vars(args), caller_cwd=caller_cwd, runtime_dir=str(runtime_dir),
                    completed_policy_steps=step,
                    environment_transitions=int(np.sum(arrays["active_mask"])),
                    wall_time_s=time.monotonic()-started, **timing,
                    sample_design="fixed per-environment episode quotas" if quotas is not None else "fixed policy-step budget",
                    episode_quotas=None if quotas is None else quotas.tolist(),
                    completed_per_env=completed.tolist(),
                    in_progress_lengths_steps=[env.stats.length for env in group.envs],
                    termination_names=list(c.TERMINATIONS), episodes=episode_records,
                    limitations=["MuJoCo/PhysX contact, passive friction and solver dynamics differ",
                                 "project hard velocity cap is stricter than PhysX soft constraint; none omits it",
                                 "MuJoCo has one sliding coefficient; dynamic draw is recorded but not applied"],
                    completed=fulfilled, stop_reason="requested_budget_complete" if fulfilled else "finite_step_budget_exhausted")
    Path(base+"_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    print(f"[INFO] evaluation {'complete' if fulfilled else 'partial'}: {base}", flush=True)
    return 0 if fulfilled else 2


if __name__ == "__main__":
    raise SystemExit(main())
