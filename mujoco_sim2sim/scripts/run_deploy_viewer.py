#!/usr/bin/env python3
"""View Trackstand-Balance-v1: 11 observations, Beta means, explicit motor PD.

Default: verified trackstand0916 MJCF, no-DR-trained model75300, 64 independent
bikes. --dr enables reset randomization; all six v1 terminations are enabled
unless --pure_survival selects timeout/fallen only. zero_agent disables motor
torques; free_fall additionally removes passive losses and does not auto-reset.

On closing deploy_sim, save completed-episode histogram, metrics and actuator
torque traces under logs/YYYYMMDDHHMM_<model>_<dr|no_dr>_<type>.
MuJoCo contact/velocity-limit approximations are explicit in README.md.
"""

import argparse
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")  # headless: render to file only, never to a display
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer  # not auto-imported by `import mujoco`
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from newbike_mujoco import BikeBalanceEnv, DeployedPolicy  # noqa: E402
from newbike_mujoco import contract as c  # noqa: E402
from newbike_mujoco.viewer_envs import ViewerEnvGroup  # noqa: E402
from newbike_mujoco.contract import DEFAULT_CHECKPOINT, DEFAULT_MJCF, TERMINATIONS  # noqa: E402
from newbike_mujoco.model import augment_trackstand_model, build_env_from_mjcf  # noqa: E402
from newbike_mujoco.reporting import (  # noqa: E402
    write_metrics_log, _evaluation_output_base, _plot_episode_length_distribution,
    _plot_torque_traces, _plot_multi_torque_traces,
)


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["deploy_sim", "zero_agent", "free_fall"], default="deploy_sim",
                    help="deploy_sim: run the deployed policy; zero_agent: no control + auto-reset; "
                         "free_fall: no control, no reset.")
    ap.add_argument("--lock_steer", action="store_true",
                    help="zero_agent only: weld the steering at 0 so the bike topples like "
                         "a rigid body instead of caster-weaving.")
    ap.add_argument("--mjcf", default=None, help="Bike MJCF (default: verified trackstand0916 model).")
    ap.add_argument("--scene", default=None, help="Alias for --mjcf; do not supply both.")
    ap.add_argument("--onnx", default=None,
                    help="Exported v1 Beta mean policy, 11 inputs / 2 actions.")
    ap.add_argument("--pt", default=None,
                    help="V1 Beta checkpoint (default: 20261021 no-DR model75300).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--physics_hz", type=float, default=None,
                    help="Physics and PD rate in Hz (default: 400); multiple of 50; policy stays at 50 Hz.")
    ap.add_argument("--num_envs", type=_positive_int, default=64,
                    help="Number of independent bikes shown in one viewer (default: 64); "
                         "use 1 for the original single-bike view. CPU physics, seeds seed+i.")
    ap.add_argument("--dr", action="store_true",
                    help="enable the domain-randomization terms (the sim's BalanceEventsCfg): "
                         "per-episode mass/COM/tire-friction/actuator-gain randomization, "
                         "DR initial state (roll U(+-0.10) rad, roll-rate U(+-0.25) rad/s), "
                         "and IMU turn-on residuals.")
    ap.add_argument("--pure_survival", action="store_true", help="Only time_out/fallen; default uses all six v1 terms.")
    ap.add_argument("--velocity_limit_mode", choices=("project", "none"), default="project",
                    help="project: hard speed projection23.9/61.6 (NOT PhysX-equivalent); none: diagnostic only.")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="Playback speed (1.0 = real time, 0.5 = half speed).")
    ap.add_argument("--base_inertia", type=float, default=1.0,
                    help="Non-v1 diagnostic override; keep1 for normal comparison.")
    args = ap.parse_args()
    try:
        physics_dt, substeps = c.physics_timing(args.physics_hz)
    except ValueError as error:
        ap.error(str(error))
    args.physics_hz = 1.0 / physics_dt
    if args.mjcf and args.scene:
        ap.error("Use --mjcf or --scene, not both")
    if not np.isfinite(args.speed) or args.speed <= 0 or not np.isfinite(args.base_inertia) or args.base_inertia <= 0:
        ap.error("--speed and --base_inertia must be positive and finite")
    if args.lock_steer and args.task != "zero_agent":
        ap.error("--lock_steer is only valid for --task zero_agent")
    env, _ = build_env_from_mjcf(args.mjcf or args.scene or DEFAULT_MJCF, args.seed, enable_dr=args.dr,
                                velocity_limit_mode=args.velocity_limit_mode, physics_hz=args.physics_hz)
    group = ViewerEnvGroup(env, args.num_envs, args.seed)
    print(f"[INFO] policy={1.0/c.CONTROL_DT:g} Hz, physics/PD={args.physics_hz:g} Hz "
          f"({substeps} substeps/policy step)")
    if not np.isclose(physics_dt, .0025, rtol=0., atol=1e-12):
        print("[WARN] non-default physics/PD rate; Isaac v1 reference remains 400 Hz")
    print(f"[INFO] environments: {args.num_envs}; independent seeds {args.seed}.."
          f"{args.seed + args.num_envs - 1}; CPU physics, batched policy inference")
    dr_state = ("ON (per-episode mass/COM/friction/gain DR + DR initial state + IMU residuals)"
                if args.dr else "off")
    print(f"[INFO] domain randomization: {dr_state}")
    print(f"[WARN] velocity limit mode={args.velocity_limit_mode}; neither mode reproduces PhysX soft constraints")
    if args.base_inertia != 1.0:
        for instance in group.envs:
            m = instance.model
            m.body_inertia[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")] *= args.base_inertia
            instance.refresh_dr_nominal()
        print(f"[WARN] base inertia scaled x{args.base_inertia}: deliberate non-v1 diagnostic override")

    auto_reset = True
    tau_eps = []  # per-episode (drive, steer) applied-torque traces
    ep_tau = []
    torque_traces = []  # per-global-step (num_envs, 2), drive then steering
    reset_events = []  # (global step, environment ID)
    drive_lim = steer_lim = None
    if args.task == "deploy_sim":
        onnx_path = args.onnx
        pt_path = args.pt if args.pt is not None else (None if args.onnx else str(DEFAULT_CHECKPOINT))
        policy = DeployedPolicy(onnx_path=onnx_path, pt_path=pt_path)
        loaded_policy_path = onnx_path if policy.backend == "onnxruntime" else pt_path
        print(f"[INFO] task=deploy_sim  policy backend: {policy.backend}")
        print(f"[INFO]   onnx: {onnx_path}")
        for instance in group.envs:
            instance.enabled_terms = {"time_out", "fallen"} if args.pure_survival else set(TERMINATIONS)
        print(f"[INFO] terminations: {sorted(env.enabled_terms)}")
        # torque tracing: data.actuator_force = the post-forcerange-clip applied
        # torque (the MuJoCo analog of the sim's applied_torque)
        steer_act = env.model.actuator("steering_act")
        drive_act = env.model.actuator("drive_act")
        steer_lim = float(env.model.actuator_forcerange[steer_act.id][1])
        drive_lim = float(env.model.actuator_forcerange[drive_act.id][1])
    else:
        policy = None
        print(f"[INFO] task={args.task}  (actuators disabled — free dynamics only)")
        for instance in group.envs:
            instance.motors_enabled = False
            instance.enabled_terms = {"time_out", "fallen"}
        if args.lock_steer:
            for instance in group.envs:
                instance.lock_steer = True
                instance.reset_ranges = ((-0.35, 0.35), (-1.5, 1.5))
            # strong initial lean so the bike topples within ~1 s, before any
            # forward roll-away can build up
            print("[INFO] steering welded at 0 (--lock_steer), strong reset perturbation")
        if args.task == "free_fall":
            auto_reset = False
            for instance in group.envs:
                instance.reset_ranges = ((-0.35, 0.35), (-1.5, 1.5))
                instance.model.dof_damping[:] = 0.
                instance.model.dof_frictionloss[:] = 0.
                instance.velocity_limit_mode = "none"
            print("[INFO] free_fall: no auto-reset — the bike falls once and stays")

    obs = group.reset()
    episodes = np.ones(args.num_envs, dtype=int)
    announced = np.zeros(args.num_envs, dtype=bool)
    global_step = 0
    ep_records = []  # (length, death, EpisodeStats) for the metrics log
    t_next = time.perf_counter()
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        with viewer.lock():
            group.configure_camera(viewer.cam)
            group.update_scene(viewer.user_scn, viewer.opt)
        print("[INFO] viewer open — close the window (or press Esc) to quit")
        while viewer.is_running():
            actions = policy.act_batch(obs) if policy is not None else np.zeros((args.num_envs, 2))
            obs, done, torque = group.step(actions)
            global_step += 1
            if args.task == "deploy_sim":
                if args.num_envs == 1:
                    ep_tau.append(tuple(torque[0]))
                else:
                    torque_traces.append(torque)
            with viewer.lock():
                group.update_scene(viewer.user_scn, viewer.opt)
            viewer.sync()
            for index in np.flatnonzero(done):
                instance = group.envs[index]
                s = instance.stats
                if auto_reset:
                    prefix = f"env {index:3d}: " if args.num_envs > 1 else ""
                    print(f"{prefix}episode {episodes[index]:3d}: {s.length:5d} steps ({s.length * 0.02:5.2f} s)  "
                          f"{'timeout' if s.timeout else s.death}")
                    if args.task == "deploy_sim":
                        ep_records.append((s.length, s.death, s))
                        reset_events.append((global_step, index))
                        if args.num_envs == 1:
                            tau_eps.append(np.array(ep_tau))
                            ep_tau = []
                    episodes[index] += 1
                    obs[index] = instance.reset()
                elif not announced[index]:
                    print(f"env {index}: bike fell: {s.death} at step {s.length} — simulating on, no reset")
                    announced[index] = True
            # pace to (approximately) real time: 0.02 s per control step
            t_next += 0.02 / args.speed
            delay = t_next - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                t_next = time.perf_counter()

    if args.task == "deploy_sim":
        if ep_tau:
            tau_eps.append(np.array(ep_tau))  # the partial episode before the window closed
        output_dir = os.path.join(ROOT, "logs")
        os.makedirs(output_dir, exist_ok=True)
        out_base = _evaluation_output_base(output_dir, loaded_policy_path, args.dr)
        if ep_records:
            write_metrics_log(ep_records, out_base + "_summary.log", num_envs=args.num_envs,
                              timing=dict(policy_hz=1.0/c.CONTROL_DT, physics_hz=args.physics_hz,
                                          pd_hz=args.physics_hz, decimation=substeps))
        _plot_episode_length_distribution(ep_records, out_base, loaded_policy_path, args.dr)
        if tau_eps:
            _plot_torque_traces(tau_eps, drive_lim, steer_lim, out_base)
        if torque_traces:
            _plot_multi_torque_traces(torque_traces, reset_events, drive_lim, steer_lim, out_base, args.seed)


if __name__ == "__main__":
    main()
