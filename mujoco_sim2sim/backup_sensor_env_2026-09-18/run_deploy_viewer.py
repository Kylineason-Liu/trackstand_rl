#!/usr/bin/env python3
"""Open the bike in the MuJoCo GUI and run one of two tasks:

  deploy_sim : load model_500 and run the deployment loop (the 50 Hz
               obs -> policy -> scaled-actions cycle, auto-reset on death),
               like run_deploy_eval.py but visualized. Also records the
               per-step applied drive/steer torque and saves a play_eval-style
               torque profile (torque_profile.png/.npz) next to the metrics log.
  zero_agent : no policy at all — actuators disabled, the bike just falls
               freely from the reset pose and resets on death (sanity check
               that the model + reset protocol behave physically).
  free_fall  : motors off and ALL joints fully free, strong initial lean,
               NO auto-reset — the bike falls once and stays where it lands.

The bike model to load: --mjcf (any MJCF file; the standard floor is added
around it) or, by default, the bundled scene models/newbike/scene.xml.

Fusion2Mujoco exports (the trackstand_rl_0916_bike model): such XMLs carry joints
but NO actuators and NO sensors, so loading one triggers the deploy-contract
augmentation — joints renamed to the env's names (base_free / steering_joint /
rearwheel_joint / frontwheel_joint), an imu site with gyro + velocimeter
sensors at the base-link origin, steering_act (position servo PD 10/0.2,
+-5.6 Nm, +-1.5708 rad target) and drive_act (torque ctrl - 0.05*qvel,
+-6.28 Nm) mirroring the validated newbike scene, timestep 0.0025 s,
ground friction 0.75 0 0 + stiff solref/solimp. deploy_sim on the augmented
model defaults to the trackstand model_15650 policy export.

Requires a display (local desktop or X forwarding). Close the window or
press Esc to quit.

Usage:
    python scripts/run_deploy_viewer.py --task deploy_sim --seed 42
    python scripts/run_deploy_viewer.py --task zero_agent
    python scripts/run_deploy_viewer.py --task deploy_sim --mjcf /path/to/bike.xml
    python scripts/run_deploy_viewer.py --task deploy_sim \
        --mjcf models/trackstand_rl_0916_bike/trackstand_rl_0916.xml
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

from newbike_mujoco import BikeBalanceEnv, DeployedPolicy, compose_scene_xml  # noqa: E402


def write_metrics_log(ep_records, log_path):
    """play_eval-style summary over the viewer's episodes, to a log file."""
    n = len(ep_records)
    lens = np.array([r[0] for r in ep_records], dtype=float)
    deaths = {}
    timeouts = 0
    for _, _, st in ep_records:
        timeouts += int(st.timeout)
        if st.death:
            deaths[st.death] = deaths.get(st.death, 0) + 1
    roll_rms = [np.sqrt(st.roll_sq / max(st.length, 1)) for _, _, st in ep_records]
    roll_maxs = [st.roll_max for _, _, st in ep_records]
    drift_maxs = [st.drift_max for _, _, st in ep_records]
    yaw_maxs = [st.yaw_max for _, _, st in ep_records]
    turns = [st.turn for _, _, st in ep_records]
    drive_means = [st.drive_speed_sum / max(st.length, 1) for _, _, st in ep_records]
    drive_maxs = [st.drive_speed_max for _, _, st in ep_records]
    steer_means = [st.steer_speed_sum / max(st.length, 1) for _, _, st in ep_records]
    steer_maxs = [st.steer_speed_max for _, _, st in ep_records]

    lines = []
    add = lines.append
    add("===== DEPLOY_SIM METRICS (MuJoCo, play_eval-style) =====")
    add(f"  episodes               : {n}")
    add(f"  mean_episode_length    : {np.mean(lens):9.1f} steps  ({np.mean(lens) * 0.02:6.2f} s)")
    add(f"  min/max episode length : {np.min(lens):9.1f} / {np.max(lens):9.1f}")
    add(f"  std_episode_length     : {np.std(lens):9.1f}")
    add(f"  timeout_rate (survived): {timeouts / n:9.3f}   (reached the horizon vs fell)")
    add(f"    time_out             : {timeouts / n * 100:8.1f}%")
    for k, v in sorted(deaths.items(), key=lambda kv: -kv[1]):
        add(f"    {k:20s}: {v / n * 100:8.1f}%")
    add(f"  roll_RMS               : {np.degrees(np.mean(roll_rms)):9.2f} deg   (tightness of balance; lower = steadier)")
    add(f"  max_roll_angle         : {np.degrees(np.mean(roll_maxs)):9.2f} deg")
    add(f"  mean_worst_lean/episode: {np.degrees(np.mean(roll_maxs)):9.2f} deg")
    add(f"  drive_speed mean/max   : {np.mean(drive_means):8.2f} / {np.mean(drive_maxs):8.2f} rad/s")
    add(f"  steer_speed mean/max   : {np.mean(steer_means):8.2f} / {np.mean(steer_maxs):8.2f} rad/s")
    add(f"  drift mean/max         : {np.mean(drift_maxs):9.3f} m   (position drift from spawn)")
    add(f"  yaw mean/max           : {np.mean(yaw_maxs):9.2f} deg (heading vs spawn)")
    add(f"  turn/episode mean      : {np.mean(turns):9.1f} deg (orbiting detection: 360 deg = full circle)")

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"[INFO] metrics log written to: {log_path}")


def _plot_torque_traces(tau_eps, drive_lim, steer_lim, out_base):
    """play_eval-style torque profile: applied drive/steer torque vs step with
    +-forcerange dashed lines. The single-env MuJoCo analog of play_eval's
    6-panel plot (MuJoCo's ``actuator_force`` IS the post-forcerange-clip
    applied torque, matching the sim's ``applied_torque``). Episode boundaries
    are dotted vertical lines. Saved as <out_base>_torque_profile.png (+ raw .npz).
    """
    boundaries = np.cumsum([len(e) for e in tau_eps])[:-1]
    steps = np.concatenate([np.arange(len(e), dtype=float) for e in tau_eps])
    drive = np.concatenate([e[:, 0] for e in tau_eps])
    steer = np.concatenate([e[:, 1] for e in tau_eps])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, tau, lim, name in ((axes[0], drive, drive_lim, "drive"), (axes[1], steer, steer_lim, "steer")):
        ax.plot(steps, tau, color="#1f77b4", lw=0.7)
        ax.axhline(lim, color="k", ls="--", lw=0.8)
        ax.axhline(-lim, color="k", ls="--", lw=0.8)
        for b in boundaries:
            ax.axvline(b, color="gray", ls=":", lw=0.6)
        ax.set_ylim(-1.15 * lim, 1.15 * lim)
        ax.set_ylabel(f"{name} τ (N·m)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_title(f"applied motor torque vs step — MuJoCo deploy_sim ({len(tau_eps)} episode(s))")
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out_base + "_torque_profile.png", dpi=150)
    plt.close(fig)
    np.savez(out_base + "_torque_profile.npz", steps=steps, boundaries=boundaries, drive=drive, steer=steer)
    print(f"[INFO] torque profile written to: {out_base}_torque_profile.png / .npz")


TRACKSTAND_EXPORT_ONNX = ("/home/lmbike/lmbike_cyclerl-mod/isaaclab_train/scripts/deployment/"
                          "trackstand_rl_rsl_2026-09-17_19-46-37_model_41050.onnx")
TRACKSTAND_EXPORT_PT = ("/home/lmbike/lmbike_cyclerl-mod/isaaclab_train/logs/rsl_rl/trackstand_rl_rsl/"
                        "2026-09-17_19-46-37/model_41050.pt")

# Fusion2Mujoco joint names -> the deploy-env names
_TRACKSTAND_JOINT_RENAMES = {"": "base_free", "steering_link": "steering_joint",
                             "rear_wheel_link": "rearwheel_joint",
                             "front_wheel_link": "frontwheel_joint"}


def augment_trackstand_model(spec, model_dir):
    """Turn a Fusion2Mujoco export (joints, no actuators/sensors) into the
    deploy-contract model, mirroring the validated newbike scene.xml:
      - joints renamed to the env's names
      - imu site + gyro/velocimeter sensors at the model_root origin (the
        sim-equivalent base frame)
      - steering_act: position servo PD 10/0.2, +-5.6 Nm, +-1.5708 rad target
        (the sim's ImplicitActuator k10/d0.2 contract; encoding verified
        bit-identical to the newbike scene's compiled actuator)
      - drive_act: torque ctrl - 0.05*qvel damping, +-6.28 Nm
        (the sim's ImplicitActuator k0/d0.05 contract)
      - timestep 0.0025 s (8 substeps per 0.02 s control period)
      - ground contact: floor + collision-default friction 0.75 0 0 (the
        sim's effective pair) with the newbike scene's stiff solref/solimp
    """
    for j in spec.joints:
        if j.name not in _TRACKSTAND_JOINT_RENAMES:
            raise ValueError(f"Unrecognized joint name in trackstand model: {j.name!r}")
        j.name = _TRACKSTAND_JOINT_RENAMES[j.name]

    # the imu site goes on model_root (the FREE body): its frame is the
    # sim-equivalent BASE frame (identity at home), while the exported
    # base_link body sits inside it with a ~180 deg CAD-flip quat. The sim's
    # articulation root is the USD base_link at identity, so the gyro must
    # read in model_root's frame to match root_ang_vel_b.
    # the imu site would render as a small sphere in the viewer — give it an
    # alpha-0 material (sites draw as spheres by default; the newbike model
    # hides its imu site the same way). Rendering only, the sensors keep
    # reading the site's frame.
    spec.add_material(name="imu_invisible", rgba=[0.0, 0.0, 0.0, 0.0])
    site = spec.body("model_root").add_site(name="imu", pos=[0.0, 0.0, 0.0], size=[0.003])
    site.material = "imu_invisible"
    spec.add_sensor(name="gyro", type=mujoco.mjtSensor.mjSENS_GYRO,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    spec.add_sensor(name="vel", type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")

    spec.add_actuator(name="steering_act", target="steering_joint",
                      trntype=mujoco.mjtTrn.mjTRN_JOINT,
                      gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                      gainprm=[10.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                      biasprm=[0.0, -10.0, -0.2, 0, 0, 0, 0, 0, 0, 0],
                      ctrllimited=True, ctrlrange=[-1.5708, 1.5708],
                      forcelimited=True, forcerange=[-5.6, 5.6])
    spec.add_actuator(name="drive_act", target="rearwheel_joint",
                      trntype=mujoco.mjtTrn.mjTRN_JOINT,
                      gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                      gainprm=[1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                      biasprm=[0.0, 0.0, -0.05, 0, 0, 0, 0, 0, 0, 0],
                      ctrllimited=True, ctrlrange=[-6.28, 6.28],
                      forcelimited=True, forcerange=[-6.28, 6.28])

    spec.option.timestep = 0.0025
    spec.compiler.meshdir = model_dir
    spec.stat.center = [0.0, 0.0, 0.4]
    spec.stat.extent = 1.2

    floor = spec.geom("floor")
    floor.friction = [0.75, 0.0, 0.0]
    floor.solref = [0.005, 1.5]
    floor.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]
    collision_default = spec.find_default("collision")
    collision_default.geom.friction = [0.75, 0.0, 0.0]
    collision_default.geom.solref = [0.005, 1.5]
    collision_default.geom.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]


def build_env_from_mjcf(mjcf_path, seed):
    """Load --mjcf. Fusion2Mujoco exports (no actuators AND no sensors) get the
    deploy-contract augmentation; models without a floor get the composed
    scene wrapper. Returns (env, trackstand_mode)."""
    mjcf_path = os.path.abspath(mjcf_path)
    spec = mujoco.MjSpec.from_file(mjcf_path)
    needs_aug = len(spec.actuators) == 0 and len(spec.sensors) == 0
    has_floor = any(g.name == "floor" for g in spec.geoms)

    if needs_aug:
        print(f"[INFO] {os.path.basename(mjcf_path)} has no actuators/sensors — applying the "
              "deploy-contract augmentation (joint renames, PD steering + torque drive, "
              "IMU sensors, 0.0025 s timestep, ground friction 0.75 0 0)")
        model_dir = os.path.dirname(mjcf_path)
        augment_trackstand_model(spec, model_dir)
        scene_path = os.path.join(model_dir, ".trackstand_deploy_augmented.xml")
        with open(scene_path, "w") as f:
            f.write(spec.to_xml())
        print(f"[INFO] augmented scene written: {scene_path}")
        return BikeBalanceEnv(scene_path=scene_path, seed=seed), True

    if not has_floor:
        print(f"[INFO] loading bike model: {mjcf_path} (standard floor added)")
        scene_path = compose_scene_xml(mjcf_path)
    else:
        print(f"[INFO] loading bike model: {mjcf_path} (model carries its own floor)")
        scene_path = mjcf_path
    return BikeBalanceEnv(scene_path=scene_path, seed=seed), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["deploy_sim", "zero_agent", "free_fall"], default="deploy_sim",
                    help="deploy_sim: run the deployed policy; zero_agent: no control + auto-reset; "
                         "free_fall: no control, no reset.")
    ap.add_argument("--lock_steer", action="store_true",
                    help="zero_agent only: weld the steering at 0 so the bike topples like "
                         "a rigid body instead of caster-weaving.")
    ap.add_argument("--mjcf", default=None,
                    help="Bike MJCF to load (default: the bundled models/newbike/scene.xml).")
    ap.add_argument("--scene", default=os.path.join(ROOT, "models/newbike/scene.xml"))
    ap.add_argument("--onnx", default=None,
                    help="Deploy ONNX policy (default: the NewBike model_500, or the trackstand "
                         "model_15650 export for an augmented trackstand model).")
    ap.add_argument("--pt", default=None,
                    help="Deploy torch checkpoint (same defaulting as --onnx).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="Playback speed (1.0 = real time, 0.5 = half speed).")
    ap.add_argument("--base_inertia", type=float, default=1.0,
                    help="Scale the base link's rotational inertia (default 1 = the USD "
                         "value: the fall matches the sim's full-gravity rate). Use e.g. "
                         "3000 for the long-survival closed-loop calibration (the fall "
                         "then looks like slow motion -- see the README's trade-off).")
    args = ap.parse_args()

    if args.mjcf is not None:
        env, trackstand_mode = build_env_from_mjcf(args.mjcf, args.seed)
    else:
        trackstand_mode = False
        print(f"[INFO] loading scene: {args.scene}")
        env = BikeBalanceEnv(scene_path=args.scene, seed=args.seed)
    if args.base_inertia != 1.0:
        m = env.model
        m.body_inertia[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")] *= args.base_inertia
        print(f"[INFO] base inertia scaled x{args.base_inertia}")

    auto_reset = True
    tau_eps = []  # per-episode (drive, steer) applied-torque traces
    ep_tau = []
    drive_lim = steer_lim = None
    if args.task == "deploy_sim":
        onnx_path = args.onnx if args.onnx is not None else (
            TRACKSTAND_EXPORT_ONNX if trackstand_mode else os.path.join(ROOT, "policies/model_500.onnx"))
        pt_path = args.pt if args.pt is not None else (
            TRACKSTAND_EXPORT_PT if trackstand_mode else os.path.join(ROOT, "policies/model_500.pt"))
        policy = DeployedPolicy(onnx_path=onnx_path, pt_path=pt_path)
        print(f"[INFO] task=deploy_sim  policy backend: {policy.backend}")
        print(f"[INFO]   onnx: {onnx_path}")
        # deploy_sim watches balance only: every termination except the
        # roll-angle limit (fallen) is disabled
        env.enabled_terms = {"fallen"}
        print("[INFO] terminations: fallen (roll limit) only")
        # torque tracing: data.actuator_force = the post-forcerange-clip applied
        # torque (the MuJoCo analog of the sim's applied_torque)
        steer_act = env.model.actuator("steering_act")
        drive_act = env.model.actuator("drive_act")
        steer_lim = float(env.model.actuator_forcerange[steer_act.id][1])
        drive_lim = float(env.model.actuator_forcerange[drive_act.id][1])
    else:
        policy = None
        print(f"[INFO] task={args.task}  (actuators disabled — free dynamics only)")
        # zero the actuator gains so even the servo/damping terms are inert
        m = env.model
        for i in range(m.nu):
            m.actuator_gainprm[i, 0] = 0.0
            m.actuator_gainprm[i, 1] = 0.0
            m.actuator_biasprm[i, :] = 0.0
        if args.lock_steer:
            env.lock_steer = True
            # strong initial lean so the bike topples within ~1 s, before any
            # forward roll-away can build up
            env.reset_ranges = ((-0.35, 0.35), (-1.5, 1.5))
            print("[INFO] steering welded at 0 (--lock_steer), strong reset perturbation")
        if args.task == "free_fall":
            auto_reset = False
            env.reset_ranges = ((-0.35, 0.35), (-1.5, 1.5))
            print("[INFO] free_fall: no auto-reset — the bike falls once and stays")

    obs = env.reset()
    ep = 1
    announced = False
    ep_records = []  # (length, death, EpisodeStats) for the metrics log
    t_next = time.perf_counter()
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        print("[INFO] viewer open — close the window (or press Esc) to quit")
        while viewer.is_running():
            action = policy.act(obs) if policy is not None else (0.0, 0.0)
            obs, done = env.step(action)
            if args.task == "deploy_sim":
                d = env.data
                ep_tau.append((float(d.actuator_force[drive_act.id]),
                               float(d.actuator_force[steer_act.id])))
            viewer.sync()
            if done:
                s = env.stats
                if auto_reset:
                    print(f"episode {ep:3d}: {s.length:5d} steps ({s.length * 0.02:5.2f} s)  "
                          f"{'timeout' if s.timeout else s.death}")
                    if args.task == "deploy_sim":
                        ep_records.append((s.length, s.death, s))
                        tau_eps.append(np.array(ep_tau))
                        ep_tau = []
                    ep += 1
                    obs = env.reset()
                elif not announced:
                    print(f"bike fell: {s.death} at step {s.length} — simulating on, no reset")
                    announced = True
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
        out_base = os.path.join(ROOT, "logs", f"deploy_sim_{time.strftime('%m%d%H%M')}")
        if ep_records:
            write_metrics_log(ep_records, out_base + ".log")
        if tau_eps:
            _plot_torque_traces(tau_eps, drive_lim, steer_lim, out_base)


if __name__ == "__main__":
    main()
