#!/usr/bin/env python3
"""Open the bike in the MuJoCo GUI and run one of two tasks:

  deploy_sim : load model_500 and run the deployment loop (the 50 Hz
               obs -> policy -> scaled-actions cycle, auto-reset on death),
               like run_deploy_eval.py but visualized. Also records the
               per-step applied drive/steer torque and saves a play_eval-style
               torque plot/data (*_torque.png/.npz) next to the summary log.
               On exit, saves a completed-episode length histogram: ten
               3-second bins covering 0-30 s, aggregated across all bikes.
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
rearwheel_joint / frontwheel_joint), steering_act (position servo PD 10/0.2,
+-5.6 Nm, +-1.5708 rad target) and drive_act (torque ctrl - 0.05*qvel,
+-6.28 Nm) mirroring the validated newbike scene, timestep 0.0025 s,
ground friction 0.1 0 0 (pass-through floor: the tire's 0.65 dominates
MAX-combine, matching the sim's min-combine pair = tire) + stiff solref/solimp.
No sensors are added — the
env reads the base velocities straight from base_free's qvel (see env.py's
observe()), so no imu site exists to render. deploy_sim on the augmented
model defaults to the trackstand model_41050 policy export.

Starts 64 independent environments by default, shown as a render-only grid in
one window. Use --num_envs 1 for the original single-bike view. Physics is
stepped on the CPU; 50 Hz is the simulated control rate, not a wall-time promise.
Requires a display (local desktop or X forwarding). Close the window or press Esc to quit.

Usage:
    python scripts/run_deploy_viewer.py --task deploy_sim --seed 42
    python scripts/run_deploy_viewer.py --task zero_agent
    python scripts/run_deploy_viewer.py --task deploy_sim --mjcf /path/to/bike.xml
    python scripts/run_deploy_viewer.py --task deploy_sim \
        --mjcf models/trackstand_rl_0916_bike/trackstand_rl_0916.xml
    python scripts/run_deploy_viewer.py --task deploy_sim --dr \
        --mjcf models/trackstand_rl_0916_bike/trackstand_rl_0916.xml
    # --dr turns on the per-episode domain randomization (the sim's
    # BalanceEventsCfg: mass/COM/tire-friction/actuator-gain randomization,
    # DR initial state, IMU turn-on residuals) — off by default.
"""

import argparse
import os
import re
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
from newbike_mujoco.viewer_envs import ViewerEnvGroup  # noqa: E402


def write_metrics_log(ep_records, log_path, num_envs=1):
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
    add(f"  environments           : {num_envs}")
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


def _evaluation_output_base(output_dir, policy_path, enable_dr):
    """Choose one timestamp/model/DR prefix for all reports, preserving earlier runs."""
    policy_name = os.path.basename(policy_path)
    model = re.search(r"model_(\d+)", policy_name)
    model_id = model.group(1) if model else os.path.splitext(policy_name)[0]
    stem = f"{time.strftime('%Y%m%d%H%M')}_{model_id}_{'dr' if enable_dr else 'no_dr'}"
    out_base = os.path.join(output_dir, stem)
    endings = ("_epi_len_histogram.png", "_epi_len_histogram.csv", "_torque.png", "_torque.npz", "_summary.log")
    suffix = 2
    while any(os.path.exists(out_base + ending) for ending in endings):
        out_base = os.path.join(output_dir, f"{stem}_{suffix}")
        suffix += 1
    return out_base


def _plot_episode_length_distribution(ep_records, out_base, policy_path, enable_dr):
    """Count completed episodes in [0, 3), ..., [27, 30] s; save PNG and counts."""
    lengths_s = np.asarray([record[0] for record in ep_records]) * BikeBalanceEnv.CTRL_DT
    edges_s = np.arange(0.0, 31.0, 3.0)
    counts, _ = np.histogram(lengths_s, bins=edges_s)
    policy_name = os.path.basename(policy_path)
    model = re.search(r"model_\d+", policy_name)
    model_label = model.group() if model else policy_name

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(edges_s[:-1], counts, width=3.0, align="edge", color="#1f77b4", edgecolor="white")
    ax.bar_label(bars, padding=3)
    ax.set_xticks(edges_s)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, max(1, int(counts.max()) * 1.15))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Episode length (s)")
    ax.set_ylabel("Number of completed episodes")
    ax.set_title(f"Episode length distribution — {len(ep_records)} completed episodes\n"
                 f"{model_label} | Domain randomization: {'ON' if enable_dr else 'OFF'}")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.5, 0.02, "Bins: [0, 3), ..., [27, 30] s. Timeouts included; unfinished episodes excluded.",
             ha="center", fontsize=9)
    fig.savefig(out_base + "_epi_len_histogram.png", dpi=150)
    plt.close(fig)
    np.savetxt(
        out_base + "_epi_len_histogram.csv",
        np.column_stack((edges_s[:-1], edges_s[1:], counts)), delimiter=",",
        header="start_s,end_s,episode_count", comments="", fmt=("%.1f", "%.1f", "%d"),
    )
    print(f"[INFO] episode length distribution written: {out_base}_epi_len_histogram.png / .csv")


def _plot_torque_traces(tau_eps, drive_lim, steer_lim, out_base):
    """play_eval-style torque profile: applied drive/steer torque vs step with
    +-forcerange dashed lines. The single-env MuJoCo analog of play_eval's
    6-panel plot (MuJoCo's ``actuator_force`` IS the post-forcerange-clip
    applied torque, matching the sim's ``applied_torque``). Episode boundaries
    are dotted vertical lines. Saved as <out_base>_torque.png (+ raw .npz).
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
    fig.savefig(out_base + "_torque.png", dpi=150)
    plt.close(fig)
    np.savez(out_base + "_torque.npz", steps=steps, boundaries=boundaries, drive=drive, steer=steer)
    print(f"[INFO] torque profile written to: {out_base}_torque.png / .npz")


def _plot_multi_torque_traces(traces, resets, drive_lim, steer_lim, out_base, seed):
    """All environments on a shared control-step axis; preserve IDs in the NPZ."""
    torque = np.asarray(traces)
    steps = np.arange(len(torque))
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, values, limit, name in (
        (axes[0], torque[:, :, 0], drive_lim, "drive"),
        (axes[1], torque[:, :, 1], steer_lim, "steer"),
    ):
        ax.plot(steps, values, color="#1f77b4", alpha=0.15, lw=0.5)
        ax.plot(steps, values.mean(axis=1), color="black", lw=0.9, label="mean across environments")
        ax.axhline(limit, color="gray", ls="--", lw=0.8)
        ax.axhline(-limit, color="gray", ls="--", lw=0.8)
        ax.set_ylim(-1.15 * limit, 1.15 * limit)
        ax.set_ylabel(f"{name} torque (N m)")
        ax.legend(fontsize=8)
    axes[0].set_title(f"Applied motor torque — {torque.shape[1]} independent environments")
    axes[-1].set_xlabel("global control step (0.02 s each)")
    fig.tight_layout()
    fig.savefig(out_base + "_torque.png", dpi=150)
    plt.close(fig)
    np.savez(
        out_base + "_torque.npz", steps=steps,
        drive=torque[:, :, 0], steer=torque[:, :, 1],
        env_ids=np.arange(torque.shape[1]), seeds=seed + np.arange(torque.shape[1]),
        reset_events=np.asarray(resets, dtype=int).reshape(-1, 2),
        control_dt=BikeBalanceEnv.CTRL_DT,
    )
    print(f"[INFO] all-environment torque profile written: {out_base}_torque.png / .npz")


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


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
      - NO sensors: the env reads the base velocities directly from the
        model_root freejoint qvel (angvel body-frame, linvel world-frame)
      - steering_act: position servo PD 10/0.2, +-5.6 Nm, +-1.5708 rad target
        (the sim's ImplicitActuator k10/d0.2 contract; encoding verified
        bit-identical to the newbike scene's compiled actuator)
      - drive_act: torque ctrl - 0.05*qvel damping, +-6.28 Nm
        (the sim's ImplicitActuator k0/d0.05 contract)
      - timestep 0.0025 s (8 substeps per 0.02 s control period; the sim
        integrates at 0.01 s but MuJoCo's contact solver CANNOT hold the
        balance at 0.01 — measured sweep 2026-09-18, see cfg_record.txt)
      - ground contact: floor friction 0.1 0 0 (pass-through) + collision-default
        friction 0.65 0 0 — MAX-combine pair = the tire's 0.65, mirroring the
        sim's min-combine (pair = tire value) with the newbike scene's stiff
        solref/solimp
    """
    for j in spec.joints:
        if j.name not in _TRACKSTAND_JOINT_RENAMES:
            raise ValueError(f"Unrecognized joint name in trackstand model: {j.name!r}")
        j.name = _TRACKSTAND_JOINT_RENAMES[j.name]

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
    floor.friction = [0.1, 0.0, 0.0]  # pass-through: tire 0.65 dominates MAX-combine
    floor.solref = [0.005, 1.5]
    floor.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]
    collision_default = spec.find_default("collision")
    collision_default.geom.friction = [0.65, 0.0, 0.0]
    # the export carries EXPLICIT friction AND solref on the bike collision
    # geoms (they do not inherit the collision default) — set them directly.
    # solref [0.005, 1.5] = the validated newbike scene's stiff contact (the
    # export's explicit "0.02 1" is just MuJoCo's built-in default; audit
    # 2026-09-22: it silently overrode the stiff intent — fixed here).
    for g in spec.geoms:
        if g.name != "floor":
            g.friction = [0.65, 0.0, 0.0]
            g.solref = [0.005, 1.5]
    collision_default.geom.solref = [0.005, 1.5]
    collision_default.geom.solimp = [0.9, 0.95, 0.001, 0.5, 2.0]


def build_env_from_mjcf(mjcf_path, seed, enable_dr=False):
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
              "no sensors — obs read base_free qvel directly, 0.0025 s timestep, "
              "ground friction 0.1 0 0 pass-through)")
        model_dir = os.path.dirname(mjcf_path)
        augment_trackstand_model(spec, model_dir)
        scene_path = os.path.join(model_dir, ".trackstand_deploy_augmented.xml")
        with open(scene_path, "w") as f:
            f.write(spec.to_xml())
        print(f"[INFO] augmented scene written: {scene_path}")
        return BikeBalanceEnv(scene_path=scene_path, seed=seed, enable_dr=enable_dr), True

    if not has_floor:
        print(f"[INFO] loading bike model: {mjcf_path} (standard floor added)")
        scene_path = compose_scene_xml(mjcf_path)
    else:
        print(f"[INFO] loading bike model: {mjcf_path} (model carries its own floor)")
        scene_path = mjcf_path
    return BikeBalanceEnv(scene_path=scene_path, seed=seed, enable_dr=enable_dr), False


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
    ap.add_argument("--num_envs", type=_positive_int, default=64,
                    help="Number of independent bikes shown in one viewer (default: 64); "
                         "use 1 for the original single-bike view. CPU physics, seeds seed+i.")
    ap.add_argument("--dr", action="store_true",
                    help="enable the domain-randomization terms (the sim's BalanceEventsCfg): "
                         "per-episode mass/COM/tire-friction/actuator-gain randomization, "
                         "DR initial state (roll U(+-0.10) rad, roll-rate U(+-0.25) rad/s), "
                         "and IMU turn-on residuals.")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="Playback speed (1.0 = real time, 0.5 = half speed).")
    ap.add_argument("--base_inertia", type=float, default=1.0,
                    help="Scale the base link's rotational inertia (default 1 = the USD "
                         "value: the fall matches the sim's full-gravity rate). Use e.g. "
                         "3000 for the long-survival closed-loop calibration (the fall "
                         "then looks like slow motion -- see the README's trade-off).")
    args = ap.parse_args()

    if args.mjcf is not None:
        env, trackstand_mode = build_env_from_mjcf(args.mjcf, args.seed, enable_dr=args.dr)
    else:
        trackstand_mode = False
        print(f"[INFO] loading scene: {args.scene}")
        env = BikeBalanceEnv(scene_path=args.scene, seed=args.seed, enable_dr=args.dr)
    group = ViewerEnvGroup(env, args.num_envs, args.seed)
    print(f"[INFO] environments: {args.num_envs}; independent seeds {args.seed}.."
          f"{args.seed + args.num_envs - 1}; CPU physics, batched policy inference")
    dr_state = ("ON (per-episode mass/COM/friction/gain DR + DR initial state + IMU residuals)"
                if args.dr else "off")
    print(f"[INFO] domain randomization: {dr_state}")
    if args.base_inertia != 1.0:
        for instance in group.envs:
            m = instance.model
            m.body_inertia[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")] *= args.base_inertia
            if args.dr:
                instance.refresh_dr_nominal()  # keep scaled inertia as the DR baseline
        print(f"[INFO] base inertia scaled x{args.base_inertia}")

    auto_reset = True
    tau_eps = []  # per-episode (drive, steer) applied-torque traces
    ep_tau = []
    torque_traces = []  # per-global-step (num_envs, 2), drive then steering
    reset_events = []  # (global step, environment ID)
    drive_lim = steer_lim = None
    if args.task == "deploy_sim":
        onnx_path = args.onnx if args.onnx is not None else (
            TRACKSTAND_EXPORT_ONNX if trackstand_mode else os.path.join(ROOT, "policies/model_500.onnx"))
        pt_path = args.pt if args.pt is not None else (
            TRACKSTAND_EXPORT_PT if trackstand_mode else os.path.join(ROOT, "policies/model_500.pt"))
        policy = DeployedPolicy(onnx_path=onnx_path, pt_path=pt_path)
        loaded_policy_path = onnx_path if policy.backend == "onnxruntime" else pt_path
        print(f"[INFO] task=deploy_sim  policy backend: {policy.backend}")
        print(f"[INFO]   onnx: {onnx_path}")
        # deploy_sim terminations: the env's default envelope — time_out
        # (1500 steps, resets the episode) + fallen (roll limit); the other
        # terms (yaw / wheel-net / drift / diverged) were DELETED from the env
        for instance in group.envs:
            instance.enabled_terms = {"time_out", "fallen"}
        print("[INFO] terminations: time_out (1500 steps) + fallen (roll limit)")
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
        for instance in group.envs:
            m = instance.model
            m.actuator_gainprm[:, :2] = 0.0
            m.actuator_biasprm[:, :] = 0.0
            if args.dr:
                instance.refresh_dr_nominal()  # zero remains zero after gain DR
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
            write_metrics_log(ep_records, out_base + "_summary.log", num_envs=args.num_envs)
        _plot_episode_length_distribution(ep_records, out_base, loaded_policy_path, args.dr)
        if tau_eps:
            _plot_torque_traces(tau_eps, drive_lim, steer_lim, out_base)
        if torque_traces:
            _plot_multi_torque_traces(torque_traces, reset_events, drive_lim, steer_lim, out_base, args.seed)


if __name__ == "__main__":
    main()
