"""Offline deployment analysis: measured traces versus explicitly labelled replays.

No ROS executor, CAN socket, simulator steps or motor commands are started.
"""
import argparse
import csv
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch


def stats(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "min": float(x.min()), "median": float(np.median(x)),
            "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)),
            "max": float(x.max()), "mean": float(x.mean()), "rms": float(np.sqrt(np.mean(x*x)))}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_axes(root):
    """Read-only kinematics of the already verified model; no dynamic steps."""
    import mujoco
    sys.path.insert(0, str(root/"mujoco_sim2sim/src"))
    module = importlib.util.spec_from_file_location("trackstand_viewer_axes", root/"mujoco_sim2sim/scripts/run_deploy_viewer.py")
    viewer = importlib.util.module_from_spec(module)
    module.loader.exec_module(viewer)
    path = root/"mujoco_sim2sim/models/trackstand_rl_0916_bike/trackstand_rl_0916.xml"
    spec = mujoco.MjSpec.from_file(str(path))
    viewer.augment_trackstand_model(spec, str(path.parent))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rear, front = model.joint("rearwheel_joint").id, model.joint("frontwheel_joint").id
    forward = data.xanchor[front]-data.xanchor[rear]
    forward /= np.linalg.norm(forward)
    velocity = -np.cross(data.xaxis[rear], [0, 0, -.07])
    assert np.dot(velocity, forward) > 0
    return {"model_path": str(path.relative_to(root)), "sha256": sha(path),
            "rear_joint_world_axis": data.xaxis[rear].tolist(),
            "steering_joint_world_axis": data.xaxis[model.joint("steering_joint").id].tolist(),
            "rear_to_front_direction": forward.tolist(), "positive_rear_rotation_no_slip_world_velocity": velocity.tolist(),
            "method": "in-memory viewer augmentation, compile, mj_forward only; zero dynamic steps"}


def quaternion_rates(imu):
    # inv(q_before) * q_after: body-frame rotation over ~40 ms.
    q, t, n = imu[:, 2:6], imu[:, 1], 4
    av, aw = -q[:-n, :3], q[:-n, 3:]
    bv, bw = q[n:, :3], q[n:, 3:]
    vector = aw*bv + bw*av + np.cross(av, bv)
    scalar = (aw*bw)[:, 0] - np.sum(av*bv, axis=1)
    flip = scalar < 0
    vector[flip] *= -1
    scalar[flip] *= -1
    norm = np.linalg.norm(vector, axis=1)
    angle = 2*np.arctan2(norm, scalar)
    dt = t[n:]-t[:-n]
    rates = vector*np.divide(angle, norm, out=np.ones_like(angle)*2, where=norm>1e-12)[:, None]/dt[:, None]
    center = (t[n:]+t[:-n])/2
    result = {}
    for i, axis in enumerate("xyz"):
        mask = (dt>.02)&(dt<.065)&(np.abs(rates[:, i])>.1)
        gyro = np.interp(center, t, imu[:, 6+i])
        result[axis] = {"n": int(mask.sum()), "correlation": float(np.corrcoef(rates[mask, i], gyro[mask])[0, 1]),
                        "gyro_to_quaternion_rate_gain": float(np.dot(rates[mask, i], gyro[mask])/np.dot(rates[mask, i], rates[mask, i]))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    a = dict(np.load(args.cache/"data.npz"))
    d = json.loads((args.cache/"details.json").read_text())
    col = lambda key, name: a[key][:, 1+d["schemas"][key].index(name)]
    m, ms, obs, raw, pt, imu = (a[key] for key in ("motor_control", "motor_state", "observation", "raw_action", "policy_timing", "imu"))
    assert len(obs) == len(raw) == int((col("policy_timing", "inference_success")>.5).sum())
    enabled = a["motor_command"][a["motor_command"][:, 1]>.5]
    assert len(enabled) == len(raw)
    scaled = raw[:, 1:]*[1.57, 6.28]
    limited = scaled.copy()
    limited[:, 0] = np.clip(limited[:, 0], -np.pi/2, np.pi/2)
    scale_error = float(np.max(np.abs(limited-enabled[:, 2:])))
    assert scale_error < 1e-9

    model = root/"deploy/source/trackstand_deploy/models/trackstand_rl_rsl_2026-09-22_19-45-06_model_91850.onnx"
    checkpoint = root/"isaaclab_train/logs/rsl_rl/trackstand_rl_rsl/2026-09-22_19-45-06/model_91850.pt"
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = opt.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=opt, providers=["CPUExecutionProvider"])
    x = obs[:, 1:].astype(np.float32)
    replay = session.run(None, {"observation": x})[0]
    assert np.max(np.abs(replay-raw[:, 1:])) < 1e-5
    # Repository-owned checkpoint; original ELU architecture and no obs normalization.
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
    net = torch.nn.Sequential(torch.nn.Linear(13, 32), torch.nn.ELU(), torch.nn.Linear(32, 32),
                              torch.nn.ELU(), torch.nn.Linear(32, 2))
    net.load_state_dict({k[4:]:v for k, v in sd.items() if k.startswith("mlp.")}, strict=True)
    torch.set_num_threads(1)
    with torch.no_grad():
        original = net(torch.from_numpy(x)).numpy()
    zero = np.zeros((1, 13), np.float32)
    zero[0, 1], zero[0, 12] = -1, 1
    perfect = session.run(None, {"observation": zero})[0][0]
    cf_obs = x.copy()
    cf_obs[:, [5, 8, 10]] *= -1  # vx estimate, wrapped rear position, rear speed
    counterfactual = session.run(None, {"observation": cf_obs})[0]

    active = col("motor_control", "drive_send_success")>.5
    inferred = col("policy_timing", "inference_success")>.5
    qx, qy, qz, qw = imu[:, 2:6].T
    gy = -2*(qy*qz+qw*qx)
    gz = -(1-2*(qx*qx+qy*qy))
    summary = {k:d[k] for k in ("bag", "bytes", "sha256", "duration_s", "counts", "issues")}
    summary.update({
        "decoded_message_count": int(sum(d["counts"].values())), "model": model.name,
        "model_sha256": sha(model), "checkpoint_sha256": sha(checkpoint),
        "onnx_replay_max_error": float(np.max(np.abs(replay-raw[:, 1:]))),
        "torch_checkpoint_replay_max_error": float(np.max(np.abs(original-raw[:, 1:]))),
        "action_scaling_max_error": scale_error,
        "ideal_upright_zero_raw_action": perfect.tolist(),
        "ideal_upright_zero_scaled_action": (perfect*[1.57, 6.28]).tolist(),
        "steering_requested_outside_90_count": int((np.abs(scaled[:, 0])>np.pi/2).sum()),
        "rear_requested_outside_6_28_count": int((np.abs(scaled[:, 1])>6.28).sum()),
        "policy_samples": len(obs), "successful_rear_send_ticks": int(active.sum()),
        "counterfactual_not_closed_loop": {
            "changed_indices_zero_based": [5, 8, 10], "original_rear_raw_abs_gt_1_count": int((np.abs(raw[:, 2])>1).sum()),
            "flipped_rear_raw_abs_gt_1_count": int((np.abs(counterfactual[:, 1])>1).sum()),
            "warning": "Same recorded poses/rates/steering, not the trajectory that a corrected controller would create."},
        "motor_active_timing": {name:stats(col("motor_control", name)[active]) for name in (
            "cycle_interval_ms", "work_duration_ms", "command_age_ms", "steering_pair_age_ms", "drive_pair_age_ms", "mit_actual_send_interval_ms")},
        "policy_active_timing": {name:stats(col("policy_timing", name)[inferred]) for name in (
            "cycle_interval_ms", "work_duration_ms", "inference_ms", "imu_age_ms", "motor_age_ms", "rc_age_ms")},
        "imu_quaternion_norm": stats(np.linalg.norm(imu[:, 2:6], axis=1)),
        "imu_quaternion_gyro_check": quaternion_rates(imu),
        "rc_average_recorded_hz": float((len(a["kickstands_up"])-1)/np.ptp(a["kickstands_up"][:, 0])),
        "disabled_motor_command_count": int((a["motor_command"][:, 1]<.5).sum()),
        "parameter_event_count": len(d["text"].get("parameter_events", [])),
        "user_conditions": "Both wheels grounded, no manual restraint during all four attempts; physically straight steering on each motor-node start/restart; no changed settings, Pi repo reported up-to-date.",
    })
    drive_equation = np.clip(col("motor_control", "bounded_drive_target_nm")[active]
                             -.05*col("motor_control", "drive_velocity_rad_s")[active], -6.28, 6.28)
    summary["rear_control_equation_max_error_nm"] = float(np.max(np.abs(drive_equation-col("motor_control", "drive_final_torque_nm")[active])))
    assert summary["rear_control_equation_max_error_nm"] < 1e-9
    summary["encoded_mit_gains"] = {name:np.unique(col("motor_control", name)[active]).tolist() for name in (
        "mit_encoded_kp_nm_per_motor_rad", "mit_encoded_kd_nm_s_per_motor_rad")}
    attempts = []
    cuts = np.r_[0, np.flatnonzero(np.diff(obs[:, 0])>.1)+1, len(obs)]
    for number, (l, h) in enumerate(zip(cuts[:-1], cuts[1:]), 1):
        l, h = int(l), int(h)
        beg, end = obs[l, 0], obs[h-1, 0]
        mask = (m[:, 0]>=beg-.01)&(m[:, 0]<=end+.025)&active
        state_mask = (ms[:, 0]>=beg-.01)&(ms[:, 0]<=end+.025)
        item = {"attempt": number, "first_s": float(beg), "last_inference_s": float(end), "policy_samples": h-l,
                "initial_observation": obs[l, 1:].tolist(), "initial_raw_action": raw[l, 1:].tolist(),
                "initial_unclipped_steering_deg": float(np.rad2deg(scaled[l, 0])),
                "initial_rear_request_nm": float(scaled[l, 1]), "active_rear_send_ticks": int(mask.sum()),
                "steering_target_clipping_fraction": float(np.mean(np.abs(scaled[l:h, 0])>np.pi/2)),
                "rear_request_exceedance_fraction": float(np.mean(np.abs(scaled[l:h, 1])>6.28)),
                "raw_rear_action": stats(raw[l:h, 2]), "counterfactual_raw_rear_action": stats(counterfactual[l:h, 1]),
                "inference_events": [v for v in d["text"]["policy_event"] if beg-.1<v["t"]<end+.2],
                "motor_events": [v for v in d["text"]["motor_event"] if beg-.1<v["t"]<end+.2]}
        if mask.any():
            t = col("motor_control", "monotonic_time_s")[mask]
            v = col("motor_control", "drive_velocity_rad_s")[mask]
            q = np.unwrap(col("motor_control", "drive_position_rad")[mask])
            s = col("motor_control", "steering_position_rad")[mask]
            tau = col("motor_control", "drive_final_torque_nm")[mask]
            item.update({"drive_speed_rad_s": stats(v), "steering_position_deg": stats(np.rad2deg(s)),
                         "drive_command_nm": stats(tau), "drive_current_derived_torque_nm": stats(ms[state_mask, 7]),
                         "drive_net_rotation_rad": float(q[-1]-q[0]),
                         "drive_command_saturation_fraction": float(np.mean(np.abs(tau)>6.279)),
                         "positive_command_negative_speed_fraction": float(np.mean((tau>0)&(v<-1))),
                         "mit_send_interval_ms": stats(col("motor_control", "mit_actual_send_interval_ms")[mask])})
            for label, hit in (("rear_above_50_rad_s", np.abs(v)>50), ("rear_net_above_7_2_rad", np.abs(q-q[0])>7.2),
                               ("steering_above_85_deg", np.abs(s)>np.deg2rad(85))):
                if hit.any():
                    idx = np.flatnonzero(hit)[0]
                    item[label+"_after_first_send_ms"] = float((t[idx]-t[0])*1000)
            imask = (imu[:, 0]>=beg)&(imu[:, 0]<=end+.15)
            for label, hit in (("training_fallen_gravity_threshold", np.abs(gy)>.5), ("inversion", gz>0)):
                hits = np.flatnonzero(imask&hit)
                item[label+"_bag_s"] = float(imu[hits[0], 0]) if len(hits) else None
        attempts.append(item)
    summary["attempts"] = attempts

    # Re-examine the archived Isaac experiment, not a new simulator evaluation.
    training_path = root/"isaaclab_train/reports/motor_limits_09221945_91850_2026-10-01/motor_limit_samples.npz"
    train = np.load(training_path)
    start = np.ones_like(train["episode_id"], bool)
    start[1:] = train["episode_id"][1:] != train["episode_id"][:-1]
    target = train["processed_target"][:, :, 1]
    summary["archived_isaac_comparison"] = {
        "path": str(training_path.relative_to(root)), "sha256": sha(training_path),
        "episode_starts": int(start.sum()), "startup_rear_requested_nm": stats(target[start]),
        "non_startup_rear_abs_peak_nm": float(np.max(np.abs(target[~start]))),
        "startup_rear_velocity_after_20ms_rad_s": stats(train["velocity_after_rad_s"][:, -1, :, 1][start]),
        "positive_startup_rear_velocity_count": int((train["velocity_after_rad_s"][:, -1, :, 1][start]>0).sum()),
        "non_startup_request_exceedances": int((np.abs(target[~start])>6.28+1e-5).sum()),
        "feedforward_buffer_equals_processed_target": bool(np.array_equal(train["physx_effort_target_nm"][:, :, :, 1],
                                                                         np.repeat(target[:, None, :], 8, axis=1)))}
    summary["events"] = {k:d["text"].get(k, []) for k in ("motor_event", "policy_event", "kickstands_command")}
    summary["non_recorder_logs"] = [v for v in d["text"]["rosout"] if v["node"] != "rosbag2_recorder"]
    summary["read_only_model_sign_check"] = model_axes(root)
    source_paths = ["deploy/source/trackstand_deploy/trackstand_deploy/"+name for name in (
        "policy_controller.py", "policy.py", "contracts.py", "motor_mit_io.py", "can_protocol.py", "mit_protocol.py", "rc_kickstand.py", "imu_publisher.py")]
    summary["source_hashes"] = {path:sha(root/path) for path in source_paths}
    (out/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    names = ["gravity_y", "gravity_z", "angvel_x_rad_s", "angvel_y_rad_s", "angvel_z_rad_s", "estimated_vx_m_s", "placeholder_vy_m_s",
             "steering_pos_rad", "rear_pos_rad", "steering_vel_rad_s", "rear_vel_rad_s", "sin_steer", "cos_steer"]
    with (out/"policy_samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["bag_time_s", *names, "raw_steer", "raw_rear", "unclipped_steer_rad", "rear_request_nm",
                         "limited_steer_rad", "counterfactual_raw_steer", "counterfactual_raw_rear"])
        writer.writerows(np.column_stack([obs, raw[:, 1:], scaled, limited[:, 0], counterfactual]))
    selected = ("sequence", "monotonic_time_s", "steering_position_rad", "drive_velocity_rad_s", "requested_steering_target_rad",
                "requested_drive_target_nm", "bounded_drive_target_nm", "drive_damping_torque_nm", "drive_final_torque_nm",
                "drive_encoded_current_a", "mit_encoded_position_rad", "drive_pair_age_ms", "mit_actual_send_interval_ms")
    with (out/"active_motor_commands.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["bag_time_s", *selected])
        writer.writerows(np.column_stack([m[active, 0], *[col("motor_control", n)[active] for n in selected]]))

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out/(name+".png"), dpi=160)
        fig.savefig(out/(name+".svg"))
        plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(18, 12), squeeze=False)
    for j, item in enumerate(attempts):
        beg, end = item["first_s"], item["last_inference_s"]+.12
        cm = (m[:, 0]>=beg-.03)&(m[:, 0]<=end)
        sm = (ms[:, 0]>=beg-.03)&(ms[:, 0]<=end)
        im = (imu[:, 0]>=beg-.03)&(imu[:, 0]<=end)
        t = m[cm, 0]-beg
        ax = axes[0, j]
        ax.step(t, np.rad2deg(col("motor_control", "requested_steering_target_rad")[cm]), where="post", label="Clamped policy target")
        ax.plot(ms[sm, 0]-beg, np.rad2deg(ms[sm, 1]), label="A3 feedback")
        ax.set_ylim(-105, 190 if j==2 else 110)
        ax.axhline(90, color="gray", ls=":")
        ax.set_title(f"Attempt {j+1}, bag t={beg:.3f} s")
        ax.legend(fontsize=7)
        ax = axes[1, j]
        ax.step(t, col("motor_control", "drive_final_torque_nm")[cm], where="post", label="Outgoing C0 request")
        ax.plot(ms[sm, 0]-beg, ms[sm, 7], label="A1/C0 current-derived torque")
        ax.axhline(6.28, color="gray", ls=":")
        ax.set_ylim(-7, 7)
        ax.legend(fontsize=7)
        axes[2, j].plot(ms[sm, 0]-beg, ms[sm, 4], label="A2 signed rear speed")
        axes[2, j].axhline(0, color="gray", ls=":")
        axes[2, j].set_ylim(-70, 70)
        axes[3, j].plot(imu[im, 0]-beg, gy[im], label="gravity_y")
        axes[3, j].plot(imu[im, 0]-beg, gz[im], label="gravity_z")
        axes[3, j].axhline(.5, color="gray", ls=":")
        axes[3, j].axhline(-.5, color="gray", ls=":")
        axes[3, j].set_ylim(-1.1, 1.1)
        axes[3, j].legend(fontsize=7)
        for ax in axes[:, j]:
            ax.grid(alpha=.2)
        axes[3, j].set_xlabel("Seconds since first inference (record timestamps)")
    for ax, name in zip(axes[:, 0], ("Steering (deg)", "Rear torque (N m)", "Rear speed (rad/s)", "Projected gravity (g)")):
        ax.set_ylabel(name)
    save(fig, "attempts")

    fig, axes = plt.subplots(2, 4, figsize=(17, 6), squeeze=False)
    for j, (l, h) in enumerate(zip(cuts[:-1], cuts[1:])):
        t = obs[l:h, 0]-obs[l, 0]
        axes[0, j].plot(t, np.rad2deg(scaled[l:h, 0]), label="Unclipped request")
        axes[0, j].plot(t, np.rad2deg(limited[l:h, 0]), label="Sent ROS steering target")
        axes[0, j].axhline(90, color="gray", ls=":")
        axes[0, j].set_title(f"Attempt {j+1}")
        axes[1, j].plot(t, scaled[l:h, 1], label="Scaled policy request")
        axes[1, j].axhline(6.28, color="gray", ls=":", label="Motor-I/O bound")
        for ax in axes[:, j]:
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        axes[1, j].set_xlabel("Seconds since first inference")
    axes[0, 0].set_ylabel("Steering request (deg)")
    axes[1, 0].set_ylabel("Rear request (N m)")
    save(fig, "policy_requests")

    fig, axes = plt.subplots(4, 3, figsize=(14, 11), squeeze=False)
    for j, idx in enumerate((0, 1, 3)):
        beg = attempts[idx]["first_s"]
        cm = (m[:, 0]>=beg)&(m[:, 0]<=beg+.35)
        sm = (ms[:, 0]>=beg)&(ms[:, 0]<=beg+.35)
        om = (obs[:, 0]>=beg)&(obs[:, 0]<=beg+.35)
        t = m[cm, 0]-beg
        axes[0, j].step(t, np.rad2deg(col("motor_control", "requested_steering_target_rad")[cm]), where="post", label="Policy target after clamp")
        axes[0, j].plot(ms[sm, 0]-beg, np.rad2deg(ms[sm, 1]), label="A3 steering feedback")
        axes[0, j].set_title(f"Attempt {idx+1}: first 350 ms")
        axes[1, j].step(t, col("motor_control", "drive_final_torque_nm")[cm], where="post", label="Outgoing C0 torque request")
        axes[1, j].plot(ms[sm, 0]-beg, ms[sm, 7], label="Current-derived torque")
        axes[2, j].plot(ms[sm, 0]-beg, ms[sm, 4], label="Recorded A2 rear speed")
        axes[3, j].plot(obs[om, 0]-beg, raw[om, 2], label="Recorded raw rear action")
        axes[3, j].plot(obs[om, 0]-beg, counterfactual[om, 1], ls="--", label="Offline flipped rear inputs")
        for ax in axes[:, j]:
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        axes[3, j].set_xlabel("Seconds since first inference")
    for ax, name in zip(axes[:, 0], ("Steering (deg)", "Rear torque (N m)", "Rear speed (rad/s)", "Raw rear action")):
        ax.set_ylabel(name)
    save(fig, "startup")

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(a["kickstands_up"][:, 0], a["kickstands_up"][:, 1], label="RC up request", lw=.8)
    axes[0].plot(pt[:, 0], col("policy_timing", "policy_enabled"), label="Policy enabled", lw=.8)
    axes[0].legend()
    axes[1].plot(pt[:, 0], col("policy_timing", "cycle_interval_ms"), label="Policy tick interval")
    axes[1].axhline(20, color="gray", ls=":")
    axes[1].set_ylabel("Policy interval (ms)")
    def with_gaps(values):
        display = values.copy()
        display[1:][np.diff(m[:, 0])>.1] = np.nan
        return display
    axes[2].plot(m[:, 0], with_gaps(col("motor_control", "cycle_interval_ms")), label="Motor tick interval", lw=.6)
    axes[2].set_ylabel("Motor interval (ms)")
    axes[3].plot(m[:, 0], with_gaps(col("motor_control", "drive_pair_age_ms")), label="Rear encoder-pair age", lw=.6)
    axes[3].axhline(50, color="red", ls=":", label="50 ms watchdog")
    axes[3].set_ylabel("Feedback age (ms)")
    axes[3].set_xlabel("Elapsed bag time (s)")
    for ax in axes:
        ax.grid(alpha=.2)
    save(fig, "timing")
    gallery = "".join(f'<h2>{html.escape(title)}</h2><a href="{name}.svg"><img style="max-width:100%" src="{name}.png"></a>' for name, title in (
        ("attempts", "Four attempts: measured targets, feedback and tilt"), ("policy_requests", "Unclipped policy requests"),
        ("startup", "First 350 ms; dashed replay is NOT a corrected real-bike experiment"), ("timing", "Scheduling and watchdog evidence")))
    (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Deployment 2026-10-12</title>'
                                 '<h1>Trackstand deployment diagnosis — 2026-10-12</h1><p><a href="README.md">Report</a> · '
                                 '<a href="summary.json">Statistics</a> · <a href="policy_samples.csv">Policy data</a> · '
                                 '<a href="active_motor_commands.csv">Motor-command data</a></p>'+gallery)
    print(json.dumps({"attempt_samples": [v["policy_samples"] for v in attempts],
                      "model": model.name, "replay_error": summary["onnx_replay_max_error"],
                      "scaling_error": scale_error, "rear_equation_error": summary["rear_control_equation_max_error_nm"]}, indent=2))


if __name__ == "__main__":
    main()
