"""Reproducible, read-only analysis of the corrected-sign ROS deployment.

Only the requested output directory is written. No ROS executor, CAN socket,
motor command, simulator step or training/configuration edit is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml


def make_plots(output, arrays, details, summary, phase_samples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = lambda key, name: arrays[key][:, 1 + details["schemas"][key].index(name)]
    motor, state, obs, raw, imu = [arrays[name] for name in (
        "motor_control", "motor_state", "observation", "raw_action", "imu")]
    gravity, roll = gravity_and_roll(imu[:, 2:6])
    active = col("motor_control", "steering_send_success") > .5
    count = len(summary["attempts"])

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / (name + ".png"), dpi=150)
        fig.savefig(output / (name + ".svg"))
        plt.close(fig)

    def shade(axes, end):
        if end > 1:
            for ax in axes:
                ax.axvspan(1, end, color="gray", alpha=.12)
                ax.axvline(1, color="gray", lw=.7, ls=":")
        for ax in axes:
            ax.grid(alpha=.2)
            ax.legend(fontsize=6, loc="best")

    for name, early in (("attempts", False), ("startup", True)):
        fig, axes = plt.subplots(5, count, figsize=(5 * count, 13), squeeze=False)
        for j, item in enumerate(summary["attempts"]):
            begin = item["first_s"]
            end = min(item["last_policy_s"] + .03 - begin, .8) if early else item["last_policy_s"] + .03 - begin
            mm = active & (motor[:, 0] >= begin - .01) & (motor[:, 0] <= begin + end)
            sm = (state[:, 0] >= begin - .02) & (state[:, 0] <= begin + end)
            im = (imu[:, 0] >= begin - .02) & (imu[:, 0] <= begin + end)
            om = (obs[:, 0] >= begin) & (obs[:, 0] <= begin + end)
            t = motor[mm, 0] - begin
            axes[0, j].plot(imu[im, 0] - begin, np.rad2deg(roll[im]), label="Quaternion Euler roll")
            axes[0, j].axhline(-30, color="red", ls=":", label="Training 30 deg envelope")
            axes[1, j].step(t, np.rad2deg(col("motor_control", "requested_steering_target_rad")[mm]),
                            where="post", label="Clipped policy target")
            axes[1, j].step(t, np.rad2deg(col("motor_control", "mit_encoded_position_rad")[mm]),
                            where="post", label="Actually submitted MIT target")
            axes[1, j].plot(state[sm, 0] - begin, np.rad2deg(state[sm, 1]), label="A3 position feedback")
            axes[2, j].step(t, col("motor_control", "drive_final_torque_nm")[mm], where="post", label="Final C0 torque request")
            axes[2, j].plot(state[sm, 0] - begin, state[sm, 7], label="Current-derived rear torque")
            axes[2, j].axhline(6.28, color="gray", ls=":")
            axes[2, j].axhline(-6.28, color="gray", ls=":")
            axes[3, j].plot(state[sm, 0] - begin, state[sm, 4], label="A2 signed rear speed")
            axes[3, j].axhline(0, color="gray", lw=.7)
            axes[4, j].step(obs[om, 0] - begin, obs[om, 9], where="post", label="Wrapped rear-position observation")
            axes[0, j].set_title(f"Attempt {j + 1}: start {begin:.3f} bag s")
            axes[4, j].set_xlabel("Seconds since first inference (record time)")
            for ax in axes[:, j]:
                ax.set_xlim(-.02, end)
            shade(axes[:, j], end)
        for ax, label in zip(axes[:, 0], ("Roll (deg)", "Steering (deg)", "Rear torque (N m)",
                                         "Rear speed (rad/s)", "Rear phase (rad)")):
            ax.set_ylabel(label)
        fig.suptitle("Corrected-sign deployment: user reports early stand contact (<1 s after lifting); "
                     "gray marks later support-affected data, not an exact contact time", fontsize=10)
        save(fig, name)

    fig, axes = plt.subplots(2, count, figsize=(5 * count, 7), squeeze=False)
    for j, item in enumerate(summary["attempts"]):
        begin = item["first_s"]
        mask = (obs[:, 0] >= begin) & (obs[:, 0] <= item["last_policy_s"])
        t = obs[mask, 0] - begin
        axes[0, j].plot(t, np.rad2deg(raw[mask, 1] * 1.57), label="Raw action x 1.57")
        axes[0, j].plot(t, np.rad2deg(np.clip(raw[mask, 1] * 1.57, -np.pi / 2, np.pi / 2)), label="Clipped ROS target")
        axes[1, j].plot(t, raw[mask, 2] * 6.28, label="Raw action x 6.28")
        axes[1, j].axhline(6.28, color="gray", ls=":", label="Motor-I/O request cap")
        axes[1, j].axhline(-6.28, color="gray", ls=":")
        axes[0, j].set_title(f"Attempt {j + 1}")
        axes[1, j].set_xlabel("Seconds since first inference")
        shade(axes[:, j], t[-1])
    axes[0, 0].set_ylabel("Unclipped steering request (deg)")
    axes[1, 0].set_ylabel("Unclipped rear request (N m)")
    save(fig, "policy_requests")

    fig, axes = plt.subplots(2, count, figsize=(5 * count, 7), squeeze=False)
    for j, item in enumerate(summary["attempts"]):
        begin = item["first_s"]
        mask = active & (motor[:, 0] >= begin) & (motor[:, 0] < begin + .8)
        sm = (state[:, 0] >= begin) & (state[:, 0] < begin + .8)
        t = motor[mask, 0] - begin
        axes[0, j].step(t, col("motor_control", "bounded_drive_target_nm")[mask], where="post", label="Policy request after cap")
        axes[0, j].step(t, col("motor_control", "drive_damping_torque_nm")[mask], where="post", label="Host damping contribution")
        axes[0, j].step(t, col("motor_control", "drive_final_torque_nm")[mask], where="post", label="Final C0 request")
        axes[0, j].plot(state[sm, 0] - begin, state[sm, 7], label="Published measured current x Kt")
        axes[1, j].plot(state[sm, 0] - begin, state[sm, 4], label="Rear speed")
        axes[1, j].axhline(0, color="gray", lw=.7)
        axes[0, j].set_title(f"Attempt {j + 1}: early rear loop")
        axes[1, j].set_xlabel("Seconds since first inference")
        shade(axes[:, j], .8)
    axes[0, 0].set_ylabel("Rear request/feedback (N m)")
    axes[1, 0].set_ylabel("Rear speed (rad/s)")
    save(fig, "rear_control")

    fig, axes = plt.subplots(2, count, figsize=(5 * count, 7), squeeze=False)
    for j, item in enumerate(summary["attempts"]):
        begin = item["first_s"]
        mask = (phase_samples[:, 0] >= begin) & (phase_samples[:, 0] < begin + .8)
        p = phase_samples[mask]
        axes[0, j].step(p[:, 0] - begin, p[:, 1], where="post", label="Recorded wrapped rear angle")
        axes[0, j].plot(p[:, 0] - begin, p[:, 2], label="Offline unwrap (same record)")
        axes[1, j].plot(p[:, 0] - begin, p[:, 4], label="Recorded rear policy request")
        axes[1, j].plot(p[:, 0] - begin, p[:, 5], ls="--", label="Replay: negative angle branch +2pi")
        axes[1, j].axhline(6.28, color="gray", ls=":")
        axes[0, j].set_title(f"Attempt {j + 1}: phase sensitivity, NOT an alternate trial")
        axes[1, j].set_xlabel("Seconds since first inference")
        shade(axes[:, j], .8)
    axes[0, 0].set_ylabel("Rear position (rad)")
    axes[1, 0].set_ylabel("Unclipped request (N m)")
    save(fig, "rear_phase")

    fig, axes = plt.subplots(2, count, figsize=(5 * count, 7), squeeze=False)
    scans = np.genfromtxt(output / "tracking_lag_scan.csv", delimiter=",", names=True, dtype=None, encoding="utf8")
    for j, item in enumerate(summary["attempts"]):
        for kind, label in (("policy_target", "Clipped policy to A3"), ("encoded_target", "Sent MIT to A3"),
                            ("first_second_encoded", "Sent MIT to A3, first second")):
            subset = scans[(scans["attempt"] == j + 1) & (scans["comparison"] == kind)]
            axes[0, j].plot(subset["lag_ms"], subset["correlation"], label=label)
        kind = scans[(scans["attempt"] == j + 1) & (scans["comparison"] == "rear_torque_published")]
        axes[1, j].plot(kind["lag_ms"], kind["correlation"], label="Final C0 to published current-derived torque")
        axes[0, j].set_title(f"Attempt {j + 1}: descriptive alignment, not pure dead time")
        axes[1, j].set_xlabel("Held-input shift (ms)")
        shade(axes[:, j], 0)
    axes[0, 0].set_ylabel("Steering input/feedback correlation")
    axes[1, 0].set_ylabel("Rear request/feedback correlation")
    save(fig, "tracking_alignment")

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    pt = arrays["policy_timing"]
    axes[0].plot(arrays["kickstands_up"][:, 0], arrays["kickstands_up"][:, 1], lw=.8, label="RC up request")
    axes[0].plot(pt[:, 0], col("policy_timing", "policy_enabled"), lw=.8, label="Policy enabled")
    axes[1].plot(pt[:, 0], col("policy_timing", "cycle_interval_ms"), lw=.8, label="Policy interval")
    axes[1].axhline(20, color="gray", ls=":")
    axes[2].plot(motor[:, 0], col("motor_control", "steering_pair_age_ms"), lw=.8, label="Steering A3/A2 pair age")
    axes[2].plot(motor[:, 0], col("motor_control", "drive_pair_age_ms"), lw=.8, label="Rear A3/A2 pair age")
    axes[2].axhline(50, color="red", ls=":", label="Motor feedback watchdog")
    axes[3].plot(pt[:, 0], col("policy_timing", "imu_age_ms"), lw=.8, label="Policy callback IMU age")
    axes[3].plot(pt[:, 0], col("policy_timing", "motor_age_ms"), lw=.8, label="Policy callback motor-state age")
    axes[0].set_ylabel("Enabled (bool)")
    axes[1].set_ylabel("Interval (ms)")
    axes[2].set_ylabel("CAN feedback age (ms)")
    axes[3].set_ylabel("ROS callback age (ms)")
    axes[3].set_xlabel("Elapsed bag time (s)")
    for ax in axes:
        ax.set_xlim(0, min(details["duration_s"], 42))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    save(fig, "timing")


OBS_NAMES = (
    "gravity_y", "gravity_z", "angular_x_rad_s", "angular_y_rad_s", "angular_z_rad_s",
    "estimated_vx_m_s", "placeholder_vy_m_s", "steering_position_rad", "rear_position_rad",
    "steering_velocity_rad_s", "rear_velocity_rad_s", "sin_steer", "cos_steer",
)


def stats(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    return {"n": int(len(values)), "min": float(values.min()), "median": float(np.median(values)),
            "max": float(values.max()), "mean": float(values.mean()),
            "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
            "rms": float(np.sqrt(np.mean(values * values)))}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def gravity_and_roll(quaternion):
    q = quaternion / np.linalg.norm(quaternion, axis=1, keepdims=True)
    x, y, z, w = q.T
    gy = -2 * (y * z + w * x)
    gz = -(1 - 2 * (x * x + y * y))
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return np.column_stack((gy, gz)), roll


def relationship(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    if len(x) < 10 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return {"n": int(len(x))}
    fit = np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]
    return {"n": int(len(x)), "correlation": float(np.corrcoef(x, y)[0, 1]),
            "slope": float(fit[0]), "offset": float(fit[1]), "error": stats(y - x)}


def held_values(times, values, query):
    indices = np.searchsorted(times, query, side="right") - 1
    return values[np.clip(indices, 0, len(values) - 1)]


def lag_profile(command_t, command, feedback_t, feedback, upper=.2):
    """Descriptive closed-loop input/output alignment, not a firmware dead time.

    Use zero-order-held input, never smooth feedback, and fit offset/gain at
    each 1 ms shift. Each comparison excludes pre-command and end-of-input data.
    """
    rows = []
    for delay in np.arange(0, upper + .0005, .001):
        mask = (feedback_t - delay >= command_t[0]) & (feedback_t <= command_t[-1])
        x = held_values(command_t, command, feedback_t[mask] - delay)
        y = feedback[mask]
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if len(x) < 20 or np.std(x) < .05 or np.std(y) < .05:
            continue
        gain, bias = np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]
        rows.append([delay * 1000, np.corrcoef(x, y)[0, 1], gain, bias,
                     np.sqrt(np.mean((y - gain * x - bias) ** 2)), len(x)])
    if not rows:
        return {"n": 0}, np.empty((0, 6))
    result = np.asarray(rows)
    best = result[np.nanargmax(result[:, 1])]
    return {"lag_ms": float(best[0]), "correlation": float(best[1]),
            "fit_gain": float(best[2]), "fit_bias": float(best[3]),
            "fit_residual_rms": float(best[4]), "n": int(best[5]),
            "warning": "Closed-loop held-input correlation lag; not universal CAN/firmware/physical delay."}, result


def quaternion_gyro_check(imu):
    t, q = imu[:, 1], imu[:, 2:6]
    n = 4
    av, aw, bv, bw = -q[:-n, :3], q[:-n, 3:], q[n:, :3], q[n:, 3:]
    vector = aw * bv + bw * av + np.cross(av, bv)
    scalar = (aw * bw)[:, 0] - np.sum(av * bv, axis=1)
    flip = scalar < 0
    vector[flip] *= -1
    scalar[flip] *= -1
    norm = np.linalg.norm(vector, axis=1)
    dt = t[n:] - t[:-n]
    rates = vector * np.divide(2 * np.arctan2(norm, scalar), norm,
                              out=np.ones_like(norm) * 2, where=norm > 1e-12)[:, None] / dt[:, None]
    center = (t[n:] + t[:-n]) / 2
    result = {}
    for axis, name in enumerate("xyz"):
        mask = (dt > .02) & (dt < .065) & (np.abs(rates[:, axis]) > .1)
        gyro = np.interp(center, t, imu[:, 6 + axis])
        base = relationship(rates[mask, axis], gyro[mask])
        if int(mask.sum()) < 10:
            result[name] = base
            continue
        shifts = []
        for shift in np.arange(-.06, .0601, .001):
            g = np.interp(center + shift, t, imu[:, 6 + axis])
            valid = mask & (center + shift >= t[0]) & (center + shift <= t[-1])
            shifts.append((shift * 1000, relationship(rates[valid, axis], g[valid]).get("correlation", -2)))
        best = max(shifts, key=lambda row: row[1])
        base.update(best_gyro_relative_time_shift_ms=float(best[0]), best_shift_correlation=float(best[1]))
        result[name] = base
    return result


def encoder_consistency(state, position_column, speed_column):
    """40 ms-scale central differences, avoiding missing-data gaps."""
    t, q = state[:, 0], np.unwrap(state[:, position_column])
    center_t, rate, speed = [], [], []
    for index in range(2, len(t) - 2):
        left = np.searchsorted(t, t[index] - .02, side="right") - 1
        right = np.searchsorted(t, t[index] + .02)
        if left < 0 or right >= len(t):
            continue
        dt = t[right] - t[left]
        if not .025 <= dt <= .065 or np.max(np.diff(t[left:right + 1])) > .04:
            continue
        # Compare the same interval, not an averaged angle derivative with an
        # instantaneous speed during rapid oscillation. A2 samples remain
        # asynchronous with A3; publication time is not acquisition time.
        v = np.trapezoid(state[left:right + 1, speed_column], t[left:right + 1]) / dt
        if abs(v) < 1:
            continue
        center_t.append(t[index])
        rate.append((q[right] - q[left]) / dt)
        speed.append(v)
    x, y = np.asarray(rate), np.asarray(speed)
    return relationship(x, y), np.column_stack((center_t, rate, speed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    arrays = dict(np.load(args.cache / "data.npz"))
    details = json.loads((args.cache / "details.json").read_text())
    col = lambda key, name: arrays[key][:, 1 + details["schemas"][key].index(name)]
    obs, raw, motor, state, imu, timing = [arrays[name] for name in (
        "observation", "raw_action", "motor_control", "motor_state", "imu", "policy_timing")]
    inferred = col("policy_timing", "inference_success") > .5
    enabled = arrays["motor_command"][arrays["motor_command"][:, 1] > .5]
    assert len(obs) == len(raw) == len(enabled) == int(inferred.sum())
    assert not details["issues"]
    metadata_path = Path(details["bag"]).parent / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text())["rosbag2_bagfile_information"]
    expected_counts = {row["topic_metadata"]["name"]: row["message_count"]
                       for row in metadata["topics_with_message_count"]}
    assert all(details["counts"].get(topic, 0) == count for topic, count in expected_counts.items())
    assert sum(details["counts"].values()) == metadata["message_count"]
    assert details["first_ns"] == metadata["starting_time"]["nanoseconds_since_epoch"]
    assert details["last_ns"] - details["first_ns"] == metadata["duration"]["nanoseconds"]
    summary = {key: details[key] for key in ("bag", "bytes", "sha256", "duration_s", "counts", "issues")}
    summary.update(decoded_message_count=sum(details["counts"].values()), metadata_matches=True,
                   user_conditions="Both wheels grounded; no manual disturbance. Only rear encoder sign changed. "
                   "User reports kickstand contact within less than one second after each release; exact contact time unrecorded.")
    gravity, roll = gravity_and_roll(imu[:, 2:6])

    # Independent reconstruction from the received sensor topics. Candidate
    # search allows cross-topic recorder ordering; it does not fabricate a new
    # acquisition timestamp or use linear interpolation to make samples match.
    sensor_errors, source_offsets, sensor_rows = [], [], []
    for t, observation in zip(obs[:, 0], obs[:, 1:]):
        imask, smask = np.abs(imu[:, 0] - t) < .09, np.abs(state[:, 0] - t) < .09
        icandidates = np.column_stack((gravity[imask], imu[imask, 6:9])).astype(np.float32)
        scandidates = state[smask, 1:5].astype(np.float32)
        goal_i, goal_s = observation[:5], observation[[7, 8, 9, 10]]
        ii = np.argmin(np.max(np.abs(icandidates - goal_i), axis=1))
        si = np.argmin(np.max(np.abs(scandidates - goal_s), axis=1))
        ie, se = icandidates[ii] - goal_i, scandidates[si] - goal_s
        sensor_errors.append(np.r_[ie, se])
        source_offsets.append([t - imu[imask, 0][ii], t - state[smask, 0][si]])
        sensor_rows.append([imu[imask, 0][ii], state[smask, 0][si]])
    sensor_errors = np.asarray(sensor_errors)
    assert np.max(np.abs(sensor_errors)) < 1e-6
    summary["observation_audit"] = {
        "sensor_reconstruction_max_error": float(np.max(np.abs(sensor_errors))),
        "finite_all": bool(np.isfinite(obs).all()),
        "vx_radius_max_error": float(np.max(np.abs(obs[:, 6] - .07 * obs[:, 11]))),
        "vy_all_zero": bool(np.all(obs[:, 7] == 0)),
        "sin_steer_max_error": float(np.max(np.abs(np.sin(obs[:, 8]) - obs[:, 12]))),
        "cos_steer_max_error": float(np.max(np.abs(np.cos(obs[:, 8]) - obs[:, 13]))),
        "quaternion_norm": stats(np.linalg.norm(imu[:, 2:6], axis=1)),
        "quaternion_gyro": quaternion_gyro_check(imu),
        "range_by_term": {name: stats(obs[:, i + 1]) for i, name in enumerate(OBS_NAMES)},
        "sensor_match_warning": "Match within +/-90ms recorded-topic window. Exact value match is not a hardware-acquisition-age proof.",
    }
    for name, pcol, vcol in (("steering", 1, 3), ("rear", 2, 4)):
        result, samples = encoder_consistency(state, pcol, vcol)
        summary["observation_audit"][name + "_a3_derivative_vs_a2"] = result
        np.savetxt(output / (name + "_encoder_consistency.csv"), samples, delimiter=",",
                   header="bag_time_s,a3_interval_rate_rad_s,interval_average_a2_rad_s", comments="")

    model = root / "deploy/source/trackstand_deploy/models/trackstand_rl_rsl_2026-09-22_19-45-06_model_91850.onnx"
    checkpoint = root / "isaaclab_train/logs/rsl_rl/trackstand_rl_rsl/2026-09-22_19-45-06/model_91850.pt"
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    replay = session.run(None, {"observation": obs[:, 1:].astype(np.float32)})[0]
    assert np.max(np.abs(replay - raw[:, 1:])) < 1e-5
    torch.set_num_threads(1)
    actor = torch.nn.Sequential(torch.nn.Linear(13, 32), torch.nn.ELU(), torch.nn.Linear(32, 32),
                                torch.nn.ELU(), torch.nn.Linear(32, 2))
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
    actor.load_state_dict({key[4:]: value for key, value in sd.items() if key.startswith("mlp.")}, strict=True)
    with torch.no_grad():
        checkpoint_replay = actor(torch.from_numpy(obs[:, 1:].astype(np.float32))).numpy()
    scaled = raw[:, 1:] * [1.57, 6.28]
    limited = np.column_stack((np.clip(scaled[:, 0], -np.pi / 2, np.pi / 2), scaled[:, 1]))
    assert np.max(np.abs(limited - enabled[:, 2:])) < 1e-9
    ideal = np.zeros((1, 13), np.float32)
    ideal[0, [1, 12]] = [-1, 1]
    ideal_raw = session.run(None, {"observation": ideal})[0][0]
    summary["policy"] = {
        "model": model.name, "model_sha256": digest(model), "checkpoint_sha256": digest(checkpoint),
        "onnx_max_error": float(np.max(np.abs(replay - raw[:, 1:]))),
        "checkpoint_max_error": float(np.max(np.abs(checkpoint_replay - raw[:, 1:]))),
        "scale_command_max_error": float(np.max(np.abs(limited - enabled[:, 2:]))),
        "ideal_raw": ideal_raw.tolist(), "ideal_scaled": (ideal_raw * [1.57, 6.28]).tolist(),
        "steering_requests_beyond_90": int((np.abs(scaled[:, 0]) > np.pi / 2).sum()),
        "rear_requests_beyond_6_28": int((np.abs(scaled[:, 1]) > 6.28).sum()),
        "sample_count": len(raw), "raw_steering": stats(raw[:, 1]), "raw_rear": stats(raw[:, 2]),
    }
    active = col("motor_control", "drive_send_success") > .5
    steer_sent = col("motor_control", "steering_send_success") > .5
    mtime = col("motor_control", "monotonic_time_s")
    summary["actuation"] = {
        "rear_successful_send_ticks": int(active.sum()), "steering_successful_send_ticks": int(steer_sent.sum()),
        "rear_equation_error_nm": float(np.max(np.abs(
            np.clip(col("motor_control", "bounded_drive_target_nm")[active]
                    - .05 * col("motor_control", "drive_velocity_rad_s")[active], -6.28, 6.28)
            - col("motor_control", "drive_final_torque_nm")[active]))),
        "mit_encoded_target_error_rad": stats(col("motor_control", "mit_encoded_position_rad")[steer_sent]
                                               - col("motor_control", "slewed_steering_target_rad")[steer_sent]),
        "rear_current_encoding_error_nm": stats(col("motor_control", "drive_encoded_torque_nm")[active]
                                                 - col("motor_control", "drive_final_torque_nm")[active]),
        "gains_ranges": {name: np.unique(col("motor_control", name)[steer_sent]).tolist() for name in (
            "mit_encoded_kp_nm_per_motor_rad", "mit_encoded_kd_nm_s_per_motor_rad", "mit_position_max_rad",
            "mit_velocity_max_rad_s", "mit_torque_max_nm", "mit_native_zero_rad")},
        "policy_timing": {name: stats(col("policy_timing", name)[inferred]) for name in (
            "cycle_interval_ms", "work_duration_ms", "inference_ms", "imu_age_ms", "motor_age_ms", "rc_age_ms")},
        "motor_timing": {name: stats(col("motor_control", name)[active]) for name in (
            "cycle_interval_ms", "work_duration_ms", "command_age_ms", "steering_pair_age_ms", "drive_pair_age_ms",
            "mit_actual_send_interval_ms", "mit_f1_age_ms")},
    }
    assert summary["actuation"]["rear_equation_error_nm"] < 1e-9
    cuts = np.r_[0, np.flatnonzero(np.diff(obs[:, 0]) > .1) + 1, len(obs)]
    attempts, lag_rows = [], []
    for number, (lo, hi) in enumerate(zip(cuts[:-1], cuts[1:]), 1):
        lo, hi = int(lo), int(hi)
        beg, end = obs[lo, 0], obs[hi - 1, 0]
        mask = active & (motor[:, 0] >= beg - .01) & (motor[:, 0] <= end + .03)
        smask = steer_sent & (motor[:, 0] >= beg - .01) & (motor[:, 0] <= end + .03)
        state_mask = (state[:, 0] >= beg - .01) & (state[:, 0] <= end + .03)
        imu_mask = (imu[:, 0] >= beg) & (imu[:, 0] <= end + .05)
        first_second = obs[lo:hi, 0] < beg + 1
        speed = col("motor_control", "drive_velocity_rad_s")[mask]
        torque = col("motor_control", "drive_final_torque_nm")[mask]
        commanded = col("motor_control", "requested_steering_target_rad")[smask]
        encoded = col("motor_control", "mit_encoded_position_rad")[smask]
        measured = col("motor_control", "steering_position_rad")[smask]
        # All cross-topic tracking graphs/lags below use recorded timestamps.
        # Within-motor diagnostics offer additional cached-sample consistency,
        # but comparing monotonic time with elapsed bag time would be invalid.
        lag_policy, scan_policy = lag_profile(motor[smask, 0], commanded, state[state_mask, 0], state[state_mask, 1])
        lag_encoded, scan_encoded = lag_profile(motor[smask, 0], encoded, state[state_mask, 0], state[state_mask, 1])
        shifted_time = state[state_mask, 0] - lag_encoded.get("lag_ms", 0) / 1000
        shifted_valid = (shifted_time >= motor[smask, 0][0]) & (state[state_mask, 0] <= motor[smask, 0][-1])
        aligned_steering_error = state[state_mask, 1][shifted_valid] - held_values(
            motor[smask, 0], encoded, shifted_time[shifted_valid])
        lag_first, scan_first = lag_profile(motor[smask & (motor[:, 0] < beg + 1), 0],
                                             col("motor_control", "mit_encoded_position_rad")[smask & (motor[:, 0] < beg + 1)],
                                             state[state_mask & (state[:, 0] < beg + 1), 0],
                                             state[state_mask & (state[:, 0] < beg + 1), 1])
        for kind, scan in (("policy_target", scan_policy), ("encoded_target", scan_encoded), ("first_second_encoded", scan_first)):
            if len(scan):
                lag_rows.extend([[number, kind, *row] for row in scan])
        item = {
            "attempt": number, "first_s": float(beg), "last_policy_s": float(end), "policy_count": hi - lo,
            "initial_observation": obs[lo, 1:].tolist(), "initial_raw": raw[lo, 1:].tolist(),
            "initial_scaled": scaled[lo].tolist(), "rear_send_ticks": int(mask.sum()),
            "rear_velocity_rad_s": stats(speed), "steering_deg": stats(np.rad2deg(measured)),
            "rear_command_nm": stats(torque), "rear_feedback_nm": stats(state[state_mask, 7]),
            "raw_steer": stats(raw[lo:hi, 1]), "raw_rear": stats(raw[lo:hi, 2]),
            "steer_clip_count": int((np.abs(scaled[lo:hi, 0]) > np.pi / 2).sum()),
            "rear_clip_count": int((np.abs(scaled[lo:hi, 1]) > 6.28).sum()),
            "first_second_steer_clip_count": int((np.abs(scaled[lo:hi, 0][first_second]) > np.pi / 2).sum()),
            "first_second_rear_clip_count": int((np.abs(scaled[lo:hi, 1][first_second]) > 6.28).sum()),
            "first_second_policy_count": int(first_second.sum()),
            "policy_to_position_alignment": lag_policy, "encoded_to_position_alignment": lag_encoded,
            "first_second_encoded_to_position_alignment": lag_first,
            "steering_simultaneous_target_error_deg": stats(np.rad2deg(measured - commanded)),
            "steering_simultaneous_encoded_error_deg": stats(np.rad2deg(measured - encoded)),
            "steering_time_aligned_unscaled_encoded_error_deg": stats(np.rad2deg(aligned_steering_error)),
            "roll_deg": stats(np.rad2deg(roll[imu_mask])),
            "motor_events": [v for v in details["text"]["motor_event"] if beg - .05 <= v["t"] <= end + .15],
            "policy_events": [v for v in details["text"]["policy_event"] if beg - .05 <= v["t"] <= end + .15],
        }
        frame_dt = np.diff(col("policy_timing", "monotonic_time_s")[inferred][lo:hi])
        requested_changes = np.diff(limited[lo:hi, 0])
        effective_speeds = np.abs(requested_changes) / frame_dt
        item["steering_requested_change_rad_s"] = stats(effective_speeds)
        item["steering_requested_change_above_23_9_count"] = int((effective_speeds > 23.9).sum())
        item["steering_full_180deg_reversals_count"] = int((np.abs(requested_changes) > np.pi - 1e-6).sum())
        item["steering_slew_limited_tick_count"] = int((col("motor_control", "steering_slew_limited")[smask] > .5).sum())
        item["query_failures_increase"] = {
            name: int(col("motor_control", name)[mask][-1] - col("motor_control", name)[mask][0]) for name in (
                "steering_position_query_failures", "steering_velocity_query_failures",
                "drive_position_query_failures", "drive_velocity_query_failures")}
        # Match each published rear torque estimate to a held final C0 request.
        # The feedback cache's own acquisition timestamp is not in motor_state;
        # these are recorded-topic comparisons, not a firmware-latency claim.
        rear_sample_t, rear_tau = state[state_mask, 0], state[state_mask, 7]
        rear_command = col("motor_control", "drive_encoded_torque_nm")[mask]
        rear_alignment, rear_scan = lag_profile(motor[mask, 0], rear_command, rear_sample_t, rear_tau, upper=.08)
        item["rear_command_to_published_torque_alignment"] = rear_alignment
        finite = np.isfinite(rear_tau) & (rear_sample_t >= motor[mask, 0][0]) & (rear_sample_t <= motor[mask, 0][-1])
        rear_held = held_values(motor[mask, 0], rear_command, rear_sample_t)
        item["rear_torque_simultaneous_error_nm"] = stats(rear_tau[finite] - rear_held[finite])
        lag = rear_alignment.get("lag_ms", 0) / 1000
        rear_shifted = held_values(motor[mask, 0], rear_command, rear_sample_t - lag)
        aligned_finite = finite & (rear_sample_t - lag >= motor[mask, 0][0])
        item["rear_torque_aligned_error_nm"] = stats(rear_tau[aligned_finite] - rear_shifted[aligned_finite])
        requested_rear = col("motor_control", "requested_drive_target_nm")[mask]
        held = (requested_rear[1:] == requested_rear[:-1]) & (np.diff(mtime[mask]) < .025)
        reverses = (torque[1:] * torque[:-1] < 0) & (np.abs(torque[1:]) >= .5) & (np.abs(torque[:-1]) >= .5)
        held_flips = held & reverses
        item["rear_strong_torque_sign_flips"] = int(reverses.sum())
        item["rear_strong_torque_flips_same_policy_request"] = int(held_flips.sum())
        examples = np.flatnonzero(held_flips)[:8]
        item["same_policy_request_torque_reversal_examples"] = [
            {"bag_s_before": float(motor[mask, 0][i]), "bag_s_after": float(motor[mask, 0][i + 1]),
             "policy_request_nm": float(requested_rear[i]),
             "rear_speed_before_rad_s": float(speed[i]), "rear_speed_after_rad_s": float(speed[i + 1]),
             "final_torque_before_nm": float(torque[i]), "final_torque_after_nm": float(torque[i + 1])}
            for i in examples]
        high_command = aligned_finite & (np.abs(rear_shifted) >= 3)
        item["rear_abs_command_at_least3_count"] = int(high_command.sum())
        item["rear_feedback_under_half_high_command_count"] = int(
            (high_command & (np.abs(rear_tau) < .5 * np.abs(rear_shifted))).sum())
        if len(rear_scan):
            lag_rows.extend([[number, "rear_torque_published", *row] for row in rear_scan])
        # Compare direct firmware F1 feedback to A3, and to sent MIT targets.
        # Approximate kernel receive times are derived from diagnostic age:
        # age is sampled after tick entry, so uncertainty is bounded by work.
        positions = np.flatnonzero(smask)
        positions = positions[1:]
        rx_time = mtime[positions] - col("motor_control", "mit_f1_age_ms")[positions] / 1000
        f1_joint = -(col("motor_control", "mit_feedback_position_motor_rad")[positions]
                     - col("motor_control", "mit_native_zero_rad")[positions])
        unique = np.r_[True, np.diff(rx_time) > .0005]
        rx_time, f1_joint = rx_time[unique], f1_joint[unique]
        f1_lag, _ = lag_profile(mtime[smask], encoded, rx_time, f1_joint)
        item["encoded_to_f1_estimated_receive_alignment"] = f1_lag
        f1_lag["timestamp_warning"] = "Kernel reception estimated from age sampled after loop entry; not firmware acquisition time."
        direct_f1 = -(col("motor_control", "mit_feedback_position_motor_rad")[smask]
                      - col("motor_control", "mit_native_zero_rad")[smask])
        item["f1_joint_vs_a3_cached"] = relationship(measured, direct_f1)
        policy_starts = col("policy_timing", "monotonic_time_s")[inferred][lo:hi]
        forward_latencies = []
        for index, (target_pair, tick_start) in enumerate(zip(enabled[lo:hi, 2:], policy_starts)):
            eligible = smask & (mtime >= tick_start) & (mtime < tick_start + .08)
            exact = eligible & (np.abs(col("motor_control", "requested_steering_target_rad") - target_pair[0]) < 1e-9)
            exact &= np.abs(col("motor_control", "requested_drive_target_nm") - target_pair[1]) < 1e-9
            first = np.flatnonzero(exact)
            if len(first):
                forward_latencies.append((mtime[first[0]] - tick_start) * 1000)
        item["policy_tick_to_first_successful_dispatch_tick_ms"] = stats(forward_latencies)
        for label, condition in (("abs_gravity_y_0_5", np.abs(gravity[:, 0]) > .5),
                                 ("abs_euler_roll_20_deg", np.abs(roll) > np.deg2rad(20)),
                                 ("abs_euler_roll_30_deg", np.abs(roll) > np.deg2rad(30))):
            hits = np.flatnonzero(imu_mask & condition)
            item["first_" + label + "_s"] = float(imu[hits[0], 0]) if len(hits) else None
        first_t = mtime[mask][0]
        for label, hit in (("rear_forward_50", speed > 50), ("rear_backward_30", speed < -30)):
            index = np.flatnonzero(hit)
            item[label + "_after_first_send_ms"] = float((mtime[mask][index[0]] - first_t) * 1000) if len(index) else None
        rear_angle = np.unwrap(col("motor_control", "drive_position_rad")[mask])
        item["rear_net_rotation_rad"] = float(rear_angle[-1] - rear_angle[0])
        crossings = np.flatnonzero(np.abs(rear_angle - rear_angle[0]) > 7.2)
        item["rear_net_7_2_after_first_send_ms"] = float((mtime[mask][crossings[0]] - first_t) * 1000) if len(crossings) else None
        # Threshold crossings avoid counting tiny fluctuations around zero.
        direction, reversals = 0, []
        for tick, v in zip(mtime[mask], speed):
            current = 1 if v > 5 else -1 if v < -5 else 0
            if current and current != direction:
                if direction:
                    reversals.append(tick)
                direction = current
        item["rear_sign_reversal_count_over_5_rad_s"] = len(reversals)
        item["rear_reversal_spacing_ms"] = stats(np.diff(reversals) * 1000)
        pre = (imu[:, 0] >= beg - .8) & (imu[:, 0] < beg - .02)
        item["prestart_roll_deg"] = stats(np.rad2deg(roll[pre]))
        item["prestart_gyro_rad_s"] = {axis: stats(imu[pre, 6 + i]) for i, axis in enumerate("xyz")}
        attempts.append(item)
    summary["attempts"] = attempts
    # Input sensitivity, not alternate real trajectories. A branch change is
    # especially revealing because +/-pi wrapping discards whole-turn phase.
    x = obs[:, 1:].astype(np.float32)
    branch_x = x.copy()
    branch_x[branch_x[:, 8] < 0, 8] += 2 * np.pi
    branch_raw = session.run(None, {"observation": branch_x})[0]
    continuous_x, per_episode_zero_x = x.copy(), x.copy()
    wrap_events = []
    for number, (lo, hi) in enumerate(zip(cuts[:-1], cuts[1:]), 1):
        lo, hi = int(lo), int(hi)
        q = obs[lo:hi, 9]
        continuous_x[lo:hi, 8] = np.unwrap(q)
        per_episode_zero_x[lo:hi, 8] = np.unwrap(q) - q[0]
        for hit in np.flatnonzero(np.abs(np.diff(q)) > np.pi) + 1:
            index = lo + int(hit)
            wrap_events.append({"attempt": number, "policy_index": index, "bag_s": float(obs[index, 0]),
                                "since_first_inference_ms": float((obs[index, 0] - obs[lo, 0]) * 1000),
                                "rear_pos_before": float(obs[index - 1, 9]), "rear_pos_after": float(obs[index, 9]),
                                "rear_vel_before": float(obs[index - 1, 11]), "rear_vel_after": float(obs[index, 11]),
                                "raw_rear_before": float(raw[index - 1, 2]), "raw_rear_after": float(raw[index, 2]),
                                "same_state_positive_branch_raw_rear": float(branch_raw[index, 1])})
    continuous_raw = session.run(None, {"observation": continuous_x})[0]
    zero_referenced_raw = session.run(None, {"observation": per_episode_zero_x})[0]
    no_vx_x = x.copy()
    no_vx_x[:, 5] = 0
    no_vx_raw = session.run(None, {"observation": no_vx_x})[0]
    negative = x[:, 8] < 0
    summary["rear_position_representation"] = {
        "negative_phase_count": int(negative.sum()),
        "recorded_torque_on_negative_phase": stats(scaled[negative, 1]),
        "recorded_torque_on_nonnegative_phase": stats(scaled[~negative, 1]),
        "wrap_event_count": len(wrap_events), "wrap_events": wrap_events,
        "counterfactual_negative_branch_add_2pi": {
            "rear_over_limit_count": int((np.abs(branch_raw[:, 1]) > 1).sum()),
            "rear_torque": stats(branch_raw[:, 1] * 6.28),
        },
        "counterfactual_continuous_per_attempt": {
            "rear_over_limit_count": int((np.abs(continuous_raw[:, 1]) > 1).sum()),
            "rear_torque": stats(continuous_raw[:, 1] * 6.28),
        },
        "counterfactual_continuous_zero_per_attempt": {
            "rear_over_limit_count": int((np.abs(zero_referenced_raw[:, 1]) > 1).sum()),
            "rear_torque": stats(zero_referenced_raw[:, 1] * 6.28),
        },
        "counterfactual_vx_zero": {
            "rear_over_limit_count": int((np.abs(no_vx_raw[:, 1]) > 1).sum()),
            "rear_torque": stats(no_vx_raw[:, 1] * 6.28),
        },
        "warning": "Fixed-record input replays; alternate physics/contact/poses are NOT simulated. "
                   "Branch choice or zero reset alone is NOT an authorized/proven hardware fix.",
    }
    for item, (lo, hi) in zip(attempts, zip(cuts[:-1], cuts[1:])):
        first = np.arange(lo, hi)[obs[lo:hi, 0] < obs[lo, 0] + 1]
        item["first_second_counterfactual_rear_exceedances"] = {
            "recorded": int((np.abs(raw[first, 2]) > 1).sum()),
            "positive_branch": int((np.abs(branch_raw[first, 1]) > 1).sum()),
            "continuous": int((np.abs(continuous_raw[first, 1]) > 1).sum()),
            "continuous_zero_start": int((np.abs(zero_referenced_raw[first, 1]) > 1).sum()),
            "vx_zero": int((np.abs(no_vx_raw[first, 1]) > 1).sum()),
        }
    # Sending resumes after long RC-down gaps: discard the first send's
    # previous-episode interval and pre-enable F1 age from active-rate metrics.
    actual_intervals, live_f1_ages = [], []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        mask = steer_sent & (motor[:, 0] >= obs[lo, 0] - .01) & (motor[:, 0] <= obs[hi - 1, 0] + .03)
        positions = np.flatnonzero(mask)
        actual_intervals.extend(col("motor_control", "mit_actual_send_interval_ms")[positions[1:]])
        live_f1_ages.extend(col("motor_control", "mit_f1_age_ms")[positions[1:]])
    summary["actuation"]["motor_timing"]["mit_actual_send_interval_ms"] = stats(actual_intervals)
    summary["actuation"]["motor_timing"]["mit_f1_age_ms"] = stats(live_f1_ages)
    active_state = np.zeros(len(state), dtype=bool)
    for item in attempts:
        active_state |= (state[:, 0] >= item["first_s"]) & (state[:, 0] <= item["last_policy_s"])
    steer_tau = state[active_state, 6]
    summary["actuation"]["steering_current_derived_torque_nm"] = stats(steer_tau)
    summary["actuation"]["steering_current_derived_torque_above_5_6_count"] = int((np.abs(steer_tau) > 5.6).sum())
    with (output / "rear_phase_sensitivity.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["bag_s", "wrapped_rear_rad", "continuous_attempt_rear_rad", "continuous_zero_start_rear_rad",
                         "recorded_rear_request_nm", "positive_branch_replay_nm", "continuous_replay_nm",
                         "continuous_zero_replay_nm", "vx_zero_replay_nm"])
        writer.writerows(np.column_stack((obs[:, 0], x[:, 8], continuous_x[:, 8], per_episode_zero_x[:, 8], scaled[:, 1],
                                          branch_raw[:, 1] * 6.28, continuous_raw[:, 1] * 6.28,
                                          zero_referenced_raw[:, 1] * 6.28, no_vx_raw[:, 1] * 6.28)))
    summary["events"] = {name: details["text"].get(name, []) for name in ("motor_event", "policy_event", "kickstands_command")}
    summary["rosout"] = [row for row in details["text"]["rosout"] if row.get("node") != "rosbag2_recorder"]
    summary["recorded_models"] = sorted({row["model"] for row in details["text"]["status"]})
    summary["rc_average_hz"] = float((len(arrays["kickstands_up"]) - 1) / np.ptp(arrays["kickstands_up"][:, 0]))
    summary["disabled_command_count"] = int((arrays["motor_command"][:, 1] <= .5).sum())
    source_names = ("contracts.py", "policy_controller.py", "policy.py", "motor_mit_io.py", "can_protocol.py",
                    "mit_protocol.py", "imu_publisher.py", "imu_serial.py", "rc_kickstand.py", "kickstand_driver.py")
    summary["source_hashes"] = {str(path.relative_to(root)): digest(path) for path in (
        root / "deploy/source/trackstand_deploy/trackstand_deploy" / name for name in source_names)}
    summary["config_hash"] = digest(root / "deploy/source/trackstand_deploy/config/params.yaml")
    train_path = root / "isaaclab_train/reports/motor_limits_09221945_91850_2026-10-01/motor_limit_samples.npz"
    with np.load(train_path) as train:
        summary["archived_isaac"] = {
            "path": str(train_path.relative_to(root)), "sha256": digest(train_path),
            "rear_joint_position_policy": stats(train["position_before_policy_rad"][:, :, 1]),
            "rear_joint_position_beyond_pi_count": int((np.abs(train["position_before_policy_rad"][:, :, 1]) > np.pi).sum()),
            "steering_target": stats(train["processed_target"][:, :, 0]),
            "rear_target": stats(train["processed_target"][:, :, 1]),
        }
    with (output / "policy_samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["bag_time_s", *OBS_NAMES, "raw_steer", "raw_rear", "scaled_steer_rad", "scaled_rear_nm", "clipped_steer_rad"])
        writer.writerows(np.column_stack((obs, raw[:, 1:], scaled, limited[:, 0])))
    with (output / "tracking_lag_scan.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["attempt", "comparison", "lag_ms", "correlation", "fit_gain", "fit_bias", "fit_rms", "samples"])
        writer.writerows(lag_rows)
    selected = ("sequence", "monotonic_time_s", "requested_steering_target_rad", "slewed_steering_target_rad",
                "mit_encoded_position_rad", "steering_position_rad", "steering_velocity_rad_s", "requested_drive_target_nm",
                "bounded_drive_target_nm", "drive_damping_torque_nm", "drive_final_torque_nm", "drive_encoded_torque_nm",
                "drive_encoded_current_a", "drive_velocity_rad_s", "drive_position_rad", "command_age_ms",
                "steering_pair_age_ms", "drive_pair_age_ms", "mit_actual_send_interval_ms")
    with (output / "active_motor_commands.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["bag_time_s", *selected])
        writer.writerows(np.column_stack([motor[active, 0], *[col("motor_control", name)[active] for name in selected]]))
    phase_samples = np.column_stack((obs[:, 0], x[:, 8], continuous_x[:, 8], per_episode_zero_x[:, 8], scaled[:, 1],
                                    branch_raw[:, 1] * 6.28, continuous_raw[:, 1] * 6.28,
                                    zero_referenced_raw[:, 1] * 6.28, no_vx_raw[:, 1] * 6.28))
    make_plots(output, arrays, details, summary, phase_samples)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({"attempts": len(attempts), "policy_samples": len(raw), "model": model.name,
                      "sensor_reconstruction_max_error": summary["observation_audit"]["sensor_reconstruction_max_error"],
                      "policy_replay_max_error": summary["policy"]["onnx_max_error"],
                      "action_scaling_max_error": summary["policy"]["scale_command_max_error"],
                      "rear_equation_max_error": summary["actuation"]["rear_equation_error_nm"]}, indent=2))


if __name__ == "__main__":
    main()
