"""Reproducible statistics and plots for the three diagnostic recordings."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rpy_from_xyzw(q):
    x, y, z, w = (q / np.linalg.norm(q, axis=1, keepdims=True)).T
    return np.column_stack([
        np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
        np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)),
        np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
    ])


def statistics(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    return {
        "n": len(values), "min": float(values.min()), "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
        "max": float(values.max()), "mean": float(values.mean()), "std": float(values.std()),
        "rms": float(np.sqrt(np.mean(values ** 2))),
    }


def runs(times, mask, maximum_gap_s=None):
    """Boolean runs; the final tick of each run is included using a median interval."""
    changes = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    step = np.median(np.diff(times)) if len(times) > 1 else 0
    intervals = []
    for a, b in changes.reshape(-1, 2):
        splits = [] if maximum_gap_s is None else (a + 1 + np.flatnonzero(np.diff(times[a:b]) > maximum_gap_s)).tolist()
        bounds = [int(a), *splits, int(b)]
        intervals.extend(zip(bounds[:-1], bounds[1:]))
    return [
        {"start_s": float(times[a]), "end_s": float(times[b - 1] + step),
         "duration_s": float(times[b - 1] + step - times[a]), "samples": int(b - a),
         "first_index": int(a), "last_index": int(b - 1)}
        for a, b in intervals
    ]


def column(array, schema, name):
    return array[:, 1 + schema.index(name)]


def load(cache, stage):
    arrays = dict(np.load(cache / f"{stage}.npz"))
    details = json.loads((cache / f"{stage}.json").read_text())
    return arrays, details


def valid_imu_rows(imu):
    valid = np.all(np.isfinite(imu), axis=1) & (np.max(np.abs(imu[:, 2:6]), axis=1) <= 1.001)
    valid &= np.max(np.abs(imu[:, 6:9]), axis=1) < 1000.0
    return valid


def correlation_and_gain(reference, measured):
    if len(reference) < 3 or np.std(reference) == 0 or np.std(measured) == 0:
        return {"n": len(reference)}
    return {"n": len(reference), "correlation": float(np.corrcoef(reference, measured)[0, 1]),
            "gain": float(np.dot(reference, measured) / np.dot(reference, reference)),
            "error_rms": float(np.sqrt(np.mean((reference - measured) ** 2)))}


def encoder_consistency(arrays, details, active_only=False):
    mc = arrays["motor_control"]
    fields = details["schemas"]["motor_control"]
    get = lambda name: column(mc, fields, name)
    t = get("monotonic_time_s")
    grid = np.arange(t[0] + .03, t[-1] - .03, .005)
    result = {}
    for motor in ("steering", "drive"):
        q = np.unwrap(get(motor + "_position_rad"))
        rate = (np.interp(grid + .01, t, q) - np.interp(grid - .01, t, q)) / .02
        velocity = np.interp(grid, t, get(motor + "_velocity_rad_s"))
        mask = np.abs(rate) > .1
        # Exclude holes in the stored record, including Stage A's damaged interval.
        left = np.searchsorted(t, grid - .01)
        right = np.searchsorted(t, grid + .01)
        mask &= t[np.minimum(right, len(t)-1)] - t[left] < .04
        if active_only:
            mask &= np.interp(grid, t, get("gate_enabled")) > .5
        result[motor] = correlation_and_gain(rate[mask], velocity[mask])
    return result


def imu_consistency(arrays):
    imu = arrays["imu"]
    imu = imu[valid_imu_rows(imu)]
    t = imu[:, 1]
    q = imu[:, 2:6]
    # inv(q_before) * q_after expresses incremental rotation in the body frame.
    before, after = q[:-4], q[4:]
    av, aw = -before[:, :3], before[:, 3:]
    bv, bw = after[:, :3], after[:, 3:]
    vector = aw * bv + bw * av + np.cross(av, bv)
    scalar = (aw * bw)[:, 0] - np.sum(av * bv, axis=1)
    flip = scalar < 0
    vector[flip] *= -1
    scalar[flip] *= -1
    norm = np.linalg.norm(vector, axis=1)
    angle = 2 * np.arctan2(norm, scalar)
    dt = t[4:] - t[:-4]
    rates = vector * np.divide(angle, norm, out=np.ones_like(angle) * 2, where=norm > 1e-12)[:, None] / dt[:, None]
    gyro = np.column_stack([np.interp((t[4:] + t[:-4]) / 2, t, imu[:, 6 + j]) for j in range(3)])
    mask = (dt > .03) & (dt < .05)
    return {axis: correlation_and_gain(rates[mask & (np.abs(rates[:, i]) > .02), i],
                                      gyro[mask & (np.abs(rates[:, i]) > .02), i])
            for i, axis in enumerate(("x", "y", "z"))}


def replay_policy(arrays, details, repository):
    import hashlib
    import onnxruntime as ort
    observation = arrays["observation"][:, 1:].astype(np.float32)
    recorded = arrays["raw_action"][:, 1:]
    assert len(observation) == len(recorded)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    name = "trackstand_rl_rsl_2026-09-17_19-46-37_model_41050.onnx"
    model = repository / "deploy/source/trackstand_deploy/models" / name
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    inferred = np.vstack([session.run(None, {"observation": value[None]})[0] for value in observation])
    enabled_commands = arrays["motor_command"][arrays["motor_command"][:, 1] > .5, 2:]
    assert len(enabled_commands) == len(recorded)
    scaled = recorded * np.array([1.57, 6.28])
    scaled[:, 0] = np.clip(scaled[:, 0], -np.pi/2, np.pi/2)
    return {
        "model": name, "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "external_data_sha256": hashlib.sha256(Path(str(model) + ".data").read_bytes()).hexdigest(),
        "samples": len(observation), "max_absolute_action_error": float(np.max(np.abs(inferred - recorded))),
        "rms_action_error": float(np.sqrt(np.mean((inferred - recorded) ** 2))),
        "max_absolute_scaling_error": float(np.max(np.abs(scaled - enabled_commands))),
        "status_models": sorted(set(value["model"] for value in details["text"].get("status", []))),
    }


def powered_segments(arrays, details):
    mc = arrays["motor_control"]
    fields = details["schemas"]["motor_control"]
    get = lambda name: column(mc, fields, name)
    result = []
    for run in runs(mc[:, 0], get("gate_enabled") > .5):
        indices = np.arange(run["first_index"], run["last_index"] + 1)
        item = dict(run)
        item["steering_target_rad"] = float(np.median(get("requested_steering_target_rad")[indices]))
        item["drive_target_nm"] = float(np.median(get("requested_drive_target_nm")[indices]))
        item["steering_position_rad"] = statistics(get("steering_position_rad")[indices])
        item["steering_velocity_rad_s"] = statistics(get("steering_velocity_rad_s")[indices])
        item["steering_final_torque_nm"] = statistics(get("steering_final_torque_nm")[indices])
        item["steering_error_rad"] = statistics(get("steering_error_rad")[indices])
        for name in ("steering", "drive"):
            item[name + "_velocity_rad_s"] = statistics(get(name + "_velocity_rad_s")[indices])
            item[name + "_position_start_rad"] = float(get(name + "_position_rad")[indices[0]])
            item[name + "_position_end_rad"] = float(get(name + "_position_rad")[indices[-1]])
            item[name + "_position_delta_rad"] = item[name + "_position_end_rad"] - item[name + "_position_start_rad"]
            item[name + "_position_rad"] = statistics(get(name + "_position_rad")[indices])
        if len(indices) > 100 and item["steering_position_rad"]["std"] > .03:
            item["steering_dominant_frequency_hz"] = dominant_frequency(
                get("monotonic_time_s")[indices], get("steering_position_rad")[indices])
        torque = arrays["motor_torque"]
        for name, joint in (("steering", 0), ("drive", 1)):
            selected = (torque[:, 2] == joint) & (torque[:, 0] >= run["start_s"]) & (torque[:, 0] < run["end_s"])
            item[name + "_feedback_torque_nm"] = statistics(torque[selected, 3])
        result.append(item)
    return result


def current_response(arrays, details):
    mc = arrays["motor_control"]
    f = details["schemas"]["motor_control"]
    get = lambda name: column(mc, f, name)
    tq = arrays["motor_torque"]
    result = {}
    for motor, joint in (("steering", 0), ("drive", 1)):
        attempted = get(motor + "_send_attempted") > .5
        disabled = get("disable_attempted") > .5
        selected = attempted | disabled
        submitted = np.where(attempted[selected], get(motor + "_encoded_torque_nm")[selected], 0.0)
        received = tq[tq[:, 2] == joint]
        # Each motor has 1,289 control replies and six zero-current replies.
        assert len(received) == len(submitted), (motor, len(received), len(submitted))
        source_time = get("ros_time_s")[selected]
        y = received[:, 3]
        result[motor] = {
            "submitted_c0_count": len(submitted), "feedback_count": len(received),
            "receipt_minus_tick_start_ms": statistics((received[:, 1] - source_time) * 1000),
            "same_order_command_feedback": correlation_and_gain(submitted, y),
            "preceding_command_feedback": correlation_and_gain(submitted[:-1], y[1:]),
        }
    return result


def control_equation_check(arrays, details):
    mc = arrays["motor_control"]; f = details["schemas"]["motor_control"]
    get = lambda name: column(mc, f, name)
    mask = get("gate_enabled") > .5
    steering = np.clip(10 * get("steering_error_rad")[mask] - .2 * get("steering_velocity_rad_s")[mask], -5.6, 5.6)
    drive = np.clip(get("bounded_drive_target_nm")[mask] - .05 * get("drive_velocity_rad_s")[mask], -6.28, 6.28)
    return {"samples": int(mask.sum()),
            "steering_max_error_nm": float(np.max(np.abs(steering - get("steering_final_torque_nm")[mask]))),
            "drive_max_error_nm": float(np.max(np.abs(drive - get("drive_final_torque_nm")[mask])))}


def dominant_frequency(times, values):
    grid = np.arange(times[0], times[-1], .005)
    series = np.interp(grid, times, values)
    series -= series.mean()
    spectrum = np.abs(np.fft.rfft(series * np.hanning(len(series)))) ** 2
    frequencies = np.fft.rfftfreq(len(series), .005)
    selected = (frequencies >= 2) & (frequencies <= 50)
    return float(frequencies[selected][np.argmax(spectrum[selected])])


def summarize(stage, arrays, details):
    result = {"counts": details["counts"], "files": details["files"],
              "metadata_duration_s": details["duration_s"], "count_differences": details["count_differences"]}
    result["topics"] = {}
    for name, value in arrays.items():
        span = value[-1, 0] - value[0, 0]
        result["topics"][name] = {
            "n": len(value), "first_s": float(value[0, 0]), "last_s": float(value[-1, 0]),
            "rate_over_present_span_hz": float((len(value) - 1) / span) if span else 0,
            "record_interval_ms": statistics(np.diff(value[:, 0]) * 1000),
        }
    mc = arrays["motor_control"]
    fields = details["schemas"]["motor_control"]
    get = lambda name: column(mc, fields, name)
    result["motor"] = {
        "dry_run_values": np.unique(get("dry_run")).tolist(),
        "sequence_skips": int(np.sum(np.maximum(np.diff(get("sequence")) - 1, 0))),
        "sequence_resets": int(np.sum(np.diff(get("sequence")) < 0)),
        "enabled_runs": runs(mc[:, 0], get("gate_enabled") > 0.5),
        "fault_runs": runs(mc[:, 0], get("fault_latched") > 0.5),
    }
    start_s = details["start_ns"] * 1e-9
    result["clock_checks"] = {
        "motor_record_minus_tick_ms": statistics((mc[:, 0] - (get("ros_time_s") - start_s)) * 1000),
        "imu_publication_interval_ms": statistics(np.diff(arrays["imu"][:, 1]) * 1000),
        "imu_record_minus_publish_ms": statistics((arrays["imu"][:, 0] - (arrays["imu"][:, 1] - start_s)) * 1000),
        "imu_publication_gaps_over_50ms": [
            {"before_record_s": float(arrays["imu"][i, 0]), "after_record_s": float(arrays["imu"][i+1, 0]),
             "before_publication_s": float(arrays["imu"][i, 1] - start_s),
             "after_publication_s": float(arrays["imu"][i+1, 1] - start_s),
             "gap_ms": float((arrays["imu"][i+1, 1] - arrays["imu"][i, 1]) * 1000)}
            for i in np.flatnonzero(np.diff(arrays["imu"][:, 1]) > .05)
        ],
    }
    names = [
        "cycle_interval_ms", "work_duration_ms", "command_age_ms", "steering_pair_age_ms", "drive_pair_age_ms",
        "steering_position_query_ms", "steering_velocity_query_ms", "drive_position_query_ms", "drive_velocity_query_ms",
        "steering_position_rad", "steering_velocity_rad_s", "drive_position_rad", "drive_velocity_rad_s",
        "steering_final_torque_nm", "drive_final_torque_nm", "requested_drive_target_nm", "requested_steering_target_rad",
        "steering_send_duration_ms", "drive_send_duration_ms", "steering_position_age_ms", "steering_velocity_age_ms",
        "drive_position_age_ms", "drive_velocity_age_ms",
    ]
    result["motor"]["statistics"] = {name: statistics(get(name)) for name in names}
    active = get("gate_enabled") > .5
    if np.any(active):
        result["motor"]["active_statistics"] = {name: statistics(get(name)[active]) for name in names}
        consecutive = active[1:] & active[:-1]
        result["motor"]["active_consecutive_steering_torque_step_nm"] = statistics(
            np.abs(np.diff(get("steering_final_torque_nm"))[consecutive]))
    flags = ["requested_enable", "gate_enabled", "fault_latched", "steering_target_clipped", "drive_target_clipped",
             "steering_slew_limited", "steering_torque_saturated", "drive_torque_saturated",
             "steering_send_attempted", "steering_send_success", "drive_send_attempted", "drive_send_success", "disable_attempted"]
    result["motor"]["flag_counts"] = {name: int(np.sum(get(name) > 0.5)) for name in flags}
    failures = [f"{motor}_{kind}_query_failures" for motor in ("steering", "drive") for kind in ("position", "velocity")]
    result["motor"]["query_failure_counters"] = {
        name: {"first": float(get(name)[0]), "last": float(get(name)[-1]),
               "increase": float(get(name)[-1] - get(name)[0])} for name in failures
    }
    result["events"] = {
        name: details["text"].get(name, []) for name in ("rosout", "motor_event", "policy_event", "parameter_events")
    }
    result["motor_status_unique"] = []
    for record in details["text"].get("motor_status", []):
        key = {k: v for k, v in record.items() if k != "t" and not k.endswith("_age_ms")}
        if key not in result["motor_status_unique"]:
            result["motor_status_unique"].append(key)
    if "policy_timing" in arrays:
        pt = arrays["policy_timing"]
        pf = details["schemas"]["policy_timing"]
        pg = lambda name: column(pt, pf, name)
        result["policy"] = {
            "statistics": {name: statistics(pg(name)) for name in pf if name.endswith("_ms")},
            "enabled_runs": runs(pt[:, 0], pg("policy_enabled") > 0.5),
            "sequence_skips": int(np.sum(np.maximum(np.diff(pg("sequence")) - 1, 0))),
            "sequence_resets": int(np.sum(np.diff(pg("sequence")) < 0)),
            "successes": int(np.sum(pg("inference_success") > 0.5)),
        }
        enabled = pg("policy_enabled") > .5
        result["policy"]["active_statistics"] = {name: statistics(pg(name)[enabled]) for name in pf if name.endswith("_ms")}
    if "raw_action" in arrays:
        result["raw_action"] = {name: statistics(arrays["raw_action"][:, i + 1]) for i, name in enumerate(("steering", "drive"))}
        result["observation"] = [statistics(arrays["observation"][:, i + 1]) for i in range(13)]
    imu_all = arrays["imu"]
    valid_imu = np.all(np.isfinite(imu_all), axis=1) & (np.max(np.abs(imu_all[:, 2:6]), axis=1) <= 1.001)
    valid_imu &= np.max(np.abs(imu_all[:, 6:9]), axis=1) < 1000.0
    imu = imu_all[valid_imu]
    q = imu[:, 2:6]
    angles = rpy_from_xyzw(q)
    result["imu"] = {
        "invalid_indices": np.flatnonzero(~valid_imu).tolist(),
        "invalid_record_times_s": imu_all[~valid_imu, 0].tolist(),
        "quaternion_norm": statistics(np.linalg.norm(q, axis=1)),
        "rpy_deg": {name: statistics(np.rad2deg(angles[:, i])) for i, name in enumerate(("roll", "pitch", "yaw"))},
        "gyro_rad_s": {name: statistics(imu[:, 6 + i]) for i, name in enumerate(("x", "y", "z"))},
        "covariance_first_values": np.unique(imu[:, 12:15], axis=0).tolist(),
    }
    if "motor_command" in arrays:
        cmd = arrays["motor_command"]
        result["commands"] = {
            "unique_values": np.unique(cmd[:, 1:], axis=0).tolist() if stage == "C" else None,
            "enabled_runs": runs(cmd[:, 0], cmd[:, 1] > 0.5, maximum_gap_s=.08),
            "enabled_count": int(np.sum(cmd[:, 1] > 0.5)),
        }
    if "motor_torque" in arrays:
        tq = arrays["motor_torque"]
        result["torque_feedback"] = {
            name: statistics(tq[tq[:, 2] == i, 3]) for i, name in enumerate(("steering", "drive"))
        }
    return result


def plot_stage_c(arrays, details, output):
    mc = arrays["motor_control"]; fields = details["schemas"]["motor_control"]
    get = lambda name: column(mc, fields, name)
    elapsed = get("monotonic_time_s") - get("monotonic_time_s")[0] + mc[0, 0]
    enabled = get("gate_enabled") > .5
    tq = arrays["motor_torque"]
    # Convert feedback receipt stamps to the same ROS elapsed origin as tick diagnostics.
    torque_elapsed = tq[:, 1] - get("ros_time_s")[0] + elapsed[0]
    for filename, interval in (("stage_C_steering_overview.png", (49, 106)),
                               ("stage_C_steering_zoom.png", (72.3, 73.0))):
        fig, axes = plt.subplots(5, 1, figsize=(13, 11), sharex=True)
        select = (elapsed >= interval[0]) & (elapsed <= interval[1])
        x = elapsed[select]
        axes[0].plot(x, np.rad2deg(get("steering_position_rad")[select]), label="A3 steering position")
        target = np.where(enabled, get("bounded_steering_target_rad"), np.nan)
        axes[0].plot(x, np.rad2deg(target[select]), label="enabled fixed target", color="black")
        axes[0].set_ylabel("angle [deg]"); axes[0].legend(loc="upper right")
        axes[1].plot(x, get("steering_velocity_rad_s")[select], label="A2 speed")
        axes[1].set_ylabel("speed [rad/s]")
        torque_select = (tq[:, 2] == 0) & (torque_elapsed >= interval[0]) & (torque_elapsed <= interval[1])
        for field, label, style in (("steering_p_torque_nm", "P term", "-"),
                                    ("steering_d_torque_nm", "D term", "-"),
                                    ("steering_final_torque_nm", "C0 commanded torque", "--")):
            axes[2].plot(x, get(field)[select], style, label=label, alpha=.8)
        feedback_time, feedback_value = break_at_gaps(torque_elapsed[torque_select], tq[torque_select, 3], .03)
        axes[2].plot(feedback_time, feedback_value, label="current-derived feedback", color="black", alpha=.65)
        axes[2].set_ylabel("torque [N m]"); axes[2].legend(loc="upper right", ncol=2)
        axes[3].plot(x, get("steering_position_age_ms")[select], label="A3 receipt age")
        axes[3].plot(x, get("steering_velocity_age_ms")[select], label="A2 receipt age", alpha=.65)
        axes[3].plot(x, get("cycle_interval_ms")[select], label="control interval", alpha=.7)
        axes[3].set_ylabel("time [ms]"); axes[3].legend(loc="upper right")
        axes[4].plot(x, get("gate_enabled")[select], label="gate enabled", color="green")
        axes[4].plot(x, get("fault_latched")[select], label="fault latched", color="red")
        axes[4].set_ylabel("flag"); axes[4].set_ylim(-.1, 1.2); axes[4].legend(loc="upper right")
        for axis in axes:
            axis.grid(alpha=.25)
        axes[-1].set_xlabel("seconds from Stage C bag start (tick clock; feedback receipt time)")
        axes[-1].set_xlim(*interval)
        fig.suptitle("Stage C: fixed commands, policy absent, wheels untouched and free")
        fig.tight_layout(); fig.savefig(output / filename, dpi=150); plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(elapsed, np.where(enabled, get("requested_drive_target_nm"), np.nan), label="requested drive torque")
    axes[0].plot(elapsed, np.where(enabled, get("drive_final_torque_nm"), np.nan), label="C0 torque (includes damping)")
    rear = tq[:, 2] == 1
    axes[0].plot(torque_elapsed[rear], tq[rear, 3], ".", markersize=2, label="current-derived feedback")
    axes[0].set_ylabel("torque [N m]"); axes[0].legend(loc="upper right")
    axes[1].plot(elapsed, np.rad2deg(get("drive_position_rad")))
    axes[1].set_ylabel("rear angle [deg]")
    axes[2].plot(elapsed, get("drive_velocity_rad_s")); axes[2].set_ylabel("rear speed [rad/s]")
    axes[2].set_xlabel("seconds from Stage C bag start"); axes[2].set_xlim(84, 105)
    for axis in axes:
        axis.grid(alpha=.25)
    fig.suptitle("Stage C: rear-wheel +/-0.05 N m tests (steering also enabled at target zero)")
    fig.tight_layout(); fig.savefig(output / "stage_C_drive.png", dpi=150); plt.close(fig)


def break_at_gaps(times, values, maximum_gap_s):
    """Do not depict nonexistent feedback between powered runs as a measured line."""
    split_indices = np.flatnonzero(np.diff(times) > maximum_gap_s) + 1
    return np.insert(times, split_indices, np.nan), np.insert(values, split_indices, np.nan)


def plot_stage_b(arrays, details, output):
    action = arrays["raw_action"]
    command = arrays["motor_command"]; command = command[command[:, 1] > .5]
    pt = arrays["policy_timing"]; pf = details["schemas"]["policy_timing"]
    pg = lambda name: column(pt, pf, name)
    obs = arrays["observation"]
    fig, axes = plt.subplots(5, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(action[:, 0], action[:, 1], ".", markersize=2, label="steering raw action")
    axes[0].plot(action[:, 0], action[:, 2], ".", markersize=2, label="drive raw action")
    axes[0].set_ylabel("raw action"); axes[0].legend(loc="upper right")
    axes[1].plot(command[:, 0], np.rad2deg(command[:, 2]), ".", markersize=2)
    axes[1].set_ylabel("steer target [deg]")
    axes[2].plot(command[:, 0], command[:, 3], ".", markersize=2); axes[2].set_ylabel("rear target [N m]")
    axes[3].plot(pt[:, 0], pg("cycle_interval_ms"), label="policy interval [ms]")
    axes[3].plot(pt[:, 0], pg("inference_ms"), label="inference duration [ms]")
    axes[3].set_ylabel("time [ms]"); axes[3].legend(loc="upper right")
    for channel, label in ((2, "gyro x"), (3, "gyro y"), (4, "gyro z")):
        axes[4].plot(obs[:, 0], obs[:, channel + 1], ".", markersize=2, label=label)
    axes[4].set_ylabel("gyro obs [rad/s]"); axes[4].legend(loc="upper right")
    axes[-1].set_xlabel("seconds from Stage B bag start"); axes[-1].set_xlim(0, 95)
    for axis in axes:
        axis.grid(alpha=.25)
    fig.suptitle("Stage B: model_41050, 1,718 inference steps, dry run (no C0 sent)")
    fig.tight_layout(); fig.savefig(output / "stage_B_policy.png", dpi=150); plt.close(fig)


def plot_stage_a(arrays, output):
    imu = arrays["imu"]; valid = valid_imu_rows(imu); imu = imu[valid]
    angles = np.rad2deg(rpy_from_xyzw(imu[:, 2:6]))
    state = arrays["motor_state"]
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(imu[:, 0], angles[:, 0], label="roll")
    axes[0].plot(imu[:, 0], angles[:, 1], label="pitch")
    axes[0].set_ylabel("angle [deg]"); axes[0].legend(loc="upper right")
    for j, axis in enumerate("xyz"):
        axes[1].plot(imu[:, 0], imu[:, 6 + j], label=f"gyro {axis}")
    axes[1].set_ylabel("gyro [rad/s]"); axes[1].legend(loc="upper right")
    axes[2].plot(state[:, 0], np.rad2deg(state[:, 1]), label="steering")
    axes[2].set_ylabel("position [deg]")
    axes[3].plot(state[:, 0], state[:, 3], label="steering")
    axes[3].plot(state[:, 0], state[:, 4], label="rear")
    axes[3].set_ylabel("speed [rad/s]"); axes[3].legend(loc="upper right")
    for axis in axes:
        axis.axvspan(106.31774, 107.11775, color="red", alpha=.18)
        axis.grid(alpha=.25)
    axes[-1].set_xlabel("seconds from Stage A bag start; red band = damaged/missing-record interval")
    fig.suptitle("Stage A: disabled sensors; one malformed IMU record excluded")
    fig.tight_layout(); fig.savefig(output / "stage_A_sensors.png", dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    bags = {stage: load(args.cache_dir, stage) for stage in "ABC"}
    results = {stage: summarize(stage, *bags[stage]) for stage in "ABC"}
    for stage in "ABC":
        arrays, details = bags[stage]
        results[stage]["encoder_consistency"] = encoder_consistency(arrays, details, active_only=stage == "C")
        results[stage]["imu_gyro_quaternion_consistency"] = imu_consistency(arrays)
    repository = Path(__file__).resolve().parents[3]
    results["B"]["policy_replay"] = replay_policy(*bags["B"], repository)
    results["C"]["powered_segments"] = powered_segments(*bags["C"])
    results["C"]["current_response"] = current_response(*bags["C"])
    results["C"]["control_equation_check"] = control_equation_check(*bags["C"])
    arrays, details = bags["C"]
    get = lambda name: column(arrays["motor_control"], details["schemas"]["motor_control"], name)
    times = arrays["motor_control"][:, 0]
    first_reversal = (times >= 71.637) & (times < 71.899)
    results["C"]["first_reversal_threshold_times_s"] = {
        str(limit): float(times[first_reversal & (np.abs(np.rad2deg(get("steering_position_rad"))) > limit)][0])
        for limit in (5, 10, 15)
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    plot_stage_a(bags["A"][0], args.output_dir)
    plot_stage_b(*bags["B"], args.output_dir)
    plot_stage_c(*bags["C"], args.output_dir)
    for stage, result in results.items():
        print(stage, "dry_run", result["motor"]["dry_run_values"], "encoder", result["encoder_consistency"],
              "imu", result["imu_gyro_quaternion_consistency"], flush=True)
    print("B policy replay", results["B"]["policy_replay"])
    print("C current response", results["C"]["current_response"])
    print("C control equation", results["C"]["control_equation_check"])
    for run in results["C"]["powered_segments"]:
        print("C segment", json.dumps(run))


if __name__ == "__main__":
    main()
