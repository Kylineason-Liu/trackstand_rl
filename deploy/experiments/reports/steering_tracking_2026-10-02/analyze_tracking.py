"""Read-only A/B/C tracking analysis; write new diagnostic results, never motor commands.

Input caches come from ../rosbag_ABC_2026-10-02/decode_bags.py. All control
intervals use recorded host monotonic timestamps. Angles are signed encoder
feedback; torque is estimated from measured Q-axis current, not a shaft sensor.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ENCODER_COUNT_RAD = 2 * np.pi / 16384


def stats(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    return {
        "n": int(len(values)), "min": float(values.min()), "max": float(values.max()),
        "median": float(np.median(values)), "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
    }


def settling_time(times, error, band, confirmation_s=.25):
    """First recorded time after the last out-of-band sample; require a .25 s tail.

    This is a finite-record criterion, not a guarantee of continued stability.
    None means either no qualifying settlement or too short a confirmation tail.
    """
    outside = np.flatnonzero(np.abs(error) > band + 1e-12)
    index = int(outside[-1] + 1) if len(outside) else 0
    if index >= len(times) or times[-1] - times[index] < confirmation_s:
        return None
    return float(times[index] * 1000)


def starts_and_stops(mask):
    return np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)


def first_crossing(times, position, initial, target, fraction):
    amplitude = abs(target - initial)
    if amplitude < ENCODER_COUNT_RAD / 2:
        return None
    reached = (position - initial) * np.sign(target - initial) >= fraction * amplitude
    indices = np.flatnonzero(reached)
    return None if not len(indices) else float(times[indices[0]] * 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    output = Path(__file__).resolve().parent
    stages = {}
    provenance = {}
    for stage in ("A", "B", "C"):
        cache_file = args.cache_dir / f"{stage}.npz"
        arrays = dict(np.load(cache_file))
        details = json.loads((args.cache_dir / f"{stage}.json").read_text())
        stages[stage] = arrays, details
        files = []
        for source in details["files"]:
            path = Path(source["path"])
            if not path.is_absolute():
                path = args.repository / path
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(block)
            actual = digest.hexdigest()
            if actual != source["sha256"]:
                raise ValueError(f"Bag changed since decoding: {path}")
            files.append({"path": source["path"], "sha256": actual})
        provenance[stage] = {
            "bags": files, "cache_sha256": hashlib.sha256(cache_file.read_bytes()).hexdigest(),
            "decoded_count": sum(details["counts"].values()),
            "metadata_discrepancy": details["count_differences"],
        }

    arrays, details = stages["C"]
    control = arrays["motor_control"]
    fields = details["schemas"]["motor_control"]
    get = lambda key: control[:, 1 + fields.index(key)]
    enabled = get("gate_enabled") > .5
    ranges = starts_and_stops(enabled)
    # Clock conversion: use each tick's ROS timestamp, not delayed bag-receipt time.
    # Current samples are selected by host receipt time; no command/reply ordering
    # assumption is needed for the initial-current-window statistics.
    current = arrays["motor_torque"][arrays["motor_torque"][:, 2] == 0]
    results = []
    segment_data = []
    for number, (start, stop) in enumerate(ranges, 1):
        t_absolute = get("monotonic_time_s")[start:stop]
        times = t_absolute - t_absolute[0]
        position = get("steering_position_rad")[start:stop]
        velocity = get("steering_velocity_rad_s")[start:stop]
        target = get("requested_steering_target_rad")[start:stop]
        torque = get("steering_final_torque_nm")[start:stop]
        if np.max(np.abs(target - target[0])) > 1e-12:
            raise ValueError("Expected one constant target per Stage C powered interval")
        initial = position[0]
        last_half = times >= max(0, times[-1] - .5)
        final_median = float(np.median(position[last_half]))
        changed = np.flatnonzero(np.abs(position - initial) > ENCODER_COUNT_RAD / 2)
        deadline_indices = np.flatnonzero(times < .020)
        index_20 = deadline_indices[-1]
        command_times = t_absolute - get("command_age_ms")[start:stop] / 1000
        updates = np.r_[0, 1 + np.flatnonzero(np.abs(np.diff(command_times)) > 1e-6)]
        lengths = np.diff(np.r_[updates, len(times)])
        # Discard first/last held-message groups: recording starts/stops mid-group.
        group_lengths = lengths[1:-1]
        source_times = t_absolute - get("steering_position_age_ms")[start:stop] / 1000
        fresh = np.r_[0, 1 + np.flatnonzero(np.abs(np.diff(source_times)) > 1e-6)]
        current_times = current[:, 1] - get("ros_time_s")[start]
        current_mask = (current_times >= 0) & (current_times <= times[-1])
        measured = current[current_mask, 3]
        receipt_times = current_times[current_mask]
        nonzero = np.flatnonzero(np.abs(measured) >= .010)
        first150 = (receipt_times >= 0) & (receipt_times < .150)
        initial150 = times < .150
        item = {
            "segment": number, "start_bag_s": float(control[start, 0]),
            "sampled_enabled_duration_ms": float(times[-1] * 1000), "control_ticks": int(stop - start),
            "successful_steering_sends": int(np.sum(get("steering_send_success")[start:stop] > .5)),
            "target_deg": float(np.rad2deg(target[0])), "initial_position_deg": float(np.rad2deg(initial)),
            "position_range_deg": [float(np.rad2deg(position.min())), float(np.rad2deg(position.max()))],
            "first_changed_position_control_tick_ms": None if not len(changed) else float(times[changed[0]] * 1000),
            "last_tick_before_20ms": {
                "elapsed_ms": float(times[index_20] * 1000), "ticks_before_deadline": int(len(deadline_indices)),
                "position_deg": float(np.rad2deg(position[index_20])),
                "position_receipt_age_ms": float(get("steering_position_age_ms")[start + index_20]),
            },
            "first_10percent_crossing_ms": first_crossing(times, position, initial, target[0], .1),
            "first_50percent_crossing_ms": first_crossing(times, position, initial, target[0], .5),
            "first_90percent_crossing_ms": first_crossing(times, position, initial, target[0], .9),
            "settling_within_target_plusminus_0p1deg_ms": settling_time(times, position - target, np.deg2rad(.1)),
            "settling_within_target_plusminus_0p3deg_ms": settling_time(times, position - target, np.deg2rad(.3)),
            "final_halfsecond_median_deg": float(np.rad2deg(final_median)),
            "final_halfsecond_std_deg": float(np.rad2deg(np.std(position[last_half]))),
            "final_median_target_error_deg": float(np.rad2deg(final_median - target[0])),
            "settling_around_observed_final_plusminus_one_encoder_count_ms": settling_time(
                times, position - final_median, ENCODER_COUNT_RAD),
            "control_interval_ms": stats(np.diff(t_absolute) * 1000),
            "observed_received_command_interval_ms": stats(np.diff(command_times[updates]) * 1000),
            "control_ticks_per_received_command_excluding_edge_groups": {
                str(int(value)): int(np.sum(group_lengths == value)) for value in np.unique(group_lengths)
            },
            "distinct_A3_receipt_timestamps": int(len(fresh)),
            "fraction_ticks_reusing_A3_receipt_timestamp": float(1 - len(fresh) / len(times)),
            "distinct_A3_receipt_interval_ms": stats(np.diff(source_times[fresh]) * 1000),
            "position_feedback_age_ms": stats(get("steering_position_age_ms")[start:stop]),
            "speed_feedback_age_ms": stats(get("steering_velocity_age_ms")[start:stop]),
            "first_current_feedback_abs_atleast_0p01Nm_receipt_ms": None if not len(nonzero)
            else float(receipt_times[nonzero[0]] * 1000),
            "first150ms_commanded_torque_nm": stats(torque[initial150]),
            "first150ms_measured_current_derived_torque_nm": stats(measured[first150]),
        }
        results.append(item)
        segment_data.append((times, position, target, velocity, torque, receipt_times, measured))

    b, b_details = stages["B"]
    commands = b["motor_command"]
    active = commands[commands[:, 1] > .5]
    delta = np.diff(active[:, 2])
    same_episode = np.diff(active[:, 0]) < .05
    # These are supported dry-run target increments, NOT live-balancing bandwidth.
    b_result = {
        "enabled_commands": int(len(active)),
        "within_contiguous_recorded_intervals_abs_target_change_deg": stats(np.abs(np.rad2deg(delta[same_episode]))),
        "warning": "Dry run only; not a measured live policy-to-steering response or bandwidth requirement.",
    }
    summary = {
        "method": {
            "control_time": "host monotonic tick start",
            "current_time": "host ROS receipt time relative to first enabled tick ROS timestamp",
            "angles": "signed A3-derived motor coordinates; encoder resolution 360/16384 degree",
            "settling": "All remaining recorded ticks in band, with at least .25 s confirmation tail",
            "first_crossing": "Crosses a fraction of initial-to-target displacement; NOT settling",
            "freshness": "Distinct reconstructed A3 host receipt timestamps, NOT firmware sampling timestamps",
            "limits": "No control-rate sweep or powered policy run exists in these bags; cannot identify a sufficient rate",
        },
        "provenance": provenance, "C_segments": results, "B_target_changes": b_result,
    }
    (output / "tracking_results.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for column, segment_index in enumerate((1, 3)):
        t, q, target, v, tau, receipt, measured = segment_data[segment_index]
        axes[0, column].step(t * 1000, np.rad2deg(q), where="post", label="A3 position available to loop")
        axes[0, column].plot(t * 1000, np.rad2deg(target), "--", label="constant requested target")
        axes[0, column].axvline(20, color="grey", ls=":", label="one policy period (20 ms)")
        axes[1, column].step(t * 1000, tau, where="post", label="computed C0 torque target")
        axes[1, column].plot(receipt * 1000, measured, ".-", ms=3, lw=.8, label="current-derived torque (receipt time)")
        for row in range(2):
            axes[row, column].set_xlim(0, 250)
            axes[row, column].grid(alpha=.25)
            axes[row, column].legend(fontsize=8)
        axes[0, column].set_title(f"C interval {segment_index + 1}: target {results[segment_index]['target_deg']:+.3f} deg")
        axes[0, column].set_ylabel("Steering angle (deg)")
        axes[1, column].set_ylabel("Torque estimate / target (N m)")
        axes[1, column].set_xlabel("Milliseconds from first enabled motor-control tick")
    fig.suptitle("Steering step / re-enable response: a 20 ms target period is not a settling deadline")
    fig.savefig(output / "step_and_reenable_response.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    t, q, target, v, tau, receipt, measured = segment_data[3]
    axes[0].step(t, np.rad2deg(q), where="post", label="encoder position")
    axes[0].plot(t, np.rad2deg(target), "--", label="constant target")
    axes[0].set_ylabel("Angle (deg)")
    axes[1].step(t, tau, where="post", label="C0 torque target")
    axes[1].plot(receipt, measured, ".-", ms=2, lw=.7, label="current-derived torque")
    axes[1].set_ylabel("Torque (N m)")
    a, z = ranges[3]
    axes[2].plot(t, get("steering_position_age_ms")[a:z], label="A3 receipt age")
    axes[2].plot(t, get("steering_velocity_age_ms")[a:z], label="A2 receipt age", alpha=.75)
    axes[2].set_ylabel("Feedback age (ms)")
    axes[2].set_xlabel("Seconds from first enabled motor-control tick")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(loc="upper right")
    fig.suptitle("C interval 4: 364 control ticks with one constant target, yet no stable tracking")
    fig.savefig(output / "constant_target_long_hold.png", dpi=160)
    plt.close(fig)
    for item in results:
        print(json.dumps({key: item[key] for key in (
            "segment", "control_ticks", "sampled_enabled_duration_ms", "target_deg",
            "first_changed_position_control_tick_ms", "first_90percent_crossing_ms",
            "settling_within_target_plusminus_0p1deg_ms", "final_halfsecond_median_deg",
            "first_current_feedback_abs_atleast_0p01Nm_receipt_ms",
        )}))
    print("Results and plots written to", output)


if __name__ == "__main__":
    main()
