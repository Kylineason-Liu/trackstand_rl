"""Concatenate all active deployment attempts without rescaling their time.

Run directly with Python; reads only the archived report decode and statistics.
No ROS, CAN, policy inference, smoothing or fitted delay is involved.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np


HERE = Path(__file__).resolve().parent
ORANGE, GREEN, RED = "#d7760e", "#15805a", "#cf2f32"


def write_csv(name, rows):
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def break_recording_gaps(times, values, threshold_s=.05):
    """Break the drawn curve across missing data without changing saved samples."""
    indices = np.flatnonzero(np.diff(times) > threshold_s) + 1
    return np.insert(times, indices, times[indices]), np.insert(values, indices, np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", action="store_true", help="Plot the October 12 18:06 model_91850 deployment")
    args = parser.parse_args()
    inputs = (("previous_20261012_180607_data.npz", "previous_20261012_180607_details.json.gz",
               "../deployment_20261012_180607/summary.json") if args.previous else
              ("decoded_data.npz", "decoded_details.json.gz", "statistics.json"))
    input_hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in inputs}
    arrays = np.load(HERE / inputs[0])
    with gzip.open(HERE / inputs[1], "rt") as stream:
        details = json.load(stream)
    if args.previous:
        summary = json.loads((HERE / inputs[2]).read_text())
        assert summary["sha256"] == details["sha256"] == (
            "24be09c76b25d338c58ca6ae3d95bd468107a4afb4fa5b38139a1d8e568c3e78"
        )
        assert summary["counts"] == details["counts"] and not details["issues"]
        events = details["text"]["policy_event"]
        episodes = []
        for start in (event for event in events if event["mode"] == "balancing"):
            stop = next(event for event in events if event["sequence"] > start["sequence"]
                        and event["mode"] != "balancing")
            episodes.append(dict(episode=len(episodes) + 1, start_s=start["t"], stop_s=stop["t"],
                                 reason=",".join(stop["latched_fault_reasons"]) or "RC_down"))
        assert len(episodes) == len(summary["attempts"]) == 3
        prefix = "20261012_180607_concatenated"
        title = "October 12, 18:06 deployment · model_91850 · all 3 attempts concatenated"
        target_label = "MIT position target after host slew"
    else:
        episodes = json.loads((HERE / inputs[2]).read_text())["episodes"]
        prefix = "concatenated"
        title = "October 29 deployment · model_2500 · all 21 attempts concatenated"
        target_label = "MIT position target after 50 Hz slew"
    stem = f"{prefix}_motor_target_response"
    control, state = arrays["motor_control"], arrays["motor_state"]
    fields = details["schemas"]["motor_control"]
    col = lambda name: control[:, 1 + fields.index(name)]
    specs = (
        ("steering", "steering_send_success", "mit_encoded_position_rad", 1, "rad",
         target_label, "A3 steering position"),
        ("rear", "drive_send_success", "drive_encoded_torque_nm", 7, "Nm",
         "Encoded C0 torque target", "A1 current × Kt torque estimate"),
    )
    attempt_count = len(episodes)
    assert [ep["episode"] for ep in episodes] == list(range(1, attempt_count + 1))
    series, boundaries, rows = [], [], []
    elapsed = 0.0
    for ep in episodes:
        start, stop = ep["start_s"], ep["stop_s"]
        # A few first packets can precede the corresponding policy-event
        # reception by microseconds. Use the published policy-event boundaries
        # consistently for both motors; do not borrow from another attempt.
        span = stop - start
        assert span > 0
        boundary = dict(episode=ep["episode"], concatenated_start_s=elapsed,
                        concatenated_stop_s=elapsed + span, bag_start_s=start,
                        bag_stop_s=stop, plotted_duration_s=span, stop_reason=ep["reason"])
        episode_series = []
        for motor, success, target_field, feedback_column, unit, target_label, feedback_label in specs:
            sent = ((control[:, 0] >= start) & (control[:, 0] < stop) & (col(success) > .5))
            measured = (state[:, 0] >= start) & (state[:, 0] < stop)
            tx, ty = control[sent, 0], col(target_field)[sent]
            fx, fy = state[measured, 0], state[measured, feedback_column]
            assert len(tx) and np.all(np.isfinite(ty))
            # Preserve missing current feedback as NaN (not an invented zero).
            assert np.all(np.diff(tx) >= 0) and np.all(np.diff(fx) >= 0)
            episode_series.append((tx - start + elapsed, ty, fx - start + elapsed, fy))
            boundary[motor + "_target_samples"] = len(tx)
            boundary[motor + "_feedback_samples"] = len(fx)
            for kind, times, values in (("target", tx, ty), ("response", fx, fy)):
                for t, value in zip(times, values):
                    rows.append(dict(episode=ep["episode"], motor=motor, series=kind, unit=unit,
                                     bag_time_s=t, concatenated_time_s=t - start + elapsed, value=value))
        boundaries.append(boundary)
        series.append(episode_series)
        elapsed += span

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .2, "savefig.dpi": 220})
    fig, axes = plt.subplots(2, 1, figsize=(24, 8.5), sharex=True, constrained_layout=True)
    all_values, limits = [], []
    for j, (motor, _, _, _, unit, target_label, feedback_label) in enumerate(specs):
        ax = axes[j]
        values = []
        # Separate Line2D objects intentionally prevent invented reset jumps.
        for i, episode_series in enumerate(series):
            tx, ty, fx, fy = episode_series[j]
            plotted_tx, plotted_ty = break_recording_gaps(tx, ty)
            plotted_fx, plotted_fy = break_recording_gaps(fx, fy)
            ax.step(plotted_tx, plotted_ty, where="post", color=ORANGE, lw=1.05, alpha=.95,
                    label=target_label if i == 0 else None, zorder=3)
            ax.plot(plotted_fx, plotted_fy, color=GREEN, lw=1.05, alpha=.95,
                    label=feedback_label if i == 0 else None, zorder=4)
            values.extend((ty, fy))
        values = np.concatenate(values)
        finite = values[np.isfinite(values)]
        all_values.append(finite)
        # Symmetric full-data limits with modest headroom and useful ticks.
        tick = .5 if motor == "steering" else 2.0
        maximum = np.max(abs(finite))
        granularity = .25 if motor == "steering" else .5
        limit = np.ceil(maximum * 1.04 / granularity) * granularity
        ax.set_ylim(-limit, limit)
        ax.yaxis.set_major_locator(MultipleLocator(tick))
        limits.append([-float(limit), float(limit)])
        ax.set_ylabel("Steering position (rad)" if motor == "steering" else "Rear drive torque (Nm)")
        ax.axhline(0, color="#707070", lw=.55, alpha=.6, zorder=1)
        ax.legend(loc="lower right", fontsize=10, framealpha=.94)
        ax.text(.004, .985, f"Recorded target/response range: {finite.min():.3f} to {finite.max():.3f} {unit}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=.9, edgecolor="none"), zorder=8)

    # Label close boundaries on separate vertical tiers. This preserves the
    # genuine 39 ms attempt 14 rather than artificially stretching its length.
    lane_last = [-np.inf] * 3
    lane_assignment = []
    for boundary in boundaries:
        t = boundary["concatenated_start_s"]
        available = [lane for lane, previous in enumerate(lane_last) if t - previous > .36]
        lane = available[0] if available else int(np.argmin(lane_last))
        lane_last[lane] = t
        lane_assignment.append(lane)
        for ax in axes:
            ax.axvline(t, color=RED, lw=.85, alpha=.9, zorder=2)
            ax.text(t + .018, 1.015 + lane * .06, str(boundary["episode"]), color=RED,
                    transform=ax.get_xaxis_transform(), fontsize=9, va="bottom", ha="left", clip_on=False)
    axes[-1].set_xlim(0, elapsed)
    axes[-1].xaxis.set_major_locator(MultipleLocator(1.0))
    axes[-1].set_xlabel("Cumulative active recording time (s) — pauses removed; timing within each attempt unchanged")
    subtitle = "Red lines and numbers mark attempt starts; curves are not joined across attempt boundaries"
    if args.previous:
        subtitle += "\nUser reported early kickstand contact: long enabled intervals are not balancing survival"
    fig.suptitle(title + "\n" + subtitle,
                 fontsize=15)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(HERE / f"{stem}.{extension}", bbox_inches="tight")
    plt.close(fig)

    write_csv(f"{prefix}_attempt_boundaries.csv", boundaries)
    write_csv(f"{prefix}_motor_samples.csv", rows)
    for j, values in enumerate(all_values):
        assert values.min() >= limits[j][0] and values.max() <= limits[j][1]
    assert len(boundaries) == attempt_count
    assert abs(elapsed - sum(ep["stop_s"] - ep["start_s"] for ep in episodes)) < 1e-12
    assert all(hashlib.sha256((HERE / name).read_bytes()).hexdigest() == value
               for name, value in input_hashes.items())
    verification = dict(attempts=attempt_count, panels=2, red_start_lines_per_panel=attempt_count, saved_sample_rows=len(rows),
                        cumulative_plotted_duration_s=elapsed, input_hashes=input_hashes,
                        y_limits=limits, all_finite_values_visible=True,
                        unchanged_native_sample_values=True, rescaled_time=False,
                        smoothing=False, fitted_delay=False, curves_joined_between_attempts=False,
                        drawn_curves_break_at_gap_s=.05,
                        target="Successful recorded MIT position / C0 current-to-torque commands",
                        rear_feedback="Current-derived A1 × Kt estimate, not shaft torque",
                        source_time="MCAP reception time, translated by each attempt's removed idle interval")
    verification.update(bag=details["bag"], bag_sha256=details["sha256"],
                        support_caveat="Early kickstand contact reported" if args.previous else "No stand-position measurement")
    (HERE / f"{prefix}_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
