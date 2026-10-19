"""Plot recorded actor outputs, successful CAN-bound requests and measured response.

Only this presentation directory is written. No policy inference, ROS executor,
CAN socket, simulator or deployment configuration is started/modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


STEERING_COLOR = "#1565C0"
REAR_COLOR = "#D55E00"
GAP_S = 0.05
TAIL_S = 0.03
CONTEXT_S = 0.02
STEM = "policy_to_motor_3x3"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def break_gaps(series: np.ndarray, threshold_s: float = GAP_S) -> np.ndarray:
    """Insert NaN breaks, never interpolate across a recording gap."""
    if series.ndim != 2 or series.shape[1] != 3 or not len(series):
        raise ValueError("series must be nonempty [time, steering_rad, rear_nm]")
    if not np.all(np.isfinite(series[:, 0])) or np.any(np.diff(series[:, 0]) < 0):
        raise ValueError("record times must be finite and nondecreasing")
    rows = [series[0]]
    for previous, current in zip(series[:-1], series[1:]):
        if current[0] - previous[0] > threshold_s:
            rows.append(np.array([current[0], np.nan, np.nan]))
        rows.append(current)
    return np.asarray(rows)


def symmetric_limit(values: list[np.ndarray]) -> float:
    finite = np.concatenate(values)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise ValueError("no finite values for axis")
    magnitude = max(1e-3, float(np.max(np.abs(finite))) * 1.1)
    step = 10.0 ** math.floor(math.log10(magnitude))
    return math.ceil(magnitude / step) * step


def load_series(cache: Path, report: Path) -> tuple[list[dict], dict]:
    summary = json.loads((report / "summary.json").read_text())
    details = cache / "details.json"
    labels = json.loads(details.read_text())["schemas"] if details.exists() else json.loads(
        (cache / "labels.json").read_text()
    )
    with np.load(cache / "data.npz", allow_pickle=False) as arrays:
        raw, control, state = [arrays[key].copy() for key in ("raw_action", "motor_control", "motor_state")]
    if raw.shape != (summary["policy"]["sample_count"], 3):
        raise ValueError("raw action count/shape does not match report")

    def field(key: str, data: np.ndarray, name: str) -> np.ndarray:
        if data.ndim != 2 or data.shape[1] != 1 + len(labels[key]):
            raise ValueError(f"{key} schema/width mismatch")
        return data[:, 1 + labels[key].index(name)]

    # Filter submission flags, not merely prepared/dry-run/guard-rejected payloads.
    sent = (field("motor_control", control, "steering_send_success") > 0.5) & (
        field("motor_control", control, "drive_send_success") > 0.5
    )
    if sent.sum() != summary["actuation"]["rear_successful_send_ticks"]:
        raise ValueError("successful submission count does not match report")
    scaled = np.column_stack((raw[:, 0], raw[:, 1] * 1.57, raw[:, 2] * 6.28))
    requests = np.column_stack((
        control[sent, 0],
        field("motor_control", control, "mit_encoded_position_rad")[sent],
        field("motor_control", control, "drive_encoded_torque_nm")[sent],
    ))
    response = np.column_stack((
        state[:, 0],
        field("motor_state", state, "steering_position_rad"),
        field("motor_state", state, "drive_torque_nm"),
    ))
    # Check the exported policy values, but retain actual raw_action record times.
    with (report / "policy_samples.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = np.array([[float(row["scaled_steer_rad"]), float(row["scaled_rear_nm"])] for row in rows])
    if not np.array_equal(scaled[:, 1:], expected):
        raise ValueError("scaled raw values do not match verified policy CSV")

    attempts = []
    for item in summary["attempts"]:
        begin, end = item["first_s"], item["last_policy_s"] + TAIL_S
        series = []
        for data in (scaled, requests, response):
            selected = data[(data[:, 0] >= begin - CONTEXT_S) & (data[:, 0] <= end)].copy()
            if not len(selected):
                raise ValueError(f"empty attempt {item['attempt']}")
            selected[:, 0] -= begin
            break_gaps(selected)  # Validate clocks before rendering.
            series.append(selected)
        if len(series[0]) != item["policy_count"] or len(series[1]) != item["rear_send_ticks"]:
            raise ValueError(f"attempt {item['attempt']} counts do not match report")
        attempts.append({
            "attempt": item["attempt"], "first_inference_bag_s": begin,
            "end_relative_s": end - begin, "series": series,
        })
    provenance = {
        "bag": summary["bag"], "bag_sha256": summary["sha256"],
        "policy": summary["policy"]["model"],
        "report_summary_sha256": digest(report / "summary.json"),
        "decoded_data_sha256": digest(cache / "data.npz"),
        "source_fields": {
            "raw": "raw_action: steering*1.57 rad, rear*6.28 N m",
            "request": "motor_control: mit_encoded_position_rad, drive_encoded_torque_nm; both send_success=true",
            "response": "motor_state: steering_position_rad (A3), drive_torque_nm (measured current * signed B0)",
        },
        "time_base": "bag record time minus first inference time; each topic keeps its own timestamps",
        "feedback_tail_s": TAIL_S, "pre_inference_context_s": CONTEXT_S,
        "gap_break_threshold_s": GAP_S,
        "warning": "Rear feedback is current-derived, not independently measured shaft torque. "
                   "No time shifting, filtering, interpolation or synthetic feedback. "
                   "This recording predates removal of command limits. Kickstand contact <1 s after lift "
                   "was reported by the user; exact contact time is unknown.",
    }
    return attempts, provenance


def draw(attempts: list[dict], output: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "svg.fonttype": "none",
        "pdf.fonttype": 42, "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(3, 3, figsize=(19, 12), sharex="col", squeeze=False)
    fig.subplots_adjust(left=0.095, right=0.91, bottom=0.15, top=0.86, wspace=0.28, hspace=0.18)
    raw_limits = [symmetric_limit([a["series"][0][:, col] for a in attempts]) for col in (1, 2)]
    lower_limits = [symmetric_limit([a["series"][row][:, col] for a in attempts for row in (1, 2)])
                    for col in (1, 2)]
    limits = [raw_limits, lower_limits, lower_limits]
    row_labels = (
        "1. Scaled raw policy output\nSteering [rad]",
        "2. Final motor-bound request\nSteering [rad]",
        "3. Measured motor response\nSteering [rad]",
    )
    right_labels = ("Scaled rear output [N·m]", "Rear motor request [N·m]", "Current-derived rear torque [N·m]")
    for col, attempt in enumerate(attempts):
        for row in range(3):
            left = axes[row, col]
            right = left.twinx()
            series = break_gaps(attempt["series"][row])
            if row < 2:
                left.step(series[:, 0], series[:, 1], where="post", color=STEERING_COLOR, lw=1.25)
                right.step(series[:, 0], series[:, 2], where="post", color=REAR_COLOR, lw=1.05, alpha=0.85)
            else:
                left.plot(series[:, 0], series[:, 1], color=STEERING_COLOR, lw=1.15)
                right.plot(series[:, 0], series[:, 2], color=REAR_COLOR, lw=1.05, alpha=0.85)
            left.set_ylim(-limits[row][0], limits[row][0])
            right.set_ylim(-limits[row][1], limits[row][1])
            left.set_xlim(-CONTEXT_S, attempt["end_relative_s"])
            left.tick_params(axis="y", colors=STEERING_COLOR)
            right.tick_params(axis="y", colors=REAR_COLOR)
            left.yaxis.set_major_locator(MaxNLocator(nbins=4))
            right.yaxis.set_major_locator(MaxNLocator(nbins=4))
            left.xaxis.set_major_locator(MaxNLocator(nbins=5))
            left.spines["left"].set_color(STEERING_COLOR)
            right.spines["right"].set_color(REAR_COLOR)
            left.axhline(0, color="#808080", lw=0.6, alpha=0.5)
            left.grid(axis="x", alpha=0.18)
            if col == 0:
                left.set_ylabel(row_labels[row], color=STEERING_COLOR, labelpad=10)
            if col == 2:
                right.set_ylabel(right_labels[row], color=REAR_COLOR, labelpad=10)
            if row == 0:
                left.set_title(f"Attempt {attempt['attempt']} — start {attempt['first_inference_bag_s']:.3f} bag s")
            if row == 2:
                left.set_xlabel("Time since first inference [s]")
    fig.suptitle("Policy output → final motor request → measured response", fontsize=22, y=0.98)
    fig.text(0.5, 0.94, "Model 91850  |  Corrected rear encoder sign  |  2026-10-12 18:06 recording",
             ha="center", fontsize=13, color="#37474F")
    fig.legend(handles=[
        Line2D([0], [0], color=STEERING_COLOR, lw=2, label="Steering: radians — LEFT axis"),
        Line2D([0], [0], color=REAR_COLOR, lw=2, label="Rear motor: N·m — RIGHT axis"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.915), ncol=2, frameon=False, fontsize=13)
    fig.text(0.095, 0.075,
             "Requests: decoded MIT/C0 payload equivalents on successful-send ticks, after all host limits. "
             "Response: published A3 position and measured Q-axis current × B0 torque constant.", fontsize=10.5)
    fig.text(0.095, 0.049,
             "Recorded timestamps; no lag correction or smoothing. Common scales within each row; "
             "request/response rows use identical scales. Gaps >50 ms are broken; 30 ms feedback tail.", fontsize=10.5)
    fig.text(0.095, 0.023,
             "Command limits were ENABLED in this bag. Rear torque is NOT an independent shaft-torque measurement. "
             "Kickstand support occurred <1 s after lift (user report); exact contact time unknown.", fontsize=10.5)
    assert axes.shape == (3, 3) and len(fig.axes) == 18  # Nine panels, each with a secondary unit axis.
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"{STEM}.{suffix}", dpi=300)
    plt.close(fig)
    return {"primary_panels": 9, "layout": [3, 3], "left_axis_rad_limits": [v[0] for v in limits],
            "right_axis_nm_limits": [v[1] for v in limits]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="Decoded data.npz plus details.json or labels.json")
    parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    attempts, provenance = load_series(args.cache, args.report)
    bag = Path(provenance["bag"])
    if digest(bag) != provenance["bag_sha256"]:
        raise ValueError("original recording hash mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    provenance.update(draw(attempts, args.output))
    provenance["attempts"] = []
    for attempt in attempts:
        counts = []
        for stage, data in zip(("raw_scaled", "motor_bound", "measured"), attempt["series"]):
            with (args.output / f"attempt_{attempt['attempt']}_{stage}.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["seconds_since_first_inference", "steering_rad", "rear_nm"])
                writer.writerows(data)
            counts.append({"stage": stage, "rows": len(data),
                           "finite_steering": int(np.isfinite(data[:, 1]).sum()),
                           "finite_rear": int(np.isfinite(data[:, 2]).sum()),
                           "gaps_over_50ms": int((np.diff(data[:, 0]) > GAP_S).sum())})
        provenance["attempts"].append({"attempt": attempt["attempt"],
                                      "first_inference_bag_s": attempt["first_inference_bag_s"], "counts": counts})
    provenance["generator_sha256"] = digest(Path(__file__))
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps({"figure": str(args.output / f"{STEM}.png"),
                      "layout": provenance["layout"], "attempts": provenance["attempts"]}, indent=2))


if __name__ == "__main__":
    main()
