"""Six motor-specific overlays, with a five-second window centred on each attempt.

Reuse the verified 3x3 presentation CSVs; never extrapolate a short attempt.
Only this presentation directory is written. No hardware/configuration changes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


WINDOW_S = 5.0
STAGES = ("raw_scaled", "motor_bound", "measured")
COLORS = ("#1565C0", "#D55E00", "#009E73")
STEM = "motor_overlay_middle_5s"


def centred_window(first_s: float, last_s: float) -> tuple[float, float]:
    if not np.isfinite([first_s, last_s]).all() or last_s <= first_s:
        raise ValueError("attempt endpoints must be finite and increasing")
    start = (first_s + last_s - WINDOW_S) / 2
    return start, start + WINDOW_S


def crop(series: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    """Preserve samples/timestamps, with no interpolation or edge extrapolation."""
    if series.ndim != 2 or series.shape[1] != 3 or not len(series):
        raise ValueError("expected nonempty [time, steering_rad, rear_nm]")
    if not np.isfinite(series[:, 0]).all() or np.any(np.diff(series[:, 0]) < 0):
        raise ValueError("record timestamps must be finite and nondecreasing")
    selected = series[(series[:, 0] >= start_s) & (series[:, 0] <= end_s)].copy()
    if not len(selected):
        raise ValueError("no recorded samples in requested window")
    selected[:, 0] -= start_s
    return selected


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_attempts(source: Path) -> tuple[list[dict], dict]:
    summary = json.loads((source.parent / "summary.json").read_text())
    provenance = json.loads((source / "provenance.json").read_text())
    if sha256(source.parent / "summary.json") != provenance["report_summary_sha256"]:
        raise ValueError("source summary hash mismatch")
    if sha256(Path(provenance["bag"])) != provenance["bag_sha256"]:
        raise ValueError("recording hash mismatch")
    attempts, hashes = [], {}
    for item in summary["attempts"]:
        duration = item["last_policy_s"] - item["first_s"]
        start, end = centred_window(0, duration)
        series = []
        for stage in STAGES:
            path = source / f"attempt_{item['attempt']}_{stage}.csv"
            with path.open(newline="") as stream:
                reader = csv.reader(stream)
                if next(reader) != ["seconds_since_first_inference", "steering_rad", "rear_nm"]:
                    raise ValueError(f"unexpected CSV columns: {path}")
                rows = np.array([[float(v) for v in row] for row in reader])
            series.append(crop(rows, start, end))
            hashes[path.name] = sha256(path)
        attempts.append({
            "attempt": item["attempt"], "active_duration_s": duration,
            "shorter_than_window": duration < WINDOW_S,
            "window_since_first_inference_s": [start, end],
            "window_bag_s": [item["first_s"] + start, item["first_s"] + end],
            "series": series,
        })
    if [a["attempt"] for a in attempts] != [1, 2, 3]:
        raise ValueError("expected the three verified deployment attempts")
    provenance.update({
        "source_presentation": str(source.resolve()), "source_csv_sha256": hashes,
        "source_provenance_sha256": sha256(source / "provenance.json"),
        "time_base": "recorded timestamps within a centred 5 s window; x=0 is its start",
        "window_duration_s": WINDOW_S, "short_attempt_handling":
        "Keep the available recorded attempt (including existing <=20 ms context/30 ms tail). "
        "Leave the rest of its centred 5 s window blank; never invent or repeat data.",
    })
    for obsolete in ("attempts", "layout", "primary_panels", "left_axis_rad_limits", "right_axis_nm_limits"):
        provenance.pop(obsolete, None)
    return attempts, provenance


def draw(attempts: list[dict], source: Path, output: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    # Reuse already-tested gap/axis helpers, without running their CLI.
    spec = importlib.util.spec_from_file_location("policy_plot_helpers", source / "plot_policy_to_motor.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 13,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 3, figsize=(19, 10), sharex=True, sharey="row", squeeze=False)
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.19, top=0.82, wspace=0.16, hspace=0.17)
    limits = [helper.symmetric_limit([s[:, motor + 1] for a in attempts for s in a["series"]])
              for motor in range(2)]
    for col, attempt in enumerate(attempts):
        for row in range(2):
            ax = axes[row, col]
            if attempt["shorter_than_window"]:
                left = min(s[0, 0] for s in attempt["series"])
                right = max(s[-1, 0] for s in attempt["series"])
                ax.axvspan(0, left, color="#ECEFF1", zorder=0)
                ax.axvspan(right, WINDOW_S, color="#ECEFF1", zorder=0)
                for x in (left / (2 * WINDOW_S), (right + WINDOW_S) / (2 * WINDOW_S)):
                    ax.text(x, 0.88, "No attempt data", transform=ax.transAxes,
                            ha="center", fontsize=10, color="#607D8B",
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
            for stage, (series, color) in enumerate(zip(attempt["series"], COLORS)):
                broken = helper.break_gaps(series)
                kwargs = {"color": color, "lw": (1.05, 1.3, 1.4)[stage],
                          "alpha": (0.7, 0.9, 0.95)[stage], "zorder": 2 + stage}
                if stage < 2:
                    ax.step(broken[:, 0], broken[:, row + 1], where="post", **kwargs)
                else:
                    ax.plot(broken[:, 0], broken[:, row + 1], **kwargs)
            ax.set_xlim(0, WINDOW_S)
            ax.set_ylim(-limits[row], limits[row])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.set_xticks(range(6))
            ax.axhline(0, color="#808080", lw=0.6, alpha=0.5, zorder=1)
            ax.grid(alpha=0.15)
            if col == 0:
                ax.set_ylabel(("Steering position [rad]", "Rear torque [N·m]")[row])
            if row == 0:
                begin, end = attempt["window_bag_s"]
                label = f"Attempt {attempt['attempt']} | bag {begin:.3f}–{end:.3f} s"
                if attempt["shorter_than_window"]:
                    label += f"\nOnly {attempt['active_duration_s']:.2f} s of policy activity"
                else:
                    label += "\nMiddle 5 seconds of the attempt"
                ax.set_title(label, pad=10)
            if row == 1:
                ax.set_xlabel("Time within centred window [s]")
    fig.suptitle("Policy output, final motor request and measured response", fontsize=22, y=0.97)
    fig.text(0.5, 0.927, "Model 91850 | Corrected rear sign | 2026-10-12 18:06 | Five-second windows",
             ha="center", fontsize=13, color="#37474F")
    fig.legend(handles=[Line2D([0], [0], color=c, lw=2, label=label) for c, label in zip(COLORS, (
        "Scaled raw policy output (×1.57 steering / ×6.28 rear)",
        "Final encoded motor request", "Measured motor response"))],
        loc="upper center", bbox_to_anchor=(0.5, 0.902), ncol=3, frameon=False, fontsize=11.5)
    notes = (
        "Same linear physical-unit axis for all three curves; full raw range retained. Common y scale across attempts per motor.",
        "Final request: encoded MIT steering position / C0 torque equivalent on successful-send ticks. Measured rear torque: current × B0, not independent shaft torque.",
        "Attempt 1 is shorter than 5 s: available data only; shaded surroundings have no attempt data. Recorded clocks, no lag correction, smoothing or interpolation.",
        "Command limits were ENABLED in this bag. Kickstand contact <1 s after lift was reported; later data are not unsupported balancing.",
    )
    for y, note in zip((0.12, 0.09, 0.06, 0.03), notes):
        fig.text(0.065, y, note, fontsize=10)
    assert axes.shape == (2, 3) and len(fig.axes) == 6
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"{STEM}.{suffix}", dpi=300)
    plt.close(fig)
    return {"layout": [2, 3], "primary_panels": 6, "y_scale": "linear",
            "symmetric_y_limits": {"steering_rad": limits[0], "rear_nm": limits[1]}}


def main() -> None:
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=folder.parent / "presentation_3x3")
    parser.add_argument("--output", type=Path, default=folder)
    args = parser.parse_args()
    attempts, provenance = load_attempts(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    provenance.update(draw(attempts, args.source, args.output))
    provenance["generator_sha256"] = sha256(Path(__file__))
    provenance["helper_sha256"] = sha256(args.source / "plot_policy_to_motor.py")
    provenance["attempts"] = []
    for attempt in attempts:
        details = {k: v for k, v in attempt.items() if k != "series"}
        details["samples_per_stage"] = dict(zip(STAGES, [len(s) for s in attempt["series"]]))
        provenance["attempts"].append(details)
        for stage, series in zip(STAGES, attempt["series"]):
            with (args.output / f"attempt_{attempt['attempt']}_{stage}.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["seconds_since_window_start", "steering_rad", "rear_nm"])
                writer.writerows(series)
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"figure": str(args.output / f"{STEM}.png"),
                      "attempts": provenance["attempts"]}, indent=2))


if __name__ == "__main__":
    main()
