"""Plot reproducible review evidence; reads recorded data only, never opens CAN."""

import argparse
import csv
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from review_analysis import HERE, linear_integral, load, original, write_csv


def csv_rows(name):
    with (HERE / name).open() as stream:
        return [{k: float(v) if k not in ("name", "method") else v for k, v in row.items()}
                for row in csv.DictReader(stream)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    meta, phases = load(args.cache)
    review = json.loads((HERE / "review_statistics.json").read_text())
    kt = meta["torque_constant_nm_per_a"]
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .22,
                         "legend.fontsize": 8, "savefig.dpi": 170})
    figures = []
    with PdfPages(HERE / "review_figures.pdf") as pdf:

        def save(fig, name, caption):
            for ax in fig.axes:
                ax.margins(x=.015)
            fig.tight_layout(rect=(0, .035, 1, .94))
            for suffix in ("png", "svg", "pdf"):
                fig.savefig(HERE / f"{name}.{suffix}")
            pdf.savefig(fig)
            figures.append((name, caption))
            plt.close(fig)

        gains = csv_rows("gain_method_comparison.csv")
        labels = ["Original time fit\nA3 integral", "Frequency fit\nA2 error harmonic",
                  "Time fit\nA2 error integral", "Time diagnostic\nA3 rescaled*",
                  "Time fit\nzero baseline only", "Frequency fit\nzero baseline only"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
        for ax, key, label in zip(axes, ("kp", "ki"), ("P [Nm / (rad/s)]", "I [Nm / rad]")):
            y = np.array([g[f"{key}_nm_per_rad_s" if key == "kp" else "ki_nm_per_rad"] for g in gains])
            lo = np.array([g[f"{key}_conditional_low"] for g in gains])
            hi = np.array([g[f"{key}_conditional_high"] for g in gains])
            ax.errorbar(np.arange(6), y, yerr=[y-lo, hi-y], fmt="o", color="#2466a4", capsize=5)
            ax.set_xticks(np.arange(6), labels, rotation=20, ha="right")
            ax.set_ylabel(label)
            ax.set_title("P is fairly consistent" if key == "kp" else "I is sensitive to the integral reconstruction")
            if key == "kp":
                ax.axhline(.30, color="#c94b40", ls="--", lw=1, label="Provisional P = 0.30")
                ax.legend(loc="lower right")
        fig.suptitle("Rear C1 effective PI gains — unchanged firmware, airborne wheel")
        fig.text(.5, .01, "Bars: conditional 95% whole-trial bootstrap intervals; systematic uncertainty remains. "
                 "*Diagnostic A3 scale fitted from repeat 1 only; no raw data changed.", ha="center", fontsize=9)
        save(fig, "gain_comparison", "P estimates agree approximately; I is strongly affected by the error integral.")

        consistency = csv_rows("velocity_position_consistency.csv")
        high = [r for r in consistency if abs(r["a2_time_average_rad_s"]) >= 1.]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
        for repeat, color in ((1, "#2466a4"), (2, "#e38b28"), (3, "#299561")):
            rows = [r for r in high if r["repeat"] == repeat]
            x = np.array([r["a2_time_average_rad_s"] for r in rows])
            y = np.array([r["a3_mean_velocity_rad_s"] for r in rows])
            axes[0].scatter(x, y, color=color, label=f"Repeat {repeat}")
            axes[1].scatter(x, (y/x-1)*100, color=color, label=f"Repeat {repeat}")
        axes[0].plot([-11, 11], [-11, 11], color="black", ls="--", label="Equal measurements")
        axes[0].set(xlabel="A2 time-average speed [rad/s]", ylabel="A3 angle travel / same interval [rad/s]")
        scale = review["kinematic_consistency"]["scale_a3_speed_per_a2_speed"]
        axes[1].axhline((scale-1)*100, color="#c94b40", ls="--", label=f"Fitted difference {(scale-1)*100:.3f}%")
        axes[1].axhline(0, color="black", lw=.8)
        axes[1].set(xlabel="A2 time-average speed [rad/s]", ylabel="A3-derived speed / A2 speed − 1 [%]")
        for ax in axes:
            ax.legend(loc="lower right")
        fig.suptitle("Independent A2/A3 consistency check: identical actual endpoints, 24 steady trials")
        fig.text(.5, .01, "No time shift or physical calibration applied. "
                 "Both directions and all three repeats show the discrepancy.",
                 ha="center", fontsize=9)
        save(fig, "a2_a3_consistency", "A3 travel is about 1.58% higher than integrated A2 speed at steady motion.")

        phase = next(p for p in phases if p["segment"]["repeat"] == 3 and p["segment"]["kind"] == "steady"
                     and p["segment"]["base_rad_s"] == 10.)
        ct, target = phase["streams"]["command"]
        vt, velocity = phase["streams"]["velocity"]
        pt, position = phase["streams"]["position"]
        it, current = phase["streams"]["current"]
        a, b = 2.02, 4.95
        qtime = np.linspace(a, b, 1000)
        command_integral = original.held_integral(ct, target, qtime)-original.held_integral(ct, target, a)
        a2_integral = linear_integral(vt, velocity, qtime)-linear_integral(vt, velocity, a)
        a3_travel = np.interp(qtime, pt, position)-np.interp(a, pt, position)
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        mask = (vt >= a) & (vt <= b)
        axes[0].plot(vt[mask]-a, velocity[mask], color="#2466a4", lw=.8, label="Fresh A2 velocity")
        axes[0].axhline(target[-1], color="#e38b28", ls="--", label="Encoded C1 target")
        axes[0].set_ylabel("Velocity [rad/s]")
        axes[1].plot(qtime-a, command_integral-a3_travel, label="∫C1 dt − A3 travel (original proxy)")
        axes[1].plot(qtime-a, command_integral-a2_integral, label="∫(C1 − A2) dt")
        axes[1].set_ylabel("Integral error [rad]")
        mask = (it >= a) & (it <= b)
        axes[2].plot(it[mask]-a, current[mask]*kt, color="#299561", lw=.8, label="Fresh A1 × B0 torque estimate")
        axes[2].set(ylabel="Motor torque [Nm]", xlabel="Time from selected steady-window start [s]")
        for ax in axes:
            ax.legend(loc="lower right")
        fig.suptitle("Why the original I estimate is biased — repeat 3, +10 rad/s steady case")
        fig.text(.5, .01, "Integral origins set to zero at the analysis-window start only; "
                 "hardware encoder origins are unchanged.",
                 ha="center", fontsize=9)
        save(fig, "integral_proxy_bias",
             "A3-based accumulated error drifts despite near-zero average A2 tracking error.")

        fit = review["alternative_fits"]["integrated_a2_error"]
        beta = np.array([fit["kp_a_per_rad_s"], fit["ki_a_per_rad"]])
        examples = [next(p for p in phases if p["segment"]["name"].endswith(suffix))
                    for suffix in ("r3_0270", "r3_0266", "r3_0331", "r3_0312", "r3_0306")]
        fig, axes = plt.subplots(2, len(examples), figsize=(19, 7))
        for col, p in enumerate(examples):
            s = p["segment"]
            ct, target = p["streams"]["command"]
            vt, velocity = p["streams"]["velocity"]
            it, current = p["streams"]["current"]
            start, stop = (.35, 1.5) if s["kind"] == "step" else (1., 2. if s["frequency_hz"] < 20 else 1.3)
            query = it-fit["apparent_delay_ms"]/1000
            keep = ((query >= max(ct[0], vt[0], .03)) & (query <= min(vt[-1], s["duration_s"]-.03)))
            times, query, measured = it[keep], query[keep], current[keep]
            idx = np.clip(np.searchsorted(ct, query, side="right")-1, 0, len(ct)-1)
            x = np.column_stack((target[idx]-np.interp(query, vt, velocity),
                                 original.held_integral(ct, target, query)-linear_integral(vt, velocity, query)))
            prediction = x@beta + (measured-x@beta).mean()
            top, bottom = axes[:, col]
            mask = (ct >= start) & (ct <= stop)
            top.step(ct[mask], target[mask], where="post", color="#e38b28", lw=1.5, label="Actual encoded C1")
            mask = (vt >= start) & (vt <= stop)
            top.plot(vt[mask], velocity[mask], color="#2466a4", lw=.85, label="Fresh A2 feedback")
            mask = (it >= start) & (it <= stop)
            bottom.plot(it[mask], current[mask]*kt, color="#299561", lw=.85, label="A1 × B0 torque")
            mask = (times >= start) & (times <= stop)
            bottom.plot(times[mask], prediction[mask]*kt, color="#ac4695", lw=.85, alpha=.9,
                        label="PI reconstruction*")
            title = f"Step {s['base_rad_s']:g} → {s['base_rad_s']+s['amplitude_rad_s']:g}" if s["kind"] == "step" else (
                f"Sine ±{s['amplitude_rad_s']:g}, {s['frequency_hz']:g} Hz")
            top.set_title(title+f"\n{s['name'].split('_')[-1]}")
            top.set_ylabel("Velocity [rad/s]")
            bottom.set(ylabel="Torque [Nm]", xlabel="Time from segment start [s]")
            for ax in (top, bottom):
                ax.set_xlim(start, stop)
                ax.legend(loc="lower right")
        fig.suptitle("Measured C1 responses — repeat 3; near-zero reversals differ from rotating-wheel steps")
        fig.text(.5, .01, "*Uses observed error/integral, all-data fitted gains and a per-trial offset; "
                 "this is not a forward simulation. Raw feedback timestamps are unshifted.", ha="center", fontsize=9)
        save(fig, "representative_responses",
             "Fresh velocity/current traces and their PI reconstruction; near-zero nonlinear behavior is visible.")

        tracking = csv_rows("sine_tracking_by_trial.csv")
        grouped = []
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        for col, amp in enumerate((.25, 1.)):
            for base, color in ((-3., "#2466a4"), (0., "#e38b28"), (3., "#299561")):
                groups = []
                for f in (.5, 1., 2., 5., 10., 20.):
                    rows = [r for r in tracking if r["amplitude_rad_s"] == amp and r["base_rad_s"] == base
                            and r["frequency_hz"] == f]
                    row = dict(base_rad_s=base, amplitude_rad_s=amp, frequency_hz=f, repeats=len(rows))
                    for key in ("tracking_fundamental_gain", "phase_equivalent_lag_ms",
                                "unshifted_velocity_error_rms_rad_s"):
                        row[key+"_mean"] = float(np.mean([r[key] for r in rows]))
                        row[key+"_min"] = min(r[key] for r in rows)
                        row[key+"_max"] = max(r[key] for r in rows)
                    grouped.append(row)
                    groups.append(row)
                for row_index, key in enumerate(("tracking_fundamental_gain", "unshifted_velocity_error_rms_rad_s")):
                    ax = axes[row_index, col]
                    ax.plot([r["frequency_hz"] for r in groups], [r[key+"_mean"] for r in groups],
                            "o-", color=color, label=f"Baseline {base:g} rad/s")
                    ax.fill_between([r["frequency_hz"] for r in groups], [r[key+"_min"] for r in groups],
                                    [r[key+"_max"] for r in groups], color=color, alpha=.1)
                    ax.set_xscale("log")
                    ax.set_xticks([.5, 1, 2, 5, 10, 20], ["0.5", "1", "2", "5", "10", "20"])
                    ax.legend(loc="lower right")
                    ax.set_xlabel("Sine frequency [Hz]")
            axes[0, col].set(title=f"Velocity sine amplitude ±{amp:g} rad/s", ylabel="Fundamental response / target")
            axes[0, col].axhline(1, color="black", ls="--", lw=.8)
            axes[1, col].set_ylabel("Unshifted velocity error RMS [rad/s]")
        steady = csv_rows("steady_conditions.csv")
        for repeat, color in ((1, "#2466a4"), (2, "#e38b28"), (3, "#299561")):
            rows = sorted([r for r in steady if r["repeat"] == repeat], key=lambda r: r["target_rad_s"])
            x = [r["target_rad_s"] for r in rows]
            axes[0, 2].plot(x, [r["a2_velocity_mean_rad_s"]-r["target_rad_s"] for r in rows], "o-",
                            color=color, label=f"Repeat {repeat}")
            axes[1, 2].plot(x, [r["a2_velocity_std_rad_s"] for r in rows], "o-", color=color)
        axes[0, 2].set(title="Steady ±0.05…10 rad/s", xlabel="Target speed [rad/s]", ylabel="Mean A2 − target [rad/s]")
        axes[1, 2].set(xlabel="Target speed [rad/s]", ylabel="A2 velocity standard deviation [rad/s]")
        for ax in axes[:, 2]:
            ax.set_xscale("symlog", linthresh=.1)
            ticks = [-10, -3, -1, -.1, 0, .1, 1, 3, 10]
            ax.set_xticks(ticks, [f"{value:g}" for value in ticks])
        axes[0, 2].legend(loc="lower right")
        fig.suptitle("Tracking is operating-condition dependent — all three repeats")
        fig.text(.5, .01, "Lines = mean of three trials; shading = trial range. Gains use actual held commands; "
                 "RMS includes speed ripple and has no delay compensation.", ha="center", fontsize=9)
        save(fig, "tracking_by_condition",
             "Zero-crossing small sine tracking is much worse than tracking around ±3 rad/s.")
        write_csv(HERE / "tracking_summary.csv", grouped)

    html = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Rear C1 gain identification</title><style>
body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:0 20px;line-height:1.5}
img{max-width:100%;height:auto}section{margin-top:36px}a{color:#1465a5}</style></head><body>
<h1>Rear C1 effective PI identification</h1>
<p><strong>Provisional velocity P ≈0.30 Nm/(rad/s); I ≈3.3 Nm/rad is a model starting point,
not calibrated firmware gain units.</strong></p>
<p>A3 travel / integrated A2 speed ≈1.0158. Read the review before using the original automated I estimate.</p>
<p><a href="README.md">Analysis report</a> · <a href="review_statistics.json">Reviewed fits</a> ·
<a href="capture_audit.json">Raw-packet audit</a> · <a href="review_figures.pdf">Review figures PDF</a> ·
<a href="tracking_summary.csv">Tracking table</a> · <a href="gain_method_comparison.csv">Gain table</a></p>
"""
    for name, caption in figures:
        html += f'<section><h2>{escape(caption)}</h2><a href="{name}.png">'
        html += f'<img src="{name}.png" alt="{escape(caption)}"></a>'
        html += f'<p><a href="{name}.svg">SVG</a> · <a href="{name}.pdf">PDF</a></p></section>'
    html += '<h2>Original automated figures</h2><p>Preserved for reproducibility; '
    html += 'their original A3-integral fit is biased here.</p>'
    for name in ("steady_conditions", "frequency_response", "timing_sensitivity",
                 "example_1", "example_2", "example_3", "example_4"):
        html += f'<p><a href="{name}.png">{escape(name)}</a></p>'
    html += '<p><a href="automated_README.md">Original automated report</a> · '
    html += '<a href="statistics.json">Original statistics</a> · '
    html += '<a href="all_figures.pdf">Original PDF</a></p></body></html>'
    (HERE / "index.html").write_text(html)


if __name__ == "__main__":
    main()
