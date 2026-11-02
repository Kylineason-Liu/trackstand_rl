"""Presentation figures for matched MIT/C1 velocity tests; raw timestamps retained."""

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

from analyze import HERE, STEMS, integral, read_capture
from check_response import paired

COLORS = {"C1": "#2166ac", "MIT": "#dd7c20"}


def rows(name):
    with (HERE/name).open() as stream:
        return [{k: (v if k in ("mode", "name", "kind") else float(v) if v else None) for k, v in r.items()}
                for r in csv.DictReader(stream)]


def extrema(ax):
    values = [line.get_ydata() for line in ax.lines if len(line.get_ydata()) > 2]
    if values:
        y = np.concatenate(values)
        ax.text(.015, .97, f"Shown min/max: {min(y):.3g} / {max(y):.3g}", transform=ax.transAxes,
                va="top", fontsize=7, bbox=dict(facecolor="white", edgecolor="none", alpha=.75))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    args = p.parse_args()
    captures = {k: read_capture(k, args.cache) for k in STEMS}
    steady, sine = rows("steady_summary.csv"), rows("sine_summary.csv")
    law = json.loads((HERE/"current_response_fit.json").read_text())["mit_current_response"]
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .23,
                         "legend.fontsize": 8, "savefig.dpi": 175})
    gallery = []
    with PdfPages(HERE/"all_figures.pdf") as pdf:

        def save(fig, name, caption):
            fig.tight_layout(rect=(0, .045, 1, .94))
            for suffix in ("png", "svg", "pdf"):
                fig.savefig(HERE/f"{name}.{suffix}")
            pdf.savefig(fig)
            gallery.append((name, caption))
            plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for label in STEMS:
            selected = sorted([r for r in steady if r["mode"] == label], key=lambda r: r["base_rad_s"])
            for ax, narrow in ((axes[0, 0], False), (axes[0, 1], True)):
                rr = [r for r in selected if not narrow or abs(r["base_rad_s"]) <= .8]
                x = [r["base_rad_s"] for r in rr]
                y = [r["velocity_mean_rad_s_mean"] for r in rr]
                ax.plot(x, y, "o-", color=COLORS[label], label=label)
                ax.fill_between(x, [r["velocity_mean_rad_s_min"] for r in rr],
                                [r["velocity_mean_rad_s_max"] for r in rr], color=COLORS[label], alpha=.15)
            x = [r["base_rad_s"] for r in selected]
            axes[1, 0].plot(x, [r["velocity_error_mean_rad_s_mean"] for r in selected], "o-",
                            color=COLORS[label], label=label)
            axes[1, 1].plot(x, [r["torque_mean_nm_mean"] for r in selected], "o-",
                            color=COLORS[label], label=label)
        for ax, limit in ((axes[0, 0], 10.5), (axes[0, 1], .85)):
            ax.plot([-limit, limit], [-limit, limit], "k--", lw=1, label="Perfect tracking")
            ax.set(xlabel="Intended constant target [rad/s]", ylabel="Mean fresh A2 speed [rad/s]")
        axes[0, 0].set_title("Steady response: all levels")
        axes[0, 1].set_title("Near zero: MIT barely moves through ±0.4 rad/s")
        axes[1, 0].set(xlabel="Intended target [rad/s]", ylabel="Mean A2 − effective target [rad/s]",
                       title="MIT has approximately 0.33-rad/s moving-speed offset")
        axes[1, 1].set(xlabel="Intended target [rad/s]", ylabel="Mean current-derived torque [Nm]",
                       title="C1 can accumulate the torque required at small targets")
        for ax in axes.flat:
            ax.legend(loc="lower right")
        fig.suptitle("Rear velocity control: C1 PI versus MIT P (requested 0.30; encoded 0.300366)")
        fig.text(.5, .01, "Three repetitions per signed target; lines = means, shading = range. "
                 "Steady window 2–5 s. No smoothing or time shift. Airborne/free, matching settings.",
                 ha="center", fontsize=9)
        save(fig, "steady_response",
             "MIT's small steady targets produce almost no travel; large targets retain a bias.")

        for metric, filename in (("fundamental_gain", "sine_amplitude"), ("velocity_error_rms_rad_s", "sine_error")):
            fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
            for row_index, amp in enumerate((.25, 1.)):
                for col, base in enumerate((0., -3., 3.)):
                    ax = axes[row_index, col]
                    for label in STEMS:
                        rr = sorted([r for r in sine if r["mode"] == label and r["base_rad_s"] == base
                                     and r["amplitude_rad_s"] == amp], key=lambda r: r["frequency_hz"])
                        x = [r["frequency_hz"] for r in rr]
                        ax.plot(x, [r[metric+"_mean"] for r in rr], "o-", color=COLORS[label], label=label)
                        ax.fill_between(x, [r[metric+"_min"] for r in rr], [r[metric+"_max"] for r in rr],
                                        color=COLORS[label], alpha=.15)
                        if filename == "sine_error":
                            ax.plot(x, [abs(r["velocity_error_mean_rad_s_mean"]) for r in rr], "--",
                                    color=COLORS[label], lw=1, label=label+" |mean bias|")
                    ax.set_xscale("log")
                    ax.set_xticks([.5, 1, 2, 5, 10, 20], ["0.5", "1", "2", "5", "10", "20"])
                    ax.set(xlabel="Sine frequency [Hz]", title=f"Baseline {base:g}, amplitude ±{amp:g} rad/s")
                    ax.set_ylabel("Response fundamental / actual target" if filename == "sine_amplitude"
                                  else "Velocity error RMS / |bias| [rad/s]")
                    if filename == "sine_amplitude":
                        ax.axhline(1, color="black", ls="--", lw=.8)
                        ax.set_ylim(-.03, 1.2)
                    ax.legend(loc="lower right")
            fig.suptitle("Sine tracking: zero-crossing behavior differs sharply from an already rotating wheel")
            fig.text(.5, .01, "Means and ranges of three repeats. Actual held encoded targets; "
                     "MIT reference includes zero-FF codec correction. "
                     "RMS is unshifted and includes mean-speed bias.", ha="center", fontsize=9)
            save(fig, filename, "Sine response amplitude" if filename == "sine_amplitude"
                 else "Sine error and mean bias")

        selections = [("steady", .1, 0., 0.), ("steady", .4, 0., 0.),
                      ("sine", 0., .25, .5), ("sine", 0., 1., .5)]
        fig, axes = plt.subplots(3, 4, figsize=(19, 10))
        plotted = []
        for col, (kind, base, amp, frequency) in enumerate(selections):
            for label, (meta, phases, _) in captures.items():
                phase = next(p for p in phases if p["segment"]["repeat"] == 3
                             and p["segment"]["kind"] == kind and p["segment"]["base_rad_s"] == base
                             and p["segment"]["amplitude_rad_s"] == amp
                             and p["segment"]["frequency_hz"] == frequency)
                start, stop = (0., 5.) if kind == "steady" else (1., 9.)
                ct, command = phase["streams"]["command"]
                vt, velocity = phase["streams"]["velocity"]
                it, current = phase["streams"]["current"]
                pt, q = phase["streams"]["position"]
                color = COLORS[label]
                for row_index, (t, y, field) in enumerate(((vt, velocity, "a2_velocity_rad_s"),
                                                          (it, current*meta["torque_constant_nm_per_a"],
                                                           "a1_torque_nm"),
                                                          (pt, q, "a3_relative_position_rad"))):
                    keep = (t >= start) & (t < stop)
                    t, y = t[keep], y[keep]
                    if row_index == 2:
                        y = y-y[0]
                    axes[row_index, col].plot(t, y, color=color, lw=.85, label=label)
                    plotted.extend(dict(panel=f"r{row_index}_c{col}", mode=label, field=field, time_s=float(a),
                                        value=float(b)) for a, b in zip(t, y))
                keep = (ct >= start) & (ct < stop)
                axes[0, col].step(ct[keep], command[keep], where="post", color="black", lw=1,
                                  alpha=.8, ls="--" if label == "MIT" else ":", label=label+" effective target")
                qorigin = pt[(pt >= start) & (pt < stop)][0]
                reference_time = np.linspace(max(qorigin, ct[0]), stop, 1000)
                reference_q = integral(ct, command, reference_time)-integral(ct, command, reference_time[0])
                axes[2, col].plot(reference_time, reference_q, color="black", lw=.8,
                                  alpha=.6, ls="--" if label == "MIT" else ":", label=label+" target integral")
            axes[0, col].set_title(f"Steady +{base:g} rad/s" if kind == "steady"
                                   else f"Zero-base sine ±{amp:g}, {frequency:g} Hz")
            for row_index in range(3):
                axes[row_index, col].set_xlim(start, stop)
                axes[row_index, col].set_xlabel("Time from segment start [s]")
                axes[row_index, col].set_ylabel(("Velocity [rad/s]", "Motor torque estimate [Nm]",
                                               "Relative displacement [rad]")[row_index])
                axes[row_index, col].legend(loc="lower right", fontsize=7)
                extrema(axes[row_index, col])
        fig.suptitle("Same intended conditions, repeat 3: raw target, response, "
                     "current-derived torque and encoder travel")
        fig.text(.5, .01, "Each position trace subtracts its own first shown A3; reference is a velocity integral, "
                 "not a position command. Separate runs, no smoothing/time shifting/physical angle matching.",
                 ha="center", fontsize=9)
        save(fig, "representative_traces",
             "Near-zero commands: current exists while displacement barely changes under MIT.")
        from analyze import write_csv
        write_csv(HERE/"plotted_response_samples.csv", plotted)

        meta, phases, _ = captures["MIT"]
        kd = meta["mit_configuration"]["velocity_kd_encoded_nm_per_rad_s"]
        kt = meta["torque_constant_nm_per_a"]
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
        for phase in phases:
            if phase["segment"]["kind"] != "steady":
                continue
            t, e, torque = paired(phase, 0., kt)
            keep = t >= 2.
            axes[0].scatter(e[keep].mean(), torque[keep].mean(), s=20, color=COLORS["MIT"], alpha=.7)
        x = np.linspace(-.65, .65, 100)
        axes[0].plot(x, kd*x, "k--", label=f"Requested encoded P={kd:.6f}")
        axes[0].set(xlabel="Steady mean effective target − A2 [rad/s]", ylabel="Mean A1 × B0 torque [Nm]",
                    title="Steady current matches the requested P law")
        phase = next(p for p in phases if p["segment"]["kind"] == "steady"
                     and p["segment"]["base_rad_s"] == 10. and p["segment"]["repeat"] == 3)
        it, error, torque = paired(phase, law["apparent_alignment_ms"]/1000, kt)
        keep = it <= .8
        axes[1].plot(it[keep], torque[keep], color=COLORS["MIT"], lw=.9, label="Fresh A1 × B0")
        axes[1].plot(it[keep], kd*error[keep], color="#358b52", lw=.9, label="P law using observed A2*")
        axes[1].set(xlabel="Time from +10-rad/s segment start [s]", ylabel="Torque [Nm]",
                    title="Transient current response")
        for phase in phases:
            if phase["segment"]["kind"] not in ("sine", "step") or phase["segment"]["repeat"] != 3:
                continue
            _, error, torque = paired(phase, law["apparent_alignment_ms"]/1000, kt)
            axes[2].scatter(kd*error[::30], torque[::30], s=3, alpha=.12, color=COLORS["MIT"])
        axes[2].plot([-1, 1], [-1, 1], "k--", label="Equal torque")
        axes[2].set(xlabel="P law with observed A2 [Nm]", ylabel="Fresh A1 × B0 [Nm]",
                    title=f"Repeat 3: fixed-P R²={law['fixed_encoded_kd_heldout']['r_squared']:.3f}")
        for ax in axes:
            ax.legend(loc="lower right")
        fig.suptitle("MIT actuation check: measured current supports the requested P control")
        fig.text(.5, .01, "*Model alignment =1 ms selected with repeat 2, not measured firmware delay; "
                 "raw current times unchanged. Current-derived torque is not independent shaft torque.",
                 ha="center", fontsize=9)
        save(fig, "mit_torque_law", "Current response supports the P-control/friction explanation.")

    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>MIT versus C1 velocity tests</title>
<style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:0 20px;line-height:1.5}
img{max-width:100%}</style>
</head><body><h1>Rear MIT P versus C1 PI</h1>
<p>MIT Kd≈0.30 is materially worse for small desired movements;
steady high-speed tracking retains about 0.33 rad/s bias.</p>
<p><a href="README.md">Report</a> · <a href="all_figures.pdf">All figures PDF</a> ·
<a href="steady_summary.csv">Steady results</a> · <a href="sine_summary.csv">Sine results</a> ·
<a href="raw_audit.json">Raw audit</a> · <a href="current_response_fit.json">Current-law check</a></p>'''
    for name, caption in gallery:
        html += f'<h2>{escape(caption)}</h2><a href="{name}.png">'
        html += f'<img src="{name}.png" alt="{escape(caption)}"></a>'
        html += f'<p><a href="{name}.svg">SVG</a> · <a href="{name}.pdf">PDF</a></p>'
    (HERE/"index.html").write_text(html+"</body></html>")


if __name__ == "__main__":
    main()
