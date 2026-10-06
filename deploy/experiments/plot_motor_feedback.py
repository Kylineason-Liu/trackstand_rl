#!/usr/bin/env python3
"""Plot recorded steering C2 targets and A3/A2/A1 feedback on a shared clock.

Produces an offline HTML page with four vertically stacked axes and selectable
time windows, plus a full-record PNG/PDF and machine-readable provenance.
"""

import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def number(row, field):
    try:
        return float(row[field])
    except (KeyError, ValueError):
        return math.nan


def series(rows, time_field, value_field):
    samples = sorted((number(row, time_field), number(row, value_field)) for row in rows
                     if math.isfinite(number(row, time_field)))
    return np.asarray(samples, dtype=float).reshape(-1, 2)


def load_recording(source):
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    feedback = [row for row in rows if row["motor"] == "steering" and row["record_type"] == "feedback"]
    commands = [row for row in rows if row["motor"] == "steering" and row["record_type"] == "c2_target"]
    if not feedback or not commands:
        raise ValueError("Need both steering feedback and recorded C2 targets")
    zeros = {int(number(row, "zero_count")) for row in feedback if math.isfinite(number(row, "zero_count"))}
    if len(zeros) != 1:
        raise ValueError("Expected one stable collector startup zero; cannot silently merge multiple origins")
    zero = zeros.pop()
    # Infer the count-to-angle mapping from this recording, rather than assume
    # which CLI sign or gear ratio was used. Validate every finite A3 sample.
    slopes = [number(row, "position_unwrapped_rad") / (number(row, "encoder_count") - zero)
              for row in feedback if math.isfinite(number(row, "position_unwrapped_rad"))
              and math.isfinite(number(row, "encoder_count")) and number(row, "encoder_count") != zero]
    if not slopes:
        raise ValueError("No nonzero A3 displacement available to establish the logged angle conversion")
    slope = statistics.median(slopes)
    for row in feedback:
        count, position = number(row, "encoder_count"), number(row, "position_unwrapped_rad")
        if math.isfinite(count) and math.isfinite(position) and not math.isclose(
                position, (count - zero) * slope, abs_tol=1e-9, rel_tol=1e-9):
            raise ValueError("Inconsistent A3 count-to-angle mapping")
    command = series(commands, "elapsed_s", "c2_target_count")
    if not np.all(np.isfinite(command)):
        raise ValueError("A captured command has a missing target count")
    command[:, 1] = (command[:, 1] - zero) * slope
    traces = [
        command,
        series(feedback, "position_elapsed_s", "position_unwrapped_rad"),
        series(feedback, "velocity_elapsed_s", "velocity_rad_s"),
        series(feedback, "current_elapsed_s", "torque_nm"),
    ]
    if any(len(trace) == 0 or not np.any(np.isfinite(trace[:, 1])) for trace in traces):
        raise ValueError("One of the requested traces has no valid samples")
    lo, hi = float(np.min(command[:, 1])), float(np.max(command[:, 1]))
    center = (lo + hi) / 2
    crossings = []
    for (t0, y0), (t1, y1) in zip(command, command[1:]):
        if y0 < center <= y1:
            crossings.append(float(t0 + (center - y0) * (t1 - t0) / (y1 - y0)))
    periods = np.diff(crossings)
    duration = max(number(row, "elapsed_s") for row in feedback)
    summary = {
        "source": str(source.resolve()), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "total_rows": len(rows), "steering_feedback_rows": len(feedback), "steering_c2_rows": len(commands),
        "collector_zero_count": zero, "radians_per_native_count_in_logged_coordinates": slope,
        "recording_duration_s": duration,
        "c2_observed_start_s": float(command[0, 0]), "c2_observed_end_s": float(command[-1, 0]),
        "command_min_rad": lo, "command_max_rad": hi, "command_peak_to_peak_rad": hi - lo,
        "period_median_s": float(np.median(periods)) if len(periods) else None,
        "period_crossing_count": len(crossings),
        "timestamp_semantics": "Host-receipt elapsed timestamps on one recorder monotonic clock; no phase shifts",
        "traces": [],
    }
    for name, trace in zip(("C2 position command", "A3 position feedback", "A2 velocity feedback", "A1 torque feedback"),
                           traces):
        valid = trace[np.isfinite(trace[:, 1])]
        summary["traces"].append({
            "name": name, "valid_samples": len(valid), "missing_values_at_known_times": len(trace) - len(valid),
            "min": float(np.min(valid[:, 1])), "max": float(np.max(valid[:, 1])),
            "mean_observed_rate_hz": (len(valid) - 1) / float(valid[-1, 0] - valid[0, 0]),
            "median_interval_ms": float(np.median(np.diff(valid[:, 0])) * 1000),
            "max_interval_ms": float(np.max(np.diff(valid[:, 0])) * 1000),
            "gaps_over_50_ms": [[float(valid[i, 0]), float(valid[i+1, 0])]
                                for i in np.flatnonzero(np.diff(valid[:, 0]) > 0.05)],
        })
    return traces, summary


def draw_figure(traces, summary, window, title):
    colors = ("#2563eb", "#087f73", "#8256bd", "#b85a16")
    titles = ("1  Steering position command · observed C2 targets",
              "2  Steering position feedback · A3",
              "3  Steering velocity feedback · A2",
              "4  Steering torque feedback · A1 current × B0 constant")
    units = ("Position (rad)", "Position (rad)", "Velocity (rad/s)", "Torque (N m)")
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 10.4))
    fig.subplots_adjust(left=0.095, right=0.975, top=0.935, bottom=0.075, hspace=0.36)
    fig.suptitle(f"Steering sine test — {title}", fontsize=17, x=0.095, ha="left", color="#172a3a")
    command_start, command_end = summary["c2_observed_start_s"], summary["c2_observed_end_s"]
    position_limit = max(abs(summary["command_min_rad"]), abs(summary["command_max_rad"]),
                         abs(summary["traces"][1]["min"]), abs(summary["traces"][1]["max"])) * 1.1
    for index, (ax, trace, color, panel_title, unit) in enumerate(zip(axes, traces, colors, titles, units)):
        selected = trace[(trace[:, 0] >= window[0] - 0.025) & (trace[:, 0] <= window[1] + 0.025)]
        if index == 0:
            ax.step(selected[:, 0], selected[:, 1], where="post", color=color, linewidth=0.85)
        else:
            # A straight line across a long missing interval would suggest a
            # response never measured. NaN is a render-only separator here.
            gaps = np.flatnonzero(np.diff(selected[:, 0]) > 0.05)
            plotted = np.insert(selected, gaps + 1, np.nan, axis=0)
            ax.plot(plotted[:, 0], plotted[:, 1], color=color, linewidth=0.85)
            for gap in gaps:
                ax.axvspan(selected[gap, 0], selected[gap+1, 0], color="#fca5a5", alpha=0.3)
        ax.set_title(panel_title, fontsize=11, loc="left", pad=8, fontweight="bold", color=color)
        ax.set_ylabel(unit, fontsize=10)
        ax.grid(True, color="#dce4e9", linewidth=0.6)
        ax.axhline(0, color="#738493", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)
        if index < 2:
            ax.set_ylim(-position_limit, position_limit)  # Identical command/position scales for honest comparison.
        else:
            magnitude = max(abs(summary["traces"][index]["min"]), abs(summary["traces"][index]["max"]))
            ax.set_ylim(-magnitude * 1.1, magnitude * 1.1)
        for boundary in (command_start, command_end):
            if window[0] < boundary < window[1]:
                ax.axvline(boundary, color="#94a3b8", linewidth=0.8, linestyle="--")
        if window[0] < command_start:
            ax.axvspan(window[0], min(window[1], command_start), color="#edf1f4", alpha=0.7)
        if window[1] > command_end:
            ax.axvspan(max(window[0], command_end), window[1], color="#edf1f4", alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(*window)
    axes[-1].set_xlabel("Elapsed time since feedback recorder started (s) — shared time axis", fontsize=11)
    fig.text(0.095, 0.015, "Collector startup A3 zero; positive = right. Grey = outside C2 interval. "
             "Red = feedback gap >50 ms. Torque = current-based estimate.",
             fontsize=8, color="#526474")
    assert all(ax.get_xlim() == axes[0].get_xlim() for ax in axes)
    return fig


def write_report(source, output):
    traces, summary = load_recording(source)
    output.mkdir(parents=True, exist_ok=True)
    start, end = summary["c2_observed_start_s"], summary["c2_observed_end_s"]
    middle = (start + end) / 2
    width = min(1.0, end - start)
    windows = [
        ("Full recording", (0.0, summary["recording_duration_s"])),
        ("C2 interval", (start, end)),
        ("First 1 s", (start, start + width)),
        ("Middle 1 s", (middle - width / 2, middle + width / 2)),
        ("Final 1 s", (end - width, end)),
    ]
    plots = []
    plt.rcParams.update({"svg.fonttype": "none", "path.simplify": False, "font.family": "DejaVu Sans"})
    for index, (label, window) in enumerate(windows):
        figure = draw_figure(traces, summary, window, label)
        svg_path = output / f"steering_view_{index}.svg"
        figure.savefig(svg_path, facecolor="white")
        if index == 0:
            figure.savefig(output / "steering_sine_report.png", dpi=160, facecolor="white")
            figure.savefig(output / "steering_sine_report.pdf", facecolor="white")
        elif index == 3:
            figure.savefig(output / "steering_sine_zoom.png", dpi=160, facecolor="white")
        plt.close(figure)
        svg = svg_path.read_text()
        svg = svg[svg.index("<svg"):]
        svg = svg.replace('id="', f'id="v{index}-').replace('url(#', f'url(#v{index}-')
        svg = svg.replace('href="#', f'href="#v{index}-')
        plots.append(f'<section class="view{" active" if index == 0 else ""}" id="view-{index}">{svg}</section>')
    summary["views"] = [{"label": label, "xlim_s": window} for label, window in windows]
    source_link = html.escape(os.path.relpath(source.resolve(), output.resolve()), quote=True)
    buttons = "".join(f'<button data-view="{index}" aria-pressed="{"true" if index == 0 else "false"}">{label}</button>'
                      for index, (label, _) in enumerate(windows))
    ranges = json.dumps([f"{window[0]:.3f}–{window[1]:.3f} s" for _, window in windows])
    period = summary["period_median_s"]
    period_text = f"{period:.5f} s" if period is not None else "not estimable"
    range_note = (f"Recorded C2 targets span {summary['command_min_rad']:.3f} to "
                  f"{summary['command_max_rad']:.3f} rad ({summary['command_peak_to_peak_rad']:.3f} rad peak-to-peak). "
                  f"Measured command period: {period_text}.")
    if math.isclose(summary["command_peak_to_peak_rad"], 4.0, abs_tol=0.01):
        range_note += " This is approximately ±2 rad, not ±1 rad."
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Steering sine test — aligned feedback</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f2f5f8;color:#172a3a;font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1220px;margin:24px auto;padding:0 18px}}header,.plots,footer{{background:white;border:1px solid #dce4e9;border-radius:12px;padding:20px 24px;margin-bottom:16px}}
h1{{font-size:26px;margin:0 0 6px}}p{{margin:7px 0}}.muted{{color:#526474}}.note{{padding:10px 14px;background:#fff5dd;border-left:4px solid #e1a935;border-radius:4px}}
.metrics{{display:flex;flex-wrap:wrap;gap:24px;margin:14px 0}}.metrics span{{display:block;color:#526474;font-size:12px}}.metrics strong{{font-size:18px}}
.toolbar{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px}}button{{border:1px solid #c5d1dd;background:white;color:#172a3a;border-radius:6px;padding:8px 12px;cursor:pointer}}
button[aria-pressed=true]{{background:#172a3a;color:white}}.view{{display:none}}.view.active{{display:block}}.view svg{{display:block;width:100%;height:auto}}
a{{color:#2563eb}}footer{{font-size:13px}}code{{background:#f1f4f7;padding:1px 4px;border-radius:3px}}
@media(max-width:600px){{main{{margin:10px auto;padding:0 8px}}header,.plots,footer{{padding:12px}}h1{{font-size:21px}}.plots{{overflow-x:auto}}.view{{min-width:680px}}}}
@media print{{@page{{size:A4;margin:8mm}}body{{background:white}}main{{margin:0;padding:0}}header,.plots,footer{{border:0;padding:0;margin-bottom:4px}}.toolbar,footer,.metrics,.note{{display:none}}h1{{font-size:18px}}.view svg{{max-height:240mm}}}}
</style></head><body><main>
<header><h1>Steering sine test · four aligned traces</h1>
<p class="muted">Recorded data from <a href="{source_link}">{html.escape(source.name)}</a>. All panels use the same elapsed-time clock.</p>
<p class="note">{range_note}</p>
<div class="metrics"><div><span>Captured command range</span><strong>{summary['command_min_rad']:.3f} to {summary['command_max_rad']:.3f} rad</strong></div>
<div><span>C2 frames / observed rate</span><strong>{summary['steering_c2_rows']:,} / {summary['traces'][0]['mean_observed_rate_hz']:.2f} Hz</strong></div>
<div><span>Steering feedback / observed rate</span><strong>{summary['steering_feedback_rows']:,} / {summary['traces'][1]['mean_observed_rate_hz']:.2f} Hz</strong></div></div>
</header><div class="plots"><div class="toolbar">{buttons}<span id="range" class="muted">{json.loads(ranges)[0]}</span></div>
{''.join(plots)}</div>
<footer><p><strong>Alignment:</strong> C2 uses <code>elapsed_s</code>; position uses <code>position_elapsed_s</code>; velocity uses <code>velocity_elapsed_s</code>; torque uses <code>current_elapsed_s</code>. These are recorder host-receipt times, not hardware acquisition or execution times. No phase shifts, smoothing, or extrapolated targets are applied. C2 targets are drawn as steps between observed frames.</p>
<p><strong>Position conversion:</strong> command radians = (C2 target count − {summary['collector_zero_count']}) × {summary['radians_per_native_count_in_logged_coordinates']:.12g}. This mapping is verified against every finite logged unwrapped-position sample. Both position panels have identical vertical scales. Positive means right in the logged coordinates.</p>
<p><strong>Torque:</strong> logged signed A1 Q-current × B0 torque constant; not an independent shaft-torque measurement. Grey shading denotes times outside the observed C2-command interval; it does not imply the motor was disabled.</p>
<p><strong>Feedback gaps:</strong> lines are broken and red shading added across receipt gaps longer than 50 ms. No missing measurement is replaced by a guessed value. Recorded C2 targets remain held steps; gaps do not reveal physical behavior inside the interval.</p>
<p><a href="steering_sine_report.png">Full-record PNG</a> · <a href="steering_sine_zoom.png">Middle-second PNG</a> · <a href="steering_sine_report.pdf">One-page PDF</a> · <a href="summary.json">Numerical summary and provenance</a></p></footer>
</main><script>
const ranges={ranges};
document.querySelectorAll('button[data-view]').forEach(button=>button.addEventListener('click',()=>{{
 document.querySelectorAll('.view').forEach(view=>view.classList.remove('active'));
 document.getElementById('view-'+button.dataset.view).classList.add('active');
 document.querySelectorAll('button[data-view]').forEach(other=>other.setAttribute('aria-pressed',String(other===button)));
 document.getElementById('range').textContent=ranges[Number(button.dataset.view)];
}}));
</script></body></html>'''
    (output / "steering_sine_report.html").write_text(page)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Report: {output / 'steering_sine_report.html'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_report(args.source, args.output_dir)
