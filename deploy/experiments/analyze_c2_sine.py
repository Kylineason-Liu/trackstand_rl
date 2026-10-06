#!/usr/bin/env python3
"""Offline C2 sine sweep: reuse the four-panel reports and quantify tracking.

No ROS initialization, CAN sockets, commands, or firmware changes. NumPy only
for numerical analysis; Matplotlib is supplied by plot_motor_feedback.py.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re

import numpy as np

from plot_motor_feedback import load_recording, number, plt, write_report


ROOT = Path(__file__).resolve().parent
LABEL = re.compile(r"motor_feedback_(\d{8})(\d{4})_(\d+(?:\.\d+)?)hz(\d+)rad\.csv")
SPEED_REFERENCE_RAD_S = 23.9  # Deployment reference, NOT a read-back limit of these experiments.


def labels(source: Path) -> dict:
    match = LABEL.fullmatch(source.name)
    if not match:
        raise ValueError(f"Unsupported filename: {source.name}")
    date, clock, frequency, amplitude = match.groups()
    amplitude_rad = float(f"0.{amplitude[1:]}" if amplitude.startswith("0") else amplitude)
    return dict(date=date, clock=clock, frequency_hz=float(frequency), amplitude_rad=amplitude_rad,
                amplitude_token=amplitude,
                report_name=f"steering_sine_{date}_{clock}_{frequency}hz{amplitude}rad")


def harmonic(time_s: np.ndarray, value: np.ndarray, frequency_hz: float, origin_s: float) -> dict:
    mask = np.isfinite(time_s) & np.isfinite(value)
    time_s, value = time_s[mask], value[mask]
    if len(time_s) < 6 or np.ptp(time_s) < 1 / frequency_hz:
        raise ValueError("Need at least six valid samples spanning one cycle")
    phase = 2 * np.pi * frequency_hz * (time_s - origin_s)
    design = np.column_stack((np.sin(phase), np.cos(phase), np.ones(len(time_s))))
    coefficients, _, rank, _ = np.linalg.lstsq(design, value, rcond=None)
    if rank != 3:
        raise ValueError("Rank-deficient sine fit")
    residual = value - design @ coefficients
    variance = np.var(value)
    return dict(amplitude=float(np.hypot(*coefficients[:2])),
                phase_rad=float(np.arctan2(coefficients[1], coefficients[0])),
                mean=float(coefficients[2]), residual_rms=float(np.sqrt(np.mean(residual**2))),
                r_squared=float(1 - np.mean(residual**2) / variance) if variance > 0 else None,
                samples=len(time_s))


def minimize_scalar(function, left: float, right: float) -> float:
    """Bounded golden section, without requiring SciPy."""
    ratio = (np.sqrt(5) - 1) / 2
    x1, x2 = right - ratio * (right - left), left + ratio * (right - left)
    y1, y2 = function(x1), function(x2)
    for _ in range(60):
        if y1 < y2:
            right, x2, y2 = x2, x1, y1
            x1 = right - ratio * (right - left)
            y1 = function(x1)
        else:
            left, x1, y1 = x1, x2, y2
            x2 = left + ratio * (right - left)
            y2 = function(x2)
    return float((left + right) / 2)


def held_values(command: np.ndarray, times: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(command[:, 0], times, side="right") - 1
    if np.any(indices < 0) or np.any(times > command[-1, 0]):
        raise ValueError("Do not extrapolate targets outside the observed C2 interval")
    return command[indices, 1]


def interval_stats(time_s: np.ndarray) -> dict:
    intervals = np.diff(time_s)
    if len(intervals) == 0 or np.any(intervals <= 0):
        raise ValueError("Each trace must have strictly increasing receipt times")
    return dict(mean_rate_hz=float(1 / np.mean(intervals)),
                median_ms=float(np.median(intervals) * 1000),
                p99_ms=float(np.percentile(intervals, 99) * 1000),
                max_ms=float(np.max(intervals) * 1000))


def window_metrics(traces: list[np.ndarray], frequency_hz: float, lo: float, hi: float) -> dict:
    if hi - lo < 3 / frequency_hz:
        raise ValueError("Tracking window must span at least three complete cycle lengths")
    fits = {}
    for name, trace in zip(("command_samples", "position", "velocity", "torque"), traces):
        mask = (trace[:, 0] >= lo) & (trace[:, 0] < hi)
        fits[name] = harmonic(*trace[mask].T, frequency_hz, traces[0][0, 0])
    dense_time = np.arange(lo, hi, .0001)
    fits["command_held"] = harmonic(dense_time, held_values(traces[0], dense_time),
                                    frequency_hz, traces[0][0, 0])
    command, position = fits["command_held"], fits["position"]
    lag_rad = (command["phase_rad"] - position["phase_rad"] + np.pi) % (2 * np.pi) - np.pi
    samples_lag = (fits["command_samples"]["phase_rad"] - position["phase_rad"] + np.pi) % (2*np.pi) - np.pi
    feedback = traces[1][(traces[1][:, 0] >= lo) & (traces[1][:, 0] < hi)]
    valid = np.isfinite(feedback[:, 1])
    error = held_values(traces[0], feedback[valid, 0]) - feedback[valid, 1]
    peaks = {}
    for name, trace in zip(("position", "velocity", "torque"), traces[1:]):
        values = trace[(trace[:, 0] >= lo) & (trace[:, 0] < hi), 1]
        values = values[np.isfinite(values)]
        peaks[name] = dict(min=float(values.min()), max=float(values.max()),
                           abs_max=float(np.abs(values).max()), rms=float(np.sqrt(np.mean(values**2))))
    return dict(window_s=[lo, hi], duration_s=hi-lo, cycle_lengths=(hi-lo)*frequency_hz,
                fits=fits, gain=position["amplitude"] / command["amplitude"],
                phase_lag_deg=float(np.degrees(lag_rad)), phase_equivalent_lag_ms=float(lag_rad/(2*np.pi*frequency_hz)*1000),
                lag_against_ideal_command_ms=float(samples_lag/(2*np.pi*frequency_hz)*1000),
                error_rms_rad=float(np.sqrt(np.mean(error**2))),
                error_p95_abs_rad=float(np.percentile(np.abs(error), 95)),
                error_max_abs_rad=float(np.abs(error).max()),
                mean_tracking_bias_rad=position["mean"]-command["mean"], peaks=peaks)


def correlation_check(traces: list[np.ndarray], frequency_hz: float, lo: float, hi: float) -> dict:
    """Secondary interpolated correlation; raw plotted traces are not resampled."""
    q = traces[1][np.isfinite(traces[1][:, 1])]
    q = q[(q[:, 0] >= lo) & (q[:, 0] < hi)]
    if len(q) < 6 or np.max(np.diff(q[:, 0])) > .05:
        return dict(available=False, reason="A3 gap >50 ms: do not interpolate across outages")
    max_lag_s = .49 / frequency_hz  # Closest positive branch, not an identified pure delay.
    grid = np.arange(max(lo + max_lag_s, q[0, 0]), min(hi, q[-1, 0]), .001)
    feedback = np.interp(grid, q[:, 0], q[:, 1])
    feedback -= np.mean(feedback)
    lags = np.arange(0, max_lag_s, .0005)
    scores = []
    for lag in lags:
        command = held_values(traces[0], grid-lag)
        command = command-command.mean()
        denominator = np.linalg.norm(command)*np.linalg.norm(feedback)
        scores.append(float(command @ feedback / denominator) if denominator else -1.0)
    best = int(np.argmax(scores))
    return dict(available=True, lag_ms=float(lags[best]*1000), correlation=scores[best],
                grid_ms=1.0, lag_step_ms=.5, search_range_ms=[0, max_lag_s*1000])


def final_hold(traces: list[np.ndarray]) -> dict:
    stopped, target = traces[0][-1]
    q = traces[1][np.isfinite(traces[1][:, 1])]
    post = q[q[:, 0] > stopped]
    if len(post) < 2:
        return dict(available=False, reason="No adequate A3 recording after the last observed C2")
    result = dict(available=True, last_target_rad=float(target), observed_after_last_c2_s=float(post[-1, 0]-stopped))
    tail = post[post[:, 0] >= max(stopped, post[-1, 0]-1)]
    result.update(tail_duration_s=float(tail[-1, 0]-tail[0, 0]),
                  tail_median_error_rad=float(np.median(tail[:, 1])-target),
                  tail_max_abs_error_rad=float(np.max(np.abs(tail[:, 1]-target))))
    settling = {}
    for band in (.1, .01, .003, .001):
        outside = np.flatnonzero(np.abs(post[:, 1]-target) > band)
        index = int(outside[-1]+1) if len(outside) else 0
        confirmed = index < len(post) and post[-1, 0]-post[index, 0] >= .25
        settling[str(band)] = float((post[index, 0]-stopped)*1000) if confirmed else None
    result["finite_record_settling_ms_by_band_rad"] = settling
    result["assumption"] = "No later observed C2; not proof the motor was disabled or that no other control command occurred"
    return result


def analyze(source: Path) -> tuple[dict, list[np.ndarray]]:
    info = labels(source)
    traces, provenance = load_recording(source)
    for trace in traces:
        interval_stats(trace[:, 0])
    command = traces[0]
    crossing_frequency = 1/provenance["period_median_s"]
    lo = float(command[0, 0]+max(1.0, 3/crossing_frequency))
    hi = float(command[-1, 0]-max(.25, 1/crossing_frequency))
    mask = (command[:, 0] >= lo) & (command[:, 0] < hi)
    loss = lambda f: harmonic(*command[mask].T, f, command[0, 0])["residual_rms"]
    f = minimize_scalar(loss, crossing_frequency*.99, crossing_frequency*1.01)
    metrics = window_metrics(traces, f, lo, hi)
    frequency_matches = abs(f-info["frequency_hz"]) < .02*info["frequency_hz"]
    amplitude = metrics["fits"]["command_samples"]["amplitude"]
    amplitude_matches = abs(amplitude-info["amplitude_rad"]) < max(.001, .02*info["amplitude_rad"])
    if not frequency_matches or not amplitude_matches:
        raise ValueError(f"Filename disagrees with recorded sine: {source.name}")
    with source.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["motor"] == "steering" and r["record_type"] == "feedback"]
    conversion = np.array([[number(r, key) for key in ("current_a", "torque_nm", "torque_constant_nm_per_a")]
                          for r in rows])
    valid = np.all(np.isfinite(conversion), axis=1) & (np.abs(conversion[:, 0]) > 1e-12)
    torque_sign = float(np.median(conversion[valid, 1]/(conversion[valid, 0]*conversion[valid, 2])))
    if not np.isclose(abs(torque_sign), 1) or not np.allclose(
            conversion[valid, 1], torque_sign*conversion[valid, 0]*conversion[valid, 2], atol=1e-10):
        raise ValueError("Current/torque/Kt conversion is inconsistent")
    blocks = [window_metrics(traces, f, lo+(hi-lo)*i/3, lo+(hi-lo)*(i+1)/3) for i in range(3)]
    metrics["gain_block_min_max"] = [min(b["gain"] for b in blocks), max(b["gain"] for b in blocks)]
    metrics["phase_lag_block_min_max_deg"] = [min(b["phase_lag_deg"] for b in blocks), max(b["phase_lag_deg"] for b in blocks)]
    result = dict(source=str(source.resolve()), source_sha256=provenance["source_sha256"], labels=info,
                  frequency_fit_hz=f, command_amplitude_rad=amplitude, filename_verified=True,
                  provenance=provenance, steady=metrics,
                  timing={name: interval_stats(t[:, 0]) for name, t in zip(("C2", "A3", "A2", "A1"), traces)},
                  steady_timing={name: interval_stats(t[(t[:, 0] >= lo) & (t[:, 0] < hi), 0])
                                 for name, t in zip(("C2", "A3", "A2", "A1"), traces)},
                  feedback_gaps_over_50_ms={name: [[float(t[i, 0]), float(t[i+1, 0])]
                                                   for i in np.flatnonzero(np.diff(t[:, 0]) > .05)]
                                           for name, t in zip(("A3", "A2", "A1"), traces[1:])},
                  missing_queries={field: sum(not math.isfinite(number(r, field)) for r in rows)
                                   for field in ("position_elapsed_s", "velocity_elapsed_s", "current_elapsed_s")},
                  measured_torque_sign=torque_sign,
                  torque_constant_range_nm_per_a=[float(np.nanmin(conversion[:, 2])), float(np.nanmax(conversion[:, 2]))],
                  required_sine_peak_velocity_rad_s=amplitude*2*np.pi*f,
                  fraction_targets_beyond_90_deg=float(np.mean(np.abs(command[:, 1]) > np.pi/2)),
                  correlation_check=correlation_check(traces, f, lo, hi), final_hold=final_hold(traces),
                  method="Discard first max(1 s,3 periods), last max(.25 s,1 period); sin/cos/DC fits at each signal's own receipt times; recorded ZOH command on .1 ms grid; error against latest target; no time shifting",
                  timestamp_caution="Host receipt times only; C2 loopback is not motor acceptance; A3/A2/A1 are sequential",
                  fixture_confirmation="User: steering free to move, motor conditions the same across tests; exact firmware readbacks not logged")
    return result, traces


def first_order_fit(results: list[dict]) -> dict:
    """Descriptive fit H=K/(1+j*w*tau), not firmware identification."""
    selected = [r for r in results if r["labels"]["amplitude_rad"] >= .5]
    omega = np.array([2*np.pi*r["frequency_fit_hz"] for r in selected])
    measured = np.array([r["steady"]["gain"]*np.exp(-1j*np.radians(r["steady"]["phase_lag_deg"]))
                         for r in selected])

    def candidate(log_tau):
        tau = np.exp(log_tau)
        shape = 1/(1+1j*omega*tau)
        gain = float(np.real(np.vdot(shape, measured))/np.real(np.vdot(shape, shape)))
        error = float(np.mean(np.abs(gain*shape-measured)**2))
        return tau, gain, error, gain*shape

    optimum = minimize_scalar(lambda log_tau: candidate(log_tau)[2], np.log(.001), np.log(1))
    tau, gain, error, predicted = candidate(optimum)
    return dict(model="H(jw)=K/(1+jw*tau)", selected_amplitudes_rad=[.5, 1, 2], samples=len(selected),
                tau_s=float(tau), dc_gain=gain, model_implied_corner_hz=float(1/(2*np.pi*tau)),
                complex_response_rmse=float(np.sqrt(error)),
                max_phase_residual_deg=float(np.max(np.abs(np.degrees(np.angle(measured/predicted))))),
                max_gain_residual=float(np.max(np.abs(np.abs(measured)-np.abs(predicted)))),
                caution="Descriptive effective closed-loop model at tested 2/5/10 Hz, not a uniquely identified physical or firmware model; corner below lowest tested frequency is model-implied, not directly measured; no independent validation dataset")


def table(results: list[dict], markdown: bool = False) -> str:
    headings = ("Hz", "Amplitude rad", "Actual amplitude rad", "Gain %", "Phase lag deg", "Lag ms",
                "Error RMS rad", "Peak speed rad/s", "Peak torque Nm")
    body = []
    for r in results:
        s = r["steady"]
        body.append([f'{r["frequency_fit_hz"]:.2f}', f'{r["labels"]["amplitude_rad"]:g}',
                     f'{s["fits"]["position"]["amplitude"]:.4f}', f'{s["gain"]*100:.1f}',
                     f'{s["phase_lag_deg"]:.1f}', f'{s["phase_equivalent_lag_ms"]:.2f}',
                     f'{s["error_rms_rad"]:.4f}', f'{s["peaks"]["velocity"]["abs_max"]:.2f}',
                     f'{s["peaks"]["torque"]["abs_max"]:.3f}'])
    if markdown:
        return "\n".join(["| " + " | ".join(headings) + " |", "| " + " | ".join(["---"]*len(headings)) + " |"]
                         + ["| " + " | ".join(row) + " |" for row in body])
    return '<table><thead><tr>'+''.join(f'<th>{h}</th>' for h in headings)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join(f'<td>{c}</td>' for c in row)+'</tr>' for row in body)+'</tbody></table>'


def save_tracking_plot(traces, result, output):
    lo, hi = result["steady"]["window_s"]
    width = 3/result["frequency_fit_hz"]
    middle = (lo+hi)/2
    left, right = middle-width/2, middle+width/2
    fig, ax = plt.subplots(figsize=(10, 4))
    for trace, label, color, step in ((traces[0], "Observed held C2 target", "#2563eb", True),
                                      (traces[1], "A3 position feedback", "#087f73", False)):
        mask = (trace[:, 0] >= left-.02) & (trace[:, 0] <= right+.02)
        (ax.step if step else ax.plot)(*trace[mask].T, label=label, color=color,
                                      **({"where": "post"} if step else {}))
    ax.set_xlim(left, right)
    ax.set(xlabel="Recorder elapsed time (s); no alignment shift", ylabel="Startup-relative position (rad)",
           title=f'{result["frequency_fit_hz"]:.2f} Hz / ±{result["labels"]["amplitude_rad"]:g} rad — actual tracking')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/"tracking_overlay.png", dpi=160)
    plt.close(fig)


def per_run_report(result, traces, output):
    report = f'''# C2 steering sine analysis: {result['frequency_fit_hz']:.2f} Hz, ±{result['labels']['amplitude_rad']:g} rad

{table([result], markdown=True)}

Measured window: {result['steady']['window_s'][0]:.5f}–{result['steady']['window_s'][1]:.5f} s.
Gain is the ratio of fitted A3 amplitude to the fundamental of the recorded held C2 targets.
Lag is frequency-specific phase divided by 2πf, **not pure motor/CAN delay**.
Peak speed/torque and RMS error use the same steady window; torque is A1 Q-current × B0 Kt, not a shaft sensor.

- [Five-window four-panel HTML](steering_sine_report.html)
- [Full-record PNG](steering_sine_report.png) / [PDF](steering_sine_report.pdf)
- [Middle-second PNG](steering_sine_zoom.png) / [three-cycle overlay](tracking_overlay.png)
- [Plot provenance](summary.json) / [tracking calculations](analysis.json)

Method: {result['method']}.

Timestamp caveat: {result['timestamp_caution']}.
Subwindow ranges reflect within-record variability, not a statistical confidence interval.
No firmware gains/limits/status/voltage readbacks are present. Neither this analysis nor the raw sine helper configures firmware limits.
Raw helper exit does not disable the motor. Grey plot shading does not mean disabled.
Fixture: {result['fixture_confirmation']}.

Source SHA-256: `{result['source_sha256']}`.
Feedback gaps >50 ms (recorder seconds): `{json.dumps(result['feedback_gaps_over_50_ms'])}`.
The plotting helper breaks feedback lines across these gaps and marks them red; no missing measurements are fabricated.
Final-hold statistics in analysis.json are finite-record settling after commands stop changing, not a controlled step experiment.
'''
    (output/"README.md").write_text(report)
    (output/"analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    save_tracking_plot(traces, result, output)
    path = output/"steering_sine_report.html"
    page = path.read_text()
    insertion = '<section class="plots"><h2>Steady-state tracking analysis</h2>'+table([result])+f'''
<p>Window: {result['steady']['window_s'][0]:.3f}–{result['steady']['window_s'][1]:.3f} s.
Gain uses the held-target fundamental. Lag is phase-equivalent at this frequency, not pure latency.
Torques are current-derived. No firmware limit/gain readback or loaded-bike validation.</p>
<p><a href="README.md">Methods and limitations</a> · <a href="analysis.json">Tracking statistics</a> ·
<a href="tracking_overlay.png">Three-cycle tracking overlay</a></p></section>'''
    page = page.replace('<footer>', insertion+'<footer>', 1)
    page = page.replace('</style>', 'table{border-collapse:collapse;display:block;overflow:auto}th,td{padding:7px;border:1px solid #dce4e9;text-align:right;white-space:nowrap}</style>')
    path.write_text(page)


def comparison_plots(results, model, trace_map, output):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fields = (("gain", "Amplitude gain (ratio)"), ("phase_lag_deg", "Phase lag (degrees)"),
              ("phase_equivalent_lag_ms", "Phase-equivalent lag (ms)"), ("error_rms_rad", "RMS error / command amplitude"))
    for amplitude in sorted({r["labels"]["amplitude_rad"] for r in results}):
        selected = [r for r in results if r["labels"]["amplitude_rad"] == amplitude]
        x = [r["frequency_fit_hz"] for r in selected]
        for ax, (field, ylabel) in zip(axes.flat, fields):
            values = [r["steady"][field] / (r["command_amplitude_rad"] if field == "error_rms_rad" else 1) for r in selected]
            ax.plot(x, values, "o-", label=f"±{amplitude:g} rad")
            ax.set(xlabel="Recorded sine frequency (Hz)", ylabel=ylabel)
            ax.grid(alpha=.25)
    frequency = np.linspace(2, 10, 100)
    response = model["dc_gain"]/(1+1j*2*np.pi*frequency*model["tau_s"])
    axes[0, 0].plot(frequency, np.abs(response), "k--", linewidth=1, label="Descriptive first-order fit")
    axes[0, 1].plot(frequency, -np.degrees(np.angle(response)), "k--", linewidth=1)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("C2 free-steering sweep: frequency-dependent gain and phase, not a fixed delay")
    fig.tight_layout()
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output/f"c2_sweep_comparison.{extension}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    for ax, r in zip(axes, [r for r in results if r["labels"]["amplitude_rad"] == .1]):
        lo, hi = r["steady"]["window_s"]
        middle, width = (lo+hi)/2, 3/r["frequency_fit_hz"]
        for trace, label, color, step in ((trace_map[r["source"]][0], "C2 target", "#2563eb", True),
                                         (trace_map[r["source"]][1], "A3 feedback", "#087f73", False)):
            mask = (trace[:, 0] >= middle-width/2-.02) & (trace[:, 0] <= middle+width/2+.02)
            (ax.step if step else ax.plot)(trace[mask, 0]-middle, trace[mask, 1], color=color, label=label,
                                          **({"where": "post"} if step else {}))
        ax.set_xlim(-width/2, width/2)
        ax.set_ylim(-.12, .12)
        ax.set(ylabel="Position (rad)", title=f'{r["frequency_fit_hz"]:.2f} Hz / ±0.1 rad; gain {r["steady"]["gain"]*100:.1f}%')
        ax.grid(alpha=.25)
    axes[0].legend()
    axes[-1].set_xlabel("Time relative to selected window center (s); no command/feedback shift")
    fig.tight_layout()
    fig.savefig(output/"small_amplitude_tracking.png", dpi=160)
    plt.close(fig)


def batch(logs: Path, reports: Path) -> None:
    sources = sorted(logs.glob("*.csv"))
    if not sources:
        raise ValueError(f"No CSVs in {logs}")
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    results, trace_map = [], {}
    reference = reports/"steering_sine_20261003_1439"
    preserved = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in reference.glob("*") if p.is_file()}
    for source in sources:
        result, traces = analyze(source)
        if source.name == "motor_feedback_202610031439_10hz2rad.csv" and reference.is_dir():
            output = reference
            if json.loads((output/"summary.json").read_text())["source_sha256"] != result["source_sha256"]:
                raise ValueError("Original reference report hash mismatch")
            result["reference_report_preserved"] = True
        else:
            output = reports/result["labels"]["report_name"]
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite existing report: {output}")
            with contextlib.redirect_stdout(io.StringIO()):
                write_report(source, output)
            per_run_report(result, traces, output)
        result["report_directory"] = str(output.resolve())
        results.append(result)
        trace_map[result["source"]] = traces
        print(f'{source.name}: gain={result["steady"]["gain"]:.4f}; lag={result["steady"]["phase_equivalent_lag_ms"]:.2f} ms; {output}', flush=True)
    results.sort(key=lambda r: (r["labels"]["frequency_hz"], r["labels"]["amplitude_rad"]))
    model = first_order_fit(results)
    aggregate = reports/"c2_sine_sweep_20261004"
    if aggregate.exists():
        raise FileExistsError(f"Refusing to overwrite existing sweep: {aggregate}")
    aggregate.mkdir(parents=True)
    comparison_plots(results, model, trace_map, aggregate)
    dataset = dict(results=results, descriptive_first_order_fit=model,
                   limitations=["User-reported unchanged free-steering fixture; no independent load or firmware readbacks",
                                "Each condition one record, subwindows are not independent repeats",
                                "Approximately 197 Hz outgoing C2; no 50 Hz policy or motor_io_position test",
                                "Original reference plot files preserved, their embedded source path predates the directory move",
                                "Prior isolated high-amplitude speed-limit explanation is insufficient: strong attenuation also at low demands"])
    (aggregate/"comparison.json").write_text(json.dumps(dataset, indent=2, allow_nan=False)+"\n")
    flat = [dict(frequency_hz=r["frequency_fit_hz"], amplitude_rad=r["labels"]["amplitude_rad"],
                 actual_amplitude_rad=r["steady"]["fits"]["position"]["amplitude"], gain=r["steady"]["gain"],
                 phase_lag_deg=r["steady"]["phase_lag_deg"], lag_ms=r["steady"]["phase_equivalent_lag_ms"],
                 error_rms_rad=r["steady"]["error_rms_rad"], peak_speed_rad_s=r["steady"]["peaks"]["velocity"]["abs_max"],
                 peak_torque_nm=r["steady"]["peaks"]["torque"]["abs_max"], source=r["source"], sha256=r["source_sha256"])
            for r in results]
    with (aggregate/"comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    links = "\n".join(f'- [{r["labels"]["frequency_hz"]:g} Hz / ±{r["labels"]["amplitude_rad"]:g} rad]({os.path.relpath(r["report_directory"], aggregate)}/steering_sine_report.html)' for r in results)
    holds = [r["final_hold"] for r in results if r["final_hold"]["available"]]
    hold_median_error = max(abs(h["tail_median_error_rad"]) for h in holds) if holds else None
    hold_peak_error = max(h["tail_max_abs_error_rad"] for h in holds) if holds else None
    hold_note = (f"Final recorded one-second hold windows have median target errors at most "
                 f"{hold_median_error:.9f} rad and maximum instantaneous error {hold_peak_error:.9f} rad. "
                 "These are encoder/current-record observations after the sine stops changing, not a controlled step or motor-disable test."
                 if holds else "No adequate final-hold records available.")
    narrative = f'''# C2 steering sine sweep (2026-10-03 / 04)

## Result

All 12 recorded commands match their filename frequencies and amplitudes (`01rad` = ±0.1 rad; `05rad` = ±0.5 rad).
The user confirmed free steering and unchanged motor conditions. The C2 response is repeatable but substantially attenuated and phase-lagged,
including small demands. These data support limited effective position-tracking bandwidth, not a single fixed motor delay.

{table(results, markdown=True)}

Gain uses the fundamental of the actual held C2 targets, not peak-to-peak extrema that include startup/final hold.
Every row uses a documented steady window in comparison.json: exclude first max(1 s,3 periods), last max(.25 s,1 period).
Lag is phase divided by 2πf. Peaks/error use that same window, not the full record. Torque is current-derived, not a shaft measurement.

## Conclusions and limits

1. **Attenuation is not merely the old ±2 rad/10 Hz speed demand.** At ±0.1 rad, demands at 2/5/10 Hz are only approximately 1.26/3.14/6.28 rad/s,
   all below deployment's 23.9 rad/s reference, yet the gain is still about 0.64/0.33/0.17. Larger amplitudes yield similar gain/phase at the same frequency.
   Thus limited closed-loop tracking response is strongly supported. Exact active speed/current limits and firmware gains remain unmeasured.
2. **A phase-equivalent lag is not fixed latency.** It varies approximately 67–70 ms at 2 Hz, 39–40 ms at 5 Hz, and 22–23 ms at 10 Hz,
   while phase lag increases. The earlier approximately 23 ms estimate remains specific to 10 Hz, not a universal actuator delay.
3. **A simple low-pass model summarizes the larger-amplitude records.** Equal-weight complex-response fit of ±0.5/1/2 rad gives
   `H(jw)=K/(1+jw*tau)`, K={model['dc_gain']:.4f}, tau={model['tau_s']*1000:.2f} ms, model-implied corner {model['model_implied_corner_hz']:.3f} Hz;
   complex-response RMS residual {model['complex_response_rmse']:.5f}, maximum phase discrepancy {model['max_phase_residual_deg']:.2f}°.
   This is a descriptive fit, **not a uniquely identified firmware/physical model**. Its corner is below the lowest tested 2 Hz and is not directly measured.
   The 0.1 rad points are not used to fit it; compare their small additional attenuation/phase lag rather than inventing a deadband or stiction threshold.
4. **C2 bench behavior is promising, not deployment certification.** Position traces are largely sinusoidal rather than growing autonomous oscillations,
   and final-hold metrics are provided where recording permits. {hold_note} Current-derived torque still fluctuates.
   There is no matched C0/C2 test, 50 Hz target-interface test, ground-loaded steering, or balancing policy experiment here.
   A 50 Hz policy does not mean it demands 50 Hz physical steering motion; actuator compatibility depends on the spectrum of its actual commands.
5. **Improve/characterize the actuator interface before judging the policy.** Read back B2/B3, B6–B9 and AE/status/voltage; do not copy host PD gains into firmware PI registers.
   Extend to 0.2/0.5/1 Hz and supervised small steps/enable transitions, then repeat small sine commands at the actual 50 Hz policy interface.
   Check loaded behavior while independently supporting the bike. Do not blindly increase gains, disable protections, or retry unsupported balance based on this sweep.

## Recording integrity

All source hashes and count-to-angle/current-to-torque conversions were checked; every steering query row contains its separate timestamp/value.
The 2 Hz/±2 rad record has approximately 342 ms and 106 ms feedback gaps before the selected steady window;
there are no >50 ms A3/A2/A1 gaps inside any selected steady window. Full plots mark feedback outages rather than connect through them.
They establish receipt/recording gaps, not the cause or the motor's physical behavior inside them.

## Artifacts

- [Comparison PNG](c2_sweep_comparison.png) / [PDF](c2_sweep_comparison.pdf) / [SVG](c2_sweep_comparison.svg)
- [Small-amplitude tracking overlays](small_amplitude_tracking.png)
- [Comparison CSV](comparison.csv) / [complete JSON and SHA-256 provenance](comparison.json)
- [Offline HTML index](index.html)

{links}

Each new condition includes the existing workflow's PNG/PDF, five SVG windows and selectable four-panel HTML,
plus README.md, analysis.json, and a three-cycle target/feedback overlay. The original 10 Hz/±2 rad report is preserved byte-for-byte;
its new comparison statistics and relocated source link are in this sweep, not written over the old report.

## Measurement caveats / reproduction

Recorded C2 is observer SocketCAN-loopback receipt, not exact motor acceptance. A3/A2/A1 are sequential host-receipt times.
Plots retain raw values and clocks with no smoothing or phase shifts. Fits/correlation use documented numerical sampling separately.
Subwindow ranges are within-record variability, not repeat-trial confidence intervals. No firmware sampling rate can be inferred.
Last-C2 settling assumes no later control command outside this capture; the raw sine script does not disable on exit.

From the repository root: `MPLCONFIGDIR=/tmp/trackstand_c2_mpl python3 deploy/experiments/analyze_c2_sine.py --reports /tmp/new_c2_sweep_reports`.
Use a fresh output directory: this helper refuses to overwrite existing reports. It never opens CAN or modifies source CSVs/runtime/configurations.
'''
    (aggregate/"README.md").write_text(narrative)
    cards = ''.join(f'<li><a href="{html.escape(os.path.relpath(r["report_directory"], aggregate))}/steering_sine_report.html">{r["labels"]["frequency_hz"]:g} Hz / ±{r["labels"]["amplitude_rad"]:g} rad</a> · '
                    f'<a href="{html.escape(os.path.relpath(r["source"], aggregate))}">source CSV</a></li>' for r in results)
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>C2 steering sweep comparison</title><style>body{{font:15px/1.6 system-ui;margin:24px auto;max-width:1200px;padding:0 18px;color:#172a3a}}
table{{border-collapse:collapse;display:block;overflow:auto}}th,td{{padding:7px;border:1px solid #ddd;white-space:nowrap;text-align:right}}
img{{width:100%;height:auto}}a{{color:#2563eb}}</style><h1>C2 steering sine sweep — 12 measured conditions</h1>
<p>Free-steering / user-reported unchanged motor conditions. All filename labels checked against actual recorded commands.</p>
<p>Attenuation persists at low amplitude: limited tracking bandwidth, not merely large-command speed demand.
Lag below means frequency-specific phase-equivalent lag, not pure latency. Torque is current-derived.</p>
{table(results)}<img src="c2_sweep_comparison.png" alt="Gain, phase, phase-equivalent lag and normalized RMS error by frequency">
<h2>Small demands still show attenuation</h2><img src="small_amplitude_tracking.png" alt="±0.1 rad command and actual feedback at 2,5,10 Hz">
<p>Descriptive first-order fit to amplitudes ≥0.5 rad: K={model['dc_gain']:.4f}, tau={model['tau_s']*1000:.2f} ms,
model-implied corner={model['model_implied_corner_hz']:.3f} Hz (below tested range; not directly measured).</p>
<h2>Individual four-panel reports</h2><ul>{cards}</ul>
<p><a href="README.md">Complete conclusions, methods and limitations</a> · <a href="comparison.json">JSON provenance and statistics</a> ·
<a href="comparison.csv">CSV table</a> · <a href="c2_sweep_comparison.pdf">Comparison PDF</a></p>
<p>Approximately 197 Hz outgoing targets, no policy or 50 Hz interface test; firmware gains/limits/status and ground-loaded response are unknown.
Plots are unshifted. Raw helper exit is not motor disable. Original reference report untouched; its old embedded source path predates relocation.</p></html>'''
    (aggregate/"index.html").write_text(page)
    assert hashes == {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    assert preserved == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in reference.glob("*") if p.is_file()}
    print(json.dumps(model, indent=2))
    print(f"Sweep: {aggregate/'index.html'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, default=ROOT/"logs")
    parser.add_argument("--reports", type=Path, default=ROOT/"reports")
    arguments = parser.parse_args()
    batch(arguments.logs.resolve(), arguments.reports.resolve())
