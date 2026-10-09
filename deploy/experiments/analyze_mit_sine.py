#!/usr/bin/env python3
"""Offline MIT sine reports and a hash-checked comparison with the C2 sweep.

Never opens CAN/ROS or changes a recording, controller, or existing report.
Uses the C2 analysis's numerical methods, but reads MIT's sparse wire records.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import struct

import numpy as np

from analyze_c2_sine import (analyze as analyze_c2, correlation_check, harmonic,
                             interval_stats, minimize_scalar, window_metrics)
from plot_motor_feedback import number, plt
from steering_sine_mit_io import Motor, RADIANS_PER_COUNT, decode_symmetric, target_fields


ROOT = Path(__file__).resolve().parent
LABEL = re.compile(r"(\d{8})(\d{4})_mit_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)Hz\.csv")
TRACE_NAMES = ("MIT encoded position target", "A3 position feedback", "A2 velocity feedback", "A1 current-derived torque")
METHOD = ("Exclude first max(1 s, 3 periods) and last max(.25 s, 1 period); "
          "fit sin/cos/DC at each trace's own timestamp; held encoded input on .1 ms grid; "
          "RMS error against latest encoded target, no alignment shift or smoothing")
CAUTIONS = ("Free steering/front wheel confirmed by user; not ground-loaded balancing. "
            "Host send/receive times, not wire, acquisition or execution times. "
            "Phase-equivalent lag is frequency-specific, not identified pure delay. "
            "A1 torque is Q-current times read B0 Kt, not an independent shaft sensor. "
            "C2 ran approximately 197 Hz and MIT approximately 50 Hz; different days and "
            "unrecorded C2 firmware gains/limits prevent a strictly controlled mode-only comparison. "
            "No identified C3 experimental record: C3 performance cannot be quantified here.")


def labels(source: Path) -> dict:
    match = LABEL.fullmatch(source.name)
    if not match:
        raise ValueError(f"Unsupported MIT filename: {source.name}")
    date, clock, kp, kd, frequency = match.groups()
    def value(token):
        return float("0." + token[1:] if token.startswith("0") and "." not in token else token)
    if float(frequency) <= 0:
        raise ValueError("Sine frequency must be positive")
    return dict(date=date, clock=clock, kp=value(kp), kd=value(kd), frequency_hz=float(frequency),
                amplitude_rad=1.0, report_name=f"steering_sine_{date}_{clock}_mit_{kp}_{kd}_{frequency}hz1rad")


def close_field(row: dict, field: str, expected: float) -> None:
    actual = number(row, field)
    if not math.isfinite(actual) or not math.isclose(actual, expected, abs_tol=1e-9, rel_tol=1e-9):
        raise ValueError(f"Wire/CSV mismatch for {field}: {actual} != {expected}")


def check_wire(row: dict) -> None:
    """Validate applicable values against raw payload, including independent zeros/signs."""
    payload = bytes.fromhex(row["mit_payload_hex"])
    if len(payload) != int(row["can_length"]):
        raise ValueError("CAN length disagrees with payload")
    motor = Motor(row["motor"], int(row["motor_id"]), number(row, "gear_ratio"),
                  int(row["encoder_sign"]), int(row["torque_sign"]),
                  int(row["zero_count"]), number(row, "mit_zero_position_motor_rad"),
                  number(row, "torque_constant_nm_per_a"),
                  tuple(number(row, k) for k in ("mit_position_max_rad", "mit_velocity_max_rad_s", "mit_torque_max_nm")))
    if motor.gear_ratio <= 0 or motor.encoder_sign not in (-1, 1) or motor.torque_sign not in (-1, 1):
        raise ValueError("Invalid sign/gearing metadata")
    if row["record_type"] == "mit_target":
        if len(payload) != 8 or int(row["mit_can_id"]) != 0x500 | motor.device_id:
            raise ValueError("Not an integrated-sender MIT motion frame")
        for field, expected in target_fields(motor, payload).items():
            close_field(row, field, expected)
        requested = number(row, "requested_position_rad")
        if abs(requested - number(row, "mit_target_position_rad")) > motor.ranges[0] / 65535 / motor.gear_ratio + 1e-9:
            raise ValueError("Requested/encoded target discrepancy exceeds half a position LSB")
        return
    if int(row["mit_can_id"]) != motor.device_id:
        raise ValueError("Feedback motor address mismatch")
    if row["record_type"] == "mit_feedback":
        if len(payload) != 7 or payload[0] != 0xF1:
            raise ValueError("Malformed F1 record")
        codes = (int.from_bytes(payload[1:3], "big"), payload[3] << 4 | payload[4] >> 4,
                 (payload[4] & 15) << 8 | payload[5])
        native = [decode_symmetric(c, m, b) for c, m, b in zip(codes, motor.ranges, (16, 12, 12))]
        for kind, code, val in zip(("position", "velocity", "torque"), codes, native):
            close_field(row, f"mit_{kind}_code", code)
            native_field = {"position": "rad", "velocity": "rad_s", "torque": "nm"}[kind]
            close_field(row, f"mit_feedback_{kind}_motor_{native_field}", val)
        close_field(row, "mit_feedback_position_rad", motor.encoder_sign * (native[0] - motor.zero_rad) / motor.gear_ratio)
        close_field(row, "mit_feedback_velocity_rad_s", motor.encoder_sign * native[1] / motor.gear_ratio)
        close_field(row, "mit_feedback_torque_nm", motor.torque_sign * native[2])
        close_field(row, "mit_status_raw", payload[6])
        if row["mit_is_active"] != str(bool(payload[6] & 1)) or row["mit_has_fault"] != str(bool(payload[6] & 2)):
            raise ValueError("F1 status bits disagree")
        return
    opcode = int(row["can_opcode"])
    if not payload or payload[0] != opcode or len(payload) != {0xA3: 7, 0xA2: 5, 0xA1: 5}.get(opcode):
        raise ValueError("Malformed sparse A3/A2/A1 record")
    if opcode == 0xA3:
        count = struct.unpack("<i", payload[3:])[0]
        position = motor.encoder_sign * (count - motor.zero_count) * RADIANS_PER_COUNT / motor.gear_ratio
        close_field(row, "encoder_count", count)
        close_field(row, "encoder_single_turn_count", int.from_bytes(payload[1:3], "little"))
        close_field(row, "position_unwrapped_rad", position)
        close_field(row, "position_rad", (position + math.pi) % (2 * math.pi) - math.pi)
    elif opcode == 0xA2:
        close_field(row, "velocity_rad_s", motor.encoder_sign * struct.unpack("<i", payload[1:])[0]
                    * .01 * 2 * math.pi / 60 / motor.gear_ratio)
    else:
        current = struct.unpack("<i", payload[1:])[0] * .001
        close_field(row, "current_a", current)
        close_field(row, "torque_nm", motor.torque_sign * current * motor.torque_constant)


def trace(rows: list[dict], field: str) -> np.ndarray:
    samples = np.array([[number(r, "elapsed_s"), number(r, field)] for r in rows], dtype=float).reshape(-1, 2)
    if len(samples) < 6 or not np.all(np.isfinite(samples)):
        raise ValueError(f"Missing or invalid samples for {field}; no invented replacements")
    interval_stats(samples[:, 0])  # Reject duplicate/reversed clocks rather than silently sort.
    return samples


def load_recording(source: Path) -> tuple[list[np.ndarray], dict, dict]:
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(None in r or None in r.values() for r in rows):
        raise ValueError("Truncated/invalid CSV row")
    selected = [r for r in rows if r["motor"] == "steering" and r["phase"] == "motion"]
    commands = [r for r in selected if r["record_type"] == "mit_target"]
    feedback = [r for r in selected if r["record_type"] == "feedback"]
    f1 = [r for r in selected if r["record_type"] == "mit_feedback"]
    if not commands or not feedback or not f1:
        raise ValueError("Need MIT targets, sparse A3/A2/A1, and F1 motion records")
    fields = ("motor_id", "gear_ratio", "encoder_sign", "torque_sign", "reference_epoch", "zero_count",
              "mit_zero_position_motor_rad", "mit_position_max_rad", "mit_velocity_max_rad_s", "mit_torque_max_nm",
              "torque_constant_nm_per_a")
    for field in fields:
        if len({r[field] for r in commands + feedback + f1}) != 1:
            raise ValueError(f"Reference/config changed during motion: {field}")
    for row in commands + feedback + f1:
        check_wire(row)
    # Verify calibration/range metadata against actual startup readbacks, not
    # merely mutually consistent decoded CSV columns. Initial pre-write F0 may differ.
    config_rows = [r for r in rows if r["motor"] == "steering" and r["record_type"] == "can_rx"
                   and r["can_opcode"] == str(0xF0) and r["can_length"] == "7"]
    constant_rows = [r for r in rows if r["motor"] == "steering" and r["record_type"] == "can_rx"
                     and r["can_opcode"] == str(0xB0) and r["can_length"] == "7"]
    if not config_rows or not constant_rows:
        raise ValueError("Missing raw F0/B0 readbacks")
    config = bytes.fromhex(config_rows[-1]["mit_payload_hex"])
    constant = bytes.fromhex(constant_rows[-1]["mit_payload_hex"])
    if len(config) != 7 or config[0] != 0xF0 or len(constant) != 7 or constant[0] != 0xB0:
        raise ValueError("Malformed raw F0/B0 readback")
    for field, raw, scale in zip(("mit_position_max_rad", "mit_velocity_max_rad_s", "mit_torque_max_nm"),
                                 struct.unpack("<HHH", config[1:]), (.1, .01, .01)):
        close_field(commands[0], field, raw*scale)
    close_field(commands[0], "torque_constant_nm_per_a", struct.unpack("<f", constant[2:6])[0])
    for row in rows:
        if row["record_type"] == "motor_status" and row["motor"] == "steering":
            payload = bytes.fromhex(row["mit_payload_hex"])
            if len(payload) != 8 or payload[0] not in (0xAE, 0xCF):
                raise ValueError("Malformed raw status readback")
            for field, expected in (("bus_voltage_v", int.from_bytes(payload[1:3], "little")*.01),
                                    ("motor_temperature_c", payload[5]), ("motor_mode_raw", payload[6]),
                                    ("motor_fault_code", payload[7])):
                close_field(row, field, expected)
    traces = [trace(commands, "mit_target_position_rad")]
    traces.extend(trace([r for r in feedback if int(r["can_opcode"]) == opcode], field)
                  for opcode, field in ((0xA3, "position_unwrapped_rad"), (0xA2, "velocity_rad_s"), (0xA1, "torque_nm")))
    query_counts = Counter((r["query_opcode"], r["query_status"]) for r in selected if r["record_type"] == "query_result")
    gaps = {name: [[float(t[i, 0]), float(t[i+1, 0])] for i in np.flatnonzero(np.diff(t[:, 0]) > .05)]
            for name, t in zip(TRACE_NAMES, traces)}
    status = [dict(phase=r["phase"], elapsed_s=number(r, "elapsed_s"), payload_hex=r["mit_payload_hex"],
                   mode=int(r["motor_mode_raw"]), fault=int(r["motor_fault_code"]),
                   voltage_v=number(r, "bus_voltage_v"), temperature_c=number(r, "motor_temperature_c"))
              for r in rows if r["record_type"] == "motor_status" and r["motor"] == "steering"]
    events = [dict(phase=r["phase"], elapsed_s=number(r, "elapsed_s"), note=r["can_note"])
              for r in rows if r["record_type"] == "event" and r["motor"] == "steering"]
    observed_drops = [number(r, "socket_dropped_frames") for r in rows if r["socket_dropped_frames"]]
    round_trip = [number(r, "query_round_trip_s") for r in selected if r["record_type"] == "query_result"]
    summary = dict(source=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   total_rows=len(rows), row_types=dict(Counter(r["record_type"] for r in rows)),
                   metadata={k: number(commands[0], k) for k in fields},
                   command_start_s=float(traces[0][0, 0]), command_end_s=float(traces[0][-1, 0]),
                   recording_duration_s=max(number(r, "elapsed_s") for r in rows),
                   traces=[dict(name=name, valid_samples=len(t), min=float(t[:, 1].min()), max=float(t[:, 1].max()),
                                **interval_stats(t[:, 0])) for name, t in zip(TRACE_NAMES, traces)],
                   feedback_gaps_over_50_ms=gaps, query_results={f"0x{int(op):02X}/{state}": count
                                                               for (op, state), count in sorted(query_counts.items())},
                   query_round_trip_p99_ms=float(np.percentile(round_trip, 99) * 1000),
                   socket_dropped_frames_max=max(observed_drops, default=0),
                   f1_motion_active_rows=sum(r["mit_is_active"] == "True" for r in f1),
                   f1_motion_fault_rows=sum(r["mit_has_fault"] == "True" for r in f1),
                   status=status, events=events, shutdown_confirmed=any(
                       e["phase"] == "shutdown" and e["note"] == "CF disabled mode acknowledged" for e in events),
                   wire_rows_checked=len(commands)+len(feedback)+len(f1), timestamp_caution=CAUTIONS)
    extras = dict(commands=commands, requested=trace(commands, "requested_position_rad"),
                  f1_position=trace(f1, "mit_feedback_position_rad"))
    return traces, summary, extras


def analyze(source: Path) -> tuple[dict, list[np.ndarray]]:
    info = labels(source)
    traces, summary, extra = load_recording(source)
    lo = summary["command_start_s"] + max(1.0, 3 / info["frequency_hz"])
    hi = summary["command_end_s"] - max(.25, 1 / info["frequency_hz"])
    requested = extra["requested"]
    mask = (requested[:, 0] >= lo) & (requested[:, 0] < hi)
    f = minimize_scalar(lambda f: harmonic(*requested[mask].T, f, requested[0, 0])["residual_rms"],
                        info["frequency_hz"]*.99, info["frequency_hz"]*1.01)
    sine_fit = harmonic(*requested[mask].T, f, requested[0, 0])
    if abs(f - info["frequency_hz"]) > .02 * info["frequency_hz"] or abs(sine_fit["amplitude"] - 1) > .005:
        raise ValueError("Filename/user amplitude disagrees with recorded requested sine")
    commands = extra["commands"]
    encoded = {}
    for field, expected, tolerance in (("mit_kp_nm_per_motor_rad", info["kp"], 250/4095),
                                       ("mit_kd_nm_s_per_motor_rad", info["kd"], 2.5/4095)):
        values = {number(r, field) for r in commands}
        if len(values) != 1 or abs(next(iter(values)) - expected) > tolerance + 1e-9:
            raise ValueError(f"Filename gains disagree with encoded {field}")
        encoded[field] = values.pop()
    for field in ("mit_target_velocity_motor_rad_s", "mit_feedforward_torque_motor_nm"):
        values = {number(r, field) for r in commands}
        if len(values) != 1:
            raise ValueError("Target velocity/feedforward changed during motion")
        encoded[field] = values.pop()
    for field, maximum in (("mit_target_velocity_motor_rad_s", summary["metadata"]["mit_velocity_max_rad_s"]),
                           ("mit_feedforward_torque_motor_nm", summary["metadata"]["mit_torque_max_nm"])):
        if abs(encoded[field]) > maximum/4095 + 1e-9:
            raise ValueError("Velocity/feedforward exceeds quantized requested-zero range")
    steady = window_metrics(traces, f, lo, hi)
    blocks = [window_metrics(traces, f, lo+(hi-lo)*i/3, lo+(hi-lo)*(i+1)/3) for i in range(3)]
    for field in ("gain", "phase_lag_deg", "error_rms_rad"):
        steady[field+"_block_min_max"] = [min(b[field] for b in blocks), max(b[field] for b in blocks)]
    f1 = extra["f1_position"]
    fm = harmonic(*f1[(f1[:, 0] >= lo) & (f1[:, 0] < hi)].T, f, traces[0][0, 0])
    lag = (steady["fits"]["command_held"]["phase_rad"] - fm["phase_rad"] + np.pi) % (2*np.pi) - np.pi
    velocity = traces[2][(traces[2][:, 0] >= lo) & (traces[2][:, 0] < hi), 1]
    torque = traces[3][(traces[3][:, 0] >= lo) & (traces[3][:, 0] < hi), 1]
    intervals = np.array([number(r, "command_interval_s") for r in commands[1:]])
    if not np.allclose(intervals, np.diff(traces[0][:, 0]), rtol=0, atol=1e-9):
        raise ValueError("Logged command intervals disagree with command timestamps")
    return dict(source=summary["source"], source_sha256=summary["source_sha256"], labels=info,
                filename_verified=True, frequency_fit_hz=f, requested_sine_fit=sine_fit,
                command_amplitude_rad=sine_fit["amplitude"], encoded_settings=encoded, provenance=summary,
                steady=steady, blocks=blocks, method=METHOD, limitations=CAUTIONS,
                correlation_check=correlation_check(traces, f, lo, hi),
                f1_cross_check=dict(fit=fm, gain=fm["amplitude"]/steady["fits"]["command_held"]["amplitude"],
                                    phase_lag_deg=float(np.degrees(lag))),
                steady_timing={name: interval_stats(t[(t[:, 0] >= lo) & (t[:, 0] < hi), 0])
                               for name, t in zip(TRACE_NAMES, traces)},
                command_lateness_max_ms=max(number(r, "command_lateness_s") for r in commands)*1000,
                skipped_command_ticks=sum(int(r["skipped_command_ticks"]) for r in commands),
                ideal_sine_peak_speed_rad_s=2*np.pi*f,
                fraction_steady_speed_over_20_rad_s=float(np.mean(np.abs(velocity) > 20)),
                fraction_steady_torque_over_training_5_6_nm=float(np.mean(np.abs(torque) > 5.6)),
                maximum_recorded_target_slew_rad_s=float(np.max(np.abs(np.diff(traces[0][:, 1])) / intervals))), traces


def table(results: list[dict], markdown: bool = False) -> str:
    headers = ("Mode", "Hz", "Kd", "A3 amp rad", "Gain %", "Phase °", "Lag ms", "RMS error rad", "Peak torque Nm", "Position R²")
    cells = []
    for r in results:
        s = r["steady"]
        cells.append([r.get("mode", "MIT"), f'{r["frequency_fit_hz"]:.0f}',
                      f'{r["labels"]["kd"]:g}' if "kd" in r["labels"] else "unknown PI",
                      f'{s["fits"]["position"]["amplitude"]:.4f}', f'{100*s["gain"]:.1f}',
                      f'{s["phase_lag_deg"]:.1f}', f'{s["phase_equivalent_lag_ms"]:.2f}',
                      f'{s["error_rms_rad"]:.4f}', f'{s["peaks"]["torque"]["abs_max"]:.3f}',
                      f'{s["fits"]["position"]["r_squared"]:.4f}'])
    if markdown:
        return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"]
                         + ["| " + " | ".join(row) + " |" for row in cells])
    return "<table>" + "".join("<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in row) + "</tr>"
                               for row, tag in [(headers, "th")] + [(row, "td") for row in cells]) + "</table>"


def plot_line(ax, trace, color, label=None, step=False):
    if step:
        ax.step(*trace.T, where="post", color=color, label=label, linewidth=.9)
    else:
        gaps = np.flatnonzero(np.diff(trace[:, 0]) > .05)
        separated = np.insert(trace, gaps+1, np.nan, axis=0)
        ax.plot(*separated.T, color=color, label=label, linewidth=.9)
        for index in gaps:
            ax.axvspan(trace[index, 0], trace[index+1, 0], color="#fca5a5", alpha=.3)


def four_panel(traces, result, window, title):
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 10.4))
    fig.subplots_adjust(left=.09, right=.975, top=.935, bottom=.085, hspace=.36)
    info = result["labels"]
    fig.suptitle(f'MIT steering · {info["frequency_hz"]:g} Hz / ±1 rad / Kp {info["kp"]:g}, Kd {info["kd"]:g} · {title}', fontsize=15)
    pos_limit = 1.1 * max(float(np.max(np.abs(t[:, 1]))) for t in traces[:2])
    start, end = result["provenance"]["command_start_s"], result["provenance"]["command_end_s"]
    for i, (ax, t, color, unit) in enumerate(zip(axes, traces, ("#2563eb", "#087f73", "#8256bd", "#b85a16"),
                                               ("Position (rad)", "Position (rad)", "Velocity (rad/s)", "Torque (Nm)"))):
        selected = t[(t[:, 0] >= window[0]-.025) & (t[:, 0] <= window[1]+.025)]
        plot_line(ax, selected, color, step=i == 0)
        ax.set_title(f"{i+1}  {TRACE_NAMES[i]}", loc="left", fontsize=11, color=color)
        ax.set_ylabel(unit)
        limit = pos_limit if i < 2 else 1.1 * float(np.max(np.abs(t[:, 1])))
        ax.set_ylim(-max(limit, .001), max(limit, .001))
        ax.set_xlim(*window)
        ax.grid(alpha=.25)
        ax.axhline(0, color="#94a3b8", lw=.7)
        for boundary in (start, end):
            if window[0] < boundary < window[1]:
                ax.axvline(boundary, color="#94a3b8", ls="--", lw=.8)
    axes[-1].set_xlabel("Recorder elapsed time (s); no phase shift")
    fig.text(.09, .02, "Positive = logged joint direction (right); A3/F1 independent startup zeros. Red = feedback gap >50 ms.\n"
             "Torque is current-derived; final CF disabled-mode confirmation failed. Motion feedback only; no invented shutdown hold.", fontsize=8)
    return fig


STYLE = """body{font:15px/1.6 system-ui;margin:24px auto;max-width:1220px;padding:0 18px;color:#172a3a}
table{border-collapse:collapse;display:block;overflow:auto}th,td{padding:7px;border:1px solid #ddd;white-space:nowrap;text-align:right}
img,svg{width:100%;height:auto}a{color:#2563eb}.view{display:none}.view.active{display:block}
button{padding:8px 12px;margin:4px;border:1px solid #ccc;background:white;border-radius:5px;cursor:pointer}
button[aria-pressed=true]{background:#172a3a;color:white}.warning{background:#fff3d6;padding:12px;border-left:4px solid #dc9822}
@media print{button{display:none}}"""


def write_run(result, traces, output):
    output.mkdir(parents=True, exist_ok=False)
    p = result["provenance"]
    start, end = p["command_start_s"], p["command_end_s"]
    middle, width = (start+end)/2, min(1, end-start)
    windows = [("Full recording", (0, p["recording_duration_s"])), ("MIT interval", (start, end)),
               ("First 1 s", (start, start+width)), ("Middle 1 s", (middle-width/2, middle+width/2)),
               ("Final 1 s", (end-width, end))]
    sections = []
    for i, (label, window) in enumerate(windows):
        fig = four_panel(traces, result, window, label)
        svg_path = output / f"steering_view_{i}.svg"
        fig.savefig(svg_path, facecolor="white")
        if i == 0:
            fig.savefig(output/"steering_sine_report.png", dpi=160, facecolor="white")
            fig.savefig(output/"steering_sine_report.pdf", facecolor="white")
        if i == 3:
            fig.savefig(output/"steering_sine_zoom.png", dpi=160, facecolor="white")
        plt.close(fig)
        svg = svg_path.read_text()
        svg = svg[svg.index("<svg"):].replace('id="', f'id="v{i}-').replace('url(#', f'url(#v{i}-').replace('href="#', f'href="#v{i}-')
        sections.append(f'<section class="view{" active" if i == 0 else ""}" id="view-{i}">{svg}</section>')
    lo, hi = result["steady"]["window_s"]
    center, width = (lo+hi)/2, 3/result["frequency_fit_hz"]
    fig, ax = plt.subplots(figsize=(11, 4))
    for t, color, label, step in zip(traces[:2], ("#2563eb", "#087f73"), TRACE_NAMES[:2], (True, False)):
        plot_line(ax, t[(t[:, 0] >= center-width/2-.025) & (t[:, 0] <= center+width/2+.025)], color, label, step)
    ax.set(xlim=(center-width/2, center+width/2), xlabel="Recorder elapsed time (s); no alignment shift",
           ylabel="Startup-relative position (rad)", title=f'{result["frequency_fit_hz"]:.0f} Hz · three-cycle target/feedback overlay')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/"tracking_overlay.png", dpi=160)
    plt.close(fig)
    p["views"] = [dict(label=label, xlim_s=window) for label, window in windows]
    (output/"summary.json").write_text(json.dumps(p, indent=2, allow_nan=False)+"\n")
    (output/"analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    report = f'''# MIT steering sine: {result['frequency_fit_hz']:.0f} Hz, ±1 rad, Kp={result['labels']['kp']:g}, Kd={result['labels']['kd']:g}

{table([result], True)}

Gain = fitted A3 fundamental / recorded **encoded held-target** fundamental.
The A3 amplitude column is the actual sinusoidal component, not a full-record peak or fitted gain.
Steady window: {lo:.6f}–{hi:.6f} s. Method: {METHOD}.
Phase-equivalent lag is **not pure motor delay**. Position R² measures sinusoidal fit quality, not controller stability.
Subwindow ranges in analysis.json are within-record variability, not repeat-trial confidence intervals.

## Recording verification

- {p['wire_rows_checked']:,} target/A3/A2/A1/F1 rows decoded again from raw payloads; conversions and references checked.
- Encoded settings: `{json.dumps(result['encoded_settings'])}`. Quantization makes requested zero velocity/feedforward slightly nonzero.
- Target rate {p['traces'][0]['mean_rate_hz']:.3f} Hz; A3 {p['traces'][1]['mean_rate_hz']:.3f} Hz; maximum command lateness {result['command_lateness_max_ms']:.3f} ms; skipped ticks {result['skipped_command_ticks']}.
- Read outcomes: `{json.dumps(p['query_results'])}`. Late/unmatched replies can still appear as received measurements; host receipt is not acquisition time.
- Socket-reported drops maximum {p['socket_dropped_frames_max']:g}; F1 motion fault rows {p['f1_motion_fault_rows']}; gaps: `{json.dumps(p['feedback_gaps_over_50_ms'])}`.
- F1 position cross-check: gain {result['f1_cross_check']['gain']*100:.2f}%, phase {result['f1_cross_check']['phase_lag_deg']:.2f}°. It is another motor report, not an independent physical sensor.
- A2 peak {result['steady']['peaks']['velocity']['abs_max']:.3f} rad/s; ideal continuous sine demands {result['ideal_sine_peak_speed_rad_s']:.3f} rad/s.
- A1 samples above training's 5.6 Nm reference: {100*result['fraction_steady_torque_over_training_5_6_nm']:.3f}%. This is a sample fraction, not an exact time fraction or verified mechanical overload.

## Important shutdown discrepancy

**The log does not confirm disabled mode after CF.** Events: `{json.dumps(p['events'])}`.
Status/raw bytes: `{json.dumps(p['status'])}`.
The script expects CF byte 6 = 0; these recordings end with byte 6 = 5, fault byte = 0.
The local protocol documents mode 0 as disabled, but does not define mode 5 in the AE table.
This could be firmware reply/transition semantics or failure to disable; these logs do not distinguish them.
Do not treat a successfully sent CF or a fault-free reply as physical disable. Check subsequent AE/F1 and physical free state with independent support before powered deployment.
No final-hold/settling result is inferred: the experiment sends CF rather than holding a final position.

## Artifacts and limitations

- [Four-panel selectable HTML](steering_sine_report.html)
- [Full PNG](steering_sine_report.png) / [PDF](steering_sine_report.pdf)
- [Middle-second PNG](steering_sine_zoom.png) / [three-cycle overlay](tracking_overlay.png)
- [Plot summary](summary.json) / [complete tracking analysis](analysis.json)
- [Source CSV]({os.path.relpath(result['source'], output)})

{CAUTIONS}
No B2/B3 or position/speed PI gain readback is present. F0 ranges (3 rad,45 rad/s,18 Nm) are encoding metadata, not verified total-PD safety limits.
Standalone sine does not reproduce deployment's target slew, safety limits, rear motor, policy or ground-loaded bicycle.

Source SHA-256: `{result['source_sha256']}`.
'''
    (output/"README.md").write_text(report)
    buttons = "".join(f'<button data-view="{i}" aria-pressed="{str(i == 0).lower()}">{label}</button>'
                      for i, (label, _) in enumerate(windows))
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MIT steering sine — four aligned traces</title><style>{STYLE}</style></head><body>
<h1>MIT steering sine · {result['frequency_fit_hz']:.0f} Hz / ±1 rad / Kp {result['labels']['kp']:g}, Kd {result['labels']['kd']:g}</h1>
<p>Recorded encoded target, A3 position, A2 velocity and A1 current-derived torque. All panels share the unchanged recorder clock.</p>
<p class="warning">CF shutdown disabled-mode confirmation failed (reply mode 5). Torque is current-derived. Phase-equivalent lag is not pure delay.</p>
{table([result])}<p>Steady window {lo:.3f}–{hi:.3f} s; target {p['traces'][0]['mean_rate_hz']:.3f} Hz; A3 {p['traces'][1]['mean_rate_hz']:.3f} Hz.</p>
{buttons}{''.join(sections)}<h2>Three-cycle overlay</h2><img src="tracking_overlay.png" alt="Unshifted target and position feedback">
<p>{html.escape(CAUTIONS)}</p><p><a href="README.md">Methods, findings and limitations</a> · <a href="analysis.json">Analysis</a> ·
<a href="summary.json">Provenance</a> · <a href="steering_sine_report.png">Full PNG</a> · <a href="steering_sine_zoom.png">Zoom PNG</a> ·
<a href="steering_sine_report.pdf">PDF</a> · <a href="{html.escape(os.path.relpath(result['source'], output))}">Source CSV</a></p>
<script>document.querySelectorAll('button[data-view]').forEach(b=>b.addEventListener('click',()=>{{
document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));document.getElementById('view-'+b.dataset.view).classList.add('active');
document.querySelectorAll('button[data-view]').forEach(o=>o.setAttribute('aria-pressed',String(o===b)));}}));</script></body></html>'''
    (output/"steering_sine_report.html").write_text(page)


def comparison_plots(results, c2, traces_by_source, output):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fields = (("gain", "Fundamental gain (%)"), ("phase_lag_deg", "Phase lag (degrees)"),
              ("phase_equivalent_lag_ms", "Phase-equivalent lag (ms)"), ("error_rms_rad", "Held-target RMS error (rad)"),
              ("position_r_squared", "Position sinusoidal fit R²"), ("peak_torque", "Peak current-derived torque (Nm)"))
    for selected, name in [(c2, "C2 / ~197 Hz")] + [([r for r in results if r["labels"]["kd"] == kd], f"MIT Kd={kd:g} / ~50 Hz")
                                                    for kd in sorted({r["labels"]["kd"] for r in results})]:
        for ax, (field, label) in zip(axes.flat, fields):
            values = [r["steady"]["fits"]["position"]["r_squared"] if field == "position_r_squared"
                      else r["steady"]["peaks"]["torque"]["abs_max"] if field == "peak_torque"
                      else r["steady"][field]*(100 if field == "gain" else 1) for r in selected]
            ax.plot([r["frequency_fit_hz"] for r in selected], values, "o-", label=name)
            ax.set(xlabel="Sine frequency (Hz)", ylabel=label, xticks=[1, 2, 5, 10])
            ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].axhline(100, color="grey", ls="--", lw=.8)
    axes[1, 2].axhline(5.6, color="red", ls="--", lw=.8, label="Training reference 5.6 Nm")
    axes[1, 2].legend(fontsize=8)
    fig.suptitle("Free-steering ±1 rad: MIT versus recorded C2 (different target rates; no C3 measurement)")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output/f"mit_c2_comparison.{ext}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    for row_index, frequency in enumerate((1, 2, 5, 10)):
        for column, group in enumerate((c2, [r for r in results if r["labels"]["kd"] == .2],
                                       [r for r in results if r["labels"]["kd"] == .1])):
            ax = axes[row_index, column]
            run = next((r for r in group if r["labels"]["frequency_hz"] == frequency), None)
            if run is None:
                ax.text(.5, .5, "No recorded C2 1 Hz / ±1 rad test", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            lo, hi = run["steady"]["window_s"]
            center, width = (lo+hi)/2, 3/frequency
            for t, color, label, step in zip(traces_by_source[run["source"]][:2], ("#2563eb", "#087f73"),
                                            ("Encoded held target", "A3 position"), (True, False)):
                selected = t[(t[:, 0] >= center-width/2-.025) & (t[:, 0] <= center+width/2+.025)].copy()
                selected[:, 0] -= center
                plot_line(ax, selected, color, label, step)
            mode = "C2" if column == 0 else f"MIT Kd={run['labels']['kd']:g}"
            ax.set(xlim=(-width/2, width/2), ylim=(-1.2, 1.2), ylabel="Position (rad)",
                   title=f"{mode} · {frequency} Hz · gain {100*run['steady']['gain']:.1f}%")
            ax.grid(alpha=.25)
            if row_index == 3:
                ax.set_xlabel("Relative window time (s); signals unshifted")
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("Three-cycle raw tracking windows; separate experiments, not synchronized trials")
    fig.tight_layout()
    fig.savefig(output/"tracking_comparison.png", dpi=160)
    fig.savefig(output/"tracking_comparison.pdf")
    plt.close(fig)


def batch(logs: Path, reports: Path, c2_catalog: Path):
    sources = sorted(logs.glob("*_mit_*.csv"))
    if not sources:
        raise ValueError("No labeled MIT CSVs")
    aggregate = reports/"mit_sine_sweep_20261007"
    targets = [reports/labels(p)["report_name"] for p in sources] + [aggregate]
    if any(p.exists() for p in targets):
        raise FileExistsError("Refusing to overwrite an existing report; use --reports with a fresh directory")
    existing = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in reports.rglob("*") if p.is_file()}
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in logs.glob("*.csv")}
    catalog = json.loads(c2_catalog.read_text())
    c2, traces_by_source = [], {}
    for saved in catalog["results"]:
        if saved["labels"]["amplitude_rad"] != 1:
            continue
        source = logs/Path(saved["source"]).name
        result, traces = analyze_c2(source)
        if result["source_sha256"] != saved["source_sha256"]:
            raise ValueError("C2 baseline recording changed")
        for field in ("gain", "phase_lag_deg", "error_rms_rad"):
            if not math.isclose(result["steady"][field], saved["steady"][field], abs_tol=1e-8):
                raise ValueError("Recomputed C2 baseline differs from saved report")
        result["mode"] = "C2"
        c2.append(result)
        traces_by_source[result["source"]] = traces
    if sorted(r["labels"]["frequency_hz"] for r in c2) != [2, 5, 10]:
        raise ValueError("Expected the three ±1 rad C2 baselines")
    results = []
    for source in sources:
        result, traces = analyze(source)
        result["report_directory"] = str((reports/result["labels"]["report_name"]).resolve())
        results.append(result)
        traces_by_source[result["source"]] = traces
        print(f'{source.name}: gain {result["steady"]["gain"]*100:.2f}%, phase {result["steady"]["phase_lag_deg"]:.2f}°, RMS {result["steady"]["error_rms_rad"]:.4f} rad', flush=True)
    results.sort(key=lambda r: (r["labels"]["frequency_hz"], r["labels"]["kd"]))
    for result in results:
        write_run(result, traces_by_source[result["source"]], Path(result["report_directory"]))
    aggregate.mkdir(parents=True, exist_ok=False)
    comparison_plots(results, c2, traces_by_source, aggregate)
    dataset = dict(mit_results=results, c2_baselines=c2, method=METHOD, limitations=CAUTIONS,
                   c2_catalog_sha256=hashlib.sha256(c2_catalog.read_bytes()).hexdigest())
    (aggregate/"comparison.json").write_text(json.dumps(dataset, indent=2, allow_nan=False)+"\n")
    flat = [dict(mode=r.get("mode", "MIT"), frequency_hz=r["frequency_fit_hz"], amplitude_rad=1,
                 kp=r["labels"].get("kp"), kd=r["labels"].get("kd"),
                 position_amplitude_rad=r["steady"]["fits"]["position"]["amplitude"], gain=r["steady"]["gain"],
                 phase_lag_deg=r["steady"]["phase_lag_deg"], lag_ms=r["steady"]["phase_equivalent_lag_ms"],
                 error_rms_rad=r["steady"]["error_rms_rad"], position_r_squared=r["steady"]["fits"]["position"]["r_squared"],
                 peak_speed_rad_s=r["steady"]["peaks"]["velocity"]["abs_max"],
                 peak_torque_nm=r["steady"]["peaks"]["torque"]["abs_max"],
                 target_rate_hz=r["provenance"]["traces"][0]["mean_rate_hz"] if "encoded_settings" in r else r["timing"]["C2"]["mean_rate_hz"],
                 source=r["source"], sha256=r["source_sha256"]) for r in results+c2]
    with (aggregate/"comparison.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    links = "\n".join(f'- [{r["labels"]["frequency_hz"]:g} Hz / Kp 10 / Kd {r["labels"]["kd"]:g}]({os.path.relpath(r["report_directory"], aggregate)}/steering_sine_report.html)' for r in results)
    narrative = f'''# MIT versus C2 free-steering sine sweep (2026-10-07)

## Results

All eight new MIT recordings have verified ±1 rad sine requests, Kp≈10, and encoded Kd≈0.1 or 0.2.
Front wheel/steering fully unconstrained, per user. Recomputed all three comparable ±1 rad C2 baselines from original CSVs;
their hashes and statistics match the previous sweep. No measured C3 baseline was identified.

{table(results+c2, True)}

Gain uses the **held encoded command's fundamental**, not the ±1 rad ideal continuous wave.
At 10 Hz / ~50 Hz MIT setpoints, this fundamental is approximately 0.935 rad; A3 fundamental is only 0.36–0.37 rad.
No 1 Hz C2 recording exists here. Kd is not a transferable C2 PI register.

## Conclusions supported by the recordings

1. **MIT substantially improves low-frequency tracking over recorded C2.** At 2 Hz, MIT gains are 99.2–101.2% versus C2 66.2%;
   phase 8.7–15.4° versus 48.6°; RMS error 0.120–0.196 rad versus 0.531 rad (approximately 63–77% reduction).
   At 1 Hz, MIT achieves approximately unity gain and 0.059–0.099 rad RMS error; no measured C2 comparison at that frequency.
2. **At 5 Hz MIT has higher amplitude, but tradeoffs remain.** Gain 79.2–81.8% versus C2 33.7%.
   Kd=.2 gives phase 55.1° / RMS 0.606 rad versus C2 71.0° / 0.669 rad.
   Kd=.1 gives phase 62.7° / RMS 0.681 rad: higher amplitude does not automatically mean lower total tracking error.
3. **At 10 Hz MIT is not a tracking fix.** Gain 38.5–39.2% versus C2 17.9%, but MIT phase is 104.5–111.6° versus C2 81.2°,
   RMS error 0.812–0.846 rad versus C2 0.700 rad, and A3 sine-fit R² only 0.770–0.855.
   Thus higher fundamental amplitude coexists with worse overall error and substantial nonsinusoidal behavior.
   The finite traces do not prove an autonomous instability or diagnose its exact mechanism.
4. **High-frequency response is consistent with a speed plateau, not setpoint starvation.** 5/10 Hz demand ideal peaks of
   31.4/62.8 rad/s, while A2 peaks remain approximately 20.6 rad/s. At 5 Hz, |speed|>20 rad/s for 46–61% of samples.
   All eight MIT target streams run 49.718–49.728 Hz, maximum logged lateness below 0.336 ms, no skipped command ticks;
   A3/A2/A1 receipt streams approximately 196.8–196.9 Hz, no >50 ms gaps, query RTT p99 below 0.89 ms.
   This makes host scheduling stalls a weak explanation for these particular free-wheel results. It does not identify the
   active speed/current limit, voltage/back-EMF mechanism, firmware sampling frequency, or ground-loaded behavior.
5. **Torque limits are not matched to training.** Steady A1-derived peaks at 10 Hz are 9.73–9.85 Nm, above the training 5.6 Nm reference;
   roughly 7.8–8.7% of samples exceed 5.6 Nm. Kd=.1 at 5 Hz reaches 6.36 Nm (2.1% of samples above 5.6), versus .2 at 5 Hz 5.40 Nm.
   F0 T_Max=18 Nm is encoding metadata, not proof of a safe total-PD clamp. These are current/Kt estimates, not calibrated shaft torque
   or proof of mechanical damage. Read back/calibrate the actual firmware current/torque protection before balancing.
6. **Lower damping is not uniformly better.** Kd=.1 reduces low-frequency lag to approximately 12 ms versus .2 approximately 21 ms,
   but has worse 5/10 Hz RMS error, larger 5 Hz torque and slightly more than unity 1/2 Hz gain.
   The default .2 matches the training's requested damping; do not silently switch the deployment based on one unloaded trial per condition.
   These are frequency-specific phase-equivalent times, not pure transport/motor delays or per-target settling times.
7. **Shutdown verification is unresolved in all eight tests.** Each log ends on KeyboardInterrupt, sends CF, receives CF byte 6=5/fault=0,
   and records `CF sent but disabled mode not confirmed before shutdown deadline`. Startup CF had mode=0.
   Protocol AE mode table defines 0 as disabled but does not define 5; do not invent its semantics or infer permanent enable.
   With independent physical support and a means of cutting power, record subsequent AE/F1 and confirm free state after CF;
   resolve firmware reply/transition semantics before trusting the deployment's acknowledged-disable transitions.

## Scope and next experiment

This supports **MIT as a better low-frequency actuator interface than the recorded C2 settings**, not a repaired/damaged-motor verdict,
universal MIT superiority, or proof that a policy will balance. The 50 Hz host interface can clearly track 1–2 Hz physical motion here;
it does not establish the firmware loop rate or the adequacy of the previous external 200 Hz C0 PD loop.
The absolute MIT target uses fixed F1 zero + signed sine offset; A3 uses its independent fixed encoder zero.
Wire re-decoding and F1 gain/phase cross-checks support the conversion; do not attribute attenuation to an invented sign/unit fault.

Next: first resolve disabled-state verification and firmware total-torque limits. Then repeat **matched 50 Hz C2/MIT** tests at ±0.1/0.2 rad,
0.5/1/2/5 Hz, with gains/limits, voltage, temperature and status recorded, followed by independently supported **ground-loaded** steering.
Small amplitudes avoid the present high-speed demands and better resemble balancing corrections. Inspect actual policy command spectrum,
model the measured tracking/limits in simulation, and only then decide whether gain tuning or policy adaptation is needed.
Changing to C3 is not an evidenced cure: it changes reference semantics, and its performance was not measured by these CSVs.

## Methods, integrity and artifacts

{METHOD}. Three subwindows record variability, not independent-repeat confidence intervals.
Each applicable MIT target/A3/A2/A1/F1 payload was re-decoded; sparse rows use **their own elapsed_s**, never cached/imputed measurements.
Independent held-input correlation agrees within 0.5 ms with A3 phase-equivalent lag in these recordings;
F1 fitted gain/phase agrees closely, but shares the motor and receipt-clock limitations, so is not independent physical calibration.
Query timeouts/late replies are retained and counted, not silently called successful or synthesized into missing rows.
F1 reports active and fault-free during all recorded motion; socket overflow metadata reports zero dropped frames.
Fault-free F1 is not evidence of compliance with the 5.6 Nm training cap. No final-hold/step-settling statistics are invented after CF.

{CAUTIONS}

- [Offline HTML index](index.html)
- [Comparison PNG](mit_c2_comparison.png) / [PDF](mit_c2_comparison.pdf) / [SVG](mit_c2_comparison.svg)
- [Three-cycle tracking grid](tracking_comparison.png) / [PDF](tracking_comparison.pdf)
- [Comparison CSV](comparison.csv) / [complete JSON with source SHA-256s](comparison.json)

{links}

Each MIT condition has the prior workflow's five selectable four-panel SVG views, full PNG/PDF, zoom PNG,
tracking overlay, README, summary.json and analysis.json. All earlier reports and raw CSVs are preserved.
No runtime, policy, firmware setting, simulation or progress record is modified; no hardware command is sent by this analysis.

Reproduce in a fresh directory: `MPLCONFIGDIR=/tmp/trackstand_mit_mpl PYTHONDONTWRITEBYTECODE=1 python3 deploy/experiments/analyze_mit_sine.py --reports /tmp/new_mit_reports`.
'''
    (aggregate/"README.md").write_text(narrative)
    cards = "".join(f'<li><a href="{html.escape(os.path.relpath(r["report_directory"], aggregate))}/steering_sine_report.html">'
                    f'{r["labels"]["frequency_hz"]:g} Hz / Kd {r["labels"]["kd"]:g}</a></li>' for r in results)
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MIT versus C2 steering sweep</title><style>{STYLE}</style></head><body><h1>MIT versus C2 · free-steering ±1 rad sweep</h1>
<p>Eight new MIT recordings; three reverified C2 baselines. MIT Kp≈10, Kd≈0.1/.2. No measured C3 baseline.</p>
<p class="warning">MIT improves low-frequency tracking, but 10 Hz tracking remains poor with higher torque. All eight CF shutdowns lack disabled-mode confirmation.</p>
{table(results+c2)}<img src="mit_c2_comparison.png" alt="Gain, phase, lag, RMS error, fit quality and torque">
<img src="tracking_comparison.png" alt="Unshifted three-cycle tracking windows">
<h2>Individual four-panel reports</h2><ul>{cards}</ul><p>{html.escape(CAUTIONS)}</p>
<p><a href="README.md">Full conclusions, evidence and next tests</a> · <a href="comparison.csv">CSV table</a> ·
<a href="comparison.json">Complete JSON/provenance</a> · <a href="mit_c2_comparison.pdf">Comparison PDF</a> ·
<a href="tracking_comparison.pdf">Tracking PDF</a></p></body></html>'''
    (aggregate/"index.html").write_text(page)
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest for p, digest in hashes.items()):
        raise RuntimeError("Input recording changed during analysis")
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest for p, digest in existing.items()):
        raise RuntimeError("Pre-existing report changed during analysis")
    print(f"Reports: {aggregate/'index.html'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, default=ROOT/"logs")
    parser.add_argument("--reports", type=Path, default=ROOT/"reports")
    parser.add_argument("--c2-catalog", type=Path, default=ROOT/"reports/c2_sine_sweep_20261004/comparison.json")
    args = parser.parse_args()
    plt.rcParams.update({"svg.fonttype": "none", "path.simplify": False, "font.family": "DejaVu Sans"})
    batch(args.logs.resolve(), args.reports.resolve(), args.c2_catalog.resolve())
