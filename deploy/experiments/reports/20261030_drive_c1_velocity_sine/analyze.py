"""Analyze the five October 30 C1 velocity-sine runs; never open ROS/CAN.

Only this report directory is written. CSV/JSON and source files stay immutable.
All plot channels share the recorded phase-start monotonic clock. Fresh A2/A1
replies, not cached command-row state, provide velocity and torque. Steady metrics
use [2,10) s; exact held-input harmonic coefficients avoid treating 200-Hz packets
as a continuous target. Phase-equivalent lag is not identified motor-internal delay.
"""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
from html import escape
import json
import math
from pathlib import Path
import struct

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LOGS = ROOT / "deploy/experiments/logs"
SOURCE = HERE / "sender_source_snapshot.py.txt"
LIVE_SOURCE = ROOT / "deploy/experiments/drive_velocity_sine.py"
FREQUENCIES = (.5, 1., 2., 5., 10.)
FILES = (
    "20261030_103805_187108_drive_velocity_sine.csv",
    "20261030_104110_521233_drive_velocity_sine.csv",
    "20261030_104154_932459_drive_velocity_sine.csv",
    "20261030_104208_835736_drive_velocity_sine.csv",
    "20261030_104226_508475_drive_velocity_sine.csv",
)
FIXTURE = ("User confirmed rear wheel airborne/free, no hand contact, "
           "same power supply and firmware conditions for all five runs.")
ORANGE, GREEN, PURPLE = "#d27818", "#16805d", "#7244a0"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(row, name):
    value = row.get(name, "")
    return float(value) if value else math.nan


def save_csv(name, rows):
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stats(values):
    x = np.asarray(values)
    x = x[np.isfinite(x)]
    return dict(n=len(x), min=float(x.min()), median=float(np.median(x)), mean=float(x.mean()),
                p95=float(np.percentile(x, 95)), p99=float(np.percentile(x, 99)), max=float(x.max()))


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values) ** 2)))


def held(t, y, query):
    index = np.searchsorted(t, query, side="right") - 1
    result = np.full(np.shape(query), np.nan)
    valid = index >= 0
    result[valid] = y[index[valid]]
    return result


def harmonic(t, y, frequency):
    omega = 2 * math.pi * frequency
    design = np.column_stack((np.sin(omega * t), np.cos(omega * t), np.ones(len(t))))
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    return dict(amplitude=float(np.hypot(*coefficients[:2])), phase_rad=float(np.arctan2(
        coefficients[1], coefficients[0])), bias=float(coefficients[2]), residual_rms=rms(y - design @ coefficients))


def held_harmonic(t, y, frequency, lower=2., upper=10.):
    """Exact integrals of the actual piecewise-constant encoded C1 target.

    The shared eight-second window contains an integer number of all test
    periods, so sin/cos/DC are orthogonal without resampling or interpolation.
    """
    omega, duration = 2 * math.pi * frequency, upper - lower
    assert math.isclose(duration * frequency, round(duration * frequency), abs_tol=1e-10)
    ends = np.r_[t[1:], upper]
    starts, ends = np.maximum(t, lower), np.minimum(ends, upper)
    valid = ends > starts
    a, b, values = starts[valid], ends[valid], y[valid]
    sine = 2 / duration * np.sum(values * (np.cos(omega * a) - np.cos(omega * b)) / omega)
    cosine = 2 / duration * np.sum(values * (np.sin(omega * b) - np.sin(omega * a)) / omega)
    return dict(amplitude=float(np.hypot(sine, cosine)), phase_rad=float(np.arctan2(cosine, sine)),
                bias=float(np.sum(values * (b - a)) / duration))


def phase_delay_ms(output, input_fit, frequency):
    difference = (output["phase_rad"] - input_fit["phase_rad"] + math.pi) % (2 * math.pi) - math.pi
    return float(-difference / (2 * math.pi * frequency) * 1000)


def load_run(path):
    metadata_path = path.with_suffix(".json")
    meta = json.loads(metadata_path.read_text())
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    args = meta["arguments"]
    assert args["method"] == "c1" and args["amplitude"] == 4.
    assert meta["completed"] and meta["disabled_and_restored"] and meta["error"] is None
    assert meta["motor_id"] == 1 and meta["encoder_sign"] == meta["torque_sign"] == -1.
    assert meta["source_sha256"] == digest(SOURCE)
    frequency, kt = args["frequency"], meta["torque_constant_nm_per_a"]
    commands = [r for r in rows if r["record_type"] == "command"]
    origins = np.array([number(r, "host_monotonic_s") - number(r, "phase_elapsed_s") for r in commands])
    origin = float(np.median(origins))
    assert np.ptp(origins) < 1e-8
    assert len(commands) == meta["phases"][0]["commands"]
    assert [int(r["command_index"]) for r in commands] == list(range(len(commands)))
    assert sum(int(r["skipped_slots"]) for r in commands) == meta["phases"][0]["skipped_slots"]
    decoded_checks = Counter()
    for row in rows:
        kind, op = row["record_type"], row["opcode"]
        if not row["payload_hex"]:
            continue
        data = bytes.fromhex(row["payload_hex"])
        cid = int(row["can_id"], 16)
        if kind in ("tx", "command"):
            assert cid == 0x101 and int(op, 16) == data[0]
            if data[0] in (0xB1, 0xB3, 0xB5, 0xB8, 0xB9, 0xF0):
                assert len(data) == 1, "unexpected firmware gain/limit/origin write"
        elif kind == "rx":
            assert cid == 1 and int(op, 16) == data[0]
        if kind == "command":
            assert len(data) == 5 and data[0] == 0xC1
            decoded = -struct.unpack("<i", data[1:])[0] * .01 * 2 * math.pi / 60
            assert abs(decoded - number(row, "encoded_target_velocity_rad_s")) < 1e-12
            desired = args["amplitude"] * math.sin(2 * math.pi * frequency * number(row, "target_sample_time_s"))
            assert abs(desired - number(row, "target_velocity_rad_s")) < 1e-10
            assert abs(decoded - desired) <= .005 * 2 * math.pi / 60 + 1e-10
            decoded_checks["command"] += 1
        elif kind == "rx" and op in ("A2", "C1"):
            decoded = -struct.unpack("<i", data[1:])[0] * .01 * 2 * math.pi / 60
            assert abs(decoded - number(row, "velocity_rad_s")) < 1e-12
            decoded_checks[op] += 1
        elif kind == "rx" and op == "A1":
            current = struct.unpack("<i", data[1:])[0] * .001
            assert abs(current - number(row, "current_a")) < 1e-12
            assert abs(-current * kt - number(row, "current_torque_nm")) < 1e-12
            decoded_checks[op] += 1
        elif kind == "rx" and op == "B0":
            assert struct.unpack("<f", data[2:6])[0] == kt
        elif kind == "rx" and op in ("AE", "CF"):
            assert data[7] == 0
    assert decoded_checks["C1"] == decoded_checks["command"]

    def fresh(opcode, field):
        chosen = [row for row in rows if row["record_type"] == "rx" and row["opcode"] == opcode
                  and 0 <= number(row, "host_monotonic_s") - origin < 10]
        result = np.array([(number(row, "host_monotonic_s") - origin, number(row, field)) for row in chosen])
        assert len(result) and np.all(np.isfinite(result)) and np.all(np.diff(result[:, 0]) > 0)
        return result

    command = np.array([(number(row, "host_monotonic_s") - origin,
                         number(row, "encoded_target_velocity_rad_s")) for row in commands])
    velocity, torque = fresh("A2", "velocity_rad_s"), fresh("A1", "current_torque_nm")
    alternate = fresh("C1", "velocity_rad_s")
    mask = velocity[:, 0] >= 2.
    feedback, input_fit = harmonic(*velocity[mask].T, frequency), held_harmonic(*command.T, frequency)
    error = velocity[mask, 1] - held(*command.T, velocity[mask, 0])
    assert np.all(np.isfinite(error))
    alt_fit = harmonic(*alternate[alternate[:, 0] >= 2].T, frequency)
    ages = [number(row, "velocity_age_s") * 1000 for row in commands]
    interval_ms = np.diff(command[:, 0]) * 1000
    ae = [r for r in rows if r["record_type"] == "rx" and r["opcode"] == "AE"
          and 0 <= number(r, "host_monotonic_s") - origin < 10]
    torque_mask = torque[:, 0] >= 2.
    metrics = dict(frequency_hz=frequency, amplitude_requested_rad_s=4., source_csv=path.name,
                   phase_origin_monotonic_s=origin, duration_s=meta["phases"][0]["duration_s"],
                   command_samples=len(command), a2_samples=len(velocity), a1_samples=len(torque),
                   commands_skipped=meta["phases"][0]["skipped_slots"],
                   achieved_command_rate_hz=meta["phases"][0]["achieved_rate_hz"],
                   held_target_fundamental_amplitude_rad_s=input_fit["amplitude"],
                   measured_fundamental_amplitude_rad_s=feedback["amplitude"],
                   amplitude_ratio_percent=100 * feedback["amplitude"] / input_fit["amplitude"],
                   phase_equivalent_lag_from_held_target_ms=phase_delay_ms(feedback, input_fit, frequency),
                   phase_equivalent_lag_from_ideal_sine_ms=-feedback["phase_rad"] / (2 * math.pi * frequency) * 1000,
                   steady_error_rms_rad_s=rms(error), steady_error_mae_rad_s=float(np.mean(abs(error))),
                   steady_error_p95_abs_rad_s=float(np.percentile(abs(error), 95)),
                   steady_error_peak_abs_rad_s=float(np.max(abs(error))),
                   steady_fraction_within_0p5_rad_s=float(np.mean(abs(error) <= .5)),
                   steady_mean_velocity_bias_rad_s=feedback["bias"],
                   steady_velocity_nonfundamental_rms_rad_s=feedback["residual_rms"],
                   measured_peak_positive_rad_s=float(velocity[:, 1].max()),
                   measured_peak_negative_rad_s=float(velocity[:, 1].min()),
                   measured_peak_over_amplitude_percent=(float(np.max(abs(velocity[:, 1]))) / 4 - 1) * 100,
                   peak_abs_current_torque_nm=float(np.max(abs(torque[:, 1]))),
                   current_torque_rms_nm=rms(torque[:, 1]),
                   steady_current_torque_rms_nm=rms(torque[torque_mask, 1]),
                   alternate_c1_reply_amplitude_rad_s=alt_fit["amplitude"],
                   alternate_c1_reply_phase_lag_ms=phase_delay_ms(alt_fit, input_fit, frequency),
                   raw_kp=meta["raw_gains"]["B8"], raw_ki=meta["raw_gains"]["B9"],
                   torque_constant_nm_per_a=kt, current_ceiling_a=struct.unpack(
                       "<I", bytes.fromhex(meta["original_settings"]["B3"])[1:])[0] * .001,
                   b5_acceleration_raw=struct.unpack("<I", bytes.fromhex(meta["original_settings"]["B5"])[1:])[0],
                   bus_voltage_min_v=min(number(r, "bus_voltage_v") for r in ae),
                   bus_voltage_max_v=max(number(r, "bus_voltage_v") for r in ae),
                   max_temperature_c=max(number(r, "temperature_c") for r in ae))
    detail = dict(source_sha256=meta["source_sha256"], csv_sha256=digest(path), json_sha256=digest(metadata_path),
                  decoded_checks=dict(decoded_checks), metadata=meta,
                  packet_interval_ms=stats(interval_ms), velocity_receipt_age_at_command_ms=stats(ages),
                  a2_interval_ms=stats(np.diff(velocity[:, 0]) * 1000),
                  a1_interval_ms=stats(np.diff(torque[:, 0]) * 1000), input_harmonic=input_fit,
                  output_harmonic=feedback, metrics=metrics)
    return dict(command=command, velocity=velocity, torque=torque, alternate=alternate,
                metrics=metrics, detail=detail)


def break_gaps(series):
    indices = np.flatnonzero(np.diff(series[:, 0]) > .05) + 1
    return np.insert(series, indices, np.column_stack((series[indices, 0], np.full(len(indices), np.nan))), axis=0)


def draw(run, zoom=False):
    m = run["metrics"]
    f = m["frequency_hz"]
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, constrained_layout=True)
    command, velocity, torque = (break_gaps(run[key]) for key in ("command", "velocity", "torque"))
    axes[0].step(*command.T, where="post", color=ORANGE, lw=1.15, label="Transmitted C1 velocity target")
    axes[0].plot(*velocity.T, color=GREEN, lw=.95, label="Fresh A2 measured velocity")
    axes[1].plot(*torque.T, color=PURPLE, lw=.95, label="Fresh A1 Q-axis current × B0 Kt")
    axes[0].set_ylim(-5.5, 5.5)
    axes[1].set_ylim(-.55, .55)
    axes[0].yaxis.set_major_locator(MultipleLocator(2))
    axes[1].yaxis.set_major_locator(MultipleLocator(.2))
    axes[0].set_ylabel("Rear velocity (rad/s)")
    axes[1].set_ylabel("Current-derived torque (Nm)")
    axes[1].set_xlabel("Seconds from common phase start — original host monotonic send/receipt times")
    limits = (max(0., 5 - 1.5 / f), min(10., 5 + 1.5 / f)) if zoom else (0., 10.)
    axes[1].set_xlim(*limits)
    for i, (ax, traces, unit) in enumerate(((axes[0], (run["command"], run["velocity"]), "rad/s"),
                                           (axes[1], (run["torque"],), "Nm"))):
        finite = np.concatenate([
            trace[(trace[:, 0] >= limits[0]) & (trace[:, 0] <= limits[1]), 1] for trace in traces
        ])
        ax.text(.008, .985, f"Shown min {finite.min():.3f}, max {finite.max():.3f} {unit}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
        ax.axhline(0, color="#777777", lw=.6)
        ax.grid(alpha=.22)
        ax.legend(loc="lower right", fontsize=9, framealpha=.92)
    view = "three-cycle zoom" if zoom else "full 10 s"
    title = f"Rear C1 velocity tracking | {f:g} Hz sine, amplitude ±4 rad/s | {view}"
    fig.suptitle(title + "\n" + f"{m['source_csv']} | rear airborne/free | target packets ≈200 Hz\n"
                 "Same time origin for both panels; no response shift or smoothing; torque is not shaft/net torque",
                 fontsize=12)
    assert np.max(abs(run["velocity"][:, 1])) < 5.5
    assert np.max(abs(run["torque"][:, 1])) < .55
    return fig


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    paths = [LOGS / filename for filename in FILES]
    protected_paths = paths + [p.with_suffix(".json") for p in paths] + [SOURCE, LIVE_SOURCE]
    protected = {str(p.relative_to(ROOT)): digest(p) for p in protected_paths}
    runs = sorted([load_run(path) for path in paths], key=lambda run: run["metrics"]["frequency_hz"])
    assert tuple(run["metrics"]["frequency_hz"] for run in runs) == FREQUENCIES
    assert len({json.dumps(run["detail"]["metadata"]["original_settings"], sort_keys=True) for run in runs}) == 1
    save_json("statistics.json", dict(fixture=FIXTURE, metric_window_s=[2., 10.],
                                      time_base="Common monotonic phase origin; no independently re-zeroed channels",
                                      lag_definition="Fundamental phase-equivalent lag vs actual held encoded input; "
                                                     "not motor-internal latency",
                                      runs=[run["detail"] for run in runs]))
    save_csv("summary.csv", [run["metrics"] for run in runs])
    save_json("protected_input_hashes.json", protected)
    plt.rcParams.update({"font.size": 10, "savefig.dpi": 180, "pdf.fonttype": 42, "svg.fonttype": "none"})
    gallery, sample_count = [], 0
    order = (1., .5, 2., 5., 10.)
    with PdfPages(HERE / "all_runs.pdf") as full_pdf, PdfPages(HERE / "all_run_zooms.pdf") as zoom_pdf:
        for frequency in order:
            run = next(run for run in runs if run["metrics"]["frequency_hz"] == frequency)
            prefix = "c1_" + f"{frequency:g}".replace(".", "p") + "hz"
            rows = []
            for key, unit in (("command", "rad/s"), ("velocity", "rad/s"), ("torque", "Nm")):
                rows.extend(dict(series=key, phase_time_s=float(t), value=float(value), unit=unit)
                            for t, value in run[key])
            save_csv(prefix + "_samples.csv", rows)
            sample_count += len(rows)
            for zoom in (False, True):
                fig = draw(run, zoom)
                stem = prefix + ("_zoom" if zoom else "_target_response")
                for extension in ("png", "svg", "pdf"):
                    fig.savefig(HERE / f"{stem}.{extension}", bbox_inches="tight")
                (zoom_pdf if zoom else full_pdf).savefig(fig)
                plt.close(fig)
            gallery.append((frequency, prefix))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    f = [run["metrics"]["frequency_hz"] for run in runs]
    specs = (("amplitude_ratio_percent", "Fundamental amplitude / held input (%)"),
             ("phase_equivalent_lag_from_held_target_ms", "Phase-equivalent lag vs held target (ms)"),
             ("steady_error_rms_rad_s", "Unshifted tracking RMS error (rad/s)"),
             ("current_torque_rms_nm", "Current-derived torque RMS (Nm)"))
    for ax, (field, label) in zip(axes.flat, specs):
        values = [run["metrics"][field] for run in runs]
        ax.plot(f, values, "o-", color=GREEN, lw=1.4)
        for freq, value in zip(f, values):
            ax.annotate(f"{value:.2f}", (freq, value), xytext=(0, 7), textcoords="offset points", ha="center")
        ax.set(xscale="log", xticks=f, xlabel="Sine frequency (Hz)", ylabel=label)
        ax.set_xticklabels([f"{freq:g}" for freq in f])
        ax.margins(y=.22)
        ax.grid(alpha=.2)
    axes[0, 0].axhline(100, color="#999999", ls="--", lw=.8)
    fig.suptitle("Rear C1 sine response, same ±4 rad/s / airborne conditions\n"
                 "Tracking metrics [2,10) s; torque RMS full phase; one run per frequency, no MIT data")
    for extension in ("png", "svg", "pdf"):
        fig.savefig(HERE / ("frequency_summary." + extension), bbox_inches="tight")
    plt.close(fig)

    table_rows = "".join(f"<tr><td>{run['metrics']['frequency_hz']:g}</td>"
                         f"<td>{run['metrics']['amplitude_ratio_percent']:.2f}</td>"
                         f"<td>{run['metrics']['phase_equivalent_lag_from_held_target_ms']:.2f}</td>"
                         f"<td>{run['metrics']['steady_error_rms_rad_s']:.3f}</td>"
                         f"<td>{run['metrics']['peak_abs_current_torque_nm']:.3f}</td></tr>" for run in runs)
    sections = []
    for frequency, prefix in gallery:
        sections.append(f'<section><h2>{frequency:g} Hz</h2><p>'
                        f'<a href="{prefix}_target_response.png">PNG</a> · '
                        f'<a href="{prefix}_target_response.svg">SVG</a> · '
                        f'<a href="{prefix}_target_response.pdf">PDF</a> · '
                        f'<a href="{prefix}_zoom.png">Three-cycle zoom</a></p>'
                        f'<img loading="lazy" src="{prefix}_target_response.png" alt="{frequency:g} Hz full run">'
                        f'<details><summary>Show three-cycle zoom</summary><img loading="lazy" '
                        f'src="{prefix}_zoom.png" alt="{frequency:g} Hz zoom"></details></section>')
    (HERE / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>Rear C1 velocity sine — 20261030</title><style>body{max-width:1500px;margin:2rem auto;'
        'padding:0 1rem;font:16px system-ui;color:#16283b;background:#f6f8fb}img{max-width:100%}'
        'section{background:white;padding:1rem;margin:2rem 0}td,th{padding:.5rem;border:1px solid #ddd}'
        'table{border-collapse:collapse}a{color:#225ab0}</style><h1>Rear C1 velocity sine — five frequencies</h1>'
        f'<p>{escape(FIXTURE)} All amplitudes ±4 rad/s. Fresh A2 and A1 samples share one phase time origin.</p>'
        '<p><a href="README.md">Analysis report</a> · <a href="summary.csv">Metrics CSV</a> · '
        '<a href="all_runs.pdf">Five-page PDF</a> · <a href="all_run_zooms.pdf">Zoom PDF</a></p>'
        '<table><tr><th>Hz</th><th>Amplitude (%)</th><th>Phase-equivalent lag (ms)</th>'
        '<th>RMS error (rad/s)</th><th>Peak torque estimate (Nm)</th></tr>' + table_rows + '</table>'
        '<p>Lag is phase-equivalent against the held transmitted target, not identified motor-internal delay. '
        'Torque is current × Kt, not independently measured shaft/net torque.</p>'
        '<img src="frequency_summary.png" alt="Frequency response summary">' + ''.join(sections) + '</html>')
    assert all(digest(ROOT / name) == value for name, value in protected.items())
    save_json("verification.json", dict(passed=True, input_files_hash_checked=len(protected),
                                         input_files_unchanged=True, frequencies_hz=list(FREQUENCIES),
                                         raw_packet_field_checks=True, sender_snapshot_hash_matches_all_runs=True,
                                         methods=["c1"], common_phase_time_origin=True,
                                         cached_feedback_duplicates_excluded=True, plotted_native_samples=sample_count,
                                         full_figures=5, zoom_figures=5, summary_figures=1,
                                         all_finite_values_within_plot_axes=True, no_smoothing_or_time_shift=True,
                                         firmware_settings_identical=True, user_fixture=FIXTURE,
                                         original_logs_runtime_sources_preserved=True))
    print(json.dumps([run["metrics"] for run in runs], indent=2))


if __name__ == "__main__":
    main()
