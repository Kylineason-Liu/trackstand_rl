"""Read-only analysis of the two October 31 C1 velocity replays.

No ROS, CAN, policy or simulator runs. Writes only this report directory.
Fresh A2/A1/A3 replies use the same recorded core-start clock, without shifts.
"""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
from html import escape
import json
import math
from pathlib import Path
import shutil
import struct

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXPERIMENTS = ROOT / "deploy/experiments"
LOGS = EXPERIMENTS / "logs"
STEMS = {
    50: "20261031_223709_716988_50hz_drive_velocity_replay",
    200: "20261031_223442_378454_200hz_drive_velocity_replay",
}
REFERENCE = EXPERIMENTS / "replay_targets/2500_isaac_velocity_startup.csv"
NATIVE = REFERENCE.with_name(REFERENCE.stem + "_simulation.csv")
SOURCE = EXPERIMENTS / "drive_velocity_replay.py"
HELPER = EXPERIMENTS / "drive_velocity_sine.py"
FIXTURE = "User confirmed identical airborne/free rear wheel, supported frame, no hand contact or setting changes."
BLACK, BLUE, ORANGE, GREEN, GREY = "#222222", "#1478b8", "#df8418", "#208564", "#777777"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def correlation(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else None


def held(t, y, query):
    index = np.searchsorted(t, query, side="right") - 1
    return y[np.clip(index, 0, len(y) - 1)]


def held_integral(t, y, query):
    cumulative = np.r_[0., np.cumsum(y[:-1] * np.diff(t))]
    index = np.clip(np.searchsorted(t, query, side="right") - 1, 0, len(t) - 1)
    return cumulative[index] + y[index] * (query - t[index])


def distribution(x):
    x = np.asarray(x)
    return dict(n=len(x), min=float(x.min()), median=float(np.median(x)),
                p95=float(np.percentile(x, 95)), p99=float(np.percentile(x, 99)), max=float(x.max()))


def simulation():
    native, reference = read_csv(NATIVE), read_csv(REFERENCE)
    metadata = json.loads(REFERENCE.with_suffix(".json").read_text())
    assert sha(REFERENCE) == metadata["reference_sha256"]
    assert sha(NATIVE) == metadata["native_sha256"]
    values = {k: np.array([float(r[k]) for r in native]) for k in native[0]}
    ref = {k: np.array([float(r[k]) for r in reference]) for k in reference[0]}
    assert len(native) == 4001 and len(reference) == 501
    assert all(np.array_equal(values[k][::8], ref[k]) for k in ref)
    assert np.allclose(np.diff(values["time_s"]), .0025, rtol=0, atol=1e-12)
    assert metadata["capture_completed"] and not metadata["episode_terminated"]
    assert values["time_s"][0] == 0 and values["time_s"][-1] == 10
    values.update(reference=ref, metadata=metadata)
    return values


def read_run(rate, sim):
    path = LOGS / f"{STEMS[rate]}.csv"
    metadata = json.loads(path.with_suffix(".json").read_text())
    saved_reference = path.with_name(path.stem + "_reference.csv")
    assert sha(saved_reference) == sha(REFERENCE) == metadata["reference_sha256"]
    assert metadata["source_sha256"] == sha(SOURCE) and metadata["helper_sha256"] == sha(HELPER)
    assert metadata["completed"] and metadata["disabled_and_restored"] and metadata["error"] is None
    assert metadata["packet_rate_hz"] == rate and metadata["target_rate_hz"] == 50
    assert metadata["encoder_sign"] == -1 and metadata["motor_id"] == 1
    phase = next(p for p in metadata["phases"] if p["name"] == "core")
    assert phase["planned_duration_s"] == 10 and phase["skipped_slots"] == 0
    origin, kt, zero = phase["start_monotonic_s"], metadata["torque_constant_nm_per_a"], metadata["initial_multi_turn_count"]
    rows = read_csv(path)
    validation = Counter()
    for row in rows:
        payload = bytes.fromhex(row["payload_hex"]) if row["payload_hex"] else b""
        if row["record_type"] == "rx":
            assert row["can_id"] == "001"
            op = int(row["opcode"], 16)
            assert payload[0] == op
            if op in (0xA2, 0xC1):
                value = -struct.unpack("<i", payload[1:])[0] * .01 * 2 * math.pi / 60
                assert abs(value - float(row["velocity_rad_s"])) < 1e-12
            elif op == 0xA1:
                current = struct.unpack("<i", payload[1:])[0] * .001
                assert abs(current - float(row["current_a"])) < 1e-12
                assert abs(-current * kt - float(row["current_torque_nm"])) < 1e-12
            elif op == 0xA3:
                count = struct.unpack("<i", payload[3:])[0]
                assert count == int(row["multi_turn_count"])
                delta = (count - zero + 2**31) % 2**32 - 2**31
                assert abs(-delta * 2 * math.pi / 16384 - float(row["position_rad"])) < 1e-12
            elif op in (0xAE, 0xCF):
                volts, amps = struct.unpack("<HH", payload[1:5])
                assert abs(volts * .01 - float(row["bus_voltage_v"])) < 1e-12
                assert abs(amps * .01 - float(row["bus_current_a"])) < 1e-12
                assert int(row["fault_code"]) == payload[7] == 0
                assert int(row["driver_mode"]) == payload[6]
            validation[f"decoded_rx_{op:02X}"] += 1
        elif payload:
            assert row["can_id"] == "101"
            op = payload[0]
            if op in (0xB0, 0xB3, 0xB5, 0xB8, 0xB9):
                assert len(payload) == 1  # No current, acceleration, gain or calibration writes.
            if row["record_type"] == "command":
                assert op == 0xC1 and len(payload) == 5
                decoded = -struct.unpack("<i", payload[1:])[0] * .01 * 2 * math.pi / 60
                assert abs(decoded - float(row["encoded_target_velocity_rad_s"])) < 1e-12
                wanted = float(row["target_velocity_rad_s"])
                assert abs(decoded - wanted) <= .5 * .01 * 2 * math.pi / 60 + 1e-12
                if row["method"] == "core":
                    source_time = float(row["target_sample_time_s"])
                    i = round(source_time / .02)
                    assert abs(wanted - sim["reference"]["velocity_rad_s"][i]) < 1e-12
                else:
                    assert wanted == 0
                validation["decoded_commands"] += 1
    selected = [r for r in rows if r["method"] == "core" and r["record_type"] == "command"]
    ct = np.array([float(r["host_monotonic_s"]) - origin for r in selected])
    cv = np.array([float(r["encoded_target_velocity_rad_s"]) for r in selected])
    sample_times = np.array([float(r["target_sample_time_s"]) for r in selected])
    assert len(selected) == 10 * rate == phase["commands"]
    assert np.all(np.diff(ct) > 0)
    assert np.array_equal(sample_times, np.repeat(np.arange(500) * .02, rate // 50))
    assert np.allclose(np.diff(ct), 1 / rate, atol=.0002, rtol=0)

    def fresh(opcode, column):
        selected = [r for r in rows if r["method"] == "core" and r["record_type"] == "rx"
                    and r["opcode"] == opcode]
        t = np.array([float(r["host_monotonic_s"]) - origin for r in selected])
        y = np.array([float(r[column]) for r in selected])
        assert np.all(np.diff(t) > 0) and np.all(np.isfinite(y))
        return t, y

    st, speed = fresh("A2", "velocity_rad_s")
    tt, torque = fresh("A1", "current_torque_nm")
    pt, position = fresh("A3", "position_rad")
    position -= phase["initial_position_rad"]
    c1t, c1v = fresh("C1", "velocity_rad_s")
    status_rows = [r for r in rows if r["method"] == "core" and r["record_type"] == "rx" and r["opcode"] == "AE"]
    assert all(int(r["driver_mode"]) == 3 for r in status_rows)
    windows = {}
    for name, (lower, upper) in {"all": (0, 10), "startup": (0, 2), "settling": (2, 10), "last2s": (8, 10)}.items():
        keep = (st >= max(lower, ct[0])) & (st < upper)
        target = held(ct, cv, st[keep])
        error = speed[keep] - target
        native = np.interp(st[keep], sim["time_s"], sim["velocity_rad_s"])
        ktau = (tt >= lower) & (tt < upper)
        windows[name] = dict(
            lower_s=lower, upper_s=upper, samples=int(keep.sum()), velocity_rms_rad_s=rms(speed[keep]),
            target_rms_rad_s=rms(target), error_rms_rad_s=rms(error), mean_error_rad_s=float(error.mean()),
            p95_absolute_error_rad_s=float(np.percentile(abs(error), 95)), peak_absolute_error_rad_s=float(max(abs(error))),
            error_rms_vs_native_sim_rad_s=rms(speed[keep] - native), correlation=correlation(speed[keep], target),
            fraction_within_0p1_rad_s=float(np.mean(abs(error) <= .1)),
            fraction_within_0p05_rad_s=float(np.mean(abs(error) <= .05)),
            torque_rms_nm=rms(torque[ktau]), torque_peak_absolute_nm=float(max(abs(torque[ktau]))),
        )
    # Separate diagnostic: five real A3 intervals (~100 ms), no moving average of A2.
    averages = []
    for first in range(0, len(pt) - 5, 5):
        last = first + 5
        a, b = pt[first], pt[last]
        actual = (position[last] - position[first]) / (b - a)
        requested = (held_integral(ct, cv, b) - held_integral(ct, cv, a)) / (b - a)
        averages.append(dict(start_s=float(a), end_s=float(b), time_s=float((a + b) / 2),
                             actual_velocity_rad_s=float(actual), target_velocity_rad_s=float(requested),
                             error_rad_s=float(actual - requested)))
    averaged_stats = {}
    for name, lo, hi in (("startup", 0, 2), ("settling", 2, 10)):
        part = [r for r in averages if r["start_s"] >= lo and r["end_s"] < hi]
        measured = np.array([r["actual_velocity_rad_s"] for r in part])
        target = np.array([r["target_velocity_rad_s"] for r in part])
        averaged_stats[name] = dict(samples=len(part), error_rms_rad_s=rms(measured - target),
                                   actual_rms_rad_s=rms(measured), target_rms_rad_s=rms(target),
                                   correlation=correlation(measured, target))
    plateaus = []
    first = 0
    while first < len(pt) - 1:
        following = first + 1
        while following < len(pt) and position[following] == position[first]:
            following += 1
        a, b = pt[first], pt[following - 1]
        if a >= 2 and b - a >= .2:
            travel = float(held_integral(ct, cv, b) - held_integral(ct, cv, a))
            effort = torque[(tt >= a) & (tt <= b)]
            plateaus.append(dict(start_s=float(a), end_s=float(b), duration_s=float(b - a),
                                 commanded_travel_rad=travel, actual_encoder_count_change=0,
                                 commanded_travel_encoder_counts=travel * 16384 / (2 * math.pi),
                                 current_torque_min_nm=float(effort.min()), current_torque_max_nm=float(effort.max())))
        first = following
    plateaus.sort(key=lambda r: r["duration_s"], reverse=True)
    # Descriptive delay only: no delay correction is applied to displayed curves or main errors.
    lags = np.arange(-.05, .100001, .0005)
    keep = (st >= .1) & (st < 1.)
    scores = np.array([rms(speed[keep] - held(ct, cv, st[keep] - lag)) for lag in lags])
    best = int(np.argmin(scores))
    metadata_result = dict(
        rate_hz=rate, source_csv=path.name, complete=True, fault_codes=[0], fixture=FIXTURE,
        core_phase=phase, source_validation=dict(validation), settings=metadata["original_settings"],
        gains=metadata["raw_gains"], initial_count=zero, initial_native_phase_rad=float((zero % 16384) * 2 * math.pi / 16384),
        voltage_min_v=min(float(r["bus_voltage_v"]) for r in status_rows),
        voltage_max_v=max(float(r["bus_voltage_v"]) for r in status_rows),
        temperature_max_c=max(float(r["temperature_c"]) for r in status_rows),
        actual_packet_interval_ms=distribution(np.diff(ct) * 1000),
        a2_interval_ms=distribution(np.diff(st) * 1000),
        maximum_command_quantization_rad_s=max(abs(float(r["target_velocity_rad_s"]) - float(r["encoded_target_velocity_rad_s"]))
                                               for r in selected),
        windows=windows, encoder_average_100ms=averaged_stats,
        unchanged_encoder_intervals=plateaus,
        apparent_startup_lag_ms=float(lags[best] * 1000),
        lag_scan=dict(window_s=[.1, 1.], step_ms=.5, shifted_rmse_rad_s=float(scores[best]),
                      interpretation="best shifted held-target RMSE, not isolated firmware latency"),
        last_a3_time_s=float(pt[-1]), final_a3_displacement_rad=float(position[-1]),
        transmitted_velocity_integral_rad=float(held_integral(ct, cv, 10.)),
        final_displacement_minus_transmitted_integral_rad=float(position[-1] - held_integral(ct, cv, pt[-1])),
        integrated_a2_velocity_rad=float(np.trapezoid(speed, st)),
        maximum_speed_rad_s=float(max(abs(speed))), maximum_current_derived_torque_nm=float(max(abs(torque))),
        a3_lsb_rad=2 * math.pi / 16384,
    )
    for opcode, t, y, key in (("a2", st, speed, "velocity_rad_s"), ("a1", tt, torque, "torque_nm"),
                               ("a3", pt, position, "position_rad"), ("c1_feedback", c1t, c1v, "velocity_rad_s")):
        write_csv(HERE / f"{rate}hz_{opcode}_samples.csv", [dict(time_s=float(a), **{key: float(b)}) for a, b in zip(t, y)])
    write_csv(HERE / f"{rate}hz_commands.csv", [dict(time_s=float(a), velocity_rad_s=float(b), source_time_s=float(c))
                                               for a, b, c in zip(ct, cv, sample_times)])
    write_csv(HERE / f"{rate}hz_encoder_average_100ms.csv", averages)
    write_csv(HERE / f"{rate}hz_encoder_plateaus.csv", plateaus)
    write_csv(HERE / f"{rate}hz_lag_scan.csv", [dict(lag_ms=float(a * 1000), rmse_rad_s=float(b))
                                               for a, b in zip(lags, scores)])
    return dict(rate=rate, command=(ct, cv), speed=(st, speed), torque=(tt, torque), position=(pt, position),
                averages=averages, stats=metadata_result, metadata=metadata)


def decorate(ax, timeseries, lower, upper, unit, mark_origin=True):
    all_values = []
    for t, y in timeseries:
        keep = (t >= lower) & (t <= upper)
        if np.any(keep):
            all_values.append(y[keep])
    values = np.concatenate(all_values)
    lo, hi = float(values.min()), float(values.max())
    margin = max(.001, .12 * (hi - lo))
    ax.set(xlim=(lower, upper), ylim=(lo - margin, hi + margin), ylabel=unit)
    ax.grid(alpha=.22)
    if lower == 0 and mark_origin:
        ax.axvline(0, color="red", lw=1.2)
    ax.legend(loc="lower right", fontsize=8, framealpha=.94)
    ax.text(.012, .97, f"Plotted min/max: {lo:.4f} / {hi:.4f}", transform=ax.transAxes,
            va="top", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", alpha=.9))


def comparison(sim, runs, lower, upper, title):
    fig, axes = plt.subplots(3, len(runs), figsize=(9.2 * len(runs), 10), squeeze=False, sharex="col")
    for column, run in enumerate(runs):
        ct, cv = run["command"]
        st, speed = run["speed"]
        tt, torque = run["torque"]
        pt, position = run["position"]
        command_t, command_v = np.r_[ct, 10.], np.r_[cv, cv[-1]]
        ax = axes[0, column]
        ax.plot(sim["time_s"], sim["velocity_rad_s"], color=BLACK, lw=1, label="Simulation actual velocity (400 Hz)")
        ax.step(command_t, command_v, where="post", color=ORANGE, lw=1, label="Actually sent C1 velocity (held)")
        ax.plot(st, speed, color=BLUE, lw=.75, label="Real fresh A2 velocity")
        decorate(ax, [(sim["time_s"], sim["velocity_rad_s"]), (command_t, command_v), (st, speed)],
                 lower, upper, "Velocity (rad/s)")
        ax.set_title(f"C1 {run['rate']} Hz packets; 50 Hz references")
        ax = axes[1, column]
        ax.plot(tt, torque, color=GREEN, lw=.8, label="Real A1 current × B0 Kt")
        decorate(ax, [(tt, torque)], lower, upper, "Torque estimate (Nm)")
        ax = axes[2, column]
        integral_t = np.unique(np.r_[0., ct, 10.])
        integral_q = held_integral(ct, cv, integral_t)
        ax.plot(sim["time_s"], sim["position_rad"], color=GREY, ls="--", lw=1, label="Saved simulation joint displacement")
        ax.plot(integral_t, integral_q, color=ORANGE, lw=1.3, label="Integral of transmitted C1 velocity")
        ax.plot(pt, position, color=BLUE, lw=1, label="Real A3 multi-turn displacement")
        decorate(ax, [(sim["time_s"], sim["position_rad"]), (integral_t, integral_q), (pt, position)],
                 lower, upper, "Displacement (rad)")
        ax.set_xlabel("Seconds since core replay / source spawn; no time shift")
    # Matching scales within each row make the rate comparison fair.
    for row in axes:
        lo, hi = min(ax.get_ylim()[0] for ax in row), max(ax.get_ylim()[1] for ax in row)
        for ax in row:
            ax.set_ylim(lo, hi)
    fig.suptitle(f"{title}\nmodel_2500 v1; DR; seed 42; env 0 — airborne hardware", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .94))
    return fig


def averaged_figure(runs):
    fig, axes = plt.subplots(2, 2, figsize=(18, 8), sharex="col")
    for column, run in enumerate(runs):
        data = [r for r in run["averages"] if r["start_s"] >= 2 and r["end_s"] < 10]
        t = np.array([r["time_s"] for r in data])
        actual = np.array([r["actual_velocity_rad_s"] for r in data])
        target = np.array([r["target_velocity_rad_s"] for r in data])
        ax = axes[0, column]
        ax.plot(t, target, color=ORANGE, lw=1.1, label="Time-averaged transmitted target")
        ax.plot(t, actual, color=BLUE, lw=.9, marker=".", ms=3, label="A3 displacement / actual interval (~100 ms)")
        decorate(ax, [(t, target), (t, actual)], 2, 10, "Average velocity (rad/s)")
        ax.set_title(f"C1 {run['rate']} Hz — supplementary encoder average")
        ax = axes[1, column]
        ax.plot(t, actual - target, color=BLUE, lw=1, label="Encoder-average error")
        ax.axhline(0, color="black", lw=.6)
        decorate(ax, [(t, actual - target)], 2, 10, "Average error (rad/s)")
        ax.set_xlabel("Seconds since core start; averaging is diagnostic, not a shifted raw trace")
    for row in axes:
        lo, hi = min(ax.get_ylim()[0] for ax in row), max(ax.get_ylim()[1] for ax in row)
        for ax in row:
            ax.set_ylim(lo, hi)
    fig.suptitle("Small settling movements: check A3 displacement as well as the noisy A2 estimate", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .95))
    return fig


def plateau_figure(runs):
    fig, axes = plt.subplots(3, 2, figsize=(18, 10), sharex="col")
    for column, run in enumerate(runs):
        span = run["stats"]["unchanged_encoder_intervals"][0]
        start, end = span["start_s"], span["end_s"]
        duration = end - start
        ct, cv = run["command"]
        st, speed = run["speed"]
        tt, torque = run["torque"]
        pt, position = run["position"]
        ax = axes[0, column]
        ax.step(ct - start, cv, where="post", color=ORANGE, lw=1, label="Actually transmitted C1 velocity")
        ax.plot(st - start, speed, color=BLUE, lw=.8, label="Fresh A2 velocity estimate")
        decorate(ax, [(ct - start, cv), (st - start, speed)], 0, duration, "Velocity (rad/s)", mark_origin=False)
        ax.set_title(f"{run['rate']} Hz: unchanged A3 count, core t={start:.3f}–{end:.3f} s")
        ax = axes[1, column]
        ax.plot(tt - start, torque, color=GREEN, lw=1, label="A1 current × B0 Kt")
        decorate(ax, [(tt - start, torque)], 0, duration, "Torque estimate (Nm)", mark_origin=False)
        ax = axes[2, column]
        use = (pt >= start) & (pt <= end)
        origin_q = position[np.flatnonzero(use)[0]]
        local = np.unique(np.r_[start, ct[(ct > start) & (ct < end)], end])
        commanded = held_integral(ct, cv, local) - held_integral(ct, cv, start)
        ax.plot(local - start, commanded, color=ORANGE, lw=1.4, label="Travel implied by commanded velocity")
        ax.plot(pt[use] - start, position[use] - origin_q, color=BLUE, lw=1.4, label="Actual A3 count change: zero")
        ax.axhspan(-2 * math.pi / 16384, 2 * math.pi / 16384, color=BLUE, alpha=.10, label="±1 encoder count")
        decorate(ax, [(local - start, commanded), (pt[use] - start, position[use] - origin_q)],
                 0, duration, "Local travel (rad)", mark_origin=False)
        ax.set_xlabel("Seconds within this measured encoder plateau; original core time given above")
    for row in axes:
        lo, hi = min(ax.get_ylim()[0] for ax in row), max(ax.get_ylim()[1] for ax in row)
        for ax in row:
            ax.set_ylim(lo, hi)
    fig.suptitle("Tiny commands are not followed continuously: position-count plateaus", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .95))
    return fig


def generate_report(sim, runs, figures):
    a, b = [r["stats"] for r in runs]
    native_integral = float(np.trapezoid(sim["velocity_rad_s"], sim["time_s"]))
    native_position = float(sim["position_rad"][-1])
    summary = dict(model="2500", task="Trackstand-Balance-v1", dr=True, seed=42, environment=0,
                   duration_s=10, fixture=FIXTURE, runs={str(r["rate"]): r["stats"] for r in runs},
                   source_native_velocity_integral_rad=native_integral,
                   source_native_joint_displacement_rad=native_position,
                   source_velocity_position_difference_rad=native_integral - native_position,
                   reference_meaning="v1 actual simulated velocity, not a velocity-action policy output",
                   comparison_boundary="unloaded airborne C1 motion; neither loaded balance nor a v2 policy evaluation")
    write_json(HERE / "statistics.json", summary)
    flat = []
    for run in runs:
        for window, values in run["stats"]["windows"].items():
            flat.append(dict(rate_hz=run["rate"], window=window, **values))
    write_csv(HERE / "summary.csv", flat)
    table = []
    for s in (a, b):
        table.append(f"| {s['rate_hz']} | {s['windows']['all']['error_rms_rad_s']:.4f} | "
                     f"{s['windows']['startup']['error_rms_rad_s']:.4f} | "
                     f"{s['windows']['settling']['error_rms_rad_s']:.4f} | "
                     f"{s['windows']['settling']['target_rms_rad_s']:.4f} | "
                     f"{s['windows']['settling']['correlation']:.3f} | {s['apparent_startup_lag_ms']:.1f} |")
    report = f"""# Rear C1 tracking of policy-generated velocity — October 31, 2026

**Both runs reproduce the large startup movement reasonably well. Neither establishes precise tracking of the small settling waveform.** The 50-Hz run has less reported low-speed error/ripple than the 200-Hz run, but one run each, starting at different rotor angles, cannot isolate packet frequency as the cause. Calling the complete stationary-balancing velocity behavior faithfully reproduced would be premature.

[Figure gallery](index.html) · [All figures PDF](all_figures.pdf) · [Full comparison](comparison_full.png) · [Startup](comparison_startup.png) · [Settling](comparison_settling.png) · [Encoder averages](settling_encoder_100ms.png) · [Count plateaus](low_speed_count_plateaus.png) · [Metrics CSV](summary.csv) · [Exact statistics](statistics.json)

## Data, conditions and provenance

- Found exactly six new files: CSV/JSON/reference for `{STEMS[50]}` and `{STEMS[200]}` in `deploy/experiments/logs/`.
- {FIXTURE}
- Source: **model_2500**, v1, DR, seed 42, env 0, first episode, seconds 0–10; 400-Hz native states and 50-Hz references. There is no source reset in this window. V1 outputs **torque**; the resulting simulated velocity is used as a **new C1 reference**. This is not a trained-v2 velocity-action replay.
- Both reference copies have SHA256 `{sha(REFERENCE)}`; all 500 held target levels are identical across the rates, after decoding every actual command. At 200 Hz each level is resent four times; no interpolation or new between-boundary reference exists.
- Both runs completed and confirmed CF disable / CD restoration. No recorded driver fault, skipped core command, gain/current-limit/acceleration/origin write, or source/helper hash mismatch. Core commands **500/2,000** at **{a['core_phase']['achieved_rate_hz']:.5f}/{b['core_phase']['achieved_rate_hz']:.5f} Hz**. Maximum core command lateness **{a['core_phase']['max_lateness_s']*1000:.3f}/{b['core_phase']['max_lateness_s']*1000:.3f} ms**: the observed tracking mismatch is not explained by a large host packet stall.
- Raw B8/B9 **0.5/0.005**, Kt **0.300000012 Nm/A**, B3 **82.498 A configured ceiling**, B5 raw **0**. Gain units/internal PI timing remain unknown. B3 is not measured current, and its raw ceiling is not evidence of that much delivered torque.
- Recorded supply **{min(a['voltage_min_v'],b['voltage_min_v']):.2f}–{max(a['voltage_max_v'],b['voltage_max_v']):.2f} V**; maximum temperature **{max(a['temperature_max_c'],b['temperature_max_c']):.0f}°C**. Maximum current-derived effort **{a['maximum_current_derived_torque_nm']:.4f}/{b['maximum_current_derived_torque_nm']:.4f} Nm** at 50/200 Hz. These are low-demand tests, not envelope/thermal validation.
- Initial multi-turn counts **{a['initial_count']}/{b['initial_count']}** differ. Separate software origins and core-start A3 snapshots remove displacement offset, but do not physically match cogging/friction phase. One successive run at each rate does not prove repeatability or a causal rate advantage.

## Aligned plots and calculation

Every graph uses the recorded core-start monotonic origin. Native simulation velocity is black, actual held encoded C1 target orange and fresh A2 velocity blue. A1 current × B0 gives electromagnetic torque estimate, not independent shaft torque. Fresh A3 gives multi-turn displacement; cached copies in command/query rows are excluded from response statistics.

All raw graphs are unsmoothed, unshifted and retain actual asynchronous sample times. Red t=0 lines mark replay/source start; zooms beginning at t=2 do not represent a new spawn. Legends are lower-right, scales are shared across the two rate columns, and each panel labels its plotted minimum/maximum. No finite sample in its time window is clipped by the axes.

| C1 packets Hz | Full 0–10 s RMS error | Startup 0–2 s RMS error | Settling 2–10 s RMS error | Settling target RMS | Settling correlation | Apparent startup lag ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

Errors/speeds are rad/s. Primary RMS compares each fresh A2 velocity with the **latest actually transmitted encoded target**, without compensating delay. Small settling targets make peak-normalized full-run error misleading. The 50/200-Hz settling 95th-percentile absolute errors are **{a['windows']['settling']['p95_absolute_error_rad_s']:.4f}/{b['windows']['settling']['p95_absolute_error_rad_s']:.4f} rad/s**; target RMS is only about 0.030 rad/s. Absolute errors within 0.1 rad/s cover **{a['windows']['settling']['fraction_within_0p1_rad_s']*100:.1f}/{b['windows']['settling']['fraction_within_0p1_rad_s']*100:.1f}%** of settling samples. That threshold is descriptive, not an established balance-performance pass criterion.

Comparing directly to the full native simulation curve gives all-window RMS **{a['windows']['all']['error_rms_vs_native_sim_rad_s']:.4f}/{b['windows']['all']['error_rms_vs_native_sim_rad_s']:.4f} rad/s**. This comparison includes 400→50-Hz reference sampling effects, so it must not all be attributed to the motor.

![Full comparison](comparison_full.png)

## Startup, settling and apparent delay

The initial large forward velocity bump has startup correlation **{a['windows']['startup']['correlation']:.3f}/{b['windows']['startup']['correlation']:.3f}**. Actual sampled peak speed reaches **{a['maximum_speed_rad_s']:.3f}/{b['maximum_speed_rad_s']:.3f} rad/s**, versus a transmitted-target peak around **2.324 rad/s** and native simulation peak **2.484 rad/s**. This is clear coarse motion reproduction, with ripple and individual peak excursions rather than perfect tracking.

The lag values minimize shifted held-target RMS on **0.1–1.0 s**, scanning −50 to +100 ms in 0.5-ms increments. Positive means response behind the command. They describe this transient's best apparent shift, not motor-internal pure latency, an invariant delay, or 0.5-ms measurement precision. All main metrics/graphs retain the original delay. Individual peak-time differences are not substituted for controller latency.

After 2 s, the command is very small and its detailed waveform is weakly correlated with A2. Reported real speed RMS is **{a['windows']['settling']['velocity_rms_rad_s']:.4f}/{b['windows']['settling']['velocity_rms_rad_s']:.4f} rad/s**, against target RMS **{a['windows']['settling']['target_rms_rad_s']:.4f}/{b['windows']['settling']['target_rms_rad_s']:.4f}**. Good reproduction of the big bump alone is insufficient to claim precise tiny back-and-forth control.

A2 spikes may contain firmware velocity-estimator noise, not only true physical velocity fluctuation. Therefore the supplementary plot differentiates fresh A3 over five actual encoder intervals (~100 ms), and compares it with the exact target integral over those same intervals. This is explicitly an average, not a replacement for the raw trace or an independent sensor. Settling average-speed RMS errors remain **{a['encoder_average_100ms']['settling']['error_rms_rad_s']:.4f}/{b['encoder_average_100ms']['settling']['error_rms_rad_s']:.4f} rad/s**, with correlations **{a['encoder_average_100ms']['settling']['correlation']:.3f}/{b['encoder_average_100ms']['settling']['correlation']:.3f}**. Thus poor fine waveform correspondence is not explained solely by isolated A2 spikes, although shared encoder/firmware/timestamp limitations remain.

### Stronger low-speed evidence: unchanged position counts

| Packets Hz | Core interval s | Unchanged-count duration s | Travel required by C1 velocity rad | Equivalent encoder counts required | Actual count change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | {a['unchanged_encoder_intervals'][0]['start_s']:.3f}–{a['unchanged_encoder_intervals'][0]['end_s']:.3f} | {a['unchanged_encoder_intervals'][0]['duration_s']:.3f} | {a['unchanged_encoder_intervals'][0]['commanded_travel_rad']:.6f} | {a['unchanged_encoder_intervals'][0]['commanded_travel_encoder_counts']:.1f} | 0 |
| 200 | {b['unchanged_encoder_intervals'][0]['start_s']:.3f}–{b['unchanged_encoder_intervals'][0]['end_s']:.3f} | {b['unchanged_encoder_intervals'][0]['duration_s']:.3f} | {b['unchanged_encoder_intervals'][0]['commanded_travel_rad']:.6f} | {b['unchanged_encoder_intervals'][0]['commanded_travel_encoder_counts']:.1f} | 0 |

These are fresh A3 CAN replies with identical decoded multi-turn counts, not repeated cached rows. One encoder count is **0.0003835 rad**; the missing commanded travel is tens of counts, so sub-count resolution alone is insufficient to explain it. Current-derived torque varies through **{a['unchanged_encoder_intervals'][0]['current_torque_min_nm']:.4f}–{a['unchanged_encoder_intervals'][0]['current_torque_max_nm']:.4f} Nm** in the 50-Hz example and **{b['unchanged_encoder_intervals'][0]['current_torque_min_nm']:.4f}–{b['unchanged_encoder_intervals'][0]['current_torque_max_nm']:.4f} Nm** in the 200-Hz example. The staircase motion and torque buildup support intermittent near-zero motion/friction plus servo dynamics as explanations. They do not independently identify friction coefficients, firmware PI state or motor-internal acquisition times. The dedicated plateau graph uses a clearly labelled local origin, not a new episode/spawn.

## Displacement: compare to the velocity actually commanded

The source's saved position ends at **{native_position:.6f} rad**, while its own saved velocity integrates to **{native_integral:.6f} rad**. The 50-Hz held reference integrates to approximately **0.694907 rad**. These are already different before hardware is involved. We do not invent a physical cause or blame C1 for that source-side numerical/quantity discrepancy.

Actually transmitted target integrals are **{a['transmitted_velocity_integral_rad']:.6f}/{b['transmitted_velocity_integral_rad']:.6f} rad**. Real A3 travel ends at **{a['final_a3_displacement_rad']:.6f}/{b['final_a3_displacement_rad']:.6f} rad**, at **{a['last_a3_time_s']:.6f}/{b['last_a3_time_s']:.6f} s**. Errors relative to the command integral at each last A3 time are **{a['final_displacement_minus_transmitted_integral_rad']:.6f}/{b['final_displacement_minus_transmitted_integral_rad']:.6f} rad**. The integrated motion is fairly close, despite the imperfect fine speed waveform.

The graphs show saved simulation joint angle (grey dashed), the integral of the C1 velocity actually sent (orange), and actual A3 travel (blue). The orange curve is a kinematic diagnostic, not a position command sent to C1. Core-start zero snapshots are about **{a['core_phase']['initial_position_age_s']*1000:.2f}/{b['core_phase']['initial_position_age_s']*1000:.2f} ms** old, which is documented rather than silently treated as simultaneous.

## Assessment and next decisions

1. **Promising for the large transient and overall travel, not yet a precise match of the complete stationary-balancing waveform.** A 0.05–0.09-rad/s error is small compared with the big startup bump but larger than these tiny settling targets. No closed-loop tolerance proving this adequate for balance has been established.
2. **200-Hz resending has no demonstrated advantage here.** The 50-Hz run has smaller full/settling A2 error, while startup errors are similar. Different starting rotor phases, one run each and possible estimator/low-speed nonlinearities prevent claiming that 50 Hz is intrinsically superior.
3. **Most plausible fine-tracking limits are near-zero friction/stick-slip and C1 servo/velocity-estimation dynamics.** These are candidate explanations, not separately identified causes. The low-demand torque, stable recorded supply and clean host timing make torque-cap saturation, large supply sag or a major host packet stall poor explanations for this pair.
4. **Next useful test:** repeat this same reference at matched marked rotor phase, especially repeat the small 2–10-s segment; use identical supported loaded conditions as a separate comparison. Do not increase gains or change friction solely to force an apparent match. This replay still does not identify whether a simulated P approximation is sufficient or whether PI/delay/nonlinear dynamics must be modeled.

## Reproduce and inspect exact data

```bash
MPLCONFIGDIR=/tmp/trackstand_c1_velocity_replay_plot python3 \\
  deploy/experiments/reports/20261031_drive_c1_velocity_replay/analyze.py
python3 deploy/experiments/reports/20261031_drive_c1_velocity_replay/verify.py
```

CSV exports retain each plotted native sample and actual sent target. `statistics.json` records windows, timing, settings, quantization, lag and encoder-average definitions. `protected_input_hashes.json`/`verification.json` check that raw inputs and existing controllers remain unchanged. `runtime_snapshot/` preserves the exact matched sender/helper source. No hardware, simulation, policy or controller modification was performed for this analysis.
"""
    (HERE / "README.md").write_text(report)
    cards = []
    for name, title in figures:
        cards.append(f'<section><h2>{escape(title)}</h2><p><a href="{name}.png">PNG</a> · '
                     f'<a href="{name}.svg">SVG</a> · <a href="{name}.pdf">PDF</a></p>'
                     f'<a href="{name}.png"><img src="{name}.png" alt="{escape(title)}"></a></section>')
    (HERE / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>October 31 C1 velocity replay</title>'
        '<style>body{font:16px system-ui;margin:24px auto;max-width:1550px;padding:0 18px;color:#222}'
        'img{max-width:100%;height:auto}section{margin:36px 0}a{color:#1467a5}</style>'
        '<h1>Rear C1 velocity replay — 50 vs 200 Hz</h1>'
        '<p>Model 2500 v1 · DR · airborne/free · unchanged settings. Large startup motion is reproduced; '
        'precise small settling-waveform tracking is not established.</p>'
        '<p><a href="README.md">Report</a> · <a href="all_figures.pdf">All figures PDF</a> · '
        '<a href="summary.csv">Metrics CSV</a> · <a href="statistics.json">Exact statistics</a></p>'
        + "".join(cards) + '</html>')
    return summary


def main():
    paths = [REFERENCE, REFERENCE.with_suffix(".json"), NATIVE, SOURCE, HELPER,
             EXPERIMENTS / "record_drive_velocity.py", ROOT / "progress_record.md", ROOT / "progress_record.txt"]
    for stem in STEMS.values():
        paths.extend(LOGS / f"{stem}{suffix}" for suffix in (".csv", ".json", "_reference.csv"))
    original = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    write_json(HERE / "protected_input_hashes.json", original)
    snapshot = HERE / "runtime_snapshot"
    snapshot.mkdir(exist_ok=True)
    for path in (SOURCE, HELPER, EXPERIMENTS / "record_drive_velocity.py"):
        shutil.copyfile(path, snapshot / (path.name + ".txt"))
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    sim = simulation()
    runs = [read_run(rate, sim) for rate in (50, 200)]
    assert runs[0]["metadata"]["original_settings"] == runs[1]["metadata"]["original_settings"]
    definitions = [
        ("comparison_full", "Full 10-second comparison", lambda: comparison(sim, runs, 0, 10, "Full velocity replay")),
        ("comparison_startup", "Startup, 0–2 seconds", lambda: comparison(sim, runs, 0, 2, "Startup transient")),
        ("comparison_settling", "Settling, 2–10 seconds", lambda: comparison(sim, runs, 2, 10, "Small settling movements")),
        ("settling_encoder_100ms", "Supplementary 100-ms encoder averages", lambda: averaged_figure(runs)),
        ("low_speed_count_plateaus", "Low-speed position-count plateaus", lambda: plateau_figure(runs)),
        ("replay_50hz", "50-Hz replay", lambda: comparison(sim, [runs[0]], 0, 10, "50-Hz C1 replay")),
        ("replay_200hz", "200-Hz replay", lambda: comparison(sim, [runs[1]], 0, 10, "200-Hz C1 replay")),
    ]
    with PdfPages(HERE / "all_figures.pdf") as combined:
        for name, _, build in definitions:
            fig = build()
            for suffix in ("png", "svg", "pdf"):
                fig.savefig(HERE / f"{name}.{suffix}", dpi=165)
            combined.savefig(fig)
            plt.close(fig)
    summary = generate_report(sim, runs, [(name, title) for name, title, _ in definitions])
    assert all(sha(ROOT / name) == digest for name, digest in original.items())
    print(json.dumps({str(rate): summary["runs"][str(rate)]["windows"] for rate in (50, 200)}, indent=2))


if __name__ == "__main__":
    main()
