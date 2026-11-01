"""Analyze three recorded C1 replays without modifying logs or controllers."""

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
EXP = ROOT / "deploy/experiments"
OLD = EXP / "reports/20261031_drive_c1_velocity_replay"
STEMS = {
    50: "20261031_223709_716988_50hz_drive_velocity_replay",
    200: "20261031_223442_378454_200hz_drive_velocity_replay",
    400: "20261101_134940_221417_400hz_drive_velocity_replay",
}
NATIVE = EXP / "replay_targets/2500_isaac_velocity_startup_400hz_simulation.csv"
FIXTURE = "User confirmed airborne/free rear wheel, no hand contact, same firmware settings and supply."
BLACK, BLUE, ORANGE, GREEN, GREY = "#222222", "#1478b8", "#df8418", "#208564", "#777777"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values) ** 2)))


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else None


def held(t, y, query):
    return y[np.clip(np.searchsorted(t, query, side="right") - 1, 0, len(t) - 1)]


def integral(t, y, query):
    cumulative = np.r_[0., np.cumsum(y[:-1] * np.diff(t))]
    i = np.clip(np.searchsorted(t, query, side="right") - 1, 0, len(t) - 1)
    return cumulative[i] + y[i] * (query - t[i])


def distribution(values):
    return dict(min=float(min(values)), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), p99=float(np.percentile(values, 99)), max=float(max(values)))


def load_run(rate, sim):
    path = EXP / "logs" / f"{STEMS[rate]}.csv"
    meta = json.loads(path.with_suffix(".json").read_text())
    ref_path = path.with_name(path.stem + "_reference.csv")
    ref = read(ref_path)
    ref_hz = meta["target_rate_hz"]
    assert sha(ref_path) == meta["reference_sha256"]
    assert len(ref) == 10 * ref_hz + 1
    for i, row in enumerate(ref):
        assert abs(float(row["time_s"]) - i / ref_hz) < 1e-10
        assert float(row["velocity_rad_s"]) == sim["velocity_rad_s"][round(i * 400 / ref_hz)]
    snapshot = (EXP / "drive_velocity_replay.py" if rate == 400
                else OLD / "runtime_snapshot/drive_velocity_replay.py.txt")
    assert sha(snapshot) == meta["source_sha256"]
    assert sha(EXP / "drive_velocity_sine.py") == meta["helper_sha256"]
    assert meta["completed"] and meta["disabled_and_restored"] and meta["error"] is None
    assert meta["packet_rate_hz"] == rate and meta["motor_id"] == 1 and meta["encoder_sign"] == -1
    phase = next(p for p in meta["phases"] if p["name"] == "core")
    assert phase["planned_duration_s"] == 10
    origin, kt, zero = phase["start_monotonic_s"], meta["torque_constant_nm_per_a"], meta["initial_multi_turn_count"]
    rows = read(path)
    rx, tx, faults = Counter(), Counter(), set()
    for row in rows:
        payload = bytes.fromhex(row["payload_hex"]) if row["payload_hex"] else b""
        if not payload:
            continue
        op = payload[0]
        if row["record_type"] == "rx":
            assert row["can_id"] == "001" and int(row["opcode"], 16) == op
            rx[f"{op:02X}"] += 1
            if op in (0xA2, 0xC1):
                value = -struct.unpack("<i", payload[1:])[0] * .01 * 2 * math.pi / 60
                assert abs(value - float(row["velocity_rad_s"])) < 1e-12
            elif op == 0xA1:
                current = struct.unpack("<i", payload[1:])[0] * .001
                assert abs(current - float(row["current_a"])) < 1e-12
                assert abs(-current * kt - float(row["current_torque_nm"])) < 1e-12
            elif op == 0xA3:
                count = struct.unpack("<i", payload[3:7])[0]
                assert count == int(row["multi_turn_count"])
                delta = (count - zero + 2**31) % 2**32 - 2**31
                assert abs(-delta * 2 * math.pi / 16384 - float(row["position_rad"])) < 1e-12
            elif op in (0xAE, 0xCF):
                faults.add(payload[7])
                assert int(row["fault_code"]) == payload[7]
                assert int(row["driver_mode"]) == payload[6]
                assert float(row["bus_voltage_v"]) == int.from_bytes(payload[1:3], "little") * .01
        else:
            assert row["can_id"] == "101"
            tx[f"{op:02X}"] += 1
            if op in (0xB0, 0xB3, 0xB5, 0xB8, 0xB9):
                assert len(payload) == 1  # Read only: no gain/current/acceleration/origin modifications.
            if row["record_type"] == "command":
                assert op == 0xC1 and len(payload) == 5
                code = struct.unpack("<i", payload[1:])[0]
                decoded = -code * .01 * 2 * math.pi / 60
                desired = float(row["target_velocity_rad_s"])
                assert code == round(-desired * 6000 / (2 * math.pi))
                assert abs(decoded - float(row["encoded_target_velocity_rad_s"])) < 1e-12
                if row["method"] == "core":
                    i = round(float(row["target_sample_time_s"]) * ref_hz)
                    assert desired == float(ref[i]["velocity_rad_s"])
                else:
                    assert desired == 0
    assert faults == {0} and rx == tx

    def series(kind, op, key):
        selected = [r for r in rows if r["method"] == "core" and r["record_type"] == kind and r["opcode"] == op]
        t = np.array([float(r["host_monotonic_s"]) - origin for r in selected])
        y = np.array([float(r[key]) for r in selected])
        assert np.all(np.diff(t) > 0) and np.all(np.isfinite(y))
        return t, y

    ct, cv = series("command", "C1", "encoded_target_velocity_rad_s")
    st, speed = series("rx", "A2", "velocity_rad_s")
    tt, torque = series("rx", "A1", "current_torque_nm")
    pt, position = series("rx", "A3", "position_rad")
    position -= phase["initial_position_rad"]
    commands = [r for r in rows if r["method"] == "core" and r["record_type"] == "command"]
    sample_time = np.array([float(r["target_sample_time_s"]) for r in commands])
    sampled = np.array([float(r["target_velocity_rad_s"]) for r in commands])
    late = np.array([float(r["lateness_s"]) for r in commands])
    assert len(ct) == phase["commands"]
    indices = np.rint(sample_time * ref_hz).astype(int)
    missing = np.setdiff1d(np.arange(10 * ref_hz), indices) / ref_hz
    skipped = sum(int(r["skipped_slots"]) for r in rows if r["method"] == "core" and r["skipped_slots"])
    assert skipped == phase["skipped_slots"] == 10 * rate - len(ct)
    if rate == 400:
        assert len(missing) == skipped and len(np.unique(indices)) == len(indices)
    else:
        assert np.array_equal(indices, np.repeat(np.arange(500), rate // 50))
    status = [r for r in rows if r["method"] == "core" and r["record_type"] == "rx" and r["opcode"] == "AE"]
    assert all(int(r["driver_mode"]) == 3 for r in status)
    windows = {}
    for name, lo, hi in (("all", 0, 10), ("startup", 0, 2), ("settling", 2, 10), ("last2s", 8, 10)):
        use = (st >= max(lo, ct[0])) & (st < hi)
        target = held(ct, cv, st[use])
        native = np.interp(st[use], sim["time_s"], sim["velocity_rad_s"])
        error = speed[use] - target
        tau = torque[(tt >= lo) & (tt < hi)]
        windows[name] = dict(
            lower_s=lo, upper_s=hi, samples=int(use.sum()), error_rms_rad_s=rms(error),
            error_rms_vs_native_sim_rad_s=rms(speed[use] - native), target_rms_rad_s=rms(target),
            actual_rms_rad_s=rms(speed[use]), correlation=corr(speed[use], target),
            mean_error_rad_s=float(error.mean()), p95_absolute_error_rad_s=float(np.percentile(abs(error), 95)),
            maximum_absolute_error_rad_s=float(max(abs(error))),
            fraction_within_0p1_rad_s=float(np.mean(abs(error) <= .1)), torque_rms_nm=rms(tau),
            peak_current_derived_torque_nm=float(max(abs(tau))),
        )
    averages = []
    for a in range(0, len(pt) - 5, 5):
        b = a + 5
        measured = (position[b] - position[a]) / (pt[b] - pt[a])
        desired = (integral(ct, cv, pt[b]) - integral(ct, cv, pt[a])) / (pt[b] - pt[a])
        averages.append(dict(start_s=float(pt[a]), end_s=float(pt[b]), time_s=float((pt[a] + pt[b]) / 2),
                             actual_velocity_rad_s=float(measured), target_velocity_rad_s=float(desired),
                             error_rad_s=float(measured - desired)))
    averaged = {}
    for name, lo, hi in (("startup", 0, 2), ("settling", 2, 10)):
        part = [r for r in averages if r["start_s"] >= lo and r["end_s"] < hi]
        a, b = [[r[key] for r in part] for key in ("actual_velocity_rad_s", "target_velocity_rad_s")]
        averaged[name] = dict(error_rms_rad_s=rms(np.array(a) - b), correlation=corr(a, b), samples=len(part))
    plateaus, a = [], 0
    while a < len(pt) - 1:
        b = a + 1
        while b < len(pt) and position[b] == position[a]:
            b += 1
        start, stop = pt[a], pt[b - 1]
        if start >= 2 and stop - start >= .2:
            travel = float(integral(ct, cv, stop) - integral(ct, cv, start))
            tau = torque[(tt >= start) & (tt <= stop)]
            plateaus.append(dict(
                start_s=float(start), end_s=float(stop), duration_s=float(stop - start),
                commanded_travel_rad=travel, commanded_travel_encoder_counts=travel * 16384 / (2*math.pi),
                actual_encoder_count_change=0, torque_min_nm=float(min(tau)), torque_max_nm=float(max(tau)),
            ))
        a = b
    plateaus.sort(key=lambda p: p["duration_s"], reverse=True)
    lags = np.arange(-.05, .100001, .0005)
    use = (st >= .1) & (st < 1.)
    scores = np.array([rms(speed[use] - held(ct, cv, st[use] - lag)) for lag in lags])
    best = int(np.argmin(scores))
    stats = dict(
        rate_hz=rate, reference_hz=ref_hz, source_csv=path.name, completed=meta["completed"],
        disabled_and_restored=meta["disabled_and_restored"], fault_codes=sorted(faults), fixture=FIXTURE,
        core_phase=phase, decoded_rx_counts=dict(rx), decoded_tx_counts=dict(tx), settings=meta["original_settings"],
        gains=meta["raw_gains"], torque_constant_nm_per_a=kt, initial_count=zero,
        initial_native_phase_rad=(zero % 16384) * 2 * math.pi / 16384,
        command_interval_ms=distribution(np.diff(ct) * 1000), actual_logged_lateness_ms=distribution(late * 1000),
        a2_interval_ms=distribution(np.diff(st) * 1000), actual_a2_rate_hz=(len(st)-1)/(st[-1]-st[0]),
        source_sample_age_at_logged_send_ms=distribution((ct - sample_time) * 1000),
        missing_reference_times_s=missing.tolist(), skipped_slots=skipped, windows=windows,
        maximum_c1_rounding_rad_s=float(max(abs(cv - sampled))),
        encoder_average_100ms=averaged, unchanged_encoder_intervals=plateaus,
        apparent_startup_lag_ms=float(lags[best] * 1000), shifted_startup_rms_rad_s=float(scores[best]),
        voltage_min_v=min(float(r["bus_voltage_v"]) for r in status),
        voltage_max_v=max(float(r["bus_voltage_v"]) for r in status),
        temperature_max_c=max(float(r["temperature_c"]) for r in status),
        maximum_speed_rad_s=float(max(abs(speed))), maximum_current_derived_torque_nm=float(max(abs(torque))),
        final_a3_displacement_rad=float(position[-1]), last_a3_time_s=float(pt[-1]),
        transmitted_velocity_integral_rad=float(integral(ct, cv, 10.)),
        final_displacement_minus_target_integral_rad=float(position[-1] - integral(ct, cv, pt[-1])),
        integrated_a2_velocity_rad=float(np.trapezoid(speed, st)),
    )
    for label, times, values, key in (("commands", ct, cv, "velocity_rad_s"), ("a2", st, speed, "velocity_rad_s"),
                                     ("a1", tt, torque, "torque_nm"), ("a3", pt, position, "position_rad")):
        output = [dict(time_s=float(t), **{key: float(y)}) for t, y in zip(times, values)]
        if label == "commands":
            for r, source_t in zip(output, sample_time):
                r["source_time_s"] = float(source_t)
        write_csv(HERE / f"{rate}hz_{label}.csv", output)
    write_csv(HERE / f"{rate}hz_encoder_average_100ms.csv", averages)
    write_csv(HERE / f"{rate}hz_encoder_plateaus.csv", plateaus)
    write_csv(HERE / f"{rate}hz_lag_scan.csv",
              [dict(lag_ms=float(a*1000), rms_rad_s=float(b)) for a, b in zip(lags, scores)])
    return dict(rate=rate, reference_hz=ref_hz, stats=stats, metadata=meta, command=(ct, cv), speed=(st, speed),
                torque=(tt, torque), position=(pt, position), averages=averages, late=late, sample_time=sample_time)


def decorate(ax, series, lo, hi, unit, origin=True):
    values = np.concatenate([y[(t >= lo) & (t <= hi)] for t, y in series])
    vmin, vmax = float(min(values)), float(max(values))
    margin = max(.001, .12 * (vmax-vmin))
    ax.set(xlim=(lo, hi), ylim=(vmin-margin, vmax+margin), ylabel=unit)
    ax.grid(alpha=.22)
    if lo == 0 and origin:
        ax.axvline(0, color="red", lw=1.2)
    ax.legend(loc="lower right", fontsize=8, framealpha=.94)
    ax.text(.012, .97, f"Plotted min/max: {vmin:.4f} / {vmax:.4f}", transform=ax.transAxes,
            va="top", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", alpha=.90))


def match_scales(axes):
    for row in axes:
        lo, hi = min(ax.get_ylim()[0] for ax in row), max(ax.get_ylim()[1] for ax in row)
        for ax in row:
            ax.set_ylim(lo, hi)


def comparison(sim, runs, lo, hi, title):
    fig, axes = plt.subplots(3, len(runs), figsize=(8.2 * len(runs), 10), squeeze=False, sharex="col")
    for column, run in enumerate(runs):
        ct, cv = run["command"]
        st, speed = run["speed"]
        tt, torque = run["torque"]
        pt, position = run["position"]
        tx, target = np.r_[ct, 10.], np.r_[cv, cv[-1]]
        ax = axes[0, column]
        ax.plot(sim["time_s"], sim["velocity_rad_s"], color=BLACK, lw=1, label="Simulation actual velocity (400 Hz)")
        ax.step(tx, target, where="post", color=ORANGE, lw=.95, label="Actually sent C1 velocity (held)")
        ax.plot(st, speed, color=BLUE, lw=.7, label="Real fresh A2 velocity")
        decorate(ax, [(sim["time_s"], sim["velocity_rad_s"]), (tx, target), (st, speed)], lo, hi, "Velocity (rad/s)")
        ax.set_title(f"{run['rate']} Hz C1 packets / {run['reference_hz']} Hz reference")
        ax = axes[1, column]
        ax.plot(tt, torque, color=GREEN, lw=.8, label="Real A1 current × B0 Kt")
        decorate(ax, [(tt, torque)], lo, hi, "Torque estimate (Nm)")
        ax = axes[2, column]
        it = np.unique(np.r_[0., ct, 10.])
        iq = integral(ct, cv, it)
        ax.plot(sim["time_s"], sim["position_rad"], color=GREY, ls="--", lw=1, label="Saved simulation displacement")
        ax.plot(it, iq, color=ORANGE, lw=1.2, label="Integral of actually sent C1 velocity")
        ax.plot(pt, position, color=BLUE, lw=1, label="Real A3 multi-turn displacement")
        decorate(ax, [(sim["time_s"], sim["position_rad"]), (it, iq), (pt, position)], lo, hi, "Displacement (rad)")
        ax.set_xlabel("Seconds since core replay / source spawn; no time shift")
    match_scales(axes)
    fig.suptitle(f"{title}\nSource: model_2500 v1 · DR · seed 42 · env 0, episode 1 | Airborne/free rear wheel")
    fig.tight_layout(rect=(0, 0, 1, .94))
    return fig


def averaged_figure(runs):
    fig, axes = plt.subplots(2, len(runs), figsize=(8.2 * len(runs), 8), squeeze=False)
    for column, run in enumerate(runs):
        part = [r for r in run["averages"] if r["start_s"] >= 2 and r["end_s"] < 10]
        t, actual, target = [np.array([r[k] for r in part]) for k in
                             ("time_s", "actual_velocity_rad_s", "target_velocity_rad_s")]
        ax = axes[0, column]
        ax.plot(t, target, color=ORANGE, label="Average of actually sent C1 velocity")
        ax.plot(t, actual, color=BLUE, lw=.9, marker=".", ms=3, label="Fresh A3 displacement / interval (~100 ms)")
        decorate(ax, [(t, target), (t, actual)], 2, 10, "Average speed (rad/s)")
        ax.set_title(f"{run['rate']} Hz packets / {run['reference_hz']} Hz reference")
        ax = axes[1, column]
        ax.plot(t, actual-target, color=BLUE, label="Encoder-average tracking error")
        ax.axhline(0, color=BLACK, lw=.6)
        decorate(ax, [(t, actual-target)], 2, 10, "Average error (rad/s)")
        ax.set_xlabel("Seconds since core start; supplementary averaged diagnostic")
    match_scales(axes)
    fig.suptitle("Small settling movements: A3 encoder averages, independent of the A2 velocity estimator")
    fig.tight_layout(rect=(0, 0, 1, .95))
    return fig


def plateau_figure(runs):
    fig, axes = plt.subplots(3, len(runs), figsize=(8.2 * len(runs), 10), squeeze=False)
    for column, run in enumerate(runs):
        p = run["stats"]["unchanged_encoder_intervals"][0]
        a, b = p["start_s"], p["end_s"]
        ct, cv = run["command"]
        st, speed = run["speed"]
        tt, torque = run["torque"]
        pt, position = run["position"]
        ax = axes[0, column]
        ax.step(ct-a, cv, where="post", color=ORANGE, label="Actually sent C1 velocity")
        ax.plot(st-a, speed, color=BLUE, lw=.8, label="Fresh A2 velocity estimate")
        decorate(ax, [(ct-a, cv), (st-a, speed)], 0, b-a, "Velocity (rad/s)", origin=False)
        ax.set_title(f"{run['rate']} Hz: unchanged A3 count; core t={a:.3f}–{b:.3f} s")
        ax = axes[1, column]
        ax.plot(tt-a, torque, color=GREEN, label="A1 current × B0 Kt")
        decorate(ax, [(tt-a, torque)], 0, b-a, "Torque estimate (Nm)", origin=False)
        ax = axes[2, column]
        t = np.unique(np.r_[a, ct[(ct>a)&(ct<b)], b])
        desired = integral(ct, cv, t)-integral(ct, cv, a)
        keep = (pt>=a)&(pt<=b)
        actual = position[keep]-position[keep][0]
        ax.plot(t-a, desired, color=ORANGE, label="Travel implied by sent velocity")
        ax.plot(pt[keep]-a, actual, color=BLUE, label="Actual A3 count change: zero")
        decorate(ax, [(t-a, desired), (pt[keep]-a, actual)], 0, b-a, "Local travel (rad)", origin=False)
        ax.set_xlabel("Seconds within measured plateau; local origin is not a new spawn")
    match_scales(axes)
    fig.suptitle("Low-speed tracking: longest measured unchanged-encoder interval after t=2 s")
    fig.tight_layout(rect=(0, 0, 1, .95))
    return fig


def timing_figure(run):
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    ct, cv = run["command"]
    st, speed = run["speed"]
    series = [(ct[1:], np.diff(ct)*1000), (ct, (ct-run["sample_time"])*1000),
              (st, speed-held(ct, cv, st))]
    labels = ["Actual host command interval", "Reference sample age at logged send", "A2 minus last sent C1 target"]
    units = ["Packet interval (ms)", "Sample age (ms)", "Velocity error (rad/s)"]
    for ax, (t, y), label, unit in zip(axes, series, labels, units):
        ax.plot(t, y, color=BLUE, lw=.8, label=label)
        for i, missing in enumerate(run["stats"]["missing_reference_times_s"]):
            ax.axvline(missing, color="red", lw=.7, alpha=.5,
                       label="Skipped source slot" if i == 0 else None)
        decorate(ax, [(t, y)], 0, 10, unit)
    axes[0].axhline(2.5, color=GREY, ls="--", lw=.7)
    axes[-1].set_xlabel("Seconds since core replay start; host timestamps, not motor acquisition times")
    fig.suptitle("400 Hz timing: six skipped slots; no fault or replay abort")
    fig.tight_layout(rect=(0, 0, 1, .95))
    return fig


def main():
    manifest_path = HERE / "protected_input_hashes.json"
    if not manifest_path.exists():
        shutil.copyfile("/tmp/c1_400hz_analysis_protected.json", manifest_path)
    protected = json.loads(manifest_path.read_text())
    assert all(sha(ROOT/p) == digest for p, digest in protected.items())
    native_rows = read(NATIVE)
    sim = {key: np.array([float(r[key]) for r in native_rows]) for key in native_rows[0]}
    assert len(native_rows) == 4001 and np.allclose(sim["time_s"], np.arange(4001)/400, atol=1e-12, rtol=0)
    assert sha(NATIVE) == sha(EXP/"replay_targets/2500_isaac_velocity_startup_400hz.csv")
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    runs = [load_run(rate, sim) for rate in STEMS]
    assert all(r["metadata"]["original_settings"] == runs[0]["metadata"]["original_settings"] for r in runs)
    summary = dict(
        model="model_2500.pt", task="Trackstand-Balance-v1", source_dr=True, seed=42, environment=0, episode=1,
        duration_s=10, fixture=FIXTURE, runs={str(r["rate"]): r["stats"] for r in runs},
        native_velocity_integral_rad=float(np.trapezoid(sim["velocity_rad_s"], sim["time_s"])),
        native_saved_joint_displacement_rad=float(sim["position_rad"][-1]),
        comparison_caveat="400 Hz changes reference sampling, packet rate, A2 query rate and physical starting phase.",
        hardware_or_simulation_run=False,
    )
    write_json(HERE/"statistics.json", summary)
    flat = [dict(packet_hz=r["rate"], reference_hz=r["reference_hz"], window=window, **values)
            for r in runs for window, values in r["stats"]["windows"].items()]
    write_csv(HERE/"summary.csv", flat)
    definitions = [
        ("replay_400hz", "400 Hz replay, full 10 seconds",
         lambda: comparison(sim, [runs[2]], 0, 10, "400 Hz C1 replay")),
        ("replay_400hz_startup", "400 Hz startup, 0–2 s", lambda: comparison(sim, [runs[2]], 0, 2, "400 Hz startup")),
        ("replay_400hz_settling", "400 Hz settling, 2–10 s",
         lambda: comparison(sim, [runs[2]], 2, 10, "400 Hz settling")),
        ("comparison_full", "50/200/400 Hz, full replay", lambda: comparison(sim, runs, 0, 10, "C1 replay comparison")),
        ("comparison_startup", "50/200/400 Hz startup", lambda: comparison(sim, runs, 0, 2, "Startup comparison")),
        ("comparison_settling", "50/200/400 Hz settling", lambda: comparison(sim, runs, 2, 10, "Settling comparison")),
        ("settling_encoder_100ms", "Supplementary encoder averages", lambda: averaged_figure(runs)),
        ("low_speed_count_plateaus", "Unchanged encoder counts", lambda: plateau_figure(runs)),
        ("400hz_timing", "400 Hz packet timing", lambda: timing_figure(runs[2])),
    ]
    with PdfPages(HERE/"all_figures.pdf") as combined:
        for name, _, build in definitions:
            fig = build()
            for suffix in ("png", "svg", "pdf"):
                fig.savefig(HERE/f"{name}.{suffix}", dpi=165)
            combined.savefig(fig)
            plt.close(fig)
    cards = []
    for name, title, _ in definitions:
        cards.append(f'<section><h2>{escape(title)}</h2><p><a href="{name}.png">PNG</a> · '
                     f'<a href="{name}.svg">SVG</a> · <a href="{name}.pdf">PDF</a></p>'
                     f'<img src="{name}.png" alt="{escape(title)}"></section>')
    (HERE/"index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>400 Hz C1 replay</title>'
        '<style>body{font:16px system-ui;max-width:1700px;margin:24px auto;padding:0 18px;color:#222}'
        'img{max-width:100%;height:auto}section{margin:36px 0}a{color:#1467a5}</style>'
        '<h1>Rear C1 replay — 50 / 200 / 400 Hz</h1>'
        '<p>Model 2500 v1 · source DR · airborne/free. The 400 Hz run also uses a denser reference.</p>'
        '<p><a href="README.md">Analysis</a> · <a href="summary.csv">Metrics CSV</a> · '
        '<a href="statistics.json">Exact statistics</a> · <a href="all_figures.pdf">All figures PDF</a></p>'
        + ''.join(cards) + '</html>')
    snapshot = HERE/"runtime_snapshot"
    snapshot.mkdir(exist_ok=True)
    for filename in ("drive_velocity_replay.py", "drive_velocity_sine.py", "record_drive_velocity.py"):
        shutil.copyfile(EXP/filename, snapshot/(filename+".txt"))
    assert all(sha(ROOT/p) == digest for p, digest in protected.items())
    print(json.dumps({str(r["rate"]): r["stats"] for r in runs}, indent=2))


if __name__ == "__main__":
    main()
