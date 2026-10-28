"""Read-only rear replay audit: decode raw CAN independently; no ROS/CAN/simulation."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import struct

os.environ.setdefault("MPLCONFIGDIR", "/tmp/trackstand_rear_diagnosis_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXPERIMENTS = ROOT / "deploy/experiments"
STEMS = (
    "20261026_163432_217938_scaled_50hz_drive_policy_replay",
    "20261026_163545_994664_scaled_200hz_drive_policy_replay",
)
RAD_PER_COUNT = 2 * math.pi / 16384


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def held(times, values, at):
    indexes = np.searchsorted(times, at, side="right") - 1
    assert np.all(indexes >= 0)
    return values[indexes]


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values) ** 2)))


def read_run(stem):
    path = EXPERIMENTS / "logs" / (stem + ".csv")
    meta = json.loads(path.with_suffix(".json").read_text())
    plan_path = path.with_name(stem + "_plan.csv")
    plan = read_csv(plan_path)
    rows = read_csv(path)
    args, rate = meta["arguments"], meta["packet_hz"]
    zero, kt = meta["encoder_zero_count"], meta["torque_constant_nm_per_a"]
    assert args["encoder_sign"] == args["torque_sign"] == -1
    assert meta["completed"] and meta["disable_confirmed"] and meta["watchdog_restored"]
    assert meta["final_fault_code"] == 0
    source = EXPERIMENTS / "replay_targets" / Path(args["targets"]).name
    assert sha(source) == meta["targets_sha256"]
    archive = EXPERIMENTS / "reports/20261026_policy_replay_hardware/runtime_snapshot"
    assert all(sha(archive / name) == digest for name, digest in meta["source_sha256"].items())
    source_rows = read_csv(source)
    source_core = [r for r in source_rows if 12.5 <= float(r["source_time_s"]) < 17.5]
    assert len(source_core) == 250 and len(plan) == 350
    assert all(float(r["requested_torque_nm"]) == 0 for r in plan[:100])
    assert all(float(p["requested_torque_nm"]) == float(s["scaled_drive_nm"])
               for p, s in zip(plan[100:], source_core))

    tx = [r for r in rows if r["record_type"] == "drive_target"]
    times = np.array([float(r["replay_time_s"]) for r in tx])
    assert len(tx) == len(plan) * (rate // 50) and np.all(np.diff(times) > 0)
    commands, payloads = [], []
    for i, row in enumerate(tx):
        payload = bytes.fromhex(row["payload_hex"])
        assert row["can_direction"] == "local_tx" and int(row["can_id_hex"], 16) == 0x101
        assert len(payload) == 5 and payload[0] == 0xC0
        code = struct.unpack("<i", payload[1:])[0]
        target = float(plan[i // (rate // 50)]["requested_torque_nm"])
        assert code == round(-target / kt * 1000)
        assert int(row["packet_index"]) == i and int(row["target_index"]) == i // (rate // 50)
        assert float(row["encoded_command_torque_nm"]) == -code * .001 * kt
        commands.append(-code * .001 * kt)
        payloads.append(payload.hex())
    commands = np.array(commands)
    assert all(int(r["can_id_hex"], 16) == 0x101 for r in rows if r["can_direction"] == "local_tx")
    writes = [bytes.fromhex(r["payload_hex"]) for r in rows
              if r["can_direction"] == "local_tx" and int(r["can_length"]) > 1]
    assert {p[0] for p in writes} == {0xC0, 0xCD}  # No origin/gain/slew/limit writes.

    series = {}
    for op, key in ((0xA1, "torque_nm"), (0xC0, "torque_nm"), (0xA2, "velocity_rad_s"),
                    (0xA3, "position_unwrapped_rad"), (0xAE, "bus_voltage_v")):
        selected = [r for r in rows if r["phase"] == "motion" and r["can_direction"] == "bus_rx"
                    and r["can_opcode"] == str(op)]
        t, values = [], []
        for row in selected:
            data = bytes.fromhex(row["payload_hex"])
            if op in (0xA1, 0xC0):
                value = -struct.unpack("<i", data[1:])[0] * .001 * kt
            elif op == 0xA2:
                value = -struct.unpack("<i", data[1:])[0] * .01 * 2 * math.pi / 60
            elif op == 0xA3:
                value = -(struct.unpack("<i", data[3:])[0] - zero) * RAD_PER_COUNT
            else:
                value = struct.unpack("<H", data[1:3])[0] * .01
                assert data[6] == 2 and data[7] == 0
            assert math.isclose(value, float(row[key]), rel_tol=0, abs_tol=1e-12)
            t.append(float(row["replay_time_s"]))
            values.append(value)
        series[op] = np.array(t), np.array(values)
        assert np.all(np.diff(series[op][0]) > 0)

    a1t, a1 = series[0xA1]
    vt, v = series[0xA2]
    qt, q = series[0xA3]
    end = min(float(r["replay_time_s"]) for r in rows
              if r["phase"] == "shutdown" and r["can_direction"] == "local_tx")
    integral_v = np.r_[0, np.cumsum((v[1:] + v[:-1]) * .5 * np.diff(vt))]
    error = a1[a1t >= 2] - held(times, commands, a1t[a1t >= 2])
    first_move = np.flatnonzero(abs(q) > .002)[0]
    stats = dict(
        stem=stem, packet_hz=rate, packets=len(tx), zero_count=zero,
        initial_native_angle_deg=(zero % 16384) * 360 / 16384,
        last_motion_count=int(round(zero - q[-1] / RAD_PER_COUNT)),
        encoder_displacement_rad=float(q[-1]), integrated_A2_displacement_rad=float(integral_v[-1]),
        A2_integral_vs_A3_RMS_rad=rms(np.interp(vt, qt, q) - q[0] - integral_v),
        speed_min_rad_s=float(v.min()), A1_core_RMS_error_Nm=rms(error),
        A1_core_bias_Nm=float(error.mean()), A1_core_mean_Nm=float(a1[a1t >= 2].mean()),
        first_A3_displacement_over_0_002rad_s=float(qt[first_move]),
        first_movement_after_core_ms=float((qt[first_move] - times[100 * (rate // 50)]) * 1000),
        prezero_A3_minmax_rad=[float(q[qt < 2].min()), float(q[qt < 2].max())],
        prezero_A2_minmax_rad_s=[float(v[vt < 2].min()), float(v[vt < 2].max())],
        final_prezero_A3_rad=float(q[qt < 2][-1]),
        voltage_minmax_v=[float(series[0xAE][1].min()), float(series[0xAE][1].max())],
        temperature_minmax_c=[min(float(r["motor_temperature_c"]) for r in rows
                                 if r["phase"] == "motion" and r["motor_temperature_c"]),
                              max(float(r["motor_temperature_c"]) for r in rows
                                  if r["phase"] == "motion" and r["motor_temperature_c"])],
        signed_command_impulse_Nm_s=float(np.sum(commands * np.diff(np.r_[times, end]))),
        achieved_packet_hz=float((len(times) - 1) / (times[-1] - times[0])),
        duration_before_shutdown_s=end, B4_raw=meta["B4_current_slew_raw"],
        B3_A=meta["B3_position_speed_limit_a"], firmware_Kt_Nm_A=kt,
        query_results=sorted({r["query_status"] for r in rows if r["query_status"]}),
        windows={}, C0_reply_errors_by_copy={},
    )
    for lo, hi in ((2., 2.2), (2., 3.44), (4.16, 4.86), (5., 7.)):
        mask, vmask, qmask = (a1t >= lo) & (a1t < hi), (vt >= lo) & (vt < hi), (qt >= lo) & (qt < hi)
        stats["windows"][f"{lo:g}-{hi:g}"] = dict(
            A1_mean_Nm=float(a1[mask].mean()), A1_min_Nm=float(a1[mask].min()),
            A2_minmax_rad_s=[float(v[vmask].min()), float(v[vmask].max())],
            A3_minmax_rad=[float(q[qmask].min()), float(q[qmask].max())],
        )
    ct, current = series[0xC0]
    indexes = np.searchsorted(times, ct, side="right") - 1
    for repeat in range(rate // 50):
        mask = (ct >= 2) & (indexes % (rate // 50) == repeat)
        stats["C0_reply_errors_by_copy"][repeat + 1] = rms(current[mask] - commands[indexes[mask]])
    input_hashes = {str(p.relative_to(ROOT)): sha(p) for p in (path, path.with_suffix(".json"), plan_path, source)}
    return dict(stats=stats, series=series, times=times, commands=commands, payloads=payloads,
                target_times=times[::rate // 50], input_hashes=input_hashes)


def plot(runs, name, limits):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True, sharey="row", constrained_layout=True)
    for col, run in enumerate(runs):
        series, t, command = run["series"], run["times"], run["commands"]
        axes[0, col].step(t, command, where="post", color="black", lw=1.2, label="Sent C0 torque")
        axes[0, col].plot(*series[0xC0], color="#89a8cc", lw=.6, alpha=.55, label="C0 current × Kt")
        axes[0, col].plot(*series[0xA1], color="#00935e", lw=.9, label="A1 current × Kt")
        axes[1, col].plot(*series[0xA2], color="#4472c4", lw=1.1, label="Fresh A2 speed")
        axes[2, col].plot(*series[0xA3], color="#be5714", lw=1.2, label="Fresh A3 multi-turn displacement")
        for row, (op, unit) in enumerate(((0xA1, "Current-derived torque (Nm)"),
                                         (0xA2, "Wheel speed (rad/s)"), (0xA3, "Startup-relative angle (rad)"))):
            ax = axes[row, col]
            ax.set_xlim(*limits)
            ax.axvline(2, color="red", lw=.8, ls="--")
            if limits[0] < 2:
                ax.axvspan(0, 2, color="gray", alpha=.10)
            ax.grid(alpha=.2)
            ax.legend(loc="lower right", fontsize=8)
            tx, y = series[op]
            y = y[(tx >= limits[0]) & (tx <= limits[1])]
            ax.text(.02, .95, f"Feedback min {y.min():.4f}  max {y.max():.4f}",
                    transform=ax.transAxes, va="top", fontsize=8)
            if col == 0:
                ax.set_ylabel(unit)
        axes[0, col].set_title(f"{run['stats']['packet_hz']} Hz C0 — native start "
                               f"{run['stats']['initial_native_angle_deg']:.2f}°")
        axes[2, col].set_xlabel("Actual seconds after first C0 send; core begins at 2 s")
    for row, ops in enumerate(((0xA1, 0xC0), (0xA2,), (0xA3,))):
        visible = []
        for run in runs:
            channels = [run["series"][op] for op in ops]
            if row == 0:
                channels.append((run["times"], run["commands"]))
            for t, y in channels:
                visible.extend(y[(t >= limits[0]) & (t <= limits[1])])
        lo, hi = min(visible), max(visible)
        margin = max((hi - lo) * .08, .001)
        axes[row, 0].set_ylim(lo - margin, hi + margin)
    fig.suptitle("Same rear torque commands, different motion\n"
                 "Actual timestamps; no time shift or bias correction; current-derived torque is not shaft torque")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=180)
    plt.close(fig)


def main():
    runs = [read_run(stem) for stem in STEMS]
    a, b = runs
    assert a["payloads"] == b["payloads"][::4]
    assert all(len(set(b["payloads"][i:i + 4])) == 1 for i in range(0, len(b["payloads"]), 4))
    edges = np.unique(np.r_[a["times"], b["times"], 2., 6.98])
    edges = edges[(edges >= 2) & (edges <= 6.98)]
    diff = held(a["times"], a["commands"], edges[:-1]) - held(b["times"], b["commands"], edges[:-1])
    widths = np.diff(edges)
    comparison = dict(
        common_target_onset_max_difference_ms=float(abs(a["target_times"] - b["target_times"]).max() * 1000),
        actual_command_difference_L1_Nm_s=float(np.sum(abs(diff) * widths)),
        actual_command_difference_RMS_Nm=float(np.sqrt(np.sum(diff ** 2 * widths) / widths.sum())),
        command_comparison_window_s=[2, 6.98],
        displacement_magnitude_ratio=abs(a["stats"]["encoder_displacement_rad"]
                                         / b["stats"]["encoder_displacement_rad"]),
        second_start_count_equals_first_end=a["stats"]["last_motion_count"] == b["stats"]["zero_count"],
        native_start_phase_difference_shortest_deg=((b["stats"]["zero_count"] - a["stats"]["zero_count"])
                                                   * 360 / 16384 + 180) % 360 - 180,
    )
    summary = {"runs": [r["stats"] for r in runs], "comparison": comparison,
               "input_hashes": {p: d for r in runs for p, d in r["input_hashes"].items()},
               "interpretation": "Two confounded runs. A2 integration is a cross-check, not independent metrology. "
                                 "Motor acquisition timing and physical shaft torque are unknown. "
                                 "No hardware, model fitting or simulation executed."}
    (HERE / "statistics.json").write_text(json.dumps(summary, indent=2) + "\n")
    simple = [{k: v for k, v in r["stats"].items() if not isinstance(v, (dict, list))} for r in runs]
    with (HERE / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(simple[0]))
        writer.writeheader()
        writer.writerows(simple)
    plot(runs, "rear_response", (0, 7))
    plot(runs, "first_200ms", (1.98, 2.22))
    (HERE / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Rear replay diagnosis</title>'
        '<style>body{max-width:1400px;margin:24px auto;font-family:sans-serif}img{width:100%}</style>'
        '<h1>Rear replay diagnosis — 2026-10-28</h1><p><a href="README.md">Report</a> · '
        '<a href="statistics.json">Statistics</a> · <a href="summary.csv">Summary CSV</a></p>'
        '<p>Red line: beginning of recorded torque sequence. Both runs start at rest, but different physical angles. '
        'All signals retain actual timestamps; no new motor operation.</p>'
        '<h2>Complete replay</h2><img src="rear_response.png" alt="Rear torque, speed and angle">'
        '<h2>First torque burst</h2><img src="first_200ms.png" alt="First 200 milliseconds of core">\n'
    )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
