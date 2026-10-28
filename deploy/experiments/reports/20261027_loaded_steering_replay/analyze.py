"""Compare recorded loaded/airborne steering replays without CAN or simulation execution."""

import argparse
from collections import Counter
import csv
import html
import io
import json
from pathlib import Path
import struct

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXPERIMENTS = ROOT / "deploy/experiments"
BASELINE = EXPERIMENTS / "reports/20261027_steering_initial_transient_hardware"

# Reuse the existing report's pure reader, windowing and metric functions.
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("loaded_replay_baseline", BASELINE / "analyze.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
base.HERE = HERE  # Write new aligned tables here; never regenerate the airborne report.
helpers, runtime, plt = base.old, base.runtime, base.plt
STEMS = (*base.STEMS, "20261027_190254_424315_scaled_200hz_policy_replay")
LABELS = ("Airborne 1", "Airborne 2", "Ground-loaded")
COLORS = ("tab:blue", "tab:orange", "tab:red")
FIGURES = ("tracking_comparison", "startup_comparison", "holding_load", "timing", "source_aligned_comparison")


def validate(run):
    """Reconstruct plans/packets and independently decode raw measurements."""
    meta = run.meta
    args = argparse.Namespace(**meta["arguments"])
    args.targets = EXPERIMENTS / "replay_targets" / Path(args.targets).name
    for name, key in (("steering_policy_replay.py", "script_sha256"),
                      ("steering_sine_mit_io.py", "transport_sha256")):
        assert helpers.digest(BASELINE / "runtime_snapshot" / name) == meta[key]
    if hasattr(args, "entry_point"):
        assert helpers.digest(BASELINE / "runtime_snapshot" / args.entry_point) == args.entry_point_sha256
    assert meta["completed"] and meta["disable_confirmed"] and meta["final_fault_code"] == 0
    assert (args.start, args.duration, args.ramp, args.pre_hold, args.post_hold) == (0., 5., 0., 0., 1.)
    assert (args.packet_rate, args.query_rate, args.current_abort_a, args.max_lateness_ms) == (200, 100., 3., 20.)
    assert (args.kp, args.kd, args.sign, args.torque_sign, args.motor_id) == (10., .2, -1, -1, 2)
    assert meta["encoder_zero_count"] == 0 and meta["steering_origin_setup"]["verified"]
    assert meta["steering_origin_setup"]["F1_position_code"] == 32767
    assert not meta["rear_motor_commanded"] and meta["command_schedule"] == "send_relative_no_catchup"
    plan, digest = runtime.load_plan(args)
    assert digest == meta["targets_sha256"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=runtime.PLAN_FIELDS)
    writer.writeheader()
    writer.writerows(plan)
    assert stream.getvalue().encode() == run.path.with_name(run.stem + "_plan.csv").read_bytes()
    assert len(plan) == 300 and len(run.tx) == 1200
    motor = runtime.mit.Motor("steering", 2, 1., -1, -1, zero_count=0,
                              zero_rad=meta["native_zero_rad"], ranges=tuple(meta["mit_ranges"]))
    for index, row in enumerate(run.tx):
        assert int(row["packet_index"]) == index and int(row["target_index"]) == index // 4
        assert row["skipped_command_ticks"] == "0"
        assert float(row["requested_position_rad"]) == plan[index // 4]["target_rad"]
        assert bytes.fromhex(row["mit_payload_hex"]) == runtime.payload(plan[index // 4]["target_rad"], motor, args)
        encoded = int.from_bytes(bytes.fromhex(row["mit_payload_hex"])[:2], "big") * 6 / 65535 - 3
        assert abs(float(row["mit_target_position_rad"]) + encoded - meta["native_zero_rad"]) < 1e-12
    assert np.all(np.diff(run.ct) >= .005)
    decoded = Counter()
    for row in run.rows:
        if row["can_direction"] != "bus_rx":
            continue
        data = bytes.fromhex(row["mit_payload_hex"])
        assert int(row["mit_can_id"]) == 2 and len(data) == int(row["can_length"])
        decoded[f"{data[0]:02X}"] += 1
        if row["phase"] != "motion":
            continue
        if data[0] == 0xA3:
            expected = -struct.unpack("<i", data[3:7])[0] * helpers.RAD_PER_COUNT
            assert abs(float(row["position_unwrapped_rad"]) - expected) < 1e-12
        elif data[0] == 0xA2:
            expected = -struct.unpack("<i", data[1:5])[0] * .01 * 2 * np.pi / 60
            assert abs(float(row["velocity_rad_s"]) - expected) < 1e-12
        elif data[0] == 0xA1:
            current = struct.unpack("<i", data[1:5])[0] * .001
            assert abs(float(row["current_a"]) - current) < 1e-12
            assert abs(float(row["torque_nm"]) + current * meta["torque_constant_nm_per_a"]) < 1e-12
        elif data[0] == 0xF1:
            position_code = int.from_bytes(data[1:3], "big")
            torque_code = ((data[4] & 15) << 8) | data[5]
            position = -(position_code * 6 / 65535 - 3 - meta["native_zero_rad"])
            assert position_code not in (0, 65535) and data[6] == 1
            assert abs(float(row["mit_feedback_position_rad"]) - position) < 1e-12
            assert abs(float(row["mit_feedback_torque_nm"]) + torque_code * 36 / 4095 - 18) < 1e-12
        elif data[0] == 0xAE:
            assert data[6] == 5 and data[7] == 0
    assert sum(r["can_opcode"] == "177" and r["can_direction"] == "local_tx" for r in run.rows) == 1
    return dict(raw_reply_counts=dict(decoded), exact_plan_and_packets=True,
                zero_signs_units_status_verified=True, recorded_runtime_hashes_verified=True)


def extra_metrics(run, result):
    result["sampled_torque_by_window_nm"] = {}
    for name, window in result["actual_windows_s"].items():
        channels = {}
        for key, opcode in (("torque_nm", 0xA1), ("mit_feedback_torque_nm", 0xF1)):
            time, values = run.series(key, opcode)
            _, values = run.select(time, values, window)
            channels[key] = helpers.describe(values)
            channels[key]["mean_absolute"] = float(np.mean(abs(values)))
        result["sampled_torque_by_window_nm"][name] = channels
    result["sent_packets"] = len(run.tx)
    result["maximum_F1_position_rad"] = float(run.fq.max())
    result["maximum_A3_position_rad"] = float(run.q.max())
    result["last_source_target_rad"] = float(run.request[996])
    return result


def save(fig, name):
    for extension in ("png", "svg", "pdf"):
        fig.savefig(HERE / f"{name}.{extension}", dpi=170)
    plt.close(fig)


def panel(ax, window, units):
    base.panel(ax, window, units)


def plots(runs):
    fig, axes = plt.subplots(4, 3, figsize=(20, 14), layout="constrained")
    for col, run in enumerate(runs):
        full = (0., 6.35)
        later = (run.ct[200], run.ct[1000])
        for row, window in ((0, full), (1, later)):
            base.target(axes[row, col], run)
            axes[row, col].plot(run.qtime, run.q, color="tab:blue", label="A3 position")
            axes[row, col].plot(run.ft, run.fq, color="tab:orange", alpha=.6, lw=.6, label="F1 position")
            panel(axes[row, col], window, "Steering position (rad)")
        axes[2, col].plot(run.qtime, np.degrees(run.q - helpers.held(run.ct, run.cv, run.qtime)),
                          color="tab:blue", label="A3 minus latest sent target")
        panel(axes[2, col], full, "Unshifted position error (deg)")
        axes[3, col].plot(run.vtime, run.v, color="tab:green", label="A2 speed")
        panel(axes[3, col], full, "Steering speed (rad/s)")
        for row in range(4):
            axes[row, col].set_title(f"{LABELS[col]} | actual motion {run.meta['actual_motion_duration_s']:.3f} s")
    fig.suptitle("Same steering source 0-5 s plus final hold | MIT Kp=10 / Kd=0.2, nominal 200 Hz\n"
                 "Actual send/RX timestamps; no time shift, mean subtraction or smoothing")
    save(fig, FIGURES[0])

    fig, axes = plt.subplots(4, 3, figsize=(20, 13), layout="constrained")
    for col, run in enumerate(runs):
        base.target(axes[0, col], run)
        axes[0, col].plot(run.qtime, run.q, "o-", ms=2.5, label="A3 position")
        axes[0, col].plot(run.ft, run.fq, ".-", ms=3, label="F1 position")
        panel(axes[0, col], (0., .3), "Position (rad)")
        for time, values, channel in ((run.qtime, run.q, "A3"), (run.ft, run.fq, "F1")):
            axes[1, col].plot(time, np.degrees(values - helpers.held(run.ct, run.cv, time)),
                              ".-", ms=3, label=f"{channel} minus sent target")
        panel(axes[1, col], (0., .3), "Position error (deg)")
        axes[2, col].plot(run.vtime, run.v, "o-", ms=3, label="A2 speed")
        panel(axes[2, col], (0., .3), "Speed (rad/s)")
        ft, torque = run.series("mit_feedback_torque_nm", 0xF1)
        axes[3, col].plot(run.ttime, run.torque, "o-", ms=3, label="A1 current x measured Kt")
        axes[3, col].plot(ft, torque, ".-", ms=3, label="F1 firmware-reported torque")
        axes[3, col].axhline(3 * run.meta["torque_constant_nm_per_a"], ls="--", color="red",
                             label="3 A x Kt (A1-only trip)")
        panel(axes[3, col], (0., .3), "Torque report (Nm)")
        for row in range(4):
            axes[row, col].set_title(LABELS[col])
    fig.suptitle("Initial transient, first 300 ms | ground-loaded versus two airborne runs\n"
                 "F1 and A1 are asynchronous firmware reports, not independent shaft torque measurements")
    save(fig, FIGURES[1])

    fig, axes = plt.subplots(3, 3, figsize=(20, 10), layout="constrained")
    for col, run in enumerate(runs):
        end = run.meta["actual_motion_duration_s"]
        window = (end - 1., end)
        base.target(axes[0, col], run)
        axes[0, col].plot(run.qtime, run.q, label="A3 position")
        axes[0, col].plot(run.ft, run.fq, lw=.7, alpha=.7, label="F1 position")
        panel(axes[0, col], window, "Position (rad)")
        axes[1, col].plot(run.qtime, np.degrees(run.q - helpers.held(run.ct, run.cv, run.qtime)),
                          label="A3 minus sent target")
        panel(axes[1, col], window, "Holding error (deg)")
        ft, torque = run.series("mit_feedback_torque_nm", 0xF1)
        axes[2, col].plot(run.ttime, run.torque, label="A1 current x Kt")
        axes[2, col].plot(ft, torque, alpha=.65, label="F1 torque")
        panel(axes[2, col], window, "Torque report (Nm)")
        for row in range(3):
            axes[row, col].set_title(LABELS[col])
    # Matching scales expose the persistent load without exaggerating tiny sensor steps.
    for row in range(3):
        lo = min(ax.get_ylim()[0] for ax in axes[row])
        hi = max(ax.get_ylim()[1] for ax in axes[row])
        for ax in axes[row]:
            ax.set_ylim(lo, hi)
    fig.suptitle("Final one second of actual motion | same held steering target\n"
                 "Ground-loaded hold has a small position deficit and sustained opposing-load torque")
    save(fig, FIGURES[2])

    fig, axes = plt.subplots(2, 3, figsize=(20, 7), layout="constrained")
    for col, run in enumerate(runs):
        axes[0, col].plot(run.ct[1:], np.diff(run.ct) * 1000, lw=.7, label="Actual send interval")
        axes[0, col].axhline(5, ls="--", color="black", label="Nominal 5 ms")
        panel(axes[0, col], (0., 6.35), "Packet interval (ms)")
        axes[1, col].plot(run.ct, [float(r["command_lateness_s"]) * 1000 for r in run.tx],
                          lw=.7, label="Lateness relative to next deadline")
        panel(axes[1, col], (0., 6.35), "Lateness (ms)")
        for row in range(2):
            axes[row, col].set_title(LABELS[col])
    fig.suptitle("Identical packets, slightly different timing | all 1,200 packets retained in each run\n"
                 "Send-relative scheduling stretches six nominal seconds to 6.263-6.314 actual seconds")
    save(fig, FIGURES[3])

    fig, axes = plt.subplots(3, 1, figsize=(16, 11), layout="constrained")
    for col, run in enumerate(runs):
        plan_time = np.array([float(r["target_time_s"]) for r in run.tx[::4]])
        # Only this diagnostic remaps elapsed time, without shifting motor response relative to input.
        qnom = np.interp(run.qtime, run.ct[::4], plan_time)
        vnom = np.interp(run.vtime, run.ct[::4], plan_time)
        q = np.interp(run.ct[::4], run.qtime, run.q)
        axes[0].plot(plan_time, q, color=COLORS[col], lw=1, label=LABELS[col])
        axes[1].plot(qnom, np.degrees(run.q - helpers.held(run.ct, run.cv, run.qtime)),
                      color=COLORS[col], lw=.8, label=LABELS[col])
        axes[2].plot(vnom, run.v, color=COLORS[col], lw=.8, label=LABELS[col])
    axes[0].step([float(r["target_time_s"]) for r in runs[0].plan],
                 [float(r["target_rad"]) for r in runs[0].plan], where="post", color="black",
                 lw=.8, alpha=.7, label="Identical target plan")
    for row, (window, units) in enumerate((((0, 5), "Position (rad)"),
                                          ((1, 5), "Unshifted error (deg)"), ((0, .3), "Speed (rad/s)"))):
        panel(axes[row], window, units)
        axes[row].set_xlabel("Nominal plan time (s); diagnostic time remapping only")
    fig.suptitle("Airborne/loaded waveform comparison | same targets, similar observed responses\n"
                 "Only this figure remaps actual command times to nominal plan coordinates; metrics use fresh samples")
    save(fig, FIGURES[4])


def main():
    runs = [helpers.Run(stem) for stem in STEMS]
    results = {run.stem: extra_metrics(run, base.metrics(run, validate(run))) for run in runs}
    assert all(run.plan == runs[0].plan for run in runs)
    assert all([r["mit_payload_hex"] for r in run.tx] == [r["mit_payload_hex"] for r in runs[0].tx] for run in runs)
    assert runs[-1].meta["arguments"]["fixture"] == "supported-loaded"
    inputs = [p for run in runs for p in (run.path, run.path.with_suffix(".json"),
                                         run.path.with_name(run.stem + "_plan.csv"))]
    dependencies = [Path(__file__), BASELINE / "analyze.py",
                    EXPERIMENTS / "reports/20261026_policy_replay_hardware/analyze.py",
                    *sorted((BASELINE / "runtime_snapshot").glob("*.py"))]
    output = dict(runs=results, labels=dict(zip(STEMS, LABELS)), identical_plans_and_payloads=True,
                  setup="User confirms front tyre carries normal weight on ground; frame supported upright; "
                        "steering free. Longitudinal freedom/support details and exact normal force not measured.",
                  airborne_setup="Previously confirmed airborne/free, straight startup and unchanged settings.",
                  input_sha256={str(p.relative_to(ROOT)): helpers.digest(p) for p in inputs},
                  analysis_dependencies_sha256={str(p.relative_to(ROOT)): helpers.digest(p) for p in dependencies})
    (HERE / "statistics.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    with (HERE / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("run", "stem", "completed", "packets", "actual_duration_s", "actual_packet_hz",
                         "actual_source_target_hz", "A3_core_RMS_deg", "A3_first_source_second_RMS_deg",
                         "A3_source_1_to_5_RMS_deg", "A3_source_1_to_5_p95_abs_deg", "F1_core_max_abs_error_deg",
                         "startup_alignment_ms", "later_alignment_ms", "peak_A2_speed_rad_s", "peak_A1_current_A",
                         "peak_F1_torque_Nm", "peak_F1_equivalent_current_A", "last_half_second_A3_bias_deg",
                         "last_half_second_A1_mean_torque_Nm", "last_half_second_F1_mean_torque_Nm"))
        for label, run in zip(LABELS, runs):
            r = results[run.stem]
            a3 = r["position_errors_rad"]["A3"]
            hold = r["sampled_torque_by_window_nm"]["final_actual_half_second"]
            writer.writerow((label, run.stem, run.meta["completed"], len(run.tx),
                             run.meta["actual_motion_duration_s"], r["actual_packet_hz"], r["actual_core_target_hz"],
                             np.degrees(a3["entire_source_0_to_5_s"]["rms"]),
                             np.degrees(a3["first_source_0_to_1_s"]["rms"]),
                             np.degrees(a3["source_1_to_5_s"]["rms"]),
                             np.degrees(a3["source_1_to_5_s"]["p95_abs"]),
                             np.degrees(r["position_errors_rad"]["F1"]["entire_source_0_to_5_s"]["maximum_abs"]),
                             r["alignment"]["startup_actual_point1_to_point6_s"]["best_shift_ms"],
                             r["alignment"]["source_1_to_5_s"]["best_shift_ms"],
                             float(abs(run.v).max()), r["A1_current_peak_abs_a"],
                             float(abs(run.series("mit_feedback_torque_nm", 0xF1)[1]).max()),
                             r["F1_peak_equivalent_current_abs_a"],
                             np.degrees(a3["final_actual_half_second"]["bias"]),
                             hold["torque_nm"]["mean"], hold["mit_feedback_torque_nm"]["mean"] ))
    plots(runs)
    gallery = "".join(f'<section><h2>{html.escape(name.replace("_", " ").title())}</h2>'
                      f'<a href="{name}.png"><img src="{name}.png" alt="{name}"></a>'
                      f'<p><a href="{name}.svg">SVG</a> · <a href="{name}.pdf">PDF</a></p></section>'
                      for name in FIGURES)
    (HERE / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Loaded steering replay comparison</title>'
        '<style>body{font:16px system-ui;max-width:1500px;margin:2em auto;padding:0 1em}img{width:100%}'
        'section{margin:3em 0}a{color:#1458a0}</style><h1>Loaded versus airborne steering replay</h1>'
        '<p>2026-10-27 · same initial source 0–5 s + final hold · MIT Kp=10/Kd=0.2, nominal 200 Hz.</p>'
        '<p>Loaded tracking remains close to airborne: approximately 0.60° RMS after source second 1 and '
        '~20 ms descriptive lag. Final hold shows a small persistent position deficit and increased torque.</p>'
        '<p><a href="README.md">Report</a> · <a href="summary.csv">Summary CSV</a> · '
        '<a href="statistics.json">Exact statistics</a></p>' + gallery + '</html>')
    print((HERE / "summary.csv").read_text())


if __name__ == "__main__":
    main()
