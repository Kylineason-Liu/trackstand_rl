"""Analyze two physical startup replays; fresh raw feedback only, no CAN or simulation execution."""

import argparse
from collections import Counter
import csv
import html
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXPERIMENTS = ROOT / "deploy/experiments"
STEMS = ("20261027_181229_866490_scaled_200hz_steering_initial_transient",
         "20261027_181349_675883_scaled_200hz_steering_initial_transient")
FIGURES = ("tracking", "startup_detail", "current_sampling", "host_timing", "repeatability")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


old = module("startup_report_helpers", EXPERIMENTS / "reports/20261026_policy_replay_hardware/analyze.py")
plt = old.plt
sys.path.insert(0, str(HERE / "runtime_snapshot"))
runtime = module("startup_recorded_runtime", HERE / "runtime_snapshot/steering_policy_replay.py")
sys.path.pop(0)


def validate(run):
    meta = run.meta
    args = argparse.Namespace(**meta["arguments"])
    args.targets = EXPERIMENTS / "replay_targets" / Path(args.targets).name
    for name, digest in (("steering_policy_replay.py", meta["script_sha256"]),
                         ("steering_sine_mit_io.py", meta["transport_sha256"]),
                         (args.entry_point, args.entry_point_sha256)):
        assert old.digest(HERE / "runtime_snapshot" / name) == digest
    assert meta["completed"] and meta["disable_confirmed"] and meta["final_fault_code"] == 0
    assert (args.start, args.duration, args.ramp, args.pre_hold, args.post_hold) == (0., 5., 0., 0., 1.)
    assert (args.packet_rate, args.query_rate, args.current_abort_a, args.max_lateness_ms) == (200, 100., 3., 20.)
    assert args.kp == 10 and args.kd == .2 and args.sign == args.torque_sign == -1
    assert meta["encoder_zero_count"] == 0 and meta["steering_origin_setup"]["verified"]
    assert meta["steering_origin_setup"]["F1_position_code"] == 32767
    assert not meta["rear_motor_commanded"] and meta["command_schedule"] == "send_relative_no_catchup"
    plan, digest = runtime.load_plan(args)
    assert digest == meta["targets_sha256"]
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=runtime.PLAN_FIELDS)
    writer.writeheader()
    writer.writerows(plan)
    assert text.getvalue().encode() == run.path.with_name(run.stem + "_plan.csv").read_bytes()
    assert len(plan) == 300 and len(run.tx) == 1200
    assert [row["phase"] for row in plan] == ["core"] * 250 + ["post_hold"] * 50
    motor = runtime.mit.Motor("steering", 2, 1., -1, -1, zero_count=0,
                              zero_rad=meta["native_zero_rad"], ranges=tuple(meta["mit_ranges"]))
    for i, row in enumerate(run.tx):
        assert int(row["packet_index"]) == i and int(row["target_index"]) == i // 4
        assert row["skipped_command_ticks"] == "0"
        assert float(row["requested_position_rad"]) == plan[i // 4]["target_rad"]
        assert bytes.fromhex(row["mit_payload_hex"]) == runtime.payload(plan[i // 4]["target_rad"], motor, args)
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
            count = struct.unpack("<i", data[3:7])[0]
            assert abs(float(row["position_unwrapped_rad"]) + count * old.RAD_PER_COUNT) < 1e-12
        elif data[0] == 0xA2:
            speed = -struct.unpack("<i", data[1:5])[0] * .01 * 2 * np.pi / 60
            assert abs(float(row["velocity_rad_s"]) - speed) < 1e-12
        elif data[0] == 0xA1:
            current = struct.unpack("<i", data[1:5])[0] * .001
            assert abs(float(row["current_a"]) - current) < 1e-12
            assert abs(float(row["torque_nm"]) + current * meta["torque_constant_nm_per_a"]) < 1e-12
        elif data[0] == 0xF1:
            pos = int.from_bytes(data[1:3], "big")
            torque_code = ((data[4] & 15) << 8) | data[5]
            torque = -(torque_code * 36 / 4095 - 18)
            position = -(pos * 6 / 65535 - 3 - meta["native_zero_rad"])
            assert pos not in (0, 65535) and data[6] == 1
            assert abs(float(row["mit_feedback_torque_nm"]) - torque) < 1e-12
            assert abs(float(row["mit_feedback_position_rad"]) - position) < 1e-12
        elif data[0] == 0xAE:
            assert data[6] == 5 and data[7] == 0
    assert sum(row["can_opcode"] == "177" and row["can_direction"] == "local_tx" for row in run.rows) == 1
    return dict(raw_reply_counts=dict(decoded), exact_plan_and_packets=True, zero_signs_units_status_verified=True)


def errors(run, window, channel="A3"):
    t, q = run.select(run.qtime if channel == "A3" else run.ft,
                      run.q if channel == "A3" else run.fq, window)
    return old.error_stats(q, old.held(run.ct, run.cv, t))


def metrics(run, validation):
    start, end_core = run.ct[0], run.ct[1000]
    end = run.meta["actual_motion_duration_s"]
    windows = dict(first_source_0_to_point2_s=(start, run.ct[40]),
                   first_source_0_to_1_s=(start, run.ct[200]),
                   source_1_to_5_s=(run.ct[200], end_core),
                   entire_source_0_to_5_s=(start, end_core), final_actual_half_second=(end - .5, end))
    output = dict(metadata=run.meta, validation=validation, measurement_rows=len(run.rows),
                  actual_windows_s={k: list(v) for k, v in windows.items()},
                  position_errors_rad={channel: {k: errors(run, w, channel) for k, w in windows.items()}
                                       for channel in ("A3", "F1")},
                  actual_packet_hz=(len(run.ct) - 1) / (run.ct[-1] - run.ct[0]),
                  actual_core_target_hz=249 / (run.ct[996] - run.ct[0]),
                  packet_interval_ms=old.describe(np.diff(run.ct) * 1000),
                  lateness_ms=old.describe([float(r["command_lateness_s"]) * 1000 for r in run.tx]),
                  target_update_interval_ms=old.describe(np.diff(run.ct[:1001:4]) * 1000),
                  motion_query_results=dict(Counter(r["query_status"] for r in run.rows
                                                   if r["phase"] == "motion" and r["record_type"] == "query_result")))
    for name, time, values in (("A2_speed_rad_s", run.vtime, run.v), ("A1_torque_nm", run.ttime, run.torque),
                               ("F1_torque_nm", *run.series("mit_feedback_torque_nm", 0xF1))):
        peak = int(np.argmax(abs(values)))
        output[name] = old.describe(values)
        output[name]["peak_at_replay_s"] = float(time[peak])
    kt = run.meta["torque_constant_nm_per_a"]
    ft, torque = run.series("mit_feedback_torque_nm", 0xF1)
    output["A1_current_peak_abs_a"] = float(abs(run.torque).max() / kt)
    output["F1_peak_equivalent_current_abs_a"] = float(abs(torque).max() / kt)
    over = abs(torque) > run.meta["arguments"]["current_abort_a"] * kt
    output["F1_above_A1_trip_equivalent"] = [dict(replay_time_s=float(t), torque_nm=float(y),
                                                  equivalent_current_a=float(y / kt))
                                           for t, y in zip(ft[over], torque[over])]
    moved = np.flatnonzero(abs(run.fq) > 2 * old.RAD_PER_COUNT)[0]
    output["first_detectable_F1_motion_bracket_ms"] = [float((run.ft[i] - start) * 1000)
                                                        for i in (moved - 1, moved)]
    output["alignment"] = {"startup_actual_point1_to_point6_s": run.alignment((.1, .6)),
                           "source_1_to_5_s": run.alignment(windows["source_1_to_5_s"])}
    for key in ("bus_voltage_v", "motor_temperature_c", "receive_queue_delay_s"):
        output[key] = old.describe(run.series(key)[1])
    for channel, time in (("A3", run.qtime), ("A1", run.ttime), ("F1", run.ft)):
        output[channel + "_feedback_hz"] = (len(time) - 1) / (time[-1] - time[0])
    with (HERE / (run.stem + "_aligned.csv")).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("replay_time_s", "latest_nominal_source_time_s", "encoded_target_rad",
                         "A3_position_rad", "unshifted_error_deg"))
        for t, q in zip(run.qtime, run.q):
            index = np.searchsorted(run.ct, t, side="right") - 1
            source = run.tx[index]["source_time_s"]
            writer.writerow((t, source, run.cv[index], q, np.degrees(q - run.cv[index])))
    return output


def panel(ax, window, units):
    ax.set(xlim=window, xlabel="Actual time from replay start (s)", ylabel=units)
    ax.grid(alpha=.25)
    values = []
    for line in ax.lines:
        if line.get_transform() is ax.transData:
            t, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
            keep = (t >= window[0]) & (t <= window[1]) & np.isfinite(y)
            if keep.any():
                values.extend(y[keep])
    if values:
        lo, hi = min(values), max(values)
        pad = max((hi - lo) * .13, .001)
        ax.set_ylim(lo - pad, hi + pad)
        ax.text(.01, .98, f"min {lo:.5g} | max {hi:.5g}", va="top", transform=ax.transAxes, fontsize=8,
                bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
    ax.legend(loc="lower right", fontsize=8)


def target(ax, run):
    ax.step(np.r_[run.ct, run.meta["actual_motion_duration_s"]], np.r_[run.cv, run.cv[-1]],
            where="post", color="black", lw=1.1, label="Encoded target, actual send times")


def save(fig, name):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=160)
    plt.close(fig)


def plots(runs):
    fig, axes = plt.subplots(4, 2, figsize=(17, 14), layout="constrained")
    for col, run in enumerate(runs):
        for row, window in ((0, (0., 6.35)), (1, (run.ct[200], run.ct[1000]))):
            ax = axes[row, col]
            target(ax, run)
            ax.plot(run.qtime, run.q, color="tab:blue", label="A3 position")
            ax.plot(run.ft, run.fq, color="tab:orange", lw=.7, alpha=.6, label="F1 position")
            panel(ax, window, "Steering position (rad)")
        axes[2, col].plot(run.qtime, np.degrees(run.q - old.held(run.ct, run.cv, run.qtime)),
                          color="tab:blue", label="A3 − sent target, unshifted")
        panel(axes[2, col], (0., 6.35), "Position error (degrees)")
        axes[3, col].plot(run.vtime, run.v, color="tab:green", label="A2 speed")
        panel(axes[3, col], (0., 6.35), "Speed (rad/s)")
        for row in range(4):
            axes[row, col].set_title(
                f"Run {col + 1} | nominal 200 Hz packets | {run.meta['actual_motion_duration_s']:.3f} s")
    fig.suptitle("Two real initial-transient steering replays | same source 0–5 s, then final hold\n"
                 "Verified zero, front wheel airborne/free, MIT Kp=10/Kd=0.2; actual timing, no curve shifting")
    save(fig, FIGURES[0])

    fig, axes = plt.subplots(3, 2, figsize=(17, 10), layout="constrained")
    for col, run in enumerate(runs):
        target(axes[0, col], run)
        axes[0, col].plot(run.qtime, run.q, "o-", ms=3, color="tab:blue", label="A3 ~99 Hz")
        axes[0, col].plot(run.ft, run.fq, ".-", ms=3, color="tab:orange", label="F1 ~190 Hz")
        panel(axes[0, col], (0., .3), "Position (rad)")
        axes[1, col].plot(run.qtime, np.degrees(run.q - old.held(run.ct, run.cv, run.qtime)),
                          "o-", ms=3, color="tab:blue", label="A3 error")
        axes[1, col].plot(run.ft, np.degrees(run.fq - old.held(run.ct, run.cv, run.ft)),
                          ".-", ms=3, color="tab:orange", label="F1 error")
        panel(axes[1, col], (0., .3), "Position error (degrees)")
        axes[2, col].plot(run.vtime, run.v, "o-", ms=3, color="tab:green", label="A2 speed")
        panel(axes[2, col], (0., .3), "Speed (rad/s)")
        for row in range(3):
            axes[row, col].set_title(f"Run {col + 1} | first 300 ms")
    fig.suptitle("Startup lag and fast movement | no preparation ramp\n"
                 "Sparse A3 misses the largest sampled position error visible in denser F1 feedback")
    save(fig, FIGURES[1])

    fig, axes = plt.subplots(3, 2, figsize=(17, 11), layout="constrained")
    for col, run in enumerate(runs):
        kt = run.meta["torque_constant_nm_per_a"]
        ft, tq = run.series("mit_feedback_torque_nm", 0xF1)
        for row, window in ((0, (0., .1)), (1, (0., 6.35))):
            axes[row, col].plot(run.ttime, run.torque / kt, "o-", ms=3, color="tab:blue",
                               label="Signed A1 current: actual abort input")
            axes[row, col].plot(ft, tq / kt, ".-", ms=3, color="tab:orange",
                               label="F1 reported torque / B0 Kt: equivalent current")
            panel(axes[row, col], window, "Current / equivalent current (A)")
            for limit in (-3., 3.):
                axes[row, col].axhline(limit, color="red", ls="--", lw=.8,
                                      label="A1 abort threshold ±3 A" if limit == 3 else None)
            axes[row, col].set_ylim(-3.5, 4.5)
            axes[row, col].legend(loc="lower right", fontsize=8)
        axes[2, col].eventplot([run.ttime[(run.ttime >= 0) & (run.ttime < .04)],
                                ft[(ft >= 0) & (ft < .04)], run.ct[run.ct < .04]],
                               lineoffsets=(2, 1, 0), colors=("tab:blue", "tab:orange", "black"), linelengths=.6)
        axes[2, col].set(yticks=(0, 1, 2), yticklabels=("MIT sends", "F1 replies", "A1 replies"),
                         xlim=(0, .04), xlabel="Actual time from replay start (s)")
        axes[2, col].grid(axis="x", alpha=.25)
        for row in range(3):
            axes[row, col].set_title(f"Run {col + 1} | current sampling")
    fig.suptitle("A1-only trip can miss a startup peak visible in F1\n"
                 "F1 is documented real-time torque; torque/Kt is an equivalent current, not another A1 measurement")
    save(fig, FIGURES[2])

    fig, axes = plt.subplots(3, 2, figsize=(17, 10), layout="constrained")
    for col, run in enumerate(runs):
        axes[0, col].plot(run.ct, [float(r["command_lateness_s"]) * 1000 for r in run.tx],
                          color="tab:blue", label="Lateness relative to next send deadline")
        panel(axes[0, col], (0., 6.35), "Send lateness (ms)")
        axes[1, col].plot(run.ct[1:], np.diff(run.ct) * 1000, color="tab:blue", label="MIT packet interval")
        panel(axes[1, col], (0., 6.35), "Packet interval (ms)")
        axes[2, col].plot(run.ct[4:1001:4], np.diff(run.ct[:1001:4]) * 1000,
                          color="tab:purple", label="Actual interval between new source targets")
        panel(axes[2, col], (0., 6.35), "Target-update interval (ms)")
        for row in range(3):
            axes[row, col].set_title(f"Run {col + 1}")
    fig.suptitle("200 Hz is nominal: actual packets ~190–191 Hz and source updates ~47.4–47.6 Hz\n"
                 "All targets/repeats are retained; relaxed scheduling stretches nominal six-second duration")
    save(fig, FIGURES[3])

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), layout="constrained")
    for i, run in enumerate(runs):
        source_t = [float(row["target_time_s"]) for row in run.plan]
        sample_t = np.array([float(row["target_time_s"]) for row in run.tx[::4]])
        sample_q = np.interp(run.ct[::4], run.qtime, run.q)
        axes[0].plot(sample_t, sample_q, label=f"Run {i + 1}: A3 interpolated only at actual target-change times")
        source_axis = old.held(run.ct, np.array([float(r["target_time_s"]) for r in run.tx]), run.qtime)
        axes[1].scatter(source_axis, np.degrees(run.q - old.held(run.ct, run.cv, run.qtime)),
                        s=4, alpha=.65, label=f"Run {i + 1}: fresh unshifted error, nominal bin")
        if i == 0:
            axes[0].step(source_t, [float(r["target_rad"]) for r in run.plan], where="post", color="black",
                         lw=.8, label="Identical target plan")
    for ax, units in zip(axes, ("Position (rad)", "Error (degrees)")):
        ax.set(xlim=(0, 5), xlabel="Nominal plan time (s); diagnostic binning, not actual wall time", ylabel=units)
        ax.grid(alpha=.25)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Repetition comparison | only this figure remaps samples to nominal plan coordinates\n"
                 "Interpolated positions are descriptive; all main tracking/error metrics use original fresh samples")
    save(fig, FIGURES[4])


def main():
    runs = [old.Run(stem) for stem in STEMS]
    results = {run.stem: metrics(run, validate(run)) for run in runs}
    assert runs[0].plan == runs[1].plan
    assert [r["mit_payload_hex"] for r in runs[0].tx] == [r["mit_payload_hex"] for r in runs[1].tx]
    plots(runs)
    inputs = [run.path.with_suffix(suffix) for run in runs for suffix in (".csv", ".json")]
    inputs += [run.path.with_name(run.stem + "_plan.csv") for run in runs]
    output = dict(runs=results, identical_plan_and_payload_sequence=True,
                  user_confirmed_setup="Front wheel airborne/free, physically straight, no hand contact; no settings "
                                       "changes. No failed 50 Hz logs provided; neither trip flag changed.",
                  failed_50hz_data_available=False,
                  input_sha256={str(p.relative_to(ROOT)): old.digest(p) for p in inputs})
    (HERE / "statistics.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    with (HERE / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("run", "actual_motion_s", "actual_packet_hz", "actual_core_target_hz",
                         "A3_full_core_RMS_deg", "A3_first_source_second_RMS_deg", "A3_source_1_to_5_RMS_deg",
                         "F1_peak_position_error_deg", "A1_peak_A", "F1_peak_equivalent_A"))
        for i, run in enumerate(runs, 1):
            r = results[run.stem]
            a3 = r["position_errors_rad"]["A3"]
            writer.writerow((i, r["metadata"]["actual_motion_duration_s"], r["actual_packet_hz"],
                             r["actual_core_target_hz"], np.degrees(a3["entire_source_0_to_5_s"]["rms"]),
                             np.degrees(a3["first_source_0_to_1_s"]["rms"]),
                             np.degrees(a3["source_1_to_5_s"]["rms"]),
                             np.degrees(r["position_errors_rad"]["F1"]["entire_source_0_to_5_s"]["maximum_abs"]),
                             r["A1_current_peak_abs_a"], r["F1_peak_equivalent_current_abs_a"]))
    with (HERE / "startup_current_samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("run", "replay_time_s", "opcode", "raw_payload_hex", "policy_signed_A1_current_A",
                         "policy_signed_F1_torque_Nm", "F1_equivalent_current_A"))
        for i, run in enumerate(runs, 1):
            for row in run.rows:
                if (row["phase"] != "motion" or row["can_direction"] != "bus_rx"
                        or row["can_opcode"] not in ("161", "241") or float(row["replay_time_s"]) >= .1):
                    continue
                a1 = -float(row["current_a"]) if row["can_opcode"] == "161" else ""
                torque = float(row["mit_feedback_torque_nm"]) if row["can_opcode"] == "241" else ""
                writer.writerow((i, row["replay_time_s"], row["can_opcode"], row["mit_payload_hex"], a1,
                                 torque, torque / run.meta["torque_constant_nm_per_a"] if torque != "" else ""))
    gallery = "".join(f'<section><h2>{html.escape(name.replace("_", " ").title())}</h2>'
                      f'<a href="{name}.png"><img src="{name}.png" alt="{name}"></a>'
                      f'<p><a href="{name}.svg">SVG</a> · <a href="{name}.pdf">PDF</a></p></section>'
                      for name in FIGURES)
    (HERE / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Initial steering replay</title>'
        '<style>body{font:16px system-ui;max-width:1400px;margin:2em auto;padding:0 1em}img{width:100%}'
        'section{margin:3em 0}a{color:#1458a0}</style><h1>Two initial-transient steering replays</h1>'
        '<p>2026-10-27 · nominal 200 Hz · source 0–5 s + final hold · airborne/free steering.</p>'
        '<p>Large brief startup error, then ~0.59° A3 RMS over source 1–5 s. F1 reveals a startup '
        'torque peak missed by the A1-only 3 A trip. Failed 50 Hz logs are unavailable.</p>'
        '<p><a href="README.md">Report</a> · <a href="statistics.json">Exact statistics</a></p>' + gallery + '</html>')
    print(json.dumps({run.stem: dict(packet_hz=results[run.stem]["actual_packet_hz"],
                                     A1_peak_a=results[run.stem]["A1_current_peak_abs_a"],
                                     F1_peak_equivalent_a=results[run.stem]["F1_peak_equivalent_current_abs_a"])
                      for run in runs}, indent=2))


if __name__ == "__main__":
    main()
