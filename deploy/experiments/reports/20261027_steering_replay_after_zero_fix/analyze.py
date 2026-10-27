"""Analyze the two corrected-zero hardware replays without opening CAN or changing raw data."""

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
LEGACY = EXPERIMENTS / "reports/20261026_policy_replay_hardware"
STEMS = (
    "20261027_144059_652887_scaled_50hz_policy_replay",
    "20261027_144436_005178_scaled_200hz_policy_replay",
)
CORE = (2.0, 7.0)
FIGURES = ("steering_tracking", "tracking_error_and_delay", "initialization_and_timing",
           "simulator_comparison", "before_after_zero_fix")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Reuse the previous report's checked fresh-feedback reader and numerical helpers.
# Its main() is not called and none of its existing artifacts are regenerated.
old = load_module("replay_report_helpers", LEGACY / "analyze.py")
plt = old.plt
sys.path.insert(0, str(EXPERIMENTS))
runtime = load_module("corrected_steering_runtime", EXPERIMENTS / "steering_policy_replay.py")
sys.path.pop(0)


def validate(run):
    """Check raw payloads, source/plan reconstruction, origin setup and complete transmissions."""
    meta, args = run.meta, argparse.Namespace(**run.meta["arguments"])
    args.targets = EXPERIMENTS / "replay_targets" / Path(args.targets).name
    assert meta["completed"] and meta["disable_confirmed"] and meta["final_fault_code"] == 0
    assert meta["script_sha256"] == old.digest(EXPERIMENTS / "steering_policy_replay.py")
    assert meta["transport_sha256"] == old.digest(EXPERIMENTS / "steering_sine_mit_io.py")
    assert meta["target_hz"] == 50 and args.query_rate == 100 and args.sign == args.torque_sign == -1
    setup = meta["steering_origin_setup"]
    assert all(setup[key] for key in ("attempted", "acknowledged", "verified"))
    assert meta["encoder_zero_count"] == setup["A3_zero_count"] == 0
    assert setup["F1_position_code"] == 32767
    assert abs(meta["native_zero_rad"]) < 6 / 65535
    writes = [r for r in run.rows if r["can_direction"] == "local_tx"]
    assert sum(r["can_opcode"] == "177" for r in writes) == 1
    assert all(int(r["mit_can_id"]) in (0x102, 0x502) for r in writes)
    assert all(r["can_length"] == "1" for r in writes if r["record_type"] != "mit_target")
    plan, source_hash = runtime.load_plan(args)  # Pure parsing: never calls hardware startup/motion.
    assert source_hash == meta["targets_sha256"]
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=runtime.PLAN_FIELDS)
    writer.writeheader()
    writer.writerows(plan)
    assert text.getvalue().encode() == run.path.with_name(run.stem + "_plan.csv").read_bytes()
    repeats = meta["packet_hz"] // 50
    assert len(run.tx) == len(plan) * repeats
    motor = runtime.mit.Motor("steering", 2, 1.0, -1, -1, zero_count=0,
                              zero_rad=meta["native_zero_rad"], ranges=tuple(meta["mit_ranges"]))
    expected_packets = [runtime.payload(row["target_rad"], motor, args) for row in plan]
    for i, row in enumerate(run.tx):
        index = i // repeats
        assert int(row["packet_index"]) == i and int(row["target_index"]) == index
        assert float(row["requested_position_rad"]) == plan[index]["target_rad"]
        assert row["phase"] == plan[index]["phase"]
        assert abs(float(row["target_time_s"]) - index * .02) < 1e-12
        raw = bytes.fromhex(row["mit_payload_hex"])
        assert raw == expected_packets[index]
        fields = runtime.mit.target_fields(motor, raw)
        assert all(abs(float(row[key]) - value) < 1e-12 for key, value in fields.items())
        assert abs(run.cv[i] - run.request[i]) <= 3 / 65535 + 1e-12
    decoded = Counter()
    for row in run.rows:
        if row["can_direction"] != "bus_rx":
            continue
        raw = bytes.fromhex(row["mit_payload_hex"])
        assert int(row["mit_can_id"]) == 2 and len(raw) == int(row["can_length"])
        decoded[f"{raw[0]:02X}"] += 1
        if row["phase"] != "motion":
            continue
        if raw[0] == 0xA3:
            count = struct.unpack("<i", raw[3:7])[0]
            assert count == int(row["encoder_count"])
            assert abs(float(row["position_unwrapped_rad"]) + count * old.RAD_PER_COUNT) < 1e-12
        elif raw[0] == 0xA2:
            speed = -struct.unpack("<i", raw[1:5])[0] * .01 * 2 * np.pi / 60
            assert abs(float(row["velocity_rad_s"]) - speed) < 1e-12
        elif raw[0] == 0xA1:
            current = struct.unpack("<i", raw[1:5])[0] * .001
            assert abs(float(row["current_a"]) - current) < 1e-12
            assert abs(float(row["torque_nm"]) + current * meta["torque_constant_nm_per_a"]) < 1e-12
        elif raw[0] == 0xF1:
            code = int.from_bytes(raw[1:3], "big")
            native = code * 6 / 65535 - 3
            assert code not in (0, 65535) and raw[6] == 1
            assert abs(float(row["mit_feedback_position_rad"]) + native - meta["native_zero_rad"]) < 1e-12
        elif raw[0] == 0xAE:
            assert raw[6] == 5 and raw[7] == 0
    return dict(raw_reply_counts=dict(decoded), all_tx_payloads_and_plan_bytes_verified=True,
                B1_sent_once=True, fresh_feedback_units_signs_and_reference_verified=True)


def lag_curve(run):
    time, q = run.select(run.qtime, run.q, CORE)
    shifts = np.arange(0, .10001, .0001)
    errors = np.array([q - old.held(run.ct, run.cv, time - shift) for shift in shifts])
    raw_rms = np.sqrt(np.mean(errors ** 2, axis=1))
    shape_rms = np.sqrt(np.mean((errors - errors.mean(axis=1)[:, None]) ** 2, axis=1))
    best = int(shape_rms.argmin())
    near = shifts[shape_rms <= shape_rms[best] * 1.01]
    return dict(best_shift_ms=float(shifts[best] * 1000),
                within_1pct_loss_ms=[float(near.min() * 1000), float(near.max() * 1000)],
                demeaned_rms_at_best_rad=float(shape_rms[best]),
                raw_rms_best_shift_ms=float(shifts[raw_rms.argmin()] * 1000),
                interpretation="Waveform alignment only; ~10 ms A3 sampling, unknown acquisition/wire times. "
                               "0.1 ms scan spacing is not delay precision; loss band is not a confidence interval.",
                shifts_ms=shifts * 1000, rms_deg=np.degrees(shape_rms))


def metrics(run, validation):
    qtime, q = run.select(run.qtime, run.q, CORE)
    ft, fq = run.select(run.ft, run.fq, CORE)
    lag = lag_curve(run)
    output = dict(metadata=run.meta, validation=validation, rows=len(run.rows),
                  packets_sent=len(run.tx), packets_planned=len(run.plan) * run.meta["packet_hz"] // 50,
                  achieved_host_tx_hz=(len(run.ct) - 1) / (run.ct[-1] - run.ct[0]),
                  core_tracking_rad=run.tracking(CORE),
                  core_tracking_deg={k: float(np.degrees(v)) if k != "n" else v
                                     for k, v in run.tracking(CORE).items()},
                  phase_tracking_rad={name: run.tracking(window) for name, window in
                                      (("ramp", (.1, 1.)), ("pre_hold_last_half", (1.5, 2.)),
                                       ("post_hold_last_half", (7.5, 8.)))},
                  lag={k: v for k, v in lag.items() if k not in ("shifts_ms", "rms_deg")},
                  F1_core_tracking_rad=old.error_stats(fq, old.held(run.ct, run.cv, ft)),
                  A3_minus_interpolated_F1_core_rad=old.describe(q - np.interp(qtime, run.ft, run.fq)),
                  command_interval_ms=old.describe(np.diff(run.ct) * 1000),
                  command_lateness_ms=old.describe([float(r["command_lateness_s"]) * 1000 for r in run.tx]),
                  motion_query_results=dict(Counter(r["query_status"] for r in run.rows
                                                   if r["record_type"] == "query_result" and r["phase"] == "motion")))
    for key, time, y in (("speed_rad_s", run.vtime, run.v), ("current_derived_torque_nm", run.ttime, run.torque)):
        output[key] = old.describe(run.select(time, y, CORE)[1])
    for name, time in (("A3", run.qtime), ("A2", run.vtime), ("A1", run.ttime)):
        output[name + "_feedback_hz"] = (len(time) - 1) / (time[-1] - time[0])
    for key in ("bus_voltage_v", "motor_temperature_c", "receive_queue_delay_s"):
        output[key] = old.describe(run.series(key)[1])
    output["encoded_fields"] = {key: float(run.tx[0][key]) for key in
                                ("mit_target_velocity_motor_rad_s", "mit_kp_nm_per_motor_rad",
                                 "mit_kd_nm_s_per_motor_rad", "mit_feedforward_torque_motor_nm")}
    output["core_target_rad"] = old.describe(old.held(run.ct, run.cv, qtime))
    # Independent fresh A3 points, paired only with commands already sent by their kernel RX times.
    with (HERE / (run.stem + "_aligned.csv")).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("replay_time_s", "requested_position_rad", "encoded_position_rad", "A3_position_rad",
                         "error_rad", "error_deg"))
        for t, y in zip(run.qtime, run.q):
            request = float(old.held(run.ct, run.request, np.array([t]))[0])
            target = float(old.held(run.ct, run.cv, np.array([t]))[0])
            writer.writerow((t, request, target, y, y - target, np.degrees(y - target)))
    return output


def finish(ax, window, unit):
    ax.set(xlim=window, xlabel="Time from replay start (s)", ylabel=unit)
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
        pad = max((hi - lo) * .13, .0002)
        ax.set_ylim(lo - pad, hi + pad)
        ax.text(.01, .98, f"min {lo:.5g} | max {hi:.5g}", va="top", transform=ax.transAxes, fontsize=8,
                bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
    for at in (2., 7.):
        if window[0] < at < window[1]:
            ax.axvline(at, color="gray", ls=":", lw=.8)
    ax.legend(loc="lower right", fontsize=8)


def target_line(ax, run):
    ax.step(np.r_[run.ct, 8.], np.r_[run.cv, run.cv[-1]], where="post", color="black", lw=1.1,
            label="Encoded target actually sent (50 Hz held)")


def save(fig, name):
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(HERE / f"{name}.{suffix}", dpi=160)
    plt.close(fig)


def plots(runs, results, sims, before_after):
    fig, axes = plt.subplots(4, 2, figsize=(17, 14), layout="constrained")
    for col, run in enumerate(runs):
        for row, window in ((0, (0., 8.)), (1, CORE)):
            ax = axes[row, col]
            target_line(ax, run)
            ax.plot(run.qtime, run.q, color="tab:blue", label="A3 position from verified straight zero")
            ax.plot(run.ft, run.fq, color="tab:orange", alpha=.55, lw=.7, label="F1 position from verified zero")
            ax.set_title(f"MIT packets {run.meta['packet_hz']} Hz | " + ("full replay" if row == 0 else "core"))
            finish(ax, window, "Position (rad)")
        axes[2, col].plot(run.vtime, run.v, label="A2 measured speed", color="tab:blue")
        finish(axes[2, col], CORE, "Speed (rad/s)")
        axes[3, col].plot(run.ttime, run.torque, label="A1 current × B0, signed", color="tab:green")
        finish(axes[3, col], CORE, "Current-derived torque (Nm)")
    for row in (2, 3):
        limits = [ax.get_ylim() for ax in axes[row]]
        for ax in axes[row]:
            ax.set_ylim(min(v[0] for v in limits), max(v[1] for v in limits))
    fig.suptitle("Corrected-zero hardware replay | identical model 2500 Isaac-recorded targets\n"
                 "Front wheel airborne/free; requested MIT Kp=10, Kd=0.2; no curve shifting or bias correction")
    save(fig, FIGURES[0])

    fig, axes = plt.subplots(3, 2, figsize=(17, 11), layout="constrained")
    for col, run in enumerate(runs):
        target_line(axes[0, col], run)
        axes[0, col].plot(run.qtime, run.q, label="A3 position", color="tab:blue")
        finish(axes[0, col], (.0, 1.3), "Ramp position (rad)")
        error = np.degrees(run.q - old.held(run.ct, run.cv, run.qtime))
        axes[1, col].plot(run.qtime, error, label="Unshifted A3 − encoded target", color="tab:blue")
        finish(axes[1, col], CORE, "Position error (degrees)")
        axes[1, col].axhline(0, color="black", lw=.5)
        lag = lag_curve(run)
        axes[2, col].plot(lag["shifts_ms"], lag["rms_deg"], color="tab:purple",
                          label="Core error RMS after removing mean")
        best = lag["best_shift_ms"]
        axes[2, col].axvline(best, color="red", ls="--", label=f"Best alignment {best:g} ms")
        axes[2, col].set(xlabel="Target shift (ms)", ylabel="Demeaned error RMS (degrees)", xlim=(0, 100))
        axes[2, col].grid(alpha=.25)
        axes[2, col].legend(loc="lower right", fontsize=8)
        for row in range(3):
            axes[row, col].set_title(f"MIT packets {run.meta['packet_hz']} Hz")
    fig.suptitle("Tracking and descriptive response alignment\n"
                 "Only the bottom row shifts a reference; ~10 ms A3 sampling does not identify pure CAN/firmware delay")
    save(fig, FIGURES[1])

    fig, axes = plt.subplots(3, 2, figsize=(17, 11), layout="constrained")
    for col, run in enumerate(runs):
        ack = next(float(r["elapsed_s"]) for r in run.rows
                   if r["can_direction"] == "bus_rx" and r["can_opcode"] == "177")
        channels = ((163, "encoder_count", old.RAD_PER_COUNT, "Native A3", "tab:blue"),
                    (241, "mit_feedback_position_motor_rad", 1., "Native F1", "tab:orange"))
        for opcode, field, scale, label, color in channels:
            rows = [r for r in run.rows if r["phase"] == "startup" and r["can_direction"] == "bus_rx"
                    and r["can_opcode"] == str(opcode) and r[field]]
            axes[0, col].plot([float(r["elapsed_s"]) for r in rows], [float(r[field]) * scale for r in rows],
                              "o-", ms=4, label=label, color=color)
        finish(axes[0, col], (0., .014), "Startup native angle (rad)")
        axes[0, col].axvline(ack, color="red", ls=":", label="B1 acknowledgement")
        axes[0, col].legend(loc="lower right", fontsize=8)
        qt, queue = run.series("receive_queue_delay_s")
        axes[1, col].plot(qt, queue * 1000, color="tab:orange", alpha=.65, label="Kernel RX → host-processing age")
        axes[1, col].plot(run.ct, [float(r["command_lateness_s"]) * 1000 for r in run.tx],
                          color="tab:blue", lw=.8, label="Command send lateness")
        finish(axes[1, col], (0., 8.), "Host timing (ms)")
        axes[1, col].axhline(3., color="red", ls="--", label="Existing 3 ms abort threshold")
        axes[1, col].set_ylim(-.1, 3.5)
        axes[1, col].legend(loc="lower right", fontsize=8)
        axes[2, col].plot(run.ct[1:], np.diff(run.ct) * 1000, color="tab:blue",
                          label="Actual host send intervals")
        finish(axes[2, col], (0., 8.), "Packet interval (ms)")
        for row in range(3):
            axes[row, col].set_title(f"MIT packets {run.meta['packet_hz']} Hz | completed")
        axes[0, col].set_xlabel("Time from logger start (s); B1 changes coordinates")
    fig.suptitle("Both B1 references verified near zero; both packet-rate runs completed\n"
                 "Startup coordinate reset is not physical wheel motion; timestamps are host-side")
    save(fig, FIGURES[2])

    fig, axes = plt.subplots(2, 2, figsize=(17, 8), layout="constrained")
    for col, run in enumerate(runs):
        for row, window in enumerate(((0., 8.), CORE)):
            ax = axes[row, col]
            target_line(ax, run)
            for tag, sim in sims.items():
                ax.plot(sim["time"], sim["steering"], lw=.8, alpha=.8, label=tag + " saved nominal fixed-root replay")
            ax.plot(run.qtime, run.q, color="tab:blue", lw=1.4,
                    label=f"Hardware A3, {run.meta['packet_hz']} Hz packets")
            finish(ax, window, "Position (rad)")
    fig.suptitle("Identical input, nominal saved simulators versus airborne hardware\n"
                 "No simulation rerun; support/loading, current dynamics and physical parameters are not fully matched")
    save(fig, FIGURES[3])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for col, (field, title) in enumerate((("rms", "Unshifted RMS error"), ("bias", "Mean signed error"))):
        for offset, key, label, color in ((-.18, "old", "Before B1 fix (Oct 26)", "tab:red"),
                                          (.18, "new", "After B1 fix (Oct 27)", "tab:blue")):
            values = [before_after[str(rate)][key][field] * 180 / np.pi for rate in (50, 200)]
            bars = axes[col].bar(np.arange(2) + offset, values, .36, label=label, color=color)
            axes[col].bar_label(bars, fmt="%.3f", padding=3)
        axes[col].set(xticks=(0, 1), xticklabels=("50 Hz packets", "200 Hz packets"), ylabel="Degrees", title=title)
        axes[col].grid(axis="y", alpha=.2)
        # Leave room to the right so the lower-right legend does not cover the small new-error bars.
        axes[col].set_xlim(-.6, 2.7)
        axes[col].legend(loc="lower right", fontsize=8)
    fig.suptitle("Zero-reference correction | same common core interval 2.00–5.36 s\n"
                 "Original 200 Hz run aborted later; no bias removal or response shifting in these metrics")
    save(fig, FIGURES[4])


def main():
    runs = [old.Run(stem) for stem in STEMS]
    results = {r.stem: metrics(r, validate(r)) for r in runs}
    assert runs[0].plan == runs[1].plan
    assert runs[0].path.with_name(runs[0].stem + "_plan.csv").read_bytes() == \
        runs[1].path.with_name(runs[1].stem + "_plan.csv").read_bytes()
    assert all(a["mit_payload_hex"] == b["mit_payload_hex"] for a, b in zip(runs[0].tx, runs[1].tx[::4]))
    before_after = {}
    for run, stem in zip(runs, old.FILES[:2]):
        previous = old.Run(stem)
        before_after[str(run.meta["packet_hz"])] = dict(
            old=previous.tracking(old.COMMON), new=run.tracking(old.COMMON), window_s=list(old.COMMON))
    sims = old.simulator_data()
    sim_metrics = {}
    for tag, sim in sims.items():
        with np.load(old.SIM / tag / "airborne.npz") as source:
            assert np.allclose(source["target_rad"][:, 4],
                               np.array([float(r["target_rad"]) for r in runs[0].plan]), rtol=0, atol=1e-7)
        keep = (sim["time"] > 2.) & (sim["time"] <= 7.)
        sim_metrics[tag] = old.error_stats(sim["steering"][keep], sim["target"][keep])
    plots(runs, results, sims, before_after)
    files = [r.path.with_suffix(suffix) for r in runs for suffix in (".csv", ".json")]
    files += [r.path.with_name(r.stem + "_plan.csv") for r in runs]
    output = dict(runs=results, before_after_common_core=before_after, saved_simulator_core_error_rad=sim_metrics,
                  pair_checks=dict(identical_plan_bytes=True, identical_payload_for_every_50hz_target=True,
                                   both_complete_and_disabled=True, both_B1_verified=True),
                  user_confirmed_conditions="Front wheel airborne/free, no restraints/hand contact; physically "
                                            "straight before each run, no settings changes between runs.",
                  source_sha256={str(p.relative_to(ROOT)): old.digest(p) for p in files},
                  scope="Two single-run unloaded hardware replays, not balance or full actuator-envelope tests.")
    (HERE / "statistics.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    with (HERE / "summary.csv").open("w", newline="") as stream:
        fields = ("packet_hz", "completed", "sent", "RMS_deg", "bias_deg", "p95_abs_deg", "max_abs_deg",
                  "alignment_ms", "achieved_tx_hz", "max_lateness_ms")
        writer = csv.writer(stream)
        writer.writerow(fields)
        for run in runs:
            m = results[run.stem]
            writer.writerow((run.meta["packet_hz"], True, m["packets_sent"], m["core_tracking_deg"]["rms"],
                             m["core_tracking_deg"]["bias"], m["core_tracking_deg"]["p95_abs"],
                             m["core_tracking_deg"]["maximum_abs"], m["lag"]["best_shift_ms"],
                             m["achieved_host_tx_hz"], m["command_lateness_ms"]["maximum"]))
    sections = "".join(f'<section><h2>{html.escape(name.replace("_", " ").title())}</h2>'
                       f'<a href="{name}.png"><img src="{name}.png" alt="{name}"></a>'
                       f'<p><a href="{name}.svg">SVG</a> · <a href="{name}.pdf">PDF</a></p></section>'
                       for name in FIGURES)
    (HERE / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Corrected-zero steering replay</title>'
        '<style>body{font:16px system-ui;max-width:1400px;margin:2em auto;padding:0 1em}img{width:100%}'
        'section{margin:3em 0}a{color:#1458a0}</style><h1>Steering replay after the zero-reference fix</h1>'
        '<p>2026-10-27 · same 50 Hz-held targets · 50 versus 200 Hz MIT packets · airborne/free steering.</p>'
        '<p>Both runs completed with verified near-zero references. Core RMS errors: 0.701° / 0.788°. '
        'No improvement established at 200 Hz. Plotted responses are unshifted.</p>'
        '<p><a href="README.md">Report</a> · <a href="statistics.json">Exact statistics</a> · '
        '<a href="summary.csv">Summary CSV</a></p>' + sections + '</html>')
    print(json.dumps({r.meta["packet_hz"]: results[r.stem]["core_tracking_deg"] for r in runs}, indent=2))


if __name__ == "__main__":
    main()
