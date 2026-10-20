#!/usr/bin/env python3
"""Offline analysis of the 20261020 rear C1 sweep; never opens CAN or changes runtime code.

Requires numpy, pandas, matplotlib. Run with system Python from repository root:
  MPLCONFIGDIR=/tmp/drive-damping-matplotlib python3 \
    deploy/experiments/reports/drive_damping_20261020_154728/analysis.py
Uses one mean per complete plateau, not repeated cached CSV rows. The user
confirmed the rear wheel was airborne, unconstrained and free to rotate.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "deploy/experiments/logs/20261020_154728_360275_drive_speed_sweep.csv"
COLS = (
    "record_type", "elapsed_s", "repeat", "step", "phase", "phase_elapsed_s", "opcode_hex", "payload_hex",
    "query_status", "query_round_trip_s", "velocity_rad_s", "current_a", "torque_estimate_nm",
    "torque_constant_nm_per_a", "velocity_age_s", "current_age_s", "status_age_s", "motor_temperature_c",
    "bus_voltage_v", "bus_current_raw", "bus_current_a", "motor_mode", "fault_code", "current_limit_a",
    "watchdog_ms", "watchdog_enabled", "plateau_speed_rad_s", "command_speed_rad_s", "encoded_command_rad_s",
    "command_interval_s", "command_lateness_s", "note", "motor_id", "encoder_sign", "torque_sign", "gear_ratio",
    "reply_native_rpm", "reply_velocity_rad_s",
)


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_loss(w, torque, constrained=False):
    """tau = B*w + Tc*sign(w) + offset; constrained fit requires B,Tc >= 0."""
    w, torque = np.asarray(w, float), np.asarray(torque, float)
    x = np.column_stack((w, np.sign(w), np.ones(len(w))))
    if np.linalg.matrix_rank(x) != 3:
        raise ValueError("Need both directions and multiple speed magnitudes")
    choices = ((0, 1, 2), (0, 2), (1, 2), (2,)) if constrained else ((0, 1, 2),)
    best = None
    for free in choices:
        coefficients = np.zeros(3)
        coefficients[list(free)] = np.linalg.lstsq(x[:, free], torque, rcond=None)[0]
        if constrained and np.any(coefficients[:2] < -1e-12):
            continue
        residual = torque-x@coefficients
        sse = float(residual@residual)
        if best is None or sse < best[0]:
            best = (sse, coefficients)
    sse, coefficients = best
    result = dict(zip(("B_nm_s_per_rad", "coulomb_nm", "offset_nm"), coefficients.tolist()))
    result.update(n_plateaus=len(w), rmse_nm=float(np.sqrt(sse/len(w))), sse_nm2=sse)
    if not constrained and len(w) > 3:
        se = np.sqrt(np.diag(np.linalg.inv(x.T@x)*sse/(len(w)-3)))
        result["standard_errors"] = se.tolist()
        if len(w) == 19:
            # Student-t(.975,16), independently evaluated with installed Isaac Python scipy.stats.t.
            critical = 2.1199052992212546
            result["conditional_95pct_t_intervals"] = np.column_stack(
                (coefficients-critical*se, coefficients+critical*se)).tolist()
    return result


def raw_checks(d):
    counts = {}
    for row in d.loc[d.record_type == "rx"].itertuples():
        op = row.opcode_hex
        if op not in ("A1", "A2", "B0", "AE", "CF", "C1", "B3", "CD"):
            continue
        data = bytes.fromhex(row.payload_hex)
        if op == "A1":
            current = struct.unpack("<i", data[1:])[0]*.001
            assert abs(current-row.current_a) < 1e-12
            assert abs(row.torque_sign*current*row.torque_constant_nm_per_a-row.torque_estimate_nm) < 1e-12
        elif op in ("A2", "C1"):
            rpm = struct.unpack("<i", data[1:])[0]*.01
            speed = row.encoder_sign*rpm*2*np.pi/60/row.gear_ratio
            assert abs(rpm-row.reply_native_rpm) < 1e-12
            assert abs(speed-row.reply_velocity_rad_s) < 1e-12
            if op == "A2":
                assert abs(speed-row.velocity_rad_s) < 1e-12
        elif op == "B0":
            assert abs(struct.unpack("<f", data[2:6])[0]-row.torque_constant_nm_per_a) < 1e-12
        elif op in ("AE", "CF"):
            voltage, current = struct.unpack("<HH", data[1:5])
            assert abs(voltage*.01-row.bus_voltage_v) < 1e-12
            assert current == row.bus_current_raw and abs(current*.01-row.bus_current_a) < 1e-12
            assert (data[5], data[6], data[7]) == (row.motor_temperature_c, row.motor_mode, row.fault_code)
        elif op == "B3":
            assert abs(struct.unpack("<I", data[1:])[0]*.001-row.current_limit_a) < 1e-12
        elif op == "CD":
            assert data[1] == row.watchdog_enabled
            assert struct.unpack("<H", data[2:4])[0] == row.watchdog_ms
        counts[op] = counts.get(op, 0)+1
    return counts


def plateau_means(d, trim=1.0):
    rx = d[(d.record_type == "rx") & (d.phase == "hold")]
    rows = []
    for (repeat, step), group in rx.groupby(["repeat", "step"], sort=True):
        if group.phase_elapsed_s.max() < 14.9:
            continue
        window = group[(group.phase_elapsed_s >= trim) & (group.phase_elapsed_s < 15-trim)]
        velocity = window[window.opcode_hex == "A2"]
        current = window[window.opcode_hex == "A1"]
        if min(len(velocity), len(current)) < 10:
            continue
        assert (current.current_age_s.abs() < 1e-9).all()
        assert (velocity.velocity_age_s.abs() < 1e-9).all()
        t = velocity.elapsed_s.to_numpy()
        w = velocity.velocity_rad_s.to_numpy()
        slope = np.polyfit(t-t[0], w, 1)[0]
        row = dict(repeat=int(repeat), step=int(step), target_rad_s=float(group.plateau_speed_rad_s.iloc[0]),
                   window_start_s=float(window.elapsed_s.min()), window_end_s=float(window.elapsed_s.max()),
                   n_a2=len(velocity), n_a1=len(current), speed_mean_rad_s=float(w.mean()),
                   speed_std_rad_s=float(w.std(ddof=1)), speed_linear_trend_rad_s2=float(slope),
                   current_mean_a=float(current.current_a.mean()), torque_mean_nm=float(current.torque_estimate_nm.mean()),
                   torque_std_nm=float(current.torque_estimate_nm.std()), temperature_mean_c=float(current.motor_temperature_c.mean()),
                   current_peak_abs_a=float(current.current_a.abs().max()),
                   fraction_current_at_least_95pct_ceiling=float((current.current_a.abs() >= .95*current.current_limit_a).mean()),
                   fraction_current_above_ceiling=float((current.current_a.abs() > current.current_limit_a).mean()))
        row["speed_error_pct"] = 100*(row["speed_mean_rad_s"]-row["target_rad_s"])/abs(row["target_rad_s"])
        rows.append(row)
    return pd.DataFrame(rows)


def fit_table(g, constrained=False):
    return fit_loss(g.speed_mean_rad_s, g.torque_mean_nm, constrained)


def bootstrap(g, repetitions=10000):
    """Descriptive plateau-resampling interval, stratified by repeat and direction; not sensor calibration."""
    rng = np.random.default_rng(42)
    strata = [z.index.to_numpy() for _, z in g.groupby([g.repeat, np.sign(g.target_rad_s)])]
    estimates = []
    for _ in range(repetitions):
        indices = np.concatenate([rng.choice(z, len(z), replace=True) for z in strata])
        try:
            estimates.append(fit_table(g.loc[indices])["B_nm_s_per_rad"])
        except ValueError:
            continue
    return dict(seed=42, repetitions=len(estimates), stratified_by="repeat and direction",
                B_percentile_2_5_50_97_5=np.percentile(estimates, [2.5, 50, 97.5]).tolist())


def summarize(d, g, checks):
    rx = d[d.record_type == "rx"]
    motion = rx[rx.phase.isin(["ramp_up", "settle", "hold", "ramp_down", "rest"])]
    status = motion[motion.opcode_hex == "AE"]
    command = d[d.record_type == "command"]
    failure = d[(d.record_type == "query") & (d.query_status != "received")]
    failed = failure.iloc[-1]
    sent = d[(d.record_type == "tx") & (d.opcode_hex == failed.opcode_hex) &
             (d.elapsed_s < failed.elapsed_s)].iloc[-1]
    late = rx[(rx.opcode_hex == failed.opcode_hex) & (rx.elapsed_s > failed.elapsed_s)].iloc[0]
    table = g.assign(magnitude=g.target_rad_s.abs(), abs_torque=g.torque_mean_nm.abs()).groupby("magnitude").agg(
        plateaus=("repeat", "size"), mean_absolute_torque_nm=("abs_torque", "mean"))
    command_dt = np.diff(command.elapsed_s)
    out = dict(rows=len(d), duration_s=float(d.elapsed_s.max()), raw_validated_reply_counts=checks,
               input_rows_by_type=d.record_type.value_counts().to_dict(), complete_hold_plateaus=len(g),
               retained_samples_a1=int(g.n_a1.sum()), retained_samples_a2=int(g.n_a2.sum()),
               fit_window="Each complete hold, phase_elapsed_s in [1,14); independent fresh A1/A2 rows only",
               user_confirmed_condition="Rear wheel completely airborne, unconstrained, free to rotate",
               torque_constant_nm_per_a=sorted(rx.torque_constant_nm_per_a.dropna().unique().tolist()),
               original_q_current_ceiling_readback_a=float(
                   rx[(rx.phase == "startup") & (rx.opcode_hex == "B3")].current_limit_a.iloc[0]),
               active_q_current_ceiling_readback_a=sorted(motion.current_limit_a.unique().tolist()),
               gear_ratio=sorted(d.gear_ratio.unique().tolist()),
               encoder_sign=sorted(d.encoder_sign.unique().tolist()), torque_sign=sorted(d.torque_sign.unique().tolist()),
               voltage_range_v=[float(status.bus_voltage_v.min()), float(status.bus_voltage_v.max())],
               temperature_range_c=[float(status.motor_temperature_c.min()), float(status.motor_temperature_c.max())],
               motion_modes=sorted(status.motor_mode.unique().tolist()), motion_faults=sorted(status.fault_code.unique().tolist()),
               command_rate_hz=1/float(command_dt.mean()), max_command_interval_ms=float(command_dt.max()*1000),
               query_failures=failure[["elapsed_s", "repeat", "step", "phase", "opcode_hex", "query_status"]].to_dict("records"),
               stop_detail=dict(request_elapsed_s=float(sent.elapsed_s), timeout_check_elapsed_s=float(failed.elapsed_s),
                                late_receipt_elapsed_s=float(late.elapsed_s),
                                host_request_to_timeout_check_ms=float((failed.elapsed_s-sent.elapsed_s)*1000),
                                host_request_to_late_receipt_ms=float((late.elapsed_s-sent.elapsed_s)*1000),
                                last_C1_elapsed_s=float(command.elapsed_s.max()),
                                explanation="A1 20ms deadline; host receipt gap cannot distinguish driver delay, CAN delay, or host scheduling"),
               unconstrained_fit=fit_table(g), physical_nonnegative_fit=fit_table(g, True),
               conditional_bootstrap=bootstrap(g),
               by_repeat={str(r): fit_table(z) for r, z in g.groupby("repeat")},
               by_direction={str(int(sign)): np.polyfit(z.speed_mean_rad_s.abs(), z.torque_mean_nm.abs(), 1).tolist()
                             for sign, z in g.groupby(np.sign(g.target_rad_s))},
               mean_absolute_torque_by_target=table.reset_index().to_dict("records"),
               positive_viscous_only_fit_B_nm_s_per_rad=float(
                   np.dot(g.speed_mean_rad_s, g.torque_mean_nm)/np.dot(g.speed_mean_rad_s, g.speed_mean_rad_s)),
               peak_plateau_speed_error_pct=float(g.speed_error_pct.abs().max()),
               speed_std_range_rad_s=[float(g.speed_std_rad_s.min()), float(g.speed_std_rad_s.max())],
               largest_plateau_linear_speed_trend_rad_s2=float(g.speed_linear_trend_rad_s2.abs().max()),
               leave_one_plateau_out_B=[fit_table(g.drop(i))["B_nm_s_per_rad"] for i in g.index])
    for opcode in ("A2", "A1"):
        stream = motion[motion.opcode_hex == opcode]
        out[f"{opcode.lower()}_observed_rate_hz"] = float((len(stream)-1)/(stream.elapsed_s.max()-stream.elapsed_s.min()))
    out["trim_sensitivity"] = {str(trim): fit_table(plateau_means(d, trim)) for trim in range(6)}
    out["first_query_metadata"] = json.loads(d.iloc[0].note)
    return out


def save_plots(d, g, result, output):
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .22})
    b = result["unconstrained_fit"]
    physical = result["physical_nonnegative_fit"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for repeat, points in g.groupby("repeat"):
        for sign, z in points.groupby(np.sign(points.target_rad_s)):
            label = f"Repeat {repeat}, {'forward' if sign > 0 else 'backward'}"
            color = "tab:blue" if sign > 0 else "tab:orange"
            axes[0].scatter(z.speed_mean_rad_s, z.torque_mean_nm, marker="o" if repeat == 1 else "s",
                            s=55, color=color, edgecolor="black", linewidth=.5, label=label, zorder=3)
            axes[1].scatter(z.speed_mean_rad_s.abs(), z.torque_mean_nm.abs()/z.speed_mean_rad_s.abs(),
                            marker="o" if repeat == 1 else "s", s=55, color=color)
    for sign in (-1, 1):
        w = sign*np.linspace(.4, 8.3, 100)
        axes[0].plot(w, physical["B_nm_s_per_rad"]*w+physical["coulomb_nm"]*sign+physical["offset_nm"],
                     color="black", label="Physical constrained fit" if sign == 1 else None)
        axes[0].plot(w, b["B_nm_s_per_rad"]*w+b["coulomb_nm"]*sign+b["offset_nm"],
                     "--", color="tab:green", label="Unconstrained fit" if sign == 1 else None)
    axes[0].set(xlabel="Measured mean output speed (rad/s)", ylabel="Mean current-derived torque (N m)",
                title="19 plateau means (one point per hold): nearly constant drag")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].axhline(.05, color="tab:red", linestyle="--", label="Old 0.05 coefficient (comparison only)")
    axes[1].set(xlabel="Measured mean speed magnitude (rad/s)", ylabel="|mean torque| / |mean speed| (N m s/rad)",
                title="This quotient includes Coulomb friction: it is NOT viscous B")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Rear drive: B constrained = {physical['B_nm_s_per_rad']:.6f}; "
                 f"moving friction = {physical['coulomb_nm']:.5f} N m\n"
                 "Airborne wheel; 13 s/hold; Kt = 0.300000012 N m/A; small positive B unresolved", fontsize=12)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output/f"drive_friction_fit.{suffix}", dpi=180)
    plt.close(fig)
    rx = d[d.record_type == "rx"]
    a2, a1 = rx[rx.opcode_hex == "A2"], rx[rx.opcode_hex == "A1"]
    command = d[d.record_type == "command"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    axes[0].plot(command.elapsed_s, command.encoded_command_rad_s, color="black", lw=1, label="C1 target")
    axes[0].plot(a2.elapsed_s.iloc[::3], a2.velocity_rad_s.iloc[::3], lw=.5, alpha=.7, label="Fresh A2 (every third for display)")
    axes[0].set(ylabel="Speed (rad/s)", title="Full recording; shaded portions used for friction fit")
    axes[1].plot(a1.elapsed_s.iloc[::3], a1.torque_estimate_nm.iloc[::3], color="tab:orange", lw=.4,
                 alpha=.55, label="Fresh A1-derived torque (every third)")
    bins = a1.groupby(np.floor(a1.elapsed_s)).agg(t=("elapsed_s", "mean"), torque=("torque_estimate_nm", "mean"))
    axes[1].plot(bins.t, bins.torque, color="black", lw=1, label="1 s mean (display only)")
    axes[1].set(ylabel="Torque (N m)")
    status = rx[rx.opcode_hex == "AE"]
    axes[2].plot(status.elapsed_s, status.motor_temperature_c, color="tab:red", label="Temperature")
    axes[2].set(xlabel="Host elapsed time (s)", ylabel="Temperature (C)")
    voltage = axes[2].twinx()
    voltage.plot(status.elapsed_s, status.bus_voltage_v, color="tab:green", alpha=.5, label="Bus voltage")
    voltage.set_ylabel("Bus voltage (V)")
    for ax in axes:
        for z in g.itertuples():
            ax.axvspan(z.window_start_s, z.window_end_s, color="tab:green", alpha=.08)
        ax.axvline(result["stop_detail"]["timeout_check_elapsed_s"], color="tab:red", ls="--")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("20261020_154728 rear C1 sweep: A1 timeout at 576.061 s, not a B3 tolerance failure")
    for suffix in ("png", "svg"):
        fig.savefig(output/f"sweep_overview.{suffix}", dpi=170)
    plt.close(fig)


def write_report(g, result, output):
    b, physical = result["unconstrained_fit"], result["physical_nonnegative_fit"]
    ci = b["conditional_95pct_t_intervals"][0]
    bootstrap_ci = result["conditional_bootstrap"]["B_percentile_2_5_50_97_5"]
    lines = ["# Rear drive friction/damping identification — 2026-10-20", "", "## Conclusion", "",
             f"The nonnegative viscous-plus-Coulomb fit gives **B = {physical['B_nm_s_per_rad']:.6f} N m s/rad**, "
             f"**moving-friction magnitude = {physical['coulomb_nm']:.6f} N m**, "
             f"and direction/bias offset = {physical['offset_nm']:.6f} N m.", "",
             "**B is not resolved as positive by this recording; this is NOT proof of exactly zero physical damping.** "
             "The main observed loss is approximately speed-independent over 1–8 rad/s. "
             "Do not assign all this drag to viscous friction or retain 0.05 as a measured passive coefficient.", "",
             f"Unconstrained slope **{b['B_nm_s_per_rad']:.9f} N m s/rad**; its negative sign is an estimation result, "
             f"not physical negative damping. Conditional 95% t interval **[{ci[0]:.7f}, {ci[1]:.7f}]** "
             "uses 19 plateau means and 16 residual degrees of freedom. "
             f"Stratified plateau bootstrap (10,000 resamples, seed42) gives [{bootstrap_ci[0]:.7f}, {bootstrap_ci[2]:.7f}]. "
             "These are descriptive regression intervals, NOT calibrated physical bounds: one full repeat plus nine holds, "
             "thermal/order dependence, torque scaling uncertainty and systematic losses remain.", "",
             "## Conditions and data contract", "",
             "User confirms rear wheel completely airborne, unconstrained and free to rotate. "
             "Source: ../../logs/20261020_154728_360275_drive_speed_sweep.csv. "
             f"{result['rows']:,} CSV rows / {result['duration_s']:.6f} s; 19 complete 15 s holds "
             "(repeat1 all10, repeat2 first9). Third repeat and second repeat's -8 rad/s hold were not reached.", "",
             "Raw A1/A2/B0/AE/CF bytes independently re-decoded and compared to logged units/signs. "
             "Use only record_type=rx, opcode_hex=A1 for current/torque and A2 for velocity. "
             "Other rows repeat cached state and are not independent samples. "
             "Native current and speed signs are both -1, protocol/output ratio1; no extra 8:1 scaling. "
             "A1 is signed 0.001 A, A2 signed 0.01 rpm, B0 float32 torque constant. "
             "AE bus current is unsigned protocol data and is NOT used as Q-axis current.", "",
             f"Kt={result['torque_constant_nm_per_a'][0]:.12f} N m/A; active B3 readback0.999 A; "
             f"voltage {result['voltage_range_v'][0]:.2f}–{result['voltage_range_v'][1]:.2f} V, "
             f"temperature {result['temperature_range_c'][0]:g}–{result['temperature_range_c'][1]:g} C. "
             "Motion AE status consistently mode3, fault0. "
             f"Observed C1/A2/A1 rates approximately {result['command_rate_hz']:.2f}/"
             f"{result['a2_observed_rate_hz']:.2f}/{result['a1_observed_rate_hz']:.2f} Hz, not exactly50/100/100.", "",
             "## Identification method", "",
             "Model at output shaft: J*domega/dt = Kt*iq - B*omega - Tc*sign(omega) - offset. "
             "For average steady motion with no external contact load, fit "
             "mean(torque)=B*mean(omega)+Tc*sign(mean(omega))+offset. "
             "The offset can absorb direction asymmetry/current bias; it is not independently identified as one physical mechanism.", "",
             "Discard ramps/settling/rest, the first and last1 s of each hold, and the interrupted -8 ramp. "
             f"Retain {result['retained_samples_a2']:,} fresh A2 and {result['retained_samples_a1']:,} fresh A1 replies. "
             "Average each signal separately on the same13 s windows. One equally weighted point per plateau "
             "avoids pseudoreplication and duplicate cached rows. Do not select only individual at_target samples "
             "or discard large currents: that would bias a closed-loop average.", "",
             "OLS fits all three terms. Physical fit enumerates the B>=0/Tc>=0 active sets, leaving offset free. "
             "No wheel inertia is assumed, but cancellation of the average inertial term is approximate, not exact. "
             "Residual speed ripple and finite-window endpoint motion contribute uncertainty.", "",
             f"Maximum plateau mean speed error {result['peak_plateau_speed_error_pct']:.2f}%; "
             f"within-window speed standard deviations {result['speed_std_range_rad_s'][0]:.3f}–"
             f"{result['speed_std_range_rad_s'][1]:.3f} rad/s. "
             "There is substantial instantaneous speed/current ripple despite approximately correct mean speed. "
             "This is effective running drag, not an isolated bearing or rotor material coefficient.", "",
             "## Evidence: pooled magnitudes", "",
             "| Nominal speed magnitude (rad/s) | Complete plateaus | Mean torque magnitude (N m) |", "|---:|---:|---:|"]
    for row in result["mean_absolute_torque_by_target"]:
        lines.append(f"| {row['magnitude']:g} | {row['plateaus']} | {row['mean_absolute_torque_nm']:.6f} |")
    lines += ["", "Full per-plateau measured speeds, torques, ripple, temperature and current-ceiling fractions: [plateaus.csv](plateaus.csv).", "",
              "![Friction fit](drive_friction_fit.png)", "", "## Sensitivity and limits", "",
              "| Trim at EACH end (s) | Unconstrained B (N m s/rad) | Tc (N m) |", "|---:|---:|---:|"]
    for trim, fit in result["trim_sensitivity"].items():
        lines.append(f"| {trim} | {fit['B_nm_s_per_rad']:.9f} | {fit['coulomb_nm']:.6f} |")
    lines += ["", "Window sensitivity changes the tiny slope's sign; all estimates retain approximately0.10–0.11 N m moving drag. "
              "Repeat-only, direction-only, leave-one-plateau-out and bootstrap results are in [fit_summary.json](fit_summary.json). "
              f"Forcing a viscous-only fit through the origin gives B={result['positive_viscous_only_fit_B_nm_s_per_rad']:.6f}; "
              "that lumps Coulomb friction into B and is not the identified viscous coefficient.", "",
              "The old B=0.05 hypothesis alone predicts0.4 N m at8 rad/s; the observed mean drag there is about0.105 N m. "
              "It does not match this free-wheel friction curve. This comparison concerns passive B, not whether an "
              "intentional controller Kd=0.05 should be kept.", "",
              "Current-derived torque uses firmware Kt, not an independent shaft torque transducer. "
              "Absolute B/Tc estimates scale with its calibration and depend on how gearbox losses are represented. "
              "The test includes bearings/gearbox/wheel effects and cannot separate all losses, loaded tyre rolling resistance, "
              "static breakaway friction, motor electrical dynamics or a thermal/torque-speed envelope. "
              "Angle-dependent cogging/gear ripple or wheel imbalance can also bias finite windows that do not span "
              "an integer number of revolutions. This experiment does not isolate each contribution. "
              "Reported B3 is a configured ceiling, not proof that measured current never overshoots it; "
              "per-plateau excursions are retained in the CSV, not removed from the mean.", "",
              "## Why recording stopped", "",
              f"One A1 query at {result['stop_detail']['request_elapsed_s']:.9f} s timed out; "
              f"the host checked the20 ms deadline at {result['stop_detail']['timeout_check_elapsed_s']:.9f} s "
              f"({result['stop_detail']['host_request_to_timeout_check_ms']:.3f} ms after send). "
              f"The A1 reply was then received during shutdown at {result['stop_detail']['late_receipt_elapsed_s']:.9f} s "
              f"({result['stop_detail']['host_request_to_late_receipt_ms']:.3f} ms after send). "
              "This is NOT a B3 tolerance failure. Host timestamps cannot establish whether the delay occurred in "
              "the motor/CAN/USB path or the Pi's scheduler/logging; do not call it measured motor latency. "
              "The first two shutdown CF replies still reported mode3; a later CF plus AE confirmed mode0/fault0. "
              "Original B3/CD values were subsequently read back and final CF again reported mode0. "
              "The valid earlier holds remain usable.", "",
              "![Sweep overview](sweep_overview.png)", "", "## Simulation interpretation / next step", "",
              "If a simple model must be tried, this dataset supports a PROVISIONAL B=0 baseline together with "
              f"moving friction about{physical['coulomb_nm']:.3f} N m, not a lossless rear wheel. "
              "B belongs in passive viscous friction; moving friction is a separate term. "
              "Static friction was not identified, and a negative fitted B must never be installed. "
              "Controller damping is a separate design decision. No Isaac Lab, MuJoCo, deployment or motor configuration was edited.", "",
              "For a tighter small-B estimate, repeat in randomized speed/direction order at matched temperature, "
              "average longer intervals covering many ripple cycles, verify torque/current calibration, "
              "and collect A3 position if per-revolution/angle-synchronous averaging is desired, "
              "and compare the fitted law with an independent recording. A calibrated inertia/coast-down or "
              "transient-torque validation could test the approximation; do not infer J from the current steady data.", "",
              "## Reproduce and artifacts", "", "```bash", "MPLCONFIGDIR=/tmp/drive-damping-matplotlib python3 \\",
              "  deploy/experiments/reports/drive_damping_20261020_154728/analysis.py", "```", "",
              "Requires numpy/pandas/matplotlib; no SciPy, ROS or CAN access. Original input and runtime sources "
              "are fingerprinted in JSON. The captured CSV is read-only. "
              "The environment's system SciPy has a NumPy ABI mismatch, so the analysis does not import it; "
              "the df16 t critical was independently obtained with Isaac Python SciPy.", "",
              "[HTML](index.html) · [fit PNG](drive_friction_fit.png) · [fit SVG](drive_friction_fit.svg) · "
              "[fit PDF](drive_friction_fit.pdf) · [overview SVG](sweep_overview.svg) · "
              "[summary JSON](fit_summary.json) · [analysis source](analysis.py) · [tests](test_analysis.py)"]
    (output/"README.md").write_text("\n".join(lines)+"\n")
    (output/"index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>Rear-drive friction: 20261020</title><style>body{font:17px system-ui;max-width:1400px;margin:2em auto;'
        'padding:0 1em}img{width:100%}</style><h1>Rear drive friction / damping identification</h1>'
        f'<p>19 complete plateaus. Constrained B = {physical["B_nm_s_per_rad"]:.6f} N m s/rad; '
        f'moving friction = {physical["coulomb_nm"]:.6f} N m. Small positive B is unresolved, not proved zero.</p>'
        '<p><a href="README.md">Full report and caveats</a> | <a href="plateaus.csv">Plateau data</a> | '
        '<a href="fit_summary.json">Numerical summary</a></p>'
        '<img src="drive_friction_fit.png" alt="Measured torque versus speed and apparent torque/speed ratio">'
        '<img src="sweep_overview.png" alt="Recorded command, speed, torque and thermal conditions"></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    source_hash = fingerprint(args.input)
    d = pd.read_csv(args.input, usecols=COLS, low_memory=False)
    checks = raw_checks(d)
    g = plateau_means(d)
    assert len(g) == 19, "This report expects the captured19 complete holds; inspect changed inputs separately"
    result = summarize(d, g, checks)
    result["provenance_sha256"] = {str(p.relative_to(ROOT)): fingerprint(p) for p in (
        args.input, Path(__file__), ROOT/"deploy/experiments/drive_speed_sweep.py",
        ROOT/"deploy/source/trackstand_deploy/trackstand_deploy/can_protocol.py")}
    args.output.mkdir(parents=True, exist_ok=True)
    g.to_csv(args.output/"plateaus.csv", index=False, float_format="%.12g")
    save_plots(d, g, result, args.output)
    write_report(g, result, args.output)
    assert fingerprint(args.input) == source_hash
    (args.output/"fit_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: result[key] for key in ("complete_hold_plateaus", "retained_samples_a1", "retained_samples_a2",
                     "unconstrained_fit", "physical_nonnegative_fit", "conditional_bootstrap", "stop_detail")}, indent=2))


if __name__ == "__main__":
    main()
