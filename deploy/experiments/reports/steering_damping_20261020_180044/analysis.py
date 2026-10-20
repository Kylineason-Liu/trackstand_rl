#!/usr/bin/env python3
"""Offline steering-friction identification; reads CSV only, never opens CAN.

Uses the rear report's identical least-squares/nonnegative fitting functions.
Run from any directory with numpy/pandas/matplotlib installed. Output defaults
to this report directory; no simulator, deployment or motor settings are edited.
"""

import argparse
import importlib.util
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
SOURCE = ROOT / "deploy/experiments/logs/20261020_180044_997520_steering_speed_sweep.csv"
REAR_ANALYSIS = HERE.parent / "drive_damping_20261020_154728/analysis.py"
spec = importlib.util.spec_from_file_location("rear_friction_analysis", REAR_ANALYSIS)
rear = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rear)
fit_loss, fingerprint = rear.fit_loss, rear.fingerprint
T_CRITICAL_DF39 = 2.022690920036761  # scipy.stats.t.ppf(.975, 39), installed Isaac Python.


def validate_raw(d):
    """Independently decode every RX and outgoing C1, including raw A3 counts."""
    counts = {}
    lengths = {"A1": 5, "A2": 5, "A3": 7, "B0": 7, "B5": 5, "AE": 8, "CF": 8, "C1": 5}
    # Scalar comparisons avoid a costly NumPy assertion for every CAN frame.
    def check(actual, expected):
        if np.isnan(actual) and np.isnan(expected):
            return
        if not np.isclose(actual, expected, rtol=1e-12, atol=1e-12):
            raise AssertionError(f"Raw decode mismatch: {actual} != {expected}")

    for row in d[d.record_type.isin(["rx", "command"])].itertuples():
        data = bytes.fromhex(row.payload_hex)
        op = row.opcode_hex
        assert len(data) == lengths[op] and data[0] == int(op, 16)
        assert int(row.can_id_hex, 16) == (row.motor_id if row.record_type == "rx" else 0x100 | row.motor_id)
        if row.record_type == "command":
            assert op == "C1"
            code = struct.unpack("<i", data[1:])[0]
            rpm = code * .01
            check(rpm, row.command_native_rpm)
            check(row.command_sign * rpm * 2*np.pi/60/row.gear_ratio, row.encoded_command_rad_s)
            assert abs(row.encoded_command_rad_s-row.command_speed_rad_s) <= .005*2*np.pi/60/row.gear_ratio+1e-12
            continue
        if op == "A1":
            current = struct.unpack("<i", data[1:])[0] * .001
            check(current, row.current_a)
            check(row.torque_sign * current * row.torque_constant_nm_per_a, row.torque_estimate_nm)
            check(row.current_age_s, 0.)
        elif op in ("A2", "C1"):
            rpm = struct.unpack("<i", data[1:])[0] * .01
            speed = row.encoder_sign * rpm * 2*np.pi/60/row.gear_ratio
            check(rpm, row.reply_native_rpm)
            check(speed, row.reply_velocity_rad_s)
            if op == "A2":
                check(speed, row.velocity_rad_s)
                check(row.velocity_age_s, 0.)
        elif op == "A3":
            single, count = struct.unpack("<Hi", data[1:])
            check(single, row.encoder_single_turn_count)
            check(count, row.encoder_count)
            delta = (count-int(row.zero_count)+2**31) % 2**32-2**31
            check(row.encoder_sign * delta * 2*np.pi/16384/row.gear_ratio, row.position_rad)
            check(row.position_age_s, 0.)
        elif op == "B0":
            check(struct.unpack("<f", data[2:6])[0], row.torque_constant_nm_per_a)
        elif op == "B5":
            check(struct.unpack("<I", data[1:])[0]*.01*2*np.pi/60/row.gear_ratio,
                  row.firmware_acceleration_rad_s2)
        elif op in ("AE", "CF"):
            voltage, current = struct.unpack("<HH", data[1:5])
            check(voltage*.01, row.bus_voltage_v)
            check(current*.01, row.bus_current_a)
            check(current, row.bus_current_raw)
            assert (data[5], data[6], data[7]) == (row.motor_temperature_c, row.motor_mode, row.fault_code)
        counts[op] = counts.get(op, 0)+1
    return counts


def window_mean(stream, column, start, end):
    """Time mean; interpolate boundaries, nearest-value extension at hold edges."""
    t, y = stream.elapsed_s.to_numpy(), stream[column].to_numpy()
    assert len(t) > 1 and end > start and np.all(np.diff(t) > 0)
    inside = (t > start) & (t < end)
    grid = np.r_[start, t[inside], end]
    return float(np.trapezoid(np.interp(grid, t, y), grid)/(end-start))


def plateau_means(d, hold=10., trim=1., full_turns=False):
    """One equal-weight point per complete hold; fresh RX rows only, no outlier cuts."""
    rows = []
    for (repeat, step), group in d[(d.record_type == "rx") & (d.phase == "hold")].groupby(["repeat", "step"]):
        if group.phase_elapsed_s.max() < hold-.05:
            continue
        phase_start = float(np.median(group.elapsed_s-group.phase_elapsed_s))
        start, end = phase_start+trim, phase_start+hold-trim
        q_all = group[group.opcode_hex == "A3"]
        q = q_all[(q_all.elapsed_s >= start) & (q_all.elapsed_s < end)]
        if full_turns:
            sign = np.sign(group.plateau_speed_rad_s.iloc[0])
            distance = sign*(q.position_rad.to_numpy()-q.position_rad.iloc[0])
            turns = int(np.floor(distance[-1]/(2*np.pi)))
            if turns < 1:
                continue
            index = int(np.flatnonzero(distance >= turns*2*np.pi)[0])
            start = float(q.elapsed_s.iloc[0])
            end = float(np.interp(turns*2*np.pi, distance[index-1:index+1], q.elapsed_s.iloc[index-1:index+1]))
        window = group[(group.elapsed_s >= start) & (group.elapsed_s < end)]
        v, a, q = [window[window.opcode_hex == op] for op in ("A2", "A1", "A3")]
        if min(len(v), len(a), len(q)) < 10:
            continue
        assert (v.velocity_age_s == 0).all() and (a.current_age_s == 0).all()
        assert v.elapsed_s.is_unique and a.elapsed_s.is_unique and q.elapsed_s.is_unique
        t, w = v.elapsed_s.to_numpy(), v.velocity_rad_s.to_numpy()
        qt, position = q.elapsed_s.to_numpy(), q.position_rad.to_numpy()
        q_slope = float(np.polyfit(qt-qt.mean(), position, 1)[0])
        a2_at_q = np.interp(qt, group[group.opcode_hex == "A2"].elapsed_s,
                            group[group.opcode_hex == "A2"].velocity_rad_s)
        integrated_speed = float(np.trapezoid(a2_at_q, qt)/(qt[-1]-qt[0]))
        row = dict(repeat=int(repeat), step=int(step), target_rad_s=float(group.plateau_speed_rad_s.iloc[0]),
                   window_start_s=start, window_end_s=end, n_a2=len(v), n_a1=len(a), n_a3=len(q),
                   speed_mean_rad_s=float(w.mean()), speed_std_rad_s=float(w.std(ddof=1)),
                   speed_linear_trend_rad_s2=float(np.polyfit(t-t.mean(), w, 1)[0]),
                   speed_time_mean_rad_s=window_mean(group[group.opcode_hex == "A2"], "velocity_rad_s", start, end),
                   torque_mean_nm=float(a.torque_estimate_nm.mean()), torque_std_nm=float(a.torque_estimate_nm.std()),
                   torque_time_mean_nm=window_mean(group[group.opcode_hex == "A1"], "torque_estimate_nm", start, end),
                   current_mean_a=float(a.current_a.mean()), current_peak_abs_a=float(a.current_a.abs().max()),
                   temperature_mean_c=float(a.motor_temperature_c.mean()),
                   a3_speed_regression_rad_s=q_slope,
                   a3_displacement_over_a2_integral=float((position[-1]-position[0])/(integrated_speed*(qt[-1]-qt[0]))),
                   sampled_rotation_turns=float(abs(position[-1]-position[0])/(2*np.pi)),
                   endpoint_speed_change_rad_s=float(v[v.elapsed_s >= end-1].velocity_rad_s.mean()
                                                    - v[v.elapsed_s < start+1].velocity_rad_s.mean()))
        row["speed_error_pct"] = 100*(row["speed_mean_rad_s"]-row["target_rad_s"])/abs(row["target_rad_s"])
        rows.append(row)
    return pd.DataFrame(rows)


def fit_table(g, speed="speed_mean_rad_s", torque="torque_mean_nm", constrained=False):
    return fit_loss(g[speed], g[torque], constrained)


def summarize(d, g, args, raw_counts):
    fit = fit_table(g)
    assert len(g) == 42  # Fixed captured experiment; don't silently overwrite it with another dataset.
    coefficients = np.array([fit[k] for k in ("B_nm_s_per_rad", "coulomb_nm", "offset_nm")])
    se = np.array(fit["standard_errors"])
    fit["conditional_95pct_t_intervals"] = np.column_stack(
        (coefficients-T_CRITICAL_DF39*se, coefficients+T_CRITICAL_DF39*se)).tolist()
    fit["residual_degrees_of_freedom"] = 39
    rx = d[d.record_type == "rx"]
    status = rx[(rx.opcode_hex == "AE") & (rx.phase != "shutdown")]
    commands = d[d.record_type == "command"]
    a1 = rx[rx.opcode_hex == "A1"]
    peak = a1.loc[a1.current_a.abs().idxmax()]
    pooled = g.assign(magnitude=g.target_rad_s.abs(), abs_torque=g.torque_mean_nm.abs()).groupby("magnitude").agg(
        plateaus=("repeat", "size"), mean_absolute_torque_nm=("abs_torque", "mean"))
    result = dict(rows=len(d), duration_s=float(d.elapsed_s.max()), startup_arguments=args,
                  user_confirmed_condition="Front wheel fully airborne, unconstrained, no hand contact/cable tightening/stops",
                  raw_validated_reply_counts=raw_counts, input_rows_by_type=d.record_type.value_counts().to_dict(),
                  complete_hold_plateaus=len(g), retained_samples_a1=int(g.n_a1.sum()),
                  retained_samples_a2=int(g.n_a2.sum()), retained_samples_a3=int(g.n_a3.sum()),
                  fit_window="Fresh A1/A2 separately averaged over each complete hold's [1,9) seconds; 42 equal-weight points",
                  torque_constant_nm_per_a=sorted(rx.torque_constant_nm_per_a.dropna().unique().tolist()),
                  unconstrained_fit=fit, physical_nonnegative_fit=fit_table(g, constrained=True),
                  conditional_bootstrap=rear.bootstrap(g),
                  by_repeat={str(r): fit_table(z) for r, z in g.groupby("repeat")},
                  by_direction={str(int(s)): dict(zip(("B_nm_s_per_rad", "intercept_nm"),
                                np.polyfit(z.speed_mean_rad_s.abs(), z.torque_mean_nm.abs(), 1).tolist()))
                                for s, z in g.groupby(np.sign(g.target_rad_s))},
                  leave_one_plateau_out_B=[fit_table(g.drop(i))["B_nm_s_per_rad"] for i in g.index],
                  trim_sensitivity={str(t): fit_table(plateau_means(d, args["hold"], t)) for t in range(5)},
                  time_weighted_fit=fit_table(g, "speed_time_mean_rad_s", "torque_time_mean_nm"),
                  a3_speed_fit=fit_table(g, "a3_speed_regression_rad_s"),
                  full_turn_window_fit=fit_table(plateau_means(d, args["hold"], full_turns=True)),
                  speed_subset_fits={str(c): fit_table(g[g.target_rad_s.abs() >= c]) for c in (1, 2, 4, 6)},
                  mean_absolute_torque_by_target=pooled.reset_index().to_dict("records"),
                  pure_viscous_only_B_nm_s_per_rad=float(np.dot(g.speed_mean_rad_s, g.torque_mean_nm)
                                                       /np.dot(g.speed_mean_rad_s, g.speed_mean_rad_s)),
                  peak_plateau_mean_speed_error_pct=float(g.speed_error_pct.abs().max()),
                  speed_std_range_rad_s=[float(g.speed_std_rad_s.min()), float(g.speed_std_rad_s.max())],
                  max_plateau_endpoint_speed_change_rad_s=float(g.endpoint_speed_change_rad_s.abs().max()),
                  a3_over_a2_displacement_ratio=dict(mean=float(g.a3_displacement_over_a2_integral.mean()),
                      median=float(g.a3_displacement_over_a2_integral.median()),
                      min=float(g.a3_displacement_over_a2_integral.min()), max=float(g.a3_displacement_over_a2_integral.max())),
                  voltage_range_v=[float(status.bus_voltage_v.min()), float(status.bus_voltage_v.max())],
                  temperature_range_c=[float(status.motor_temperature_c.min()), float(status.motor_temperature_c.max())],
                  motion_status_modes=sorted(status.motor_mode.unique().tolist()),
                  motion_status_faults=sorted(status.fault_code.unique().tolist()),
                  peak_sampled_current=dict(elapsed_s=float(peak.elapsed_s), phase=peak.phase,
                      target_rad_s=float(peak.plateau_speed_rad_s), current_a=float(peak.current_a),
                      current_based_torque_nm=float(peak.torque_estimate_nm)),
                  peak_fitted_window_current_a=float(g.current_peak_abs_a.max()),
                  firmware_b5_acceleration_readback_rad_s2=sorted(rx.firmware_acceleration_rad_s2.dropna().unique().tolist()),
                  unknown_firmware_settings="B3 current ceiling/CD watchdog/gains not read by this recording",
                  completion_events=d[(d.record_type == "event") & d.note.str.contains(
                      "sweep completed|CF disable acknowledged", na=False)][["elapsed_s", "note"]].to_dict("records"),
                  error_rows=int(d.record_type.isin(["can_error", "rx_error"]).sum()),
                  reported_kernel_drop_rows=int(d.note.str.contains("kernel cumulative RX drops", na=False).sum()))
    for op in ("A1", "A2", "A3", "C1", "AE", "B0"):
        stream = commands if op == "C1" else rx[rx.opcode_hex == op]
        dt = np.diff(stream.elapsed_s)
        result[f"{op.lower()}_timing"] = dict(observed_rate_hz=float(1/dt.mean()),
                                               maximum_gap_ms=float(dt.max()*1000))
    result["tx_rx_counts"] = {
        op: dict(sent=int(((d.record_type.isin(["tx", "command"])) & (d.opcode_hex == op)).sum()), received=int(n))
        for op, n in raw_counts.items()}
    assert all(counts["sent"] == counts["received"] for counts in result["tx_rx_counts"].values())
    assert result["error_rows"] == result["reported_kernel_drop_rows"] == 0
    assert result["motion_status_faults"] == [0.]
    return result


def save_plots(d, g, result, output):
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .22})
    b = result["physical_nonnegative_fit"]
    palette, markers = {1: "tab:blue", -1: "tab:orange"}, {1: "o", 2: "s", 3: "^"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for (repeat, sign), z in g.groupby([g.repeat, np.sign(g.target_rad_s)]):
        label = f"Repeat {repeat}, {'+' if sign > 0 else '-'} direction"
        axes[0].scatter(z.speed_mean_rad_s, z.torque_mean_nm, marker=markers[repeat], s=48,
                        color=palette[sign], edgecolor="black", lw=.4, label=label)
        axes[1].scatter(z.speed_mean_rad_s.abs(), z.torque_mean_nm.abs()/z.speed_mean_rad_s.abs(),
                        marker=markers[repeat], s=48, color=palette[sign])
    for sign in (-1, 1):
        w = sign*np.linspace(.7, 12.3, 100)
        axes[0].plot(w, b["B_nm_s_per_rad"]*w+b["coulomb_nm"]*sign+b["offset_nm"], "k-",
                     label="B*w + Tc*sign(w) + offset" if sign == 1 else None)
        axes[1].plot(abs(w), abs(b["B_nm_s_per_rad"]*w+b["coulomb_nm"]*sign+b["offset_nm"])/abs(w),
                     color=palette[sign], alpha=.5)
    axes[0].set(xlabel="Mean A2 joint speed (rad/s)", ylabel="Mean A1 x B0 signed torque (N m)",
                title="42 complete holds: positive slope plus nearly constant drag")
    axes[0].legend(fontsize=8)
    axes[1].axhline(b["B_nm_s_per_rad"], color="black", ls="--", label="Fitted viscous B only")
    axes[1].set(xlabel="Mean speed magnitude (rad/s)", ylabel="|mean torque| / |mean speed| (N m s/rad)",
                title="Torque/speed includes Coulomb drag; it is NOT B")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Steering: B = {b['B_nm_s_per_rad']:.7f} N m s/rad; Tc = {b['coulomb_nm']:.5f} N m\n"
                 "Airborne, unconstrained; 8 s per hold; current-derived torque, not a shaft sensor")
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output/f"steering_friction_fit.{suffix}", dpi=180)
    plt.close(fig)

    rx = d[d.record_type == "rx"]
    a1, a2 = [rx[rx.opcode_hex == op] for op in ("A1", "A2")]
    commands = d[d.record_type == "command"]
    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True, constrained_layout=True)
    axes[0].plot(commands.elapsed_s, commands.encoded_command_rad_s, "k-", lw=.8, label="C1 target")
    axes[0].plot(a2.elapsed_s.iloc[::3], a2.velocity_rad_s.iloc[::3], lw=.5, alpha=.7, label="Fresh A2 (every 3rd)")
    axes[0].set(ylabel="Speed (rad/s)")
    show = a1.iloc[np.unique(np.r_[np.arange(0, len(a1), 3), np.argmax(a1.current_a), np.argmin(a1.current_a)])]
    axes[1].plot(show.elapsed_s, show.torque_estimate_nm, color="tab:orange", lw=.5,
                 label="Current-derived torque (every 3rd + extrema)")
    axes[1].set(ylabel="Torque (N m)", title="Includes step transients; these are excluded from the steady-friction fit")
    axes[2].scatter((g.window_start_s+g.window_end_s)/2, g.torque_mean_nm, s=18, color="tab:blue", label="8 s hold mean")
    axes[2].set(ylabel="Mean torque (N m)", title="Separate steady-window scale")
    status = rx[rx.opcode_hex == "AE"]
    axes[3].plot(status.elapsed_s, status.motor_temperature_c, color="tab:red", label="Temperature")
    axes[3].set(xlabel="Host elapsed time (s)", ylabel="Temperature (C)")
    twin = axes[3].twinx()
    twin.plot(status.elapsed_s, status.bus_voltage_v, color="tab:green", alpha=.6, label="Bus voltage")
    twin.set_ylabel("Bus voltage (V)")
    twin.legend(loc="lower left", fontsize=8)
    for ax in axes:
        for row in g.itertuples():
            ax.axvspan(row.window_start_s, row.window_end_s, color="tab:green", alpha=.07)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("20261020_180044 steering C1 sweep: all 3 repeats completed; green = fitted windows")
    for suffix in ("png", "svg"):
        fig.savefig(output/f"sweep_overview.{suffix}", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for repeat, z in g.groupby("repeat"):
        for sign, points in z.groupby(np.sign(z.target_rad_s)):
            axes[0, 0].plot(points.speed_mean_rad_s.abs(), points.torque_mean_nm.abs(),
                            marker=markers[repeat], color=palette[sign], lw=.7, alpha=.8,
                            label=f"Repeat {repeat}, {'+' if sign > 0 else '-'}")
        prediction = b["B_nm_s_per_rad"]*z.speed_mean_rad_s+b["coulomb_nm"]*np.sign(z.speed_mean_rad_s)+b["offset_nm"]
        axes[0, 1].scatter(z.temperature_mean_c, (z.torque_mean_nm-prediction)*1000,
                           marker=markers[repeat], label=f"Repeat {repeat}")
        axes[1, 0].scatter(z.target_rad_s, (z.a3_displacement_over_a2_integral-1)*100, marker=markers[repeat])
        axes[1, 1].scatter(z.speed_mean_rad_s.abs(), z.torque_std_nm, marker=markers[repeat], label=f"Repeat {repeat}")
    axes[0, 0].set(xlabel="Mean |speed| (rad/s)", ylabel="|Mean torque| (N m)", title="Mild low-speed nonlinearity; not a fitted Stribeck law")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].axhline(0, color="black", lw=.8)
    axes[0, 1].set(xlabel="Mean motor temperature (C)", ylabel="Simple-model residual (mN m)",
                   title="Ordered sweep: temperature/order are not independently controlled")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].set(xlabel="Signed target (rad/s)", ylabel="A3 displacement / integrated A2 - 1 (%)",
                   title="Approx. 1.6% position/speed consistency discrepancy")
    axes[1, 1].set(xlabel="Mean |speed| (rad/s)", ylabel="Instantaneous torque standard deviation (N m)",
                   title="Ripple is much larger than repeat-to-repeat mean variation")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Steering friction: diagnostics and identification limits")
    for suffix in ("png", "svg"):
        fig.savefig(output/f"identification_diagnostics.{suffix}", dpi=170)
    plt.close(fig)


def write_report(g, result, output):
    b = result["physical_nonnegative_fit"]
    ci = result["unconstrained_fit"]["conditional_95pct_t_intervals"][0]
    boot = result["conditional_bootstrap"]["B_percentile_2_5_50_97_5"]
    peak = result["peak_sampled_current"]
    ratio = result["a3_over_a2_displacement_ratio"]
    rear_summary = json.loads((REAR_ANALYSIS.parent/"fit_summary.json").read_text())
    drive = rear_summary["physical_nonnegative_fit"]
    lines = ["# Steering friction/damping identification — 2026-10-20", "", "## Conclusion", "",
             f"Using the rear-wheel report's identical three-term model and nonnegative fit: **B = {b['B_nm_s_per_rad']:.9f} N m s/rad**, "
             f"**moving Coulomb-like friction Tc = {b['coulomb_nm']:.6f} N m**, "
             f"and **signed offset = {b['offset_nm']:.6f} N m**.", "",
             "Unlike the rear-wheel sweep, this steering dataset resolves a small positive viscous slope within the simple regression model. "
             "It does not establish one exact, load-independent intrinsic motor constant. These are effective losses of the tested, "
             "powered, unconstrained steering assembly, inferred from motor current.", "",
             "```text", "J * acceleration = signed(Kt * iq) - B * speed - Tc * sign(speed) - offset",
             f"steady torque estimate = {b['B_nm_s_per_rad']:.9f} * speed + {b['coulomb_nm']:.6f} * sign(speed) + {b['offset_nm']:.6f}",
             "```", "",
             f"At 12 rad/s, the viscous component is only **{12*b['B_nm_s_per_rad']:.5f} N m**, "
             f"versus **{b['coulomb_nm']:.5f} N m** moving friction. Constant-like drag dominates over the tested 1–12 rad/s range. "
             "The offset may absorb current bias, asymmetric losses or ordered-test effects; it is not a separately identified friction mechanism.", "",
             "## Data and verified units", "",
             "User explicitly confirms the front wheel was fully airborne and steering unconstrained throughout, "
             "without hand contact, cable tightening or mechanical stops. "
             "Source: [20261020_180044_997520_steering_speed_sweep.csv](../../logs/20261020_180044_997520_steering_speed_sweep.csv). "
             f"**{result['rows']:,} rows / {result['duration_s']:.6f} s; all 42 ten-second holds completed**, "
             "three repeats of +1,+2,+4,+6,+8,+10,+12,-1,-2,-4,-6,-8,-10,-12 rad/s. "
             "Startup JSON confirms settle 5 / hold 10 / rest 2 s, C1 50 Hz, each A3/A2/A1 100 Hz, AE 5 Hz, B0 1 Hz. "
             "Final events report sweep completion and CF plus fresh AE disabled acknowledgement.", "",
             "Every raw RX payload and outgoing C1 payload is independently decoded and compared to CSV units/signs. "
             "A1 is signed int32, 0.001 A; A2/C1 feedback is signed int32, 0.01 rpm; "
             "A3 is uint16 single-turn plus int32 multi-turn, 16384 counts/revolution; B0 is float32 bytes2–5. "
             "Encoder/current/command signs are -1, gear1. No extra gearbox factor is invented. "
             f"All B0 replies agree on **Kt={result['torque_constant_nm_per_a'][0]:.12f} N m/A**. "
             "Torque is -A1_current*Kt, not AE bus current and not an independent shaft torque sensor. "
             "Raw byte definitions were checked against `tsinghua_bike_project/can_protocol.txt` and both sweep sources.", "",
             f"Observed C1 rate **{result['c1_timing']['observed_rate_hz']:.2f} Hz**, "
             f"A2/A1 rates **{result['a2_timing']['observed_rate_hz']:.2f}/{result['a1_timing']['observed_rate_hz']:.2f} Hz**. "
             f"Maximum A2 host receipt gap **{result['a2_timing']['maximum_gap_ms']:.2f} ms**. "
             "All sent read/command reply counts match over the complete recording; no logged CAN error, malformed RX, "
             "or kernel-drop rows. This is not proof of simultaneous samples or zero transport latency. "
             f"AE temperature **{result['temperature_range_c'][0]:g}–{result['temperature_range_c'][1]:g} C**, "
             f"voltage **{result['voltage_range_v'][0]:.2f}–{result['voltage_range_v'][1]:.2f} V**, fault0 throughout; "
             "mode3 during speed control, mode0 initially/after shutdown. B5 readback0 has no documented special-zero semantics. "
             "B3 current ceiling, CD watchdog and firmware gains were not read; their actual settings cannot be reconstructed here.", "",
             "## Same identification method as rear drive", "",
             "For each complete hold, discard its first and last1 s. Earlier5 s settling, speed steps and zero-speed rest are excluded. "
             "Average independent fresh A2 velocities and fresh A1-derived torques separately over the same [1,9) s window. "
             f"Retain **{result['retained_samples_a2']:,} fresh A2 / {result['retained_samples_a1']:,} fresh A1** samples; "
             "fit one equally weighted point per hold, not hundreds of thousands of cached CSV rows. "
             "No high-current/outlier/at-target filtering is used within selected windows. "
             "OLS fits B, Tc and an unconstrained signed offset; the physical fit enumerates B>=0/Tc>=0 active sets, exactly as in the rear report. "
             "Both fits coincide here because both fitted friction terms are positive. "
             "No inertia J is assumed: the steady-window average inertial term is neglected, not proven exactly zero.", "",
             f"Maximum mean A2 speed error **{result['peak_plateau_mean_speed_error_pct']:.3f}%**; "
             f"within-window speed SD **{result['speed_std_range_rad_s'][0]:.3f}–{result['speed_std_range_rad_s'][1]:.3f} rad/s**. "
             f"Largest end-minus-start one-second mean speed change **{result['max_plateau_endpoint_speed_change_rad_s']:.4f} rad/s**. "
             "This supports approximately steady averages, not ripple-free instantaneous motion.", "",
             "## Evidence and comparison", "",
             "| Parameter | Steering, this recording | Rear drive, previous recording |",
             "|---|---:|---:|",
             f"| Passive viscous B (N m s/rad), constrained fit | {b['B_nm_s_per_rad']:.9f} | {drive['B_nm_s_per_rad']:.6f} (positive B unresolved) |",
             f"| Moving Coulomb-like Tc (N m) | {b['coulomb_nm']:.6f} | {drive['coulomb_nm']:.6f} |",
             f"| Signed offset (N m) | {b['offset_nm']:.6f} | {drive['offset_nm']:.6f} |",
             f"| Per-plateau fit RMSE (N m) | {b['rmse_nm']:.6f} | {drive['rmse_nm']:.6f} |",
             "| Static breakaway friction | Not identified | Not identified |", "",
             "Different assemblies, torque constants, temperatures and test protocols: this is a descriptive comparison, not a controlled motor contest.", "",
             "| Nominal speed magnitude (rad/s) | Holds | Mean torque magnitude (N m) |",
             "|---:|---:|---:|"]
    for row in result["mean_absolute_torque_by_target"]:
        lines.append(f"| {row['magnitude']:g} | {row['plateaus']} | {row['mean_absolute_torque_nm']:.6f} |")
    lines += ["", "[Full plateau statistics](plateaus.csv).", "", "![Friction fit](steering_friction_fit.png)", "",
              "## Repeatability, uncertainty and nonlinear losses", "",
              "| Repeat | B (N m s/rad) | Tc (N m) | Offset (N m) |", "|---:|---:|---:|---:|"]
    for repeat, fit in result["by_repeat"].items():
        lines.append(f"| {repeat} | {fit['B_nm_s_per_rad']:.9f} | {fit['coulomb_nm']:.6f} | {fit['offset_nm']:.6f} |")
    lines += ["", f"Conditional OLS 95% t interval for B: **[{ci[0]:.9f}, {ci[1]:.9f}]** "
              "(42 plateau means,39 residual degrees of freedom). "
              f"Stratified plateau bootstrap,10,000 draws/seed42,2.5–97.5%: **[{boot[0]:.9f}, {boot[2]:.9f}]**. "
              "These are descriptive regression intervals, not calibrated physical bounds: current/Kt scaling, "
              "correlated ordered samples, angle ripple, torque bias and thermal dependence are not fully represented.", "",
              "| Trim at EACH hold end (s) | B (N m s/rad) | Tc (N m) |", "|---:|---:|---:|"]
    for trim, fit in result["trim_sensitivity"].items():
        lines.append(f"| {trim} | {fit['B_nm_s_per_rad']:.9f} | {fit['coulomb_nm']:.6f} |")
    lines += ["", f"Time-weighted windows give B={result['time_weighted_fit']['B_nm_s_per_rad']:.9f}, "
              f"Tc={result['time_weighted_fit']['coulomb_nm']:.6f}; "
              f"integer-A3-turn window sensitivity gives B={result['full_turn_window_fit']['B_nm_s_per_rad']:.9f}, "
              f"Tc={result['full_turn_window_fit']['coulomb_nm']:.6f}. "
              "Those extra windows are defined using the documented encoder conversion, not independent angular calibration.", "",
              "Mean drag first decreases slightly from1 to4 rad/s, then increases. This can be compatible with "
              "Stribeck-like behavior, angle-dependent cogging/gearing, finite-revolution averaging, current-loop ripple or ordered heating; "
              "the recording cannot uniquely distinguish them or identify a trustworthy Stribeck/static-friction law. "
              f"Using only |speed|>=4 gives B={result['speed_subset_fits']['4']['B_nm_s_per_rad']:.9f}, "
              f"Tc={result['speed_subset_fits']['4']['coulomb_nm']:.6f}. "
              "Therefore the simple coefficients depend on the fitted speed band. Never install all optional terms as additive losses without a joint identification. "
              f"A forced viscous-only through-origin fit gives B={result['pure_viscous_only_B_nm_s_per_rad']:.6f}; "
              "that folds Coulomb friction into B and is not the separated viscous coefficient.", "",
              "## Important additional observations", "",
              f"- **Position/speed consistency:** per-window A3 displacement divided by integrated A2 has median "
              f"**{ratio['median']:.6f}**, mean {ratio['mean']:.6f}, range {ratio['min']:.6f}–{ratio['max']:.6f}. "
              "Both signs agree, but documented A3 units imply about1.6% more rotation than A2. Raw decode checks pass; "
              "which feedback/scale/clock is physically more accurate is not established. Do not silently correct a gear ratio or count resolution. "
              f"Replacing A2 means with A3 position-regression speeds gives B={result['a3_speed_fit']['B_nm_s_per_rad']:.9f}, "
              f"Tc={result['a3_speed_fit']['coulomb_nm']:.6f}; the central friction conclusion survives this small difference.",
              f"- **Step transients are not friction:** the largest sampled A1 magnitude is **{abs(peak['current_a']):.3f} A** "
              f"at {peak['elapsed_s']:.6f} s ({peak['phase']}, leaving a {peak['target_rad_s']:g} rad/s step), "
              f"equivalent to **{abs(peak['current_based_torque_nm']):.3f} N m** under firmware Kt. "
              f"The maximum within fitted windows is only {result['peak_fitted_window_current_a']:.3f} A. "
              "The full-record voltage minimum 18.59 V is also a transition sample, not a steady loss measurement. "
              "These are sampled current-derived values, not independently validated shaft torques, firmware current limits or safe continuous ratings. "
              "Abrupt C1 transitions are not recommended evidence of safe repeated stress. No new hardware tests were run here.",
              "- **Not identified:** static/breakaway friction, very-low-speed friction, load/tyre-ground losses, separate rotor/gear/bearing friction, "
              "J, torque-speed envelope, peak-torque duration, electrical dynamics and true torque calibration. Enabled C1 zero-speed rest "
              "actively regulates current and jitters; it is not a passive breakaway test.", "",
              "![Identification diagnostics](identification_diagnostics.png)", "",
              "## Simulation interpretation / next verification", "",
              f"A provisional passive model over1–12 rad/s is **viscous_friction≈{b['B_nm_s_per_rad']:.6f} N m s/rad** plus "
              f"**dynamic/moving friction≈{b['coulomb_nm']:.3f} N m**. "
              "Static friction must remain explicitly unmeasured; using an equal static threshold to satisfy a simulator's static>=dynamic "
              "requirement would be a provisional modelling assumption, not a result of this test. The small offset should not automatically become a constant external torque.", "",
              "These losses are separate from steering controller Kd=0.2. Do not replace the working MIT Kd with the fitted B, "
              "nor add passive resistance again on the real Pi. If adding passive friction in simulation, keep controller terms distinct "
              "and check that asset-native losses do not already duplicate the identified assembly losses. "
              "No Isaac Lab, MuJoCo, deployment, sweep runtime, firmware or progress-record settings were changed in this analysis.", "",
              "For tighter identification: verify A3 counts and A2 RPM against independently observed rotations; verify Kt/current calibration; "
              "repeat randomized speed/direction tests at stable temperature with safe acceleration/current settings; "
              "add small near-zero steady speeds and a separately guarded slow breakaway test if static/Stribeck friction is needed. "
              "Validate the fitted model on an independent recording and, if J is known, a controlled coast-down/transient test. "
              "No commanded-speed latency or motor internal-loop frequency is inferred from these averages.", "",
              "![Sweep overview](sweep_overview.png)", "", "## Reproduce and artifacts", "",
              "```bash", "MPLCONFIGDIR=/tmp/steering-damping-matplotlib python3 \\",
              "  deploy/experiments/reports/steering_damping_20261020_180044/analysis.py", "",
              "python3 -B -m unittest discover \\",
              "  -s deploy/experiments/reports/steering_damping_20261020_180044 -p 'test_analysis.py' -v", "```", "",
              "Requires numpy/pandas/matplotlib, no SciPy, CAN or ROS. Mathematical fit/bootstrap helpers are reused from "
              "the read-only rear report. Input CSV, helper, protocol and runtime source SHA256 fingerprints are retained in JSON; "
              "input and runtime file hashes are checked unchanged after analysis. Conditional df39 t critical was evaluated separately "
              "using the installed Isaac Python SciPy, not a hard-coded normal approximation.", "",
              "[HTML](index.html) · [fit PNG](steering_friction_fit.png) · [SVG](steering_friction_fit.svg) · [PDF](steering_friction_fit.pdf) · "
              "[plateaus](plateaus.csv) · [numerical JSON](fit_summary.json) · [source](analysis.py) · [tests](test_analysis.py)"]
    (output/"README.md").write_text("\n".join(lines)+"\n")
    (output/"index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>Steering friction / damping, 20261020</title><style>body{font:17px system-ui;max-width:1400px;'
        'margin:2em auto;padding:0 1em}img{width:100%}</style><h1>Steering friction / damping identification</h1>'
        f'<p>42 complete holds. B={b["B_nm_s_per_rad"]:.9f} N m s/rad; moving friction={b["coulomb_nm"]:.6f} N m. '
        'Airborne, unconstrained; current-derived estimates. Static friction is not measured.</p>'
        '<p><a href="README.md">Full report / caveats</a> | <a href="plateaus.csv">Plateaus</a> | '
        '<a href="fit_summary.json">Numerical summary</a></p>'
        '<img src="steering_friction_fit.png" alt="Steering friction model and independent plateau means">'
        '<img src="identification_diagnostics.png" alt="Repeatability, nonlinearity, temperature and A3 versus A2">'
        '<img src="sweep_overview.png" alt="Full sweep including large step transients"></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    protected = [args.input, REAR_ANALYSIS, REAR_ANALYSIS.parent/"fit_summary.json",
                 ROOT/"deploy/experiments/steering_speed_sweep.py", ROOT/"deploy/experiments/drive_speed_sweep.py",
                 ROOT/"deploy/source/trackstand_deploy/trackstand_deploy/can_protocol.py",
                 ROOT/"tsinghua_bike_project/can_protocol.txt"]
    before = {str(p.relative_to(ROOT)): fingerprint(p) for p in protected}
    d = pd.read_csv(args.input, dtype={"can_id_hex": str}, low_memory=False)
    assert d.elapsed_s.is_monotonic_increasing
    config = json.loads(d.iloc[0].note)
    assert config["motor_id"] == 2 and config["hold"] == 10 and config["repeats"] == 3
    assert (d.motor_id == 2).all() and (d.gear_ratio == 1).all()
    assert (d[["encoder_sign", "command_sign", "torque_sign"]] == -1).all().all()
    raw_counts = validate_raw(d)
    g = plateau_means(d, config["hold"])
    result = summarize(d, g, config, raw_counts)
    result["provenance_sha256"] = before
    args.output.mkdir(parents=True, exist_ok=True)
    g.to_csv(args.output/"plateaus.csv", index=False, float_format="%.12g")
    save_plots(d, g, result, args.output)
    write_report(g, result, args.output)
    for p in protected:
        assert fingerprint(p) == before[str(p.relative_to(ROOT))], f"Protected input changed: {p}"
    result["analysis_source_sha256"] = fingerprint(Path(__file__))
    (args.output/"fit_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: result[k] for k in ("rows", "complete_hold_plateaus", "retained_samples_a1", "retained_samples_a2",
                                          "raw_validated_reply_counts", "unconstrained_fit", "physical_nonnegative_fit",
                                          "conditional_bootstrap", "full_turn_window_fit", "a3_speed_fit",
                                          "peak_sampled_current")}, indent=2))


if __name__ == "__main__":
    main()
