#!/usr/bin/env python3
"""PC-only effective C1 PI identification; no motor access or firmware-unit guess.

Joint P/I fits use measured current, velocity, unwrapped position and actual sent
C1 commands. Time-domain fits include per-trial offsets and a delay sweep, with
complete trials held out. Frequency-domain fits independently estimate I/error
and fit PI + delay + optional first-order current/measurement lag. Units: A/(rad/s)
and A/rad; multiply by measured B0 Kt for Nm/(rad/s) and Nm/rad.
"""

import argparse
import csv
import hashlib
from html import escape
import json
import math
from pathlib import Path
import struct

import numpy as np


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_capture(path):
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta.get("schema") != "drive_c1_kp_identify_v1":
        raise ValueError("expected this experiment's CSV and JSON, not a velocity replay/sine log")
    phases = {p["segment"]["name"]: dict(p, streams={k: [] for k in ("command", "current", "velocity", "position")})
              for p in meta.get("phases", [])}
    gain_writes, faults = [], set()
    readbacks = {"B8": set(), "B9": set()}
    last_count, travel = meta.get("initial_multi_turn_count"), 0
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            data = bytes.fromhex(row["payload_hex"]) if row["payload_hex"] else b""
            if not data:
                continue
            opcode = data[0]
            if row["record_type"] in ("tx", "command") and opcode in (0xB8, 0xB9) and len(data) > 1:
                gain_writes.append(row["host_monotonic_s"])
            if row["record_type"] == "rx" and opcode in (0xAE, 0xCF):
                faults.add(data[7])
            if row["record_type"] == "rx" and opcode in (0xB8, 0xB9):
                readbacks[f"{opcode:02X}"].add(data.hex())
            position_value = None
            if row["record_type"] == "rx" and opcode == 0xA3 and last_count is not None:
                count = struct.unpack("<i", data[3:7])[0]
                travel += (count-last_count+2**31) % 2**32-2**31
                last_count = count
                position_value = -travel*2*math.pi/16384
                assert abs(position_value-float(row["position_rad"])) < 1e-8
            if row["method"] not in phases:
                continue
            phase = phases[row["method"]]
            time = float(row["host_monotonic_s"])-phase["start_monotonic_s"]
            if row["record_type"] == "command" and opcode == 0xC1:
                value = -struct.unpack("<i", data[1:])[0]*.01*2*math.pi/60
                assert abs(value-float(row["encoded_target_velocity_rad_s"])) < 1e-12
                phase["streams"]["command"].append((time, value))
            elif row["record_type"] == "rx" and opcode in (0xA1, 0xA2, 0xA3):
                if opcode == 0xA1:
                    field, value = "current", -struct.unpack("<i", data[1:])[0]*.001
                elif opcode == 0xA2:
                    field, value = "velocity", -struct.unpack("<i", data[1:])[0]*.01*2*math.pi/60
                else:
                    if position_value is None:
                        continue
                    field, value = "position", position_value
                phase["streams"][field].append((time, value))
    if gain_writes:
        raise ValueError("B8/B9 writes found: this is not the required gain-read-only experiment")
    meta["analysis_gain_readback_values"] = {key: sorted(values) for key, values in readbacks.items()}
    usable, excluded = [], []
    b3 = bytes.fromhex(meta.get("original_settings", {}).get("B3", ""))
    configured_current = int.from_bytes(b3[1:5], "little")*.001 if len(b3) == 5 else math.inf
    kt = meta.get("torque_constant_nm_per_a")
    physical_current = 6.28/kt if kt is not None and math.isfinite(kt) and kt > 0 else math.inf
    fit_current_limit = .98*min(configured_current if configured_current > 0 else math.inf, physical_current)
    for phase in phases.values():
        if phase["actual_duration_s"]+1e-7 < phase["segment"]["duration_s"]:
            excluded.append(dict(name=phase["segment"]["name"], reason="partial segment"))
            continue
        if min(map(len, phase["streams"].values())) < 4:
            excluded.append(dict(name=phase["segment"]["name"], reason="insufficient fresh streams"))
            continue
        for key, values in phase["streams"].items():
            a = np.array(values, dtype=float)
            if not np.all(np.isfinite(a)) or not np.all(np.diff(a[:, 0]) > 0):
                raise ValueError(f"invalid {key} stream in {phase['segment']['name']}")
            phase["streams"][key] = (a[:, 0], a[:, 1])
        if np.mean(abs(phase["streams"]["current"][1]) >= fit_current_limit) > .01:
            excluded.append(dict(name=phase["segment"]["name"], reason="possible current/torque clipping"))
            continue
        usable.append(phase)
    return meta, usable, excluded, sorted(faults)


def held_integral(time, values, query):
    index = np.clip(np.searchsorted(time, query, side="right")-1, 0, len(time)-1)
    integral = np.r_[0., np.cumsum(values[:-1]*np.diff(time))]
    return integral[index]+values[index]*(query-time[index])


def paired(phase, delay):
    """Reconstruct error/integral at fresh A1 times; interpolation is analysis only."""
    ct, target = phase["streams"]["command"]
    vt, velocity = phase["streams"]["velocity"]
    pt, position = phase["streams"]["position"]
    it, current = phase["streams"]["current"]
    query = it-delay
    end = phase["segment"]["duration_s"]
    keep = (query >= max(ct[0], vt[0], pt[0], .03)) & (query <= min(vt[-1], pt[-1], end-.03))
    keep &= (it >= 0) & (it < end)
    # Large gaps affect analysis validity, never cause a runtime experiment abort.
    for t in (vt, pt):
        index = np.clip(np.searchsorted(t, query), 1, len(t)-1)
        keep &= t[index]-t[index-1] <= .1
    query, times, measured = query[keep], it[keep], current[keep]
    index = np.clip(np.searchsorted(ct, query, side="right")-1, 0, len(ct)-1)
    error = target[index]-np.interp(query, vt, velocity)
    integral_error = held_integral(ct, target, query)-(np.interp(query, pt, position)-position[0])
    return times, np.column_stack((error, integral_error)), measured


def sufficient(phase, delay):
    _, x, y = paired(phase, delay)
    if len(y) < 12:
        return None
    x, y = x-x.mean(axis=0), y-y.mean()
    return dict(g=x.T@x/len(y), h=x.T@y/len(y), yy=float(y@y/len(y)), n=len(y), phase=phase)


def solve(stats):
    if not stats:
        return None
    g = sum(s["g"] for s in stats)
    h = sum(s["h"] for s in stats)
    if min(np.diag(g)) <= 1e-16 or np.linalg.det(g) <= 1e-10*np.prod(np.diag(g)):
        return None
    beta = np.linalg.solve(g, h)
    rho = g[0, 1]/math.sqrt(g[0, 0]*g[1, 1])
    return beta, float((1+abs(rho))/(1-abs(rho)))


def score(stats, beta):
    if not stats:
        return dict(rms_a=None, r_squared=None)
    residual = max(0., np.mean([s["yy"]-2*beta@s["h"]+beta@s["g"]@beta for s in stats]))
    variance = np.mean([s["yy"] for s in stats])
    return dict(rms_a=float(math.sqrt(residual)), r_squared=float(1-residual/variance) if variance else None)


def split_indices(segments):
    ids = [p.get("segment", p)["repeat"] for p in segments]
    repeats = sorted(set(ids))
    if len(repeats) >= 3:
        validation = [i for i, r in enumerate(ids) if r == repeats[-2]]
        test = [i for i, r in enumerate(ids) if r == repeats[-1]]
    elif len(repeats) == 2:
        validation = [i for i, r in enumerate(ids) if r == repeats[-1]]
        test = []
    else:
        validation = [i for i in range(len(segments)) if i % 5 == 3]
        test = [i for i in range(len(segments)) if i % 5 == 4]
    return [i for i in range(len(segments)) if i not in validation+test], validation, test


def time_fit(phases, bootstrap, seed):
    dynamic = [p for p in phases if p["segment"]["kind"] in ("step", "sine")]
    train, validation_indices, test = split_indices(dynamic)
    if len(train) < 3 or not validation_indices:
        return None, []
    sweep = []
    for delay_ms in range(-10, 21):
        stats = [sufficient(p, delay_ms/1000) for p in dynamic]
        fit = solve([stats[i] for i in train if stats[i] is not None])
        if fit is None:
            continue
        beta, condition = fit
        validation = score([stats[i] for i in validation_indices if stats[i] is not None], beta)
        if validation["rms_a"] is not None:
            sweep.append(dict(delay_ms=delay_ms, kp_a_per_rad_s=float(beta[0]), ki_a_per_rad=float(beta[1]),
                              condition=condition, validation=validation))
    if not sweep:
        return None, []
    best = min(sweep, key=lambda r: r["validation"]["rms_a"])
    stats = [sufficient(p, best["delay_ms"]/1000) for p in dynamic]
    valid = [s for s in stats if s is not None]
    pre_test = solve([stats[i] for i in train+validation_indices if stats[i] is not None])
    test_score = score([stats[i] for i in test if stats[i] is not None], pre_test[0]) if test else None
    beta, condition = solve(valid)
    rng, draws = np.random.default_rng(seed), []
    for _ in range(bootstrap):
        selected = [valid[i] for i in rng.integers(len(valid), size=len(valid))]
        fitted = solve(selected)
        if fitted is not None:
            draws.append(fitted[0])
    groups = {}
    for key in ("repeat", "kind", "base_rad_s", "amplitude_rad_s", "frequency_hz"):
        for value in sorted({s["phase"]["segment"][key] for s in valid}, key=str):
            group = [s for s in valid if s["phase"]["segment"][key] == value]
            fitted = solve(group)
            if fitted is not None:
                groups[f"{key}={value}"] = dict(trials=len(group), kp_a_per_rad_s=float(fitted[0][0]),
                                                ki_a_per_rad=float(fitted[0][1]), condition=fitted[1])
    return dict(
        kp_a_per_rad_s=float(beta[0]), ki_a_per_rad=float(beta[1]), apparent_delay_ms=best["delay_ms"],
        condition=condition, trials=len(valid), training_trials=len(train), validation_trials=len(validation_indices),
        heldout_trials=len(test), heldout_score_before_all_data_refit=test_score,
        delay_selection_validation_score=best["validation"], all_data_score=score(valid, beta),
        conditional_trial_bootstrap_95ci=(np.percentile(draws, [2.5, 97.5], axis=0).T.tolist() if draws else None),
        timing_sensitivity=sweep, by_condition=groups,
        warning="PI approximation with one nuisance offset per trial. Bootstrap holds fitted delay fixed; "
                "timing/model/calibration bias is additional. Negative delay is diagnostic, not firmware causality.",
    ), valid


def harmonic(time, values, frequency, start, stop):
    keep = (time >= start) & (time < stop)
    t, y = time[keep], values[keep]
    if len(t) < 16:
        return None
    matrix = np.column_stack((np.cos(2*math.pi*frequency*t), np.sin(2*math.pi*frequency*t),
                              np.ones(len(t)), t-t.mean()))
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, y, rcond=None)
    if rank < 4:
        return None
    residual = y-matrix@coefficients
    covariance = np.linalg.pinv(matrix.T@matrix)*(residual@residual/max(1, len(y)-4))
    return complex(coefficients[0], -coefficients[1]), float(math.sqrt(covariance[0, 0]+covariance[1, 1]))


def held_harmonic(time, values, frequency, start, stop):
    omega = 2*math.pi*frequency
    left = np.maximum(time, start)
    right = np.minimum(np.r_[time[1:], stop], stop)
    keep = right > left
    integral = np.sum(values[keep]*(np.exp(-1j*omega*left[keep])-np.exp(-1j*omega*right[keep]))/(1j*omega))
    return complex(2*integral/(stop-start))


def frequency_samples(phases):
    output = []
    for phase in phases:
        s = phase["segment"]
        if s["kind"] != "sine":
            continue
        start, stop, f = s["settle_s"], s["duration_s"], s["frequency_hz"]
        vt, velocity = phase["streams"]["velocity"]
        it, current = phase["streams"]["current"]
        ct, target = phase["streams"]["command"]
        v, i = harmonic(vt, velocity, f, start, stop), harmonic(it, current, f, start, stop)
        if v is None or i is None:
            continue
        command = held_harmonic(ct, target, f, start, stop)
        error = command-v[0]
        signal_ratio = abs(error)/max(v[1], 1e-12)
        usable = abs(error) >= .001 and signal_ratio >= 3
        gain = i[0]/error if abs(error) > 1e-12 else complex(float("nan"), float("nan"))
        output.append(dict(name=s["name"], repeat=s["repeat"], frequency_hz=f, base_rad_s=s["base_rad_s"],
                           amplitude_rad_s=s["amplitude_rad_s"], error_harmonic_rad_s=float(abs(error)),
                           approximate_error_signal_to_coefficient_se=float(signal_ratio), usable=usable,
                           gain_real_a_per_rad_s=float(gain.real) if math.isfinite(gain.real) else None,
                           gain_imag_a_per_rad_s=float(gain.imag) if math.isfinite(gain.imag) else None))
    return output


def fit_complex(samples, delay_ms, lag_ms):
    omega = np.array([2*math.pi*r["frequency_hz"] for r in samples])
    y = np.array([complex(r["gain_real_a_per_rad_s"], r["gain_imag_a_per_rad_s"]) for r in samples])
    response = np.exp(-1j*omega*delay_ms/1000)/(1+1j*omega*lag_ms/1000)
    x = np.column_stack((response, response/(1j*omega)))
    weights = 1/np.maximum(abs(y), .001)
    x, target = x*weights[:, None], y*weights
    matrix, target = np.vstack((x.real, x.imag)), np.r_[target.real, target.imag]
    beta, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
    return beta if rank == 2 else None


def complex_score(samples, beta, delay_ms, lag_ms):
    omega = np.array([2*math.pi*r["frequency_hz"] for r in samples])
    measured = np.array([complex(r["gain_real_a_per_rad_s"], r["gain_imag_a_per_rad_s"]) for r in samples])
    predicted = (beta[0]+beta[1]/(1j*omega))*np.exp(-1j*omega*delay_ms/1000)/(1+1j*omega*lag_ms/1000)
    return float(np.sqrt(np.mean(abs((predicted-measured)/np.maximum(abs(measured), .001))**2)))


def frequency_fit(samples, bootstrap, seed):
    usable = [r for r in samples if r["usable"]]
    if len(usable) < 8 or len({r["frequency_hz"] for r in usable}) < 3:
        return None
    tr, val, test = split_indices(usable)
    train, validation = [usable[i] for i in tr], [usable[i] for i in val]
    if not train or not validation:
        return None
    candidates = []
    for delay in range(-10, 21):
        for lag in (0., 1., 2.5, 5., 10., 20.):
            beta = fit_complex(train, delay, lag)
            if beta is not None:
                candidates.append((complex_score(validation, beta, delay, lag), delay, lag))
    error, delay, lag = min(candidates)
    pre_test = fit_complex(train+validation, delay, lag)
    test_error = complex_score([usable[i] for i in test], pre_test, delay, lag) if test else None
    beta = fit_complex(usable, delay, lag)
    rng, draws = np.random.default_rng(seed), []
    for _ in range(bootstrap):
        sample = [usable[i] for i in rng.integers(len(usable), size=len(usable))]
        fit = fit_complex(sample, delay, lag)
        if fit is not None:
            draws.append(fit)
    return dict(kp_a_per_rad_s=float(beta[0]), ki_a_per_rad=float(beta[1]), apparent_delay_ms=delay,
                current_or_measurement_lag_ms=lag, usable_trials=len(usable),
                validation_relative_rms=error, training_trials=len(train), heldout_trials=len(test),
                validation_trials=len(validation), untouched_test_relative_rms=test_error,
                training_frequencies_hz=sorted({r["frequency_hz"] for r in train}),
                validation_frequencies_hz=sorted({r["frequency_hz"] for r in validation}),
                test_frequencies_hz=sorted({usable[i]["frequency_hz"] for i in test}),
                conditional_trial_bootstrap_95ci=np.percentile(draws, [2.5, 97.5], axis=0).T.tolist(),
                model="(P+I/(j*omega))*exp(-j*omega*delay)/(1+j*omega*lag)",
                warning="Delay/lag and all gains are effective host-I/O estimates; firmware timing not identified. "
                        "CI holds delay/lag fixed. Relative-error weighting treats frequency points comparably.")


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def steady_summary(phases, kt):
    output = []
    for phase in phases:
        s = phase["segment"]
        if s["kind"] != "steady":
            continue
        start, stop = s["settle_s"], s["duration_s"]
        it, current = phase["streams"]["current"]
        vt, velocity = phase["streams"]["velocity"]
        pt, position = phase["streams"]["position"]
        i = current[(it >= start) & (it < stop)]
        v = velocity[(vt >= start) & (vt < stop)]
        t = pt[(pt >= start) & (pt < stop)]
        q = position[(pt >= start) & (pt < stop)]
        if min(len(i), len(v), len(t)) < 2:
            continue
        output.append(dict(name=s["name"], repeat=s["repeat"], target_rad_s=s["base_rad_s"],
                           a2_velocity_mean_rad_s=float(v.mean()), a2_velocity_std_rad_s=float(v.std()),
                           a3_average_velocity_rad_s=float((q[-1]-q[0])/(t[-1]-t[0])),
                           current_mean_a=float(i.mean()), current_std_a=float(i.std()),
                           current_derived_torque_mean_nm=float(i.mean()*kt)))
    return output


def make_figures(folder, phases, samples, td, fd, stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    figures = []
    steady = steady_summary(phases, stats["torque_constant_nm_per_a"])
    if steady:
        fig, axes = plt.subplots(2, 1, figsize=(11, 8))
        axes[0].scatter([s["target_rad_s"] for s in steady], [s["a3_average_velocity_rad_s"] for s in steady],
                        label="Steady fresh A3 average")
        limits = [min(s["target_rad_s"] for s in steady), max(s["target_rad_s"] for s in steady)]
        axes[0].plot(limits, limits, ls="--", color="grey", label="Ideal velocity tracking")
        axes[0].set(xlabel="Steady C1 target (rad/s)", ylabel="Measured velocity (rad/s)")
        axes[1].scatter([s["a3_average_velocity_rad_s"] for s in steady],
                        [s["current_derived_torque_mean_nm"] for s in steady], label="Steady current-derived effort")
        axes[1].set(xlabel="Measured velocity (rad/s)", ylabel="Electromagnetic torque estimate (Nm)")
        for ax in axes:
            ax.grid(alpha=.25)
            ax.legend(loc="lower right")
        fig.suptitle("Steady conditions: drag/tracking context; torque/error is NOT a P estimate with integral action")
        fig.tight_layout()
        figures.append(("steady_conditions", fig))
    usable = [r for r in samples if r["usable"]]
    if usable:
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        f = np.array([r["frequency_hz"] for r in usable])
        gain = np.array([complex(r["gain_real_a_per_rad_s"], r["gain_imag_a_per_rad_s"]) for r in usable])
        axes[0].scatter(f, abs(gain), s=16, alpha=.5, label="Measured coherent I / velocity-error")
        axes[1].scatter(f, np.angle(gain, deg=True), s=16, alpha=.5, label="Measured host-I/O phase")
        if fd:
            grid = np.geomspace(min(f), max(f), 300)
            w = 2*math.pi*grid
            model = ((fd["kp_a_per_rad_s"]+fd["ki_a_per_rad"]/(1j*w))
                     * np.exp(-1j*w*fd["apparent_delay_ms"]/1000)/(1+1j*w*fd["current_or_measurement_lag_ms"]/1000))
            axes[0].plot(grid, abs(model), color="orange", label="Joint PI + delay/lag model")
            axes[1].plot(grid, np.angle(model, deg=True), color="orange", label="Joint PI + delay/lag model")
        axes[0].set(ylabel="Current / speed-error gain (A/(rad/s))", yscale="log")
        axes[1].set(ylabel="Phase (degrees)", xlabel="Excitation frequency (Hz)", xscale="log")
        for ax in axes:
            ax.grid(alpha=.25)
            ax.legend(loc="lower right")
        fig.suptitle("C1 gains unchanged: independent frequency-domain PI identification")
        fig.tight_layout()
        figures.append(("frequency_response", fig))
    if td:
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        sweep = td["timing_sensitivity"]
        axes[0].plot([r["delay_ms"] for r in sweep], [r["kp_a_per_rad_s"] for r in sweep], label="Fitted P")
        axes[1].plot([r["delay_ms"] for r in sweep], [r["validation"]["rms_a"] for r in sweep], label="Held-out error")
        axes[0].set_ylabel("Effective P (A/(rad/s))")
        axes[1].set(ylabel="Demeaned current RMS error (A)", xlabel="Assumed response delay (ms)")
        for ax in axes:
            ax.axvline(td["apparent_delay_ms"], color="red", label="Selected apparent delay")
            ax.grid(alpha=.25)
            ax.legend(loc="lower right")
        fig.suptitle("Sensitivity to asynchronous feedback timing; not motor-internal latency")
        fig.tight_layout()
        figures.append(("timing_sensitivity", fig))
    selected = []
    for kind in ("step", "sine"):
        for p in phases:
            s = p["segment"]
            if s["kind"] == kind and s["repeat"] == 1 and abs(s["base_rad_s"]) < .001:
                selected.append(p)
                if sum(q["segment"]["kind"] == kind for q in selected) == 2:
                    break
    for number, phase in enumerate(selected):
        ct, target = phase["streams"]["command"]
        vt, velocity = phase["streams"]["velocity"]
        it, current = phase["streams"]["current"]
        kt = stats["torque_constant_nm_per_a"]
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        axes[0].step(ct, target, where="post", color="orange", label="Actually sent C1 target")
        axes[0].plot(vt, velocity, lw=.8, label="Fresh A2 velocity")
        axes[1].plot(it, current*kt, lw=.8, label="Fresh A1 current × B0 Kt")
        if td:
            t, x, y = paired(phase, td["apparent_delay_ms"]/1000)
            beta = np.array([td["kp_a_per_rad_s"], td["ki_a_per_rad"]])
            prediction = x@beta
            prediction += np.mean(y-prediction)  # Explicit unknown per-trial initial-integrator offset.
            axes[1].plot(t, prediction*kt, color="orange", label="PI prediction + fitted trial offset")
        axes[0].set_ylabel("Velocity (rad/s)")
        axes[1].set(ylabel="Current-derived torque (Nm)", xlabel="Seconds from trial start; raw timestamps")
        for ax in axes:
            ax.grid(alpha=.25)
            ax.axvline(0, color="red")
            ax.legend(loc="lower right")
            ax.margins(y=.15)
        fig.suptitle(phase["segment"]["name"])
        fig.tight_layout()
        figures.append((f"example_{number+1}", fig))
    if not figures:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        ax.text(.05, .5, "Insufficient completed/excited trials for a gain estimate. Inspect statistics.json.")
        figures.append(("insufficient_data", fig))
    with PdfPages(folder/"all_figures.pdf") as pdf:
        for name, fig in figures:
            for extension in ("png", "svg", "pdf"):
                fig.savefig(folder/f"{name}.{extension}", dpi=160)
            pdf.savefig(fig)
            plt.close(fig)
    return [name for name, _ in figures]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.bootstrap < 20:
        parser.error("use at least 20 trial bootstrap draws")
    folder = args.output_dir or Path(__file__).resolve().parent/"reports"/(args.csv.stem+"_analysis")
    if folder.exists():
        raise FileExistsError("choose a new report directory; previous results are never overwritten")
    meta, phases, excluded, faults = read_capture(args.csv)
    kt = meta.get("torque_constant_nm_per_a")
    if kt is None or not math.isfinite(kt) or kt <= 0:
        raise ValueError("a valid measured B0 torque constant is required for Nm units")
    folder.mkdir(parents=True)
    td, _ = time_fit(phases, args.bootstrap, args.seed)
    harmonics = frequency_samples(phases)
    fd = frequency_fit(harmonics, args.bootstrap, args.seed)
    flags = []
    if not meta["completed"]:
        flags.append("Partial run: only completed segments were used")
    b5 = bytes.fromhex(meta.get("original_settings", {}).get("B5", ""))
    if len(b5) == 5 and int.from_bytes(b5[1:5], "little") != 0:
        flags.append("B5 acceleration setting is nonzero; resolve internal-target ramping before trusting PI gains")
    if any(len(values) > 1 for values in meta["analysis_gain_readback_values"].values()):
        flags.append("Reported B8/B9 values changed during the experiment")
    if td is None or fd is None:
        flags.append("Insufficient excitation for independent time/frequency estimates; initial gain withheld")
    else:
        if min(td["kp_a_per_rad_s"], fd["kp_a_per_rad_s"]) <= 0:
            flags.append("Nonpositive P estimate: simple model not supported")
        elif max(td["kp_a_per_rad_s"], fd["kp_a_per_rad_s"])/min(td["kp_a_per_rad_s"], fd["kp_a_per_rad_s"]) > 1.3:
            flags.append("Time/frequency P estimates differ by >30%")
        if fd["validation_relative_rms"] > .35:
            flags.append("Frequency-model held-out relative RMS >35%")
        if fd["untouched_test_relative_rms"] is None:
            flags.append("No untouched test trials remain for independent validation")
        elif fd["untouched_test_relative_rms"] > .35:
            flags.append("Frequency-model untouched-test relative RMS >35%")
        if min(len(fd[key]) for key in ("training_frequencies_hz", "validation_frequencies_hz",
                                       "test_frequencies_hz")) < 3:
            flags.append("Fewer than three usable frequencies in a train/validation/test partition")
        if td["condition"] > 1000:
            flags.append("P/I regressors are poorly separated (normalized condition >1000)")
        td_ci, fd_ci = td["conditional_trial_bootstrap_95ci"], fd["conditional_trial_bootstrap_95ci"]
        if td_ci is None or fd_ci is None:
            flags.append("Insufficient independent trials for a conditional confidence interval")
        else:
            if min(td_ci[1][1], fd_ci[1][1]) < 0:
                flags.append("Integral confidence interval is entirely negative; PI model unsupported")
            if min(td_ci[0][0], fd_ci[0][0]) <= 0:
                flags.append("Conditional P confidence interval includes zero")
        if fd["current_or_measurement_lag_ms"] == 20. or fd["apparent_delay_ms"] in (-10, 20):
            flags.append("Delay/lag optimum touches the search boundary")
        if td["apparent_delay_ms"] in (-10, 20):
            flags.append("Time-domain delay optimum touches the search boundary")
    fatal = [f for f in flags if not f.startswith("Partial run")]
    initial = fd["kp_a_per_rad_s"]*kt if fd and not fatal else None
    stats = dict(
        experiment="C1 gains strictly read only; joint effective P/I identification", completed=meta["completed"],
        torque_constant_nm_per_a=kt, time_domain=td, frequency_domain=fd, quality_notes=flags,
        provisional_initial_simulation_velocity_kp_nm_per_rad_s=initial,
        simulation_mapping="For a torque-producing velocity P law, use this P as IdealPDActuator damping/MIT Kd. "
                           "A full C1 PI model additionally needs the identified integral term, limits and dynamics.",
        gain_units="current P A/(rad/s), current I A/rad; multiply both by measured Kt for torque gains",
        uncertainty="Bootstrap is conditional on model, selected delay/lag and current-derived torque calibration. "
                    "Asynchronous sampling, internal filters and firmware-integrator state are unobserved.",
        selection_rules="30% method-agreement and 35% normalized held-out-error screens are analysis heuristics, "
                        "never motor aborts or proof of real-world accuracy.",
        completed_segments=len(phases), excluded_segments=excluded, recorded_fault_codes=faults,
        reported_gain_readbacks=meta["analysis_gain_readback_values"],
        source_csv_sha256=file_sha(args.csv), source_json_sha256=file_sha(args.csv.with_suffix(".json")),
        analyzer_sha256=file_sha(__file__), hardware_or_simulation_run=False,
    )
    for fitted in (td, fd):
        if fitted:
            fitted.update(kp_nm_per_rad_s=fitted["kp_a_per_rad_s"]*kt, ki_nm_per_rad=fitted["ki_a_per_rad"]*kt)
            fitted["conditional_trial_bootstrap_95ci_torque_units"] = (
                (np.asarray(fitted["conditional_trial_bootstrap_95ci"])*kt).tolist()
                if fitted["conditional_trial_bootstrap_95ci"] is not None else None)
    (folder/"statistics.json").write_text(json.dumps(stats, indent=2, allow_nan=False)+"\n")
    write_csv(folder/"harmonic_estimates.csv", harmonics)
    write_csv(folder/"steady_conditions.csv", steady_summary(phases, kt))
    if td:
        write_csv(folder/"condition_estimates.csv", [dict(group=k, **v) for k, v in td["by_condition"].items()])
    names = make_figures(folder, phases, harmonics, td, fd, stats)
    report = ["# C1 effective velocity-loop gains", "", "B8/B9 were read only; P and I are fitted together.", "",
              f"Provisional initial velocity Kp: **{initial} Nm/(rad/s)**. `None` means evidence screens failed.", "",
              "This is the velocity-P/MIT-Kd/IdealPD damping coefficient, not passive mechanical damping.", "",
              "## Estimates", "", "| Method | P A/(rad/s) | P Nm/(rad/s) | I Nm/rad |", "|---|---:|---:|---:|"]
    for label, fitted in (("Time domain", td), ("Frequency domain", fd)):
        if fitted:
            report.append(f"| {label} | {fitted['kp_a_per_rad_s']:.6g} | {fitted['kp_nm_per_rad_s']:.6g} | "
                          f"{fitted['ki_nm_per_rad']:.6g} |")
    report += ["", "## Evidence and limits", ""] + [f"- {f}" for f in flags]
    report += ["- Unknown units of the firmware registers are not inferred from their numeric values.",
               "- Steady torque/error is not used as P: integral action makes that quotient misleading.",
               "- Each time-domain trial has a nuisance offset for unknown prior integral state.",
               "- Whole-trial validation/bootstraps preserve within-trial sample correlation; systematic bias remains.",
               "- Frequency fits use exact actually-sent held targets and separate asynchronous harmonic fits.",
               "- Current-derived torque is electromagnetic torque estimate, not independent shaft torque.",
               "- Passive friction/inertia remain separate plant properties; these gains alone do not reproduce C1.",
               "", "[Exact statistics](statistics.json) · [Figure gallery](index.html)"]
    (folder/"README.md").write_text("\n".join(report)+"\n")
    cards = ''.join(f'<h2>{escape(n)}</h2><a href="{n}.png"><img src="{n}.png" alt="{escape(n)}"></a>' for n in names)
    (folder/"index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>C1 identification</title>'
        '<style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:0 15px}img{max-width:100%}</style>'
        '<h1>C1 P/I identification with gains unchanged</h1><p><a href="README.md">Report</a> · '
        '<a href="statistics.json">Statistics</a> · <a href="all_figures.pdf">PDF</a></p>'+cards+'</html>')
    print(json.dumps(dict(report=str(folder), provisional_initial_kp_nm_per_rad_s=initial,
                          quality_notes=flags), indent=2))
    return 0


if __name__ == "__main__":
    main()
