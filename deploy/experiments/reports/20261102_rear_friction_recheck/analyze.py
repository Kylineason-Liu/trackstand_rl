"""Re-estimate unloaded rear running losses from fresh C1 feedback; no motor access."""

import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
from pathlib import Path
import struct

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
STEM = ROOT / "deploy/experiments/logs/20261101_154126_577381_drive_c1_kp_identify"
PRIOR = HERE.parent / "drive_damping_20261020_154728"
FIELDS = ("velocity", "torque", "position", "temperature")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def capture(cache):
    source = STEM.with_suffix(".csv")
    digest = sha(source)
    meta = json.loads(STEM.with_suffix(".json").read_text())
    assert meta["completed"] and meta["disabled_and_restored"] and meta["error"] is None
    assert meta["encoder_sign"] == meta["torque_sign"] == -1
    phases = [dict(p, streams={f: [] for f in FIELDS}) for p in meta["phases"]
              if p["segment"]["kind"] == "steady"]
    assert len(phases) == 54
    assert all(p["segment"]["settle_s"] == 2. and p["segment"]["duration_s"] == 5.
               and p["actual_duration_s"] >= 5. for p in phases)
    if (cache / "audit.json").exists():
        audit = json.loads((cache / "audit.json").read_text())
        assert audit["csv_sha256"] == digest
        with np.load(cache / "streams.npz", allow_pickle=False) as data:
            for i, p in enumerate(phases):
                p["streams"] = {f: data[f"{i}_{f}"] for f in FIELDS}
        return meta, phases, audit
    cache.mkdir(parents=True, exist_ok=True)
    by_name = {p["segment"]["name"]: p for p in phases}
    checks, faults, codes = Counter(), Counter(), Counter()
    voltage, temperature, settings = [], [], {}
    kt = meta["torque_constant_nm_per_a"]
    rows = 0
    with source.open(newline="") as stream:
        for row in csv.DictReader(stream):
            rows += 1
            if row["record_type"] != "rx":
                continue
            data = bytes.fromhex(row["payload_hex"])
            assert int(row["can_id"], 16) == 1
            op = data[0]
            codes[f"{op:02X}"] += 1
            field, value = None, None
            if op == 0xA1:
                native = struct.unpack("<i", data[1:])[0] * .001
                value, field = -native * kt, "torque"
                assert abs(value - float(row["current_torque_nm"])) < 1e-12
                assert abs(native - float(row["current_a"])) < 1e-12
                checks["A1"] += 1
            elif op == 0xA2:
                value, field = -struct.unpack("<i", data[1:])[0] * .01 * 2 * np.pi / 60, "velocity"
                assert abs(value - float(row["velocity_rad_s"])) < 1e-12
                checks["A2"] += 1
            elif op == 0xA3:
                # Within each short hold these int32 multi-turn counts do not roll over.
                count = struct.unpack("<i", data[3:7])[0]
                assert count == int(row["multi_turn_count"])
                value, field = -count * 2 * np.pi / 16384, "position"
                checks["A3"] += 1
            elif op in (0xAE, 0xCF):
                faults[data[7]] += 1
                voltage.append(int.from_bytes(data[1:3], "little") * .01)
                temperature.append(data[5])
                value, field = float(data[5]), "temperature"
            elif op in (0xB0, 0xB3, 0xB5, 0xB8, 0xB9):
                settings.setdefault(f"{op:02X}", set()).add(data.hex().upper())
                if op == 0xB0:
                    assert abs(struct.unpack("<f", data[2:6])[0] - kt) < 1e-12
            p = by_name.get(row["method"])
            if p is not None and field is not None:
                t = float(row["host_monotonic_s"]) - p["start_monotonic_s"]
                p["streams"][field].append((t, value))
    assert set(faults) == {0}
    arrays = {}
    for i, p in enumerate(phases):
        for field, samples in p["streams"].items():
            a = np.asarray(samples, dtype=float).reshape(-1, 2)
            assert len(a) >= 8 and np.isfinite(a).all() and (np.diff(a[:, 0]) > 0).all()
            p["streams"][field] = a
            arrays[f"{i}_{field}"] = a
    np.savez_compressed(cache / "streams.npz", **arrays)
    audit = dict(csv_sha256=digest, csv_rows=rows, fresh_replies=dict(codes),
                 independently_checked_raw_feedback=dict(checks), faults=dict(faults),
                 torque_constant_nm_per_a=kt, voltage_range_v=[min(voltage), max(voltage)],
                 temperature_range_c=[min(temperature), max(temperature)],
                 settings={k: sorted(v) for k, v in settings.items()})
    save_json(cache / "audit.json", audit)
    return meta, phases, audit


def means(phases, lower=2., upper=5., weighted=False):
    rows = []
    for p in phases:
        s, streams = p["segment"], p["streams"]
        lo, hi = lower, upper
        if weighted:
            hi = min(upper, *(streams[f][-1, 0] for f in ("torque", "velocity", "position")))
            lo = max(lower, *(streams[f][0, 0] for f in ("torque", "velocity", "position")))
        r = dict(name=s["name"], repeat=s["repeat"], target_rad_s=s["base_rad_s"],
                 window_start_s=lo, window_stop_s=hi, time_weighted=weighted)
        for field in ("velocity", "torque", "temperature"):
            t, y = streams[field].T
            mask = (t >= lo) & (t < hi)
            a, b = t[mask], y[mask]
            assert len(a) >= 5
            mean = float(np.mean(b))
            if weighted:
                knots = np.r_[lo, t[(t > lo) & (t < hi)], hi]
                values = np.interp(knots, t, y)
                mean = float(np.trapezoid(values, knots) / (hi - lo))
            unit = {"velocity": "rad_s", "torque": "nm", "temperature": "c"}[field]
            r[f"{field}_mean_{unit}"] = mean
            r[f"{field}_sd_{unit}"] = float(np.std(b))
            r[f"{field}_samples"] = len(b)
            if field == "velocity":
                r["a2_wrong_direction_fraction"] = float(np.mean(b * np.sign(s["base_rad_s"]) < 0))
                r["velocity_linear_trend_rad_s2"] = float(np.polyfit(a - a.mean(), b, 1)[0])
                r["velocity_endpoint_mean_difference_rad_s"] = float(
                    np.mean(y[(t >= hi - .25) & (t < hi)]) - np.mean(y[(t >= lo) & (t < lo + .25)]))
        pt, pos = streams["position"].T
        pl, ph = max(lo, pt[0]), min(hi, pt[-1])
        r["a3_average_speed_rad_s"] = float((np.interp(ph, pt, pos) - np.interp(pl, pt, pos)) / (ph - pl))
        r["a3_travel_rad"] = float(np.interp(ph, pt, pos) - np.interp(pl, pt, pos))
        rows.append(r)
    return rows


def xy(rows):
    return np.array([r["velocity_mean_rad_s"] for r in rows]), np.array([r["torque_mean_nm"] for r in rows])


def fit(rows, nonnegative=False, decay_speed=None):
    v, t = xy(rows)
    x = np.column_stack((v, np.sign(v), np.ones(len(v))))
    if decay_speed is not None:
        x = np.column_stack((v, np.sign(v), np.sign(v) * np.exp(-abs(v) / decay_speed), np.ones(len(v))))
    count = x.shape[1] - 1
    options = itertools.product((False, True), repeat=count) if nonnegative else [(True,) * count]
    best = None
    for selected in options:
        indices = [i for i, on in enumerate(selected) if on] + [count]
        beta = np.zeros(count + 1)
        beta[indices] = np.linalg.lstsq(x[:, indices], t, rcond=None)[0]
        if nonnegative and np.min(beta[:count]) < -1e-12:
            continue
        residual = t - x @ beta
        loss = float(np.mean(residual**2))
        if best is None or loss < best[0]:
            best = loss, beta
    loss, beta = best
    keys = ["B_nm_s_per_rad", "Tc_nm"] + (["low_speed_excess_nm"] if decay_speed else []) + ["bias_nm"]
    result = dict(zip(keys, beta.tolist()))
    result.update(rmse_nm=float(np.sqrt(loss)), n_holds=len(rows))
    if decay_speed is not None:
        result["decay_speed_rad_s"] = float(decay_speed)
    return result


def speed_dependent_fit(rows):
    return min((fit(rows, True, v) for v in np.geomspace(.015, 3., 160)), key=lambda x: x["rmse_nm"])


def predict(model, velocity):
    v = np.asarray(velocity)
    loss = model["Tc_nm"] + model.get("low_speed_excess_nm", 0.) * np.exp(
        -abs(v) / model.get("decay_speed_rad_s", 1.))
    return model["B_nm_s_per_rad"] * v + np.sign(v) * loss + model["bias_nm"]


def bootstraps(rows, n=5000):
    rng = np.random.default_rng(42)
    # Whole holds within each signed target; never treat 100-Hz samples as independent trials.
    strata = [[r for r in rows if r["target_rad_s"] == target]
              for target in sorted({r["target_rad_s"] for r in rows})]
    draws = []
    for _ in range(n):
        sample = [group[i] for group in strata for i in rng.integers(len(group), size=len(group))]
        draws.append(fit(sample))
    ci = {key: np.percentile([r[key] for r in draws], [2.5, 50, 97.5]).tolist()
          for key in ("B_nm_s_per_rad", "Tc_nm", "bias_nm")}
    return ci, draws


def plots(rows, pooled, fits, phases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .22})

    def save(fig, stem):
        for ext in ("png", "svg", "pdf"):
            fig.savefig(HERE / f"{stem}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    v, tau = xy(rows)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    colors = {1: "tab:blue", -1: "tab:orange"}
    for sign in (-1, 1):
        mask = np.sign(v) == sign
        axs[0].scatter(v[mask], tau[mask], s=24, color=colors[sign],
                       label=f"{'Forward' if sign > 0 else 'Backward'} holds")
        selected = [r for r in pooled if r["direction"] == sign]
        axs[1].errorbar([r["speed_mean_rad_s"] for r in selected], [r["torque_magnitude_mean_nm"] for r in selected],
                        yerr=[r["torque_between_hold_sd_nm"] for r in selected], fmt="o-", capsize=3,
                        color=colors[sign], label=f"{'Forward' if sign > 0 else 'Backward'} mean +/- across-repeat SD")
        grid = sign * np.linspace(.01, 10.3, 300)
        axs[0].plot(grid, predict(fits["configured"], grid), color="black", linestyle="--",
                    label="Configured +/-0.106 Nm" if sign == 1 else None)
        axs[0].plot(grid, predict(fits["moving_physical"], grid), color="tab:green",
                    label="New moving-range fit" if sign == 1 else None)
    grid = np.geomspace(.04, 11, 300)
    axs[1].axhline(.106, color="black", linestyle="--", label="Configured moving friction 0.106 Nm")
    empirical = fits["speed_dependent_descriptive"]
    axs[1].plot(grid, empirical["Tc_nm"] + empirical["B_nm_s_per_rad"] * grid
                + empirical["low_speed_excess_nm"] * np.exp(-grid / empirical["decay_speed_rad_s"]),
                color="tab:purple", linestyle=":", label="Descriptive speed-dependent fit (not breakaway)")
    axs[0].set(xlabel="Mean measured A2 speed (rad/s)", ylabel="Mean A1 x B0 torque (Nm)",
               title="54 steady holds: signed current-derived running torque", ylim=(-.16, .16))
    axs[1].set(xlabel="Absolute mean A2 speed (rad/s, log scale)", ylabel="Direction-aligned mean torque (Nm)",
               title="Higher effective running torque near zero", xscale="log", ylim=(.075, .165))
    for ax in axs:
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Rear C1 friction recheck | airborne/free | 3 repeats | final 3 s of each 5-s hold")
    save(fig, "rear_friction_curve")

    fig, axs = plt.subplots(3, 3, figsize=(15, 9), sharex="col", constrained_layout=True)
    plotted = []
    for col, level in enumerate((.05, .4, 3.)):
        p = next(p for p in phases if p["segment"]["base_rad_s"] == level and p["segment"]["repeat"] == 1)
        axs[0, col].set_title(f"+{level:g} rad/s, repeat 1")
        for row_index, field in enumerate(("velocity", "torque", "position")):
            t, values = p["streams"][field].T
            y = values - values[0] if field == "position" else values
            axs[row_index, col].plot(t, y, color="tab:blue", linewidth=.9, label=f"Fresh {field}")
            axs[row_index, col].axvspan(2., 5., color="gold", alpha=.13, label="Fit window")
            if field == "velocity":
                axs[row_index, col].axhline(level, color="tab:red", linestyle="--", label="C1 target")
            if field == "torque":
                axs[row_index, col].axhline(.106, color="black", linestyle="--", label="Configured 0.106 Nm")
            axs[row_index, col].legend(loc="lower right", fontsize=8)
            axs[row_index, col].set_ylabel({"velocity": "A2 speed (rad/s)", "torque": "A1 x B0 (Nm)",
                                           "position": "A3 travel (rad)"}[field])
            plotted.extend(dict(phase=p["segment"]["name"], field=field, time_s=float(a), value=float(b))
                           for a, b in zip(t, y))
        axs[2, col].set_xlabel("Seconds after hold starts")
    fig.suptitle("Raw steady-hold feedback: speed ripple and finite-angle averaging remain in the estimate")
    save(fig, "steady_feedback_examples")
    save_csv(HERE / "plotted_native_samples.csv", plotted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/trackstand_c1_friction_recheck_cache"))
    args = parser.parse_args()
    meta, phases, audit = capture(args.cache)
    rows = means(phases)
    save_csv(HERE / "steady_holds.csv", rows)
    # Independently reproduce the prior C1 review's per-hold values before fitting.
    previous = list(csv.DictReader((HERE.parent / "20261101_drive_c1_kp_identification/steady_conditions.csv").open()))
    lookup = {r["name"]: r for r in previous}
    maximum_previous_error = max(abs(r["torque_mean_nm"] - float(lookup[r["name"]]["current_derived_torque_mean_nm"]))
                                 for r in rows)
    assert maximum_previous_error < 1e-12
    assert max(abs(r["velocity_mean_rad_s"] - float(lookup[r["name"]]["a2_velocity_mean_rad_s"]))
               for r in rows) < 1e-12
    moving = [r for r in rows if abs(r["target_rad_s"]) >= 1.]
    configured = dict(B_nm_s_per_rad=0., Tc_nm=.106, bias_nm=0.)
    fits = dict(configured=configured, all_range_ols=fit(rows), all_range_physical=fit(rows, True),
                moving_ols=fit(moving), moving_physical=fit(moving, True),
                speed_dependent_descriptive=speed_dependent_fit(rows))
    ci, draws = bootstraps(moving)
    save_csv(HERE / "moving_bootstrap_draws.csv", draws)
    sensitivity = []
    for cutoff in (0., .1, .2, .4, .8, 1., 3.):
        for lower, upper, weighted in ((2., 5., False), (2., 5., True), (2.5, 5., False),
                                        (3., 5., False), (3.5, 5., False), (4., 5., False)):
            selected = [r for r in means(phases, lower, upper, weighted) if abs(r["target_rad_s"]) >= cutoff]
            for physical in (False, True):
                sensitivity.append(dict(minimum_target_magnitude_rad_s=cutoff, window_lower_s=lower,
                                        window_upper_s=upper, time_weighted=weighted, physical=physical,
                                        **fit(selected, physical)))
    save_csv(HERE / "fit_sensitivity.csv", sensitivity)
    repeat_fits = {str(i): fit([r for r in moving if r["repeat"] == i]) for i in (1, 2, 3)}
    zero_viscous = dict(B_nm_s_per_rad=0., Tc_nm=float(np.mean(
        [abs(r["torque_mean_nm"]) for r in moving])), bias_nm=float(np.mean(
            [r["torque_mean_nm"] for r in moving])))
    joint_intervals = means(phases, weighted=True)
    a3_coordinate = [dict(r, velocity_mean_rad_s=r["a3_average_speed_rad_s"])
                     for r in joint_intervals if abs(r["target_rad_s"]) >= 1.]
    comparable_old_range = [r for r in moving if abs(r["target_rad_s"]) <= 8.]
    # A fixed B=0 fit is useful for the actual model baseline, not an identified exact zero.
    zero_v, zero_t = xy(moving)
    zero_viscous["rmse_nm"] = float(np.sqrt(np.mean((zero_t - predict(zero_viscous, zero_v))**2)))
    direction_fits = []
    for sign in (-1, 1):
        group = [r for r in moving if np.sign(r["target_rad_s"]) == sign]
        v, t = xy(group)
        b, intercept = np.polyfit(abs(v), sign * t, 1)
        direction_fits.append(dict(direction=sign, B_nm_s_per_rad=float(b), intercept_nm=float(intercept)))
    save_csv(HERE / "moving_direction_fits.csv", direction_fits)
    pooled = []
    for target in sorted({r["target_rad_s"] for r in rows}):
        group = [r for r in rows if r["target_rad_s"] == target]
        sign = int(np.sign(target))
        torque = [sign * r["torque_mean_nm"] for r in group]
        pooled.append(dict(target_rad_s=target, direction=sign, repeats=len(group),
                           speed_mean_rad_s=float(np.mean([abs(r["velocity_mean_rad_s"]) for r in group])),
                           torque_magnitude_mean_nm=float(np.mean(torque)),
                           torque_between_hold_sd_nm=float(np.std(torque, ddof=1))))
    save_csv(HERE / "torque_by_speed_and_direction.csv", pooled)
    validation = []
    for repeat in (1, 2, 3):
        train, test = [r for r in rows if r["repeat"] != repeat], [r for r in rows if r["repeat"] == repeat]
        models = dict(configured=configured, moving_fit=fit([r for r in train if abs(r["target_rad_s"]) >= 1.], True),
                      all_range_constant=fit(train, True), descriptive=speed_dependent_fit(train))
        v, t = xy(test)
        for name, model in models.items():
            for label, mask in (("all", np.ones(len(v), bool)), ("moving", abs(v) >= 1.), ("low", abs(v) < .5)):
                error = t[mask] - predict(model, v[mask])
                validation.append(dict(held_out_repeat=repeat, model=name, evaluated_range=label,
                                       holds=int(mask.sum()), rmse_nm=float(np.sqrt(np.mean(error**2))),
                                       mean_error_nm=float(error.mean())))
    save_csv(HERE / "held_out_repeat_errors.csv", validation)
    prior = json.loads((PRIOR / "fit_summary.json").read_text())
    v, t = xy(moving)
    all_v, all_t = xy(rows)
    summary = dict(
        source_stem=str(STEM.relative_to(ROOT)), raw_audit=audit, moving_target_range_rad_s=[1.5, 10.],
        moving_holds=len(moving), all_holds=len(rows), retained_fresh_a1=sum(r["torque_samples"] for r in rows),
        retained_fresh_a2=sum(r["velocity_samples"] for r in rows),
        current_configuration=configured, models=fits, moving_conditional_bootstrap_95ci=ci,
        moving_zero_viscous_fit=zero_viscous, moving_a3_speed_coordinate_fit=fit(a3_coordinate),
        overlapping_1_to_8_rad_s_range_fit=fit(comparable_old_range, True),
        bootstrap=dict(replicates=5000, seed=42, resampling="whole holds within each signed target"),
        moving_repeat_fits=repeat_fits, moving_direction_fits=direction_fits,
        prior_20261020=prior["physical_nonnegative_fit"],
        moving_vs_configured_coulomb_percent=100 * (fits["moving_physical"]["Tc_nm"] / .106 - 1),
        configured_rmse_moving_nm=float(np.sqrt(np.mean((t - predict(configured, v))**2))),
        configured_rmse_all_nm=float(np.sqrt(np.mean((all_t - predict(configured, all_v))**2))),
        mean_torque_magnitude_moving_nm=float(np.mean(np.sign(v) * t)),
        peak_moving_speed_linear_trend_rad_s2=max(abs(r["velocity_linear_trend_rad_s2"]) for r in moving),
        peak_moving_endpoint_speed_difference_rad_s=max(abs(r["velocity_endpoint_mean_difference_rad_s"])
                                                       for r in moving),
        old_review_maximum_reproduced_torque_error_nm=maximum_previous_error,
        notes=["Current-derived electromagnetic estimate, not independent shaft torque.",
               "Airborne/free assembly; static breakaway and loaded tyre resistance not identified.",
               "Near-zero ripple, controller/measurement effects and finite rotor phase coverage "
               "confound a single law.",
               "Positive viscous slope must be distinguished from statistical/systematic uncertainty.",
               "No simulation, deployment, firmware or progress-record configuration changed."])
    save_json(HERE / "statistics.json", summary)
    plots(rows, pooled, fits, phases)
    print(json.dumps({k: summary[k] for k in ("models", "moving_conditional_bootstrap_95ci",
                                            "moving_vs_configured_coulomb_percent")}, indent=2))


if __name__ == "__main__":
    main()
