"""Check MIT's actual current response and encoder evidence against its P command."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze import HERE, STEMS, held, read_capture, write_csv, write_json


def paired(phase, delay_s, kt):
    ct, target = phase["streams"]["command"]
    vt, velocity = phase["streams"]["velocity"]
    it, current = phase["streams"]["current"]
    q = it-delay_s
    end = phase["segment"]["duration_s"]
    keep = (q >= max(ct[0], vt[0], .03)) & (q <= min(vt[-1], end-.03)) & (it < end)
    q = q[keep]
    error = held(ct, target, q)-np.interp(q, vt, velocity)
    return it[keep], error, current[keep]*kt


def stats(phase, delay, kt):
    _, error, torque = paired(phase, delay, kt)
    return dict(g=np.array([[np.mean(error**2), error.mean()], [error.mean(), 1.]]),
                h=np.array([np.mean(error*torque), torque.mean()]), yy=np.mean(torque**2))


def solve(rows):
    return np.linalg.solve(sum(r["g"] for r in rows), sum(r["h"] for r in rows))


def score(rows, beta):
    mse = np.mean([r["yy"]-2*beta@r["h"]+beta@r["g"]@beta for r in rows])
    variance = np.mean([r["yy"] for r in rows])-np.mean([r["h"][1] for r in rows])**2
    return dict(rms_nm=float(np.sqrt(max(mse, 0))), r_squared=float(1-mse/variance))


def fit(phases, kt):
    dynamic = [p for p in phases if p["segment"]["kind"] in ("sine", "step")]
    split = {r: [i for i, p in enumerate(dynamic) if p["segment"]["repeat"] == r] for r in (1, 2, 3)}
    sweep = []
    for ms in range(-5, 11):
        sufficient = [stats(p, ms/1000, kt) for p in dynamic]
        beta = solve([sufficient[i] for i in split[1]])
        validation = score([sufficient[i] for i in split[2]], beta)
        sweep.append(dict(delay_ms=ms, validation_rms_nm=validation["rms_nm"]))
    best = min(sweep, key=lambda r: r["validation_rms_nm"])
    rows = [stats(p, best["delay_ms"]/1000, kt) for p in dynamic]
    pretest = solve([rows[i] for i in split[1]+split[2]])
    beta = solve(rows)
    kd = 246*5/4095
    return dict(fitted_velocity_p_nm_per_rad_s=float(beta[0]), fitted_offset_nm=float(beta[1]),
                apparent_alignment_ms=best["delay_ms"],
                pretest_coefficients=pretest.tolist(),
                heldout_repeat3=score([rows[i] for i in split[3]], pretest),
                fixed_encoded_kd_heldout=score([rows[i] for i in split[3]], np.array([kd, 0.])),
                all_data=score(rows, beta), delay_sweep=sweep,
                repeat_fits={str(r): solve([rows[i] for i in split[r]]).tolist() for r in split},
                note="Equal whole-trial weighting; one global offset. Uses observed A2, not forward plant prediction. "
                     "Train repeat1, select alignment repeat2, test repeat3, then all-data refit. "
                     "Alignment includes sensor/host effects, not measured firmware delay.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    args = p.parse_args()
    captures = {label: read_capture(label, args.cache) for label in STEMS}
    meta, phases, _ = captures["MIT"]
    law = fit(phases, meta["torque_constant_nm_per_a"])
    audits = []
    for label, (meta, phases, _) in captures.items():
        for phase in phases:
            s = phase["segment"]
            if s["kind"] != "steady":
                continue
            pt, q = phase["streams"]["position"]
            vt, v = phase["streams"]["velocity"]
            keep = (pt >= s["settle_s"]) & (pt < s["duration_s"]) & (pt >= vt[0]) & (pt <= vt[-1])
            qt, q = pt[keep], q[keep]
            nodes = np.r_[qt[0], vt[(vt > qt[0]) & (vt < qt[-1])], qt[-1]]
            values = np.interp(nodes, vt, v)
            a2_travel = np.sum(np.diff(nodes)*(values[:-1]+values[1:])*.5)
            audits.append(dict(mode=label, name=s["name"], repeat=s["repeat"], target_rad_s=s["base_rad_s"],
                               a3_mean_velocity_rad_s=float((q[-1]-q[0])/(qt[-1]-qt[0])),
                               a2_mean_velocity_rad_s=float(a2_travel/(qt[-1]-qt[0])),
                               a3_range_rad=float(np.ptp(q)),
                               exact_unchanged_window=bool(np.ptp(q) == 0)))
    consistency = {}
    for label in captures:
        rows = [r for r in audits if r["mode"] == label and abs(r["a2_mean_velocity_rad_s"]) >= 1]
        x = np.array([r["a2_mean_velocity_rad_s"] for r in rows])
        y = np.array([r["a3_mean_velocity_rad_s"] for r in rows])
        consistency[label] = dict(trials=len(rows), a3_per_a2_slope=float(x@y/(x@x)))
    with (HERE/"all_trial_metrics.csv").open() as stream:
        steady = [r for r in csv.DictReader(stream) if r["mode"] == "MIT" and r["kind"] == "steady"]
    error = np.array([-float(r["velocity_error_mean_rad_s"]) for r in steady])
    torque = np.array([float(r["torque_mean_nm"]) for r in steady])
    law["steady_mean_p_through_origin_nm_per_rad_s"] = float(error@torque/(error@error))
    law["steady_mean_fixed_kd_rms_nm"] = float(np.sqrt(np.mean((torque-246*5/4095*error)**2)))
    result = dict(mit_current_response=law, steady_kinematic_consistency=consistency)
    write_json(HERE/"current_response_fit.json", result)
    write_csv(HERE/"encoder_steady_checks.csv", audits)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
