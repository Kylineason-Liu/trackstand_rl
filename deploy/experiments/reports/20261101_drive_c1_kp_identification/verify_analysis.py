"""Independent numerical checks for this saved-data report, with no CAN/simulation."""

import argparse
import json
from pathlib import Path

import numpy as np

from review_analysis import HERE, linear_integral, load


def features(phase, delay_s):
    ct, command = phase["streams"]["command"]
    vt, velocity = phase["streams"]["velocity"]
    pt, _ = phase["streams"]["position"]
    it, current = phase["streams"]["current"]
    q = it-delay_s
    keep = (q >= max(ct[0], vt[0], pt[0], .03))
    keep &= q <= min(vt[-1], pt[-1], phase["segment"]["duration_s"]-.03)
    for t in (vt, pt):
        j = np.clip(np.searchsorted(t, q), 1, len(t)-1)
        keep &= t[j]-t[j-1] <= .1
    q, y = q[keep], current[keep]
    # Independently derive the held-command integral from rectangular intervals.
    j = np.clip(np.searchsorted(ct, q, side="right")-1, 0, len(ct)-1)
    command_area = np.r_[0., np.cumsum(np.diff(ct)*command[:-1])]
    integral = command_area[j]+command[j]*(q-ct[j])-linear_integral(vt, velocity, q)
    error = command[j]-np.interp(q, vt, velocity)
    x = np.column_stack((error, integral))
    return x-x.mean(axis=0), y-y.mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    args = parser.parse_args()
    meta, phases = load(args.cache)
    reviewed = json.loads((HERE / "review_statistics.json").read_text())
    expected = reviewed["alternative_fits"]["integrated_a2_error"]
    dynamic = [p for p in phases if p["segment"]["kind"] in ("sine", "step")]
    pairs = [features(p, expected["apparent_delay_ms"]/1000) for p in dynamic]

    def solve(selected):
        x = np.concatenate([pairs[i][0]/np.sqrt(len(pairs[i][1])) for i in selected])
        y = np.concatenate([pairs[i][1]/np.sqrt(len(pairs[i][1])) for i in selected])
        return np.linalg.lstsq(x, y, rcond=None)[0]

    beta = solve(range(len(pairs)))
    assert np.allclose(beta, [expected["kp_a_per_rad_s"], expected["ki_a_per_rad"]], atol=1e-10, rtol=0)
    pretest = solve([i for i, p in enumerate(dynamic) if p["segment"]["repeat"] < 3])
    test = [pairs[i] for i, p in enumerate(dynamic) if p["segment"]["repeat"] == 3]
    residual = np.mean([np.mean((y-x@pretest)**2) for x, y in test])
    rms, r2 = float(np.sqrt(residual)), float(1-residual/np.mean([np.mean(y*y) for _, y in test]))
    assert abs(rms-expected["heldout_score_before_all_data_refit"]["rms_a"]) < 1e-10
    assert abs(r2-expected["heldout_score_before_all_data_refit"]["r_squared"]) < 1e-10
    max_integral_error = 0.
    for p in dynamic[::19]:
        vt, velocity = p["streams"]["velocity"]
        for t in np.linspace(vt[0], vt[-1], 13):
            points = np.r_[vt[vt < t], t]
            values = np.interp(points, vt, velocity)
            direct = np.sum(np.diff(points)*(values[1:]+values[:-1])*.5)
            max_integral_error = max(max_integral_error, abs(direct-linear_integral(vt, velocity, t)))
    assert max_integral_error < 1e-10
    result = dict(equal_trial_weight_lstsq_p_i_current_units=beta.tolist(),
                  pretest_p_i_torque_units=(pretest*meta["torque_constant_nm_per_a"]).tolist(),
                  untouched_repeat3_rms_a=rms, untouched_repeat3_r_squared=r2,
                  independent_a2_integral_max_error_rad=float(max_integral_error),
                  tested_dynamic_trials=len(dynamic), passed=True)
    (HERE / "numerical_verification.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
