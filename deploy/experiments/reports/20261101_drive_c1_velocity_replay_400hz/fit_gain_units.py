"""Exploratory PI fit, not a determination of firmware units or loop frequency.

Fit user-positive current = a*velocity_error + b*integrated_error + offset.
The integral uses integral(sent C1 velocity) minus fresh A3 displacement.
Fresh A2/A1/A3 are asynchronous; filtering, current-loop dynamics, initial
integrator state and internal scaling are unknown. No motor/configuration writes.
"""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
STEMS = {
    50: "20261031_223709_716988_50hz_drive_velocity_replay",
    200: "20261031_223442_378454_200hz_drive_velocity_replay",
    400: "20261101_134940_221417_400hz_drive_velocity_replay",
}


def fit(rate, stem):
    path = ROOT / "deploy/experiments/logs" / (stem + ".csv")
    metadata = json.loads(path.with_suffix(".json").read_text())
    phase = next(p for p in metadata["phases"] if p["name"] == "core")
    origin = phase["start_monotonic_s"]
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    def fresh(kind, opcode, field):
        selected = [r for r in rows if r["method"] == "core" and r["record_type"] == kind and r["opcode"] == opcode]
        return (np.array([float(r["host_monotonic_s"]) - origin for r in selected]),
                np.array([float(r[field]) for r in selected]))

    ct, target = fresh("command", "C1", "encoded_target_velocity_rad_s")
    vt, velocity = fresh("rx", "A2", "velocity_rad_s")
    pt, position = fresh("rx", "A3", "position_rad")
    position -= phase["initial_position_rad"]
    it, current = fresh("rx", "A1", "current_a")
    current *= -1  # Rear native-positive current is user-negative torque.
    cumulative = np.r_[0., np.cumsum(target[:-1] * np.diff(ct))]
    use = (it >= .02) & (it < 9.98)  # Shared window keeps all timing probes inside captured states.
    current = current[use]
    results = []
    for lag_ms in (-5, -2, 0, 2, 5):
        time = it[use] - lag_ms / 1000
        index = np.clip(np.searchsorted(ct, time, side="right") - 1, 0, len(ct)-1)
        error = target[index] - np.interp(time, vt, velocity)
        integrated = (cumulative[index] + target[index] * (time - ct[index])
                      - np.interp(time, pt, position))
        matrix = np.column_stack([error, integrated, np.ones(len(time))])
        coefficients, _, rank, _ = np.linalg.lstsq(matrix, current, rcond=None)
        assert rank == 3 and np.all(np.isfinite(coefficients))
        predicted = matrix @ coefficients
        a, b, offset = map(float, coefficients)
        results.append(dict(
            assumed_response_lag_ms=lag_ms, samples=len(time),
            effective_kp_a_per_rad_s=a, effective_ki_a_per_rad=b, offset_a=offset,
            current_rms_residual_a=float(np.sqrt(np.mean((current-predicted)**2))),
            r_squared=float(1-np.sum((current-predicted)**2)/np.sum((current-current.mean())**2)),
        ))
    nominal = next(r for r in results if r["assumed_response_lag_ms"] == 0)
    raw = metadata["raw_gains"]
    # If BOTH gains share the same unit scale and Ki is per update, the ratio
    # gives this conditional period. It is not a measurement of firmware timing.
    nominal["conditional_per_update_period_s"] = (
        raw["B9"] / raw["B8"] * nominal["effective_kp_a_per_rad_s"] / nominal["effective_ki_a_per_rad"])
    return dict(packet_hz=rate, source_csv=path.name, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                raw_kp=raw["B8"], raw_ki=raw["B9"], nominal_fit=nominal, timing_sensitivity=results)


def main():
    runs = {str(rate): fit(rate, stem) for rate, stem in STEMS.items()}
    # Algebraic gain conversions alone cannot tell which internal units are used.
    current_per_rad_speed = .5
    milliamp_per_centirpm = .5 * .001 * 6000 / (2*math.pi)
    report = dict(
        method="OLS: user-positive A1 current = a*(held C1 target-interpolated A2 velocity) "
               "+ b*(integral held C1 target-interpolated A3 displacement) + offset",
        fitting_window_s=[.02, 9.98], runs=runs,
        kp_0p5_if_a_per_rad_s=current_per_rad_speed,
        kp_0p5_if_ma_per_centirpm_converted_to_a_per_rad_s=milliamp_per_centirpm,
        conclusion="Firmware gain units not identified. Strong relative integral action is compatible with "
                   "a per-update Ki or another internal integral scaling. P is sensitive to feedback timing.",
        assumptions="Ideal PI approximation, constant gains, reconstructed error from asynchronous external feedback; "
                    "internal measurement/filter/current-loop dynamics and integrator state are not observed.",
        conditional_period_warning="Derived only under an ideal PI/shared unit-scale/per-update hypothesis; "
                                   "not a measured firmware period and not a recommended MIT gain setting.",
        hardware_or_simulation_run=False,
    )
    (HERE / "gain_units_fit.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v["nominal_fit"] for k, v in runs.items()}, indent=2))


if __name__ == "__main__":
    main()
