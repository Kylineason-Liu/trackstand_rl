"""Independent raw selected-trace and numerical verification; no hardware calls."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np

from analyze import EXP, HERE, STEMS, read_capture, write_json

SELECTIONS = [("steady", .1, 0., 0.), ("steady", .4, 0., 0.),
              ("sine", 0., .25, .5), ("sine", 0., 1., .5)]


def raw_selected(mode, meta):
    wanted = {}
    for col, (kind, base, amp, frequency) in enumerate(SELECTIONS):
        p = next(p for p in meta["phases"] if p["segment"]["repeat"] == 3
                 and p["segment"]["kind"] == kind and p["segment"]["base_rad_s"] == base
                 and p["segment"]["amplitude_rad_s"] == amp and p["segment"]["frequency_hz"] == frequency)
        wanted[p["segment"]["name"]] = (col, p["start_monotonic_s"],
                                        0. if kind == "steady" else 1., 5. if kind == "steady" else 9.)
    streams = defaultdict(list)
    last, total, position_endpoints, f1_positions = meta["initial_multi_turn_count"], 0, 0, 0
    low_names = {p["segment"]["name"] for p in meta["phases"] if p["segment"]["kind"] == "steady"
                 and abs(p["segment"]["base_rad_s"]) <= .4}
    low_endpoints, low_samples = 0, 0
    low_by_trial = defaultdict(lambda: dict(samples=0, endpoint_samples=0))
    with (EXP/"logs"/(STEMS[mode]+".csv")).open(newline="") as f:
        for r in csv.DictReader(f):
            if r["record_type"] != "rx":
                continue
            data = bytes.fromhex(r["payload_hex"])
            op = data[0]
            position = None
            if op == 0xA3:
                count = int.from_bytes(data[3:7], "little", signed=True)
                delta = count-last
                if delta >= 2**31:
                    delta -= 2**32
                elif delta < -2**31:
                    delta += 2**32
                total += delta
                last = count
                position = -total*(2*math.pi/16384)
            elif op == 0xF1:
                pcode = int.from_bytes(data[1:3], "big")
                expected = (pcode/65535*2-1)*95.5
                assert abs(expected-float(r["mit_position_native_rad"])) < 1e-10
                position_endpoints += int(pcode in (0, 65535))
                f1_positions += 1
                if r["method"] in low_names:
                    low_endpoints += int(pcode in (0, 65535))
                    low_samples += 1
                    low_by_trial[r["method"]]["samples"] += 1
                    low_by_trial[r["method"]]["endpoint_samples"] += int(pcode in (0, 65535))
            if r["method"] not in wanted:
                continue
            col, origin, start, stop = wanted[r["method"]]
            t = float(r["host_monotonic_s"])-origin
            if not start <= t < stop:
                continue
            if op == 0xA2:
                value = -int.from_bytes(data[1:], "little", signed=True)/100*2*math.pi/60
                row_index, field = 0, "a2_velocity_rad_s"
            elif op == 0xA1:
                value = -int.from_bytes(data[1:], "little", signed=True)/1000*meta["torque_constant_nm_per_a"]
                row_index, field = 1, "a1_torque_nm"
            elif op == 0xA3:
                value = position
                row_index, field = 2, "a3_relative_position_rad"
            else:
                continue
            streams[(f"r{row_index}_c{col}", mode, field)].append((t, value))
    for key in streams:
        a = np.array(streams[key])
        if key[2] == "a3_relative_position_rad":
            a[:, 1] -= a[0, 1]
        streams[key] = a
    return streams, dict(f1_positions_decoded=f1_positions, f1_position_endpoint_codes=position_endpoints,
                         low_steady_f1_position_endpoints=low_endpoints, low_steady_f1_samples=low_samples,
                         low_steady_f1_position_by_trial=dict(low_by_trial))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    args = p.parse_args()
    captures = {m: read_capture(m, args.cache) for m in STEMS}
    exported = defaultdict(list)
    with (HERE/"plotted_response_samples.csv").open() as f:
        for r in csv.DictReader(f):
            exported[r["panel"], r["mode"], r["field"]].append((float(r["time_s"]), float(r["value"])))
    verified, maximum_error, extra = 0, 0., {}
    for mode, (meta, _, _) in captures.items():
        raw, extra[mode] = raw_selected(mode, meta)
        for key, samples in raw.items():
            saved = np.array(exported.pop(key))
            assert samples.shape == saved.shape
            maximum_error = max(maximum_error, float(np.max(abs(samples-saved))))
            assert np.allclose(samples, saved, rtol=0, atol=1e-10)
            verified += len(samples)
    assert not exported
    # Cross-check all baseline metrics against the independently saved C1 report.
    old = EXP/"reports/20261101_drive_c1_kp_identification"
    with (HERE/"all_trial_metrics.csv").open() as f:
        new = {r["name"]: r for r in csv.DictReader(f) if r["mode"] == "C1"}
    checks = 0
    for filename, mapping in (
        ("steady_conditions.csv", {"a2_velocity_mean_rad_s": "velocity_mean_rad_s",
                                   "a2_velocity_std_rad_s": "velocity_std_rad_s",
                                   "current_derived_torque_mean_nm": "torque_mean_nm"}),
        ("sine_tracking_by_trial.csv", {"tracking_fundamental_gain": "fundamental_gain",
                                       "unshifted_velocity_error_rms_rad_s": "velocity_error_rms_rad_s"}),
    ):
        with (old/filename).open() as f:
            for r in csv.DictReader(f):
                for a, b in mapping.items():
                    assert abs(float(r[a])-float(new[r["name"]][b])) < 1e-10
                    checks += 1
    # Independently solve the equal-trial-weight MIT dynamic regression.
    meta, phases, _ = captures["MIT"]
    xs, ys = [], []
    for phase in phases:
        if phase["segment"]["kind"] not in ("sine", "step"):
            continue
        ct, command = phase["streams"]["command"]
        vt, velocity = phase["streams"]["velocity"]
        it, current = phase["streams"]["current"]
        q = it-.001
        mask = (q >= max(ct[0], vt[0], .03)) & (q <= min(vt[-1], phase["segment"]["duration_s"]-.03))
        mask &= it < phase["segment"]["duration_s"]
        j = np.searchsorted(ct, q[mask], side="right")-1
        e = command[j]-np.interp(q[mask], vt, velocity)
        x = np.column_stack((e, np.ones(len(e))))
        y = current[mask]*meta["torque_constant_nm_per_a"]
        xs.append(x/np.sqrt(len(y))); ys.append(y/np.sqrt(len(y)))
    beta = np.linalg.lstsq(np.concatenate(xs), np.concatenate(ys), rcond=None)[0]
    expected = json.loads((HERE/"current_response_fit.json").read_text())["mit_current_response"]
    assert np.allclose(beta, [expected["fitted_velocity_p_nm_per_rad_s"], expected["fitted_offset_nm"]],
                       rtol=0, atol=1e-10)
    write_json(HERE/"verification.json", dict(raw_plotted_response_samples_verified=verified,
                                             maximum_raw_export_error=maximum_error,
                                             baseline_metrics_reproduced=checks, extra_raw_checks=extra,
                                             independent_dynamic_lstsq_coefficients=beta.tolist(),
                                             passed=True, hardware_or_simulator_run=False))
    print((HERE/"verification.json").read_text())


if __name__ == "__main__":
    main()
