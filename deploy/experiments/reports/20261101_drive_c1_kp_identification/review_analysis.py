"""Review the real C1 identification with independent error-integral definitions.

No motor/simulator calls or source-configuration changes. The position consistency
correction is a diagnostic coordinate probe, not a physical calibration claim.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
sys.path.insert(0, str(EXP))
import analyze_drive_c1_kp as original

INPUT = EXP / "logs/20261101_154126_577381_drive_c1_kp_identify.csv"


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def load(cache):
    if cache and (cache/"phase_streams.npz").exists():
        saved = json.loads((cache/"phase_metadata.json").read_text())
        assert saved["source_sha256"] == original.file_sha(INPUT)
        with np.load(cache/"phase_streams.npz", allow_pickle=False) as data:
            phases = saved["phases"]
            for index, phase in enumerate(phases):
                phase["streams"] = {key: (data[f"p{index}_{key}_t"], data[f"p{index}_{key}_y"])
                                    for key in ("command", "velocity", "current", "position")}
        return saved["meta"], phases
    meta, phases, excluded, faults = original.read_capture(INPUT)
    assert not excluded and faults == [0]
    if cache:
        cache.mkdir(exist_ok=True, parents=True)
        arrays = {f"p{index}_{key}_{axis}": stream[j]
                  for index, phase in enumerate(phases) for key, stream in phase["streams"].items()
                  for j, axis in enumerate(("t", "y"))}
        np.savez_compressed(cache/"phase_streams.npz", **arrays)
        saved = dict(source_sha256=original.file_sha(INPUT), meta=meta,
                     phases=[{k: v for k, v in p.items() if k != "streams"} for p in phases])
        (cache/"phase_metadata.json").write_text(json.dumps(saved))
    return meta, phases


def linear_integral(time, velocity, query):
    cumulative = np.r_[0., np.cumsum((velocity[:-1]+velocity[1:])*.5*np.diff(time))]
    index = np.clip(np.searchsorted(time, query, side="right")-1, 0, len(time)-2)
    elapsed = query-time[index]
    slope = (velocity[index+1]-velocity[index])/(time[index+1]-time[index])
    return cumulative[index]+velocity[index]*elapsed+.5*slope*elapsed**2


def consistency(phases):
    rows = []
    for phase in phases:
        segment = phase["segment"]
        if segment["kind"] != "steady":
            continue
        lo, hi = segment["settle_s"], segment["duration_s"]
        pt, position = phase["streams"]["position"]
        vt, velocity = phase["streams"]["velocity"]
        selected = np.flatnonzero((pt >= lo) & (pt < hi) & (pt >= vt[0]) & (pt <= vt[-1]))
        a, b = pt[selected[0]], pt[selected[-1]]
        qtravel = position[selected[-1]]-position[selected[0]]
        vtravel = float(linear_integral(vt, velocity, b)-linear_integral(vt, velocity, a))
        rows.append(dict(name=segment["name"], repeat=segment["repeat"], target_rad_s=segment["base_rad_s"],
                         start_s=float(a), stop_s=float(b), a3_travel_rad=float(qtravel),
                         integrated_a2_travel_rad=vtravel, a3_mean_velocity_rad_s=float(qtravel/(b-a)),
                         a2_time_average_rad_s=float(vtravel/(b-a)),
                         a3_to_a2_ratio=float(qtravel/vtravel) if abs(vtravel)>1e-8 else None))
    high = [r for r in rows if abs(r["a2_time_average_rad_s"]) >= 1.]

    def slope(part):
        x = np.array([r["a2_time_average_rad_s"] for r in part])
        y = np.array([r["a3_mean_velocity_rad_s"] for r in part])
        return float(x@y/(x@x))

    rng = np.random.default_rng(42)
    bootstrap = [slope([high[i] for i in rng.integers(len(high), size=len(high))]) for _ in range(2000)]
    summary = dict(
        all_steady_trials=len(rows), high_speed_trials=len(high), minimum_selected_abs_speed_rad_s=1.,
        scale_a3_speed_per_a2_speed=slope(high),
        conditional_trial_bootstrap_95ci=np.percentile(bootstrap, [2.5, 97.5]).tolist(),
        repeat_scales={str(r): slope([row for row in high if row["repeat"]==r]) for r in (1,2,3)},
        positive_scale=slope([r for r in high if r["target_rad_s"]>0]),
        negative_scale=slope([r for r in high if r["target_rad_s"]<0]),
        warning="A3-derived speed versus A2 report inconsistency; neither is independently externally calibrated. "
                "Scale is diagnostic only. Same actual time intervals, no fitted timing shifts.",
    )
    return rows, summary


def time_probes(phases, scale):
    stock_pair = original.paired

    def speed_integral(phase, delay):
        t, x, y = stock_pair(phase, delay)
        query = t-delay
        ct, command = phase["streams"]["command"]
        vt, velocity = phase["streams"]["velocity"]
        x[:,1] = original.held_integral(ct,command,query)-linear_integral(vt,velocity,query)
        return t,x,y

    def corrected_position(phase, delay):
        t,x,y = stock_pair(phase,delay)
        query = t-delay
        ct,command = phase["streams"]["command"]
        pt,position = phase["streams"]["position"]
        x[:,1] = original.held_integral(ct,command,query)-(np.interp(query,pt,position)-position[0])/scale
        return t,x,y

    results = {}
    try:
        for name, pair, selected in (
            ("integrated_a2_error",speed_integral,phases),
            ("a3_consistency_probe",corrected_position,phases),
            ("zero_baseline_a3_only",stock_pair,[p for p in phases if p["segment"]["base_rad_s"]==0.]),
        ):
            original.paired = pair
            fitted,_ = original.time_fit(selected,500,42)
            results[name]=fitted
    finally:
        original.paired = stock_pair
    return results


def tracking(phases,kt):
    rows=[]
    for phase in phases:
        s=phase["segment"]
        if s["kind"] != "sine":
            continue
        start,stop,f=s["settle_s"],s["duration_s"],s["frequency_hz"]
        ct,target=phase["streams"]["command"]
        vt,velocity=phase["streams"]["velocity"]
        it,current=phase["streams"]["current"]
        wanted=original.held_harmonic(ct,target,f,start,stop)
        measured=original.harmonic(vt,velocity,f,start,stop)[0]
        transfer=measured/wanted
        mask=(vt>=max(start,ct[0]))&(vt<stop)
        index=np.clip(np.searchsorted(ct,vt[mask],side="right")-1,0,len(ct)-1)
        error=velocity[mask]-target[index]
        effort=current[(it>=start)&(it<stop)]*kt
        rows.append(dict(name=s["name"], repeat=s["repeat"], base_rad_s=s["base_rad_s"],
                         amplitude_rad_s=s["amplitude_rad_s"],
                         frequency_hz=f,tracking_fundamental_gain=float(abs(transfer)),
                         phase_equivalent_lag_ms=float(-np.angle(transfer)/(2*math.pi*f)*1000),
                         unshifted_velocity_error_rms_rad_s=float(np.sqrt(np.mean(error**2))),
                         measured_velocity_rms_rad_s=float(np.sqrt(np.mean(velocity[mask]**2))),
                         peak_torque_estimate_nm=float(max(abs(effort)))))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path)
    args=parser.parse_args()
    meta,phases=load(args.cache)
    kt=meta["torque_constant_nm_per_a"]
    consistency_rows,kinematics=consistency(phases)
    probes=time_probes(phases,kinematics["repeat_scales"]["1"])
    baseline=json.loads((HERE/"statistics.json").read_text())
    gains=[]
    for name,fit in [("stock_time_domain",baseline["time_domain"]),
                     ("stock_frequency_domain",baseline["frequency_domain"]),
                     *list(probes.items())]:
        ci=fit["conditional_trial_bootstrap_95ci"]
        gains.append(dict(method=name,kp_nm_per_rad_s=fit["kp_a_per_rad_s"]*kt,ki_nm_per_rad=fit["ki_a_per_rad"]*kt,
                          kp_conditional_low=ci[0][0]*kt,kp_conditional_high=ci[0][1]*kt,
                          ki_conditional_low=ci[1][0]*kt,ki_conditional_high=ci[1][1]*kt,
                          apparent_delay_ms=fit["apparent_delay_ms"]))
    harmonic_samples=original.frequency_samples(phases)
    zero=original.frequency_fit([r for r in harmonic_samples if r["base_rad_s"]==0.],500,42)
    if zero:
        probes["zero_baseline_frequency"]=zero
        gains.append(dict(method="zero_baseline_frequency",kp_nm_per_rad_s=zero["kp_a_per_rad_s"]*kt,
                          ki_nm_per_rad=zero["ki_a_per_rad"]*kt,
                          kp_conditional_low=zero["conditional_trial_bootstrap_95ci"][0][0]*kt,
                          kp_conditional_high=zero["conditional_trial_bootstrap_95ci"][0][1]*kt,
                          ki_conditional_low=zero["conditional_trial_bootstrap_95ci"][1][0]*kt,
                          ki_conditional_high=zero["conditional_trial_bootstrap_95ci"][1][1]*kt,
                          apparent_delay_ms=zero["apparent_delay_ms"]))
    (HERE/"review_statistics.json").write_text(json.dumps(dict(
        source_sha256=original.file_sha(INPUT),kinematic_consistency=kinematics,alternative_fits=probes,
        gain_comparison=gains,diagnostic_scale_fitted_from_repeat1_only=True,
        hardware_or_simulation_run=False),indent=2,allow_nan=False)+"\n")
    write_csv(HERE/"velocity_position_consistency.csv",consistency_rows)
    write_csv(HERE/"gain_method_comparison.csv",gains)
    write_csv(HERE/"sine_tracking_by_trial.csv",tracking(phases,kt))
    print(json.dumps(dict(kinematics=kinematics,gains=gains,
                         scores={k:v.get("heldout_score_before_all_data_refit",v.get("untouched_test_relative_rms"))
                                 for k,v in probes.items()}),indent=2))


if __name__=="__main__":
    main()
