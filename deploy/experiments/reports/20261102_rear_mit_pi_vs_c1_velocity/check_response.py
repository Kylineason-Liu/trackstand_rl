"""Check the known P-plus-host-I torque law against fresh A1 current telemetry."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze import HERE, captures, held, write_csv, write_json


def paired(phase, delay_s, kt):
    ct, target = phase["streams"]["command"]
    vt, velocity = phase["streams"]["velocity"]
    it, current = phase["streams"]["current"]
    q = it-delay_s
    end = phase["segment"]["duration_s"]
    keep = (q >= max(ct[0],vt[0],.03)) & (q <= min(vt[-1],end-.03)) & (it < end)
    q = q[keep]
    error = held(ct,target,q)-np.interp(q,vt,velocity)
    i_tau = held(*phase["streams"]["integral"],q)
    return it[keep], error, i_tau, current[keep]*kt


def stats(phase, delay, kt):
    _, error, i_tau, torque = paired(phase,delay,kt)
    x = np.column_stack((error,i_tau,np.ones(len(error))))
    return dict(g=x.T@x/len(x),h=x.T@torque/len(x),yy=np.mean(torque**2))


def solve(rows):
    return np.linalg.solve(sum(r["g"] for r in rows),sum(r["h"] for r in rows))


def score(rows,beta):
    mse = np.mean([r["yy"]-2*beta@r["h"]+beta@r["g"]@beta for r in rows])
    variance = np.mean([r["yy"] for r in rows])-np.mean([r["h"][2] for r in rows])**2
    return dict(rms_nm=float(np.sqrt(max(0.,mse))),r_squared=float(1-mse/variance))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,required=True)
    args = p.parse_args()
    data = captures(args.cache)
    meta, phases, _ = data["PI"]
    kt = meta["torque_constant_nm_per_a"]
    kd = meta["mit_configuration"]["velocity_kd_encoded_nm_per_rad_s"]
    dynamic = [p for p in phases if p["segment"]["kind"] in ("step","sine")]
    split = {r:[i for i,p in enumerate(dynamic) if p["segment"]["repeat"]==r] for r in (1,2,3)}
    sweep = []
    for ms in range(-5,11):
        rows = [stats(p,ms/1000,kt) for p in dynamic]
        beta = solve([rows[i] for i in split[1]])
        val = score([rows[i] for i in split[2]],beta)
        sweep.append(dict(delay_ms=ms,validation_rms_nm=val["rms_nm"]))
    best = min(sweep,key=lambda x:x["validation_rms_nm"])["delay_ms"]
    rows = [stats(p,best/1000,kt) for p in dynamic]
    pretest = solve([rows[i] for i in split[1]+split[2]])
    beta = solve(rows)
    law = dict(apparent_alignment_ms=best,fitted_p_nm_per_rad_s=float(beta[0]),
               fitted_integral_torque_factor=float(beta[1]),fitted_offset_nm=float(beta[2]),
               pretest_coefficients=pretest.tolist(),
               heldout_repeat3_fitted=score([rows[i] for i in split[3]],pretest),
               heldout_repeat3_known_pi=score([rows[i] for i in split[3]],np.array([kd,1.,0.])),
               heldout_repeat3_without_i=score([rows[i] for i in split[3]],np.array([kd,0.,0.])),
               delay_sweep=sweep,
               note="Current reconstruction uses measured A2 and logged host integral torque, not forward motion. "
                    "Train repeat1, select alignment repeat2, test repeat3; one global offset; equal trial weights. "
                    "Alignment is descriptive host/sensor timing, not firmware delay. I factor should approach 1, "
                    "not Ki=3.3: the logged I torque already includes Ki.")
    audits, consistency = [], {}
    for mode,(meta,phases,_) in data.items():
        for phase in phases:
            s = phase["segment"]
            if s["kind"]!="steady":
                continue
            pt,q = phase["streams"]["position"]
            vt,v = phase["streams"]["velocity"]
            keep = (pt>=s["settle_s"]) & (pt<s["duration_s"]) & (pt>=vt[0]) & (pt<=vt[-1])
            t,qq = pt[keep],q[keep]
            nodes = np.r_[t[0],vt[(vt>t[0]) & (vt<t[-1])],t[-1]]
            values = np.interp(nodes,vt,v)
            travel = np.sum(np.diff(nodes)*(values[:-1]+values[1:])*.5)
            audits.append(dict(mode=mode,name=s["name"],target_rad_s=s["base_rad_s"],
                               a3_mean_velocity_rad_s=float((qq[-1]-qq[0])/(t[-1]-t[0])),
                               a2_mean_velocity_rad_s=float(travel/(t[-1]-t[0])),a3_range_rad=float(np.ptp(qq))))
        rr = [a for a in audits if a["mode"]==mode and abs(a["a2_mean_velocity_rad_s"])>=1]
        x = np.array([a["a2_mean_velocity_rad_s"] for a in rr])
        y = np.array([a["a3_mean_velocity_rad_s"] for a in rr])
        consistency[mode] = dict(trials=len(x),a3_per_a2_slope=float(x@y/(x@x)))
    write_json(HERE/"current_response_fit.json",dict(pi_current_response=law,encoder_consistency=consistency))
    write_csv(HERE/"encoder_steady_checks.csv",audits)
    print(json.dumps(dict(law=law,consistency=consistency),indent=2))


if __name__ == "__main__":
    main()
