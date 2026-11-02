"""Independent raw-byte checks for plotted traces and comparison statistics."""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np

from analyze import EXP, HERE, OLD, STEM, base, captures, write_json


def raw_selected(stem,meta):
    selections = [("steady",.1,0.,0.),("steady",.4,0.,0.),("sine",0.,.25,.5),("sine",0.,1.,.5)]
    wanted = {}
    for col,(kind,b,a,f) in enumerate(selections):
        p = next(p for p in meta["phases"] if p["segment"]["kind"]==kind and p["segment"]["repeat"]==3
                 and p["segment"]["base_rad_s"]==b and p["segment"]["amplitude_rad_s"]==a
                 and p["segment"]["frequency_hz"]==f)
        wanted[p["segment"]["name"]] = (col,p["start_monotonic_s"],
                                        0. if kind=="steady" else 1.,5. if kind=="steady" else 9.)
    streams = defaultdict(list)
    history_phases = {p["segment"]["name"]:p for p in meta["phases"] if p["segment"]["kind"]=="step"
                      and p["segment"]["base_rad_s"]==0. and p["segment"]["amplitude_rad_s"]==.1}
    history = defaultdict(lambda:defaultdict(list))
    last,total = meta["initial_multi_turn_count"],0
    with (EXP/"logs"/(stem+".csv")).open(newline="") as stream:
        for r in csv.DictReader(stream):
            name = r["method"]
            h = history_phases.get(name)
            ht = float(r["host_monotonic_s"])-h["start_monotonic_s"] if h else None
            if h and r["record_type"]=="command" and r["integral_torque_nm"]:
                history[name]["integral"].append((ht,float(r["integral_torque_nm"])))
            if r["record_type"]!="rx":
                continue
            data = bytes.fromhex(r["payload_hex"])
            op,position = data[0],None
            if op==0xA3:
                count = int.from_bytes(data[3:7],"little",signed=True)
                delta = count-last
                if delta>=2**31:
                    delta -= 2**32
                elif delta< -2**31:
                    delta += 2**32
                total += delta
                last = count
                position = -total*(2*math.pi/16384)
            if h and op in (0xA2,0xA3):
                if op==0xA2:
                    value = -int.from_bytes(data[1:],"little",signed=True)/100*2*math.pi/60
                    history[name]["velocity"].append((ht,value))
                else:
                    history[name]["position"].append((ht,position))
            if r["method"] not in wanted:
                continue
            col,origin,start,stop = wanted[r["method"]]
            t = float(r["host_monotonic_s"])-origin
            if not start<=t<stop:
                continue
            if op==0xA2:
                y = -int.from_bytes(data[1:],"little",signed=True)/100*2*math.pi/60
                row,field = 0,"a2_velocity_rad_s"
            elif op==0xA1:
                y = -int.from_bytes(data[1:],"little",signed=True)/1000*meta["torque_constant_nm_per_a"]
                row,field = 1,"a1_torque_nm"
            elif op==0xA3:
                y,row,field = position,2,"a3_relative_position_rad"
            else:
                continue
            streams[(f"r{row}_c{col}",field)].append((t,y))
    for key in streams:
        a = np.array(streams[key])
        if key[1]=="a3_relative_position_rad":
            a[:,1] -= a[0,1]
        streams[key] = a
    return streams,history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,required=True)
    args = p.parse_args()
    data = captures(args.cache)
    exported = defaultdict(list)
    with (HERE/"plotted_response_samples.csv").open() as f:
        for r in csv.DictReader(f):
            exported[(r["mode"],r["panel"],r["field"])].append((float(r["time_s"]),float(r["value"])))
    verified,maximum = 0,0.
    stems = dict(base.STEMS,PI=STEM)
    raw_history = None
    for mode,(meta,_,_) in data.items():
        raw_streams,history = raw_selected(stems[mode],meta)
        if mode=="PI":
            raw_history = history
        for (panel,field),raw in raw_streams.items():
            saved = np.array(exported.pop((mode,panel,field)))
            assert raw.shape == saved.shape
            error = float(np.max(abs(raw-saved)))
            assert error<1e-10
            maximum = max(maximum,error)
            verified += len(raw)
    assert not exported
    history_checks = 0
    with (HERE/"small_step_history.csv").open() as f:
        for r in csv.DictReader(f):
            streams = raw_history[r["name"]]
            ct,i_tau = np.array(streams["integral"]).T
            vt,v = np.array(streams["velocity"]).T
            qt,q = np.array(streams["position"]).T
            pre = np.searchsorted(ct,.6,side="left")-1
            end = np.searchsorted(ct,1.,side="left")-1
            keep = (qt>=.6) & (qt<1.)
            checks = dict(i_before_pulse_nm=i_tau[pre],i_end_pulse_nm=i_tau[end],
                          speed_before_rad_s=v[(vt>=.45) & (vt<.6)].mean(),
                          speed_end_pulse_rad_s=v[(vt>=.85) & (vt<1.)].mean(),
                          a3_pulse_travel_rad=q[keep][-1]-q[keep][0])
            for k,value in checks.items():
                assert abs(value-float(r[k]))<1e-10
                history_checks += 1
    with (OLD/"all_trial_metrics.csv").open() as f:
        old = {r["name"]:r for r in csv.DictReader(f)}
    checks = 0
    with (HERE/"all_trial_metrics.csv").open() as f:
        for r in csv.DictReader(f):
            if r["mode"]=="PI":
                continue
            previous = old.pop(r["name"])
            for k,value in r.items():
                if k in ("mode","name","kind") or not value:
                    assert value==previous[k]
                else:
                    assert abs(float(value)-float(previous[k]))<1e-10
            checks += 1
    assert not old
    xs,ys = [],[]
    meta,phases,_ = data["PI"]
    for p in phases:
        if p["segment"]["kind"] not in ("sine","step"):
            continue
        ct,c = p["streams"]["command"]
        vt,v = p["streams"]["velocity"]
        it,current = p["streams"]["current"]
        q = it-.001
        end = p["segment"]["duration_s"]
        keep = (q>=max(ct[0],vt[0],.03)) & (q<=min(vt[-1],end-.03)) & (it<end)
        t = q[keep]
        j = np.searchsorted(ct,t,side="right")-1
        error = c[j]-np.interp(t,vt,v)
        i_tau = p["streams"]["integral"][1][j]
        x = np.column_stack((error,i_tau,np.ones(len(t))))
        y = current[keep]*meta["torque_constant_nm_per_a"]
        xs.append(x/np.sqrt(len(t))); ys.append(y/np.sqrt(len(t)))
    beta = np.linalg.lstsq(np.concatenate(xs),np.concatenate(ys),rcond=None)[0]
    law = json.loads((HERE/"current_response_fit.json").read_text())["pi_current_response"]
    assert law["apparent_alignment_ms"]==1
    expected = [law["fitted_p_nm_per_rad_s"],law["fitted_integral_torque_factor"],law["fitted_offset_nm"]]
    assert np.allclose(beta,expected,rtol=0,atol=1e-10)
    result = dict(passed=True,raw_plotted_response_samples_verified=verified,maximum_raw_export_error=maximum,
                  previous_baseline_trials_reproduced=checks,independent_dynamic_coefficients=beta.tolist(),
                  raw_small_step_history_fields_verified=history_checks,
                  motor_or_simulator_run=False)
    write_json(HERE/"verification.json",result)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
