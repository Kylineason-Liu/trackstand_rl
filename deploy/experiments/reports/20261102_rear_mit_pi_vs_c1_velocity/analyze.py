"""Compare the completed rear MIT PI campaign with saved MIT P and C1 captures."""

import argparse
from collections import Counter, defaultdict
import csv
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
OLD = HERE.parent/"20261101_rear_mit_vs_c1_velocity"
spec = importlib.util.spec_from_file_location("mit_comparison_base", OLD/"analyze.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
sys.path.insert(0, str(EXP))
from drive_c1_kp_identify import Segment

STEM = "20261102_104627_769455_drive_mit_velocity_test"
MODES = ("C1", "MIT", "PI")
STREAMS = base.STREAMS+("integral", "error_integral", "estimated_torque", "cached_velocity", "sample_time")
write_csv, write_json, sha = base.write_csv, base.write_json, base.sha
held, integral = base.held, base.integral


def read_pi(cache):
    path = EXP/"logs"/(STEM+".csv")
    digest = sha(path)
    saved, arrays = cache/"PI.json", cache/"PI.npz"
    if saved.exists() and arrays.exists():
        d = json.loads(saved.read_text())
        assert d["audit"]["csv_sha256"] == digest
        with np.load(arrays, allow_pickle=False) as a:
            for i, p in enumerate(d["phases"]):
                p["streams"] = {s:(a[f"{i}_{s}_t"], a[f"{i}_{s}_y"]) for s in STREAMS}
        return d["meta"], d["phases"], d["audit"]
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["completed"] and meta["disabled_and_restored"] and meta["disable_confirmed"]
    assert not meta["pending_restore_payloads"] and meta["arguments"]["velocity_ki"] == 3.3
    assert sha(EXP/"drive_mit_velocity_test.py") == meta["source_sha256"]
    assert sha(EXP/"drive_c1_kp_identify.py") == meta["scheduler_sha256"]
    assert sha(EXP/"drive_velocity_sine.py") == meta["helper_sha256"]
    phases = {p["segment"]["name"]:dict(p, streams={s:[] for s in STREAMS}) for p in meta["phases"]}
    with path.with_name(path.stem+"_plan.csv").open() as f:
        plan = list(csv.DictReader(f))
    assert len(phases) == len(meta["plan"]) == len(plan) == 342
    for a, b, p in zip(plan, meta["plan"], phases.values()):
        assert b == p["segment"] and all(a[k] == str(v) for k,v in b.items())
    segments = {k:Segment(**p["segment"]) for k,p in phases.items()}
    counts, faults, modes, flags, ff_codes = Counter(), Counter(), Counter(), Counter(), Counter()
    errors, endpoints, reads = Counter(), Counter(), defaultdict(set)
    writes, statuses, f0_history, volts, temps, late = [], [], [], [], [], []
    kt, ki, ranges = meta["torque_constant_nm_per_a"], 3.3, None
    last_count, travel, zeroes, rows, event_skips, command_skips = meta["initial_multi_turn_count"], 0, 0, 0, 0, 0
    peaks = Counter()

    def check(key, value, recorded, tolerance=1e-10):
        error = abs(value-float(recorded))
        errors[key] = max(errors[key], error)
        assert error <= tolerance, (key, value, recorded)

    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            rows += 1
            kind, op, name = r["record_type"], r["opcode"], r["method"]
            t = float(r["host_monotonic_s"])
            counts[f"{kind}_{op or 'event'}"] += 1
            if kind == "event":
                event_skips += int(r["skipped_slots"] or 0)
                if r["note"].startswith("Capture software zero:"):
                    zeroes += 1
                    assert int(r["multi_turn_count"]) == last_count and float(r["position_rad"]) == 0.
                continue
            data, cid = bytes.fromhex(r["payload_hex"]), int(r["can_id"], 16)
            phase = phases.get(name)
            rel = t-phase["start_monotonic_s"] if phase else None
            if kind == "tx" and len(data)>1:
                assert cid == 0x101 and data[0] in (0xF0, 0xCD)
                writes.append(dict(time=t, payload=data.hex().upper()))
            if kind == "command":
                assert cid == 0x501 and len(data) == 8 and ranges == (95.5,13.,18.)
                desired = segments[name].target(float(r["target_sample_time_s"]))
                check("desired_target", desired, r["target_velocity_rad_s"])
                vcode, pcode = data[2]<<4 | data[3]>>4, (data[3]&15)<<8 | data[4]
                dcode, tcode = data[5]<<4 | data[6]>>4, (data[6]&15)<<8 | data[7]
                assert pcode == 0 and dcode == 246
                kd, ff = dcode*5/4095, (tcode*2/4095-1)*ranges[2]
                wire = -(vcode*2/4095-1)*ranges[1]
                i_tau, i_error = float(r["integral_torque_nm"]), float(r["integral_error_rad"])
                check("integral_gain", ki*i_error, i_tau)
                check("wire_ff", ff, r["mit_feedforward_native_nm"])
                assert tcode == round((-i_tau/ranges[2]+1)*4095/2)
                residual = ff+i_tau
                assert vcode == round(((-desired-residual/kd)/ranges[1]+1)*4095/2)
                effective = wire-residual/kd  # P target; do not turn desired I torque into velocity demand.
                check("wire_velocity", wire, r["encoded_target_velocity_rad_s"])
                check("encoded_kd", kd, r["mit_kd"])
                check("position_kp", 0., r["mit_kp"])
                estimated = kd*(effective-float(r["velocity_rad_s"]))+i_tau
                check("estimated_torque", estimated, r["estimated_mit_total_torque_nm"])
                assert abs(effective-desired) <= ranges[1]/4095+1e-10
                errors["maximum_effective_quantization_rad_s"] = max(
                    errors["maximum_effective_quantization_rad_s"], abs(effective-desired))
                for s,y in (("command",effective), ("wire_command",wire), ("desired",desired),
                            ("integral",i_tau), ("error_integral",i_error), ("estimated_torque",estimated),
                            ("cached_velocity",float(r["velocity_rad_s"])),
                            ("sample_time",float(r["target_sample_time_s"]))):
                    phase["streams"][s].append((rel,y))
                ff_codes[tcode] += 1
                late.append(float(r["lateness_s"]))
                command_skips += int(r["skipped_slots"] or 0)
                peaks["estimated_torque_nm"] = max(peaks["estimated_torque_nm"], abs(estimated))
                peaks["integral_torque_nm"] = max(peaks["integral_torque_nm"], abs(i_tau))
                continue
            if kind != "rx":
                continue
            assert cid == 1
            code, stream, value = data[0], None, None
            if code in (0xB0,0xB3,0xB5,0xB8,0xB9,0xCD,0xF0):
                reads[f"{code:02X}"].add(data.hex().upper())
            if code == 0xF0:
                p,v,z = struct.unpack("<HHH",data[1:])
                ranges = (p*.1,v*.01,z*.01)
                f0_history.append(dict(time=t,payload=data.hex().upper()))
            elif code in (0xAE,0xCF):
                faults[data[7]] += 1; modes[data[6]] += 1
                volts.append(int.from_bytes(data[1:3],"little")*.01); temps.append(data[5])
                statuses.append(dict(time=t,opcode=op,mode=data[6],fault=data[7]))
            elif code == 0xA1:
                raw = struct.unpack("<i",data[1:])[0]*.001
                check("native_current",raw,r["current_a"]); check("current_torque",-raw*kt,r["current_torque_nm"])
                stream,value = "current",-raw
                peaks["a1_torque_nm"] = max(peaks["a1_torque_nm"],abs(raw*kt))
            elif code == 0xA2:
                stream,value = "velocity",-struct.unpack("<i",data[1:])[0]*.01*2*math.pi/60
                check("a2_velocity",value,r["velocity_rad_s"])
                peaks["a2_velocity_rad_s"] = max(peaks["a2_velocity_rad_s"],abs(value))
            elif code == 0xA3:
                count = struct.unpack("<i",data[3:7])[0]
                travel += (count-last_count+2**31)%2**32-2**31
                last_count = count
                stream,value = "position",-travel*2*math.pi/16384
                check("a3_position",value,r["position_rad"])
                assert count == int(r["multi_turn_count"])
            elif code == 0xF1:
                pcode = int.from_bytes(data[1:3],"big")
                vcode,tcode = data[3]<<4 | data[4]>>4, (data[4]&15)<<8 | data[5]
                flags[data[6]] += 1
                assert data[6] == 1
                if ranges:
                    fv,ft = -(vcode*2/4095-1)*ranges[1],-(tcode*2/4095-1)*ranges[2]
                    check("f1_velocity",fv,r["mit_velocity_rad_s"]); check("f1_torque",ft,r["mit_torque_nm"])
                    check("f1_position",(pcode*2/65535-1)*ranges[0],r["mit_position_native_rad"])
                    for key,c,maxcode in (("position",pcode,65535),("velocity",vcode,4095),("torque",tcode,4095)):
                        endpoints[key] += int(c in (0,maxcode))
                    if phase:
                        phase["streams"]["f1_velocity"].append((rel,fv))
                        phase["streams"]["f1_torque"].append((rel,ft))
            if phase and stream:
                phase["streams"][stream].append((rel,value))
    assert set(faults) == {0} and zeroes == 1 and statuses[-1]["mode"] == 0
    for key in ("B8","B9","B3","B5"):
        assert reads[key] == {meta["original_settings"][key]}
    assert statuses[-1]["fault"] == 0
    assert f0_history[-1]["payload"] == meta["original_settings"]["F0"]
    assert bytes.fromhex(sorted(reads["B0"])[0])[2:6] == struct.pack("<f",kt)
    phases, rates, gaps, reconstruction = list(phases.values()), [], [], []
    for p in phases:
        s = p["segment"]
        assert p["actual_duration_s"] >= s["duration_s"]
        rr = dict(mode="PI",name=s["name"],repeat=s["repeat"],kind=s["kind"])
        for key in STREAMS:
            a = np.array(p["streams"][key],dtype=float).reshape((-1,2))
            assert np.isfinite(a).all() and (not len(a) or np.all(np.diff(a[:,0])>0))
            p["streams"][key] = (a[:,0],a[:,1])
            if key in ("command","velocity","current","position"):
                rr[key+"_hz"] = float((len(a)-1)/(a[-1,0]-a[0,0]))
        ct,cv = p["streams"]["command"]
        assert len(ct) == p["commands"] and p["streams"]["integral"][1][0] == 0.
        assert abs(p["streams"]["integral"][1][-1]-p["final_integral_torque_nm"]) < 1e-12
        error = p["streams"]["desired"][1]-p["streams"]["cached_velocity"][1]
        ts = p["streams"]["sample_time"][1]
        reconstructed = np.r_[0.,np.cumsum((error[1:]+error[:-1])*.5*np.diff(ts))]
        reconstruction.append(float(np.max(abs(reconstructed-p["streams"]["error_integral"][1]))))
        gaps.extend(np.diff(ct)); rates.append(rr)
    skipped = sum(p["skipped_slots"] for p in phases)
    assert event_skips+command_skips == skipped
    audit = dict(csv_sha256=digest,rows=rows,packet_counts=dict(counts),maximum_errors=dict(errors),
                 completed_segments=len(phases),commands=sum(p["commands"] for p in phases),skipped_slots=skipped,
                 duration_s=sum(p["actual_duration_s"] for p in phases),peaks=dict(peaks),
                 fault_codes=dict(faults),driver_modes=dict(modes),f1_flags=dict(flags),
                 f1_endpoint_counts=dict(endpoints),readbacks={k:sorted(v) for k,v in reads.items()},
                 configuration_writes=writes,final_statuses=statuses[-4:],f0_history=f0_history,
                 voltage_min_max_v=[min(volts),max(volts)],temperature_min_max_c=[min(temps),max(temps)],
                 command_gap_ms=(np.percentile(gaps,[0,50,95,99,100])*1000).tolist(),
                 command_lateness_ms=(np.percentile(late,[0,50,95,99,100])*1000).tolist(),
                 ff_codes_used=len(ff_codes),rates=rates,
                 anti_windup_limited_updates=sum(p["integral_limited_updates"] for p in phases),
                 integral_sample_stamp_reconstruction_max_error_rad=max(reconstruction),
                 reconstruction_note="Sample timestamps bracket rather than equal the private integration clock.")
    np.savez_compressed(arrays,**{f"{i}_{s}_{a}":p["streams"][s][j] for i,p in enumerate(phases)
                                for s in STREAMS for j,a in enumerate(("t","y"))})
    write_json(saved,dict(meta=meta,audit=audit,
                          phases=[{k:v for k,v in p.items() if k!="streams"} for p in phases]))
    return meta,phases,audit


def captures(cache):
    cache.mkdir(parents=True,exist_ok=True)
    legacy = Path("/tmp/trackstand_mit_c1_analysis_20261101_qz68bvv1")
    result = {m:base.read_capture(m,legacy if legacy.exists() else cache)
              for m in ("C1","MIT")}
    result["PI"] = read_pi(cache)
    for parts in zip(*(result[m][1] for m in MODES)):
        cleaned = [{k:v for k,v in p["segment"].items() if k not in ("name","mode")} for p in parts]
        assert cleaned[0] == cleaned[1] == cleaned[2]
    return result


def small_step_history(phases):
    rows = []
    for p in phases:
        s = p["segment"]
        if s["kind"]!="step" or s["base_rad_s"]!=0. or s["amplitude_rad_s"]!=.1:
            continue
        ct,_ = p["streams"]["command"]
        i_tau = p["streams"]["integral"][1]
        vt,v = p["streams"]["velocity"]
        qt,q = p["streams"]["position"]
        before = np.searchsorted(ct,.6,side="left")-1
        end = np.searchsorted(ct,1.,side="left")-1
        keep = (qt>=.6) & (qt<1.)
        rows.append(dict(name=s["name"],repeat=s["repeat"],initial_velocity_rad_s=p["initial_velocity_rad_s"],
                         i_before_pulse_nm=float(i_tau[before]),i_end_pulse_nm=float(i_tau[end]),
                         speed_before_rad_s=float(v[(vt>=.45) & (vt<.6)].mean()),
                         speed_end_pulse_rad_s=float(v[(vt>=.85) & (vt<1.)].mean()),
                         a3_pulse_travel_rad=float(q[keep][-1]-q[keep][0])))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,required=True)
    args = parser.parse_args()
    data = captures(args.cache)
    rows = [base.metrics(m,p,meta["torque_constant_nm_per_a"]) for m,(meta,phases,_) in data.items() for p in phases]
    steady = base.aggregate([r for r in rows if r["kind"]=="steady"],("mode","base_rad_s"))
    sine = base.aggregate([r for r in rows if r["kind"]=="sine"],
                          ("mode","base_rad_s","amplitude_rad_s","frequency_hz"))
    step = base.aggregate([r for r in rows if r["kind"]=="step"],("mode","base_rad_s","amplitude_rad_s"))
    write_csv(HERE/"small_step_history.csv",small_step_history(data["PI"][1]))
    for name,values in (("all_trial_metrics",rows),("steady_summary",steady),("sine_summary",sine),
                        ("step_summary",step),
                        ("capture_phase_rates",[x for _,_,a in data.values() for x in a["rates"]])):
        write_csv(HERE/(name+".csv"),values)
    write_json(HERE/"raw_audit.json",{m:{k:v for k,v in a.items() if k!="rates"} for m,(_,_,a) in data.items()})
    stems = dict(base.STEMS,PI=STEM)
    write_json(HERE/"statistics.json",dict(
        input_sha256={stem+suffix:sha(EXP/"logs"/(stem+suffix)) for stem in stems.values()
                      for suffix in (".csv",".json","_plan.csv")},
        fixture="User confirmed identical airborne/free conditions, supply/motor settings; "
                "no CAN configuration changes",
        intended_plans_match=True,steady=steady,sine=sine,step=step,
        mean_trial_rms_rad_s={m:{k:float(np.mean([r["velocity_error_rms_rad_s"] for r in rows
                                                 if r["mode"]==m and r["kind"]==k]))
                                for k in ("steady","step","sine")} for m in MODES},
        hardware_or_simulator_run=False))
    print(json.dumps(dict(audit=data["PI"][2],steady=[r for r in steady if r["mode"]=="PI"]),indent=2))


if __name__ == "__main__":
    main()
