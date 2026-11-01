"""Offline MIT-P versus C1-PI comparison. Reads raw data; never opens CAN/simulation."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
sys.path.insert(0, str(EXP))
import analyze_drive_c1_kp as core
from drive_c1_kp_identify import Segment

STEMS = {"MIT": "20261101_212240_940924_drive_mit_velocity_test",
         "C1": "20261101_154126_577381_drive_c1_kp_identify"}
STREAMS = ("command", "wire_command", "desired", "velocity", "current", "position", "f1_velocity", "f1_torque")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def held(t, values, query):
    j = np.clip(np.searchsorted(t, query, side="right")-1, 0, len(t)-1)
    return values[j]


def integral(t, values, query):
    j = np.clip(np.searchsorted(t, query, side="right")-1, 0, len(t)-1)
    cumulative = np.r_[0., np.cumsum(values[:-1]*np.diff(t))]
    return cumulative[j]+values[j]*(query-t[j])


def read_capture(label, cache):
    path = EXP / "logs" / (STEMS[label]+".csv")
    digest = sha(path)
    saved, arrays = cache/(label+".json"), cache/(label+".npz")
    if saved.exists() and arrays.exists():
        data = json.loads(saved.read_text())
        assert data["audit"]["csv_sha256"] == digest
        phases = data["phases"]
        with np.load(arrays, allow_pickle=False) as archive:
            for i, phase in enumerate(phases):
                phase["streams"] = {s: (archive[f"{i}_{s}_t"], archive[f"{i}_{s}_y"]) for s in STREAMS}
        return data["meta"], phases, data["audit"]
    meta = json.loads(path.with_suffix(".json").read_text())
    phases = {p["segment"]["name"]: dict(p, streams={s: [] for s in STREAMS}) for p in meta["phases"]}
    assert meta["completed"] and meta["disabled_and_restored"]
    assert len(phases) == len(meta["plan"]) == 342
    with path.with_name(path.stem+"_plan.csv").open() as stream:
        plan = list(csv.DictReader(stream))
    for a, b, c in zip(plan, meta["plan"], phases.values()):
        assert b == c["segment"] and all(a[k] == str(v) for k, v in b.items())
    segments = {name: Segment(**p["segment"]) for name, p in phases.items()}
    counts, faults, modes, f1_flags = Counter(), Counter(), Counter(), Counter()
    reads, writes = defaultdict(set), []
    voltages, temps, late = [], [], []
    errors, gains, ff_codes = Counter(), Counter(), Counter()
    current_ranges = None
    last_count, travel = meta["initial_multi_turn_count"], 0
    zero_events, event_skips, command_skips, rows = 0, 0, 0, 0
    peak_torque, peak_speed, f1_saturated = 0., 0., Counter()
    statuses, f0_history = [], []
    kt = meta["torque_constant_nm_per_a"]

    def check(key, actual, recorded, tolerance=1e-10):
        error = abs(actual-float(recorded))
        errors[key] = max(errors[key], error)
        assert error <= tolerance, (key, actual, recorded)

    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            rows += 1
            kind, name, op = row["record_type"], row["method"], row["opcode"]
            stamp = float(row["host_monotonic_s"])
            counts[f"{kind}_{op or 'event'}"] += 1
            if kind == "event":
                event_skips += int(row["skipped_slots"] or 0)
                if row["note"].startswith("Capture software zero:"):
                    zero_events += 1
                    assert int(row["multi_turn_count"]) == meta["initial_multi_turn_count"]
                    assert float(row["position_rad"]) == 0.
                continue
            data = bytes.fromhex(row["payload_hex"])
            if not data:
                continue
            cid = int(row["can_id"], 16)
            phase = phases.get(name)
            rel = stamp-phase["start_monotonic_s"] if phase else None
            if kind == "tx" and len(data) > 1:
                assert cid == 0x101 and data[0] in ((0xF0, 0xCD) if label == "MIT" else (0xCD,))
                writes.append(dict(time=stamp, payload=data.hex().upper()))
            if kind == "command":
                assert phase is not None
                desired = segments[name].target(float(row["target_sample_time_s"]))
                check("desired_target", desired, row["target_velocity_rad_s"])
                if label == "MIT":
                    assert cid == 0x501 and len(data) == 8 and current_ranges == (95.5, 13., 18.)
                    vcode = (data[2] << 4) | (data[3] >> 4)
                    pcode = ((data[3] & 15) << 8) | data[4]
                    dcode = (data[5] << 4) | (data[6] >> 4)
                    tcode = ((data[6] & 15) << 8) | data[7]
                    assert pcode == 0
                    kd = dcode*5/4095
                    wire = -(vcode*2/4095-1)*current_ranges[1]
                    ff = (tcode*2/4095-1)*current_ranges[2]
                    effective = wire-ff/kd
                    check("wire_ff", ff, row["mit_feedforward_native_nm"])
                    check("encoded_kd", kd, row["mit_kd"])
                    check("position_kp", 0., row["mit_kp"])
                    check("integral_error", 0., row["integral_error_rad"])
                    check("integral_torque", 0., row["integral_torque_nm"])
                    check("software_torque_estimate", kd*(effective-float(row["velocity_rad_s"])),
                          row["estimated_mit_total_torque_nm"])
                    assert abs(effective-desired) <= current_ranges[1]/4095+1e-10
                    gains[dcode] += 1
                    ff_codes[tcode] += 1
                else:
                    assert cid == 0x101 and data[0] == 0xC1
                    wire = -struct.unpack("<i", data[1:])[0]*.01*2*math.pi/60
                    effective = wire
                check("wire_target", wire, row["encoded_target_velocity_rad_s"])
                errors["maximum_effective_quantization_rad_s"] = max(
                    errors["maximum_effective_quantization_rad_s"], abs(effective-desired))
                for s, value in (("command", effective), ("wire_command", wire), ("desired", desired)):
                    phase["streams"][s].append((rel, value))
                late.append(float(row["lateness_s"]))
                command_skips += int(row["skipped_slots"] or 0)
                continue
            if kind != "rx":
                continue
            assert cid == 1
            code = data[0]
            value, stream_name = None, None
            if code in (0xB0, 0xB3, 0xB5, 0xB8, 0xB9, 0xCD, 0xF0):
                reads[f"{code:02X}"].add(data.hex().upper())
            if code == 0xF0:
                p, v, t = struct.unpack("<HHH", data[1:])
                current_ranges = (p*.1, v*.01, t*.01)
                f0_history.append(dict(time=stamp, payload=data.hex().upper()))
            elif code in (0xAE, 0xCF):
                faults[data[7]] += 1
                modes[data[6]] += 1
                voltages.append(int.from_bytes(data[1:3], "little")*.01)
                temps.append(data[5])
                statuses.append(dict(time=stamp, opcode=f"{code:02X}", mode=data[6], fault=data[7]))
            elif code in (0xA2, 0xC1):
                value = -struct.unpack("<i", data[1:])[0]*.01*2*math.pi/60
                check("a2_or_c1_velocity", value, row["velocity_rad_s"])
                peak_speed = max(peak_speed, abs(value))
                if code == 0xA2:
                    stream_name = "velocity"
            elif code == 0xA1:
                raw = struct.unpack("<i", data[1:])[0]*.001
                check("native_current", raw, row["current_a"])
                check("current_torque", -raw*kt, row["current_torque_nm"])
                value, stream_name = -raw, "current"
                peak_torque = max(peak_torque, abs(value*kt))
            elif code == 0xA3:
                count = struct.unpack("<i", data[3:7])[0]
                travel += (count-last_count+2**31) % 2**32-2**31
                last_count = count
                value, stream_name = -travel*2*math.pi/16384, "position"
                check("a3_position", value, row["position_rad"])
                assert count == int(row["multi_turn_count"])
            elif code == 0xF1:
                vcode = (data[3] << 4) | (data[4] >> 4)
                tcode = ((data[4] & 15) << 8) | data[5]
                f1_flags[data[6]] += 1
                assert not data[6] & 2
                if current_ranges is not None:
                    fv = -(vcode*2/4095-1)*current_ranges[1]
                    ft = -(tcode*2/4095-1)*current_ranges[2]
                    check("f1_velocity", fv, row["mit_velocity_rad_s"])
                    check("f1_torque", ft, row["mit_torque_nm"])
                    f1_saturated["velocity_endpoint"] += int(vcode in (0, 4095))
                    f1_saturated["torque_endpoint"] += int(tcode in (0, 4095))
                    if phase is not None:
                        phase["streams"]["f1_velocity"].append((rel, fv))
                        phase["streams"]["f1_torque"].append((rel, ft))
            if phase is not None and stream_name is not None:
                phase["streams"][stream_name].append((rel, value))
    phases = list(phases.values())
    rates, gaps = [], []
    for phase in phases:
        s = phase["segment"]
        assert phase["actual_duration_s"] >= s["duration_s"]
        row = dict(mode=label, name=s["name"], repeat=s["repeat"], kind=s["kind"])
        for key in STREAMS:
            values = np.array(phase["streams"][key], dtype=float).reshape((-1, 2))
            assert np.all(np.isfinite(values))
            assert len(values) == 0 or np.all(np.diff(values[:, 0]) > 0)
            phase["streams"][key] = (values[:, 0], values[:, 1])
            if key in ("command", "velocity", "current", "position"):
                assert len(values) >= 4
                row[key+"_hz"] = float((len(values)-1)/(values[-1, 0]-values[0, 0]))
        ct, _ = phase["streams"]["command"]
        assert len(ct) == phase["commands"]
        gaps.extend(np.diff(ct))
        rates.append(row)
    assert set(faults) == {0}
    assert reads["B8"] == {meta["original_settings"]["B8"]}
    assert reads["B9"] == {meta["original_settings"]["B9"]}
    if label == "MIT":
        assert gains == {246: sum(gains.values())} and set(ff_codes) == {2048}
        assert zero_events == 1 and f0_history[-1]["payload"] == meta["original_settings"]["F0"]
        assert statuses[-1]["mode"] == 0
    skipped = sum(p["skipped_slots"] for p in phases)
    assert event_skips+command_skips == skipped
    audit = dict(csv_sha256=digest, rows=rows, packet_counts=dict(counts), maximum_errors=dict(errors),
                 completed_segments=len(phases), commands=sum(p["commands"] for p in phases), skipped_slots=skipped,
                 duration_s=sum(p["actual_duration_s"] for p in phases),
                 gain_codes=dict(gains), feedforward_codes=dict(ff_codes),
                 configuration_writes=writes, readbacks={k: sorted(v) for k, v in reads.items()},
                 f0_history=f0_history, final_statuses=statuses[-4:], fault_codes=dict(faults),
                 driver_modes=dict(modes), f1_flags=dict(f1_flags), f1_endpoint_counts=dict(f1_saturated),
                 voltage_min_max_v=[min(voltages), max(voltages)], temperature_min_max_c=[min(temps), max(temps)],
                 peak_a1_torque_nm=peak_torque, peak_a2_or_c1_speed_rad_s=peak_speed,
                 command_gap_ms=np.percentile(gaps, [0, 50, 95, 99, 100]).tolist(),
                 command_lateness_ms=(np.percentile(late, [0, 50, 95, 99, 100])*1000).tolist(), rates=rates)
    audit["command_gap_ms"] = (np.percentile(gaps, [0, 50, 95, 99, 100])*1000).tolist()
    raw = {f"{i}_{s}_{axis}": phase["streams"][s][j] for i, phase in enumerate(phases)
           for s in STREAMS for j, axis in enumerate(("t", "y"))}
    np.savez_compressed(arrays, **raw)
    write_json(saved, dict(meta=meta, audit=audit,
                          phases=[{k: v for k, v in p.items() if k != "streams"} for p in phases]))
    return meta, phases, audit


def metrics(label, phase, kt):
    s, streams = phase["segment"], phase["streams"]
    row = dict(mode=label, name=s["name"], repeat=s["repeat"], kind=s["kind"],
               base_rad_s=s["base_rad_s"], amplitude_rad_s=s["amplitude_rad_s"], frequency_hz=s["frequency_hz"])
    start, stop = s["settle_s"], s["duration_s"]
    if s["kind"] == "step":
        start, stop = s["pulse_start_s"], s["pulse_end_s"]
    elif s["kind"] == "zero":
        start = 1.
    ct, command = streams["command"]
    vt, velocity = streams["velocity"]
    it, current = streams["current"]
    pt, position = streams["position"]
    mask = (vt >= max(start, ct[0])) & (vt < stop)
    reference = held(ct, command, vt[mask])
    error = velocity[mask]-reference
    imask, pmask = (it >= start) & (it < stop), (pt >= start) & (pt < stop)
    qtime, q = pt[pmask], position[pmask]
    target_travel = integral(ct, command, qtime[-1])-integral(ct, command, qtime[0])
    row.update(start_s=start, stop_s=stop, velocity_mean_rad_s=float(velocity[mask].mean()),
               velocity_std_rad_s=float(velocity[mask].std()), velocity_error_mean_rad_s=float(error.mean()),
               velocity_error_rms_rad_s=float(np.sqrt(np.mean(error**2))),
               torque_mean_nm=float(current[imask].mean()*kt),
               torque_rms_nm=float(np.sqrt(np.mean(current[imask]**2))*kt),
               torque_peak_nm=float(np.max(abs(current[imask]))*kt), a3_range_rad=float(np.ptp(q)),
               a3_mean_speed_rad_s=float((q[-1]-q[0])/(qtime[-1]-qtime[0])),
               commanded_travel_rad=float(target_travel), actual_a3_travel_rad=float(q[-1]-q[0]),
               unchanged_a3_fraction=float(np.mean(np.diff(q) == 0)),
               fundamental_gain=None, phase_equivalent_lag_ms=None, a3_velocity_fundamental_gain=None,
               step_increment_gain=None)
    if s["kind"] == "sine":
        f = s["frequency_hz"]
        target_h = core.held_harmonic(ct, command, f, start, stop)
        vh, vse = core.harmonic(vt, velocity, f, start, stop)
        qh, _ = core.harmonic(pt, position, f, start, stop)
        h = vh/target_h
        row["fundamental_gain"] = float(abs(h))
        if abs(h) >= .1 and abs(vh) >= 3*vse:
            row["phase_equivalent_lag_ms"] = float(-np.angle(h)/(2*math.pi*f)*1000)
        row["a3_velocity_fundamental_gain"] = float(abs(2j*math.pi*f*qh/target_h))
    elif s["kind"] == "step":
        base = velocity[(vt >= s["pulse_start_s"]-.15) & (vt < s["pulse_start_s"])]
        pulse = velocity[(vt >= s["pulse_end_s"]-.15) & (vt < s["pulse_end_s"])]
        row["step_increment_gain"] = float((pulse.mean()-base.mean())/s["amplitude_rad_s"])
    return row


def aggregate(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    output = []
    fields = ("velocity_mean_rad_s", "velocity_std_rad_s", "velocity_error_mean_rad_s", "velocity_error_rms_rad_s",
              "torque_mean_nm", "torque_rms_nm", "a3_mean_speed_rad_s", "unchanged_a3_fraction",
              "fundamental_gain", "phase_equivalent_lag_ms", "a3_velocity_fundamental_gain", "step_increment_gain")
    for key, part in sorted(groups.items(), key=lambda item: str(item[0])):
        row = dict(zip(keys, key))
        row["trials"] = len(part)
        for field in fields:
            values = [r[field] for r in part if r[field] is not None]
            row[field+"_mean"] = float(np.mean(values)) if values else None
            row[field+"_min"] = min(values) if values else None
            row[field+"_max"] = max(values) if values else None
        output.append(row)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    captures = {label: read_capture(label, args.cache) for label in STEMS}
    for mit, c1 in zip(captures["MIT"][1], captures["C1"][1]):
        a, b = dict(mit["segment"]), dict(c1["segment"])
        for key in ("name", "mode"):
            a.pop(key); b.pop(key)
        assert a == b
    rows = [metrics(label, p, meta["torque_constant_nm_per_a"])
            for label, (meta, phases, _) in captures.items() for p in phases]
    steady = aggregate([r for r in rows if r["kind"] == "steady"], ("mode", "base_rad_s"))
    sine = aggregate([r for r in rows if r["kind"] == "sine"],
                     ("mode", "base_rad_s", "amplitude_rad_s", "frequency_hz"))
    step = aggregate([r for r in rows if r["kind"] == "step"], ("mode", "base_rad_s", "amplitude_rad_s"))
    write_csv(HERE/"all_trial_metrics.csv", rows)
    write_csv(HERE/"steady_summary.csv", steady)
    write_csv(HERE/"sine_summary.csv", sine)
    write_csv(HERE/"step_summary.csv", step)
    write_csv(HERE/"capture_phase_rates.csv", [r for _, _, audit in captures.values() for r in audit["rates"]])
    write_json(HERE/"raw_audit.json", {k: {a: b for a, b in v[2].items() if a != "rates"}
                                      for k, v in captures.items()})
    group_rms = {label: {kind: float(np.mean([r["velocity_error_rms_rad_s"] for r in rows
                                            if r["mode"] == label and r["kind"] == kind]))
                         for kind in ("steady", "step", "sine")} for label in STEMS}
    input_hashes = {STEMS[label]+suffix: sha(EXP/"logs"/(STEMS[label]+suffix))
                    for label in STEMS for suffix in (".csv", ".json", "_plan.csv")}
    write_json(HERE/"statistics.json", dict(input_sha256=input_hashes, intended_plans_match=True,
                                           fixture="User confirmed identical airborne/free conditions and settings",
                                           steady=steady, sine=sine, step=step, mean_trial_rms_rad_s=group_rms,
                                           motor_or_simulator_run_during_analysis=False))
    print(json.dumps(dict(steady=[r for r in steady if r["base_rad_s"] in (.05, .1, .2, .4, .8, 3., 10.)],
                         audit={k: {a: b for a, b in v[2].items() if a in
                                    ("rows", "commands", "skipped_slots", "peak_a1_torque_nm", "duration_s",
                                     "peak_a2_or_c1_speed_rad_s", "voltage_min_max_v", "temperature_min_max_c")}
                                for k, v in captures.items()}), indent=2))


if __name__ == "__main__":
    main()
