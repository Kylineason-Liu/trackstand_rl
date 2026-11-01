"""Independently decode every raw packet and check the recorded identification plan."""

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
sys.path.insert(0, str(EXP))
from drive_c1_kp_identify import Segment

SOURCE = EXP / "logs/20261101_154126_577381_drive_c1_kp_identify.csv"


def main():
    meta = json.loads(SOURCE.with_suffix(".json").read_text())
    phases = {p["segment"]["name"]: p for p in meta["phases"]}
    segments = {name: Segment(**p["segment"]) for name, p in phases.items()}
    with SOURCE.with_name(SOURCE.stem + "_plan.csv").open() as stream:
        plan_rows = list(csv.DictReader(stream))
    assert len(plan_rows) == len(meta["plan"]) == len(phases) == 342
    for csv_plan, saved_plan, phase in zip(plan_rows, meta["plan"], meta["phases"]):
        assert saved_plan == phase["segment"]
        assert all(str(saved_plan[k]) == value for k, value in csv_plan.items())
    counts, types, modes, faults, op_values = Counter(), Counter(), Counter(), Counter(), defaultdict(set)
    times = defaultdict(lambda: defaultdict(list))
    maximum_errors = Counter()
    maximum_speed = maximum_torque = 0.
    voltage, temperatures, late = [], [], []
    changes, skips, events, statuses, cd_readbacks = [], [], [], [], []
    command_count = 0
    last_count, travel = None, 0
    kt = meta["torque_constant_nm_per_a"]
    rows = 0
    with SOURCE.open(newline="") as stream:
        for row in csv.DictReader(stream):
            rows += 1
            kind, name = row["record_type"], row["method"]
            types[kind] += 1
            if kind == "event":
                events.append({k: row[k] for k in ("elapsed_s", "method", "note", "skipped_slots")})
            payload = bytes.fromhex(row["payload_hex"])
            if not payload:
                continue
            op, t = payload[0], float(row["host_monotonic_s"])
            opcode = f"{op:02X}"
            counts[f"{kind}_{opcode}"] += 1
            if name in phases:
                times[name][f"{kind}_{opcode}"].append(t)
            if kind in ("tx", "command") and len(payload) > 1:
                changes.append(dict(time=t, opcode=opcode, payload=payload.hex(), phase=name)) if op != 0xC1 else None
                assert op in (0xCD, 0xC1), f"Unexpected setting write: {payload.hex()}"
            if kind == "command":
                assert op == 0xC1 and row["can_id"] == "101" and name in phases
                command_count += 1
                sample = float(row["target_sample_time_s"])
                desired = segments[name].target(sample)
                maximum_errors["desired_target_rad_s"] = max(
                    maximum_errors["desired_target_rad_s"], abs(desired-float(row["target_velocity_rad_s"])))
                code = struct.unpack("<i", payload[1:])[0]
                assert code == round(-desired*60/(2*math.pi)*100)
                encoded = -code*.01*2*math.pi/60
                maximum_errors["encoded_target_rad_s"] = max(
                    maximum_errors["encoded_target_rad_s"], abs(encoded-float(row["encoded_target_velocity_rad_s"])))
                late.append(float(row["lateness_s"]))
                if int(row["skipped_slots"] or 0):
                    skips.append(dict(phase=name, sample_time_s=sample, skipped_slots=int(row["skipped_slots"])))
            if kind != "rx":
                continue
            assert row["can_id"] == "001"
            if op in (0xB0, 0xB3, 0xB5, 0xB8, 0xB9, 0xCD):
                op_values[opcode].add(payload.hex().upper())
            if op == 0xCD:
                cd_readbacks.append(dict(time=t, payload=payload.hex().upper()))
            if op in (0xA2, 0xC1):
                value = -struct.unpack("<i", payload[1:])[0]*.01*2*math.pi/60
                maximum_errors["velocity_rad_s"] = max(
                    maximum_errors["velocity_rad_s"], abs(value-float(row["velocity_rad_s"])))
                maximum_speed = max(maximum_speed, abs(value))
            if op == 0xA1:
                current = struct.unpack("<i", payload[1:])[0]*.001
                torque = -current*kt
                maximum_errors["raw_current_a"] = max(
                    maximum_errors["raw_current_a"], abs(current-float(row["current_a"])))
                maximum_errors["torque_nm"] = max(
                    maximum_errors["torque_nm"], abs(torque-float(row["current_torque_nm"])))
                maximum_torque = max(maximum_torque, abs(torque))
            if op == 0xA3:
                count = struct.unpack("<i", payload[3:7])[0]
                if last_count is None:
                    assert count == meta["initial_multi_turn_count"]
                else:
                    travel += (count-last_count+2**31) % 2**32-2**31
                last_count = count
                position = -travel*2*math.pi/16384
                maximum_errors["position_rad"] = max(
                    maximum_errors["position_rad"], abs(position-float(row["position_rad"])))
                assert count == int(row["multi_turn_count"])
            if op in (0xAE, 0xCF):
                volts, amps = struct.unpack("<HH", payload[1:5])
                voltage.append(volts*.01)
                temperatures.append(payload[5])
                modes[payload[6]] += 1
                faults[payload[7]] += 1
                statuses.append(dict(time=t, opcode=opcode, mode=payload[6], fault=payload[7],
                                     payload=payload.hex().upper(), voltage_v=volts*.01, bus_current_a=amps*.01))
    assert all(v < 1e-10 for v in maximum_errors.values()), maximum_errors
    phase_rows = []
    gaps = []
    for name, p in phases.items():
        rates = dict(name=name, kind=p["segment"]["kind"], repeat=p["segment"]["repeat"],
                     duration_s=p["actual_duration_s"])
        for key in ("command_C1", "rx_A1", "rx_A2", "rx_A3", "rx_AE"):
            stamp = np.array(times[name][key])
            rates[key + "_count"] = len(stamp)
            rates[key + "_hz"] = (len(stamp)-1)/(stamp[-1]-stamp[0])
            rates[key + "_max_gap_ms"] = float(max(np.diff(stamp))*1000)
            if key == "command_C1":
                assert len(stamp) == p["commands"]
                gaps.extend(np.diff(stamp))
        phase_rows.append(rates)
    with (HERE / "capture_phase_rates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(phase_rows[0]))
        writer.writeheader()
        writer.writerows(phase_rows)
    assert faults == {0: sum(faults.values())}
    assert op_values["B8"] == {"B80000003F"} and op_values["B9"] == {"B90AD7A33B"}
    assert statuses[-1]["opcode"] == "CF" and statuses[-1]["mode"] == 0
    assert changes[-1]["payload"].upper() == meta["original_settings"]["CD"]
    assert cd_readbacks[-1]["payload"] == meta["original_settings"]["CD"]
    assert cd_readbacks[-1]["time"] > changes[-1]["time"]
    assert command_count == sum(p["commands"] for p in phases.values())
    skipped_in_events = sum(int(r["skipped_slots"] or 0) for r in events)
    skipped_in_commands = sum(r["skipped_slots"] for r in skips)
    skipped_in_metadata = sum(p["skipped_slots"] for p in phases.values())
    assert skipped_in_events + skipped_in_commands == skipped_in_metadata == 1
    audit = dict(
        source=str(SOURCE.relative_to(EXP)), rows=rows, record_types=dict(types), packet_counts=dict(counts),
        planned_and_completed_phases=len(phases), kinds=dict(Counter(p["segment"]["kind"] for p in phases.values())),
        duration_s=sum(p["actual_duration_s"] for p in phases.values()), commands=command_count,
        planned_commands=sum(round(p["segment"]["duration_s"]*200) for p in phases.values()),
        skipped_slots=skips, total_skipped_slots=skipped_in_metadata,
        event_records=events, maximum_decode_errors=dict(maximum_errors),
        settings_writes=changes, cd_readback_history=cd_readbacks,
        gain_and_setting_readbacks={k: sorted(v) for k, v in op_values.items()},
        maximum_measured_abs_speed_rad_s=maximum_speed, maximum_measured_abs_current_torque_nm=maximum_torque,
        voltage_min_max_v=[min(voltage), max(voltage)], temperature_min_max_c=[min(temperatures), max(temperatures)],
        fault_counts=dict(faults), driver_mode_counts=dict(modes), final_statuses=statuses[-5:],
        command_gap_percentiles_ms=np.percentile(gaps, [0, 50, 95, 99, 100]).tolist(),
        lateness_percentiles_ms=(np.percentile(late, [0, 50, 95, 99, 100])*1000).tolist(),
        notes=["command_gap_percentiles_ms converted below from seconds",
               "CSV current_a is native A1 current; current_torque_nm uses verified forward-positive sign -1.",
               "All target samples and payloads match the plan; no gain/zero/current-limit writes.",
               "Fresh A1/A2/A3 rows only were used; repeated state snapshots are not independent samples."],
    )
    audit["command_gap_percentiles_ms"] = (np.percentile(gaps, [0, 50, 95, 99, 100])*1000).tolist()
    audit["notes"].pop(0)
    (HERE / "capture_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k not in ("event_records", "packet_counts")}, indent=2))


if __name__ == "__main__":
    main()
