"""Check derived conclusions and extracted traces against raw CAN without importing the analyzer."""

from collections import Counter, defaultdict
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import struct

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    stats = json.loads((HERE / "statistics.json").read_text())
    protected = stats["input_sha256"]
    references = {"progress_record.md", "progress_record.txt"}
    reference_changes = {}
    for name, digest in protected.items():
        current = sha(ROOT / name)
        if name in references:
            if current != digest:
                reference_changes[name] = dict(analysis_sha256=digest, verification_sha256=current)
        else:
            assert current == digest, name
    source = ROOT / (stats["source_stem"] + ".csv")
    metadata = json.loads(source.with_suffix(".json").read_text())
    trials = {r["trial_id"]: r for r in stats["trials"]}
    outputs = {}
    for name in ("startup_samples.csv", "selected_trial_traces.csv"):
        with (HERE / name).open() as stream:
            outputs[name] = list(csv.DictReader(stream))
    expected_rows = {int(r["csv_line"]): r for rows in outputs.values() for r in rows}
    checked_lines = set()
    commands, position, torque, velocity = (defaultdict(list) for _ in range(4))
    latest_count, query_results, line_count = None, Counter(), 0
    kt = metadata["torque_constant_nm_per_a"]
    count_rad = 2 * math.pi / 16384

    with source.open() as stream:
        for line, row in enumerate(csv.DictReader(stream), 2):
            line_count += 1
            if row["query_status"]:
                query_results[row["query_status"]] += 1
            if row["can_direction"] not in ("local_tx", "bus_rx"):
                continue
            data = bytes.fromhex(row["payload_hex"])
            op = data[0]
            trial = int(row["trial_id"] or 0)
            t = float(row["elapsed_s"])
            if row["can_direction"] == "local_tx":
                if op == 0xC0:
                    expected = trials[trial]
                    expected_ma = round(-expected["requested_torque_nm"] / kt * 1000)
                    assert data == bytes([0xC0]) + struct.pack("<i", expected_ma)
                    if not commands[trial]:
                        assert latest_count == expected["start_count"]
                    commands[trial].append(t)
                continue
            if op == 0xA3:
                latest_count = struct.unpack("<i", data[3:7])[0]
            active = trial and row["phase"] == "torque" and commands[trial] and t >= commands[trial][0]
            if not active or op not in (0xA1, 0xA2, 0xA3, 0xC0):
                continue
            kernel = float(row["kernel_time_unix_s"])
            if op == 0xA3:
                value = -(latest_count - trials[trial]["start_count"]) * count_rad
                position[trial].append((t, kernel, value))
            elif op == 0xA2:
                value = -struct.unpack("<i", data[1:5])[0] * .01 * math.pi / 30
                velocity[trial].append((t, kernel, value))
            else:
                value = -struct.unpack("<i", data[1:5])[0] * .001 * kt
                if op == 0xA1:
                    torque[trial].append(value)
            if line in expected_rows:
                expected = expected_rows[line]
                assert int(expected["trial"]) == trial and expected["opcode"] == f"{op:02X}"
                assert abs(float(expected["time_s"]) - (t - commands[trial][0])) < 1e-10
                assert abs(float(expected["value"]) - value) < 1e-10, (line, expected, value)
                checked_lines.add(line)

    assert line_count == 344039 and source.stat().st_size == 109534334
    assert checked_lines == set(expected_rows)
    assert dict(query_results) == {"received": 106332, "cancelled_phase_boundary": 36}
    assert sum(map(len, commands.values())) == 12445
    kernel_velocity_deficits = []
    for trial, info in trials.items():
        assert len(commands[trial]) == info["C0_packets"]
        assert abs(max(np.diff(commands[trial])) - info["max_command_gap_s"]) < 1e-10
        assert len(position[trial]) == info["A3_samples"]
        assert len(velocity[trial]) == info["A2_samples"]
        assert len(torque[trial]) == info["A1_samples"]
        assert abs(position[trial][-1][2] - info["displacement_rad"]) < 1e-10
        assert abs(sum(torque[trial]) / len(torque[trial]) - info["torque_mean_nm"]) < 1e-10
        if info["sustained_rotation"]:
            # Independently use raw kernel realtime rather than the mapped elapsed column.
            q = np.array(position[trial])
            v = np.array(velocity[trial])
            start, end = q[0, 1], q[-1, 1]
            within = (v[:, 1] > start) & (v[:, 1] < end)
            times = np.concatenate(([start], v[within, 1], [end]))
            speeds = np.interp(times, v[:, 1], v[:, 2])
            integral = np.sum((speeds[1:] + speeds[:-1]) / 2 * np.diff(times))
            kernel_velocity_deficits.append(100 * (1 - integral / (q[-1, 2] - q[0, 2])))

    expected_counts = [(3, 2), (3, 4), (5, 3), (3, 5), (5, 5)]
    actual_counts = []
    for level in (.12, .13, .14, .15, .16):
        actual_counts.append(tuple(sum(r["sustained_rotation"] for r in trials.values()
                                       if r["torque_level_nm"] == level and r["direction"] == direction)
                                   for direction in ("forward", "backward")))
    assert actual_counts == expected_counts
    moving = [r for r in trials.values() if r["sustained_rotation"]]
    other = [r for r in trials.values() if not r["sustained_rotation"]]
    assert len(moving) == 38 and len(other) == 12
    assert min(abs(r["displacement_rad"]) for r in moving) > 103
    assert max(r["max_excursion_rad"] for r in other) < .032
    assert all(r["pre_half_second_position_excursion_rad"] <= count_rad for r in list(trials.values())[1:])
    assert max(abs(r["start_velocity_rad_s"]) for r in trials.values()) < .063
    assert min(kernel_velocity_deficits) > 1.36 and max(kernel_velocity_deficits) < 1.57

    for filename in ("analyze.py", "plot.py", "verify.py"):
        ast.parse((HERE / filename).read_text())
    local_links = re.findall(r"\]\(([^)]+)\)", (HERE / "README.md").read_text())
    for target in local_links:
        if target != "verification.json":
            assert (HERE / target).is_file(), target
    for name, digest in protected.items():
        if name not in references:
            assert sha(ROOT / name) == digest, name

    result = dict(passed=True, raw_csv_rows=line_count, raw_C0_transmissions=12445,
                  reconstructed_trial_count=50, checked_extracted_raw_lines=len(checked_lines),
                  data_and_runtime_source_hashes_unchanged=True,
                  progress_reference_changes_since_analysis_snapshot=reference_changes,
                  outcome_counts_verified=True,
                  kernel_timestamp_A2_below_A3_percent_range=[min(kernel_velocity_deficits),
                                                             max(kernel_velocity_deficits)],
                  local_links_checked=len(local_links), all_three_python_files_parse=True,
                  artifacts_sha256={p.name: sha(p) for p in sorted(HERE.iterdir())
                                    if p.is_file() and p.name != "verification.json"})
    (HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "artifacts_sha256"}, indent=2))


if __name__ == "__main__":
    main()
