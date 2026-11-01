"""Independent checks of original data, exported traces, RMS, plateaus and figures."""

import ast
from bisect import bisect_right
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
statistics = json.loads((HERE / "statistics.json").read_text())
protected = json.loads((HERE / "protected_input_hashes.json").read_text())
for name, digest in protected.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


def read(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


checked_rows, collapsed_codes = 0, {}
for rate in (50, 200):
    run = statistics["runs"][str(rate)]
    source = ROOT / "deploy/experiments/logs" / run["source_csv"]
    rows = read(source)
    origin = run["core_phase"]["start_monotonic_s"]
    series = {}
    for name, kind, opcode, field in (("commands", "command", "C1", "encoded_target_velocity_rad_s"),
                                      ("a2_samples", "rx", "A2", "velocity_rad_s"),
                                      ("a1_samples", "rx", "A1", "current_torque_nm"),
                                      ("a3_samples", "rx", "A3", "position_rad"),
                                      ("c1_feedback_samples", "rx", "C1", "velocity_rad_s")):
        selected = [r for r in rows if r["method"] == "core" and r["record_type"] == kind and r["opcode"] == opcode]
        exported = read(HERE / f"{rate}hz_{name}.csv")
        assert len(exported) == len(selected)
        output_field = {"commands": "velocity_rad_s", "a2_samples": "velocity_rad_s",
                        "a1_samples": "torque_nm", "a3_samples": "position_rad",
                        "c1_feedback_samples": "velocity_rad_s"}[name]
        times, values = [], []
        for raw, saved in zip(selected, exported):
            time = float(raw["host_monotonic_s"]) - origin
            value = float(raw[field])
            if name == "a3_samples":
                value -= run["core_phase"]["initial_position_rad"]
            assert float(saved["time_s"]) == time
            assert float(saved[output_field]) == value
            times.append(time)
            values.append(value)
        series[name] = times, values
        checked_rows += len(exported)
    command_times, targets = series["commands"]
    velocity_times, velocity = series["a2_samples"]
    # Pure-Python causal lookup/RMS is separate from the NumPy analysis implementation.
    for name, metrics in run["windows"].items():
        errors, actual = [], []
        for time, value in zip(velocity_times, velocity):
            if max(metrics["lower_s"], command_times[0]) <= time < metrics["upper_s"]:
                target = targets[bisect_right(command_times, time) - 1]
                errors.append(value - target)
                actual.append(value)
        error_rms = math.sqrt(sum(e * e for e in errors) / len(errors))
        assert abs(error_rms - metrics["error_rms_rad_s"]) < 1e-12
        assert len(errors) == metrics["samples"]
    tx = [r for r in rows if r["method"] == "core" and r["record_type"] == "command"]
    codes = [struct.unpack("<i", bytes.fromhex(r["payload_hex"])[1:])[0] for r in tx]
    collapsed_codes[rate] = codes[::rate // 50]
    for i in range(500):
        assert len(set(codes[i * (rate // 50):(i + 1) * (rate // 50)])) == 1
    # Reintegrate the exact packet segments without using held_integral().
    integral = sum(value * (stop - start) for start, stop, value in zip(
        command_times, command_times[1:] + [10.], targets))
    assert abs(integral - run["transmitted_velocity_integral_rad"]) < 1e-12
    plateau = run["unchanged_encoder_intervals"][0]
    a3 = [r for r in rows if r["method"] == "core" and r["record_type"] == "rx" and r["opcode"] == "A3"]
    p = [r for r in a3 if plateau["start_s"] <= float(r["host_monotonic_s"]) - origin <= plateau["end_s"]]
    counts = [struct.unpack("<i", bytes.fromhex(r["payload_hex"])[3:])[0] for r in p]
    assert len(counts) >= 30 and len(set(counts)) == 1
    travel = 0.
    for start, stop, value in zip(command_times, command_times[1:] + [10.], targets):
        travel += max(0., min(stop, plateau["end_s"]) - max(start, plateau["start_s"])) * value
    assert abs(travel - plateau["commanded_travel_rad"]) < 1e-12
    assert travel * 16384 / (2 * math.pi) > 30
    # Check the ~100-ms A3 averages independently using exported raw positions.
    pt, pq = series["a3_samples"]
    averages = read(HERE / f"{rate}hz_encoder_average_100ms.csv")
    for first, saved in zip(range(0, len(pt) - 5, 5), averages):
        last = first + 5
        measured = (pq[last] - pq[first]) / (pt[last] - pt[first])
        assert abs(measured - float(saved["actual_velocity_rad_s"])) < 1e-12

assert collapsed_codes[50] == collapsed_codes[200]
figures = dict(comparison_full=6, comparison_startup=6, comparison_settling=6,
               settling_encoder_100ms=4, low_speed_count_plateaus=6, replay_50hz=3, replay_200hz=3)
for name, count in figures.items():
    assert (HERE / f"{name}.png").read_bytes().startswith(b"\x89PNG")
    assert (HERE / f"{name}.pdf").read_bytes().startswith(b"%PDF")
    tree = ET.parse(HERE / f"{name}.svg")
    axes = [item for item in tree.iter() if item.get("id", "").startswith("axes_")]
    assert len(axes) == count
info = subprocess.check_output(["pdfinfo", str(HERE / "all_figures.pdf")], text=True)
assert re.search(r"Pages:\s+7\b", info)
for name in ("README.md", "index.html"):
    text = (HERE / name).read_text()
    links = re.findall(r'href="([^"]+)"', text) if name.endswith("html") else re.findall(r"\]\(([^)]+)\)", text)
    for link in links:
        if not link.startswith(("http:", "https:", "#")):
            assert (HERE / link).exists(), link
for source in HERE.glob("*.py"):
    ast.parse(source.read_text(), filename=str(source))
verification = dict(protected_input_files=len(protected), inputs_unchanged=True,
                    independently_checked_exported_native_samples=checked_rows,
                    independent_causal_window_rms=True, identical_collapsed_encoded_commands=True,
                    unchanged_encoder_count_plateaus_verified_from_raw_payloads=True,
                    command_integrals_and_encoder_averages_verified=True,
                    figures=len(figures), combined_pdf_pages=7, figure_axes_counts_verified=True,
                    local_links_exist=True, report_python_parses=True, hardware_or_simulation_run=False)
(HERE / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
print(json.dumps(verification, indent=2))
