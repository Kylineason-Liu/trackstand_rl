"""Independent pure-Python checks of raw records, statistics, exports and figures."""

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


def integrate(times, values, end, start=0.):
    return sum(max(0., min(b, end) - max(a, start)) * v
               for a, b, v in zip(times, times[1:] + [10.], values))


native = read(ROOT / "deploy/experiments/replay_targets/2500_isaac_velocity_startup_400hz.csv")
native_times = [float(r["time_s"]) for r in native]
native_velocity = [float(r["velocity_rad_s"]) for r in native]
checked_rows = 0
for rate in (50, 200, 400):
    run = statistics["runs"][str(rate)]
    source = ROOT / "deploy/experiments/logs" / run["source_csv"]
    raw = read(source)
    origin = run["core_phase"]["start_monotonic_s"]
    series = {}
    for kind, opcode, key, output in (("command", "C1", "encoded_target_velocity_rad_s", "commands"),
                                      ("rx", "A2", "velocity_rad_s", "a2"),
                                      ("rx", "A1", "current_torque_nm", "a1"),
                                      ("rx", "A3", "position_rad", "a3")):
        selected = [r for r in raw if r["method"] == "core" and r["record_type"] == kind and r["opcode"] == opcode]
        exported = read(HERE / f"{rate}hz_{output}.csv")
        assert len(selected) == len(exported)
        t, y = [], []
        for row, saved in zip(selected, exported):
            time = float(row["host_monotonic_s"]) - origin
            value = float(row[key]) - (run["core_phase"]["initial_position_rad"] if output == "a3" else 0.)
            output_key = {"commands": "velocity_rad_s", "a2": "velocity_rad_s", "a1": "torque_nm",
                          "a3": "position_rad"}[output]
            assert float(saved["time_s"]) == time and float(saved[output_key]) == value
            t.append(time)
            y.append(value)
        series[output] = t, y
        checked_rows += len(exported)
    ct, cv = series["commands"]
    st, speed = series["a2"]
    for metrics in run["windows"].values():
        errors, native_errors = [], []
        for t, v in zip(st, speed):
            if max(metrics["lower_s"], ct[0]) <= t < metrics["upper_s"]:
                errors.append(v - cv[bisect_right(ct, t)-1])
                i = min(len(native_times)-2, max(0, bisect_right(native_times, t)-1))
                fraction = (t-native_times[i])/(native_times[i+1]-native_times[i])
                interpolated = native_velocity[i] + fraction*(native_velocity[i+1]-native_velocity[i])
                native_errors.append(v-interpolated)
        assert len(errors) == metrics["samples"]
        assert abs(math.sqrt(sum(e*e for e in errors)/len(errors))-metrics["error_rms_rad_s"]) < 1e-12
        assert abs(math.sqrt(sum(e*e for e in native_errors)/len(native_errors))
                   - metrics["error_rms_vs_native_sim_rad_s"]) < 1e-12
    assert abs(integrate(ct, cv, 10.) - run["transmitted_velocity_integral_rad"]) < 1e-12
    pt, position = series["a3"]
    error = position[-1] - integrate(ct, cv, pt[-1])
    assert abs(error-run["final_displacement_minus_target_integral_rad"]) < 1e-12
    for plateau in run["unchanged_encoder_intervals"]:
        a, b = plateau["start_s"], plateau["end_s"]
        a3 = [r for r in raw if r["method"] == "core" and r["record_type"] == "rx" and r["opcode"] == "A3"
              and a <= float(r["host_monotonic_s"])-origin <= b]
        counts = {struct.unpack("<i", bytes.fromhex(r["payload_hex"])[3:7])[0] for r in a3}
        assert len(counts) == 1 and len(a3) >= 10
        assert abs(integrate(ct, cv, b, a)-plateau["commanded_travel_rad"]) < 1e-12
    for i, row in enumerate(read(HERE / f"{rate}hz_encoder_average_100ms.csv")):
        a, b = i*5, i*5+5
        actual = (position[b]-position[a])/(pt[b]-pt[a])
        target = integrate(ct, cv, pt[b], pt[a])/(pt[b]-pt[a])
        assert abs(actual-float(row["actual_velocity_rad_s"])) < 1e-12
        assert abs(target-float(row["target_velocity_rad_s"])) < 1e-12
    if rate == 400:
        command_rows = [r for r in raw if r["method"] == "core" and r["record_type"] == "command"]
        indices = [round(float(r["target_sample_time_s"])*400) for r in command_rows]
        assert len(indices) == len(set(indices)) == 3994
        missing = [i/400 for i in range(4000) if i not in set(indices)]
        assert missing == run["missing_reference_times_s"]
        assert all(t < 2 for t in missing) and run["skipped_slots"] == 6
        assert max(float(r["lateness_s"])*1000 for r in command_rows) == run["actual_logged_lateness_ms"]["max"]
        for row in command_rows:
            payload = bytes.fromhex(row["payload_hex"])
            code = struct.unpack("<i", payload[1:])[0]
            source_speed = native_velocity[round(float(row["target_sample_time_s"])*400)]
            assert code == round(-source_speed*6000/(2*math.pi))

figures = dict(replay_400hz=3, replay_400hz_startup=3, replay_400hz_settling=3,
               comparison_full=9, comparison_startup=9, comparison_settling=9,
               settling_encoder_100ms=6, low_speed_count_plateaus=9, **{"400hz_timing": 3})
for name, axes in figures.items():
    assert (HERE/f"{name}.png").read_bytes().startswith(b"\x89PNG")
    assert (HERE/f"{name}.pdf").read_bytes().startswith(b"%PDF")
    tree = ET.parse(HERE/f"{name}.svg")
    assert len([e for e in tree.iter() if e.get("id", "").startswith("axes_")]) == axes
assert re.search(r"Pages:\s+9\b", subprocess.check_output(["pdfinfo", str(HERE/"all_figures.pdf")], text=True))
for filename in ("README.md", "index.html"):
    text = (HERE/filename).read_text()
    links = re.findall(r'href="([^"]+)"', text) if filename.endswith("html") else re.findall(r"\]\(([^)]+)\)", text)
    for link in links:
        if not link.startswith(("https:", "http:", "#")):
            assert link == "verification.json" or (HERE/link).exists(), link
for source in HERE.glob("*.py"):
    ast.parse(source.read_text(), filename=str(source))
verification = dict(
    protected_input_files=len(protected), original_files_unchanged=True,
    independently_verified_export_rows=checked_rows, causal_command_rms_and_native_curve_rms=True,
    command_integrals_encoder_averages_and_raw_count_plateaus=True,
    all_3994_c1_payloads_match_native_reference=True, six_missing_slots_independently_verified=True,
    figure_files_and_axes_checked=True, figures=9, combined_pdf_pages=9,
    report_links_and_python_syntax_checked=True, hardware_or_simulation_run=False,
)
(HERE/"verification.json").write_text(json.dumps(verification, indent=2)+"\n")
print(json.dumps(verification, indent=2))
