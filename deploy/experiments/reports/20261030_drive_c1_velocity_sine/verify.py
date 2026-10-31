"""Independently check plotted samples, timing, harmonic integration and artifacts."""

import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
stats = json.loads((HERE / "statistics.json").read_text())
protected = json.loads((HERE / "protected_input_hashes.json").read_text())
assert len(protected) == 12
for name, expected in protected.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name

spec = importlib.util.spec_from_file_location("c1_report_analysis", HERE / "analyze.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
maximum_phase_error_ms = maximum_amplitude_error = 0.
checked_rows = 0
for run in stats["runs"]:
    m = run["metrics"]
    f = m["frequency_hz"]
    prefix = "c1_" + f"{f:g}".replace(".", "p") + "hz"
    with (ROOT / "deploy/experiments/logs" / m["source_csv"]).open(newline="") as stream:
        original = list(csv.DictReader(stream))
    with (HERE / f"{prefix}_samples.csv").open(newline="") as stream:
        plotted = list(csv.DictReader(stream))
    series_arrays = {}
    for name, kind, opcode, field in (("command", "command", "C1", "encoded_target_velocity_rad_s"),
                                      ("velocity", "rx", "A2", "velocity_rad_s"),
                                      ("torque", "rx", "A1", "current_torque_nm")):
        source_rows = [row for row in original if row["record_type"] == kind and row["opcode"] == opcode
                       and 0 <= float(row["host_monotonic_s"]) - m["phase_origin_monotonic_s"] < 10]
        samples = [row for row in plotted if row["series"] == name]
        assert len(source_rows) == len(samples)
        times = []
        values = []
        for source, sample in zip(source_rows, samples):
            t = float(source["host_monotonic_s"]) - m["phase_origin_monotonic_s"]
            assert t == float(sample["phase_time_s"])
            assert float(source[field]) == float(sample["value"])
            times.append(t)
            values.append(float(sample["value"]))
        series_arrays[name] = np.column_stack((times, values))
        checked_rows += len(samples)
    command = series_arrays["command"]
    # Dense midpoint quadrature is independent of the exact segment integration.
    t = 2 + (np.arange(400000) + .5) * .00002
    y = command[np.searchsorted(command[:, 0], t, side="right") - 1, 1]
    sine = 2 * np.mean(y * np.sin(2 * np.pi * f * t))
    cosine = 2 * np.mean(y * np.cos(2 * np.pi * f * t))
    phase = np.arctan2(cosine, sine)
    exact = run["input_harmonic"]
    phase_error = abs((phase - exact["phase_rad"] + np.pi) % (2 * np.pi) - np.pi) / (2 * np.pi * f) * 1000
    amplitude_error = abs(np.hypot(sine, cosine) - exact["amplitude"])
    assert phase_error < .02 and amplitude_error < .0005
    maximum_phase_error_ms = max(maximum_phase_error_ms, phase_error)
    maximum_amplitude_error = max(maximum_amplitude_error, amplitude_error)
    velocity = series_arrays["velocity"]
    mask = velocity[:, 0] >= 2
    target = command[np.searchsorted(command[:, 0], velocity[mask, 0], side="right") - 1, 1]
    rms = np.sqrt(np.mean((velocity[mask, 1] - target) ** 2))
    assert abs(rms - m["steady_error_rms_rad_s"]) < 1e-12
    assert np.max(abs(velocity[:, 1])) < 5.5
    assert np.max(abs(series_arrays["torque"][:, 1])) < .55
    for suffix in ("target_response", "zoom"):
        stem = HERE / f"{prefix}_{suffix}"
        assert stem.with_suffix(".png").read_bytes().startswith(b"\x89PNG")
        assert stem.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
        tree = ET.parse(stem.with_suffix(".svg"))
        axes = [element for element in tree.iter() if element.get("id", "").startswith("axes_")]
        assert len(axes) == 2
for filename in ("all_runs.pdf", "all_run_zooms.pdf"):
    info = subprocess.check_output(["pdfinfo", str(HERE / filename)], text=True)
    assert re.search(r"Pages:\s+5\b", info)
for filename in ("README.md", "index.html"):
    text = (HERE / filename).read_text()
    links = re.findall(r'href="([^"]+)"', text) if filename.endswith("html") else re.findall(r"\]\(([^)]+)\)", text)
    for link in links:
        if not link.startswith(("http:", "https:", "#")):
            assert (HERE / link).exists(), link
for path in HERE.glob("*.py"):
    ast.parse(path.read_text(), filename=str(path))
verification_path = HERE / "verification.json"
verification = json.loads(verification_path.read_text())
assert checked_rows == verification["plotted_native_samples"]
verification.update(independently_checked_native_csv_rows=checked_rows,
                    harmonic_quadrature_max_phase_error_ms=maximum_phase_error_ms,
                    harmonic_quadrature_max_amplitude_error_rad_s=maximum_amplitude_error,
                    independent_rms_checks=True, full_and_zoom_figures_each_two_axes=True,
                    full_and_zoom_pdf_pages_each=5, local_links_exist=True, report_python_parses=True)
verification_path.write_text(json.dumps(verification, indent=2) + "\n")
print(json.dumps(verification, indent=2))
