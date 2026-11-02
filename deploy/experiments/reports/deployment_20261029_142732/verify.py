"""Check archived analysis, figures, original input hashes and MCAP integrity."""

from collections import Counter
import ast
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
stats = json.loads((HERE / "statistics.json").read_text())
with gzip.open(HERE / "decoded_details.json.gz", "rt") as stream:
    decoded = json.load(stream)
arrays = np.load(HERE / "decoded_data.npz")
validation = np.load(HERE / "validated_analysis.npz")
protected = json.loads((HERE / "protected_input_hashes.json").read_text())
changes = [name for name, digest in protected.items()
           if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
assert not changes, changes
assert sum(decoded["counts"].values()) == 290281
assert not decoded["issues"]
assert len(stats["episodes"]) == 21
assert sum(ep["n_actions"] for ep in stats["episodes"]) == 1101
assert Counter(ep["reason"] for ep in stats["episodes"]) == {
    "roll_or_inversion_safety": 17, "motor_fault": 2, "motor_state_timeout": 1, "RC_down": 1,
}
assert np.array_equal(validation["model_action"], arrays["raw_action"][:, 1:])
assert np.array_equal(np.sum(~np.isfinite(arrays["motor_state"][:, 6:8]), axis=0), [11, 12])
assert np.all(abs(validation["model_action"]) <= 1)
assert np.array_equal(np.unique(validation["episode"]), np.arange(1, 22))
assert np.max(abs(validation["steering_slewed_rad"]
                  - validation["enabled_motor_command"][:, 2])) < 1e-10
assert np.all(arrays["imu"][validation["imu_input_index"], 0] <= arrays["observation"][:, 0])
assert np.all(arrays["motor_state"][validation["motor_input_index"], 0] <= arrays["observation"][:, 0])
simcheck = np.load(HERE / "saved_simulation_actor_check.npz")
assert len(simcheck["flat_policy_sample_index"]) == 2048
assert np.max(abs(simcheck["onnx_action"] - simcheck["recorded_isaac_action"])) < 1e-5

figure_list = json.loads((HERE / "figure_manifest.json").read_text())
assert len(figure_list) == 28
for fig in figure_list:
    for extension, magic in (("png", b"\x89PNG"), ("pdf", b"%PDF")):
        path = HERE / (fig["stem"] + "." + extension)
        assert path.read_bytes().startswith(magic), path
    assert "<svg" in (HERE / (fig["stem"] + ".svg")).read_text()[:2000]
assert len(list((HERE / "attempts").glob("*.png"))) == 21
pdfinfo = subprocess.check_output(["pdfinfo", str(HERE / "all_attempts.pdf")], text=True)
assert re.search(r"Pages:\s+21\b", pdfinfo)
for path in HERE.glob("*.py"):
    ast.parse(path.read_text(), filename=str(path))
for name in ("index.html", "README.md"):
    text = (HERE / name).read_text()
    links = re.findall(r'href="([^"]+)"', text) if name.endswith("html") else re.findall(r"\]\(([^)]+)\)", text)
    for link in links:
        if link != "verification.json" and not link.startswith(("http:", "https:", "#")):
            assert (HERE / link).exists(), link

bag_path = ROOT / decoded["bag"]
if not bag_path.exists():
    bag_path = ROOT / "deploy/experiments/logs/rosbag2_2026_10_29-14_27_32/rosbag2_2026_10_29-14_27_32_0.mcap"
metadata = yaml.safe_load((bag_path.parent / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
metadata_counts = {row["topic_metadata"]["name"]: row["message_count"]
                   for row in metadata["topics_with_message_count"] if row["message_count"]}
assert metadata_counts == decoded["counts"]
assert hashlib.sha256(bag_path.read_bytes()).hexdigest() == decoded["sha256"]

mcap_integrity = {"original_decoder_crc_validation": True, "indexed_reader_check": "not installed"}
try:
    from mcap.reader import make_reader
except ImportError:
    pass
else:
    with bag_path.open("rb") as stream:
        reader = make_reader(stream, validate_crcs=True)
        summary = reader.get_summary()
        indexed_counts = Counter(channel.topic for _, channel, _ in reader.iter_messages(log_time_order=False))
        assert dict(indexed_counts) == decoded["counts"]
        assert summary.statistics.message_count == 290281
        mcap_integrity.update(indexed_reader_check="all indexed messages read with CRC validation",
                              chunks=summary.statistics.chunk_count, messages=sum(indexed_counts.values()))

result = dict(passed=True, protected_inputs_checked=len(protected), changed_protected_inputs=changes,
              messages=290281, episodes=21, policy_contract_samples=1101,
              saved_simulation_actor_check_samples=2048, figures=28, individual_attempts=21,
              pdf_pages=21, local_links_exist=True, report_python_parses=True,
              missing_torque_values_not_filled=True, integrity=mcap_integrity,
              output_scope="This report directory only; no deployment/training/model/bag/ledger changes")
(HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
