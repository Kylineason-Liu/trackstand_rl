"""Verify this figure bundle against captures, the old archive, and exported CSVs."""

import csv
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(path for path in HERE.parents if (path / "isaaclab_train").is_dir())
ARCHIVE = ROOT / "isaaclab_train/reports/motor_limits_09221945_91850_2026-10-01/motor_limit_samples.npz"


def main():
    summary = json.loads((HERE / "summary.json").read_text())
    checks = {}
    for name, digest in summary["input_sha256"].items():
        checks[f"hash:{name}"] = hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest
    validation = json.loads((HERE / "sensor_validation.json").read_text())
    checks["sensor_24_samples"] = len(validation) == 24
    checks["sensor_acceleration_agreement"] = max(row["absolute_error_nm"] for row in validation) < 1e-5
    for case, expected in (("positive_feedforward_only", 2.0), ("negative_feedforward_only", -2.0),
                           ("oversized_feedforward", 20.0), ("saturated_implicit_drive", 5.6)):
        checks[f"sensor_expected_effort:{case}"] = all(
            abs(row["measured_joint_effort_nm"] - expected) < 2e-5 for row in validation if row["case"] == case
        )
    with np.load(HERE / "dr/motor_samples.npz") as fresh, np.load(ARCHIVE) as old:
        for key in ("raw_action", "position_after_rad", "velocity_after_rad_s", "done"):
            checks[f"archive_bitwise_identical:{key}"] = np.array_equal(fresh[key], old[key])
    for mode in ("no_dr", "dr"):
        directory = HERE / mode
        checks[f"source_unchanged:{mode}"] = not json.loads((directory / "source_integrity.json").read_text())["changed_files"]
        checks[f"saved_configuration:{mode}"] = all(json.loads((directory / "configuration_checks.json").read_text()).values())
        with np.load(directory / "motor_samples.npz") as data:
            metadata = json.loads((directory / "experiment.json").read_text())
            checks[f"shape:{mode}"] = data["projected_joint_effort_nm"].shape == (3000, 8, 64, 2)
            checks[f"checkpoint:{mode}"] = hashlib.sha256((directory / "model_91850.pt").read_bytes()).hexdigest() == metadata["checkpoint_sha256"]
            starts = np.zeros((3000, 64), dtype=bool)
            starts[0] = True
            starts[1:] = data["done"][:-1]
            mask = np.repeat(starts[:, None], 8, axis=1)
            checks[f"rear_excess_confined_to_episode_start:{mode}"] = not (
                np.abs(data["projected_joint_effort_nm"][~mask, 1]) > 6.29
            ).any()
            csv_path = HERE / f"{mode}_env0_traces.csv"
            with csv_path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            checks[f"csv_length:{mode}"] = len(rows) == 24000
            checks[f"csv_timebase:{mode}"] = bool(np.allclose(
                [float(row["command_time_s"]) for row in rows], np.arange(24000) * 0.0025, rtol=0, atol=1e-10
            )) and abs(float(rows[-1]["response_time_s"]) - 60.0) < 1e-10
            checks[f"csv_matches_sensor:{mode}"] = bool(np.allclose(
                [float(row["solver_measured_rear_joint_effort_nm"]) for row in rows],
                data["projected_joint_effort_nm"][:, :, 0, 1].reshape(-1), rtol=1e-6, atol=1e-7
            ))
    for name in ("isaac_demand_vs_actuation", "isaac_startup_demand_vs_actuation", "isaac_tracking_detail",
                 "isaac_peak_startup_demand_vs_actuation"):
        png = (HERE / f"{name}.png").read_bytes()
        checks[f"png:{name}"] = png[:8] == b"\x89PNG\r\n\x1a\n" and min(struct.unpack(">II", png[16:24])) > 1000
        checks[f"pdf:{name}"] = (HERE / f"{name}.pdf").read_bytes().startswith(b"%PDF")
        checks[f"svg:{name}"] = "<svg" in (HERE / f"{name}.svg").read_text()
    result = {"passed": all(checks.values()), "checks": checks}
    (HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise AssertionError("Figure-bundle verification failed")


if __name__ == "__main__":
    main()
