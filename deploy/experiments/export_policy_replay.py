#!/usr/bin/env python3
"""Export actual model_2500 episode commands at both levels; run on this PC, not the Pi."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "isaaclab_train/reports/20261025_2500_dr_comparison"
COLUMNS = ("sample_index", "source_time_s", "episode", "raw_steering", "raw_drive",
           "scaled_steering_rad", "conditioned_steering_rad", "scaled_drive_nm", "submitted_drive_nm")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "replay_targets")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(model="2500", training_run="2026-10-21_11-18-29", dr=True, seed=42,
                    environment=0, episode=1, policy_hz=50, units="position rad; torque N m",
                    steering_levels="scaled raw action versus held position target BEFORE PD and passive losses",
                    files={})
    for tag in ("isaac400", "mujoco400", "mujoco800"):
        folder = SOURCE / tag
        native = folder / "motor_samples.npz" if tag == "isaac400" else next(folder.glob("*_evaluation.npz"))
        physics = native if tag == "isaac400" else folder / "physics_samples.npz"
        with np.load(native) as capture, np.load(physics) as steps:
            done = capture["done"] if tag == "isaac400" else capture["termination_flags"].any(-1)
            stop = int(np.flatnonzero(done[:, 0])[0]) + 1
            actions = capture["raw_action"][:stop, 0]
            scaled = actions * np.array([1.57, 6.28], dtype=np.float32)
            held = steps["held_steering_target_rad"][:stop, 0, 0]
            rear = steps["submitted_motor_torque_nm"][:stop, 0, 0, 1]
            path = args.output_dir / f"2500_{tag}_env0_episode1.csv"
            with path.open("x", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(COLUMNS)
                for i in range(stop):
                    writer.writerow([i, i * .02, 1, *map(float, actions[i]),
                                     float(scaled[i, 0]), float(held[i]), float(scaled[i, 1]), float(rear[i])])
        manifest["files"][path.name] = dict(
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), samples=stop, duration_s=stop * .02,
            source_files={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in set((native, physics))},
            maximum_scaled_vs_conditioned_steering_difference_rad=float(np.max(abs(scaled[:, 0] - held))))
        print(path, stop, "real samples")
    with (args.output_dir / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
