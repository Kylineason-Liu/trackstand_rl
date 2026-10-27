"""Plot recorded unloaded constant-target diagnostics; no resimulation."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
with np.load(ROOT / "rear_stability_400/capture.npz") as data:
    samples_400 = data["speed_computed_applied"]
env_id = int(np.argmax(np.ptp(samples_400[len(samples_400)//2:, :, 0], axis=0)))
fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey="row", constrained_layout=True)
for column, hz in enumerate((400, 800)):
    folder = ROOT / f"rear_stability_{hz}"
    summary = json.loads((folder / "summary.json").read_text())
    with np.load(folder / "capture.npz") as data:
        samples = data["speed_computed_applied"][:, env_id]
    time_ms = np.arange(len(samples))*1000/hz
    axes[0, column].plot(time_ms, samples[:, 0], label="Actual rear speed", lw=1.3)
    axes[0, column].axhline(1., color="tab:red", ls="--", label="Held velocity target")
    axes[0, column].axhline(.47, color="tab:green", ls=":", label="1 - 0.106 / 0.2 = 0.47 rad/s")
    axes[0, column].set_title(f"{hz} Hz physics / explicit PD; 50 Hz held target")
    axes[0, column].set_ylim(-.08, 1.78)
    axes[1, column].plot(time_ms, samples[:, 2], label="Applied actuator effort", color="tab:orange", lw=1.3)
    axes[1, column].axhline(.106, color="tab:green", ls=":", label="Configured moving friction 0.106 Nm")
    axes[1, column].set_ylim(-.15, .22)
    axes[1, column].set_xlabel("Time (ms)")
    for row in range(2):
        axis = axes[row, column]
        axis.grid(alpha=.25)
        axis.legend(loc="lower right", fontsize=8)
        axis.set_xlim(0, 200)
axes[0, 0].set_ylabel("Rear velocity (rad/s)")
axes[1, 0].set_ylabel("Motor effort (Nm)")
fig.suptitle(f"V2 unloaded constant-target audit — environment {env_id}; Kp=0, Kd=0.2\n"
             "Seed 42; 64 DR environments; no geometry changes; diagnostic fixture, not a balance evaluation")
for suffix in ("png", "svg"):
    fig.savefig(ROOT / f"rear_velocity_loop_stability.{suffix}", dpi=180)
plt.close(fig)
print(f"Plotted matching environment {env_id} for both integration rates.")
