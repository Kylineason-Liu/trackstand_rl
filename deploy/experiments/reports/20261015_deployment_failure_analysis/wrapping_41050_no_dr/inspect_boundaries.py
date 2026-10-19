"""Locate first causal encoding differences in the saved boundary stress tests."""

import csv
import json
from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
OUT = parser.parse_args().results.resolve()
summary = json.loads((OUT / "summary.json").read_text())
rear_index = summary["rear_qpos_index"]
rows = []
for phase in (2.8, 3.5, -3.5):
    base = dict(np.load(OUT / f"phase_{phase:+g}_wrapped_pi.npz"))
    for mode in ("native_unwrapped", "candidate_4pi"):
        other = dict(np.load(OUT / f"phase_{phase:+g}_{mode}.npz"))
        n = min(len(base["valid"]), len(other["valid"]))
        for i in range(16):
            common = base["valid"][:n, i] & other["valid"][:n, i]
            different = np.any(base["observation"][:n, i] != other["observation"][:n, i], axis=1)
            hits = np.flatnonzero(common & different)
            if not len(hits):
                raise AssertionError(f"No encoding difference: phase={phase}, {mode}, seed={42 + i}")
            t = int(hits[0])
            channels = [j for j in range(13) if j != 8]
            physical_same = bool(np.array_equal(base["state_before"][t, i], other["state_before"][t, i]))
            others_same = bool(np.array_equal(base["observation"][t, i, channels], other["observation"][t, i, channels]))
            prefix_same = bool(np.array_equal(base["raw_action"][:t, i], other["raw_action"][:t, i])
                               and np.array_equal(base["state_before"][:t, i], other["state_before"][:t, i]))
            assert physical_same and others_same and prefix_same
            rows.append({"initial_phase_rad": phase, "alternative": mode, "seed": 42 + i,
                         "first_different_tick": t, "first_different_time_s": t * .02,
                         "native_rear_position_rad": base["state_before"][t, i, rear_index],
                         "wrapped_observation_rad": base["observation"][t, i, 8],
                         "alternative_observation_rad": other["observation"][t, i, 8],
                         "wrapped_raw_steer": base["raw_action"][t, i, 0],
                         "alternative_raw_steer": other["raw_action"][t, i, 0],
                         "wrapped_raw_rear": base["raw_action"][t, i, 1],
                         "alternative_raw_rear": other["raw_action"][t, i, 1],
                         "physical_state_identical": physical_same, "other_12_channels_identical": others_same,
                         "preceding_actions_and_states_identical": prefix_same})
with (OUT / "first_encoding_differences.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

sample = next(r for r in rows if r["initial_phase_rad"] == 2.8 and r["seed"] == 42
              and r["alternative"] == "native_unwrapped")
crossing = sample["first_different_time_s"]
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
for mode, name in (("wrapped_pi", "Current ±π"), ("native_unwrapped", "No wrapping"), ("candidate_4pi", "4π candidate")):
    data = dict(np.load(OUT / ("phase_+2.8_" + mode + ".npz")))
    valid = data["valid"][:, 0]
    t = np.arange(len(valid))[valid] * .02
    window = (t >= crossing - .25) & (t <= crossing + .75)
    axes[0].plot(t[window], data["state_before"][valid, 0, rear_index][window], label=name)
    axes[1].step(t[window], data["observation"][valid, 0, 8][window], where="post", label=name)
    axes[2].step(t[window], data["raw_action"][valid, 0, 1][window] * 6.28, where="post", label=name)
    axes[3].plot(t[window] + .02, np.rad2deg(data["roll_after_rad"][valid, 0][window]), label=name)
for ax in axes[:2]:
    ax.axhline(-np.pi, color="gray", ls=":"); ax.axhline(np.pi, color="gray", ls=":")
for ax in axes:
    ax.axvline(crossing, color="black", ls="--", lw=.8)
    ax.set_xlim(crossing - .25, crossing + .75); ax.grid(alpha=.2); ax.legend(fontsize=8)
for ax, label in zip(axes, ("Native rear angle (rad)", "Supplied rear observation (rad)",
                            "Rear policy demand (N m)", "Actual roll (deg)")):
    ax.set_ylabel(label)
axes[-1].set_xlabel("First-episode simulation time (s); roll timestamp is after its physics step")
fig.suptitle(f"41050 no DR, seed 42, fixed initial rear phase +2.8 rad\n"
             f"First difference at {crossing:.2f}s: physical state and other 12 observations still identical")
fig.tight_layout()
for suffix in ("png", "svg", "pdf"):
    fig.savefig(OUT / ("boundary_crossing_detail." + suffix), dpi=150)
plt.close(fig)
print(json.dumps({"verified_first_difference_rows": len(rows), "example": sample}, indent=2, default=float))
