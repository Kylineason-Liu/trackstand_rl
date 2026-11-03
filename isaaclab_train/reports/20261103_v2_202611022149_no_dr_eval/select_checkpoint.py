"""Select the latest saved checkpoint tied for best raw TensorBoard length."""

import csv
import hashlib
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "isaaclab_train/logs/rsl_rl/trackstand_rl_v2_rsl/2026-11-02_21-49-32"
OUT = Path(__file__).resolve().parent
TAG = "Train/mean_episode_length"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


acc = EventAccumulator(str(RUN), size_guidance={"scalars": 0, "tensors": 0})
acc.Reload()
rows = acc.Scalars(TAG)
checkpoints = {int(p.stem.split("_")[1]): p for p in RUN.glob("model_*.pt")}
eligible = [r for r in rows if r.step in checkpoints]
maximum = max(r.value for r in eligible)
ties = [r for r in eligible if r.value == maximum]
selected = max(ties, key=lambda r: (r.step, r.wall_time))
global_maximum = max(r.value for r in rows)
global_ties = [r.step for r in rows if r.value == global_maximum]
with (OUT / "raw_mean_episode_length.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["iteration", "raw_mean_episode_length_steps", "wall_time", "checkpoint_exists"])
    writer.writerows((r.step, r.value, r.wall_time, int(r.step in checkpoints)) for r in rows)
result = {
    "run": str(RUN),
    "criterion_tag": TAG,
    "criterion": "exact raw scalar, eligible saved checkpoint iterations; latest iteration wins exact ties",
    "scalar_reservoir_size": 0,
    "raw_event_count": len(rows),
    "raw_first_iteration": rows[0].step,
    "raw_last_iteration": rows[-1].step,
    "checkpoint_count": len(checkpoints),
    "checkpoint_first_iteration": min(checkpoints),
    "checkpoint_last_iteration": max(checkpoints),
    "eligible_raw_event_count": len(eligible),
    "eligible_raw_maximum": maximum,
    "eligible_exact_maximum_tie_count": len(ties),
    "eligible_exact_maximum_tie_iterations": [r.step for r in ties],
    "selected_iteration": selected.step,
    "selected_checkpoint": str(checkpoints[selected.step]),
    "selected_checkpoint_sha256": sha256(checkpoints[selected.step]),
    "global_raw_maximum": global_maximum,
    "global_exact_maximum_tie_count": len(global_ties),
    "global_exact_maximum_first_iteration": min(global_ties),
    "global_exact_maximum_last_iteration": max(global_ties),
    "global_maximum_has_available_checkpoint": any(i in checkpoints for i in global_ties),
    "time_indexed_length_tag_not_used": "Train/mean_episode_length/time",
    "smoothing_applied": False,
}
(OUT / "checkpoint_selection.json").write_text(json.dumps(result, indent=2) + "\n")
inputs = {
    str(p.relative_to(RUN)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
    for p in sorted(RUN.rglob("*")) if p.is_file()
}
(OUT / "original_run_manifest_before.json").write_text(json.dumps(inputs, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "eligible_exact_maximum_tie_iterations"}, indent=2))
print("Latest 20 tied checkpoint iterations:", result["eligible_exact_maximum_tie_iterations"][-20:])
