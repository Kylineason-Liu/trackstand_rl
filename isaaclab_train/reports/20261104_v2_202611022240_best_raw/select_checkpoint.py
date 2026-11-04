"""Read-only raw TensorBoard selection for completed v2 DR run 202611022240."""

import csv
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "isaaclab_train/logs/rsl_rl/trackstand_rl_v2_rsl/2026-11-02_22-40-01"
TAG = "Train/mean_episode_length"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def stat_manifest():
    return {
        str(path.relative_to(RUN)): {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in sorted(RUN.rglob("*")) if path.is_file()
    }


OUT.mkdir(exist_ok=True, parents=True)
before = stat_manifest()
events = sorted(RUN.glob("events.out.tfevents.*"))
assert events, "No TensorBoard event files"
protected = [*events, RUN / "params/env.yaml", RUN / "params/agent.yaml"]
before_hashes = {str(path.relative_to(RUN)): digest(path) for path in protected}
env = yaml.load((RUN / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
agent = yaml.load((RUN / "params/agent.yaml").read_text(), Loader=yaml.BaseLoader)
policy_dt = float(env["sim"]["dt"]) * int(env["decimation"])
assert env["domain_randomization"] == "true"
assert env["actions"]["drive"]["class_type"].endswith(":IntegralJointVelocityAction")
assert agent["experiment_name"] == "trackstand_rl_v2_rsl"
assert float(env["episode_length_s"]) == 30 and policy_dt == .02

accumulator = EventAccumulator(str(RUN), size_guidance={"scalars": 0, "tensors": 0}, purge_orphaned_data=False)
accumulator.Reload()
scalars = accumulator.Scalars(TAG)
direct = []
session_logs = Counter()
for event_file in events:
    for event in LegacyEventFileLoader(str(event_file)).Load():
        if event.HasField("session_log"):
            session_logs[str(event.session_log.status)] += 1
        if event.HasField("summary"):
            for value in event.summary.value:
                if value.tag == TAG:
                    assert value.HasField("simple_value"), "Unexpected tensor-only target tag"
                    direct.append((int(event.step), float(value.simple_value), float(event.wall_time)))
assert [(x.step, x.value, x.wall_time) for x in scalars] == direct, "Two raw parsers disagree"
assert direct
step_counts = Counter(row[0] for row in direct)
by_step = defaultdict(list)
for step, value, wall_time in direct:
    by_step[step].append((value, wall_time))
conflicts = {str(step): records for step, records in by_step.items() if len({x[0] for x in records}) > 1}
assert not conflicts, "Conflicting duplicate scalar iterations require explicit restart resolution"
values = {step: max(records, key=lambda x: x[1])[0] for step, records in by_step.items()}
checkpoints = {int(path.stem.split("_")[1]): path for path in RUN.glob("model_*.pt")}
eligible = {step: value for step, value in values.items() if step in checkpoints}
assert eligible
peak = max(eligible.values())
ties = sorted(step for step, value in eligible.items() if value == peak)
selected_iteration = ties[-1]
selected_path = checkpoints[selected_iteration]
selected_hash = digest(selected_path)
global_peak = max(values.values())
global_ties = sorted(step for step, value in values.items() if value == global_peak)
ranked = sorted(eligible, key=lambda step: (eligible[step], step), reverse=True)

with (OUT / "raw_mean_episode_length.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["iteration", "raw_mean_episode_length_steps", "wall_time", "checkpoint_exists"])
    writer.writerows((step, value, stamp, int(step in checkpoints)) for step, value, stamp in direct)
with (OUT / "saved_checkpoint_ranking.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["rank", "iteration", "raw_mean_episode_length_steps", "seconds", "checkpoint"])
    writer.writerows((i + 1, step, eligible[step], eligible[step] * policy_dt, checkpoints[step].name)
                     for i, step in enumerate(ranked))

import torch

torch.set_num_threads(1)
checkpoint = torch.load(selected_path, map_location="cpu", weights_only=True)
assert checkpoint["iter"] == selected_iteration
actor_state = checkpoint["actor_state_dict"]
assert actor_state["distribution._beta_policy_version"].item() == 1
assert list(actor_state["mlp.0.weight"].shape) == [32, 11]
inspection = {"checkpoint": str(selected_path), "sha256": selected_hash,
              "internal_iteration": checkpoint["iter"], "keys": list(checkpoint),
              "actor_tensor_shapes": {key: list(value.shape) for key, value in actor_state.items()}}
(OUT / "checkpoint_inspection.json").write_text(json.dumps(inspection, indent=2) + "\n")

result = {
    "task_id": "best_raw_analysis", "iteration_id": "i1", "run": str(RUN),
    "criterion_tag": TAG, "smoothing_applied": False, "scalar_reservoir_size": 0,
    "purge_orphaned_data": False, "direct_event_parser_agrees": True,
    "event_files": [path.name for path in events], "raw_event_count": len(direct),
    "raw_first_iteration": min(values), "raw_last_iteration": max(values),
    "duplicate_iterations": {str(step): count for step, count in step_counts.items() if count > 1},
    "conflicting_duplicate_iterations": conflicts,
    "target_step_regressions": sum(b[0] < a[0] for a, b in zip(direct, direct[1:])),
    "session_log_status_counts": dict(session_logs),
    "checkpoint_count": len(checkpoints), "checkpoint_first_iteration": min(checkpoints),
    "checkpoint_last_iteration": max(checkpoints), "eligible_checkpoint_count": len(eligible),
    "checkpoints_without_exact_scalar": sorted(set(checkpoints) - set(values)),
    "eligible_raw_maximum_steps": peak, "eligible_raw_maximum_seconds": peak * policy_dt,
    "eligible_exact_maximum_tie_count": len(ties), "eligible_exact_maximum_tie_iterations": ties,
    "selected_iteration": selected_iteration, "selected_checkpoint": str(selected_path),
    "selected_checkpoint_sha256": selected_hash, "selected_checkpoint_internal_iteration": checkpoint["iter"],
    "global_raw_maximum_steps": global_peak, "global_raw_maximum_seconds": global_peak * policy_dt,
    "global_exact_maximum_tie_count": len(global_ties), "global_exact_maximum_first_iteration": global_ties[0],
    "global_exact_maximum_last_iteration": global_ties[-1],
    "global_peak_saved_iterations": [step for step in global_ties if step in checkpoints],
    "latest_checkpoint_raw_mean_steps": values.get(max(checkpoints)),
    "top_10_saved": [{"iteration": step, "raw_mean_episode_length_steps": eligible[step],
                      "seconds": eligible[step] * policy_dt} for step in ranked[:10]],
    "policy_dt_s": policy_dt, "domain_randomization": True, "task": "Trackstand-Balance-v2",
    "no_new_evaluation_or_training": True,
}
(OUT / "checkpoint_selection.json").write_text(json.dumps(result, indent=2) + "\n")
after = stat_manifest()
after_hashes = {str(path.relative_to(RUN)): digest(path) for path in protected}
assert before == after, "Original run file list/stat changed during selection"
assert before_hashes == after_hashes, "Original event/config content changed during selection"
assert selected_hash == digest(selected_path)
preservation = {"passed": True, "original_run_file_stats_identical": len(before),
                "before": before, "after": after,
                "event_and_saved_config_sha256_before": before_hashes,
                "event_and_saved_config_sha256_after": after_hashes,
                "selected_checkpoint_sha256_before_and_after": selected_hash,
                "scope": "All run file sizes/mtimes/list; event/config and selected-checkpoint content hashes"}
(OUT / "preservation.json").write_text(json.dumps(preservation, indent=2) + "\n")
provenance = {"repository_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "repository_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
              "python": platform.python_version(), "torch": torch.__version__,
              "tensorboard": importlib.metadata.version("tensorboard"),
              "saved_configuration": {"domain_randomization": True, "experiment_name": agent["experiment_name"],
                  "seed": int(agent["seed"]), "policy_hz": 1 / policy_dt,
                  "episode_length_s": float(env["episode_length_s"]), "max_iterations": int(agent["max_iterations"]),
                  "save_interval": int(agent["save_interval"]), "saved_historical_log_dir": env["log_dir"],
                  "drive_action_class": env["actions"]["drive"]["class_type"],
                  "distribution": agent["actor"]["distribution_cfg"]["class_name"]},
              "device": "CPU only; CUDA_VISIBLE_DEVICES=''", "gpu_or_simulator_started": False}
(OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
print(json.dumps({key: value for key, value in result.items()
                  if key not in ("eligible_exact_maximum_tie_iterations", "global_peak_saved_iterations")}, indent=2))
