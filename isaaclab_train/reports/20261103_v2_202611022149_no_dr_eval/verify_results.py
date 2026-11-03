"""Verify the completed native evaluation, episode capture and input preservation."""

import csv
import hashlib
import json
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUN = ROOT / "isaaclab_train/logs/rsl_rl/trackstand_rl_v2_rsl/2026-11-02_21-49-32"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


summary = json.loads((OUT / "episode_capture_summary.json").read_text())
episodes = list(csv.DictReader((OUT / "completed_episodes.csv").open()))
native = (OUT / "eval.txt").read_text()
run_log = (OUT / "run.log").read_text()
lengths = [int(e["length_steps"]) for e in episodes]
timeout_count = sum(e["timeout"] == "True" for e in episodes)
flags = {
    name: sum(e[name] == "True" for e in episodes)
    for name in summary["termination_counts"]
}
assert summary["policy_steps"] == 3000 and summary["num_envs"] == 64
assert summary["environment_transitions"] == 192000
assert len(episodes) == summary["completed_episodes"]
assert timeout_count == summary["timeouts"]
assert flags == summary["termination_counts"]
assert statistics.mean(lengths) == summary["mean_length_steps"]
assert f"{len(episodes)} completed episodes" in native
assert "Traceback" not in run_log
assert "domain randomization: OFF" in run_log
assert "termination_mode       : pure_survival" not in native
assert (OUT / "torque_profile.png").stat().st_size > 0
assert (OUT / "torque_profile.npz").stat().st_size > 0

before = json.loads((OUT / "original_run_manifest_before.json").read_text())
after = {
    str(p.relative_to(RUN)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
    for p in sorted(RUN.rglob("*")) if p.is_file()
}
(OUT / "original_run_manifest_after.json").write_text(json.dumps(after, indent=2) + "\n")
assert before == after, "Original run files were changed during this task"
source = json.loads((OUT / "preflight_provenance.json").read_text())["source_sha256"]
assert all(sha256(ROOT / name) == expected for name, expected in source.items())
selection = json.loads((OUT / "checkpoint_selection.json").read_text())
assert sha256(OUT / "model_26600.pt") == selection["selected_checkpoint_sha256"]

result = {
    "passed": True,
    "policy_steps": 3000,
    "num_envs": 64,
    "environment_transitions": 192000,
    "completed_episodes": len(episodes),
    "mean_length_steps": statistics.mean(lengths),
    "mean_length_seconds": statistics.mean(lengths) * summary["step_dt"],
    "min_length_steps": min(lengths),
    "max_length_steps": max(lengths),
    "std_length_steps": statistics.pstdev(lengths),
    "timeouts": timeout_count,
    "timeout_rate": timeout_count / len(episodes),
    "non_timeout_endings": len(episodes) - timeout_count,
    "termination_counts": flags,
    "overlapping_termination_episodes": summary["overlapping_termination_episodes"],
    "censored_trajectory_count": summary["censored_trajectory_count"],
    "original_run_files_identical": True,
    "original_run_files_checked": len(before),
    "production_python_sources_identical": True,
    "production_python_sources_checked": len(source),
    "staged_checkpoint_hash_identical": True,
    "native_eval_and_capture_counts_agree": True,
    "evaluation_summary_and_torque_artifacts_present": True,
    "no_dr_confirmed": True,
    "full_task_terminations_confirmed": list(flags),
}
(OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
