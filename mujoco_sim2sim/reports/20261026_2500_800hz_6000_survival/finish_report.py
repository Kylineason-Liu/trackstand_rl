"""Finish reporting for the completed run after correcting a directory-glob bug."""

import json
from pathlib import Path

import evaluate


def main():
    report = Path(__file__).resolve().parent
    output = report / "run_6000"
    experiment = json.loads((output / "experiment.json").read_text())
    protected = experiment["protected_inputs_sha256"].copy()
    script = report / "evaluate.py"
    snapshot = report / "evaluate_as_run.py"
    original_hash = protected.pop(str(script.relative_to(evaluate.ROOT)))
    assert evaluate.sha256(snapshot) == original_hash
    # The only correction excludes worker_XX.log files from the worker-directory
    # listing. Verify this against the archived code that actually ran physics.
    expected_fixed_source = snapshot.read_text().replace(
        'for folder in sorted(output.glob("worker_*")):',
        'for folder in sorted(path for path in output.glob("worker_*") if path.is_dir()):',
    )
    assert script.read_text() == expected_fixed_source
    protected[str(snapshot.relative_to(evaluate.ROOT))] = original_hash
    progress = [json.loads(path.read_text()) for path in output.glob("worker_*/progress.json")]
    assert len(progress) == 12 and all(item["finished"] for item in progress)
    assert sum(item["completed"] for item in progress) == 6000
    evaluate.summarize_and_plot(output, 6000, protected, max(item["wall_time_s"] for item in progress))
    evaluate.write_json(output / "postprocessing_provenance.json", dict(
        corrected_after_evaluation="Exclude worker log files from worker-directory glob",
        only_one_line_of_reporting_code_changed=True,
        original_script_sha256=original_hash,
        original_script_snapshot=str(snapshot.relative_to(evaluate.ROOT)),
        corrected_script_sha256=evaluate.sha256(script),
        physical_runtime_model_and_checkpoint_hashes_unchanged=True,
        wall_time_note="summary.wall_time_s is maximum worker evaluation time, excluding process startup",
        completed_episodes=6000,
    ))


if __name__ == "__main__":
    main()
