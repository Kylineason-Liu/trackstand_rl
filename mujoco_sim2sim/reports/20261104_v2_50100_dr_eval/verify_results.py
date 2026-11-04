"""Reconcile native episode metadata, masked traces and histogram; no rollout."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[2]


def main():
    metadata_path, = REPORT.glob("*_metadata.json")
    base = metadata_path.name.removesuffix("_metadata.json")
    metadata = json.loads(metadata_path.read_text())
    summary = json.loads((REPORT / f"{base}_summary.json").read_text())
    episodes = sorted(metadata["episodes"], key=lambda item: item["env_id"])
    assert metadata["completed"] and metadata["stop_reason"] == "requested_budget_complete"
    assert metadata["episode_quotas"] == metadata["completed_per_env"] == [1] * 64
    assert [item["env_id"] for item in episodes] == list(range(64))
    assert all(item["episode_index"] == 0 for item in episodes)
    assert metadata["arguments"]["dr"] is True and not metadata["arguments"]["pure_survival"]
    assert metadata["policy_backend"] == "onnxruntime"
    assert (metadata["policy_hz"], metadata["physics_hz"], metadata["integral_hz"]) == (50, 1200, 200)
    lengths = np.array([item["length"] for item in episodes])
    seconds = lengths * .02
    terms = metadata["termination_names"]
    counts = {name: sum(bool(item["terminations"][name]) for item in episodes) for name in terms}
    assert counts == summary["termination_counts"]
    assert int(lengths.sum()) == metadata["environment_transitions"]
    assert float(lengths.mean()) == summary["mean_episode_length_steps"]
    np.testing.assert_allclose(seconds.mean(), summary["mean_episode_length_s"], rtol=0., atol=1e-12)
    counts_hist, _ = np.histogram(seconds, bins=np.arange(0., 31., 3.))
    histogram = np.loadtxt(REPORT / f"{base}_epi_len_histogram.csv", delimiter=",", skiprows=1)
    np.testing.assert_array_equal(counts_hist, histogram[:, 2])
    assert int(counts_hist.sum()) == len(episodes) == summary["completed_episodes"] == 64
    with np.load(REPORT / f"{base}_evaluation.npz") as archive:
        active = archive["active_mask"]
        assert active.shape == (1500, 64)
        np.testing.assert_array_equal(active.sum(axis=0), lengths)
        expected_resets = np.array(sorted((int(n), i) for i, n in enumerate(lengths)))
        np.testing.assert_array_equal(archive["reset_events"], expected_resets)
        actions = archive["raw_action"][active]
        assert actions.shape == (int(lengths.sum()), 2)
        assert np.isfinite(actions).all() and np.max(np.abs(actions)) <= 1.
        assert np.isfinite(archive["observation"][active]).all()
        torque = archive["physics_motor_torque_nm"][active]
        assert np.isfinite(torque).all()
        peaks = np.max(np.abs(torque), axis=(0, 1))
        assert np.all(peaks <= np.array([5.6, 6.28]) + 1e-10)
        flags = archive["termination_flags"]
        for index, episode in enumerate(episodes):
            np.testing.assert_array_equal(flags[lengths[index] - 1, index],
                                          [episode["terminations"][name] for name in terms])
    before = json.loads((REPORT / "protected_sha256_before.json").read_text())
    after = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in before}
    assert before == after
    (REPORT / "protected_sha256_after.json").write_text(json.dumps(after, indent=2) + "\n")
    with (REPORT / "episodes.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["env_id", "length_steps", "length_s", "termination", *terms])
        for episode in episodes:
            writer.writerow([episode["env_id"], episode["length"], episode["length"] * .02,
                             "+".join(name for name in terms if episode["terminations"][name]),
                             *[int(episode["terminations"][name]) for name in terms]])
    result = dict(
        passed=True, native_prefix=base, completed_episodes=64,
        mean_length_s=float(seconds.mean()), median_length_s=float(np.median(seconds)),
        min_length_s=float(seconds.min()), max_length_s=float(seconds.max()),
        std_length_s=float(seconds.std()), quartiles_s=np.quantile(seconds, [.25, .75]).tolist(),
        timeouts=counts["time_out"], termination_counts=counts,
        histogram_counts=counts_hist.tolist(),
        timeout_env_ids=[e["env_id"] for e in episodes if e["timeout"]],
        episode_mask_reset_flag_histogram_counts_agree=True,
        environment_transitions=int(lengths.sum()), physics_steps=int(lengths.sum()) * 24,
        integral_updates=int(lengths.sum()) * 4,
        peak_absolute_raw_actions=np.max(np.abs(actions), axis=0).tolist(),
        peak_absolute_motor_torque_nm=peaks.tolist(),
        motor_order=["steering", "rear"], original_inputs_hash_unchanged=len(before),
        sample_scope="64 fixed-quota initial DR episodes, seed 42, no pure_survival; one seed only",
        multiagent_used=False,
    )
    (REPORT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
