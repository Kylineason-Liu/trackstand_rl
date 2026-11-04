"""Verify matched DR trials and summarize physics/PD frequency diagnostics."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[2]
DIRECTORIES = {400: REPORT / "hz400", 800: REPORT / "hz800",
               1200: REPORT.parent / "20261104_v2_50100_dr_pure_survival_eval",
               1600: REPORT.parent / "20261104_v2_50100_frequency_sensitivity/hz1600",
               2400: REPORT.parent / "20261104_v2_50100_frequency_sensitivity/hz2400"}


def main():
    rows, per_env, metadata_by_rate = [], {}, {}
    for hz, directory in DIRECTORIES.items():
        metadata = json.loads(next(directory.glob("*_metadata.json")).read_text())
        summary = json.loads(next(directory.glob("*_summary.json")).read_text())
        episodes = sorted(metadata["episodes"], key=lambda item: item["env_id"])
        assert metadata["completed"] and metadata["completed_per_env"] == [1] * 64
        assert metadata["episode_quotas"] == [1] * 64
        assert [e["env_id"] for e in episodes] == list(range(64))
        assert metadata["physics_hz"] == metadata["pd_hz"] == hz
        assert metadata["policy_hz"] == 50 and metadata["integral_hz"] == 200
        assert metadata["arguments"]["dr"] and metadata["arguments"]["pure_survival"]
        assert metadata["arguments"]["velocity_limit_mode"] == "project"
        lengths = np.array([e["length"] for e in episodes])
        counts = {name: sum(e["terminations"][name] for e in episodes)
                  for name in metadata["termination_names"]}
        assert counts == summary["termination_counts"]
        assert counts["time_out"] + counts["fallen"] == 64
        assert all(counts[name] == 0 for name in ("diverged", "drifted", "yaw_deviation", "wheel_net"))
        per_env[hz] = lengths * .02
        np.testing.assert_allclose(lengths.mean() * .02, summary["mean_episode_length_s"], atol=1e-12)
        with np.load(next(directory.glob("*_evaluation.npz"))) as archive:
            mask = archive["active_mask"]
            np.testing.assert_array_equal(mask.sum(axis=0), lengths)
            assert int(mask.sum()) == metadata["environment_transitions"]
            assert np.isfinite(archive["raw_action"][mask]).all()
            assert np.max(np.abs(archive["raw_action"][mask])) <= 1.
            # Same 0..0.20-s window: no environment terminates before 0.22 s.
            assert mask[:10].all()
            speed = archive["physics_drive_velocity_rad_s"][:10]
            spans = np.ptp(speed, axis=2)
            torque = archive["physics_motor_torque_nm"][:10, :, :, 1]
            startup = dict(
                speed_span_per_20ms_median_p95_max_rad_s=np.percentile(spans, [50, 95, 100]).tolist(),
                adjacent_speed_jump_p95_p99_max_rad_s=np.percentile(np.abs(np.diff(speed, axis=2)),
                                                                  [95, 99, 100]).tolist(),
                rear_torque_sign_change_fraction=float(np.mean(torque[:, :, 1:] * torque[:, :, :-1] < 0.)),
            )
            if hz == 1200:
                # Concrete startup case: the largest span is an acceleration
                # toward a large requested speed, not itself alternating chatter.
                step, env = np.unravel_index(np.argmax(spans), spans.shape)
                startup["largest_span_example"] = dict(
                    env_id=int(env), policy_step=int(step),
                    target_rad_s=float(archive["held_drive_target_rad_s"][step, env]),
                    velocity_rad_s=speed[step, env].tolist(), torque_nm=torque[step, env].tolist())
        row = dict(physics_pd_hz=hz, completed=64, timeouts=counts["time_out"], fallen=counts["fallen"],
                   timeout_percent=counts["time_out"] / 64 * 100,
                   mean_s=summary["mean_episode_length_s"], median_s=float(np.median(lengths) * .02),
                   min_s=float(lengths.min() * .02), max_s=float(lengths.max() * .02),
                   transitions=int(lengths.sum()), physics_steps=int(lengths.sum()) * round(hz / 50),
                   startup_diagnostics=startup)
        rows.append(row)
        metadata_by_rate[hz] = metadata
    baseline = sorted(metadata_by_rate[1200]["episodes"], key=lambda e: e["env_id"])
    baseline_survived = per_env[1200] == 30
    for hz in (400, 800, 1600, 2400):
        other = sorted(metadata_by_rate[hz]["episodes"], key=lambda e: e["env_id"])
        assert all(a["sampled_dr"] == b["sampled_dr"] for a, b in zip(baseline, other))
        assert metadata_by_rate[hz]["policy_sha256"] == metadata_by_rate[1200]["policy_sha256"]
        row = next(r for r in rows if r["physics_pd_hz"] == hz)
        survived = per_env[hz] == 30
        row["rescued_baseline_falls"] = int(np.sum(~baseline_survived & survived))
        row["lost_baseline_survivors"] = int(np.sum(baseline_survived & ~survived))
    with (REPORT / "per_environment_lengths.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["env_id", *[f"length_{hz}hz_s" for hz in DIRECTORIES]])
        writer.writerows((i, *[per_env[hz][i] for hz in DIRECTORIES]) for i in range(64))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = [str(r["physics_pd_hz"]) for r in rows]
    for ax, key, ylabel in ((axes[0], "timeout_percent", "30-s survivors (%)"),
                            (axes[1], "mean_s", "Mean episode length (s)")):
        bars = ax.bar(labels, [r[key] for r in rows], color=["#e45756", "#f58518", "#4c78a8", "#72b7b2", "#54a24b"])
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_xlabel("Physics / PD frequency (Hz)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 100 if key == "timeout_percent" else 30)
        ax.grid(axis="y", alpha=.25)
        ax.set_axisbelow(True)
    fig.suptitle("model_50100 | DR + pure survival | 64 matched draws, seed 42\nPolicy 50 Hz; rear integral 200 Hz")
    fig.tight_layout()
    fig.savefig(REPORT / "frequency_comparison.png", dpi=160)
    plt.close(fig)
    before = json.loads((REPORT / "protected_sha256_before.json").read_text())
    after = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in before}
    assert before == after
    (REPORT / "protected_sha256_after.json").write_text(json.dumps(after, indent=2) + "\n")
    output = dict(passed=True, results=rows, same_sampled_dr_all_64=True,
                  original_inputs_unchanged=len(before), physics_and_pd_changed_together=True,
                  limitations=["One seed, 64 matched trials, not an exhaustive timestep convergence proof",
                               "Cannot separate explicit-PD discretization from contact/integration changes",
                               "Startup span/sign-change metrics describe motion; they are not identified oscillation frequencies"],
                  multiagent_used=False)
    (REPORT / "comparison.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
