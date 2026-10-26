"""Replot the saved 800-Hz model_2500 run after checking its episode records."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "trackstand_2500_hist_matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
SOURCE = ROOT / "isaaclab_train/reports/20261025_2500_dr_comparison/mujoco800"
PREFIX = "202610252053_2500_dr"
STEM = "model_2500_mujoco_800hz_episode_length_distribution"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(survival_only=False):
    source = SOURCE
    prefix = PREFIX
    suffix = "_survival_only" if survival_only else ""
    stem = STEM + suffix
    if survival_only:
        source = ROOT / "mujoco_sim2sim/reports/20261023_2500_dr_evaluation/diagnostic800_survival"
        prefix = "202610230044_2500_dr"
    metadata_path = source / f"{prefix}_metadata.json"
    summary_path = source / f"{prefix}_summary.json"
    archive_path = source / f"{prefix}_evaluation.npz"
    counts_path = source / f"{prefix}_epi_len_histogram.csv"
    experiment_path = source / "experiment.json"
    metadata = json.loads(metadata_path.read_text())
    summary = json.loads(summary_path.read_text())
    experiment = json.loads(experiment_path.read_text())
    episodes = metadata["episodes"]
    arguments = metadata["arguments"]
    num_envs = arguments["num_envs"]
    num_steps = metadata["completed_policy_steps"]

    assert metadata["completed"]
    assert metadata["task"] == "Trackstand-Balance-v1"
    assert arguments["dr"] and arguments["pure_survival"] == survival_only
    if survival_only:
        assert experiment["completed"] and experiment["pure_survival"]
        assert experiment["physics_hz"] == experiment["pd_hz"] == 800
        assert experiment["policy_hz"] == 50
    else:
        assert experiment["capture_completed"]
        assert metadata["physics_hz"] == metadata["pd_hz"] == 800.0
        assert metadata["policy_hz"] == 50.0 and metadata["decimation"] == 16
        assert experiment["physics_dt_s"] == 0.00125 and experiment["policy_dt_s"] == 0.02
        assert experiment["checkpoint_sha256"] == metadata["policy_sha256"]
    assert sha256(Path(metadata["policy"])) == metadata["policy_sha256"]
    assert Path(metadata["policy"]).name == "model_2500.pt"
    assert arguments["steps"] == 1500 and num_envs == 64 and num_steps == 3000

    with np.load(archive_path, allow_pickle=False) as archive:
        control_dt = float(archive["control_dt_s"])
        assert control_dt == 0.02 and float(archive["physics_dt_s"]) == 0.00125
        if "decimation" in archive:
            assert int(archive["decimation"]) == 16
        resets = archive["reset_events"]
        flags = archive["termination_flags"]
        names = archive["termination_names"].tolist()
        assert flags.shape == (num_steps, num_envs, len(names))
        assert names == metadata["termination_names"]
        assert archive["physics_motor_torque_nm"].shape == (num_steps, num_envs, 16, 2)
    if survival_only:
        inactive = [index for index, name in enumerate(names) if name not in ("time_out", "fallen")]
        assert not flags[:, :, inactive].any()

    # Each recorded termination must produce exactly one reset, in the same order.
    expected_resets = np.argwhere(flags.any(axis=2))
    expected_resets[:, 0] += 1
    np.testing.assert_array_equal(resets, expected_resets)
    assert len(resets) == len(episodes) == summary["completed_episodes"]
    previous_end = np.zeros(num_envs, dtype=int)
    rows = []
    for index, ((end_step, env), episode) in enumerate(zip(resets, episodes)):
        length = int(end_step - previous_end[env])
        previous_end[env] = end_step
        assert length == episode["length"] and 0 < length <= 1500
        for term_index, name in enumerate(names):
            assert bool(flags[end_step - 1, env, term_index]) == episode["terminations"][name]
        assert bool(episode["timeout"]) == episode["terminations"]["time_out"]
        if episode["timeout"]:
            assert length == 1500
        rows.append((index + 1, int(env), int(end_step), length, length * control_dt, episode["timeout"]))

    lengths_s = np.asarray([row[4] for row in rows])
    edges_s = np.arange(0.0, 31.0, 3.0)
    counts, _ = np.histogram(lengths_s, bins=edges_s)
    saved_counts = np.loadtxt(counts_path, delimiter=",", skiprows=1)
    np.testing.assert_array_equal(saved_counts[:, :2], np.column_stack((edges_s[:-1], edges_s[1:])))
    np.testing.assert_array_equal(saved_counts[:, 2], counts)
    assert int(counts.sum()) == len(rows)
    assert np.isclose(lengths_s.mean(), summary["mean_episode_length_s"], rtol=0, atol=1e-12)
    timeout_count = sum(row[5] for row in rows)
    assert timeout_count == summary["termination_counts"]["time_out"]
    assert np.isclose(timeout_count / len(rows), summary["timeout_rate"], rtol=0, atol=1e-12)
    assert sum(row[3] for row in rows) + int((num_steps - previous_end).sum()) == num_steps * num_envs

    # Match the October 15 histogram: blue counts, ten 3-second bins, exact bar labels.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(edges_s[:-1], counts, width=3.0, align="edge", color="#1f77b4", edgecolor="white")
    ax.bar_label(bars, padding=3)
    ax.set_xticks(edges_s)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, max(1, int(counts.max()) * 1.15))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Episode length (s)")
    ax.set_ylabel("Number of completed episodes")
    ax.set_title(
        f"Episode length distribution{' — survival only' if survival_only else ''}"
        f" — {len(rows)} completed episodes\n"
        "model_2500 | Domain randomization: ON\n"
        "MuJoCo | Physics: 800 Hz | Motor torque application: 800 Hz | Policy: 50 Hz",
        fontsize=12,
    )
    ax.text(
        0.43, 0.95,
        f"Mean: {lengths_s.mean():.2f} s\n30-s timeouts: {timeout_count}/{len(rows)} ({100 * timeout_count / len(rows):.1f}%)",
        transform=ax.transAxes, va="top", fontsize=10,
    )
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.text(0.5, 0.045, "Bins: [0, 3), ..., [27, 30] s. Timeouts included; unfinished episodes excluded.",
             ha="center", fontsize=9)
    mode = "Survival only: timeout + fallen" if survival_only else "All six task terminations"
    source_date = "2026-10-23" if survival_only else "2026-10-25"
    fig.text(0.5, 0.015, f"{mode} | 64 environments × 60 simulated seconds | Source: {source_date}",
             ha="center", fontsize=9)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT / f"{stem}.{extension}", dpi=150)
    plt.close(fig)

    np.savetxt(
        OUTPUT / f"{stem}.csv", np.column_stack((edges_s[:-1], edges_s[1:], counts)),
        delimiter=",", header="start_s,end_s,episode_count", comments="", fmt=("%.1f", "%.1f", "%d"),
    )
    with (OUTPUT / f"completed_episodes{suffix}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("episode_index", "env_index", "end_policy_step", "length_policy_steps", "length_s", "timeout"))
        writer.writerows(rows)

    result = {
        "passed": True,
        "new_evaluation_run": False,
        "source_files_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (metadata_path, summary_path, archive_path, counts_path, experiment_path)
        },
        "checkpoint_sha256": metadata["policy_sha256"],
        "physics_hz": 800, "motor_torque_application_hz": 800, "policy_target_update_hz": 50,
        "domain_randomization": True, "pure_survival": survival_only,
        "termination_names": names,
        "enabled_termination_names": ["time_out", "fallen"] if survival_only else names,
        "seed": arguments["seed"],
        "environments": num_envs, "simulated_seconds_per_environment": num_steps * control_dt,
        "completed_episodes": len(rows), "mean_episode_length_s": float(lengths_s.mean()),
        "timeouts": timeout_count, "timeout_rate": timeout_count / len(rows),
        "unfinished_episodes_excluded": int(np.count_nonzero(previous_end < num_steps)),
        "bin_edges_s": edges_s.tolist(), "bin_counts": counts.tolist(),
        "validation": [
            "Saved timing, policy identity and local checkpoint hash agree",
            "16 recorded motor-torque applications per 50-Hz policy step",
            "All recorded resets match termination flags and episode lengths",
            "All ten histogram counts agree with the original saved CSV",
            "Mean, timeouts and total simulation-step budget agree with source records",
        ],
    }
    (OUTPUT / f"verification{suffix}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in (
        "passed", "completed_episodes", "mean_episode_length_s", "timeouts", "timeout_rate", "bin_counts",
    )}, indent=2))
    print(OUTPUT / f"{stem}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survival-only", action="store_true", help="Plot the saved timeout/fallen-only run")
    main(survival_only=parser.parse_args().survival_only)
