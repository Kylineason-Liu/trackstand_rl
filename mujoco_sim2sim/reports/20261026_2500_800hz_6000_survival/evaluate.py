"""Collect completed survival episodes with unchanged production environment/policy code.

Independent environment streams run in subprocesses. Each policy call keeps the
production batch shape (64, 11); unused rows do not interact with the MLP's rows.
Only episode records are retained, rather than a large physics-trajectory archive.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/trackstand_6000_survival_matplotlib")

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
COMPONENT = ROOT / "mujoco_sim2sim"
CHECKPOINT = ROOT / "isaaclab_train/logs/rsl_rl/trackstand_rl_rsl/2026-10-21_11-18-29/model_2500.pt"
REFERENCE = COMPONENT / "reports/20261023_2500_dr_evaluation/diagnostic800_survival"
sys.path.insert(0, str(COMPONENT / "src"))


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, default=serial, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_first_episodes():
    metadata = json.loads((REFERENCE / "202610230044_2500_dr_metadata.json").read_text())
    with np.load(REFERENCE / "202610230044_2500_dr_evaluation.npz", allow_pickle=False) as capture:
        resets = capture["reset_events"]
    first = {}
    for reset, stats in zip(resets, metadata["episodes"]):
        first.setdefault(int(reset[1]), stats)
    assert len(first) == 64
    return first


def worker(args):
    import torch
    from newbike_mujoco import DeployedPolicy, build_env_from_mjcf
    from newbike_mujoco import contract as c
    from newbike_mujoco.viewer_envs import ViewerEnvGroup

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    policy = DeployedPolicy(pt_path=CHECKPOINT)
    first, _ = build_env_from_mjcf(c.DEFAULT_MJCF, 42, True, physics_hz=800, velocity_limit_mode="project")
    group = ViewerEnvGroup(first, 64, 42)
    owned = list(range(args.worker_id, 64, args.workers))
    quotas = [args.episodes // 64 + int(index < args.episodes % 64) for index in range(64)]
    active = [index for index in owned if quotas[index] > 0]
    completed = np.zeros(64, dtype=int)
    observations = np.zeros((64, 11), dtype=np.float32)
    reset_records = {}
    reference = reference_first_episodes()
    matched_reference = []
    calls = 0
    count = 0
    simulated_policy_steps = 0
    started = last_progress = time.monotonic()
    output = args.output_dir / f"worker_{args.worker_id:02d}"
    output.mkdir(exist_ok=False)
    write_json(output / "friction_bank.json", first._dr_friction_buckets)

    def reset(index):
        env = group.envs[index]
        env.enabled_terms = {"time_out", "fallen"}
        assert env.physics_dt == .00125 and env.decimation == 16 and env.CTRL_DT == .02
        assert env.model.opt.timestep == .00125 and env.physics_torques.shape == (16, 2)
        observation = env.reset()
        reset_records[index] = dict(dr=env.dr, qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(),
                                    kp=env.kp.copy(), kd=env.kd.copy())
        return observation

    def progress(finished=False):
        write_json(output / "progress.json", dict(
            completed=count, target=sum(quotas[index] for index in owned),
            simulated_policy_steps=simulated_policy_steps,
            wall_time_s=time.monotonic() - started, finished=finished,
            first_episode_reference_matches=matched_reference,
        ))

    for index in active:
        observations[index] = reset(index)
    progress()
    with (output / "episodes.jsonl").open("w", buffering=1) as stream:
        while active:
            actions = policy.act_batch(observations)
            calls += 1
            for index in active.copy():
                env = group.envs[index]
                observation, done = env.step(actions[index], observe_terminal=False)
                simulated_policy_steps += 1
                observations[index] = observation
                if not done:
                    continue
                stats = env.stats
                assert set(name for name, value in stats.terminations.items() if value) <= {"time_out", "fallen"}
                assert stats.timeout or stats.terminations["fallen"]
                assert 0 < stats.length <= 1500
                assert abs(env.data.time - stats.length * .02) < 1e-8
                assert np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all()
                assert np.isfinite(env.physics_torques).all()
                assert np.all(np.abs(env.physics_torques) <= np.array([5.6, 6.28]) + 1e-10)
                if stats.timeout:
                    assert stats.length == 1500
                if completed[index] == 0:
                    expected = reference[index]
                    assert stats.length == expected["length"], (index, stats.length, expected["length"])
                    assert stats.timeout == expected["timeout"]
                    assert np.isclose(stats.roll_sq, expected["roll_sq"], rtol=1e-10, atol=1e-12)
                    matched_reference.append(index)
                completed[index] += 1
                entry = dict(
                    env_index=index, seed=42 + index, episode=int(completed[index]),
                    end_policy_step=calls, duration_s=stats.length * .02,
                    stats={key: value for key, value in vars(stats).items() if key != "rolls"},
                    reset=reset_records[index],
                )
                stream.write(json.dumps(entry, default=serial, allow_nan=False) + "\n")
                count += 1
                if completed[index] == quotas[index]:
                    active.remove(index)
                    observations[index] = 0.
                else:
                    observations[index] = reset(index)
            now = time.monotonic()
            if now - last_progress >= 5:
                progress()
                last_progress = now
    assert all(completed[index] == quotas[index] for index in owned)
    progress(finished=True)
    print(f"Worker {args.worker_id}: completed {count} episodes", flush=True)


def summarize_and_plot(output, target, protected, wall_time):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from newbike_mujoco.env import EpisodeStats
    from newbike_mujoco.reporting import summarize

    entries = []
    progress = []
    banks = []
    for folder in sorted(path for path in output.glob("worker_*") if path.is_dir()):
        entries.extend(json.loads(line) for line in (folder / "episodes.jsonl").read_text().splitlines())
        progress.append(json.loads((folder / "progress.json").read_text()))
        banks.append(json.loads((folder / "friction_bank.json").read_text()))
    entries.sort(key=lambda entry: (entry["env_index"], entry["episode"]))
    assert len(entries) == target and all(item["finished"] for item in progress)
    assert all(bank == banks[0] for bank in banks)
    assert sorted(index for item in progress for index in item["first_episode_reference_matches"]) == list(range(64))
    records = []
    per_environment = []
    for index in range(64):
        cohort = [entry for entry in entries if entry["env_index"] == index]
        assert len(cohort) == target // 64 + int(index < target % 64)
        previous_step = 0
        for number, entry in enumerate(cohort, 1):
            assert entry["seed"] == 42 + index and entry["episode"] == number
            assert entry["stats"]["length"] == entry["end_policy_step"] - previous_step
            previous_step = entry["end_policy_step"]
            assert entry["duration_s"] == entry["stats"]["length"] * .02
            records.append((entry["stats"]["length"], entry["stats"]["death"], EpisodeStats(**entry["stats"])))
        per_environment.append(dict(env_index=index, seed=42 + index, completed_episodes=len(cohort)))
    assert sum(record[0] for record in records) == sum(item["simulated_policy_steps"] for item in progress)
    for filename, expected in protected.items():
        assert sha256(ROOT / filename) == expected, filename
    summary = summarize(records, 64)
    assert summary["termination_counts"]["time_out"] + summary["termination_counts"]["fallen"] >= target
    assert all(summary["termination_counts"][name] == 0 for name in ("diverged", "drifted", "yaw_deviation", "wheel_net"))
    summary.update(
        physics_hz=800, motor_torque_application_hz=800, policy_hz=50,
        pure_survival=True, domain_randomization=True, episode_horizon_s=30,
        wall_time_s=wall_time, total_simulated_seconds=sum(record[0] for record in records) * .02,
        sampling="93 or 94 completed episodes per environment stream for target=6000",
        seed_range=[42, 105], per_environment=per_environment,
    )
    write_json(output / "summary.json", summary)
    lengths = np.array([entry["duration_s"] for entry in entries])
    edges = np.arange(0., 31., 3.)
    counts, _ = np.histogram(lengths, edges)
    assert counts.sum() == target
    stem = "model_2500_mujoco_800hz_6000_survival_histogram"
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(edges[:-1], counts, width=3, align="edge", color="#1f77b4", edgecolor="white")
    ax.bar_label(bars, padding=3)
    ax.set(xlim=(0, 30), ylim=(0, max(counts) * 1.15), xticks=edges,
           xlabel="Episode length (s)", ylabel="Number of completed episodes")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(
        f"Episode length distribution — survival only — {target:,} completed episodes\n"
        "model_2500 | Domain randomization: ON\n"
        "MuJoCo | Physics: 800 Hz | Motor torque application: 800 Hz | Policy: 50 Hz", fontsize=12,
    )
    timeouts = summary["termination_counts"]["time_out"]
    ax.text(.43, .95, f"Mean: {lengths.mean():.2f} s\n30-s timeouts: {timeouts:,}/{target:,} "
            f"({100 * timeouts / target:.1f}%)", transform=ax.transAxes, va="top", fontsize=10)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout(rect=(0, .09, 1, 1))
    fig.text(.5, .045, "Bins: [0, 3), ..., [27, 30] s. Timeouts included; every counted episode completed.",
             ha="center", fontsize=9)
    fig.text(.5, .015, "Survival only: timeout + fallen | 64 seed streams | 93–94 episodes per stream",
             ha="center", fontsize=9)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"{stem}.{extension}", dpi=150)
    plt.close(fig)
    np.savetxt(output / "histogram_counts.csv", np.column_stack((edges[:-1], edges[1:], counts)),
               delimiter=",", header="start_s,end_s,episode_count", comments="", fmt=("%.1f", "%.1f", "%d"))
    with (output / "completed_episodes.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("env_index", "seed", "episode", "length_policy_steps", "length_s", "timeout", "fallen"))
        for entry in entries:
            st = entry["stats"]
            writer.writerow((entry["env_index"], entry["seed"], entry["episode"], st["length"],
                             entry["duration_s"], st["timeout"], st["terminations"]["fallen"]))
    write_json(output / "verification.json", dict(
        passed=True, completed_episodes=len(entries), bin_counts=counts,
        all_64_first_episodes_match_saved_reference=True,
        protected_input_hashes_unchanged=True, physics_hz=800, motor_torque_application_hz=800,
        policy_target_update_hz=50, pure_survival=True,
        complete_episode_and_step_accounting=True, shared_friction_bank_identical_across_workers=True,
        raw_episode_records_sha256={str(path.relative_to(output)): sha256(path)
                                   for path in sorted(output.glob("worker_*/episodes.jsonl"))},
    ))
    print(json.dumps({name: summary[name] for name in (
        "completed_episodes", "mean_episode_length_s", "timeout_rate", "termination_counts", "wall_time_s",
    )}, indent=2), flush=True)


def main(args):
    assert args.episodes >= 64 and 1 <= args.workers <= 64
    args.output_dir.mkdir(parents=True, exist_ok=False)
    paths = list((COMPONENT / "src/newbike_mujoco").glob("*.py"))
    paths.extend((COMPONENT / "models/trackstand_rl_0916_bike").rglob("*"))
    paths.extend((CHECKPOINT, Path(__file__)))
    protected = {str(path.relative_to(ROOT)): sha256(path) for path in paths if path.is_file()}
    assert sha256(CHECKPOINT) == "5389effbd6a0d3e78c9ea9caeac309ceb5082622c8ee125bb701c63d1511fe52"
    import importlib.metadata
    write_json(args.output_dir / "experiment.json", dict(
        started_utc=datetime.now(timezone.utc).isoformat(), command=[sys.executable, *sys.argv],
        target_completed_episodes=args.episodes, workers=args.workers, environment_streams=64,
        physics_hz=800, motor_torque_application_hz=800, policy_hz=50, decimation=16,
        pure_survival=True, domain_randomization=True, horizon_s=30, velocity_limit_mode="project",
        per_environment_quotas=[args.episodes // 64 + int(index < args.episodes % 64) for index in range(64)],
        policy=str(CHECKPOINT.relative_to(ROOT)), policy_sha256=sha256(CHECKPOINT),
        protected_inputs_sha256=protected,
        versions={name: importlib.metadata.version(name) for name in ("mujoco", "torch", "numpy", "matplotlib")},
    ))
    started = time.monotonic()
    processes = []
    streams = []
    try:
        for index in range(args.workers):
            log = (args.output_dir / f"worker_{index:02d}.log").open("w")
            streams.append(log)
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker-id", str(index),
                       "--episodes", str(args.episodes), "--workers", str(args.workers),
                       "--output-dir", str(args.output_dir)]
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        last_report = 0.
        while any(process.poll() is None for process in processes):
            failed = [(index, process.returncode) for index, process in enumerate(processes)
                      if process.poll() is not None and process.returncode != 0]
            if failed:
                raise RuntimeError(f"Worker failed: {failed}; inspect worker logs")
            now = time.monotonic()
            if now - last_report >= 20:
                progress = [json.loads(path.read_text()) for path in args.output_dir.glob("worker_*/progress.json")]
                completed = sum(item["completed"] for item in progress)
                elapsed = now - started
                print(f"Progress: {completed}/{args.episodes} completed episodes; {elapsed:.1f} seconds elapsed",
                      flush=True)
                last_report = now
            time.sleep(1)
        assert all(process.returncode == 0 for process in processes)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        for stream in streams:
            stream.close()
    summarize_and_plot(args.output_dir, args.episodes, protected, time.monotonic() - started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "run")
    parser.add_argument("--worker-id", type=int)
    arguments = parser.parse_args()
    arguments.output_dir = arguments.output_dir.resolve()
    if arguments.worker_id is None:
        main(arguments)
    else:
        worker(arguments)
