"""Validate saved evaluations, compare like-for-like survival, and plot results."""

import csv
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from experiment import protected_hashes, write_json

VARIANTS = ("historical_v0_91850_survival", "current400_full", "current400_survival",
            "diagnostic800_full", "diagnostic800_survival")
LABELS = ("v0 / 91850\n400 Hz\nsurvival", "v1 / 2500\n400 Hz\nfull task",
          "v1 / 2500\n400 Hz\nsurvival", "v1 / 2500\n800 Hz\nfull task",
          "v1 / 2500\n800 Hz\nsurvival")
COLORS = ("#777777", "#1f77b4", "#2a9d8f", "#b7791f", "#8e44ad")


def load(variant):
    folder = OUT / variant
    old = variant.startswith("historical")
    meta_path = folder / "metadata.json" if old else next(folder.glob("*_metadata.json"))
    archive = folder / "evaluation.npz" if old else next(folder.glob("*_evaluation.npz"))
    meta = json.loads(meta_path.read_text())
    assert meta["completed"]
    with np.load(archive) as z:
        obs, raw = z["observation"], z["raw_action"]
        resets = z["reset_events"]
        assert obs.shape == (3000, 64, 13 if old else 11) and raw.shape == (3000, 64, 2)
        assert all(np.isfinite(z[name]).all() for name in
                   ("observation", "raw_action", "position_rad", "velocity_rad_s"))
        rows = meta["episodes"]
        assert len(rows) == len(resets)
        if not old:
            for record, (_, index) in zip(rows, resets):
                record["env"] = int(index)
            torque = z["physics_motor_torque_nm"]
            hz = 800 if variant.startswith("diagnostic") else 400
            assert torque.shape == (3000, 64, hz // 50, 2)
            assert np.isfinite(torque).all() and np.abs(raw).max() <= 1
            peak = np.max(np.abs(torque), axis=(0, 1, 2))
            assert np.all(peak <= np.array([5.6, 6.28]) + 1e-6)
            assert np.max(np.abs(torque[..., 1] - (raw[..., 1]*np.float32(6.28))[..., None])) < 1e-6
            expected = np.zeros(64)
            flags = z["termination_flags"]
            held_targets = z["held_steering_target_rad"]
            for step in range(3000):
                desired = np.clip(raw[step, :, 0]*np.float32(1.57), -1.57, 1.57)
                expected += np.clip(desired - expected, -.478, .478)
                assert np.max(np.abs(expected - held_targets[step])) < 1e-6
                expected[np.any(flags[step], axis=1)] = 0
        else:
            torque = z["motor_torque_drive_steering_nm"]
            assert np.isfinite(torque).all()
            peak = np.max(np.abs(torque[..., ::-1]), axis=(0, 1))
        validation = dict(observation_shape=list(obs.shape), raw_shape=list(raw.shape),
                          raw_min=raw.min(axis=(0, 1)).tolist(), raw_max=raw.max(axis=(0, 1)).tolist(),
                          max_abs_motor_torque_steering_rear_Nm=peak.tolist(), all_samples_finite=True,
                          dr=True, physics_dt_s=float(z["physics_dt_s"]), control_dt_s=float(z["control_dt_s"]))
        per_env = np.zeros((64, 3))  # completed seconds, count, timeouts
        first, episode_index = [], np.zeros(64, dtype=int)
        for record in rows:
            index = record["env"]
            episode_index[index] += 1
            record["episode"] = int(episode_index[index])
            per_env[index] += [record["length"]*.02, 1, int(record["timeout"])]
            if record["episode"] == 1:
                first.append(record)
        first.sort(key=lambda r: r["env"])
        assert len(first) == 64 and [r["env"] for r in first] == list(range(64))
        lengths = np.array([r["length"]*.02 for r in rows])
        assert lengths.max() <= 30 and lengths.min() > 0
        unfinished = 60 - per_env[:, 0]
        assert unfinished.min() > -1e-8 and unfinished.max() < 30 + 1e-8
        counts, edges = np.histogram(lengths, bins=np.arange(0, 31, 3))
        assert counts.sum() == len(rows)
        stats = dict(completed_episodes=len(rows), mean_episode_length_s=float(lengths.mean()),
                     median_episode_length_s=float(np.median(lengths)), std_episode_length_s=float(lengths.std()),
                     timeouts=int(sum(r["timeout"] for r in rows)),
                     timeout_rate=float(sum(r["timeout"] for r in rows)/len(rows)),
                     roll_rms_deg=float(np.degrees(np.sqrt(sum(r["roll_sq"] for r in rows)/sum(r["length"] for r in rows)))),
                     max_roll_deg=float(np.degrees(max(r["roll_max"] for r in rows))),
                     mean_abs_drive_speed_rad_s=float(sum(r["drive_speed_sum"] for r in rows)/sum(r["length"] for r in rows)),
                     mean_abs_steering_speed_rad_s=float(sum(r["steer_speed_sum"] for r in rows)/sum(r["length"] for r in rows)),
                     first_episode_mean_s=float(np.mean([r["length"]*.02 for r in first])),
                     first_episode_timeouts=int(sum(r["timeout"] for r in first)),
                     first_episode_timeout_rate=float(sum(r["timeout"] for r in first)/64),
                     incomplete_episodes=int(np.count_nonzero(unfinished > 1e-8)),
                     incomplete_time_s=float(unfinished.sum()), histogram_counts=counts.tolist(),
                     validation=validation)
        terms = ("time_out", "fallen", "diverged", "drifted", "yaw_deviation", "wheel_net")
        stats["termination_counts"] = {name: sum(bool(r.get("terminations", {}).get(name, False)) for r in rows)
                                       for name in terms}
        if old:
            stats["termination_counts"] = {"time_out": stats["timeouts"],
                                            "fallen": sum(r["death"] == "fallen" for r in rows)}
        if not old:
            summary = json.loads(next(folder.glob("*_summary.json")).read_text())
            for key in ("completed_episodes", "mean_episode_length_s", "timeout_rate", "roll_rms_deg"):
                assert np.isclose(stats[key], summary[key], rtol=1e-12, atol=1e-12), key
        draws = json.loads((folder / "dr_draws.json").read_text())
        assert len(draws) == 64 + len(rows)
        for draw in draws:
            dr = draw["dr"]
            assert dr is not None and np.all((np.array(dr["mass_scale"]) >= .8) & (np.array(dr["mass_scale"]) <= 1.2))
            assert np.max(np.abs(dr["com_offset"])) <= .005
            assert np.min(dr["tire_friction"]) >= .5 and np.max(dr["tire_friction"]) <= .8
            assert .9 <= dr["steer_kp_scale"] <= 1.1 and .9 <= dr["steer_kv_scale"] <= 1.1
            if not old:
                assert np.allclose(draw["kp"][1], 0) and np.allclose(draw["kd"][1], 0)
                assert np.all(np.array(dr["tire_dynamic_friction"]) <= np.array(dr["tire_friction"]))
        stats["validation"]["recorded_dr_resets"] = len(draws)
    return stats, rows, first, per_env


def compare_bootstrap(loaded):
    rng = np.random.default_rng(20261023)
    samples = {}
    for variant, (_, rows, first, per_env) in loaded.items():
        ids = rng.integers(0, 64, size=(20000, 64))
        total = per_env[ids].sum(axis=1)
        seconds = np.array([r["length"]*.02 for r in first])
        timeouts = np.array([r["timeout"] for r in first], dtype=float)
        samples[variant] = dict(mean_s=total[:, 0]/total[:, 1], timeout_rate=total[:, 2]/total[:, 1],
                                first_mean_s=seconds[ids].mean(axis=1), first_timeout_rate=timeouts[ids].mean(axis=1))
    baseline = samples[VARIANTS[0]]
    output = {}
    for variant in VARIANTS[1:]:
        output[variant] = {}
        for metric in baseline:
            difference = samples[variant][metric] - baseline[metric]
            output[variant][metric + "_difference_95ci"] = np.quantile(difference, [.025, .975]).tolist()
    output["method"] = ("20,000 independent environment-cluster bootstrap resamples per variant, seed20261023; "
                        "percentile95% intervals for v1 minus fresh v0. Whole environment trajectories are "
                        "resampled, not independent episodes. Conditional on this single shared material bank "
                        "and these trained checkpoints; not a causal policy-only or multi-training-seed comparison.")
    return output


def main():
    loaded = {variant: load(variant) for variant in VARIANTS}
    result = {variant: value[0] for variant, value in loaded.items()}
    result["bootstrap"] = compare_bootstrap(loaded)
    assert protected_hashes() == json.loads((OUT / "protected_before.json").read_text())
    result["protected_files_unchanged"] = True
    write_json(OUT / "comparison.json", result)
    with (OUT / "episodes.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "env", "episode", "length_s", "timeout", "death"])
        for variant, (_, rows, _, _) in loaded.items():
            writer.writerows((variant, r["env"], r["episode"], r["length"]*.02, r["timeout"], r["death"]) for r in rows)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    values = [loaded[v][0] for v in VARIANTS]
    for ax, key, ylabel in ((axes[0], "mean_episode_length_s", "Mean completed episode length (s)"),
                            (axes[1], "timeout_rate", "Completed episodes reaching 30 s (%)")):
        y = [r[key]*(100 if key == "timeout_rate" else 1) for r in values]
        bars = ax.bar(np.arange(5), y, color=COLORS)
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_xticks(np.arange(5), LABELS, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(y)*1.25)
        ax.grid(axis="y", alpha=.2)
    for variant, label, color in zip(VARIANTS, LABELS, COLORS):
        lengths = np.array([r["length"]*.02 for r in loaded[variant][2]])
        t = np.arange(0, 30.001, .02)
        axes[2].step(t, (lengths[:, None] >= t).mean(axis=0)*100, where="post", label=label.replace("\n", " "), color=color)
    axes[2].set(xlabel="Time since first spawn (s)", ylabel="First episodes lasting at least t (%)", xlim=(0, 30), ylim=(0, 105))
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=.2)
    fig.suptitle("MuJoCo DR comparison — 64 environments × 3,000 policy steps, seeds42–105\n"
                 "Policy50Hz throughout; 800Hz results are diagnostic overrides, not production defaults")
    fig.tight_layout()
    prior = sorted(OUT.glob("*_2500_dr_v0_comparison.png"))
    stem = prior[0].stem if prior else time.strftime("%Y%m%d%H%M") + "_2500_dr_v0_comparison"
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, variant, label, color in zip(axes, (VARIANTS[0], VARIANTS[2], VARIANTS[1]),
                                        (LABELS[0], LABELS[2], LABELS[1]), (COLORS[0], COLORS[2], COLORS[1])):
        counts = np.array(loaded[variant][0]["histogram_counts"])
        n = counts.sum()
        bars = ax.bar(np.arange(0, 30, 3), counts/n*100, align="edge", width=3, color=color, edgecolor="white")
        ax.bar_label(bars, labels=[str(x) for x in counts], fontsize=8, padding=2)
        ax.set(title=label.replace("\n", " | ")+f"\n{n} completed episodes; labels=count", xlabel="Episode length (s)",
               xticks=np.arange(0, 31, 3), xlim=(0, 30), ylim=(0, 105))
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Share of completed episodes (%)")
    fig.suptitle("DR episode-length distributions at400Hz — equal60s/env budget, different episode counts")
    fig.tight_layout()
    fig.savefig(OUT / (stem.replace("v0_comparison", "epi_len_comparison")+".png"), dpi=160)
    plt.close(fig)
    for variant, value in loaded.items():
        s = value[0]
        print(variant, {key: s[key] for key in ("completed_episodes", "mean_episode_length_s", "timeouts", "timeout_rate",
                                               "first_episode_mean_s", "first_episode_timeouts", "roll_rms_deg")})
    print(json.dumps(result["bootstrap"], indent=2))
    print("VALIDATION PASSED; protected files unchanged")


if __name__ == "__main__":
    main()
