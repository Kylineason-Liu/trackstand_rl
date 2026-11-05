"""Verify diagnostic samples and build plots without another simulation run."""

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
BASE = REPORT.parent / "20261104_v2_50100_dr_pure_survival_eval"
ISAAC = ROOT / "isaaclab_train/reports/20261104_v2_50100_dr_diagnosis"
CASES = {"nominal": None, "zero_initial": set(), "zero_com": {"com_offset", "com_offset_mj_body"},
         "zero_lateral_com": {"com_offset", "com_offset_mj_body"},
         "only_lateral_com": {"com_offset", "com_offset_mj_body"}, "nominal_mass": {"mass_scale"},
         "nominal_contact": {"tire_friction", "tire_dynamic_friction"},
         "zero_imu_residual": {"gyro_bias", "accel_bias"}, "no_speed_projection": set(), "zero_integral": set()}


def load_case(directory):
    meta = json.loads(next(directory.glob("*_metadata.json")).read_text())
    summary = json.loads(next(directory.glob("*_summary.json")).read_text())
    assert meta["completed"] and meta["completed_per_env"] == meta["episode_quotas"] == [1] * 64
    assert meta["policy_hz"] == 50 and meta["physics_hz"] == 1200 and meta["integral_hz"] == 200
    assert meta["arguments"]["pure_survival"]
    episodes = sorted(meta["episodes"], key=lambda e: e["env_id"])
    assert [e["env_id"] for e in episodes] == list(range(64))
    lengths = np.array([e["length"] for e in episodes])
    assert sum(e["timeout"] for e in episodes) == summary["termination_counts"]["time_out"]
    assert summary["termination_counts"]["time_out"] + summary["termination_counts"]["fallen"] == 64
    np.testing.assert_allclose(lengths.mean() * .02, summary["mean_episode_length_s"], atol=1e-12)
    with np.load(next(directory.glob("*_evaluation.npz"))) as archive:
        mask = archive["active_mask"]
        np.testing.assert_array_equal(mask.sum(axis=0), lengths)
        assert np.isfinite(archive["observation"][mask]).all()
        assert np.isfinite(archive["raw_action"][mask]).all()
        assert np.max(np.abs(archive["raw_action"][mask])) <= 1.
        assert np.isfinite(archive["physics_motor_torque_nm"][mask]).all()
        if directory.name == "zero_integral":
            assert np.all(archive["physics_drive_integral_nm"][mask] == 0.)
    return meta, summary, episodes


def main():
    baseline, base_summary, base_episodes = load_case(BASE)
    rows = [dict(case="full_dr_baseline", timeouts=22, fallen=42, mean_s=12.998125)]
    for case, allowed in CASES.items():
        metadata, summary, episodes = load_case(REPORT / case)
        assert metadata["policy_sha256"] == baseline["policy_sha256"]
        assert metadata["arguments"]["dr"] == (case != "nominal")
        if allowed is not None:
            for original, changed in zip(base_episodes, episodes):
                for key, value in changed["sampled_dr"].items():
                    if key not in allowed:
                        assert value == original["sampled_dr"][key], (case, original["env_id"], key)
        rows.append(dict(case=case, timeouts=summary["termination_counts"]["time_out"],
                         fallen=summary["termination_counts"]["fallen"], mean_s=summary["mean_episode_length_s"]))
    isaac_rows = list(csv.DictReader((ISAAC / "completed_episodes.csv").open()))
    first = sorted((e for e in isaac_rows if int(e["episode_index"]) == 0), key=lambda e: int(e["env_id"]))
    assert len(first) == 64 and [int(e["env_id"]) for e in first] == list(range(64))
    isaac_result = dict(timeouts=sum(e["timeout"] == "True" for e in first),
                        mean_s=float(np.mean([float(e["length_seconds"]) for e in first])),
                        first_completed_episodes=64, same_draws_as_mujoco=False)
    assert isaac_result["timeouts"] == 58
    assert hashlib.sha256((ISAAC / "model_50100.pt").read_bytes()).hexdigest() == \
        "32e43762691fcd25a8e72182339557f0f71f43c14cde2106bb71118eb5019af1"
    matched = {}
    for hz in (800, 1200):
        directory = REPORT / f"matched_v1_{hz}"
        result = json.loads((directory / "summary.json").read_text())
        initial = json.loads((directory / "initial_states.json").read_text())
        assert all(e["sampled_dr"] == b["sampled_dr"] for e, b in zip(initial, base_episodes))
        episodes = json.loads((directory / "episodes.json").read_text())
        assert len(episodes) == 64 and sum(e["timeout"] for e in episodes) == result["timeouts"]
        with np.load(directory / "policy_samples.npz") as archive:
            np.testing.assert_array_equal(archive["active_mask"].sum(axis=0), [e["length"] for e in episodes])
        matched[str(hz)] = result
    with np.load(REPORT / "matched_v1_1200/policy_samples.npz") as v1, \
            np.load(next(BASE.glob("*_evaluation.npz"))) as v2:
        np.testing.assert_array_equal(v1["observation"][0], v2["observation"][0])
        raw1, raw2 = v1["raw_action"][0], v2["raw_action"][0]
        tau1 = raw1[:, 1] * np.float32(6.28)
        tau2 = v2["physics_motor_torque_nm"][0, :, 0, 1]
        demand = dict(initial_observations_equal_all64=True,
                      initial_abs_steering_raw_median_v1_v2=[float(np.median(abs(raw1[:, 0]))),
                                                           float(np.median(abs(raw2[:, 0])))],
                      initial_abs_rear_torque_median_v1_v2_nm=[float(np.median(abs(tau1))), float(np.median(abs(tau2)))],
                      initial_abs_rear_torque_p95_v1_v2_nm=[float(np.percentile(abs(tau1), 95)),
                                                         float(np.percentile(abs(tau2), 95))],
                      v1_torque_provenance="computed 6.28*raw rear action; native v1 torque law",
                      v2_torque_provenance="actual first-physics-step actuator_force archive")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for values, label, color in ((tau1, "v1 rear torque request", "#4c78a8"),
                                     (tau2, "v2 first applied rear torque", "#e45756")):
            axes[0].scatter(v2["observation"][0, :, 0], values, label=label, color=color, s=22, alpha=.8)
        axes[0].set_xlabel("Identical initial observed gravity_y")
        axes[0].set_ylabel("Rear torque (Nm)")
        for values, label, color in ((raw1[:, 0], "v1", "#4c78a8"), (raw2[:, 0], "v2", "#e45756")):
            axes[1].scatter(v2["observation"][0, :, 0], values, label=label, color=color, s=22, alpha=.8)
        axes[1].set_xlabel("Identical initial observed gravity_y")
        axes[1].set_ylabel("Raw steering action before scale/clip/slew")
        for ax in axes:
            ax.grid(alpha=.25)
            ax.legend(loc="lower right", fontsize=8)
        fig.suptitle("64 identical initial policy inputs: learned startup demands differ")
        fig.tight_layout()
        fig.savefig(REPORT / "initial_demands.png", dpi=160)
        plt.close(fig)
    with (REPORT / "ablation_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh([r["case"] for r in rows], [r["timeouts"] for r in rows], color="#4c78a8")
    ax.bar_label(bars, labels=[f'{r["timeouts"]}/64 ({r["timeouts"]/64*100:.1f}%)' for r in rows], padding=4)
    ax.set_xlim(0, 70)
    ax.invert_yaxis()
    ax.set_xlabel("Episodes reaching 30 s")
    ax.set_title("v2 model_50100: controlled MuJoCo diagnostics\n1200 Hz physics/PD; DR + pure survival, seed 42")
    ax.grid(axis="x", alpha=.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(REPORT / "ablation_results.png", dpi=160)
    plt.close(fig)
    protected = {}
    for name in ("protected_sha256_before.json", "additional_protected_sha256_before.json"):
        before = json.loads((REPORT / name).read_text())
        assert all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest for path, digest in before.items())
        protected.update(before)
    output = dict(passed=True, isaac_v2_frozen_policy=isaac_result, matched_v1=matched,
                  v2_ablations=rows, initial_demand_comparison=demand, preserved_inputs=len(protected),
                  new_mujoco_completed_episodes=768, new_isaac_completed_first_episodes=64,
                  production_changes=False, multiagent_used=False)
    (REPORT / "verification.json").write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
