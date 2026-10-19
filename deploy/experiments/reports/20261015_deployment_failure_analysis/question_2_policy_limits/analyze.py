"""Compare archived Isaac/bike demands with new, instrumented MuJoCo episodes.

No runtime source/model edits, GUI, ROS, or CAN. Existing outputs are never overwritten.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = next(path for path in Path(__file__).resolve().parents if (path / "isaaclab_train").is_dir())
ISAAC = ROOT / "isaaclab_train/reports/motor_limits_09221945_91850_2026-10-01"
BIKE = ROOT / "deploy/experiments/reports/deployment_20261012_180607"
MUJOCO = ROOT / "mujoco_sim2sim"
MODEL_NAME = "trackstand_rl_rsl_2026-09-22_19-45-06_model_91850.onnx"
SCALES = np.array([1.57, 6.28])
LIMITS = np.array([np.pi / 2, 6.28])
TOL = 1e-5
COLORS = ["#4477aa", "#66aa88", "#228833", "#cc6677"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(raw, starts, scaled=None, times=None):
    raw = np.asarray(raw, dtype=np.float64)
    scaled = raw * SCALES if scaled is None else np.asarray(scaled, dtype=np.float64)
    starts = np.asarray(starts, dtype=bool)
    assert raw.shape == scaled.shape and raw.shape[1] == 2 and len(starts) == len(raw)
    assert np.isfinite(raw).all() and np.isfinite(scaled).all()
    over = np.abs(scaled) > LIMITS + TOL
    result = {
        "samples": len(raw), "raw_abs_max": np.abs(raw).max(axis=0).tolist(),
        "scaled_abs_max": np.abs(scaled).max(axis=0).tolist(),
        "steer_peak_deg": float(np.rad2deg(np.abs(scaled[:, 0]).max())),
        "count": over.sum(axis=0).tolist(), "percent": (100 * over.mean(axis=0)).tolist(),
        "first_samples": int(starts.sum()), "first_count": over[starts].sum(axis=0).tolist(),
        "after_first_samples": int((~starts).sum()),
        "after_first_count": over[~starts].sum(axis=0).tolist(),
        "after_first_abs_max": np.abs(scaled[~starts]).max(axis=0).tolist(),
        "raw_abs_over_one_count": (np.abs(raw) > 1 + TOL).sum(axis=0).tolist(),
    }
    changes = np.abs(np.diff(scaled[:, 0])) / (0.02 if times is None else np.diff(times))
    within = ~starts[1:]
    result["steer_target_change_rate"] = {
        "samples": int(within.sum()), "count_over_23_9": int((changes[within] > 23.9 + TOL).sum()),
        "maximum_rad_s": float(changes[within].max()),
        "meaning": ("Rate required for exact 20-ms target tracking, not a commanded/measured motor velocity"
                    if times is None else "Uses recorded bag-clock policy intervals"),
    }
    return result


def load_archived():
    with np.load(ISAAC / "motor_limit_samples.npz", allow_pickle=False) as data:
        raw = data["raw_action"]
        target = data["processed_target"]
        starts = np.zeros(raw.shape[:2], dtype=bool)
        starts[0] = True
        starts[1:] = data["done"][:-1]
        isaac = describe(raw.reshape(-1, 2), starts.reshape(-1), target.reshape(-1, 2))
        rates = np.abs(np.diff(target[:, :, 0], axis=0)) / 0.02
        within = ~starts[1:]
        isaac["steer_target_change_rate"] = {
            "samples": int(within.sum()), "count_over_23_9": int((rates[within] > 23.9 + TOL).sum()),
            "maximum_rad_s": float(rates[within].max()),
            "meaning": "Within each environment/episode, not differences between flattened environments",
        }
        # Re-evaluate archived torque/speed evidence; these are NOT actual total PhysX torques.
        computed = data["computed_torque_nm"].reshape(-1, 2)
        diagnostic = data["applied_torque_estimate_nm"].reshape(-1, 2)
        speed = data["velocity_after_rad_s"].reshape(-1, 2)
        isaac["physics_samples"] = len(computed)
        isaac["computed_torque_abs_max"] = np.abs(computed).max(axis=0).tolist()
        isaac["computed_torque_over_count"] = (np.abs(computed) > [5.6 + TOL, 6.28 + TOL]).sum(axis=0).tolist()
        isaac["diagnostic_abs_max"] = np.abs(diagnostic).max(axis=0).tolist()
        isaac["diagnostic_over_count"] = (np.abs(diagnostic) > [5.6 + TOL, 6.28 + TOL]).sum(axis=0).tolist()
        isaac["actual_speed_abs_max"] = np.abs(speed).max(axis=0).tolist()
        isaac["speed_over_count"] = (np.abs(speed) > [23.9 + TOL, 61.6 + TOL]).sum(axis=0).tolist()
        isaac["actual_position_abs_max_rad"] = np.abs(data["position_after_rad"][..., 0]).max().item()
        isaac["rear_submitted_ff_abs_max_nm"] = np.abs(data["physx_effort_target_nm"][..., 1]).max().item()
        isaac["shape"] = list(raw.shape)
    original = json.loads((ISAAC / "motor_limit_statistics.json").read_text())
    for metric in original["metrics"]:
        if metric["metric"] == "steering_position_target_outside_joint_range":
            assert isaac["count"][0] == metric["count"]
        if metric["metric"] == "drive_feedforward_target_above_effort_limit":
            assert isaac["count"][1] == metric["count"]

    records = np.genfromtxt(BIKE / "policy_samples.csv", delimiter=",", names=True)
    raw_bike = np.column_stack([records["raw_steer"], records["raw_rear"]])
    summary = json.loads((BIKE / "summary.json").read_text())
    bike_starts = np.zeros(len(records), dtype=bool)
    attempts = []
    for attempt in summary["attempts"]:
        selected = ((records["bag_time_s"] >= attempt["first_s"] - 1e-7)
                    & (records["bag_time_s"] <= attempt["last_policy_s"] + 1e-7))
        indices = np.flatnonzero(selected)
        assert len(indices) == attempt["policy_count"]
        bike_starts[indices[0]] = True
        local_starts = np.zeros(len(indices), dtype=bool)
        local_starts[0] = True
        attempts.append({"attempt": attempt["attempt"],
                         **describe(raw_bike[indices], local_starts, times=records["bag_time_s"][indices])})
    bike = describe(raw_bike, bike_starts, times=records["bag_time_s"])
    bike["attempts"] = attempts
    commands = np.genfromtxt(BIKE / "active_motor_commands.csv", delimiter=",", names=True)
    bike["successful_send_ticks"] = len(commands)
    bike["final_encoded_abs_max"] = [float(np.abs(commands[field]).max()) for field in
                                     ["mit_encoded_position_rad", "drive_encoded_torque_nm"]]
    bike["final_encoded_over_count"] = [int((np.abs(commands[field]) > bound + TOL).sum()) for field, bound in
                                        zip(["mit_encoded_position_rad", "drive_encoded_torque_nm"], LIMITS)]
    bike["context"] = summary["user_conditions"]
    assert bike["count"] == [summary["policy"]["steering_requests_beyond_90"],
                             summary["policy"]["rear_requests_beyond_6_28"]]
    return isaac, bike, raw_bike, records


def collect_mujoco(policy):
    import newbike_mujoco

    viewer_spec = importlib.util.spec_from_file_location("q2_viewer", MUJOCO / "scripts/run_deploy_viewer.py")
    viewer = importlib.util.module_from_spec(viewer_spec)
    viewer_spec.loader.exec_module(viewer)
    xml = MUJOCO / "models/trackstand_rl_0916_bike/trackstand_rl_0916.xml"
    spec = mujoco.MjSpec.from_file(str(xml))
    viewer.augment_trackstand_model(spec, str(xml.parent))
    model = spec.compile()
    assert model.opt.timestep == 0.0025 and np.all(model.actuator_forcelimited)
    collected = {}
    metrics = {}
    native_step = mujoco.mj_step
    for dr in (False, True):
        label = "mujoco_dr" if dr else "mujoco_no_dr"
        rows, starts, episodes, physics = [], [], [], []
        for seed in range(42, 50):
            env = newbike_mujoco.BikeBalanceEnv(model=model, seed=seed, enable_dr=dr)
            obs = env.reset()
            begin = len(rows)

            def observed_step(m, d, *args, **kwargs):
                result = native_step(m, d, *args, **kwargs)
                physics.append(np.r_[d.actuator_force.copy(), d.qvel[[env._steer_dof, env._rear_dof]].copy()])
                return result

            mujoco.mj_step = observed_step
            try:
                for index in range(1500):
                    action = policy.act(obs)
                    starts.append(index == 0)
                    obs, done = env.step(action)
                    rows.append(np.r_[action, env.data.ctrl.copy(),
                                      env.data.qpos[env._steer_qadr],
                                      env.data.qvel[[env._steer_dof, env._rear_dof]],
                                      env.data.actuator_force.copy()])
                    if done:
                        break
            finally:
                mujoco.mj_step = native_step
            episodes.append({"seed": seed, "begin": begin, "end": len(rows), "steps": index + 1,
                             "timeout": bool(env.stats.timeout), "death": env.stats.death})
        data = np.asarray(rows)
        physics_data = np.asarray(physics)
        stats = describe(data[:, :2], starts)
        stats["episodes"] = episodes
        stats["physics_samples"] = len(physics_data)
        stats["actual_force_abs_max_nm"] = np.abs(physics_data[:, :2]).max(axis=0).tolist()
        stats["actual_force_over_count"] = (np.abs(physics_data[:, :2]) > [5.6 + TOL, 6.28 + TOL]).sum(axis=0).tolist()
        stats["speed_post_control_abs_max"] = np.abs(data[:, 5:7]).max(axis=0).tolist()
        stats["speed_pre_manual_clamp_abs_max"] = np.abs(physics_data[:, 2:4]).max(axis=0).tolist()
        stats["stored_ctrl_abs_max"] = np.abs(data[:, 2:4]).max(axis=0).tolist()
        stats["actual_steer_position_abs_max_rad"] = float(np.abs(data[:, 4]).max())
        assert len(physics_data) == 8 * len(data)
        assert stats["actual_force_over_count"] == [0, 0]
        metrics[label] = stats
        collected[label] = data
        collected[label + "_physics"] = physics_data
        collected[label + "_starts"] = np.asarray(starts)
        print(label, json.dumps(stats), flush=True)
    return metrics, collected


def plot_summary(groups, output):
    labels = ["Isaac\nDR", "MuJoCo\nno DR", "MuJoCo\nDR", "Bike bag\npre-disable"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for motor, title in enumerate(["Steering position demand", "Rear feed-forward torque demand"]):
        ax = axes[0, motor]
        percentages = [group["percent"][motor] for group in groups]
        ax.bar(labels, percentages, color=COLORS)
        ax.set_ylim(0, 85)
        ax.set_ylabel("Samples above reference (%)")
        ax.set_title(title + (" > 90°" if motor == 0 else " > 6.28 N·m"))
        for i, group in enumerate(groups):
            ax.text(i, percentages[i] + 1.3, f'{percentages[i]:.3f}%\n{group["count"][motor]:,}/{group["samples"]:,}',
                    ha="center", fontsize=9)
        ax = axes[1, motor]
        peaks = [group["scaled_abs_max"][motor] for group in groups]
        bound = LIMITS[motor]
        if motor == 0:
            peaks = np.rad2deg(peaks)
            bound = 90
        ax.bar(labels, peaks, color=COLORS)
        ax.axhline(bound, color="black", linestyle="--", label=f"Reference {bound:g}")
        ax.set_ylim(0, max(peaks) * 1.18)
        ax.set_ylabel("Maximum absolute demand (deg)" if motor == 0 else "Maximum absolute demand (N·m)")
        for i, peak in enumerate(peaks):
            ax.text(i, peak + max(peaks) * .025, f"{peak:.2f}\n{peak / bound:.2f}× reference", ha="center", fontsize=9)
        ax.legend(fontsize=9)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Model 91850: raw-policy demands after trained scales, before limiters", fontsize=14)
    fig.text(.5, .015, "Different datasets, not matched failure probabilities. Steering raw is position, not torque.\n"
             "Isaac: archived 64×3000; MuJoCo: new seeds 42–49, one episode each; bike: 3 attempts, early stand contact.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .95))
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(output / f"exceedance_comparison.{ext}", dpi=180)
    plt.close(fig)


def plot_mujoco(collected, metrics, output):
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    for column, key in enumerate(["mujoco_no_dr", "mujoco_dr"]):
        data = collected[key]
        time = np.arange(len(data)) * .02
        raw = data[:, :2] * SCALES
        axes[0, column].plot(time, np.rad2deg(raw[:, 0]), color="#4477aa", lw=.8, label="Scaled raw demand")
        axes[0, column].plot(time, np.rad2deg(data[:, 2]), color="#ee7733", lw=.8, label="Clipped/slewed ctrl")
        axes[0, column].plot(time, np.rad2deg(data[:, 4]), color="#228833", lw=.7, label="Actual steering angle")
        axes[1, column].plot(time, raw[:, 1], color="#4477aa", lw=.8, label="Scaled raw demand")
        axes[1, column].plot(time, np.clip(data[:, 3], -6.28, 6.28), color="#ee7733", lw=.8, label="Effective ctrl before damping")
        axes[1, column].plot(time, data[:, 8], color="#228833", lw=.8, label="Actual actuator torque")
        for row, bound in enumerate([90, 6.28]):
            ax = axes[row, column]
            ax.axhline(bound, color="black", ls="--", lw=.7)
            ax.axhline(-bound, color="black", ls="--", lw=.7)
            ax.set_yscale("symlog", linthresh=bound)
            ax.grid(alpha=.2)
            ax.set_xlabel("Concatenated episode policy time (s)")
            ax.legend(fontsize=8, loc="upper right")
            for episode in metrics[key]["episodes"][1:]:
                ax.axvline(episode["begin"] * .02, color="gray", alpha=.3, lw=.7)
        axes[0, column].set_title("MuJoCo DR " + ("ON" if key == "mujoco_dr" else "OFF"))
    axes[0, 0].set_ylabel("Steering angle (deg), symlog")
    axes[1, 0].set_ylabel("Rear torque (N·m), symlog")
    fig.suptitle("Model 91850: all eight episodes per mode — oversized demands versus limited actuation")
    fig.text(.5, .015, "Vertical lines separate seeds 42–49: not one continuous trajectory. "
             "No smoothing/lag correction. Traces sample each policy step; force caps also checked at every physics step.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .95))
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(output / f"mujoco_demand_vs_actuation.{ext}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    output = parser.parse_args().output.resolve()
    for filename in ["summary.json", "mujoco_samples.npz", "counts.csv", "exceedance_comparison.png",
                     "exceedance_comparison.svg", "exceedance_comparison.pdf", "mujoco_demand_vs_actuation.png",
                     "mujoco_demand_vs_actuation.svg", "mujoco_demand_vs_actuation.pdf"]:
        if (output / filename).exists():
            raise FileExistsError(f"Preserving previous output: {output / filename}; use --output with a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    inputs = [ISAAC / "motor_limit_samples.npz", ISAAC / "motor_limit_statistics.json", ISAAC / "experiment.json",
              BIKE / "policy_samples.csv", BIKE / "active_motor_commands.csv", BIKE / "summary.json",
              MUJOCO / "scripts/run_deploy_viewer.py", MUJOCO / "src/newbike_mujoco/env.py",
              MUJOCO / "src/newbike_mujoco/policy.py", MUJOCO / "models/trackstand_rl_0916_bike/trackstand_rl_0916.xml",
              MUJOCO / "policies" / MODEL_NAME, ROOT / "deploy/source/trackstand_deploy/models" / MODEL_NAME,
              Path(__file__).resolve()]
    before = {str(path.relative_to(ROOT)): sha(path) for path in inputs}
    assert sha(MUJOCO / "policies" / MODEL_NAME) == sha(ROOT / "deploy/source/trackstand_deploy/models" / MODEL_NAME)
    sys.path.insert(0, str(MUJOCO / "src"))
    from newbike_mujoco import DeployedPolicy

    policy = DeployedPolicy(onnx_path=str(MUJOCO / "policies" / MODEL_NAME))
    isaac, bike, raw_bike, records = load_archived()
    terms = ["gravity_y", "gravity_z", "angular_x_rad_s", "angular_y_rad_s", "angular_z_rad_s",
             "estimated_vx_m_s", "placeholder_vy_m_s", "steering_position_rad", "rear_position_rad",
             "steering_velocity_rad_s", "rear_velocity_rad_s", "sin_steer", "cos_steer"]
    observations = np.column_stack([records[field] for field in terms]).astype(np.float32)
    replay = policy.act_batch(observations)
    replay_error = float(np.abs(replay - raw_bike).max())
    assert replay_error < 1e-4
    mj_metrics, collected = collect_mujoco(policy)
    summary = {"model": "09221945/model_91850", "scales": SCALES.tolist(), "references": LIMITS.tolist(),
               "native_unit_tolerance": TOL, "onnx_bag_replay_max_error": replay_error,
               "versions": {"mujoco": mujoco.__version__, "numpy": np.__version__},
               "isaac_archived_dr": isaac, **mj_metrics, "bike_pre_disable": bike,
               "source_sha256": before,
               "mujoco_policy_row_fields": ["raw_steer", "raw_drive", "stored_ctrl_steer_rad", "stored_ctrl_drive_nm",
                                             "actual_steer_rad", "steer_velocity_rad_s", "rear_velocity_rad_s",
                                             "actual_steer_actuator_nm", "actual_drive_actuator_nm"],
               "mujoco_physics_row_fields": ["steer_actuator_nm", "drive_actuator_nm",
                                              "steer_velocity_before_manual_clamp_rad_s", "rear_velocity_before_manual_clamp_rad_s"],
               "conditions": "New MuJoCo: actual unchanged viewer augmentation/env/policy, no GUI/XML write; "
                             "seeds 42..49, one episode per seed/mode, max1500 steps, normal observation noise. "
                             "All physics steps instrumented by wrapping and delegating mj_step without state changes. "
                             "Isaac/bike are existing data, not rerun, and are not matched trajectories."}
    assert before == {str(path.relative_to(ROOT)): sha(path) for path in inputs}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(output / "mujoco_samples.npz", **collected)
    groups = [isaac, mj_metrics["mujoco_no_dr"], mj_metrics["mujoco_dr"], bike]
    with (output / "counts.csv").open("w") as stream:
        stream.write("dataset,samples,steer_excess_count,steer_excess_percent,steer_max_deg,rear_excess_count,rear_excess_percent,rear_max_nm\n")
        for name, group in zip(["isaac_dr", "mujoco_no_dr", "mujoco_dr", "bike_pre_disable"], groups):
            stream.write(f'{name},{group["samples"]},{group["count"][0]},{group["percent"][0]},'
                         f'{group["steer_peak_deg"]},{group["count"][1]},{group["percent"][1]},{group["scaled_abs_max"][1]}\n')
    plot_summary(groups, output)
    plot_mujoco(collected, mj_metrics, output)
    print(json.dumps({key: summary[key] for key in ["isaac_archived_dr", "mujoco_no_dr", "mujoco_dr", "bike_pre_disable"]}, indent=2))
    print("Artifacts:", output)


if __name__ == "__main__":
    main()
