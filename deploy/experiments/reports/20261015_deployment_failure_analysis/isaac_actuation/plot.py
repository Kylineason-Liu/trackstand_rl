"""Plot captured Isaac Lab demand, submitted target, and solver-measured response.

Run with ordinary Python after record_eval.py has completed both modes.
No smoothing, decimation, time shifting, clipping or diagnostic-as-measurement substitution.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
COLORS = ["#4477aa", "#ee7733", "#228833"]
SCALES = np.array([1.57, 6.28])


def load(mode):
    directory = HERE / mode
    metadata = json.loads((directory / "experiment.json").read_text())
    with np.load(directory / "motor_samples.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    raw = arrays["raw_action"]
    n_steps, n_envs, motors = raw.shape
    assert motors == 2 and n_envs == metadata["num_envs"]
    assert n_steps == metadata["completed_policy_steps"] == 3000
    assert metadata["decimation"] == 8
    assert metadata["policy_dt_s"] == 0.02 and metadata["physics_dt_s"] == 0.0025
    assert metadata["dr"] == (mode == "dr") and not metadata["pure_survival"]
    for key, value in arrays.items():
        assert np.isfinite(value).all(), key
    submitted = np.stack((arrays["submitted_position_rad"][:, 0, :, 0],
                          arrays["submitted_effort_nm"][:, 0, :, 1]), axis=-1)
    assert np.allclose(raw * SCALES, submitted, atol=1e-5, rtol=1e-6)
    for key in ("submitted_position_rad", "submitted_effort_nm"):
        assert np.array_equal(arrays[key], np.repeat(arrays[key][:, :1], 8, axis=1))
    assert not json.loads((directory / "source_integrity.json").read_text())["changed_files"]
    assert all(json.loads((directory / "configuration_checks.json").read_text()).values())
    arrays["scaled_raw"] = raw.astype(np.float64) * SCALES
    return metadata, arrays


def intervals(done):
    """Episode segments [start, stop); do not draw motion across a reset."""
    boundaries = np.r_[0, np.flatnonzero(done) + 1, len(done)]
    boundaries = np.unique(boundaries)
    return list(zip(boundaries[:-1], boundaries[1:]))


def draw(datasets, filename, window, envs, symlog=True):
    fig, axes = plt.subplots(2, 2, figsize=(16, 8.8), sharex=True)
    for col, mode in enumerate(("no_dr", "dr")):
        metadata, data = datasets[mode]
        env = envs[mode]
        dt, physics_dt = metadata["policy_dt_s"], metadata["physics_dt_s"]
        raw = data["scaled_raw"][:, env]
        for row, (factor, limit, unit) in enumerate(((180 / np.pi, 90.0, "deg"), (1.0, 6.28, "N·m"))):
            ax = axes[row, col]
            first = True
            for start, stop in intervals(data["done"][:, env]):
                if stop * dt < window[0] or start * dt > window[1]:
                    continue
                lo = max(start, int(np.floor(window[0] / dt)))
                hi = min(stop, int(np.ceil(window[1] / dt)) + 1)
                if hi <= lo:
                    continue
                command_times = np.arange(lo * 8, hi * 8) * physics_dt
                state_times = command_times + physics_dt
                policy_times = np.arange(lo, hi + 1) * dt
                demand = raw[lo:hi, row] * factor
                ax.step(policy_times, np.r_[demand, demand[-1]], where="post", lw=2.8, alpha=0.60,
                        color=COLORS[0], label="Scaled raw policy demand" if first else None, zorder=1)
                target = (data["submitted_position_rad"][lo:hi, :, env, 0] if row == 0
                          else data["submitted_effort_nm"][lo:hi, :, env, 1]).reshape(-1) * factor
                ax.step(np.r_[command_times, hi * dt], np.r_[target, target[-1]], where="post",
                        color=COLORS[1], lw=1.1, ls="--", label="Target submitted to PhysX" if first else None, zorder=2)
                response = (data["position_after_rad"][lo:hi, :, env, 0] if row == 0
                            else data["projected_joint_effort_nm"][lo:hi, :, env, 1]).reshape(-1) * factor
                ax.plot(state_times, response, color=COLORS[2], lw=0.9,
                        label=("Actual simulated steering angle" if row == 0 else "Solver-measured joint-axis effort")
                        if first else None, zorder=3)
                if start and window[0] <= start * dt <= window[1]:
                    ax.axvline(start * dt, color="0.55", lw=0.8, ls=":")
                first = False
            for sign in (-1, 1):
                ax.axhline(sign * limit, color="0.4", lw=0.75, ls="--", zorder=0)
            if symlog:
                ax.set_yscale("symlog", linthresh=90.0 if row == 0 else 6.28, linscale=1.0)
            ax.set_ylabel(f"{'Steering position' if row == 0 else 'Rear joint torque'} ({unit})")
            ax.set_xlim(*window)
            ax.grid(True, alpha=0.22)
            ax.legend(fontsize=8, loc="upper right")
            if row == 0:
                ax.set_title(f"{'No DR' if mode == 'no_dr' else 'Full DR'} · env {env}")
            else:
                ax.set_xlabel("Evaluation time (s; vertical dotted lines mark resets)")
    fig.suptitle("Isaac Lab · model_91850 · demand versus measured response", fontsize=17)
    scale_note = "Symmetric-log y axes retain startup spikes. " if symlog else "Linear y axes. "
    fig.text(0.5, 0.022, scale_note + "50 Hz policy / 400 Hz physics and measurement · seed 42\n"
             "Blue and orange overlap: submitted targets are not Python-clipped. "
             "Rear green is joint-axis effort, not the clipped ImplicitActuator estimate.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    for extension in ("png", "svg", "pdf"):
        fig.savefig(HERE / f"{filename}.{extension}", dpi=180)
    plt.close(fig)


def describe(metadata, data):
    measured = data["projected_joint_effort_nm"].astype(np.float64)
    limits = np.array([5.6, 6.28])
    raw = data["scaled_raw"]
    starts = np.zeros(data["done"].shape, dtype=bool)
    starts[0] = True
    starts[1:] = data["done"][:-1]
    physics_starts = np.repeat(starts[:, None], metadata["decimation"], axis=1)
    return {
        "policy_samples": int(np.prod(raw.shape[:2])),
        "physics_samples_per_motor": int(np.prod(measured.shape[:3])),
        "completed_episodes": int(data["done"].sum()),
        "scaled_demand_abs_max_rad_nm": np.abs(raw).max(axis=(0, 1)).tolist(),
        "measured_joint_effort_abs_max_nm": np.abs(measured).max(axis=(0, 1, 2)).tolist(),
        # A sensor has finite solver precision. Keep strict and conservative
        # margins separate rather than treating micro-newton-metres as material.
        "effort_over_nominal_plus_1e_5_count": (np.abs(measured) > limits + 1e-5).sum(axis=(0, 1, 2)).tolist(),
        "effort_over_nominal_plus_0_01_count": (np.abs(measured) > limits + 0.01).sum(axis=(0, 1, 2)).tolist(),
        "after_first_action_effort_abs_max_nm": np.abs(measured[~physics_starts]).max(axis=0).tolist(),
        "after_first_action_effort_over_nominal_plus_0_01_count":
            (np.abs(measured[~physics_starts]) > limits + 0.01).sum(axis=0).tolist(),
        "clipped_estimate_abs_max_nm": np.abs(data["clipped_torque_estimate_nm"]).max(axis=(0, 1, 2)).tolist(),
        "env0_effort_abs_max_nm": np.abs(measured[:, :, 0]).max(axis=(0, 1)).tolist(),
        "env0_peak_rear_demand_nm": float(np.abs(raw[:, 0, 1]).max()),
        "max_rear_demand_environment": int(np.unravel_index(np.abs(raw[..., 1]).argmax(), raw.shape[:2])[1]),
    }


def export_env(metadata, data, env, mode):
    fields = ["command_time_s", "response_time_s", "episode", "raw_steering", "raw_rear",
              "scaled_steering_rad", "scaled_rear_nm", "submitted_steering_rad", "submitted_rear_nm",
              "actual_steering_rad", "solver_measured_steering_joint_effort_nm", "solver_measured_rear_joint_effort_nm",
              "diagnostic_clipped_steering_nm", "diagnostic_clipped_rear_nm"]
    episodes = np.r_[0, np.cumsum(data["done"][:-1, env])]
    with (HERE / f"{mode}_env{env}_traces.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for step in range(len(episodes)):
            for sub in range(metadata["decimation"]):
                command_time = (step * metadata["decimation"] + sub) * metadata["physics_dt_s"]
                index = (step, sub, env)
                writer.writerow([
                    command_time, command_time + metadata["physics_dt_s"], int(episodes[step]),
                    *data["raw_action"][step, env], *data["scaled_raw"][step, env],
                    data["submitted_position_rad"][index][0], data["submitted_effort_nm"][index][1],
                    data["position_after_rad"][index][0], *data["projected_joint_effort_nm"][index],
                    *data["clipped_torque_estimate_nm"][index],
                ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=int, default=0, help="Same explicitly selected environment in both modes.")
    args = parser.parse_args()
    if (HERE / "summary.json").exists():
        raise FileExistsError("Figures already generated; preserve the existing experiment outputs.")
    datasets = {mode: load(mode) for mode in ("no_dr", "dr")}
    summary = {mode: describe(*dataset) for mode, dataset in datasets.items()}
    for mode, (metadata, data) in datasets.items():
        if not 0 <= args.env < metadata["num_envs"]:
            raise ValueError("Invalid environment index")
        export_env(metadata, data, args.env, mode)
    envs = dict.fromkeys(datasets, args.env)
    draw(datasets, "isaac_demand_vs_actuation", (0.0, 60.0), envs)
    draw(datasets, "isaac_startup_demand_vs_actuation", (0.0, 0.6), envs)
    draw(datasets, "isaac_tracking_detail", (0.5, 5.0), envs, symlog=False)
    peak_envs = {mode: result["max_rear_demand_environment"] for mode, result in summary.items()}
    draw(datasets, "isaac_peak_startup_demand_vs_actuation", (0.0, 0.6), peak_envs)
    summary["plotted_environment"] = args.env
    summary["peak_startup_environments"] = peak_envs
    inputs = [HERE / mode / name for mode in datasets for name in ("motor_samples.npz", "experiment.json")]
    inputs.extend([Path(__file__).resolve(), HERE / "sensor_validation.json"])
    summary["input_sha256"] = {str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
