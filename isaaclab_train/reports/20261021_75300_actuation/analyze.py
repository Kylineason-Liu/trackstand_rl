"""Verify 75300 captures and generate presentation figures, CSVs and HTML.

No smoothing, resampling, signal clipping or response-time alignment is used.
Commands are at step start, simulated responses at step end; resets break lines.
"""

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
MODES = ("no_dr", "dr")
TITLES = {"no_dr": "No DR · training-matched", "dr": "Full DR · robustness test"}
COLORS = ("#4477aa", "#ee7733", "#228833")
SCALES = np.array([1.57, 6.28])
FIGURES = []


def segments(done):
    edges = np.unique(np.r_[0, np.flatnonzero(done) + 1, len(done)])
    return list(zip(edges[:-1], edges[1:]))


def spawn_times(done, policy_dt_s=0.02):
    """Initial spawn and end-of-step auto-resets, including a reset at capture end."""
    return np.r_[0, np.flatnonzero(done) + 1] * policy_dt_s


def episode_statistics(data):
    lengths, timeouts = [], []
    for env in range(data["done"].shape[1]):
        endings = np.flatnonzero(data["done"][:, env]) + 1
        lengths.extend(np.diff(np.r_[0, endings]).tolist())
        timeouts.extend(data["truncated"][endings - 1, env].tolist())
    lengths, timeouts = np.array(lengths), np.array(timeouts, dtype=bool)
    counts, edges = np.histogram(lengths * 0.02, bins=np.linspace(0, 30, 11))
    return lengths, timeouts, counts, edges


def load(mode):
    directory = HERE / mode
    metadata = json.loads((directory / "experiment.json").read_text())
    with np.load(directory / "motor_samples.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    assert metadata["capture_completed"] and metadata["completed_policy_steps"] == 3000
    assert metadata["decimation"] == 8 and metadata["num_envs"] == 64
    assert metadata["dr"] == (mode == "dr") and not metadata["pure_survival"]
    assert not metadata["training_dr"] and metadata["seed"] == 42
    assert metadata["policy_dt_s"] == 0.02 and metadata["physics_dt_s"] == 0.0025
    assert data["raw_action"].shape == (3000, 64, 2) and data["observation"].shape == (3000, 64, 11)
    assert data["submitted_motor_torque_nm"].shape == (3000, 8, 64, 2)
    assert all(np.isfinite(value).all() for value in data.values())
    assert all(json.loads((directory / "configuration_checks.json").read_text()).values())
    integrity = json.loads((directory / "source_integrity.json").read_text())
    assert integrity["evaluation_completed"] and not integrity["changed_files"]
    assert np.max(np.abs(data["raw_action"])) <= 1.0
    data["scaled_raw"] = data["raw_action"].astype(np.float64) * SCALES
    held = data["held_steering_target_rad"]
    np.testing.assert_array_equal(held, np.repeat(held[:, :1], 8, axis=1))
    previous = np.r_[held[:1, 0] * 0, held[:-1, 0]].copy()
    resets = np.r_[np.ones((1, 64), dtype=bool), data["done"][:-1]]
    previous[resets] = np.clip(data["position_before_rad"][:, 0, :, 0][resets], -1.57, 1.57)
    expected_target = previous + np.clip(np.clip(data["scaled_raw"][..., 0], -1.57, 1.57) - previous, -0.478, 0.478)
    np.testing.assert_allclose(held[:, 0], expected_target, rtol=1e-6, atol=3e-7)
    assert np.max(np.abs(held)) <= 1.570001
    assert np.max(np.abs(held[:, 0] - previous)) <= 0.478001
    computed = np.empty_like(data["computed_motor_torque_nm"])
    computed[..., 0] = data["controller_kp"][..., 0] * (held - data["position_before_rad"][..., 0])
    computed[..., 0] -= data["controller_kd"][..., 0] * data["velocity_before_rad_s"][..., 0]
    computed[..., 1] = data["scaled_raw"][:, None, :, 1]
    np.testing.assert_allclose(data["computed_motor_torque_nm"], computed, rtol=1e-5, atol=2e-6)
    np.testing.assert_allclose(data["clipped_motor_torque_nm"], np.clip(computed, -np.array([5.6, 6.28]), [5.6, 6.28]),
                               rtol=1e-5, atol=2e-6)
    np.testing.assert_array_equal(data["submitted_motor_torque_nm"], data["clipped_motor_torque_nm"])
    assert (np.abs(data["submitted_motor_torque_nm"]) <= np.array([5.6, 6.28]) + 1e-5).all()
    assert not data["controller_kp"][..., 1].any() and not data["controller_kd"][..., 1].any()
    return metadata, data


def save(fig, name, caption):
    fig.suptitle("Isaac Lab · Trackstand-Balance-v1 · model_75300 · seed 42", fontsize=15)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.text(0.5, 0.012, "50 Hz policy / 400 Hz physics · " + caption, ha="center", fontsize=8)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(HERE / f"{name}.{extension}", dpi=170)
    plt.close(fig)
    FIGURES.append((name, caption))


def decorate(ax, mode, ylabel, xlabel=True):
    ax.set_title(TITLES[mode], fontsize=11)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel("Evaluation time (s)")
    ax.grid(alpha=0.22)


def traces(datasets, name, window, envs, fit_y=False, nominal_zoom=False):
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=True)
    settings = {}
    for col, mode in enumerate(MODES):
        metadata, data = datasets[mode]
        env = envs[mode]
        dt = metadata["policy_dt_s"]
        spawns = spawn_times(data["done"][:, env], dt)
        spawns = spawns[(spawns >= window[0]) & (spawns <= window[1])]
        settings[mode] = {"environment": env, "spawn_times_s": spawns.tolist(), "panels": {}}
        for row, (factor, unit, limit) in enumerate(((180 / np.pi, "deg", np.rad2deg(1.57)), (1, "N m", 6.28))):
            ax, first = axes[row, col], True
            all_values, range_values = [], []
            for start, stop in segments(data["done"][:, env]):
                lo, hi = max(start, int(window[0] / 0.02)), min(stop, int(np.ceil(window[1] / 0.02)) + 1)
                if hi <= lo:
                    continue
                command_time = np.arange(lo * 8, hi * 8) * 0.0025
                policy_time = np.arange(lo, hi + 1) * 0.02
                demand = data["scaled_raw"][lo:hi, env, row] * factor
                ax.step(policy_time, np.r_[demand, demand[-1]], where="post", color=COLORS[0],
                        lw=2.6, alpha=0.6, label="Scaled raw policy output" if first else None)
                target = (data["held_steering_target_rad"][lo:hi, :, env] if row == 0
                          else data["submitted_motor_torque_nm"][lo:hi, :, env, 1]).reshape(-1) * factor
                response = (data["position_after_rad"][lo:hi, :, env, 0] if row == 0
                            else data["projected_joint_effort_nm"][lo:hi, :, env, 1]).reshape(-1) * factor
                values = (demand, target, response)
                all_values.extend(values)
                # Only the nominal zoom's axis calculation excludes startup; the
                # plotted arrays/timestamps are unchanged, including all transients.
                settled = np.arange(lo, hi) - start >= int(round(0.5 / dt))
                range_values.extend([demand[settled], target.reshape(-1, 8)[settled].ravel(),
                                     response.reshape(-1, 8)[settled].ravel()]
                                    if nominal_zoom and mode == "no_dr" else values)
                ax.step(np.r_[command_time, hi * 0.02], np.r_[target, target[-1]], where="post", color=COLORS[1],
                        ls="--", lw=1.2, label=("Held steering target entering PD" if row == 0 else
                                               "Motor torque submitted to PhysX") if first else None)
                ax.plot(command_time + 0.0025, response, color=COLORS[2], lw=0.9,
                        label=("Actual simulated steering position" if row == 0 else
                               "Solver-measured joint-axis effort") if first else None)
                first = False
            values = np.concatenate(all_values)
            fitted = np.concatenate(range_values)
            if not fitted.size:
                fitted = values  # A short window may contain only startup data.
            low, high = float(fitted.min()), float(fitted.max())
            padding = max((high - low) * 0.1, 1e-6)
            if fit_y:
                ax.set_ylim(low - padding, high + padding)
            for sign in (-1, 1):
                if not fit_y or low - padding <= sign * limit <= high + padding:
                    ax.axhline(sign * limit, color="0.5", ls="--", lw=0.7)
            for index, time in enumerate(spawns):
                ax.axvline(time, color="#d62728", lw=0.9, alpha=0.7, zorder=4, clip_on=False,
                           label="Episode spawn" if index == 0 else None)
            decorate(ax, mode, f"{'Steering position' if row == 0 else 'Rear torque'} ({unit})")
            ax.set_title(TITLES[mode] + f" · env {env}")
            ax.set_xlim(*window)
            ax.legend(fontsize=7, loc="upper right")
            bounds = ax.get_ylim()
            settings[mode]["panels"]["steering" if row == 0 else "rear"] = {
                "units": unit, "ylim": list(bounds),
                "plotted_min_max": [float(values.min()), float(values.max())],
                "axis_fitting_excludes_first_s_each_episode": 0.5 if nominal_zoom and mode == "no_dr" else 0.0,
                "samples_outside_y_view": int(((values < bounds[0]) | (values > bounds[1])).sum()),
            }
    caption = (f"{window[0]:g}–{window[1]:g} s · red: spawns (including boundary resets) · "
               "independent linear axes · no smoothing/time shift")
    if nominal_zoom:
        caption += "\nNominal Y-zoom: bounds from episode age ≥0.5 s; startup extrema outside view. DR: full range."
    elif fit_y:
        caption += "\nAll samples visible; distant motor-cap references do not stretch the axes."
    save(fig, name, caption)
    return settings


def save_demand_settings(full, zoom, hashes):
    (HERE / "demand_plot_settings.json").write_text(json.dumps({
        "source": "existing saved captures; no simulation rerun", "input_sha256": hashes,
        "full_range": full, "vertical_zoom": zoom,
        "terminal_reset_note": "A reset at 60 s has no subsequent samples in this capture.",
    }, indent=2) + "\n")


def redraw_demand():
    """Redraw only demand figures from verified saved captures; do not rerun physics."""
    summary = json.loads((HERE / "summary.json").read_text())
    hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in summary["input_sha256"]}
    assert hashes == summary["input_sha256"]
    datasets = {mode: load(mode) for mode in MODES}
    envs = dict.fromkeys(MODES, 0)
    full = traces(datasets, "isaac_demand_vs_actuation", (0, 60), envs, fit_y=True)
    zoom = traces(datasets, "isaac_demand_vs_actuation_zoomed", (0, 60), envs, fit_y=True, nominal_zoom=True)
    save_demand_settings(full, zoom, hashes)
    print(json.dumps({"full_range": full, "vertical_zoom": zoom}, indent=2))


def simple_traces(datasets, name, fields, ylabel, reference=None):
    fig, axes = plt.subplots(len(fields), 2, figsize=(16, 3 * len(fields)), sharex=True, squeeze=False)
    for col, mode in enumerate(MODES):
        _, data = datasets[mode]
        for row, (key, column, factor, label) in enumerate(fields):
            values = data[key]
            physics = values.ndim == 4
            values = values[:, :, 0, column].reshape(-1) if physics else values[:, 0, column]
            dt = 0.0025 if physics else 0.02
            t = np.arange(len(values)) * dt + (dt if physics else 0)
            ax = axes[row, col]
            for lo, hi in segments(data["done"][:, 0]):
                stride = 8 if physics else 1
                ax.plot(t[lo * stride:hi * stride], values[lo * stride:hi * stride] * factor,
                        color=COLORS[row % 3], lw=0.85)
                if lo:
                    ax.axvline(lo * 0.02, color="0.6", ls=":", lw=0.6)
            if reference is not None:
                for value in reference[row]:
                    ax.axhline(value, color="0.4", ls="--", lw=0.7)
            decorate(ax, mode, label + " " + ylabel[row])
            ax.set_xlim(0, 60)
    save(fig, name, "env 0 in both modes · no connections across resets")


def distributions(datasets):
    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    definitions = (("scaled_raw", [1.57, 6.28], ["rad", "N m"], "Scaled raw policy demand"),
                   ("submitted_motor_torque_nm", [5.6, 6.28], ["N m", "N m"], "Submitted motor torque"),
                   ("velocity_after_rad_s", [23.9, 61.6], ["rad/s", "rad/s"], "Actual motor speed"))
    for row, (key, limits, units, title) in enumerate(definitions):
        for motor in (0, 1):
            ax = axes[row, motor]
            samples = [datasets[mode][1][key][..., motor].reshape(-1) for mode in MODES]
            bound = max(limits[motor] * 1.1, *(float(np.abs(values).max()) * 1.02 for values in samples))
            bins = np.linspace(-bound, bound, 121)
            for index, (mode, values) in enumerate(zip(MODES, samples)):
                ax.hist(values, bins=bins, histtype="step", lw=1.4, color=COLORS[index], label=TITLES[mode])
            for sign in (-1, 1):
                ax.axvline(sign * limits[motor], color="0.4", ls="--", lw=0.8)
            ax.set_yscale("log")
            ax.set_xlabel(f"{title} ({units[motor]})")
            ax.set_ylabel("Sample count (log scale)")
            ax.set_title("Steering" if motor == 0 else "Rear")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
    save(fig, "limit_distributions", "all 64 envs · demand 50 Hz; torque/speed 400 Hz · dashed: configured limits")


def observations(datasets):
    groups = (([0, 1], ["gravity_y", "gravity_z"], "Gravity (dimensionless)"),
              ([2, 3, 4], ["omega_x", "omega_y", "omega_z"], "Body angular velocity (rad/s)"),
              ([5, 6], ["steering", "rear"], "Joint position (rad)"),
              ([7, 8], ["steering", "rear"], "Joint speed (rad/s)"),
              ([9, 10], ["sin(steering)", "cos(steering)"], "Steering encoding"))
    fig, axes = plt.subplots(5, 2, figsize=(16, 14), sharex=True)
    for col, mode in enumerate(MODES):
        _, data = datasets[mode]
        for row, (indices, labels, unit) in enumerate(groups):
            ax = axes[row, col]
            for start, stop in segments(data["done"][:, 0]):
                for index, (channel, label) in enumerate(zip(indices, labels)):
                    ax.plot(np.arange(start, stop) * 0.02, data["observation"][start:stop, 0, channel],
                            color=COLORS[index], lw=0.8, label=label if start == 0 else None)
                if start:
                    ax.axvline(start * 0.02, color="0.6", ls=":", lw=0.6)
            decorate(ax, mode, unit)
            ax.legend(fontsize=7)
    save(fig, "observation_traces", "env 0 · actual noisy 11-channel policy input, captured at inference")


def torque_trace(datasets):
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=True)
    fields = ("computed_motor_torque_nm", "submitted_motor_torque_nm", "projected_joint_effort_nm")
    labels = ("PD/feedforward before motor clamp", "Submitted motor torque", "Solver joint-axis effort (not motor-only)")
    for col, mode in enumerate(MODES):
        _, data = datasets[mode]
        for motor in (0, 1):
            ax = axes[motor, col]
            for lo, hi in segments(data["done"][:, 0]):
                times = np.arange(lo * 8, hi * 8) * 0.0025
                for index, (field, label) in enumerate(zip(fields, labels)):
                    values = data[field][lo:hi, :, 0, motor].reshape(-1)
                    ax.plot(times + (0.0025 if index == 2 else 0), values, color=COLORS[index],
                            lw=1.1 if index == 1 else 0.85, ls="--" if index == 1 else "-",
                            label=label if lo == 0 else None)
                if lo:
                    ax.axvline(lo * 0.02, color="0.6", ls=":", lw=0.6)
            for sign in (-1, 1):
                ax.axhline(sign * (5.6 if motor == 0 else 6.28), color="0.4", ls=":", lw=0.7)
            decorate(ax, mode, f"{'Steering' if motor == 0 else 'Rear'} torque (N m)")
            ax.legend(fontsize=7)
            ax.set_xlim(0, 5)
    save(fig, "motor_torque_detail", "env 0 · first 5 s · passive joint losses are separate from controller Kd")


def describe(metadata, data):
    lengths, timeouts, counts, edges = episode_statistics(data)
    torque = data["submitted_motor_torque_nm"]
    target = data["held_steering_target_rad"]
    measured = data["projected_joint_effort_nm"]
    errors = data["position_after_rad"][..., 0] - target
    phase = np.zeros(data["done"].shape, dtype=int)
    for step in range(1, len(phase)):
        phase[step] = np.where(data["done"][step - 1], 0, phase[step - 1] + 1)
    ordinary = np.repeat((phase >= 25)[:, None], 8, axis=1)
    return {
        "policy_samples_per_motor": int(np.prod(data["raw_action"].shape[:2])),
        "physics_samples_per_motor": int(np.prod(torque.shape[:3])),
        "completed_episodes": len(lengths), "mean_episode_s": float(lengths.mean() * 0.02),
        "min_episode_s": float(lengths.min() * 0.02), "max_episode_s": float(lengths.max() * 0.02),
        "timeout_rate": float(timeouts.mean()), "episode_histogram_counts": counts.tolist(),
        "episode_histogram_edges_s": edges.tolist(),
        "termination_counts": dict(zip(metadata["termination_names"], data["termination_flags"].sum((0, 1)).tolist())),
        "raw_action_min": data["raw_action"].min((0, 1)).tolist(),
        "raw_action_max": data["raw_action"].max((0, 1)).tolist(),
        "raw_action_oob_count": (np.abs(data["raw_action"]) > 1.0 + 1e-6).sum((0, 1)).tolist(),
        "scaled_demand_abs_max_rad_nm": np.abs(data["scaled_raw"]).max((0, 1)).tolist(),
        "held_steering_target_abs_max_rad": float(np.abs(target).max()),
        "slew_changes_scaled_target_count": int((np.abs(target[:, 0] - data["scaled_raw"][..., 0]) > 1e-5).sum()),
        "computed_torque_abs_max_nm": np.abs(data["computed_motor_torque_nm"]).max((0, 1, 2)).tolist(),
        "submitted_torque_abs_max_nm": np.abs(torque).max((0, 1, 2)).tolist(),
        "submitted_torque_oob_count": (np.abs(torque) > np.array([5.6, 6.28]) + 1e-5).sum((0, 1, 2)).tolist(),
        "post_pd_clipping_count": (np.abs(data["computed_motor_torque_nm"]) > np.array([5.6, 6.28]) + 1e-5).sum((0, 1, 2)).tolist(),
        "solver_joint_effort_abs_max_nm": np.abs(measured).max((0, 1, 2)).tolist(),
        "solver_joint_effort_above_nominal_plus_0_01_count": (np.abs(measured) > np.array([5.6, 6.28]) + 0.01).sum((0, 1, 2)).tolist(),
        "actual_speed_abs_max_rad_s": np.abs(data["velocity_after_rad_s"]).max((0, 1, 2)).tolist(),
        "actual_speed_over_configured_plus_1e_5_count": (np.abs(data["velocity_after_rad_s"]) > np.array([23.9, 61.6]) + 1e-5).sum((0, 1, 2)).tolist(),
        "steering_tracking_rmse_rad": float(np.sqrt(np.mean(errors ** 2))),
        "steering_tracking_rmse_after_first_0_5s_each_episode_rad": float(np.sqrt(np.mean(errors[ordinary] ** 2))) if ordinary.any() else None,
        "pre_reset_physics_max_abs_roll_drift_yaw_rad_m_rad": np.abs(data["body_roll_drift_yaw"]).max((0, 1, 2)).tolist(),
        "observation_max_abs_by_channel": np.abs(data["observation"]).max((0, 1)).tolist(),
    }


def export_env(data, mode):
    columns = ["command_time_s", "response_time_s", "episode", "raw_steer", "raw_rear",
               "scaled_steer_rad", "scaled_rear_nm", "held_steer_rad", "steer_actual_rad",
               "rear_submitted_nm", "steer_submitted_nm", "rear_solver_effort_nm", "steer_solver_effort_nm",
               "steer_speed_rad_s", "rear_speed_rad_s"]
    episode = np.r_[0, np.cumsum(data["done"][:-1, 0])]
    with (HERE / f"{mode}_env0_traces.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for step in range(3000):
            for sub in range(8):
                index = (step, sub, 0)
                time = (step * 8 + sub) * 0.0025
                writer.writerow([time, time + 0.0025, episode[step], *data["raw_action"][step, 0],
                                 *data["scaled_raw"][step, 0], data["held_steering_target_rad"][index],
                                 data["position_after_rad"][index][0], *data["submitted_motor_torque_nm"][index][::-1],
                                 *data["projected_joint_effort_nm"][index][::-1], *data["velocity_after_rad_s"][index]])


def main():
    if (HERE / "summary.json").exists():
        raise FileExistsError("Preserve completed reports; use a new output directory for another experiment")
    datasets = {mode: load(mode) for mode in MODES}
    assert datasets["no_dr"][0]["checkpoint_sha256"] == datasets["dr"][0]["checkpoint_sha256"]
    summary = {mode: describe(*datasets[mode]) for mode in MODES}
    envs = dict.fromkeys(MODES, 0)
    full = traces(datasets, "isaac_demand_vs_actuation", (0, 60), envs, fit_y=True)
    zoom = traces(datasets, "isaac_demand_vs_actuation_zoomed", (0, 60), envs, fit_y=True, nominal_zoom=True)
    traces(datasets, "isaac_startup_demand_vs_actuation", (0, 0.6), envs)
    traces(datasets, "isaac_tracking_detail", (0.5, 5), envs)
    peak_envs = {mode: int(np.unravel_index(np.abs(data["scaled_raw"][:30, :, 1]).argmax(), (30, 64))[1])
                 for mode, (_, data) in datasets.items()}
    traces(datasets, "isaac_peak_startup_demand_vs_actuation", (0, 0.6), peak_envs)
    simple_traces(datasets, "raw_policy_output", [("raw_action", 0, 1, "Steering"), ("raw_action", 1, 1, "Rear")],
                  ["normalized action", "normalized action"], [[-1, 1], [-1, 1]])
    simple_traces(datasets, "balance_motion", [("body_roll_drift_yaw", 0, 180 / np.pi, "Roll"),
                  ("body_roll_drift_yaw", 1, 1, "Drift"), ("body_roll_drift_yaw", 2, 180 / np.pi, "Yaw vs spawn")],
                  ["(deg)", "(m)", "(deg)"], [[-30, 30], [0.5], [-30, 30]])
    distributions(datasets)
    observations(datasets)
    torque_trace(datasets)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, mode in zip(axes, MODES):
        values = summary[mode]
        ax.bar(np.arange(0, 30, 3), values["episode_histogram_counts"], width=2.8, align="edge", color=COLORS[0])
        ax.set_xticks(np.arange(0, 31, 3))
        ax.set_title(TITLES[mode] + f" · {values['completed_episodes']} completed episodes")
        ax.set_xlabel("Episode length (s; final bin includes 30 s)")
        ax.set_ylabel("Completed episode count")
        ax.grid(axis="y", alpha=0.2)
        for x, count in zip(np.arange(1.5, 30, 3), values["episode_histogram_counts"]):
            if count:
                ax.text(x, count, str(count), ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, max(max(summary[m]["episode_histogram_counts"]) for m in MODES) * 1.2 + 1)
    save(fig, "episode_length_histogram", "10 bins × 3 s · unfinished episodes at capture end are excluded")
    for mode, (_, data) in datasets.items():
        export_env(data, mode)
    summary["peak_startup_environments"] = peak_envs
    summary["verification_passed"] = True
    summary["input_sha256"] = {
        str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest()
        for mode in MODES for path in (HERE / mode / "motor_samples.npz", HERE / mode / "experiment.json")
    }
    summary["figures"] = [name for name, _ in FIGURES]
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save_demand_settings(full, zoom, summary["input_sha256"])
    rows = "".join(f"<tr><td>{label}</td><td>{summary['no_dr'][key]}</td><td>{summary['dr'][key]}</td></tr>"
                   for key, label in (("completed_episodes", "Completed episodes"), ("mean_episode_s", "Mean episode (s)"),
                                      ("timeout_rate", "Timeout fraction")))
    images = "".join(f'<section><h2>{html.escape(name.replace("_", " "))}</h2><p>{html.escape(caption)}</p>'
                     f'<a href="{name}.svg"><img loading="lazy" src="{name}.png" alt="{html.escape(name)}"></a>'
                     f'<p><a href="{name}.pdf">PDF</a> · <a href="{name}.svg">SVG</a></p></section>'
                     for name, caption in FIGURES)
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>75300 motor analysis</title>'
            '<style>body{max-width:1500px;margin:2em auto;padding:0 1em;font:16px system-ui;background:#f8fafc}'
            'img{width:100%;background:white}td,th{padding:.5em;border:1px solid #ccd}table{border-collapse:collapse}'
            'section{margin:3em 0}a{color:#225e99}</style><h1>model_75300 · v1 Beta policy</h1>'
            '<p>64 environments × 3000 steps, seed 42. Left: training-matched no DR. Right: full-DR stress test.</p>'
            '<p>Orange steering is the held target entering explicit PD; orange rear is submitted motor torque. '
            'Green rear is solver joint-axis effort, not isolated motor-only torque. No smoothing or time shifting.</p>'
            '<p><a href="README.md">Methods/report</a> · <a href="summary.json">Numerical summary</a></p>'
            '<table><tr><th>Metric</th><th>No DR</th><th>Full DR</th></tr>' + rows + '</table>' + images + '</html>')
    (HERE / "index.html").write_text(page)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redraw-demand", action="store_true", help="Redraw only demand figures from saved captures")
    args = parser.parse_args()
    if args.redraw_demand:
        redraw_demand()
    else:
        main()
