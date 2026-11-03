"""Plot the completed v2 MuJoCo capture without rerunning either simulator."""

import csv
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/trackstand_v2_trace_plots_mpl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = HERE.parent / "20261103_v2_parallel_backend/validation/no_dr/202611032149_26600_no_dr"
INPUTS = [Path(str(BASE) + suffix) for suffix in (
    "_evaluation.npz", "_metadata.json", "_summary.json", "_summary.log",
)]
COLORS = ("#2367ad", "#e77c16", "#208543", "#8a60a1")
OBSERVATIONS = (
    ("gravity_y", "dimensionless"), ("gravity_z", "dimensionless"),
    ("angular_velocity_x", "rad/s"), ("angular_velocity_y", "rad/s"),
    ("angular_velocity_z", "rad/s"), ("steering_position", "rad"),
    ("rear_position_native_branch", "rad"), ("steering_velocity", "rad/s"),
    ("rear_velocity", "rad/s"), ("sin_steering", "dimensionless"), ("cos_steering", "dimensionless"),
)
WINDOWS = {"full": (0.0, 30.0), "startup": (0.0, 2.0), "detail": (12.5, 17.5)}
plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_and_verify():
    with np.load(INPUTS[0], allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    meta = json.loads(INPUTS[1].read_text())
    native = json.loads(INPUTS[2].read_text())
    assert meta["task"] == "Trackstand-Balance-v2" and meta["completed"]
    assert meta["arguments"]["dr"] is False and not meta["arguments"]["pure_survival"]
    assert meta["arguments"]["seed"] == 42 and meta["completed_policy_steps"] == 1500
    assert meta["environment_transitions"] == 96000 and meta["episode_quotas"] == [1] * 64
    assert meta["completed_per_env"] == [1] * 64 and len(meta["episodes"]) == 64
    assert (meta["policy_hz"], meta["physics_hz"], meta["pd_hz"], meta["integral_hz"]) == (50, 1200, 1200, 200)
    assert all(item["length"] == 1500 and item["timeout"] and item["sampled_dr"] is None
               for item in meta["episodes"])
    assert data["observation"].shape == (1500, 64, 11) and data["raw_action"].shape == (1500, 64, 2)
    assert data["physics_motor_torque_nm"].shape == (1500, 64, 24, 2)
    assert np.all(data["active_mask"]) and data["active_mask"].sum() == 96000
    assert data["motor_order"].tolist() == ["steering", "rear"]
    assert np.isfinite(np.concatenate([data[k].ravel() for k in (
        "raw_action", "observation", "position_rad", "velocity_rad_s", "physics_motor_torque_nm",
        "physics_computed_torque_nm", "physics_drive_integral_nm", "physics_drive_error_rad_s",
    )])).all()
    assert np.max(np.abs(data["raw_action"])) <= 1
    np.testing.assert_array_equal(data["reset_events"][:, 0], [1500] * 64)
    np.testing.assert_array_equal(np.sort(data["reset_events"][:, 1]), np.arange(64))
    assert np.all(data["termination_flags"][:-1] == 0)
    np.testing.assert_array_equal(data["termination_flags"][-1, :, 0], [1] * 64)
    assert np.all(data["termination_flags"][..., 1:] == 0)
    data["scaled_raw"] = data["raw_action"] * np.array([1.57, 61.6], dtype=np.float32)
    np.testing.assert_array_equal(data["held_drive_target_rad_s"], data["scaled_raw"][..., 1])
    previous = np.zeros(64)
    for step in range(1500):
        expected = previous + np.clip(np.clip(data["scaled_raw"][step, :, 0], -1.57, 1.57) - previous, -.478, .478)
        np.testing.assert_allclose(data["held_steering_target_rad"][step], expected, rtol=0.0, atol=1e-12)
        previous = data["held_steering_target_rad"][step]
    error = data["held_drive_target_rad_s"][:, :, None] - data["physics_drive_velocity_rad_s"]
    np.testing.assert_allclose(data["physics_drive_error_rad_s"], error, rtol=0.0, atol=0.0)
    rear_calculation = .30 * error + data["physics_drive_integral_nm"]
    np.testing.assert_allclose(data["physics_computed_torque_nm"][..., 1], rear_calculation, rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(data["physics_motor_torque_nm"],
                                  np.clip(data["physics_computed_torque_nm"], [-5.6, -6.28], [5.6, 6.28]))
    # Independently reproduce all retained I states from recorded errors, including hold intervals.
    errors = error.transpose(0, 2, 1).reshape(-1, 64)
    recorded_i = data["physics_drive_integral_nm"].transpose(0, 2, 1).reshape(-1, 64)
    previous_error, stored_error, torque = np.zeros((3, 64))
    maximum_i_error = 0.0
    for tick in range(36000):
        if tick % 6 == 0:
            increment = .5 * (errors[tick] + previous_error) * .005 if tick else 0.0
            candidate = np.clip(3.3 * (stored_error + increment), -6.28, 6.28)
            delta = candidate - torque
            total = .30 * errors[tick] + candidate
            freeze = ((total > 6.28) & (delta > 0)) | ((total < -6.28) & (delta < 0))
            torque = np.where(freeze, torque, candidate)
            stored_error, previous_error = torque / 3.3, errors[tick]
        maximum_i_error = max(maximum_i_error, float(np.max(np.abs(recorded_i[tick] - torque))))
    assert maximum_i_error < 1e-12
    # Input joint channels are uncorrupted, while gravity/gyro retain their actual noisy values.
    position_before = np.concatenate([np.zeros((1, 64, 2)), data["position_rad"][:-1]])
    native_rear = 2 * np.arctan2(np.sin(position_before[..., 1] / 2), np.cos(position_before[..., 1] / 2))
    for channel, values in ((5, position_before[..., 0]), (6, native_rear),
                            (8, data["physics_drive_velocity_rad_s"][..., 0]),
                            (9, np.sin(position_before[..., 0])), (10, np.cos(position_before[..., 0]))):
        np.testing.assert_allclose(data["observation"][..., channel], values, rtol=2e-6, atol=2e-7)
    assert sha(meta["policy"]) == meta["policy_sha256"]
    return data, meta, native, dict(maximum_reconstructed_integral_error_nm=maximum_i_error,
                                   policy_samples=96000, physics_samples=2304000,
                                   unique_recorded_episodes=64, completed_timeouts=64,
                                   actual_sampled_spawn_times_s=[0.0],
                                   terminal_reset_s_without_following_samples=30.0)


def policy_times():
    return np.arange(1500) * .02


def physics_times():
    return (np.arange(1500)[:, None] * .02 + np.arange(24)[None, :] / 1200).reshape(-1)


def series(data, kind, row, env):
    tp, th = policy_times(), physics_times()
    response = tp + .02
    def item(t, y, label, color, held=False):
        if held:
            t, y = np.r_[t, t[-1] + (1 / 1200 if len(t) == 36000 else .02)], np.r_[y, y[-1]]
        return dict(t=t, y=y, label=label, color=color, held=held)
    if kind == "demand":
        if row == 0:
            return [item(tp, data["scaled_raw"][:, env, 0], "Scaled raw steering", COLORS[0], True),
                    item(tp, data["held_steering_target_rad"][:, env], "Held steering request", COLORS[1], True),
                    item(np.r_[0., response], np.r_[data["observation"][0, env, 5], data["position_rad"][:, env, 0]],
                         "Actual position (50 Hz boundaries)", COLORS[2])]
        velocity = data["physics_drive_velocity_rad_s"][:, env].reshape(-1)
        return [item(tp, data["scaled_raw"][:, env, 1], "Scaled raw rear velocity", COLORS[0], True),
                item(tp, data["held_drive_target_rad_s"][:, env], "Held rear velocity request", COLORS[1], True),
                item(np.r_[th, 30.], np.r_[velocity, data["velocity_rad_s"][-1, env, 1]],
                     "Actual rear speed (1200 Hz feedback)", COLORS[2])]
    if kind == "torque":
        computed = data["physics_computed_torque_nm"][:, env, :, row].reshape(-1)
        actual = data["physics_motor_torque_nm"][:, env, :, row].reshape(-1)
        result = []
        if row == 1:
            result += [item(th, .30 * data["physics_drive_error_rad_s"][:, env].reshape(-1),
                            "Velocity P torque", COLORS[0], True),
                       item(th, data["physics_drive_integral_nm"][:, env].reshape(-1),
                            "Retained I torque", COLORS[3], True)]
        return result + [item(th, computed, "Computed before final clamp", COLORS[1], True),
                         item(th + 1 / 1200, actual, "Applied motor actuator torque", COLORS[2])]
    if kind == "observation":
        return [item(tp, data["observation"][:, env, row], OBSERVATIONS[row][0], COLORS[0])]
    if kind == "raw":
        return [item(tp, data["raw_action"][:, env, row], "Steering action" if row == 0 else "Rear velocity action",
                     COLORS[row * 2], True)]
    if kind == "speed":
        if row == 0:
            return [item(response, data["velocity_rad_s"][:, env, 0], "Steering speed (50 Hz boundaries)", COLORS[2])]
        return [item(th, data["physics_drive_velocity_rad_s"][:, env].reshape(-1),
                     "Rear speed (1200 Hz feedback)", COLORS[2])]
    raise ValueError(kind)


def draw_panel(ax, traces, window, rasterized=False):
    values, extrema = [], {}
    for trace in traces:
        selected = (trace["t"] >= window[0] - 1e-10) & (trace["t"] <= window[1] + 1e-10)
        t, y = trace["t"][selected], trace["y"][selected]
        lo, hi = float(y.min()), float(y.max())
        label = f"{trace['label']}  [min {lo:.5g}, max {hi:.5g}]"
        style = dict(color=trace["color"], lw=1.15, alpha=.88, label=label, rasterized=rasterized)
        if trace["held"]:
            ax.step(t, y, where="post", ls="--" if trace["color"] == COLORS[1] else "-", **style)
        else:
            ax.plot(t, y, **style)
        for index in (int(y.argmin()), int(y.argmax())):
            ax.plot(t[index], y[index], "o", color=trace["color"], ms=3.0, zorder=5, rasterized=rasterized)
        extrema[trace["label"]] = dict(minimum=lo, maximum=hi, minimum_time_s=float(t[y.argmin()]),
                                       maximum_time_s=float(t[y.argmax()]), plotted_samples=len(y))
        values.append(y)
    low, high = min(x.min() for x in values), max(x.max() for x in values)
    pad = max(float(high - low) * .09, max(abs(float(low)), abs(float(high))) * 1e-5, 1e-7)
    ax.set_ylim(low - pad, high + pad)
    spawns = [0.0] if window[0] <= 0 <= window[1] else []
    for time in spawns:
        ax.axvline(time, color="#d62728", lw=1.05, alpha=.8, clip_on=False, label="Initial spawn", zorder=4)
    ax.set_xlim(*window)
    ax.grid(alpha=.18)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.tick_params(labelsize=8.5)
    ax.legend(loc="lower right", fontsize=7.0, framealpha=.86, borderpad=.35)
    ax.text(.01, .025, f"Panel min {low:.6g} · max {high:.6g}", transform=ax.transAxes, fontsize=7.5,
            va="bottom", bbox=dict(facecolor="white", edgecolor="none", alpha=.8))
    return dict(window_s=list(window), ylim=[float(low - pad), float(high + pad)],
                panel_minimum=float(low), panel_maximum=float(high), extrema=extrema,
                spawn_times_s=spawns, samples_outside_y_view=0, legend_location="lower right")


def figure(data, kind, env=0, window=(0., 30.), rasterized=False):
    rows = 11 if kind == "observation" else 2
    fig, axes = plt.subplots(rows, 1, figsize=(13.5, 26 if rows == 11 else 9), squeeze=False)
    titles = {"demand": ["Steering position: policy demand → held request → motion",
                         "Rear velocity: policy demand → held request → motion"],
              "torque": ["Steering controller demand and applied torque",
                         "Rear velocity P + retained I and applied torque"],
              "raw": ["Normalized steering Beta mean", "Normalized rear velocity Beta mean"],
              "speed": ["Actual steering speed", "Actual rear speed"]}
    settings = []
    for row, ax in enumerate(axes[:, 0]):
        settings.append(draw_panel(ax, series(data, kind, row, env), window, rasterized))
        if kind == "observation":
            ax.set_title(f"Observation {row + 1}: {OBSERVATIONS[row][0]}", fontsize=10, loc="left")
            ax.set_ylabel(OBSERVATIONS[row][1])
        else:
            ax.set_title(titles[kind][row], fontsize=11, loc="left")
            ax.set_ylabel("rad" if kind == "demand" and row == 0 else "rad/s" if kind in ("demand", "speed")
                          else "N m" if kind == "torque" else "dimensionless")
    axes[-1, 0].set_xlabel("Elapsed simulation time from initial spawn (s)")
    fig.suptitle(f"Trackstand-Balance-v2 · model_26600 · no DR · MuJoCo · env {env}/63\n"
                 f"{window[0]:g}–{window[1]:g} s · seed 42 · policy 50 Hz · physics/PD 1200 Hz · rear I 200 Hz",
                 fontsize=14, y=.985)
    footer = ("Nominal fixed reset; ordinary IMU white noise retained. Red: initial spawn at 0 s. "
              "No second episode has recorded samples.\n"
              "Steering position/speed: 50 Hz boundaries; rear feedback/torques: 1200 Hz. "
              "Applied motor torque excludes passive friction/contact joint effort.\n"
              "Fixed gains: steering 10 / 0.2; rear velocity P 0.30, Ki 3.3. "
              "Nominal tyre friction 0.65; rear frictionloss 0.106 N m, passive B 0.")
    fig.text(.07, .012, footer, fontsize=8.5, va="bottom")
    fig.subplots_adjust(top=.93 if rows == 11 else .89, bottom=.075 if rows == 11 else .15,
                        hspace=.40 if rows == 11 else .36, left=.09, right=.98)
    return fig, settings


def export_csv(data):
    tp, th = policy_times(), physics_times()
    policy_fields = ["policy_time_s", "response_time_s", "raw_steering", "raw_drive_velocity",
                     "scaled_raw_steering_rad", "scaled_raw_drive_velocity_rad_s", "held_steering_target_rad",
                     "held_drive_target_rad_s", "actual_steering_position_rad", "actual_rear_position_rad",
                     "actual_steering_velocity_rad_s", "actual_rear_velocity_rad_s"]
    policy_fields += ["obs_" + name for name, _ in OBSERVATIONS]
    table = np.column_stack([tp, tp + .02, data["raw_action"][:, 0], data["scaled_raw"][:, 0],
                             data["held_steering_target_rad"][:, 0], data["held_drive_target_rad_s"][:, 0],
                             data["position_rad"][:, 0], data["velocity_rad_s"][:, 0], data["observation"][:, 0]])
    np.savetxt(HERE / "env0_policy_traces.csv", table, delimiter=",", header=",".join(policy_fields), comments="", fmt="%.17g")
    error = data["physics_drive_error_rad_s"][:, 0].reshape(-1)
    table = np.column_stack([th, th + 1 / 1200, data["physics_drive_velocity_rad_s"][:, 0].reshape(-1), error,
                             .3 * error, data["physics_drive_integral_nm"][:, 0].reshape(-1),
                             data["physics_computed_torque_nm"][:, 0].reshape(-1, 2),
                             data["physics_motor_torque_nm"][:, 0].reshape(-1, 2)])
    fields = ["controller_time_s", "physics_end_time_s", "rear_velocity_feedback_rad_s", "rear_velocity_error_rad_s",
              "rear_P_torque_nm", "rear_I_torque_nm", "steering_preclip_torque_nm", "rear_preclip_torque_nm",
              "steering_applied_torque_nm", "rear_applied_torque_nm"]
    np.savetxt(HERE / "env0_physics_traces.csv", table, delimiter=",", header=",".join(fields), comments="", fmt="%.17g")
    rows = []
    for env in range(64):
        steer_error = data["held_steering_target_rad"][:, env] - data["position_rad"][:, env, 0]
        drive_error = data["held_drive_target_rad_s"][:, env] - data["velocity_rad_s"][:, env, 1]
        computed = data["physics_computed_torque_nm"][:, env]
        rows.append(dict(env_id=env, episode_length_s=30.0, time_out=True,
                         steering_end_interval_rmse_rad=float(np.sqrt(np.mean(steer_error ** 2))),
                         drive_end_interval_rmse_rad_s=float(np.sqrt(np.mean(drive_error ** 2))),
                         max_abs_steering_torque_nm=float(abs(computed[..., 0]).max()),
                         max_abs_rear_torque_nm=float(abs(computed[..., 1]).max()),
                         steering_torque_clip_fraction=float(np.mean(abs(computed[..., 0]) > 5.6)),
                         rear_torque_clip_fraction=float(np.mean(abs(computed[..., 1]) > 6.28))))
    with (HERE / "all64_policy_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    obs_rows = [dict(env_id=env, observation_index=index + 1, observation=name, unit=unit,
                     minimum=float(data["observation"][:, env, index].min()),
                     maximum=float(data["observation"][:, env, index].max()))
                for env in range(64) for index, (name, unit) in enumerate(OBSERVATIONS)]
    with (HERE / "observation_extrema_all64.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(obs_rows[0]))
        writer.writeheader(); writer.writerows(obs_rows)
    return rows


def all_env_envelope(data):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    signals = [data["position_rad"][..., 0], data["velocity_rad_s"][..., 1],
               data["physics_motor_torque_nm"][..., 0].transpose(0, 2, 1).reshape(-1, 64),
               data["physics_motor_torque_nm"][..., 1].transpose(0, 2, 1).reshape(-1, 64)]
    names = ["Steering position", "Rear velocity", "Steering applied motor torque", "Rear applied motor torque"]
    units = ["rad", "rad/s", "N m", "N m"]
    settings = []
    for index, (ax, values, label, unit) in enumerate(zip(axes.flat, signals, names, units)):
        times = policy_times() + .02 if index < 2 else physics_times() + 1 / 1200
        low, high = values.min(axis=1), values.max(axis=1)
        ax.fill_between(times, low, high, color=COLORS[0], alpha=.17,
                        label=f"All 64 range [{values.min():.5g}, {values.max():.5g}]")
        panel = draw_panel(ax, [dict(t=times, y=values[:, 0], label="Env 0", color=COLORS[2], held=False)], (0., 30.))
        bottom, top = float(values.min()), float(values.max())
        pad = max((top - bottom) * .09, 1e-7)
        ax.set_ylim(bottom - pad, top + pad)
        for label_artist in list(ax.texts):
            label_artist.remove()
        ax.text(.01, .025, f"All 64 min {bottom:.6g} · max {top:.6g}", transform=ax.transAxes, fontsize=7.5,
                va="bottom", bbox=dict(facecolor="white", edgecolor="none", alpha=.8))
        ax.set_title(label); ax.set_ylabel(unit); ax.set_xlabel("Elapsed simulation time (s)")
        panel.update(ylim=[bottom - pad, top + pad], panel_minimum=bottom, panel_maximum=top,
                     all64_minimum=bottom, all64_maximum=top)
        settings.append(panel)
    fig.suptitle("MuJoCo v2 · model_26600 · no DR · seed 42 · all 64 successful 30 s episodes\n"
                 "Actual recorded state/actuator torque: env 0 plus across-environment min–max envelope", fontsize=14)
    fig.tight_layout(rect=(0, .02, 1, .92))
    return fig, settings


def main():
    HERE.mkdir(exist_ok=True)
    protected_before = {str(path.relative_to(ROOT)): sha(path) for path in INPUTS}
    data, meta, native, checks = load_and_verify()
    rows = export_csv(data)
    plot_settings, figures = {}, []
    plans = [(kind, window) for kind in ("demand", "torque", "observation") for window in WINDOWS]
    plans += [("raw", "full"), ("speed", "full")]
    stems = {"demand": "mujoco_demand_vs_actuation", "torque": "mujoco_motor_torque_flow",
             "observation": "observation_traces", "raw": "raw_policy_outputs", "speed": "motor_speeds"}
    for kind, window_name in plans:
        stem = stems[kind] + ("" if window_name == "full" else "_" + window_name)
        fig, settings = figure(data, kind, window=WINDOWS[window_name])
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(HERE / f"{stem}.{suffix}", dpi=150)
        plt.close(fig)
        plot_settings[stem] = dict(kind=kind, env_id=0, panels=settings)
        figures.append(stem)
        print("Saved", stem, flush=True)
    fig, settings = all_env_envelope(data)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(HERE / f"all64_response_envelope.{suffix}", dpi=150)
    plt.close(fig)
    plot_settings["all64_response_envelope"] = dict(kind="envelope", env_id=0, panels=settings)
    figures.append("all64_response_envelope")
    for kind, stem in (("demand", "all64_demand_vs_actuation"), ("observation", "all64_observation_traces")):
        pages = []
        with PdfPages(HERE / f"{stem}.pdf") as document:
            for env in range(64):
                fig, settings = figure(data, kind, env=env, rasterized=True)
                document.savefig(fig, dpi=110); plt.close(fig)
                pages.append(dict(env_id=env, panels=settings))
            assert document.get_pagecount() == 64
        plot_settings[stem] = dict(kind=kind, pages=pages)
        print("Saved 64-page", stem, flush=True)
    summary = dict(conditions={key: meta[key] for key in ("task", "policy", "policy_sha256", "sample_design",
                                                         "policy_hz", "physics_hz", "pd_hz", "integral_hz")},
                   seed=42, dr=False, native_evaluation_summary=native, capture_checks=checks,
                   env0_end_interval_tracking=rows[0], figures=figures,
                   source_inputs_sha256=protected_before,
                   plotting_does_not_run_simulator=True,
                   unavailable=["physics-rate steering joint position/velocity", "solver net joint-axis effort",
                                "body pose/velocity histories beyond the eleven recorded policy observations"],
                   time_semantics={"policy_inputs_actions": "20 ms interval starts",
                                   "steering_motion": "Recorded 50-Hz interval-end states, plus measured initial pose",
                                   "rear_motion": "Recorded1200-Hz pre-physics feedback, final50-Hz endpoint appended",
                                   "torque": "Computed at physics start; actuator-force snapshot recorded after mj_step",
                                   "no_latency_inference": "Substep timestamp offsets are recording boundaries, not measured motor delay"})
    write_json("summary.json", summary)
    write_json("plot_settings.json", plot_settings)
    write_json("nominal_configuration.json", dict(dr=False, sampled_dr_per_episode=[None] * 64,
        initial_spawn_s=0.0, initial_roll_rad=0.0, initial_body_rates_rad_s=[0.0] * 3,
        steering_position_and_rear_position_rad=0.0, wheel_zero_references_reset=True,
        controller_gains=dict(steer_kp_nm_rad=10.0, steer_kd_nm_s_rad=0.2,
                              rear_position_kp=0.0, rear_velocity_p_nm_s_rad=0.3, rear_ki_nm_rad=3.3),
        passive=dict(steering_coulomb_nm=0.046036056104, steering_viscous_nm_s_rad=0.000910713492,
                     rear_coulomb_nm=0.106, rear_viscous_nm_s_rad=0.0),
        tyre_ground_friction=0.65, gyro_residual_rad_s=[0.0] * 3, gravity_yz_residual=[0.0] * 2,
        ordinary_observation_noise_std=[.001, .001, .0012, .0012, .0012, 0., 0., 0., 0., 0., 0.]))
    description = """# V2 model_26600: no-DR MuJoCo demand, actuation and observations

Open [the HTML gallery](index.html). This report plots the existing completed
capture; no simulation, policy retraining or production-code edit was performed.

**64 environments, seed 42, one completed 30-second episode per environment,
64/64 timeouts, all six normal terminations enabled.** Physics/PD 1200 Hz,
policy/observations/targets 50 Hz, external rear integral 200 Hz. Ordinary IMU white
noise remains enabled; no domain randomization was sampled.

## Figures and exports

- `mujoco_demand_vs_actuation`: steering raw ×1.57 rad / held request / actual
  steering position; rear raw ×61.6 rad/s / held velocity request / actual rear speed.
- `mujoco_motor_torque_flow`: steering computed/applied torque; rear P, retained I,
  P+I before clamp, and applied motor torque. **Applied motor torque is not solver
  net joint effort after passive friction/contact.** The force trace is recorded
  after each physics step; the computed trace is recorded at its start.
- `observation_traces`: all eleven actual vectors passed into policy inference,
  with the recorded noisy gravity/gyro channels. No linear-velocity channels.
- `raw_policy_outputs`, `motor_speeds`, `all64_response_envelope`: bounded actions,
  available motor-speed samples and env 0 versus the min–max range over all 64 envs.

Main figures have PNG/SVG/PDF versions. The first three have a **full 0–30 s** view,
**startup 0–2 s** view and **settling detail 12.5–17.5 s** view. No smoothing or numeric
downsampling. Independent Y axes include every displayed sample with 9% padding.
Legends are bottom right and give each curve's min/max, with colored extrema dots.
Red vertical lines mark the actual initial spawn at 0 s. Terminal resets at 30 s have
no following recorded trajectory, so no extra episode is invented or plotted.

[64-page demand PDF](all64_demand_vs_actuation.pdf) and
[64-page observation PDF](all64_observation_traces.pdf): page 1 = env 0, page 64 = env 63.
Curves in these large PDFs are rasterized; numeric records remain unchanged.

Numeric exports: [env0 policy](env0_policy_traces.csv),
[env0 physics](env0_physics_traces.csv), [all64 summaries](all64_policy_summary.csv),
[all observation extrema](observation_extrema_all64.csv),
[exact axes/extrema/time windows](plot_settings.json),
[nominal/no-DR values](nominal_configuration.json) and [summary](summary.json).

## Sampling and evidence limits

Steering position/speed are recorded only at 50 Hz policy boundaries. The initial
steering pose is the actual first uncorrupted encoder-style observation. This
capture cannot reveal complete 1200 Hz steering-state oscillations. Rear velocity
is available before every physics step at 1200 Hz; its final 30 s endpoint is also
recorded. There are no recorded solver net-effort or full body-pose histories.
Timestamp offsets label capture boundaries; they do not measure real motor delay.

Rear actions mean velocity, not torque. Rear P is 0.30 × velocity error; I is the
actual retained 200 Hz integral state, with Ki 3.3. P+I is clamped at ±6.28 N m; the
steering PD torque is clamped at ±5.6 N m. Passive friction is separate. These
records never reach either torque clamp. MuJoCo speed projection remains a
documented approximation to PhysX's soft speed constraints. No claim of hardware
performance or of survival beyond 30 s follows from these plots.

`analyze.py` checks the original shapes/counts, action scaling, 50 Hz steering slew,
all 2,304,000 rear P+I calculations and 200 Hz retained integral states, final torque
clipping and uncorrupted joint observation channels before plotting. The report's
separate `verify_plots.py` rechecks CSV values, panel bounds/extrema, PDF pages,
links and protected input hashes. Original raw data and prior reports remain
unchanged. Reproduce with `python3 -B analyze.py` from this directory.
"""
    (HERE / "README.md").write_text(description)
    cards = []
    for stem in figures:
        cards.append(f'<section><h2>{html.escape(stem.replace("_", " "))}</h2>'
                     f'<a href="{stem}.png"><img loading="lazy" src="{stem}.png" alt="{stem}"></a>'
                     f'<p><a href="{stem}.svg">SVG</a> · <a href="{stem}.pdf">PDF</a></p></section>')
    (HERE / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>V2 MuJoCo26600 no-DR traces</title>'
        '<style>body{font:16px system-ui;margin:2rem auto;max-width:1450px;padding:0 1rem;color:#182532}'
        'img{width:100%;height:auto}section{margin:2rem 0}a{color:#2367ad}</style>'
        '<h1>Trackstand-Balance-v2 · model_26600 · no DR · MuJoCo</h1>'
        '<p>Existing 64-env capture; seed 42; policy 50 Hz; physics/PD 1200 Hz; rear I 200 Hz. All 64 episodes reached 30 s.</p>'
        '<p>Rear scaled action is velocity (rad/s). Applied actuator torque is not net joint effort after passive losses.</p>'
        '<p><a href="README.md">Report and sampling limits</a> · '
        '<a href="all64_demand_vs_actuation.pdf">All 64 demand pages</a> · '
        '<a href="all64_observation_traces.pdf">All 64 observation pages</a></p>' + ''.join(cards))
    assert protected_before == {str(path.relative_to(ROOT)): sha(path) for path in INPUTS}
    write_json("provenance.json", dict(input_sha256=protected_before,
        repository_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        command="python3 -B analyze.py", cwd=str(HERE), numpy_version=np.__version__,
        matplotlib_version=matplotlib.__version__, scope="Offline plots of existing raw capture; no simulator run",
        earlier_style_reference="isaaclab_train/reports/20261025_2500_dr_comparison/analyze.py"))
    write_json("artifact_manifest.json", {str(path.relative_to(HERE)): sha(path)
        for path in sorted(HERE.rglob("*")) if path.is_file() and path.name != "artifact_manifest.json"})
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
