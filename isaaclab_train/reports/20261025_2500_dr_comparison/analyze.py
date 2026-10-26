"""Validate real captures and plot 2500 DR demand/response and all observations.

Signals retain their original samples, SI units and time offsets. Red lines are
actual initial/reset spawns. Full-range axes include every plotted extremum;
separate five-second views reveal changes without hiding the full recordings.
"""

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import dr_annotations

TAGS = ("isaac400", "mujoco400", "mujoco800")
TITLES = ("Isaac Lab · physics/PD 400 Hz", "MuJoCo · physics/PD 400 Hz", "MuJoCo · physics/PD 800 Hz")
COLORS = ("#2367ad", "#e77c16", "#208543", "#8a60a1")
SCALES = np.array([1.57, 6.28], dtype=np.float32)
OBSERVATIONS = (("gravity_y", "dimensionless"), ("gravity_z", "dimensionless"),
                ("angular_velocity_x", "rad/s"), ("angular_velocity_y", "rad/s"),
                ("angular_velocity_z", "rad/s"), ("steering_position", "rad"),
                ("rear_position", "rad"), ("steering_velocity", "rad/s"),
                ("rear_velocity", "rad/s"), ("sin_steering", "dimensionless"),
                ("cos_steering", "dimensionless"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def segments(done):
    edges = np.unique(np.r_[0, np.flatnonzero(done) + 1, len(done)])
    return list(zip(edges[:-1], edges[1:]))


def spawn_times(done, dt=.02):
    return np.r_[0, np.flatnonzero(done) + 1] * dt


def detail_window(done, dt=.02, seconds=5):
    """Central <=5 seconds of the longest COMPLETED episode; never cross reset."""
    completed = [(lo, hi) for lo, hi in segments(done) if done[hi - 1]]
    choices = completed or segments(done)
    lo, hi = max(choices, key=lambda pair: pair[1] - pair[0])
    width = min(round(seconds / dt), hi - lo)
    first = lo + (hi - lo - width) // 2
    return first * dt, (first + width) * dt


def load(tag):
    folder = HERE / tag
    meta = json.loads((folder / "experiment.json").read_text())
    assert meta["capture_completed"] and meta["training_dr"] and meta["dr"]
    assert not meta["pure_survival"] and meta["seed"] == 42
    assert meta["num_envs"] == 64 and meta["completed_policy_steps"] == 3000
    assert meta["policy_dt_s"] == .02 and meta["decimation"] in (8, 16)
    files = [folder / "experiment.json"]
    if tag == "isaac400":
        files.append(folder / "motor_samples.npz")
        data = archive(files[-1])
        assert all(json.loads((folder / "configuration_checks.json").read_text()).values())
        integrity = json.loads((folder / "source_integrity.json").read_text())
        assert integrity["evaluation_completed"] and not integrity["changed_files"]
    else:
        native = next(folder.glob("*_evaluation.npz"))
        files.extend([native, folder / "physics_samples.npz"])
        data = archive(native)
        physics = archive(files[-1])
        # Preserve the differently sampled production values for independent checks.
        data["policy_boundary_position_rad"] = data["position_rad"]
        data["policy_boundary_velocity_rad_s"] = data["velocity_rad_s"]
        data["done"] = np.any(data["termination_flags"], axis=-1)
        data["truncated"] = data["termination_flags"][..., 0]
        np.testing.assert_allclose(physics["held_steering_target_rad"][:, 0],
                                   data["held_steering_target_rad"], rtol=0, atol=0)
        np.testing.assert_array_equal(physics["actual_motor_torque_nm"].transpose(0, 2, 1, 3),
                                      data["physics_motor_torque_nm"])
        data.update(physics)
        old_folder = ROOT / "mujoco_sim2sim/reports/20261023_2500_dr_evaluation"
        old_folder /= "current400_full" if tag == "mujoco400" else "diagnostic800_full"
        with np.load(next(old_folder.glob("*_evaluation.npz"))) as old, np.load(native) as new:
            for key in old.files:
                a, b = old[key], new[key]
                assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), key
        meta["all_13_original_mujoco_arrays_byte_identical"] = True
    data["scaled_raw"] = data["raw_action"] * SCALES
    validate(meta, data, isaac=tag == "isaac400")
    meta["input_sha256"] = {str(path.relative_to(HERE)): sha(path) for path in files}
    return meta, data


def validate(meta, data, isaac):
    t, n, s = meta["completed_policy_steps"], meta["num_envs"], meta["decimation"]
    assert data["raw_action"].shape == (t, n, 2) and data["observation"].shape == (t, n, 11)
    for key in ("position_before_rad", "velocity_before_rad_s", "position_after_rad",
                "velocity_after_rad_s", "computed_motor_torque_nm", "submitted_motor_torque_nm",
                "controller_kp", "controller_kd"):
        assert data[key].shape == (t, s, n, 2), key
    assert all(np.isfinite(value).all() for value in data.values() if value.dtype.kind not in "US")
    assert np.abs(data["raw_action"]).max() <= 1
    np.testing.assert_array_equal(data["done"], np.any(data["termination_flags"], axis=-1))
    held = data["held_steering_target_rad"]
    np.testing.assert_array_equal(held, np.repeat(held[:, :1], s, axis=1))
    previous = np.r_[np.zeros((1, n)), held[:-1, 0]].copy()
    resets = np.r_[np.ones((1, n), dtype=bool), data["done"][:-1]]
    previous[resets] = np.clip(data["position_before_rad"][:, 0, :, 0][resets], -1.57, 1.57)
    expected = previous + np.clip(np.clip(data["scaled_raw"][..., 0], -1.57, 1.57) - previous, -.478, .478)
    np.testing.assert_allclose(held[:, 0], expected, rtol=1e-6, atol=3e-7)
    computed = np.empty_like(data["computed_motor_torque_nm"])
    computed[..., 0] = data["controller_kp"][..., 0] * (held - data["position_before_rad"][..., 0])
    computed[..., 0] -= data["controller_kd"][..., 0] * data["velocity_before_rad_s"][..., 0]
    computed[..., 1] = data["scaled_raw"][:, None, :, 1]
    np.testing.assert_allclose(data["computed_motor_torque_nm"], computed, rtol=1e-5, atol=2e-6)
    np.testing.assert_allclose(data["submitted_motor_torque_nm"], np.clip(computed, [-5.6, -6.28], [5.6, 6.28]),
                               rtol=1e-5, atol=2e-6)
    assert np.all(np.abs(data["submitted_motor_torque_nm"]) <= np.array([5.6, 6.28]) + 1e-5)
    q, v = data["position_before_rad"][:, 0], data["velocity_before_rad_s"][:, 0]
    expected_rear = q[..., 1] if isaac else 2 * np.arctan2(np.sin(q[..., 1] / 2), np.cos(q[..., 1] / 2))
    for channel, values in ((5, q[..., 0]), (6, expected_rear), (7, v[..., 0]), (8, v[..., 1]),
                            (9, np.sin(q[..., 0])), (10, np.cos(q[..., 0]))):
        np.testing.assert_allclose(data["observation"][..., channel], values, rtol=2e-6, atol=2e-6)
    if isaac:
        np.testing.assert_array_equal(data["submitted_motor_torque_nm"], data["clipped_motor_torque_nm"])
    else:
        np.testing.assert_array_equal(data["actual_motor_torque_nm"], data["submitted_motor_torque_nm"])


def bounds(values):
    finite = np.concatenate([np.asarray(value).ravel() for value in values])
    low, high = float(finite.min()), float(finite.max())
    padding = max((high - low) * .09, max(abs(low), abs(high)) * 1e-5, 1e-7)
    return (low - padding, high + padding), (low, high)


def response_field(tag):
    return "projected_joint_effort_nm" if tag == "isaac400" else "actual_motor_torque_nm"


def panel_series(tag, data, kind, row):
    """(field, motor/channel, policy sampled, end sampled, label) descriptors."""
    if kind == "demand":
        if row == 0:
            return [("scaled_raw", 0, True, False, "Scaled raw steering"),
                    ("held_steering_target_rad", None, False, False, "Held target entering PD"),
                    ("position_after_rad", 0, False, True, "Actual steering position")]
        return [("scaled_raw", 1, True, False, "Scaled raw rear request"),
                ("submitted_motor_torque_nm", 1, False, False, "Submitted rear motor torque"),
                (response_field(tag), 1, False, True,
                 "Solver joint-axis effort (not motor-only)" if tag == "isaac400" else "Actual motor actuator force")]
    if kind == "torque":
        return [("computed_motor_torque_nm", row, False, False, "Computed before clamp"),
                ("submitted_motor_torque_nm", row, False, False, "Submitted after clamp"),
                (response_field(tag), row, False, True,
                 "Solver joint-axis effort (not motor-only)" if tag == "isaac400" else "Actual motor actuator force")]
    if kind == "observation":
        return [("observation", row, True, False, OBSERVATIONS[row][0])]
    if kind == "raw":
        return [("raw_action", row, True, False, "Steering action" if row == 0 else "Rear action")]
    if kind == "speed":
        return [("velocity_after_rad_s", row, False, True,
                 "Joint speed" if tag == "isaac400" else "Native speed before projection")]
    raise ValueError(kind)


def plot_panel(ax, tag, meta, data, kind, row, env, window, rasterized=False):
    dt, h, s = meta["policy_dt_s"], meta["physics_dt_s"], meta["decimation"]
    window_lo, window_hi = round(window[0] / dt), round(window[1] / dt)
    series = panel_series(tag, data, kind, row)
    traces = [[] for _ in series]
    for start, stop in segments(data["done"][:, env]):
        lo, hi = max(start, window_lo), min(stop, window_hi)
        if lo >= hi:
            continue
        for index, (key, channel, policy, end, label) in enumerate(series):
            values = data[key][lo:hi, env] if policy else data[key][lo:hi, :, env]
            if channel is not None:
                values = values[..., channel]
            values = values.reshape(-1)
            times = np.arange(lo, hi) * dt if policy else np.arange(lo * s, hi * s) * h
            if end:
                times = times + h
                # Draw the measured starting steering pose, not an invented zero.
                if kind == "demand" and row == 0:
                    times = np.r_[lo * dt, times]
                    values = np.r_[data["position_before_rad"][lo, 0, env, 0], values]
            traces[index].append((times, values))
    extrema, plotted = {}, []
    for index, ((_, _, policy, end, label), pieces) in enumerate(zip(series, traces)):
        all_t = np.concatenate([piece[0] for piece in pieces])
        all_y = np.concatenate([piece[1] for piece in pieces])
        low, high = float(all_y.min()), float(all_y.max())
        legend = f"{label}  [min {low:.5g}, max {high:.5g}]"
        for piece, (times, values) in enumerate(pieces):
            keywords = dict(color=COLORS[index], lw=1.05 if index != 1 else 1.2,
                            ls="--" if index == 1 else "-", alpha=.85,
                            label=legend if piece == 0 else None, rasterized=rasterized)
            if end or kind == "observation":
                ax.plot(times, values, **keywords)
            else:
                ax.step(np.r_[times, times[-1] + (dt if policy else h)], np.r_[values, values[-1]],
                        where="post", **keywords)
        for selector in (np.argmin, np.argmax):
            at = selector(all_y)
            ax.plot(all_t[at], all_y[at], "o", ms=3.1, color=COLORS[index], zorder=5, rasterized=rasterized)
        extrema[label] = {"minimum": low, "maximum": high,
                          "minimum_time_s": float(all_t[all_y.argmin()]),
                          "maximum_time_s": float(all_t[all_y.argmax()]), "samples": len(all_y)}
        plotted.append(all_y)
    ylim, (low, high) = bounds(plotted)
    ax.set_ylim(*ylim)
    spawns = spawn_times(data["done"][:, env], dt)
    spawns = spawns[(spawns >= window[0] - 1e-10) & (spawns <= window[1] + 1e-10)]
    for index, time in enumerate(spawns):
        ax.axvline(time, color="#d62728", lw=.85, alpha=.65, clip_on=False,
                   label="Spawn / reset" if index == 0 else None, zorder=4)
    ax.set_xlim(*window)
    ax.grid(alpha=.18)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.tick_params(labelsize=8)
    ax.legend(loc="lower right", fontsize=6.5 if kind != "observation" else 7.3,
              framealpha=.83, borderpad=.35)
    ax.text(.01, .02, f"Panel min {low:.5g} · max {high:.5g}", transform=ax.transAxes, fontsize=7,
            va="bottom", bbox=dict(facecolor="white", edgecolor="none", alpha=.76))
    return dict(environment=env, window_s=list(window), ylim=list(ylim), extrema=extrema,
                panel_min=low, panel_max=high, spawn_times_s=spawns.tolist(), samples_outside_y_view=0,
                legend_location="lower right",
                sampled_dr_episodes=[item["episode"] for item in dr_annotations.episodes_in_view(tag, env, window)])


def figure(datasets, kind, env=0, detail=False, rasterized=False):
    rows = 11 if kind == "observation" else 2
    fig, axes = plt.subplots(rows, 3, figsize=(21, 25 if rows == 11 else 8.4), squeeze=False)
    settings = {}
    for col, tag in enumerate(TAGS):
        meta, data = datasets[tag]
        window = detail_window(data["done"][:, env]) if detail else (0., 60.)
        settings[tag] = []
        for row in range(rows):
            ax = axes[row, col]
            settings[tag].append(plot_panel(ax, tag, meta, data, kind, row, env, window, rasterized))
            if kind == "observation":
                label, unit = OBSERVATIONS[row]
            elif kind == "demand":
                label, unit = ("Steering position", "rad") if row == 0 else ("Rear torque / effort", "N m")
            elif kind == "torque":
                label, unit = ("Steering torque / effort" if row == 0 else "Rear torque / effort", "N m")
            elif kind == "speed":
                label, unit = ("Steering speed" if row == 0 else "Rear speed", "rad/s")
            else:
                label, unit = ("Raw steering action" if row == 0 else "Raw rear action", "normalized")
            ax.set_ylabel(f"{label}\n({unit})", fontsize=9)
            ax.set_xlabel("Evaluation time (s)", fontsize=8)
            if row == 0:
                ax.set_title(f"{TITLES[col]} · env {env}\nDR · {window[0]:.2f}–{window[1]:.2f} s", fontsize=11)
    name = {"demand": "Demand versus actuation", "observation": "All 11 actual policy observations",
            "torque": "Pre-clamp command, submitted torque and response proxy",
            "raw": "Raw bounded Beta policy outputs", "speed": "Physics-substep joint speeds"}[kind]
    fig.suptitle(f"model_2500 · {name} · env {env}" + (" · five-second detail" if detail else " · full 60 s"),
                 fontsize=16, y=.992)
    footer = "50 Hz policy · seed 42 · red = every spawn/reset in view · dots + legends = recorded minima/maxima"
    if kind in ("demand", "torque"):
        footer += ("\nGreen torque: Isaac solver joint-axis effort ≠ isolated motor torque; "
                   "MuJoCo actuator force ≠ net joint torque.")
    if kind == "speed":
        footer += ("\nMuJoCo physics velocities are BEFORE its hard post-step speed projection; "
                   "observations use the projected state.")
    if detail:
        footer += ("\nEach column shows the central ≤5 s of that environment's longest completed episode; "
                   "different absolute windows.")
    footer += ("\nIndependent full-range Y axes · no smoothing, numerical clipping or response time shifting · "
               "DR draws are not paired across engines.")
    fig.text(.5, .006, footer, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, .065 if rows == 2 else .033, 1, .955 if rows == 2 else .975), h_pad=1.2)
    dr_annotations.append_blocks(fig, [(tag, env, detail_window(datasets[tag][1]["done"][:, env])
                                      if detail else (0., 60.)) for tag in TAGS])
    return fig, settings


def describe(meta, data):
    lengths, timeouts = [], []
    for env in range(meta["num_envs"]):
        endings = np.flatnonzero(data["done"][:, env]) + 1
        lengths.extend(np.diff(np.r_[0, endings]).tolist())
        timeouts.extend(data["truncated"][endings - 1, env].tolist())
    return dict(completed_episodes=len(lengths), timeouts=int(sum(timeouts)),
                timeout_rate=float(np.mean(timeouts)), mean_episode_s=float(np.mean(lengths) * .02),
                termination_counts=data["termination_flags"].sum((0, 1)).tolist(),
                raw_action_min=data["raw_action"].min((0, 1)).tolist(),
                raw_action_max=data["raw_action"].max((0, 1)).tolist(),
                scaled_demand_min_rad_nm=data["scaled_raw"].min((0, 1)).tolist(),
                scaled_demand_max_rad_nm=data["scaled_raw"].max((0, 1)).tolist(),
                submitted_torque_min_nm=data["submitted_motor_torque_nm"].min((0, 1, 2)).tolist(),
                submitted_torque_max_nm=data["submitted_motor_torque_nm"].max((0, 1, 2)).tolist(),
                observation_min=data["observation"].min((0, 1)).tolist(),
                observation_max=data["observation"].max((0, 1)).tolist(),
                spawn_times_s_by_environment=[spawn_times(data["done"][:, i]).tolist() for i in range(64)],
                checkpoint_sha256=meta["checkpoint_sha256"], input_sha256=meta["input_sha256"])


def export_csv(tag, meta, data):
    path = HERE / f"{tag}_env0_policy_traces.csv"
    columns = ["policy_time_s", "episode", "raw_steer", "raw_rear", "scaled_steer_rad", "scaled_rear_nm",
               "held_steer_rad", "steer_position_at_policy_start_rad", "steer_position_after_20ms_rad",
               "rear_last_substep_motor_torque_nm"] + [label for label, _ in OBSERVATIONS]
    episode = np.r_[0, np.cumsum(data["done"][:-1, 0])]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for step in range(3000):
            row = [step * .02, episode[step], *data["raw_action"][step, 0],
                   *data["scaled_raw"][step, 0], data["held_steering_target_rad"][step, 0, 0],
                   data["position_before_rad"][step, 0, 0, 0],
                   data["position_after_rad"][step, -1, 0, 0],
                   data["submitted_motor_torque_nm"][step, -1, 0, 1],
                   *data["observation"][step, 0]]
            # NumPy's float32 string is round-trippable as float32 but not exact
            # when a CSV reader parses it directly as float64. Preserve values.
            writer.writerow([value.item() if isinstance(value, np.generic) else value for value in row])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-env-pdfs", action="store_true", help="Also write one page per each of 64 environments")
    parser.add_argument("--pdf-batch", nargs=3, metavar=("KIND", "FIRST", "STOP"),
                        help="Internal bounded-memory PDF worker")
    args = parser.parse_args()
    if args.all_env_pdfs:
        # Separate processes release Matplotlib/PDF image caches between batches.
        # The coordinator never loads the large physics recordings itself.
        subprocess.run([sys.executable, "-B", str(Path(__file__))], check=True)
        settings = json.loads((HERE / "plot_settings.json").read_text())
        for kind, name in (("demand", "all64_demand_vs_actuation"), ("observation", "all64_observation_traces")):
            settings[name], parts = {}, []
            for first in range(0, 64, 8):
                stop = first + 8
                subprocess.run([sys.executable, "-B", str(Path(__file__)),
                                "--pdf-batch", kind, str(first), str(stop)], check=True)
                stem = HERE / "pdf_batches" / f"{kind}_{first:02}_{stop:02}"
                parts.append(str(stem.with_suffix(".pdf")))
                settings[name].update(json.loads(stem.with_suffix(".json").read_text()))
            subprocess.run(["pdfunite", *parts, str(HERE / f"{name}.pdf")], check=True)
        (HERE / "plot_settings.json").write_text(json.dumps(settings, indent=2, allow_nan=False) + "\n")
        page = (HERE / "index.html").read_text()
        links = ('<p>All environments, PDF page = env index +1: '
                 '<a href="all64_demand_vs_actuation.pdf">64-page demand/response</a> · '
                 '<a href="all64_observation_traces.pdf">64-page observations</a>.</p>')
        (HERE / "index.html").write_text(page.replace("<table>", links + "<table>", 1))
        print("Both all-environment PDFs complete: 64 pages each", flush=True)
        return
    datasets = {tag: load(tag) for tag in TAGS}
    assert len({meta["checkpoint_sha256"] for meta, _ in datasets.values()}) == 1
    if args.pdf_batch:
        kind, first, stop = args.pdf_batch[0], int(args.pdf_batch[1]), int(args.pdf_batch[2])
        assert kind in ("demand", "observation") and 0 <= first < stop <= 64 and stop - first <= 8
        folder = HERE / "pdf_batches"
        folder.mkdir(exist_ok=True)
        stem = folder / f"{kind}_{first:02}_{stop:02}"
        settings = {}
        with PdfPages(stem.with_suffix(".pdf")) as pdf:
            for env in range(first, stop):
                fig, settings[str(env)] = figure(datasets, kind, env=env, rasterized=True)
                pdf.savefig(fig, dpi=100)
                plt.close(fig)
        stem.with_suffix(".json").write_text(json.dumps(settings, indent=2, allow_nan=False) + "\n")
        print(kind, "PDF environments", first, "through", stop - 1, "complete", flush=True)
        return
    summary = {tag: describe(*pair) for tag, pair in datasets.items()}
    plots, settings = [], {}
    for kind, detail, name in (("demand", False, "demand_vs_actuation"),
                               ("demand", True, "demand_vs_actuation_detail"),
                               ("observation", False, "observation_traces"),
                               ("observation", True, "observation_traces_detail"),
                               ("torque", False, "motor_torque_flow"), ("torque", True, "motor_torque_flow_detail"),
                               ("raw", False, "raw_policy_outputs"), ("speed", False, "motor_speeds")):
        fig, settings[name] = figure(datasets, kind, detail=detail)
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(HERE / f"{name}.{suffix}", dpi=150)
        plt.close(fig)
        plots.append(name)
        print("Written:", name, flush=True)
    for tag, (meta, data) in datasets.items():
        export_csv(tag, meta, data)
    summary["figures"] = plots
    summary["observations"] = OBSERVATIONS
    summary["validation_passed"] = True
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (HERE / "plot_settings.json").write_text(json.dumps(settings, indent=2, allow_nan=False) + "\n")
    images = "".join(f'<section><h2>{html.escape(name.replace("_", " "))}</h2>'
                     f'<a href="{name}.svg"><img src="{name}.png" loading="lazy" alt="{name}"></a>'
                     f'<p><a href="{name}.pdf">PDF</a> · <a href="{name}.svg">SVG</a></p></section>' for name in plots)
    rows = "".join(f'<tr><td>{html.escape(TITLES[i])}</td><td>{summary[tag]["completed_episodes"]}</td>'
                   f'<td>{summary[tag]["mean_episode_s"]:.3f} s</td>'
                   f'<td>{summary[tag]["timeout_rate"]:.2%}</td></tr>' for i, tag in enumerate(TAGS))
    all_env = ('<p>All environments, PDF page = env index +1: '
               '<a href="all64_demand_vs_actuation.pdf">64-page demand/response</a> · '
               '<a href="all64_observation_traces.pdf">64-page observations</a>.</p>') if args.all_env_pdfs else ""
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>2500 DR simulation traces</title>'
            '<style>body{max-width:1800px;margin:2em auto;padding:0 1em;font:16px system-ui;background:#f6f8fb}'
            'img{width:100%;background:white}td,th{padding:.6em;border:1px solid #bbc}table{border-collapse:collapse}'
            'section{margin:3em 0}a{color:#205b99}</style><h1>model_2500 · Isaac Lab and MuJoCo DR traces</h1>'
            '<p>64 environments ×3000 steps, seed42, 50 Hz policy, all six terminations. Main figures: env0. '
            'Red lines mark actual spawns. Every panel lists its recorded minima/maxima; separate detail '
            'windows use tighter axes without hiding samples from the full-range views.</p>'
            '<p>Different engines have different DR realizations. MuJoCo 800 Hz is a labeled timing variant, '
            'not the Isaac training rate. Isaac green torque is solver joint-axis effort, not isolated motor torque; '
            'MuJoCo green torque is actuator force, not net joint torque. Do not treat these as identical sensors.</p>'
            '<p><a href="README.md">Methods and caveats</a> · <a href="summary.json">Numerical summary</a> · '
            '<a href="plot_settings.json">Every axis, extremum and spawn marker</a></p>' + all_env +
            '<p>Legends are at each panel’s lower right. Sampled DR values appear beneath every figure, '
            'separately for each environment/episode in view. '
            '<a href="sampled_dr_values.json">Full-precision sampled DR values and recovery verification</a>.</p>' +
            '<table><tr><th>Run</th><th>Completed episodes</th><th>Mean episode</th><th>Timeout rate</th></tr>' +
            rows + '</table>' + images + '</html>')
    (HERE / "index.html").write_text(page)
    print(json.dumps({tag: {k: summary[tag][k] for k in ("completed_episodes", "mean_episode_s", "timeout_rate")}
                      for tag in TAGS}, indent=2), flush=True)


if __name__ == "__main__":
    main()
