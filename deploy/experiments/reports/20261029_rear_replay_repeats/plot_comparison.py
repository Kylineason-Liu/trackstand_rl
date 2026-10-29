"""Plot seven saved rear-wheel replays; no simulator, policy, ROS or CAN runs."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import struct

os.environ.setdefault("MPLCONFIGDIR", "/tmp/trackstand_rear_replay_repeats_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LOGS = ROOT / "deploy/experiments/logs"
SIM = ROOT / "isaaclab_train/reports/20261025_matched_target_replay/isaac400/airborne.npz"
SOURCE = ROOT / "deploy/experiments/replay_targets/2500_isaac400_env0_episode1.csv"
RUN_IDS = (
    "152721_598360", "152744_269780", "152804_029946", "152822_687404",
    "152846_268393", "152930_048554", "152944_019062",
)
CHANNELS = ("torque", "velocity", "position")
UNITS = dict(torque="Nm", velocity="rad/s", position="rad")
LABELS = {
    "torque": ("Torque (Nm)", "Actuator torque before passive friction"),
    "velocity": ("Angular velocity (rad/s)", "Rear-wheel velocity"),
    "position": ("Relative position (rad)", "Cumulative rear-wheel displacement"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def read_simulation():
    with np.load(SIM) as data:
        case_name = "isaac400_dr_env0_12p5_17p5_friction_on"
        case = list(data["names"]).index(case_name)
        dt = float(data["physics_dt_s"])
        assert case == 4 and math.isclose(dt, .0025)
        # Axes: [50 Hz target update, physics substep, case, steering/rear].
        before = data["position_before_rad"][:, :, case, 1].ravel().astype(float)
        after = data["position_after_rad"][:, :, case, 1].ravel().astype(float)
        increments = (after - before + 2 * math.pi) % (4 * math.pi) - 2 * math.pi
        continuity = (before[1:] - after[:-1] + 2 * math.pi) % (4 * math.pi) - 2 * math.pi
        assert np.max(abs(continuity)) < 1e-8 and np.max(abs(increments)) < .2
        # Sum measured per-step angle changes, not velocities; preserve full turns.
        position = np.cumsum(increments)
        pre_time = np.arange(len(after)) * dt
        post_time = (np.arange(len(after)) + 1) * dt
        signals = {}
        for channel, times, values in (
            ("torque", pre_time, data["submitted_nm"][:, :, case, 1].ravel()),
            ("velocity", post_time, data["velocity_after_rad_s"][:, :, case, 1].ravel()),
            ("position", post_time, position),
        ):
            keep = (times >= 2) & (times <= 7)
            if channel == "torque":
                keep &= times < 7
            signals[channel] = times[keep] - 2, np.asarray(values[keep], dtype=float)
        targets = data["rear_nm"][:350, case].astype(float)
    return signals, targets, dict(
        case_index=case, case_name=case_name, physics_dt_s=dt, fixed_base=True,
        fresh_domain_randomization=False, policy_inference=False,
        native_position_branch_rad=4 * math.pi,
        position_method="cumulative native angle increments with 4*pi branch resolved; not speed integration",
        maximum_position_increment_rad=float(max(abs(increments))),
    )


def read_hardware(run_id, sim_targets, source_rows):
    stem = f"20261029_{run_id}_scaled_50hz_drive_policy_replay"
    path = LOGS / f"{stem}.csv"
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert all(metadata[key] for key in ("completed", "disable_confirmed", "watchdog_restored"))
    assert metadata["packet_hz"] == metadata["target_hz"] == 50 and metadata["final_fault_code"] == 0
    assert metadata["command_schedule"] == "send_relative_no_catchup"
    assert metadata["targets_sha256"] == sha(SOURCE)
    args = metadata["arguments"]
    assert args["motor_id"] == 1 and args["encoder_sign"] == args["torque_sign"] == -1
    assert (args["start"], args["duration"], args["pre_zero"], args["post_hold"]) == (12.5, 5, 2, 0)
    assert args["level"] == "scaled" and metadata["host_damping_nm_s_per_rad"] == 0
    assert metadata["friction_compensation_nm"] == 0 and not metadata["steering_motor_commanded"]
    kt, zero = metadata["torque_constant_nm_per_a"], metadata["encoder_zero_count"]
    plan_path = path.with_name(path.stem + "_plan.csv")
    plan = read_csv(plan_path)
    planned = np.array([float(row["requested_torque_nm"]) for row in plan])
    assert len(plan) == 350 and np.array_equal(sim_targets, planned)
    core_plan = [row for row in plan if row["phase"] == "core"]
    assert len(core_plan) == len(source_rows) == 250
    assert np.array_equal([float(row["requested_torque_nm"]) for row in core_plan],
                          [float(row["scaled_drive_nm"]) for row in source_rows])
    assert np.array_equal([float(row["source_time_s"]) for row in core_plan],
                          [float(row["source_time_s"]) for row in source_rows])
    rows = read_csv(path)
    commands = [row for row in rows if row["record_type"] == "drive_target"]
    assert len(commands) == 350
    assert [int(row["packet_index"]) for row in commands] == list(range(350))
    assert [int(row["target_index"]) for row in commands] == list(range(350))
    times = np.array([float(row["replay_time_s"]) for row in commands])
    assert np.all(np.diff(times) > 0)
    encoded = []
    for row, target in zip(commands, planned):
        payload = bytes.fromhex(row["payload_hex"])
        assert row["can_direction"] == "local_tx" and row["can_id_hex"] == "101"
        assert payload[0] == 0xC0 and len(payload) == 5
        code = struct.unpack("<i", payload[1:])[0]
        assert code == round(-target / kt * 1000)
        assert float(row["requested_torque_nm"]) == target
        assert int(row["skipped_command_ticks"]) == 0
        encoded.append(-code * .001 * kt)
    core_start = float(commands[100]["replay_time_s"])
    end = metadata["actual_motion_duration_s"]
    signals = {}
    for name, opcode, column in (("torque", 0xA1, "torque_nm"), ("velocity", 0xA2, "velocity_rad_s"),
                                  ("position", 0xA3, "position_unwrapped_rad")):
        selected = [row for row in rows if row["phase"] == "motion" and row["can_direction"] == "bus_rx"
                    and row["can_opcode"] == str(opcode)]
        rx_time, values = [], []
        for row in selected:
            payload = bytes.fromhex(row["payload_hex"])
            assert payload[0] == opcode and row["can_id_hex"] == "001"
            if name == "torque":
                value = -struct.unpack("<i", payload[1:])[0] * .001 * kt
            elif name == "velocity":
                value = -struct.unpack("<i", payload[1:])[0] * .01 * 2 * math.pi / 60
            else:
                value = -(struct.unpack("<i", payload[3:])[0] - zero) * 2 * math.pi / 16384
            assert math.isclose(value, float(row[column]), rel_tol=0, abs_tol=1e-12)
            rx_time.append(float(row["replay_time_s"]))
            values.append(value)
        t, y = np.array(rx_time), np.array(values)
        assert np.all(np.diff(t) > 0) and np.all(np.isfinite(y))
        keep = (t >= core_start) & (t <= end)
        assert keep.sum() > 450
        signals[name] = t[keep] - core_start, y[keep]
    statuses = [row["query_status"] for row in rows if row["record_type"] == "query_result"]
    assert statuses and set(statuses) == {"received"}
    torque_time, torque = signals["torque"]
    active = np.searchsorted(times, torque_time + core_start, side="right") - 1
    error = torque - np.array(encoded)[active]
    core_intervals = np.diff(times[100:])
    metrics = dict(
        run_id=run_id, input_stem=stem, packet_hz_nominal=50,
        encoder_zero_count=zero, startup_native_phase_deg=(zero % 16384) * 360 / 16384,
        torque_constant_nm_per_a=kt, core_start_replay_time_s=core_start,
        actual_core_duration_s=end - core_start,
        mean_core_command_interval_ms=float(core_intervals.mean() * 1000),
        max_core_command_interval_ms=float(core_intervals.max() * 1000),
        achieved_core_packet_hz=float(1 / core_intervals.mean()),
        packet_count=350, preparation_packet_count=100, core_packet_count=250,
        query_result_count=len(statuses), all_query_results_received=True,
        packet_encoding_quantization_max_nm=float(max(abs(planned - encoded))),
        torque_rmse_vs_actual_sent_hold_nm=float(np.sqrt(np.mean(error**2))),
        note="Torque RMSE uses actual held encoded C0 requests with no fitted delay.",
    )
    inputs = [path, path.with_suffix(".json"), plan_path]
    return signals, metrics, {str(p.relative_to(ROOT)): sha(p) for p in inputs}


def make_figure(sim, real, metrics, number, pdf):
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "svg.fonttype": "none"})
    fig, axes = plt.subplots(3, 1, figsize=(12.8, 10.2), sharex=True)
    fig.subplots_adjust(top=.86, bottom=.11, left=.10, right=.97, hspace=.23)
    duration = metrics["actual_core_duration_s"]
    for ax, name in zip(axes, CHANNELS):
        ts, ys = sim[name]
        tr, yr = real[name]
        sim_label = "Isaac Lab 400 Hz"
        real_label = "Hardware replay, nominal C0 50 Hz"
        if name == "torque":
            ax.step(ts, ys, where="post", color="#1767a5", lw=1.5,
                    label=sim_label + " — submitted actuator effort")
            real_label += " — A1 current × Kt"
        else:
            ax.plot(ts, ys, color="#1767a5", lw=1.7, label=sim_label)
        ax.plot(tr, yr, color="#d55e00", lw=1.1, label=real_label)
        ax.set(ylabel=LABELS[name][0], title=LABELS[name][1], xlim=(0, max(5, duration)))
        low, high = float(min(ys.min(), yr.min())), float(max(ys.max(), yr.max()))
        margin = max(.07 * (high - low), .0005)
        ax.set_ylim(low - margin, high + max(margin, .13 * (high - low)))
        ax.set_xticks(np.arange(6))
        ax.grid(alpha=.25)
        ax.legend(loc="lower left" if name == "position" else "lower right", fontsize=9, framealpha=.94)
        ax.text(.015, .965, f"Sim min/max: {ys.min():.4g} / {ys.max():.4g}    "
                f"Hardware min/max: {yr.min():.4g} / {yr.max():.4g}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.83))
        metrics[name] = dict(unit=UNITS[name], sim_min=float(ys.min()), sim_max=float(ys.max()),
                             hardware_min=float(yr.min()), hardware_max=float(yr.max()),
                             sim_last=float(ys[-1]), hardware_last=float(yr[-1]),
                             sim_last_time_s=float(ts[-1]), hardware_last_time_s=float(tr[-1]),
                             sim_samples=len(ts), hardware_samples=len(tr), y_limits=list(ax.get_ylim()))
    axes[-1].set_xlabel("Elapsed time from first core command (s); actual hardware timestamps")
    clock = metrics["run_id"].split("_")[0]
    clock = f"{clock[:2]}:{clock[2:4]}:{clock[4:]}"
    fig.suptitle(f"Rear wheel: simulation versus replay {number} — 2026-10-29 {clock}", fontsize=16, y=.967)
    fig.text(.5, .913, "Same ordered model_2500 torque targets · source episode 12.5–17.5 s\n"
             f"Airborne hardware, nominal C0 50 Hz ({metrics['achieved_core_packet_hz']:.2f} Hz measured) · "
             "saved nominal fixed-base Isaac400\n"
             f"Core duration: simulation 5.000 s; hardware {duration:.3f} s · no fitted delay or time rescaling",
             ha="center", va="center", fontsize=10)
    fig.text(.10, .029, "Torque: submitted simulation effort versus current-derived hardware estimate, "
             "before passive losses.\nPosition: full turns from each startup zero; not rebased at core start. "
             "Physical wheel starting angle was not matched between runs.", fontsize=9)
    filename = f"20261029_{metrics['run_id']}_rear_sim_vs_replay_50hz"
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(HERE / f"{filename}.{suffix}", dpi=200, facecolor="white")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)
    metrics["figure_basename"] = filename


def write_report(statistics):
    rows, cards, csv_rows = [], [], []
    for number, item in enumerate(statistics, 1):
        base, run = item["figure_basename"], item["run_id"]
        pos, speed = item["position"], item["velocity"]
        rows.append(f"| {number} / {run} | [{pos['hardware_last']:.6f}]({base}.png) | "
                    f"{max(abs(speed['hardware_min']), abs(speed['hardware_max'])):.6f} | "
                    f"{item['actual_core_duration_s']:.6f} | {item['achieved_core_packet_hz']:.3f} |")
        cards.append(f'<section id="run{number}"><h2>Replay {number}: {run} — nominal 50 Hz</h2>'
                     f'<p><a href="{base}.png">PNG</a> · <a href="{base}.svg">SVG</a> · '
                     f'<a href="{base}.pdf">PDF</a></p><a href="{base}.png"><img src="{base}.png" '
                     f'alt="Replay {number}: rear torque, velocity and unwrapped position '
                     'versus Isaac400"></a></section>')
        csv_rows.append(dict(
            run=run, core_duration_s=item["actual_core_duration_s"],
            achieved_packet_hz=item["achieved_core_packet_hz"],
            hardware_last_position_rad=pos["hardware_last"], hardware_last_position_time_s=pos["hardware_last_time_s"],
            hardware_min_velocity_rad_s=speed["hardware_min"], hardware_max_velocity_rad_s=speed["hardware_max"],
            torque_rmse_vs_sent_nm=item["torque_rmse_vs_actual_sent_hold_nm"],
            startup_native_phase_deg=item["startup_native_phase_deg"],
        ))
    text = """# Seven rear-wheel simulation versus replay comparisons

Each figure follows `20261029_rear_sim_vs_replay`: three vertically aligned
panels, two curves per panel (torque, velocity and cumulative position). Blue is
the same saved Isaac Lab 400 Hz actuator replay; orange is fresh hardware feedback.
All seven are **nominal 50 Hz C0** replays, not a 50-versus-200 Hz experiment.

[HTML gallery](index.html) · [Seven-page PDF](all_replays.pdf) ·
[Summary CSV](summary.csv) · [Native plotted samples](plotted_samples.csv) ·
[Statistics and input hashes](statistics.json) · [Verification](verification.json).

| Replay / timestamp | Last position (rad; PNG) | Peak speed (rad/s) | Core duration (s) | Packet rate (Hz) |
| --- | ---: | ---: | ---: | ---: |
""" + "\n".join(rows) + """

Each plan has 100 zero-torque preparation commands and 250 core commands from
model_2500's Isaac-recorded episode window **12.5–17.5 s**. All 350 values exactly
match the saved simulator's targets. All seven logs report completion, confirmed
disable, restored watchdog and final fault code zero; every logged query completed.
The script independently verifies transmitted C0 payloads and A1/A2/A3 units/signs.

Time zero is the **actual first core command send** in each hardware log, and the
nominal first core command in simulation. This uses known command events, not a
fitted response delay. Hardware intervals retain their actual durations; no
smoothing, interpolation of the plotted curves, time rescaling or bias fitting is
applied. Send-relative scheduling stretches playback slightly. The full actual
hardware core is shown; the simulator curve stops at its nominal five-second
endpoint. Consequently, targets match in order and value, but their timing is not
identical. End-position samples are asynchronous and are not all at exactly 5.000 s.

Torque compares simulation **submitted actuator effort before passive friction**
with hardware **signed A1 current × measured B0 Kt**, not net wheel torque or an
independent torque sensor. Solver-measured joint-axis effort is not plotted.
Hardware position uses signed A3 multi-turn count minus that run's startup count.
Simulation sums native per-physics-step position increments, resolving its 4π
branch. Neither uses velocity integration, ±π wrapping or a new core-start zero.

The user confirmed all seven had the rear wheel airborne and free, without hand
contact/restraint, and unchanged firmware and supply settings. **The wheel was not
returned to a common physical starting angle.** A relative encoder zero therefore
does not make motor electrical/mechanical starting phase identical. Different
motion is visible, but these plots alone cannot identify its cause or prove
intrinsic motor randomness.

The simulation reference is the original nominal fixed-base case
`isaac400_dr_env0_12p5_17p5_friction_on` (case 4), not the original on-ground policy
episode. The source policy was DR-trained, but this actuator replay has no fresh
DR sampling or inference. Support and actuator dynamics are not fully matched.
No new simulation or hardware was run to make these plots. The original report,
recordings, target file and control code are unchanged.

Reproduce from the repository root:

```bash
python3 -B deploy/experiments/reports/20261029_rear_replay_repeats/plot_comparison.py
```
"""
    (HERE / "README.md").write_text(text)
    with (HERE / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    navigation = " · ".join(f'<a href="#run{n}">Replay {n}</a>' for n in range(1, 8))
    (HERE / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width, initial-scale=1"><title>Seven rear-wheel replays</title>'
        '<style>body{font:17px system-ui,sans-serif;max-width:1280px;margin:30px auto;padding:0 20px;'
        'color:#20272c;background:#f8fafb}img{width:100%;height:auto}section{margin:45px 0;background:white;'
        'padding:16px;border:1px solid #dbe3e8}a{color:#1767a5}p{line-height:1.55}</style>'
        '<h1>Rear wheel: seven simulation-versus-replay comparisons</h1>'
        '<p>2026-10-29 · all nominal 50 Hz C0 · model_2500 source window 12.5–17.5 s.</p>'
        '<p>Three panels per replay: torque (Nm), velocity (rad/s), multi-turn relative position (rad). '
        'Blue: saved nominal fixed-base Isaac400; orange: airborne hardware.</p>'
        '<p>Actual hardware timing is preserved. Each origin is its first core-command send; '
        'hardware lasts slightly longer than the simulator’s 5 s. Physical starting wheel angles differed. '
        'No fitted delay, smoothing, time rescaling or core-start position reset.</p>'
        '<p><a href="all_replays.pdf">Download all seven as PDF</a> · '
        '<a href="README.md">Methods and summary</a> · <a href="summary.csv">Summary CSV</a> · '
        '<a href="statistics.json">Statistics</a> · <a href="plotted_samples.csv">Plotted samples</a></p>'
        f'<nav>{navigation}</nav>' + "".join(cards) + '</html>')


def main():
    sim, sim_targets, sim_metadata = read_simulation()
    source_rows = [row for row in read_csv(SOURCE) if 12.5 <= float(row["source_time_s"]) < 17.5]
    provenance = {str(p.relative_to(ROOT)): sha(p) for p in (SIM, SIM.with_name("metadata.json"), SOURCE)}
    statistics = []
    with PdfPages(HERE / "all_replays.pdf") as pdf, (HERE / "plotted_samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("run_id", "system", "channel", "elapsed_from_core_start_s", "value", "unit"))
        for number, run_id in enumerate(RUN_IDS, 1):
            real, metrics, hashes = read_hardware(run_id, sim_targets, source_rows)
            provenance.update(hashes)
            make_figure(sim, real, metrics, number, pdf)
            statistics.append(metrics)
            for system, signals in (("simulation", sim), ("hardware", real)):
                for channel, (times, values) in signals.items():
                    writer.writerows((run_id, system, channel, float(t), float(y), UNITS[channel])
                                     for t, y in zip(times, values))
    write_report(statistics)
    metadata = dict(
        simulation=sim_metadata, source_episode_window_s=[12.5, 17.5], nominal_core_duration_s=5,
        user_confirmed_fixture="rear wheel airborne, free, no hand contact/restraint",
        user_confirmed_settings="firmware and power supply unchanged",
        user_confirmed_starting_phase="not returned to same physical wheel angle between runs",
        graph_time_zero="actual first hardware core TX; simulation nominal core start; no fitted delay",
        position_zero="startup multi-turn count, not core-start angle",
        torque_definition="simulation submitted effort; hardware signed A1 current times measured B0 Kt",
        input_sha256=provenance, plotting_script_sha256=sha(Path(__file__)),
        new_simulation_or_hardware_run=False, statistics=statistics,
    )
    (HERE / "statistics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    verification = dict(
        replay_count=len(statistics), individual_panel_count=3 * len(statistics), curves_per_panel=2,
        all_completed_and_disabled_and_watchdog_restored=True, all_final_fault_codes_zero=True,
        all_350_targets_match_simulator=True, all_250_core_targets_match_source=True,
        all_transmitted_current_payloads_verified=True,
        all_fresh_feedback_values_verified_from_raw_A1_A2_A3=True, all_query_results_received=True,
        no_plot_time_rescaling_or_smoothing_or_fitted_offset=True,
        source_hashes_unchanged_after_reading=all(sha(ROOT / path) == digest for path, digest in provenance.items()),
        input_sha256=provenance,
    )
    assert verification["source_hashes_unchanged_after_reading"]
    (HERE / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    for item in statistics:
        print(f"{item['run_id']}: {item['achieved_core_packet_hz']:.3f} Hz, "
              f"last position {item['position']['hardware_last']:.6f} rad, "
              f"torque RMSE {item['torque_rmse_vs_actual_sent_hold_nm']:.6f} Nm")


if __name__ == "__main__":
    main()
