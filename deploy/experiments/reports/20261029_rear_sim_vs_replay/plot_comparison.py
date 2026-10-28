"""Plot saved rear Isaac400/hardware replays; no policy, simulator, ROS or CAN runs."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import struct

os.environ.setdefault("MPLCONFIGDIR", "/tmp/trackstand_rear_sim_vs_replay_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SIM = ROOT / "isaaclab_train/reports/20261025_matched_target_replay/isaac400/airborne.npz"
LOGS = ROOT / "deploy/experiments/logs"
STEMS = {
    50: "20261026_163432_217938_scaled_50hz_drive_policy_replay",
    200: "20261026_163545_994664_scaled_200hz_drive_policy_replay",
}
CORE = (2., 7.)
UNITS = {"torque": "Nm", "velocity": "rad/s", "position": "rad"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_hardware(rate):
    path = LOGS / (STEMS[rate] + ".csv")
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["completed"] and metadata["disable_confirmed"] and metadata["watchdog_restored"]
    assert metadata["packet_hz"] == rate and metadata["final_fault_code"] == 0
    args = metadata["arguments"]
    assert args["encoder_sign"] == args["torque_sign"] == -1
    assert args["start"] == 12.5 and args["duration"] == 5 and args["pre_zero"] == 2
    kt, zero = metadata["torque_constant_nm_per_a"], metadata["encoder_zero_count"]
    plan_path = path.with_name(path.stem + "_plan.csv")
    with plan_path.open() as stream:
        plan = [float(row["requested_torque_nm"]) for row in csv.DictReader(stream)]
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    signals = {}
    for name, opcode, column in (("torque", 0xA1, "torque_nm"), ("velocity", 0xA2, "velocity_rad_s"),
                                  ("position", 0xA3, "position_unwrapped_rad")):
        selected = [r for r in rows if r["phase"] == "motion" and r["can_direction"] == "bus_rx"
                    and r["can_opcode"] == str(opcode)]
        times, values = [], []
        for row in selected:
            payload = bytes.fromhex(row["payload_hex"])
            if name == "torque":
                value = -struct.unpack("<i", payload[1:])[0] * .001 * kt
            elif name == "velocity":
                value = -struct.unpack("<i", payload[1:])[0] * .01 * 2 * math.pi / 60
            else:
                value = -(struct.unpack("<i", payload[3:])[0] - zero) * 2 * math.pi / 16384
            assert math.isclose(value, float(row[column]), rel_tol=0, abs_tol=1e-12)
            times.append(float(row["replay_time_s"]))
            values.append(value)
        t, y = np.array(times), np.array(values)
        assert np.all(np.diff(t) > 0) and np.all(np.isfinite(y))
        keep = (t >= CORE[0]) & (t <= CORE[1])
        signals[name] = t[keep] - CORE[0], y[keep]
    inputs = [path, path.with_suffix(".json"), plan_path]
    return signals, np.array(plan), {str(p.relative_to(ROOT)): sha(p) for p in inputs}


def read_simulation():
    with np.load(SIM) as data:
        case = list(data["names"]).index("isaac400_dr_env0_12p5_17p5_friction_on")
        assert case == 4
        dt = float(data["physics_dt_s"])
        assert math.isclose(dt, .0025)
        # Stored axes are [policy step, physics substep, replay case, steering/rear].
        before = data["position_before_rad"][:, :, case, 1].ravel().astype(float)
        after = data["position_after_rad"][:, :, case, 1].ravel().astype(float)
        velocity = data["velocity_after_rad_s"][:, :, case, 1].ravel().astype(float)
        torque = data["submitted_nm"][:, :, case, 1].ravel().astype(float)
        targets = data["rear_nm"][:, case].astype(float)
        increments = (after - before + 2 * math.pi) % (4 * math.pi) - 2 * math.pi
        continuity = (before[1:] - after[:-1] + 2 * math.pi) % (4 * math.pi) - 2 * math.pi
        assert np.max(abs(continuity)) < 1e-8
        assert np.max(abs(increments)) < .2  # No ambiguous multi-turn jump within a physics step.
        # PhysX native angles use a 4*pi branch. Integrate actual position increments,
        # not velocity, to preserve full cumulative displacement from the spawn zero.
        position = np.cumsum(increments)
        post_time = (np.arange(len(after)) + 1) * dt
        pre_time = np.arange(len(after)) * dt
        signals = {}
        for name, times, values in (("torque", pre_time, torque), ("velocity", post_time, velocity),
                                    ("position", post_time, position)):
            keep = (times >= CORE[0]) & (times <= CORE[1])
            if name == "torque":
                keep &= times < CORE[1]  # Effort applies over [pre_time, pre_time+dt).
            assert np.all(np.isfinite(values))
            signals[name] = times[keep] - CORE[0], values[keep]
        return signals, targets, dict(case_index=case, case_name=str(data["names"][case]), physics_dt_s=dt,
                                      native_angle_branch_rad=4 * math.pi,
                                      position_method="sum native per-substep increments, resolving 4*pi branch",
                                      maximum_position_increment_rad=float(max(abs(increments))))


def make_figure(sim, real, rate):
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "svg.fonttype": "none"})
    fig, axes = plt.subplots(3, 1, figsize=(12.8, 10.2), sharex=True)
    fig.subplots_adjust(top=.87, bottom=.10, left=.10, right=.97, hspace=.20)
    labels = {"torque": ("Torque (Nm)", "Actuator torque before passive friction"),
              "velocity": ("Angular velocity (rad/s)", "Rear-wheel velocity"),
              "position": ("Relative position (rad)", "Cumulative rear-wheel displacement")}
    metrics = {}
    for ax, name in zip(axes, ("torque", "velocity", "position")):
        ts, ys = sim[name]
        tr, yr = real[name]
        sim_label = "Isaac Lab 400 Hz"
        real_label = f"Hardware replay, C0 {rate} Hz"
        if name == "torque":
            sim_label += " — submitted actuator effort"
            real_label += " — A1 current × Kt"
            ax.step(ts, ys, where="post", color="#1767a5", lw=1.5, label=sim_label)
        else:
            ax.plot(ts, ys, color="#1767a5", lw=1.7, label=sim_label)
        ax.plot(tr, yr, color="#d55e00", lw=1.1, label=real_label)
        ax.set(ylabel=labels[name][0], title=labels[name][1], xlim=(0, 5))
        combined = np.r_[ys, yr]
        lo, hi = float(combined.min()), float(combined.max())
        margin = max(.07 * (hi - lo), .0005)
        ax.set_ylim(lo - margin, hi + max(margin, .13 * (hi - lo)))
        ax.grid(alpha=.25)
        ax.legend(loc="lower left" if name == "position" else "lower right", fontsize=9, framealpha=.94)
        ax.text(.015, .965, f"Sim min/max: {ys.min():.3f} / {ys.max():.3f}    "
                f"Hardware min/max: {yr.min():.3f} / {yr.max():.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.83))
        metrics[name] = dict(unit=UNITS[name], sim_min=float(ys.min()), sim_max=float(ys.max()),
                             hardware_min=float(yr.min()), hardware_max=float(yr.max()),
                             sim_last=float(ys[-1]), hardware_last=float(yr[-1]),
                             sim_samples=len(ts), hardware_samples=len(tr))
    axes[-1].set_xlabel("Time since recorded torque sequence begins (s)")
    fig.suptitle(f"Rear wheel: simulation versus {rate} Hz hardware replay", fontsize=17, y=.965)
    fig.text(.5, .918, "Same 50 Hz-held model_2500 torque sequence · source episode 12.5–17.5 s\n"
             "Airborne hardware versus saved nominal fixed-base simulation · actual timestamps, no time shift",
             ha="center", va="center", fontsize=10)
    fig.text(.10, .028, "Torque is actuator-level: simulation effort versus current-derived hardware estimate, "
             "not net torque after losses.\nPosition preserves full turns relative to each run’s initial reference; "
             "support and actuator dynamics are not fully matched.", fontsize=9)
    name = f"rear_sim_vs_replay_{rate}hz"
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(HERE / f"{name}.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)
    return metrics


def main():
    sim, targets, sim_metadata = read_simulation()
    provenance = {str(SIM.relative_to(ROOT)): sha(SIM)}
    statistics = {}
    with (HERE / "plotted_samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("replay_packet_hz", "system", "channel", "core_time_s", "value", "unit"))
        for rate in STEMS:
            real, plan, hashes = read_hardware(rate)
            assert len(plan) == 350 and np.array_equal(targets[:350], plan)
            provenance.update(hashes)
            statistics[rate] = make_figure(sim, real, rate)
            for system, signals in (("simulation", sim), ("hardware", real)):
                for channel, (times, values) in signals.items():
                    writer.writerows((rate, system, channel, float(t), float(y), UNITS[channel])
                                     for t, y in zip(times, values))
    metadata = dict(simulation=sim_metadata, core_replay_window_s=CORE, source_episode_window_s=(12.5, 17.5),
                    common_target_hz=50, hardware_packet_hz=list(STEMS), input_sha256=provenance,
                    torque_definition="Simulation submitted effort; hardware signed A1 current times measured B0",
                    plot_processing="native fresh samples; no smoothing, signal time shift or fitted bias removal",
                    graph_time_zero="replay t=2 s, after zero-current preparation",
                    position_zero="each run's initial position; not rebased at core start",
                    simulation_or_hardware_run=False, statistics=statistics,
                    plotting_script_sha256=sha(Path(__file__)))
    (HERE / "statistics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
