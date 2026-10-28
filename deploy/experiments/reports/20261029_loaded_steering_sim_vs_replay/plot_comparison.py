"""Plot saved grounded Isaac evaluation and loaded steering replay; no CAN/simulation."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import struct

os.environ.setdefault("MPLCONFIGDIR", "/tmp/trackstand_loaded_steering_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EVAL = ROOT / "isaaclab_train/reports/20261025_2500_dr_comparison"
SIM = EVAL / "isaac400/motor_samples.npz"
STEM = "20261027_190254_424315_scaled_200hz_policy_replay"
CSV = ROOT / "deploy/experiments/logs" / (STEM + ".csv")
SOURCE = ROOT / "deploy/experiments/replay_targets/2500_isaac400_env0_episode1.csv"
UNITS = {"torque": "Nm", "velocity": "rad/s", "position": "rad"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_hardware():
    meta = json.loads(CSV.with_suffix(".json").read_text())
    args, setup = meta["arguments"], meta["steering_origin_setup"]
    assert meta["completed"] and meta["disable_confirmed"] and meta["final_fault_code"] == 0
    assert args["fixture"] == "supported-loaded" and not meta["rear_motor_commanded"]
    assert (args["start"], args["duration"], args["ramp"], args["pre_hold"]) == (0., 5., 0., 0.)
    assert (args["kp"], args["kd"], args["sign"], args["torque_sign"]) == (10., .2, -1, -1)
    assert meta["packet_hz"] == 200 and meta["target_hz"] == 50
    assert all(setup[k] for k in ("attempted", "acknowledged", "verified"))
    assert setup["A3_zero_count"] == meta["encoder_zero_count"] == 0
    assert setup["F1_position_code"] == 32767 and sha(SOURCE) == meta["targets_sha256"]
    with CSV.with_name(STEM + "_plan.csv").open() as stream:
        plan = list(csv.DictReader(stream))
    with SOURCE.open() as stream:
        source = list(csv.DictReader(stream))
    plan_targets = np.array([float(row["target_rad"]) for row in plan])
    source_targets = np.array([float(row["scaled_steering_rad"]) for row in source[:250]])
    assert len(plan) == 300 and np.array_equal(plan_targets[:250], source_targets)
    assert all(float(row["source_time_s"]) == float(source[i]["source_time_s"])
               for i, row in enumerate(plan[:250]))
    with CSV.open() as stream:
        rows = list(csv.DictReader(stream))
    targets = [row for row in rows if row["record_type"] == "mit_target"]
    assert len(targets) == 1200
    tx_time, tx_position = [], []
    for i, row in enumerate(targets):
        assert int(row["target_index"]) == i // 4 and int(row["packet_index"]) == i
        assert float(row["requested_position_rad"]) == plan_targets[i // 4]
        data = bytes.fromhex(row["mit_payload_hex"])
        assert len(data) == 8 and int(row["mit_can_id"]) == 0x502
        position = -(int.from_bytes(data[:2], "big") * 6 / 65535 - 3 - meta["native_zero_rad"])
        assert abs(position - float(row["mit_target_position_rad"])) < 1e-12
        assert abs(position - plan_targets[i // 4]) <= 6 / 65535
        tx_time.append(float(row["replay_time_s"]))
        tx_position.append(position)
    tx_time, tx_position = np.array(tx_time), np.array(tx_position)
    assert np.all(np.diff(tx_time) > 0)
    signals = {}
    for name, opcode, column in (("torque", 0xA1, "torque_nm"), ("velocity", 0xA2, "velocity_rad_s"),
                                  ("position", 0xA3, "position_unwrapped_rad")):
        times, values = [], []
        for row in rows:
            if row["phase"] != "motion" or row["can_direction"] != "bus_rx" or row["can_opcode"] != str(opcode):
                continue
            data = bytes.fromhex(row["mit_payload_hex"])
            assert int(row["mit_can_id"]) == 2 and data[0] == opcode
            if name == "torque":
                value = -struct.unpack("<i", data[1:5])[0] * .001 * meta["torque_constant_nm_per_a"]
            elif name == "velocity":
                value = -struct.unpack("<i", data[1:5])[0] * .01 * 2 * math.pi / 60
            else:
                value = -(struct.unpack("<i", data[3:7])[0] - meta["encoder_zero_count"]) * 2 * math.pi / 16384
            assert abs(value - float(row[column])) < 1e-12
            times.append(float(row["replay_time_s"]))
            values.append(value)
        t, y = np.array(times), np.array(values)
        assert np.all(np.diff(t) > 0) and np.all(np.isfinite(y))
        signals[name] = t, y
    t, q = signals["position"]
    core = (t >= tx_time[0]) & (t < tx_time[1000])
    indexes = np.searchsorted(tx_time, t[core], side="right") - 1
    error = np.degrees(q[core] - tx_position[indexes])
    return signals, plan_targets[:250], dict(
        raw_A1_A2_A3_and_MIT_position_verified=True,
        fixture=args["fixture"], packet_hz=200, target_hz=50, requested_kp=args["kp"], requested_kd=args["kd"],
        B1_verified=True, encoder_zero_count=0, Kt_nm_per_a=meta["torque_constant_nm_per_a"],
        source_core_actual_start_s=float(tx_time[0]), source_core_actual_stop_s=float(tx_time[1000]),
        actual_source_target_rate_hz=250 / float(tx_time[1000] - tx_time[0]),
        actual_motion_duration_s=meta["actual_motion_duration_s"],
        complete_core_unshifted_target_RMS_deg=float(np.sqrt(np.mean(error ** 2))))


def read_simulation(targets):
    metadata = json.loads((EVAL / "isaac400/experiment.json").read_text())
    assert metadata["capture_completed"] and metadata["dr"] and metadata["seed"] == 42
    assert metadata["policy_dt_s"] == .02 and metadata["decimation"] == 8
    with np.load(SIM, allow_pickle=False) as data:
        assert not data["done"][:250, 0].any()
        held = data["held_steering_target_rad"][:250, :, 0].astype(float)
        assert np.array_equal(held, np.repeat(targets[:, None], 8, axis=1))
        before = data["position_before_rad"][:250, :, 0, 0].ravel().astype(float)
        after = data["position_after_rad"][:250, :, 0, 0].ravel().astype(float)
        speed = data["velocity_after_rad_s"][:250, :, 0, 0].ravel().astype(float)
        torque = data["submitted_motor_torque_nm"][:250, :, 0, 0].ravel().astype(float)
        initial_v = float(data["velocity_before_rad_s"][0, 0, 0, 0])
        kp = data["controller_kp"][:250, :, 0, 0].astype(float)
        kd = data["controller_kd"][:250, :, 0, 0].astype(float)
        assert np.ptp(kp) == np.ptp(kd) == 0
        assert before[0] == 0 and np.max(abs(before[1:] - after[:-1])) < 1e-8
        assert np.max(abs(after)) < math.pi
        pre_time, post_time = np.arange(len(after)) / 400, (np.arange(len(after)) + 1) / 400
        signals = {"torque": (pre_time, torque),
                   "velocity": (np.r_[0., post_time], np.r_[initial_v, speed]),
                   "position": (np.r_[0., post_time], np.r_[before[0], after])}
    draws = json.loads((EVAL / "sampled_dr_values.json").read_text())["runs"]["isaac400"]["episodes"]
    draw = next(row for row in draws if row["environment"] == 0 and row["episode"] == 1)
    assert float(kp[0, 0]) == draw["kp"][0] and float(kd[0, 0]) == draw["kd"][0]
    return signals, dict(source="Original grounded full-bike DR evaluation; not an airborne bench replay",
                         environment=0, episode=1, physics_hz=400, policy_hz=50,
                         targets_match_all_250_source_samples_exactly=True,
                         initial_position_rad=float(before[0]), initial_velocity_rad_s=initial_v,
                         kp=float(kp[0, 0]), kd=float(kd[0, 0]), sampled_DR=draw)


def plot(sim, real, window, suffix):
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "svg.fonttype": "none"})
    fig, axes = plt.subplots(3, 1, figsize=(12.8, 10.6), sharex=True)
    fig.subplots_adjust(top=.84, bottom=.12, left=.10, right=.97, hspace=.20)
    metrics = {}
    labels = {"torque": ("Torque (Nm)", "Actuator torque: simulation effort versus hardware A1 current × Kt"),
              "velocity": ("Angular velocity (rad/s)", "Steering angular velocity"),
              "position": ("Position (rad)", "Steering position from each run’s fixed straight zero")}
    for ax, name in zip(axes, UNITS):
        metrics[name] = {}
        for system, signals, color in (("Isaac Lab: grounded DR", sim, "#1767a5"),
                                        ("Hardware: loaded, upright support", real, "#d55e00")):
            t, y = signals[name]
            keep = (t >= window[0]) & (t <= window[1])
            t, y = t[keep], y[keep]
            assert len(y) > 0
            if name == "torque" and signals is sim:
                ax.step(t, y, where="post", color=color, lw=1.3, label=system)
            else:
                ax.plot(t, y, color=color, lw=1.3, label=system)
            metrics[name]["simulation" if signals is sim else "hardware"] = dict(
                samples=len(y), min=float(y.min()), max=float(y.max()), last=float(y[-1]),
                first_time_s=float(t[0]), last_time_s=float(t[-1]))
        a, b = metrics[name]["simulation"], metrics[name]["hardware"]
        lo, hi = min(a["min"], b["min"]), max(a["max"], b["max"])
        span = max(hi - lo, .001)
        ax.set(ylim=(lo - .08 * span, hi + .16 * span), xlim=window,
               ylabel=labels[name][0], title=labels[name][1])
        ax.text(.015, .965, f"Sim min/max: {a['min']:.3f} / {a['max']:.3f}    "
                f"Hardware min/max: {b['min']:.3f} / {b['max']:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
        ax.grid(alpha=.25)
        ax.legend(loc="lower right", fontsize=9, framealpha=.94)
    axes[-1].set_xlabel("Actual elapsed time from replay / source episode start (s)")
    fig.suptitle("Loaded steering replay versus Isaac Lab", fontsize=18, y=.975)
    fig.text(.5, .913, "Model_2500 · source episode 0–5 s · hardware MIT packets: nominal 200 Hz\n"
             "Recorded Isaac evaluation: 400 Hz physics, 50 Hz targets · all 250 requested targets match\n"
             "Actual clocks retained: hardware reaches the source-window end at 5.225 s; no time warp",
             ha="center", va="center", fontsize=10)
    fig.text(.10, .025, "Hardware: tyre carries normal load, frame supported upright; Kp/Kd = 10/0.2.\n"
             "Simulation: grounded DR bike balancing freely; Kp/Kd = 10.6517/0.209184. Support and loads differ.\n"
             "Torque is before passive losses / current-derived, not net shaft torque. No smoothing, "
             "fitted delay or bias correction.", fontsize=9)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(HERE / (suffix + "." + extension), dpi=200, facecolor="white")
    assert all(len(ax.lines) == 2 for ax in axes)
    plt.close(fig)
    return metrics


def main():
    inputs = [CSV, CSV.with_suffix(".json"), CSV.with_name(STEM + "_plan.csv"), SOURCE, SIM,
              EVAL / "isaac400/experiment.json", EVAL / "sampled_dr_values.json"]
    before = {str(path.relative_to(ROOT)): sha(path) for path in inputs}
    real, targets, hardware = read_hardware()
    sim, simulation = read_simulation(targets)
    windows = {"steering_loaded_sim_vs_replay_200hz": (0., 5.),
               "steering_loaded_sim_vs_replay_startup": (0., .5),
               "steering_loaded_sim_vs_replay_detail": (1., 5.)}
    statistics = {name: plot(sim, real, window, name) for name, window in windows.items()}
    with (HERE / "plotted_samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("system", "channel", "elapsed_s", "value", "unit"))
        for system, signals in (("simulation", sim), ("loaded_hardware", real)):
            for channel, (times, values) in signals.items():
                writer.writerows((system, channel, float(t), float(y), UNITS[channel])
                                 for t, y in zip(times, values) if 0 <= t <= 5)
    after = {str(path.relative_to(ROOT)): sha(path) for path in inputs}
    assert before == after
    metadata = dict(hardware=hardware, simulation=simulation, windows_s=windows,
                    input_sha256=before, statistics=statistics,
                    comparison="Original grounded DR evaluation versus supported-loaded hardware; not paired plants",
                    processing="Native timestamps; no smoothing, time shift, time warp or fitted bias subtraction",
                    script_sha256=sha(Path(__file__)), new_simulation_or_hardware_operation=False)
    (HERE / "statistics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (HERE / "verification.json").write_text(json.dumps(dict(
        input_hashes_unchanged=True, raw_CAN_units_signs_and_zero_checked=True,
        exact_250_source_target_values=True, no_early_simulation_reset=True,
        three_panels_two_data_curves_per_figure=True, figures=3, exports=9,
        hardware_is_supported_loaded=True, loaded_50hz_record_available=False,
        simulated_and_real_support_gains_and_timing_differ=True), indent=2) + "\n")
    print(json.dumps(dict(hardware=hardware, simulation_gains=(simulation["kp"], simulation["kd"]),
                          main_plot_statistics=statistics[next(iter(windows))]), indent=2))


if __name__ == "__main__":
    main()
