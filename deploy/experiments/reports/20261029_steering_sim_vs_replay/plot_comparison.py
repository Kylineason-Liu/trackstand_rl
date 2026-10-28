"""Read-only three-panel steering simulation/hardware comparison; no CAN or simulation."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import struct

os.environ.setdefault("MPLCONFIGDIR", "/tmp/trackstand_steering_sim_vs_replay_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SIM = ROOT / "isaaclab_train/reports/20261025_matched_target_replay/isaac400/airborne.npz"
LOGS = ROOT / "deploy/experiments/logs"
STEMS = {
    50: "20261027_144059_652887_scaled_50hz_policy_replay",
    200: "20261027_144436_005178_scaled_200hz_policy_replay",
}
CORE = (2., 7.)
UNITS = {"torque": "Nm", "velocity": "rad/s", "position": "rad"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_hardware(rate):
    path = LOGS / (STEMS[rate] + ".csv")
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["completed"] and meta["disable_confirmed"] and meta["final_fault_code"] == 0
    assert meta["packet_hz"] == rate and meta["target_hz"] == 50
    args, setup = meta["arguments"], meta["steering_origin_setup"]
    assert args["sign"] == args["torque_sign"] == -1 and args["fixture"] == "airborne"
    assert args["start"] == 12.5 and args["duration"] == 5 and args["ramp"] == args["pre_hold"] == 1
    assert args["kp"] == 10 and args["kd"] == .2
    assert all(setup[k] for k in ("attempted", "acknowledged", "verified"))
    assert meta["encoder_zero_count"] == setup["A3_zero_count"] == 0 and setup["F1_position_code"] == 32767
    kt, zero = meta["torque_constant_nm_per_a"], meta["encoder_zero_count"]
    source = ROOT / "deploy/experiments/replay_targets" / Path(args["targets"]).name
    assert sha(source) == meta["targets_sha256"]
    plan_path = path.with_name(path.stem + "_plan.csv")
    with plan_path.open() as stream:
        plan = np.array([float(row["target_rad"]) for row in csv.DictReader(stream)])
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    targets = [r for r in rows if r["record_type"] == "mit_target"]
    repeats = rate // 50
    assert len(plan) == 400 and len(targets) == 400 * repeats
    assert all(float(row["requested_position_rad"]) == plan[i // repeats] for i, row in enumerate(targets))
    for i, row in enumerate(targets):
        payload = bytes.fromhex(row["mit_payload_hex"])
        assert len(payload) == 8 and int(row["mit_can_id"]) == 0x502
        native_position = int.from_bytes(payload[:2], "big") * 6 / 65535 - 3
        encoded_position = -(native_position - meta["native_zero_rad"])
        assert abs(encoded_position - float(row["mit_target_position_rad"])) < 1e-12
        assert abs(encoded_position - plan[i // repeats]) < 6 / 65535
    tx_time = np.array([float(row["replay_time_s"]) for row in targets])
    tx_position = np.array([float(row["mit_target_position_rad"]) for row in targets])
    assert np.all(np.diff(tx_time) > 0)
    signals = {}
    for name, opcode, column in (("torque", 0xA1, "torque_nm"), ("velocity", 0xA2, "velocity_rad_s"),
                                  ("position", 0xA3, "position_unwrapped_rad")):
        selected = [r for r in rows if r["phase"] == "motion" and r["can_direction"] == "bus_rx"
                    and r["can_opcode"] == str(opcode)]
        times, values = [], []
        for row in selected:
            payload = bytes.fromhex(row["mit_payload_hex"])
            assert int(row["mit_can_id"]) == 2 and payload[0] == opcode
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
        keep = (t >= CORE[0]) & (t < CORE[1])
        signals[name] = t[keep] - CORE[0], y[keep]
    tp, qp = signals["position"]
    indexes = np.searchsorted(tx_time, tp + CORE[0], side="right") - 1
    assert np.all(indexes >= 0)
    error_deg = np.degrees(qp - tx_position[indexes])
    checks = dict(A3_samples=len(qp), core_tracking_rms_deg=float(np.sqrt(np.mean(error_deg ** 2))),
                  core_tracking_bias_deg=float(error_deg.mean()), B1_zero_count=zero,
                  measured_Kt_nm_per_a=kt)
    inputs = (path, path.with_suffix(".json"), plan_path, source)
    return signals, plan, {str(p.relative_to(ROOT)): sha(p) for p in inputs}, checks


def read_simulation():
    with np.load(SIM) as data:
        case = list(data["names"]).index("isaac400_dr_env0_12p5_17p5_friction_on")
        assert case == 4
        dt = float(data["physics_dt_s"])
        assert math.isclose(dt, .0025)
        # Saved order is [policy step, physics substep, replay case, steering/rear].
        before = data["position_before_rad"][:, :, case, 0].ravel().astype(float)
        after = data["position_after_rad"][:, :, case, 0].ravel().astype(float)
        velocity = data["velocity_after_rad_s"][:, :, case, 0].ravel().astype(float)
        torque = data["submitted_nm"][:, :, case, 0].ravel().astype(float)
        targets = data["target_rad"][:, case].astype(float)
        assert abs(before[0]) < 1e-10
        assert np.max(abs(before[1:] - after[:-1])) < 1e-8
        assert np.max(abs(after - before)) < .1 and np.max(abs(after)) < math.pi
        position = after - before[0]  # Fixed spawn zero; no re-zeroing at the displayed interval.
        post_time = (np.arange(len(after)) + 1) * dt
        pre_time = np.arange(len(after)) * dt
        signals = {}
        for name, times, values in (("torque", pre_time, torque), ("velocity", post_time, velocity),
                                    ("position", post_time, position)):
            keep = (times >= CORE[0]) & (times <= CORE[1])
            if name == "torque":
                keep &= times < CORE[1]
            assert np.all(np.isfinite(values))
            signals[name] = times[keep] - CORE[0], values[keep]
        return signals, targets, dict(case_index=case, case_name=str(data["names"][case]), physics_dt_s=dt,
                                      position_method="native steering angle minus fixed initial zero; no wrap/shift")


def make_figure(sim, real, rate):
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "svg.fonttype": "none"})
    fig, axes = plt.subplots(3, 1, figsize=(12.8, 10.2), sharex=True)
    fig.subplots_adjust(top=.87, bottom=.10, left=.10, right=.97, hspace=.20)
    labels = {"torque": ("Torque (Nm)", "Actuator torque before passive friction"),
              "velocity": ("Angular velocity (rad/s)", "Steering angular velocity"),
              "position": ("Relative position (rad)", "Steering angle from the fixed straight reference")}
    metrics = {}
    for ax, name in zip(axes, ("torque", "velocity", "position")):
        ts, ys = sim[name]
        tr, yr = real[name]
        sim_label, real_label = "Isaac Lab 400 Hz", f"Hardware replay, MIT {rate} Hz"
        if name == "torque":
            sim_label += " — submitted actuator effort"
            real_label += " — A1 current × Kt"
            ax.step(ts, ys, where="post", color="#1767a5", lw=1.3, label=sim_label)
        else:
            ax.plot(ts, ys, color="#1767a5", lw=1.6, label=sim_label)
        ax.plot(tr, yr, color="#d55e00", lw=1.1, label=real_label)
        ax.set(ylabel=labels[name][0], title=labels[name][1], xlim=(0, 5))
        combined = np.r_[ys, yr]
        lo, hi = float(combined.min()), float(combined.max())
        margin = max(.07 * (hi - lo), .0005)
        ax.set_ylim(lo - margin, hi + max(margin, .13 * (hi - lo)))
        ax.grid(alpha=.25)
        ax.legend(loc="lower right", fontsize=9, framealpha=.94)
        ax.text(.015, .965, f"Sim min/max: {ys.min():.3f} / {ys.max():.3f}    "
                f"Hardware min/max: {yr.min():.3f} / {yr.max():.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.83))
        metrics[name] = dict(unit=UNITS[name], sim_min=float(ys.min()), sim_max=float(ys.max()),
                             hardware_min=float(yr.min()), hardware_max=float(yr.max()),
                             sim_last=float(ys[-1]), hardware_last=float(yr[-1]),
                             sim_samples=len(ts), hardware_samples=len(tr))
    axes[-1].set_xlabel("Time since selected steering sequence begins (s)")
    fig.suptitle(f"Steering motor: simulation versus {rate} Hz hardware replay", fontsize=17, y=.965)
    fig.text(.5, .918, "Same 50 Hz-held model_2500 position targets · source episode 12.5–17.5 s\n"
             "Airborne corrected-zero MIT replay versus saved nominal fixed-base simulation · no time shift",
             ha="center", va="center", fontsize=10)
    fig.text(.10, .028, "Torque is simulation effort versus current-derived hardware estimate, not net torque "
             "after losses.\nPosition retains each run’s verified straight zero; nominal Kp/Kd = 10/0.2. "
             "Support and actuator dynamics are not fully matched.", fontsize=9)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(HERE / f"steering_sim_vs_replay_{rate}hz.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)
    return metrics


def main():
    sim, targets, sim_metadata = read_simulation()
    provenance = {str(SIM.relative_to(ROOT)): sha(SIM)}
    statistics, checks = {}, {}
    with (HERE / "plotted_samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("replay_packet_hz", "system", "channel", "core_time_s", "value", "unit"))
        for rate in STEMS:
            real, plan, hashes, checks[rate] = read_hardware(rate)
            max_target_diff = float(np.max(abs(targets - plan)))
            assert np.array_equal(targets, plan)  # Entire ramp/hold/core/final-hold requested plan.
            provenance.update(hashes)
            checks[rate]["maximum_sim_host_requested_target_difference_rad"] = max_target_diff
            statistics[rate] = make_figure(sim, real, rate)
            for system, signals in (("simulation", sim), ("hardware", real)):
                for channel, (times, values) in signals.items():
                    writer.writerows((rate, system, channel, float(t), float(y), UNITS[channel])
                                     for t, y in zip(times, values))
    metadata = dict(simulation=sim_metadata, core_replay_window_s=CORE, source_episode_window_s=(12.5, 17.5),
                    common_target_hz=50, hardware_packet_hz=list(STEMS), input_sha256=provenance,
                    torque_definition="Simulation submitted actuator effort; hardware signed A1 current times B0",
                    plot_processing="native fresh samples; no smoothing, time shift or fitted bias removal",
                    graph_time_zero="replay t=2 s, after 1-second ramp and 1-second hold",
                    position_zero="verified fixed straight zero; not rebased at core start",
                    simulation_or_hardware_run=False, statistics=statistics, checks=checks,
                    plotting_script_sha256=sha(Path(__file__)))
    (HERE / "statistics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(dict(statistics=statistics, checks=checks), indent=2))


if __name__ == "__main__":
    main()
