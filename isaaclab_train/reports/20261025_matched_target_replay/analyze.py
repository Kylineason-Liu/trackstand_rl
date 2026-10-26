"""Validate matched targets, quantify response, and plot the controlled replay."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
TAGS = ("isaac400", "mujoco400", "mujoco800", "isaac800")
LABELS = ("Isaac 400 Hz", "MuJoCo 400 Hz", "MuJoCo 800 Hz", "Isaac 800 Hz (timing variant)")
COLORS = ("tab:green", "tab:orange", "tab:red", "tab:purple")
CASE_LABELS = ("1 Hz sine, 0.1 rad", "25 Hz alternating target, +/-0.1 rad",
               "Recorded Isaac DR targets", "Recorded MuJoCo 400 Hz DR targets", "Recorded MuJoCo 800 Hz DR targets")


def archive(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def validate(tag, fixture, data, bank):
    n = 10 if fixture == "airborne" else 2
    substeps = int(tag.removeprefix("isaac").removeprefix("mujoco")) // 50
    np.testing.assert_array_equal(data["target_rad"], bank[f"{fixture}_target_rad"])
    np.testing.assert_array_equal(data["rear_nm"], bank[f"{fixture}_rear_nm"])
    assert data["position_after_rad"].shape == (400, substeps, n, 2)
    for values in data.values():
        if values.dtype.kind in "fiu":
            assert np.isfinite(values).all()
    np.testing.assert_allclose(data["position_before_rad"][0, 0], 0., atol=1e-7, rtol=0)
    np.testing.assert_allclose(data["velocity_before_rad_s"][0, 0], 0., atol=1e-7, rtol=0)
    expected = np.empty_like(data["computed_nm"])
    expected[..., 0] = 10. * (data["target_rad"][:, None, :] - data["position_before_rad"][..., 0])
    expected[..., 0] -= .2 * data["velocity_before_rad_s"][..., 0]
    expected[..., 1] = data["rear_nm"][:, None, :]
    np.testing.assert_allclose(data["computed_nm"], expected, rtol=1e-5, atol=3e-6)
    np.testing.assert_allclose(data["submitted_nm"], np.clip(expected, [-5.6, -6.28], [5.6, 6.28]),
                               rtol=1e-5, atol=3e-6)
    np.testing.assert_array_equal(data["submitted_nm"][..., 1], np.repeat(data["rear_nm"][:, None, :], substeps, 1))
    if tag.startswith("mujoco"):
        np.testing.assert_array_equal(data["submitted_nm"], data["actuator_nm"])
        if fixture == "airborne":
            assert not data["contacts"].any()
            np.testing.assert_array_equal(data["dry_friction_nm"][:, :, 1::2, 0], 0.)
            np.testing.assert_array_equal(data["passive_nm"][:, :, 1::2, 0], 0.)


def phase_measure(time, reference, actual, frequency):
    design = np.c_[np.sin(2 * np.pi * frequency * time),
                   np.cos(2 * np.pi * frequency * time), np.ones(time.size)]
    a = np.linalg.lstsq(design, reference, rcond=None)[0]
    b = np.linalg.lstsq(design, actual, rcond=None)[0]
    phase = np.arctan2(a[1], a[0]) - np.arctan2(b[1], b[0])
    lag = np.arctan2(np.sin(phase), np.cos(phase)) / (2 * np.pi * frequency) * 1000
    return dict(fundamental_gain=float(np.hypot(*b[:2]) / np.hypot(*a[:2])),
                phase_equivalent_lag_ms=float(lag),
                note="Fit to actual held reference, no response time shift; 3-7s window")


def metrics(data, fixture):
    dt = float(data["physics_dt_s"])
    t = (np.arange(400 * data["position_after_rad"].shape[1]) + 1) * dt
    output = {}
    for index, name in enumerate(data["names"]):
        target = np.repeat(data["target_rad"][:, index], data["position_after_rad"].shape[1])
        q = data["position_after_rad"][:, :, index, 0].ravel()
        v = data["velocity_after_rad_s"][:, :, index, 0].ravel()
        if fixture == "airborne":
            keep = (t > 2.) & (t <= 7.)
        else:
            keep = t <= .25  # common interval BEFORE every run's first 30-degree crossing
        error = q[keep] - target[keep]
        item = dict(tracking_rms_rad=float(np.sqrt(np.mean(error ** 2))),
                    tracking_rms_deg=float(np.degrees(np.sqrt(np.mean(error ** 2)))),
                    tracking_p95_rad=float(np.percentile(abs(error), 95)),
                    target_peak_to_peak_rad=float(np.ptp(target[keep])),
                    position_peak_to_peak_rad=float(np.ptp(q[keep])),
                    maximum_native_speed_rad_s=float(np.max(abs(v[keep]))),
                    motor_torque_saturated_percent=float(np.mean(
                        abs(data["submitted_nm"][:, :, index, 0].ravel()[keep]) >= 5.6 - 1e-6) * 100),
                    window_s=[2., 7.] if fixture == "airborne" else [0., .25])
        if fixture == "airborne":
            steady = t > 7.5
            item["final_hold_mean_error_rad"] = float(np.mean(q[steady] - target[steady]))
            item["final_hold_mean_motor_torque_nm"] = float(np.mean(
                data["submitted_nm"][:, :, index, 0].ravel()[steady]))
            item["final_hold_mean_abs_speed_rad_s"] = float(np.mean(abs(v[steady])))
            if "dry_friction_nm" in data:
                item["final_hold_mean_dry_friction_nm"] = float(np.mean(
                    data["dry_friction_nm"][:, :, index, 0].ravel()[steady]))
            if index < 4:
                fit = (t > 3.) & (t <= 7.)
                item["fundamental_response"] = phase_measure(t[fit], target[fit], q[fit], 1. if index < 2 else 25.)
        else:
            roll = data["roll_rad"][:, :, index].ravel()
            bad = np.flatnonzero(abs(roll) > np.pi / 6)
            item["first_30deg_time_s"] = float(t[bad[0]]) if bad.size else None
            item["warning"] = "Open-loop copied commands from a different DR trajectory; not a balancing evaluation"
        output[str(name)] = item
    return output


def finish_panel(ax, window, unit):
    ax.grid(alpha=.25)
    ax.set_xlim(*window)
    ax.set_ylabel(unit)
    # Spawn/fall axvlines use axes-fraction Y coordinates [0,1], not motor
    # measurements. Exclude those artists from physical extrema and scaling.
    values = np.concatenate([line.get_ydata() for line in ax.lines if line.get_transform() is ax.transData])
    lo, hi = float(values.min()), float(values.max())
    pad = max((hi - lo) * .14, 1e-5)
    ax.set_ylim(lo - pad, hi + pad)
    ax.text(.99, .02, f"min {lo:.4g} | max {hi:.4g}", ha="right", transform=ax.transAxes, fontsize=7,
            bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
    if window[0] == 0:
        ax.axvline(0, color="red", linewidth=1.4, label="Initial spawn")
    ax.legend(fontsize=7, loc="upper right")


def save(fig, name):
    for extension in ("png", "svg", "pdf"):
        fig.savefig(HERE / f"{name}.{extension}", dpi=160)
    plt.close(fig)


def plot_positions(runs, window, name):
    fig, axes = plt.subplots(5, 2, figsize=(20, 22), layout="constrained")
    for case in range(5):
        for loss in range(2):
            index, ax = 2 * case + loss, axes[case, loss]
            base = runs[TAGS[0]]["airborne"]
            policy_time = np.arange(400) * .02
            keep = (policy_time >= window[0]) & (policy_time < window[1])
            held = base["target_rad"][keep, index]
            ax.step(np.r_[policy_time[keep], window[1]], np.r_[held, held[-1]], where="post",
                    color="black", ls="--", linewidth=1.2, label="Identical held position target")
            for tag, label, color in zip(TAGS, LABELS, COLORS):
                data = runs[tag]["airborne"]
                t = (np.arange(data["position_after_rad"].shape[0] * data["position_after_rad"].shape[1]) + 1)
                t = t * float(data["physics_dt_s"])
                keep = (t >= window[0]) & (t <= window[1])
                q = data["position_after_rad"][:, :, index, 0].ravel()
                ax.plot(t[keep], q[keep], color=color, linewidth=1., alpha=.8, label=label)
            ax.set_title(f"{CASE_LABELS[case]} | steering friction {'ON' if loss == 0 else 'OFF'}", fontsize=10)
            finish_panel(ax, window, "Steering position (rad)")
            ax.set_xlabel("Time from replay spawn (s)")
    fig.suptitle("Identical 50 Hz-held targets: supported bike, wheels airborne, same nominal Kp=10/Kd=0.2\n"
                 "No policy inference or DR resampling; 2-7s is replay core; only steering losses change in OFF pairs")
    save(fig, name)


def plot_torque(runs):
    fig, axes = plt.subplots(5, 2, figsize=(20, 21), layout="constrained")
    window = (4., 4.5)
    for case in range(5):
        index = 2 * case
        for tag, label, color in zip(TAGS, LABELS, COLORS):
            d = runs[tag]["airborne"]
            t = np.arange(400 * d["submitted_nm"].shape[1]) * float(d["physics_dt_s"])
            keep = (t >= window[0]) & (t < window[1])
            for motor in (0, 1):
                axes[case, motor].step(t[keep], d["submitted_nm"][:, :, index, motor].ravel()[keep],
                                       where="post", color=color, alpha=.8, label=label)
        for motor in (0, 1):
            ax = axes[case, motor]
            ax.set_title(f"{CASE_LABELS[case]} | {'steering PD torque' if motor == 0 else 'rear submitted torque'}")
            finish_panel(ax, window, "Motor command (Nm)")
            ax.set_xlabel("Time from replay spawn (s)")
    fig.suptitle("Matched position/rear-torque inputs; steering friction ON\n"
                 "Rear commands coincide; steering PD torques can differ because feedback states differ")
    save(fig, "matched_torque_detail")


def plot_ground(runs):
    fig, axes = plt.subplots(4, 2, figsize=(20, 14), layout="constrained")
    for index in range(2):
        base = runs[TAGS[0]]["ground"]
        for row, request in ((0, "target_rad"), (2, "rear_nm")):
            held = base[request][:25, index]
            axes[row, index].step(np.r_[np.arange(25) * .02, .5], np.r_[held, held[-1]], where="post",
                                  color="black", ls="--", label="Identical input")
        for tag, label, color in zip(TAGS, LABELS, COLORS):
            data = runs[tag]["ground"]
            h = float(data["physics_dt_s"])
            t = (np.arange(400 * data["position_after_rad"].shape[1]) + 1) * h
            keep = t <= .5
            series = [data["position_after_rad"][:, :, index, 0].ravel(),
                      data["submitted_nm"][:, :, index, 0].ravel(),
                      data["submitted_nm"][:, :, index, 1].ravel(),
                      np.degrees(data["roll_rad"][:, :, index].ravel())]
            for row, values in enumerate(series):
                tx = t - h if row in (1, 2) else t
                if row in (1, 2):
                    axes[row, index].step(tx[keep], values[keep], where="post", color=color, alpha=.8, label=label)
                else:
                    axes[row, index].plot(tx[keep], values[keep], color=color, alpha=.8, label=label)
            crossing = np.flatnonzero(abs(series[3]) > 30.)
            if crossing.size:
                for ax in axes[:, index]:
                    ax.axvline(t[crossing[0]], color=color, ls=":", alpha=.6)
        axes[0, index].set_title(str(base["names"][index]))
        for row, unit in enumerate(("Steering position (rad)", "Steering command (Nm)",
                                     "Rear command (Nm)", "Roll (degrees)")):
            finish_panel(axes[row, index], (0., .5), unit)
            axes[row, index].set_xlabel("Time from replay spawn (s)")
    fig.suptitle("Original floating bike, identical recorded commands from spawn, nominal parameters\n"
                 "Colored dotted lines = first |roll|>30deg; later curves include falling. No feedback policy/reset.")
    save(fig, "ground_replay_first_half_second")


def main():
    bank = archive(HERE / "inputs.npz")
    inputs = json.loads((HERE / "inputs.json").read_text())
    assert hashlib.sha256((HERE / "inputs.npz").read_bytes()).hexdigest() == inputs["input_sha256"]
    runs, result = {}, {}
    for tag in TAGS:
        meta = json.loads((HERE / tag / "metadata.json").read_text())
        assert meta["completed"] and not meta["policy_inference"] and not meta["dr_resampling"]
        runs[tag] = {fixture: archive(HERE / tag / f"{fixture}.npz") for fixture in ("airborne", "ground")}
        result[tag] = {}
        for fixture, data in runs[tag].items():
            validate(tag, fixture, data, bank)
            result[tag][fixture] = metrics(data, fixture)
    (HERE / "statistics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot_positions(runs, (0., 8.), "matched_position_full")
    plot_positions(runs, (2., 7.), "matched_position_core")
    plot_positions(runs, (4., 4.5), "matched_position_detail")
    plot_positions(runs, (7., 8.), "matched_position_final_hold")
    plot_torque(runs)
    plot_ground(runs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
