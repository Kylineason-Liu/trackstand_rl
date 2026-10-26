"""Quantify the original graphs and separately recorded joint friction forces."""

import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("original_analysis", HERE.parent / "analyze.py")
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


def main():
    summary, traces = {}, {}
    for tag in ("isaac400", "mujoco400", "mujoco800"):
        meta, data = original.load(tag)
        target = data["held_steering_target_rad"]
        position = data["position_after_rad"][..., 0]
        velocity = data["velocity_before_rad_s"]
        after_velocity = data["velocity_after_rad_s"]
        command = data["submitted_motor_torque_nm"]
        response = data[original.response_field(tag)][..., 1]
        moving = ((abs(velocity[..., 1]) > .1) & (abs(after_velocity[..., 1]) > .1)
                  & (np.sign(velocity[..., 1]) == np.sign(after_velocity[..., 1])))
        motoring = moving & (np.sign(command[..., 1]) == np.sign(velocity[..., 1])) & (abs(command[..., 1]) > .106)
        residual = response - command[..., 1] + .106 * np.sign(velocity[..., 1])
        item = dict(samples_per_motor=int(target.size),
                    steer_tracking_rms_rad=float(np.sqrt(np.mean((position - target) ** 2))),
                    steer_tracking_p95_rad=float(np.percentile(abs(position - target), 95)),
                    rear_command_response_max_gap_nm=float(np.max(abs(response - command[..., 1]))),
                    moving_samples=int(moving.sum()), motoring_samples=int(motoring.sum()),
                    motoring_abs_effort_reduction_median_nm=float(np.median(
                        (abs(command[..., 1]) - abs(response))[motoring])),
                    moving_response_minus_command_plus_coulomb_residual_within_1e4_percent=float(
                        np.mean(abs(residual[moving]) < 1e-4) * 100))
        detail = (slice(625, 875), slice(None), 0)
        kp = data["controller_kp"][..., 0][detail]
        kd = data["controller_kd"][..., 0][detail]
        steering_velocity = velocity[..., 0][detail]
        policy_targets = target[625:875, 0, 0].astype(float)
        energy = abs(np.fft.fft(policy_targets - policy_targets.mean())) ** 2
        frequency = np.fft.fftfreq(policy_targets.size, .02)
        item["env0_12p5_to_17p5_s"] = dict(
            target_peak_to_peak_rad=float(np.ptp(target[detail])),
            position_peak_to_peak_rad=float(np.ptp(position[detail])),
            tracking_rms_deg=float(np.degrees(np.sqrt(np.mean((position[detail] - target[detail]) ** 2)))),
            kp=float(kp.flat[0]), kd=float(kd.flat[0]),
            controller_damping_abs_torque_p95_nm=float(np.percentile(abs(kd * steering_velocity), 95)),
            passive_viscous_abs_torque_p95_nm=float(np.percentile(abs(.000910713492 * steering_velocity), 95)),
            steering_motor_torque_saturation_percent=float(
                np.mean(abs(command[..., 0][detail]) >= 5.6 - 1e-6) * 100),
            dry_friction_static_equivalent_error_deg=float(np.degrees(.046036056104 / kp.flat[0])),
            target_step_abs_p95_rad=float(np.percentile(abs(np.diff(policy_targets)), 95)),
            demeaned_target_fft_energy_10_to_25_hz_percent=float(
                100 * energy[abs(frequency) >= 10].sum() / energy.sum()),
            spectrum_definition="250 held target updates at 50Hz; demeaned full DFT; rectangular 5s window")
        forces = None
        if tag != "isaac400":
            verification = json.loads((HERE / tag / "verification.json").read_text())
            assert verification["original_13_arrays_byte_identical"] and verification["production_sources_unchanged"]
            with np.load(HERE / tag / "forces.npz") as archive:
                forces = {key: archive[key] for key in archive.files}
            np.testing.assert_array_equal(forces["actuator_nm"], command)
            expected_passive = -velocity * np.array([.000910713492, 0.])
            np.testing.assert_allclose(forces["passive_nm"], expected_passive, rtol=0, atol=1e-12)
            friction = forces["dry_friction_nm"]
            item["force_verification"] = dict(
                original_13_arrays_byte_identical=True,
                damping=[.000910713492, 0.], frictionloss=[.046036056104, .106],
                friction_min_nm=friction.min((0, 1, 2)).tolist(),
                friction_max_nm=friction.max((0, 1, 2)).tolist(),
                moving_rear_abs_friction_median_nm=float(np.median(abs(friction[..., 1][moving]))),
                moving_rear_abs_friction_saturated_percent=float(np.mean(
                    abs(abs(friction[..., 1][moving]) - .106) < 1e-10) * 100),
                moving_rear_friction_opposes_pre_step_velocity_percent=float(np.mean(
                    (friction[..., 1] * velocity[..., 1])[moving] <= 1e-10) * 100),
                passive_damping_equation_max_error_nm=float(np.max(abs(forces["passive_nm"] - expected_passive))))
            by_speed = {}
            for lower in (.1, .5, 1.):
                keep = ((abs(velocity[..., 1]) > lower) & (abs(after_velocity[..., 1]) > lower)
                        & (np.sign(velocity[..., 1]) == np.sign(after_velocity[..., 1])))
                by_speed[str(lower)] = dict(
                    samples=int(keep.sum()), abs_friction_median_nm=float(np.median(abs(friction[..., 1][keep]))),
                    abs_friction_at_0p106_percent=float(np.mean(
                        abs(abs(friction[..., 1][keep]) - .106) < 1e-10) * 100))
            item["force_verification"]["rear_by_minimum_pre_and_post_speed_rad_s"] = by_speed
            # Choose a direct moving/motoring illustration, with no fabricated .106 subtraction.
            matches = motoring & (abs(friction[..., 1] + .106 * np.sign(velocity[..., 1])) < 1e-10)
            step, sub, env = np.argwhere(matches)[len(np.argwhere(matches)) // 2]
            item["measured_motoring_example"] = dict(
                env=int(env), command_time_s=float(step * .02 + sub * meta["physics_dt_s"]),
                velocity_before_rad_s=float(velocity[step, sub, env, 1]),
                submitted_motor_nm=float(command[step, sub, env, 1]),
                actuator_nm=float(forces["actuator_nm"][step, sub, env, 1]),
                joint_dry_friction_nm=float(friction[step, sub, env, 1]),
                motor_plus_joint_losses_nm=float((forces["actuator_nm"] + forces["passive_nm"]
                                                 + friction)[step, sub, env, 1]))
        local = dict(
            time_start=np.arange(3000 * meta["decimation"]) * meta["physics_dt_s"],
            time_end=(np.arange(3000 * meta["decimation"]) + 1) * meta["physics_dt_s"],
            target=target[:, :, 0].ravel(), position=position[:, :, 0].ravel(),
            rear_command=command[:, :, 0, 1].ravel(), rear_response=response[:, :, 0].ravel(),
            steer_torque=command[:, :, 0, 0].ravel(),
            steer_passive=-.000910713492 * velocity[:, :, 0, 0].ravel(),
            spawn=np.r_[0., (np.flatnonzero(data["done"][:, 0]) + 1) * .02])
        if forces is not None:
            local.update(rear_dry=forces["dry_friction_nm"][:, :, 0, 1].ravel(),
                         rear_after_joint_losses=(forces["actuator_nm"] + forces["passive_nm"]
                                                  + forces["dry_friction_nm"])[:, :, 0, 1].ravel(),
                         steer_dry=forces["dry_friction_nm"][:, :, 0, 0].ravel())
        traces[tag] = local
        summary[tag] = item
        del data, forces
    (HERE / "statistics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(HERE / "env0_traces.npz", **{
        f"{tag}_{key}": value for tag, trace in traces.items() for key, value in trace.items()})
    for window, suffix in (((12.5, 17.5), "5s"), ((12.5, 13.), "detail")):
        fig, axes = plt.subplots(3, 3, figsize=(22, 12), sharex=True, layout="constrained")
        for column, (tag, trace) in enumerate(traces.items()):
            titles = ["Isaac 400 Hz", "MuJoCo 400 Hz", "MuJoCo 800 Hz"]
            axes[0, column].set_title(titles[column])
            mask = (trace["time_start"] >= window[0]) & (trace["time_start"] < window[1])
            x = trace["time_start"][mask]
            axes[0, column].plot(x, trace["rear_command"][mask], color="tab:orange", label="Motor command")
            axes[0, column].plot(x, trace["rear_response"][mask], color="tab:green", alpha=.8,
                                 label="Solver axis effort" if tag == "isaac400" else "Motor actuator output")
            if tag != "isaac400":
                axes[0, column].plot(x, trace["rear_after_joint_losses"][mask], color="tab:purple",
                                     label="Motor + damping + joint dry friction")
            axes[1, column].step(x, trace["target"][mask], where="post", color="tab:orange", label="Held target")
            mask_end = (trace["time_end"] >= window[0]) & (trace["time_end"] < window[1])
            axes[1, column].plot(trace["time_end"][mask_end], trace["position"][mask_end],
                                 color="tab:green", label="Actual position (step end)")
            axes[2, column].plot(x, trace["steer_torque"][mask], label="Submitted PD torque", alpha=.8)
            axes[2, column].plot(x, trace["steer_passive"][mask], label="Passive viscous torque", alpha=.9)
            if tag != "isaac400":
                axes[2, column].plot(x, trace["steer_dry"][mask], label="Isolated joint dry-friction force")
            labels = ("Rear torque / effort (Nm)", "Steering position (rad)", "Steering torque (Nm)")
            for row, ylabel in enumerate(labels):
                ax = axes[row, column]
                ax.set_ylabel(ylabel)
                ax.set_xlim(*window)
                ax.grid(alpha=.25)
                low = min(line.get_ydata().min() for line in ax.lines)
                high = max(line.get_ydata().max() for line in ax.lines)
                ax.margins(y=.12)
                ax.text(.01, .98, f"min {low:.5f} | max {high:.5f}", transform=ax.transAxes,
                        va="top", fontsize=8, bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
                ax.legend(loc="lower right", fontsize=8)
                for spawn in trace["spawn"]:
                    if window[0] <= spawn <= window[1]:
                        ax.axvline(spawn, color="red", linewidth=.9)
                if row == 2:
                    ax.set_xlabel("Recording time (s)")
        fig.suptitle("Model 2500, DR, env 0: motor output versus separately applied joint losses\n"
                     "Different engine trajectories; purple is NOT a total net/contact torque "
                     "or an Isaac sensor equivalent")
        original.dr_annotations.append_blocks(fig, [(tag, 0, window) for tag in original.TAGS])
        for extension in ("png", "svg", "pdf"):
            fig.savefig(HERE / f"friction_and_tracking_{suffix}.{extension}", dpi=160)
        plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
