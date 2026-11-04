"""Compare identical initial rear commands over the first held 20-ms interval."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT = Path(__file__).resolve().parent
DIRECTORIES = {400: REPORT / "hz400", 800: REPORT / "hz800",
               1200: REPORT.parent / "20261104_v2_50100_dr_pure_survival_eval"}


def main():
    # Environment 6 has the most first-interval torque sign changes at 400 Hz.
    # This example selection is disclosed; aggregate counts use all 64 draws.
    example, initial, action = {}, None, None
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey="row")
    for column, (hz, directory) in enumerate(DIRECTORIES.items()):
        with np.load(next(directory.glob("*_evaluation.npz"))) as archive:
            obs, raw = archive["observation"][0, 6], archive["raw_action"][0, 6]
            if initial is None:
                initial, action = obs.copy(), raw.copy()
            np.testing.assert_array_equal(obs, initial)
            np.testing.assert_array_equal(raw, action)
            speed = archive["physics_drive_velocity_rad_s"][0, 6]
            torque = archive["physics_motor_torque_nm"][0, 6, :, 1]
            target = float(archive["held_drive_target_rad_s"][0, 6])
        time_ms = np.arange(len(speed)) / hz * 1000
        axes[0, column].plot(time_ms, speed, ".-", label="Simulated rear velocity", color="#1f77b4")
        axes[0, column].axhline(target, color="#d62728", ls="--", label="Held velocity target")
        axes[1, column].plot(time_ms, torque, ".-", color="#2ca02c", label="Applied motor torque")
        axes[1, column].axhline(0., color="gray", lw=.7)
        axes[0, column].set_title(f"{hz} Hz physics / PD")
        for ax in axes[:, column]:
            ax.set_xlim(0, 20)
            ax.grid(alpha=.25)
            ax.legend(loc="lower right", fontsize=8)
        axes[1, column].set_xlabel("Time since first policy command (ms)")
        example[str(hz)] = dict(target_rad_s=target, sample_times_ms=time_ms.tolist(),
                                rear_velocity_rad_s=speed.tolist(), rear_torque_nm=torque.tolist())
    axes[0, 0].set_ylabel("Rear velocity (rad/s)")
    axes[1, 0].set_ylabel("Motor torque (Nm)")
    fig.suptitle("model_50100 | matched DR environment 6 | identical initial observation/action\n"
                 "50-Hz-held target, 200-Hz integral; velocity sampled before each physics step")
    fig.tight_layout()
    fig.savefig(REPORT / "startup_env6.png", dpi=160)
    plt.close(fig)
    (REPORT / "startup_env6.json").write_text(json.dumps(dict(
        env_id=6, selection="largest first-20-ms rear torque sign-change count at 400 Hz",
        identical_initial_observation_and_raw_action=True, traces=example), indent=2) + "\n")


if __name__ == "__main__":
    main()
