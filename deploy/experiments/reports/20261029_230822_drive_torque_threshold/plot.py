"""Plot fresh feedback at its recorded timestamps; no filtering or response alignment."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
STATS = json.loads((HERE / "statistics.json").read_text())
TRIALS = {r["trial_id"]: r for r in STATS["trials"]}
COLORS = {False: "#bd5b20", True: "#176f99"}


def save(fig, stem):
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(HERE / f"{stem}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    with (HERE / "selected_trial_traces.csv").open() as stream:
        traces = list(csv.DictReader(stream))
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, sharey="row", layout="constrained")
    for col, ids in enumerate(((1, 11), (7, 27))):
        for trial in ids:
            info = TRIALS[trial]
            color = COLORS[info["sustained_rotation"]]
            label = (f"Trial {trial}: start {info['start_phase_deg']:.1f} degrees, "
                     f"{info['temperature_min_c']}-{info['temperature_max_c']} C")
            for row, opcode in enumerate(("A1", "A2", "A3")):
                values = [r for r in traces if int(r["trial"]) == trial and r["opcode"] == opcode]
                axes[row, col].plot([float(r["time_s"]) for r in values],
                                    [float(r["value"]) for r in values],
                                    color=color, lw=1, alpha=.85, label=label)
                axes[row, col].grid(alpha=.2)
            axes[2, col].text(.03, .92 if not info["sustained_rotation"] else .82,
                              f"Trial {trial} last displacement: {info['displacement_rad']:.4f} rad",
                              color=color, transform=axes[2, col].transAxes, fontsize=9)
        target = TRIALS[ids[0]]["encoded_torque_nm"]
        axes[0, col].axhline(target, color="#555555", ls="--", lw=1, label="Encoded C0 target")
        axes[0, col].set_title(f"Same +{target:.2f} Nm request; different starting conditions")
        axes[0, col].legend(loc="upper left", fontsize=8)
        axes[2, col].set_xlabel("Time since actual first C0 send (s)")
        axes[2, col].set_xlim(0, 5)
    for row, label in enumerate(("A1 current x B0 Kt (Nm)", "A2 velocity (rad/s)",
                                 "A3 displacement from trial start (rad)")):
        axes[row, 0].set_ylabel(label)
    fig.suptitle("Rear wheel airborne/free and untouched | C0 nominal 50 Hz | 5 s holds\n"
                 "Fresh replies only; torque is current-derived, not measured shaft torque", fontsize=13)
    save(fig, "same_command_response")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col", sharey="row", layout="constrained")
    for col, direction in enumerate(("forward", "backward")):
        for level in (.12, .13, .14, .15, .16):
            group = [r for r in TRIALS.values() if r["direction"] == direction and r["torque_level_nm"] == level]
            for r in group:
                # Fixed horizontal offsets reveal all five trials; they do not alter any measured quantity.
                x = level + (r["repeat"] - 3) * .00065
                color = COLORS[r["sustained_rotation"]]
                axes[0, col].scatter(x, abs(r["displacement_rad"]), c=color, s=32, zorder=3)
                axes[1, col].scatter(x, abs(r["late_mean_velocity_rad_s"]), c=color, s=32, zorder=3)
            n = sum(r["sustained_rotation"] for r in group)
            axes[0, col].text(level, 400, f"{n}/5", ha="center", fontsize=10)
        axes[0, col].set_title(direction.capitalize())
        axes[0, col].set_yscale("log")
        axes[0, col].set_ylim(.0004, 800)
        axes[0, col].axhline(2 * np.pi, ls=":", color="#666666", lw=.8)
        axes[1, col].set_ylim(-1, 55)
        axes[1, col].set_xticks([.12, .13, .14, .15, .16])
        axes[1, col].set_xlim(.116, .164)
        axes[1, col].set_xlabel("Requested torque magnitude (Nm); horizontal offsets separate repeats")
        for row in range(2):
            axes[row, col].grid(alpha=.2)
    axes[0, 0].set_ylabel("Absolute net displacement in 5 s (rad, log scale)")
    axes[1, 0].set_ylabel("Absolute mean A2 velocity over 4-5 s (rad/s)")
    for flag, label in ((True, "Sustained rotation (38)"), (False, "No sustained rotation (12)")):
        axes[1, 0].scatter([], [], c=COLORS[flag], label=label)
    axes[1, 0].legend(loc="upper left", fontsize=9)
    fig.suptitle("50 near-rest starts: one torque magnitude does not determine the outcome\n"
                 "Top annotations: rotating trials / five starts; dotted line: one revolution", fontsize=13)
    save(fig, "trial_outcomes")


if __name__ == "__main__":
    main()
