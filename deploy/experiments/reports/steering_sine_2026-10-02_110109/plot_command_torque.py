"""Plot steering commands and feedback, together and as four separate graphs.

Source ROS first; dependencies are mcap, numpy, PyYAML, and matplotlib.
No ROS node is initialized and no motor commands are sent.
"""

import argparse
import json
from pathlib import Path
import runpy
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPORT_DIR = Path(__file__).resolve().parent
REPOSITORY = REPORT_DIR.parents[2]


def save_separate_plots(command, state, steering_torque, duration_s, position_bound, bag_name, output_dir):
    """Preserve each recorded signal on its own graph, using one common time range."""
    signals = (
        (
            "steering_position_command", "Steering position command", "Position target (rad)",
            command[:, 0], command[:, 2], "#1967b3", position_bound, True,
            "/trackstand/motor_command: data[1]",
        ),
        (
            "steering_position_feedback", "Steering position feedback", "Position (rad)",
            state[:, 0], state[:, 1], "#168347", position_bound, False,
            "/trackstand/motor_state: data[0]",
        ),
        (
            "steering_velocity_feedback", "Steering velocity feedback", "Angular velocity (rad/s)",
            state[:, 0], state[:, 3], "#7745a6", None, False,
            "/trackstand/motor_state: data[2]",
        ),
        (
            "steering_torque_feedback", "Steering torque feedback", "Current-derived torque (N m)",
            steering_torque[:, 0], steering_torque[:, 3], "#db681e", None, False,
            "/trackstand/motor_torque: steering_joint, effort[0]; measured current x B0 constant, sign-adjusted",
        ),
    )
    for filename, title, ylabel, timestamps, values, color, bound, is_step, source in signals:
        figure, axis = plt.subplots(figsize=(14, 3.5), layout="constrained")
        if is_step:
            axis.step(timestamps, values, where="post", color=color, linewidth=1.5)
        else:
            axis.plot(timestamps, values, color=color, linewidth=0.85)
        if bound is None:
            bound = max(0.01, 1.15 * np.max(np.abs(values)))
        axis.set_xlim(0.0, duration_s)
        axis.set_ylim(-bound, bound)
        axis.set_xlabel("Elapsed bag-record time (s)")
        axis.set_ylabel(ylabel)
        axis.set_title(f"{title}\n{bag_name}")
        axis.grid(alpha=0.2)
        figure.supxlabel(
            f"{source}\n{len(values):,} raw samples; no smoothing, decimation, or time shift.", fontsize=9,
        )
        for extension in ("png", "svg"):
            destination = output_dir / f"{filename}.{extension}"
            figure.savefig(destination, dpi=180, facecolor="white")
            print(destination)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bag", type=Path,
        default=REPOSITORY / "rosbags" / "rosbag2_2026_10_02-11_01_09",
    )
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    decoder_path = REPORT_DIR.parent / "rosbag_ABC_2026-10-02" / "decode_bags.py"
    decoder = runpy.run_path(str(decoder_path))
    with tempfile.TemporaryDirectory(prefix="trackstand_sine_plot.") as temporary:
        cache_dir = Path(temporary)
        decoder["decode"]("SINE", args.bag.resolve(), cache_dir)
        details = json.loads((cache_dir / "SINE.json").read_text())
        with np.load(cache_dir / "SINE.npz") as arrays:
            command = arrays["motor_command"].copy()
            state = arrays["motor_state"].copy()
            torque = arrays["motor_torque"].copy()

    if details["count_differences"]:
        raise ValueError(f"Metadata/decoded message counts differ: {details['count_differences']}")
    if command.shape[1] != 4 or state.shape[1] != 6 or torque.shape[1] != 4:
        raise ValueError("Unexpected command, motor-state, or per-motor torque cache shape")
    # Decoder verifies JointState names; code 0 is steering_joint, 1 is rear_wheel_joint.
    steering_torque = torque[torque[:, 2] == 0.0]
    if len(steering_torque) == 0:
        raise ValueError("No steering torque samples in the bag")
    if not all(np.all(np.isfinite(samples)) for samples in (command, state, steering_torque)):
        raise ValueError("Non-finite plotted samples")

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False})
    figure, position_axis = plt.subplots(figsize=(14, 5.7), layout="constrained")
    torque_axis = position_axis.twinx()
    position_color = "#1967b3"
    feedback_color = "#168347"
    torque_color = "#db681e"
    torque_line, = torque_axis.plot(
        steering_torque[:, 0], steering_torque[:, 3], color=torque_color,
        linewidth=0.85, alpha=0.75, label="Steering torque feedback (N m; right axis)",
    )
    # Put the command above the denser raw torque trace, without obscuring either axis.
    position_axis.set_zorder(torque_axis.get_zorder() + 1)
    position_axis.patch.set_visible(False)
    command_line, = position_axis.step(
        command[:, 0], command[:, 2], where="post", color=position_color,
        linewidth=1.9, label="Steering position command (rad; left axis)",
    )
    # motor_io publishes [steering_pos, rear_pos, steering_speed, rear_speed, time].
    # The decoder prepends elapsed bag time, so column 1 is signed steering angle.
    feedback_line, = position_axis.plot(
        state[:, 0], state[:, 1], color=feedback_color, linestyle="--",
        linewidth=1.5, label="Steering position feedback (rad; left axis)",
    )
    position_axis.set_xlim(0.0, details["duration_s"])
    position_bound = 1.15 * max(1.0, np.max(np.abs(command[:, 2])), np.max(np.abs(state[:, 1])))
    position_axis.set_ylim(-position_bound, position_bound)
    torque_bound = 1.15 * np.max(np.abs(steering_torque[:, 3]))
    torque_axis.set_ylim(-max(0.01, torque_bound), max(0.01, torque_bound))
    position_axis.set_xlabel("Elapsed bag-record time (s)")
    position_axis.set_ylabel("Steering position (rad)", color=position_color)
    torque_axis.set_ylabel("Current-derived steering torque (N m)", color=torque_color)
    position_axis.tick_params(axis="y", colors=position_color)
    torque_axis.tick_params(axis="y", colors=torque_color)
    position_axis.grid(alpha=0.2)
    position_axis.set_title(
        "Steering sine experiment: command, position feedback, and torque\n"
        "rosbag2_2026_10_02-11_01_09",
        pad=12,
    )
    position_axis.legend(handles=[command_line, feedback_line, torque_line], loc="upper right", framealpha=0.95)
    figure.supxlabel(
        f"Raw samples: {len(command):,} commands, {len(state):,} position readings, "
        f"{len(steering_torque):,} torque readings. No smoothing or time shift.\n"
        "Torque is measured Q-axis current x B0 torque constant, sign-adjusted; not a shaft torque sensor.",
        fontsize=9,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        destination = args.output_dir / f"steering_command_torque.{extension}"
        figure.savefig(destination, dpi=180, facecolor="white")
        print(destination)
    plt.close(figure)

    save_separate_plots(
        command, state, steering_torque, details["duration_s"], position_bound, args.bag.name, args.output_dir,
    )

    provenance = {
        "bag": str(args.bag.resolve()), "files": details["files"],
        "duration_s": details["duration_s"], "topic_counts": details["counts"],
        "metadata_count_differences": details["count_differences"],
        "time_axis": "MCAP log_time minus metadata bag start, seconds; no artificial time shift",
        "command": {"topic": "/trackstand/motor_command", "field": "data[1]", "units": "rad",
                    "samples": len(command), "range": [float(command[:, 2].min()), float(command[:, 2].max())]},
        "position_feedback": {
            "topic": "/trackstand/motor_state", "field": "data[0]", "units": "rad",
            "samples": len(state), "range": [float(state[:, 1].min()), float(state[:, 1].max())],
        },
        "velocity_feedback": {
            "topic": "/trackstand/motor_state", "field": "data[2]", "units": "rad/s",
            "samples": len(state), "range": [float(state[:, 3].min()), float(state[:, 3].max())],
        },
        "torque": {"topic": "/trackstand/motor_torque", "joint": "steering_joint", "field": "effort[0]",
                   "units": "N m", "samples": len(steering_torque),
                   "range": [float(steering_torque[:, 3].min()), float(steering_torque[:, 3].max())]},
        "smoothing": "none", "decimation": "none", "original_bag_modified": False,
    }
    (args.output_dir / "plot_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
