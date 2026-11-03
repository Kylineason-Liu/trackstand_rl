"""Launch the locomotive (CycleRL riding policy) stack: live motors unless dry_run.

Every launch also records a session folder (disable with ``record:=false``)::

    <log_dir>/locomotive_YYYYmmdd_HHMMSS/
        bag/          ros2 bag of all topics (split every 5 min, so a power cut loses one segment)
        ros_logs/     every node's text log (startup B3 / original drive F0 lines included)
        config/       params.yaml + locomotive.yaml exactly as loaded
        session.json  start time, git commit/dirty state, model + SHA, dry_run, record command

``log_dir`` defaults to ``~/trackstand_logs``.
"""

import datetime
import json
import os
import shutil
import socket
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable
from launch_ros.actions import Node
import yaml


# Rosbag split size (s): bounds what an unclean stop (power loss) can corrupt.
BAG_SPLIT_S = 300


def _git_state(path: str) -> dict:
    """Best-effort provenance; a missing git checkout is recorded, not fatal."""
    directory = os.path.dirname(os.path.realpath(path))
    try:
        commit = subprocess.run(["git", "-C", directory, "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=5, check=True).stdout.strip()
        branch = subprocess.run(["git", "-C", directory, "rev-parse", "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        # Whole deploy package (config/ lives one level below it).
        dirty = subprocess.run(["git", "-C", directory, "status", "--porcelain", "--", ".."],
                               capture_output=True, text=True, timeout=5, check=True).stdout.splitlines()
        return {"commit": commit, "branch": branch, "uncommitted_files": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "note": "git state unavailable"}


def _session(context, parameters):
    if context.launch_configurations["record"].lower() not in ("true", "1", "yes"):
        return []
    stamp = datetime.datetime.now()
    root = os.path.expanduser(context.launch_configurations["log_dir"])
    session = os.path.join(root, f"locomotive_{stamp:%Y%m%d_%H%M%S}")
    os.makedirs(os.path.join(session, "config"), exist_ok=True)
    loaded = {}
    for path in parameters:
        shutil.copy2(path, os.path.join(session, "config", os.path.basename(path)))
        with open(path) as stream:
            for node, values in (yaml.safe_load(stream) or {}).items():
                loaded.setdefault(node, {}).update((values or {}).get("ros__parameters", {}))
    command = ["ros2", "bag", "record", "-a", "-o", os.path.join(session, "bag"),
               "--max-bag-duration", str(BAG_SPLIT_S)]
    controller = loaded.get("locomotive_policy_controller", {})
    motor = loaded.get("trackstand_motor_io", {})
    with open(os.path.join(session, "session.json"), "w") as stream:
        json.dump({
            "started": stamp.isoformat(timespec="seconds"), "host": socket.gethostname(),
            "stack": "locomotive", "git": _git_state(parameters[-1]),
            "model_filename": controller.get("model_filename"), "model_sha256": controller.get("model_sha256"),
            "rear_control_mode": motor.get("rear_control_mode"),
            "drive_mit_kd_nm_s_per_rad": motor.get("drive_mit_kd_nm_s_per_rad"),
            "dry_run": motor.get("dry_run"), "record_command": command,
            "note": "dry_run is the YAML value; a CLI override on a separately started node is not seen here",
        }, stream, indent=2)
        stream.write("\n")
    print(f"[locomotive.launch] recording session to {session}")
    return [
        # Applies to the nodes below: their text logs land in the session folder.
        SetEnvironmentVariable("ROS_LOG_DIR", os.path.join(session, "ros_logs")),
        ExecuteProcess(cmd=command, name="locomotive_rosbag", output="screen"),
    ]


def generate_launch_description():
    config = os.path.join(get_package_share_directory("trackstand_deploy"), "config")
    # Hardware params first; the locomotive overlay only adds/overrides its own keys.
    parameters = [os.path.join(config, "params.yaml"), os.path.join(config, "locomotive.yaml")]

    def node(executable: str, name: str) -> Node:
        return Node(
            package="trackstand_deploy",
            executable=executable,
            name=name,
            output="screen",
            parameters=parameters,
        )

    return LaunchDescription(
        [
            DeclareLaunchArgument("record", default_value="true",
                                  description="Record a rosbag + node logs + config session folder"),
            DeclareLaunchArgument("log_dir", default_value="~/trackstand_logs",
                                  description="Parent folder for locomotive_<timestamp> sessions"),
            # Runs first so ROS_LOG_DIR is set before any node starts.
            OpaqueFunction(function=_session, args=[parameters]),
            node("kickstand_driver", "trackstand_kickstand_driver"),
            node("imu_publisher", "trackstand_imu_publisher"),
            node("motor_mit_velocity_io", "trackstand_motor_io"),
            node("rc_kickstand", "trackstand_rc_kickstand"),
            node("locomotive_controller", "locomotive_policy_controller"),
        ]
    )
