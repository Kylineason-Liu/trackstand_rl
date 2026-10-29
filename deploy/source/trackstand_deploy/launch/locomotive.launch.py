"""Launch the locomotive (CycleRL riding policy) stack: live motors unless dry_run."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


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
            node("kickstand_driver", "trackstand_kickstand_driver"),
            node("imu_publisher", "trackstand_imu_publisher"),
            node("motor_mit_velocity_io", "trackstand_motor_io"),
            node("rc_kickstand", "trackstand_rc_kickstand"),
            node("locomotive_controller", "locomotive_policy_controller"),
        ]
    )
