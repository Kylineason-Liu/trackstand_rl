"""Launch all sensor, RC, actuator, and policy nodes."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    parameters = os.path.join(
        get_package_share_directory("trackstand_deploy"), "config", "params.yaml"
    )

    def node(executable: str, name: str) -> Node:
        return Node(
            package="trackstand_deploy",
            executable=executable,
            name=name,
            output="screen",
            parameters=[parameters],
        )

    return LaunchDescription(
        [
            node("kickstand_driver", "trackstand_kickstand_driver"),
            node("imu_publisher", "trackstand_imu_publisher"),
            node("motor_mit_io", "trackstand_motor_io"),
            node("rc_kickstand", "trackstand_rc_kickstand"),
            node("policy_controller", "trackstand_policy_controller"),
        ]
    )

