"""Launch the parallel v2 velocity-policy deployment with the existing bike hardware."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    parameters = os.path.join(
        get_package_share_directory("trackstand_deploy"), "config", "params_v2.yaml"
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
            node("motor_mit_io_v2", "trackstand_motor_io_v2"),
            node("rc_kickstand", "trackstand_rc_kickstand"),
            node("policy_controller_v2", "trackstand_policy_controller_v2"),
        ]
    )

