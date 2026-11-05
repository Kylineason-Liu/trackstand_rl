"""Launch the parallel v2 velocity-policy deployment with the existing bike hardware."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    parameters = os.path.join(
        get_package_share_directory("trackstand_deploy"), "config", "params_v2.yaml"
    )

    def node(executable: str, name: str, **kwargs) -> Node:
        return Node(
            package="trackstand_deploy",
            executable=executable,
            name=name,
            output="screen",
            parameters=[parameters],
            **kwargs,
        )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_rc", default_value="true",
                description="Start the RC reader; false allows manual_kickstand in a separate terminal",
            ),
            node("kickstand_driver", "trackstand_kickstand_driver"),
            node("imu_publisher", "trackstand_imu_publisher"),
            node("motor_mit_io_v2", "trackstand_motor_io_v2"),
            node("rc_kickstand", "trackstand_rc_kickstand", condition=IfCondition(LaunchConfiguration("use_rc"))),
            node("policy_controller_v2", "trackstand_policy_controller_v2"),
        ]
    )

