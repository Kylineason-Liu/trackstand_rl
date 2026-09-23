"""Launch file for bike_balance.

Usage
-----
  ros2 launch bike_balance balance_launch.py mode:=sign_check
  ros2 launch bike_balance balance_launch.py mode:=inner_loop
  ros2 launch bike_balance balance_launch.py mode:=full method:=lqr target_speed:=2.0

See DEPLOYMENT.md for the step-by-step deployment checklist.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    mode_arg = DeclareLaunchArgument(
        "mode", default_value="full",
        description="Operating mode: sign_check | inner_loop | full",
    )
    method_arg = DeclareLaunchArgument(
        "method", default_value="pole",
        description="Control method: pole | lqr",
    )
    target_speed_arg = DeclareLaunchArgument(
        "target_speed", default_value="2.7",
        description="Target forward speed (m/s)",
    )
    config_arg = DeclareLaunchArgument(
        "config_file",
        default_value="",
        description="Path to params YAML file (optional)",
    )

    balance_node = Node(
        package="bike_balance",
        executable="balance_node",
        name="bike_balance",
        output="screen",
        parameters=[{
            "mode": LaunchConfiguration("mode"),
            "method": LaunchConfiguration("method"),
            "target_speed": LaunchConfiguration("target_speed"),
        }],
    )

    return LaunchDescription([
        mode_arg,
        method_arg,
        target_speed_arg,
        config_arg,
        balance_node,
    ])
