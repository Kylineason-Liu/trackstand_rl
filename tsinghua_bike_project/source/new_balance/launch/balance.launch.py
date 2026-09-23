#!/usr/bin/env python3
"""
一键启动 new_balance 全部节点（IMU / 转向 / 后轮 / 遥控 / 主控）。

用法：
    ros2 launch new_balance balance.launch.py

    # 跳过 RC 遥控器（只用固定速度模式）
    ros2 launch new_balance balance.launch.py enable_rc:=false

    # 纯调试：只启动传感器发布节点，不启动主控
    ros2 launch new_balance balance.launch.py enable_main:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _make_node(name, executable, extra_params=None):
    """Create a Node with the shared params.yaml and optional overrides."""
    params_path = os.path.join(
        get_package_share_directory("new_balance"), "config", "params.yaml")
    params = [params_path]
    if extra_params:
        params.append(extra_params)
    return Node(
        package="new_balance",
        executable=executable,
        name=name,
        output="screen",
        parameters=params,
    )


def _launch_setup(context, *args, **kwargs):
    enable_main = LaunchConfiguration("enable_main").perform(context) == "true"
    enable_rc = LaunchConfiguration("enable_rc").perform(context) == "true"
    enable_roll = LaunchConfiguration("enable_roll").perform(context) == "true"
    enable_steer = LaunchConfiguration("enable_steer").perform(context) == "true"
    enable_rear = LaunchConfiguration("enable_rear").perform(context) == "true"

    nodes = []

    if enable_roll:
        nodes.append(_make_node("roll_info_publisher", "roll_publisher"))

    if enable_steer:
        nodes.append(_make_node("steer_info_publisher", "steer_publisher"))

    if enable_rear:
        nodes.append(_make_node("rear_info_publisher", "rear_publisher"))

    if enable_rc:
        nodes.append(_make_node("rc_publisher", "rc_publisher"))

    if enable_main:
        method = LaunchConfiguration("method").perform(context)
        vctrl = LaunchConfiguration("vctrl").perform(context)
        nodes.append(_make_node("new_balance_node", "balance_node",
                                extra_params={"method": method, "vctrl": vctrl}))

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("enable_main",  default_value="true",
                              description="启动主控节点"),
        DeclareLaunchArgument("enable_rc",    default_value="true",
                              description="启动 RC 遥控器节点"),
        DeclareLaunchArgument("enable_roll",  default_value="true",
                              description="启动 IMU 姿态发布节点"),
        DeclareLaunchArgument("enable_steer", default_value="true",
                              description="启动转向电机反馈节点"),
        DeclareLaunchArgument("enable_rear",  default_value="true",
                              description="启动后轮电机反馈节点"),
        DeclareLaunchArgument("method",       default_value="lqr",
                              description="控制方法: pole | lqr"),
        DeclareLaunchArgument("vctrl",        default_value="rc",
                              description="速度控制模式: fixed | rc"),
        OpaqueFunction(function=_launch_setup),
    ])
