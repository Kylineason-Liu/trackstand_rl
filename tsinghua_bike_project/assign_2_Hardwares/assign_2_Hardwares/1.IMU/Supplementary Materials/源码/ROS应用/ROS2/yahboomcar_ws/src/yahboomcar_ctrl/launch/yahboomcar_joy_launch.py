from launch import LaunchDescription
from launch_ros.actions import Node

import os
def generate_launch_description():
    node1 = Node(
        package='joy',
        executable='joy_node',
    )
    joy_node = Node(
        package='yahboomcar_ctrl',
        executable='yahboom_joy_A1',
    )
    bringup_node = Node(
        package='yahboomcar_bringup',
        executable='Ackman_driver_A1',
    )
    launch_description = LaunchDescription([node1,joy_node,bringup_node])
    return launch_description
