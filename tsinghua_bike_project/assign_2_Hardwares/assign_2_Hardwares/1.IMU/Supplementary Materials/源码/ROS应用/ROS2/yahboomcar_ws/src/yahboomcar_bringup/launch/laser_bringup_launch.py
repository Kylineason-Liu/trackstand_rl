from launch import LaunchDescription
from launch_ros.actions import Node 
import os
from launch.actions import IncludeLaunchDescription
from launch.conditions import LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    ROBOT_TYPE = os.getenv('ROBOT_TYPE', 'A1')
    RPLIDAR_TYPE = os.getenv('RPLIDAR_TYPE', 'c1')
    rplidar_type = RPLIDAR_TYPE = os.getenv('RPLIDAR_TYPE')

    if not os.environ.get("LASER_BRINGUP_PRINTED"):
        os.environ["LASER_BRINGUP_PRINTED"] = "1"
        print("\n-------- robot_type = {}, rplidar_type = {} --------\n".format(ROBOT_TYPE, RPLIDAR_TYPE))

    if ROBOT_TYPE == 'M1':
        bringup_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(get_package_share_directory('yahboomcar_bringup'), 'launch'),
                '/yahboomcar_bringup_M1_launch.py'
            ])
        )
    else:
        bringup_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(get_package_share_directory('yahboomcar_bringup'), 'launch'),
                '/yahboomcar_bringup_A1_launch.py'
            ])
        )

    if RPLIDAR_TYPE == 'c1':
        lidar_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(get_package_share_directory('sllidar_ros2'), 'launch'),
                '/sllidar_c1_launch.py'
            ])
        )
    else:
        lidar_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(get_package_share_directory('ydlidar_ros2_driver'), 'launch'),
                '/ydlidar_launch.py'
            ])
        )

    return LaunchDescription([
        bringup_launch,
        lidar_launch,
    ])