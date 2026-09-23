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
    CAMERA_TYPE = os.getenv('CAMERA_TYPE')

    if not os.environ.get("LASER_BRINGUP_PRINTED"):
        os.environ["LASER_BRINGUP_PRINTED"] = "1"
        print(f"\n-------- robot_type = {ROBOT_TYPE}, rplidar_type = {RPLIDAR_TYPE}, camera_type = {CAMERA_TYPE} --------\n")

    camera_type_arg = DeclareLaunchArgument(
        name='camera_type', 
        default_value=os.getenv('CAMERA_TYPE', 'nuwa'), 
        choices=['nuwa', 'usb'],
        description='The type of CAMERA'
    )

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

    tf_base_link_to_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '1.570796', '3.141592', '1.570796', 'camera_link', 'ascamera_hp60c_camera_link_0'],
        condition=LaunchConfigurationEquals('camera_type', 'nuwa')
    )
    tf_base_link_to_cameracolor = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'ascamera_hp60c_camera_link_0', 'ascamera_hp60c_color_0'],
        condition=LaunchConfigurationEquals('camera_type', 'nuwa')
    )

    tf_base_link_to_laser = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0.0060585', '0', '0.14912', '0', '3.1415', '3.1415', 'base_link', 'laser']
    )

    return LaunchDescription([
        camera_type_arg,
        bringup_launch,
        lidar_launch,
        tf_base_link_to_laser,
        tf_base_link_to_camera,
        tf_base_link_to_cameracolor,
    ])