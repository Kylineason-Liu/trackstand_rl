from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription,DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

print("---------------------robot_type = stm32v2---------------------")

def generate_launch_description():
    imu_filter_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('imu_complementary_filter'), 'launch'),
            '/complementary_filter.launch.py'])
    )

    ekf_params_file = LaunchConfiguration('ekf_params_file')
    ekf_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('robot_localization'), 'launch'),
            '/ekf.launch.py']),
        launch_arguments={'ekf_params_file': ekf_params_file}.items()
    )

    description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('yahboomcar_description'), 'launch'),
            '/description_launch.py'])
    )   
    base_link_to_imu_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_link_to_base_imu',
        #arguments=['0.0', '0.016325', '0.080691', '0', '0', '0', 'base_link', 'mpu6050_frame']
        arguments=['0.0', '0.016325', '-0.02', '0', '0', '0', 'base_link', 'imu_frame']
    )
    
    base_link_to_laser_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_link_to_base_laser',
        arguments=['0.0', '0.0', '0.138', '0', '0', '0', 'base_link', 'laser_frame']
    )

    #rf2o_laser_odometry 
    rf2o_laser_odometry_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('rf2o_laser_odometry'), 'launch'),
            '/rf2o_laser_odometry.launch.py'])
    )
    odom_to_base_footprint_node = Node(
        package='yahboomcar_bringup',
        executable='base_node',
        name='odom_to_base_footprint',
    )
    cmd_vel_scaler_node = Node(
        package='yahboomcar_bringup',
        executable='cmdvel2bl',
        name='cmd_vel_scaler',
        parameters=[{'mode': LaunchConfiguration('mode')}]
    )

    return LaunchDescription([
        imu_filter_node,
        base_link_to_imu_tf_node,
        base_link_to_laser_tf_node,
        description_launch,
        DeclareLaunchArgument(
        'mode',
        default_value='nav',
        description='mode:= nav or appslam,default,slam'
        ),
        cmd_vel_scaler_node,
        #rf2o_laser_odometry_launch,
        DeclareLaunchArgument(
            'ekf_params_file',
            default_value=os.path.join(get_package_share_directory("robot_localization"), 'params', 'ekf3.yaml'),
        ),
        ekf_node,
    ])