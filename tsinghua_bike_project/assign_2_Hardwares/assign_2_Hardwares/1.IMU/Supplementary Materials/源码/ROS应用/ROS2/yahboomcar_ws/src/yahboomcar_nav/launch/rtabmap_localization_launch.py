from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    use_sim_time = LaunchConfiguration('use_sim_time')
    qos = LaunchConfiguration('qos')
    localization = LaunchConfiguration('localization')
    config_rviz = os.path.join(
        get_package_share_directory('rtabmap_launch'), 'launch', 'config', 'rgbd.rviz'
    )

    parameters={
          'use_sim_time':use_sim_time,
          'subscribe_rgbd':True,
          'subscribe_scan':True,
          'use_action_for_goal':True,
          # RTAB-Map's parameters should be strings:
          'Reg/Strategy':'0',
          'RGBD/NeighborLinkRefining':'True',
          'publish_tf_map': 'false',      # 禁用 map->odom
          'publish_tf_odom': 'false',
          'Grid/RangeMin':'0.2', # ignore laser scan points on the robot itself
          'frame_id': 'base_link',
          'map_frame_id': 'map',
          'odom_frame_id': 'odom',
          'Reg/Force3DoF': 'true',        # 强制2D平面运动
          'Optimizer/GravitySigma': '0',   # 禁用IMU约束
          'RGBD/OptimizeMaxError': '10.0',
          'RGBD/LoopClosureAngularPrecision': '0.5',
          'Vis/MinInliers': '40'
    }

    remappings=[
         ('rgb/image', '/ascamera_hp60c/camera_publisher/rgb0/image'),
          ('rgb/camera_info', '/ascamera_hp60c/camera_publisher/rgb0/camera_info'),
          ('depth/image', '/ascamera_hp60c/camera_publisher/depth0/image_raw'),
        ]

    return LaunchDescription([

        # Launch arguments
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use simulation (Gazebo) clock if true'),
        
        DeclareLaunchArgument(
            'qos', default_value='2',
            description='QoS used for input sensor topics'),
            
        DeclareLaunchArgument(
            'localization', default_value='true',
            description='Launch in localization mode.'),
        
        DeclareLaunchArgument('rviz_cfg', default_value=config_rviz, 
                              description='Configuration path of rviz2.'),
        
        DeclareLaunchArgument('rviz', default_value='true', description='Launch RVIZ (optional).'),


        # Nodes to launch
        Node(
            package='rtabmap_sync', executable='rgbd_sync', output='screen',
            parameters=[{'approx_sync':True, 'approx_sync_max_interval':0.01, 'use_sim_time':use_sim_time, 'qos':qos}],
            remappings=remappings),

        # SLAM Mode:
        Node(
            condition=UnlessCondition(localization),
            package='rtabmap_slam', executable='rtabmap', output='screen',
            parameters=[parameters],
            remappings=remappings,
            arguments=['-d']),
            
        # Localization mode:
        Node(
            condition=IfCondition(localization),
            package='rtabmap_slam', executable='rtabmap', 
            parameters=[parameters,
              {'Mem/IncrementalMemory':'False',
               'Mem/InitWMWithAllNodes':'True'}],
            remappings=remappings),

    ])
