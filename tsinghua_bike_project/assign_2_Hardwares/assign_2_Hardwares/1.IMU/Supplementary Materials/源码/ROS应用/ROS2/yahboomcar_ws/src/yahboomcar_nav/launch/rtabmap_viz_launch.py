from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    use_sim_time = LaunchConfiguration('use_sim_time')
    qos = LaunchConfiguration('qos')

    parameters={
          'use_sim_time':use_sim_time,
          'subscribe_rgbd':True,
          'subscribe_scan':True,
          'use_action_for_goal':True,
          # RTAB-Map's parameters should be strings:
          'Reg/Strategy':'0',
          'RGBD/NeighborLinkRefining':'True',
          'Grid/RangeMin':'0.2', # ignore laser scan points on the robot itself
          'publish_tf_map': 'false',      # 禁用RTAB-Map发布的map->odom
          'publish_tf_odom': 'false',     # 禁用RTAB-Map发布的odom->base
          'frame_id': 'base_link',
          'map_frame_id': 'map',
          'odom_frame_id': 'odom',
          'Reg/Force3DoF': 'true',        # 强制2D平面运动
          'Optimizer/GravitySigma': '0'   # 禁用IMU约束
    }

    remappings=[
          ('rgb/image', '/ascamera_hp60c/camera_publisher/rgb0/image'),
          ('rgb/camera_info', '/ascamera_hp60c/camera_publisher/rgb0/camera_info'),
          ('depth/image', '/ascamera_hp60c/camera_publisher/depth0/image_raw'),
          ('odom', '/odom')
        ]

    return LaunchDescription([

        # Launch arguments
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use simulation (Gazebo) clock if true'),
        
        DeclareLaunchArgument(
            'qos', default_value='2',
            description='QoS used for input sensor topics'),
            
        Node(
            package='rtabmap_viz', executable='rtabmap_viz', output='screen',
            parameters=[parameters],
            remappings=remappings),
        
    ])
