from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
import os
from ament_index_python import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node


def generate_launch_description():
    realsense_launch_file = os.path.join(
        get_package_share_directory('realsense2_camera'),  # 替换为RealSense包的名称
        'launch',
        'rs_launch.py'
    )
    realsense_pointcloud_enable = LaunchConfiguration('pointcloud.enable', default='true')
    easy_handeye_launch_file = os.path.join(
        get_package_share_directory('easy_handeye2'),  # 替换为EasyHandEye2包的名称
        'launch',
        'calibrate.launch.py'
    )
    aruco_detect_launch_file = os.path.join(
        get_package_share_directory('ros2_aruco'),  # 替换为ArucoDetect包的名称
        'launch',
        'aruco_recognition.launch.py'
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('calibration_type', default_value='eye_in_hand', description='Type of calibration'),
            DeclareLaunchArgument('name', default_value='my_eob_calib', description='Calibration task name'),
            DeclareLaunchArgument('robot_base_frame', default_value='base_link', description='Robot base frame ID'),
            DeclareLaunchArgument('robot_effector_frame',default_value='grasp_link',description='Robot end-effector frame ID'),
            DeclareLaunchArgument('tracking_base_frame', default_value='camera_link',
                                  description='Tracking system base frame ID'),
            DeclareLaunchArgument('tracking_marker_frame', default_value='marker_571',
                                  description='Tracking system marker frame ID'),
            # IncludeLaunchDescription(
            #     PythonLaunchDescriptionSource(realsense_launch_file),
            #     launch_arguments={'pointcloud.enable': realsense_pointcloud_enable}.items()
            # ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(aruco_detect_launch_file),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(easy_handeye_launch_file),
                launch_arguments={
                    'calibration_type': LaunchConfiguration('calibration_type'),
                    'name': LaunchConfiguration('name'),
                    'robot_base_frame': LaunchConfiguration('robot_base_frame'),
                    'robot_effector_frame': LaunchConfiguration('robot_effector_frame'),
                    'tracking_base_frame': LaunchConfiguration('tracking_base_frame'),
                    'tracking_marker_frame': LaunchConfiguration('tracking_marker_frame')
                }.items()
            )
        ]
    )