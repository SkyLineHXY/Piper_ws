from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
import os
from ament_index_python import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node


def generate_launch_description():
    #启动realsense2_camera
    front_realsense_num = "148522072680"
    wrist_realsense_num = "241222073777"
    top_realsense_num = "327122074756"
    front_cam_node =Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        namespace='front_camera',
        name=f'front_camera',
        output='screen',
        parameters=
        [
            {"tf_publish_rate": 1},
            {'serial_no': front_realsense_num},
            {"enable_depth": True},
            {"enable_accel": False},
            {"enable_gyro": False},
            {"enable_infra1": False},
            {"enable_infra2": False},
            {"rgb_camera.auto_exposure_priority": False},
            {"rgb_camera.color_profile": "640x480x30"},
            {"depth_module.depth_profile": "640x480x30"},
            {"depth_module.enable_auto_exposure": False},
            {"align_depth.enable": True}
        ]
    )
    wrist_cam_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        namespace='wrist_camera',
        name=f'wrist_camera',
        output='screen',
        parameters=[{'serial_no': wrist_realsense_num},
                    {"enable_depth": False},
                    {"enable_accel": False},
                    {"enable_gyro": False},
                    {"enable_infra1": False},
                    {"enable_infra2": False},
                    # {"enable_sync": True},
                    {"rgb_camera.auto_exposure_priority": False},
                    {"rgb_camera.color_profile": "640x480x30"},
                    # {"depth_module.depth_profile": "640x480x60"},
                    # {"depth_module.enable_auto_exposure": False},
                    # {"align_depth.enable": True}
                    ]
    )
    top_cam_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        namespace='top_camera',
        name=f'top_camera',
        output='screen',
        parameters=[
                    {"tf_publish_rate":1},
                    {'serial_no': top_realsense_num},
                    {"enable_depth": True},
                    {"enable_accel": False},
                    {"enable_gyro": False},
                    {"enable_infra1": False},
                    {"enable_infra2": False},
                    {"rgb_camera.auto_exposure_priority":False},
                    {"rgb_camera.color_profile": "640x480x30"},
                    {"depth_module.depth_profile":"640x480x30"},
                    {"depth_module.enable_auto_exposure":False},
                    {"align_depth.enable":True}
                    ]
    )
    piper_description_path = os.path.join(
        get_package_share_directory('piper_description'),
        'launch',
        'display_xacro.launch.py'
    )
    piper_can_port_arg = DeclareLaunchArgument(
        'piper_can_port',
        default_value='can0',
        description='CAN port for the master arm'
    )
    auto_enable_arg = DeclareLaunchArgument(
        'auto_enable',
        default_value='true',
        description='Enable robot arm automatically'
    )
    display_xacro_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(piper_description_path)
    )

    piper_state_get_node = Node(
        package='piper_collection_pkg',
        executable='piper_state_get_node',
        name='piper_state_get_node',
        output='screen',
        parameters=[
            {'piper_can_port': LaunchConfiguration('piper_can_port')},
            {'auto_enable': LaunchConfiguration('auto_enable')},
        ],
        remappings=[('slave_joint_states','joint_states')]
    )
    return LaunchDescription([
        piper_can_port_arg,
        auto_enable_arg,
        display_xacro_launch,
        piper_state_get_node,
        front_cam_node,
        wrist_cam_node,
        top_cam_node
    ])