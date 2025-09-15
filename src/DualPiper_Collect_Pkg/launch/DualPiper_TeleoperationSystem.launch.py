from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
import os
from ament_index_python import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

def generate_launch_description():
    #����realsense2_camera
    top_realsense_num = "327122074756"
    left_wrist_realsense_num = "148522072680"
    # top_realsense_num = "327122074756"
    top_cam_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        namespace='top_camera',
        name=f'top_camera',
        output='screen',
        parameters=
        [
            {"tf_publish_rate": 1},
            {'serial_no': top_realsense_num},
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
    left_wrist_cam_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        namespace='leftwrist_camera',
        name=f'wrist_camera',
        output='screen',
        parameters=[{'serial_no': left_wrist_realsense_num},
                    {"enable_depth": True},
                    {"enable_accel": False},
                    {"enable_gyro": False},
                    {"enable_infra1": False},
                    {"enable_infra2": False},
                    # {"enable_sync": True},
                    {"rgb_camera.auto_exposure_priority": False},
                    {"rgb_camera.color_profile": "640x480x30"},
                    {"depth_module.depth_profile": "640x480x30"},
                    {"depth_module.enable_auto_exposure": False},
                    {"align_depth.enable": True}
                    ]
    )
    # top_cam_node = Node(
    #     package='realsense2_camera',
    #     executable='realsense2_camera_node',
    #     namespace='top_camera',
    #     name=f'top_camera',
    #     output='screen',
    #     parameters=[
    #                 {"tf_publish_rate":1},
    #                 {'serial_no': top_realsense_num},
    #                 {"enable_depth": True},
    #                 {"enable_accel": False},
    #                 {"enable_gyro": False},
    #                 {"enable_infra1": False},
    #                 {"enable_infra2": False},
    #                 {"rgb_camera.auto_exposure_priority":False},
    #                 {"rgb_camera.color_profile": "640x480x30"},
    #                 {"depth_module.depth_profile":"640x480x30"},
    #                 {"depth_module.enable_auto_exposure":False},
    #                 {"align_depth.enable":True}
    #                 ]
    # )
    piper_description_path = os.path.join(
        get_package_share_directory('piper_description'),
        'launch',
        'display_xacro.launch.py'
    )
    piper_can_left_port_arg = DeclareLaunchArgument(
        'piper_can_left_port',
        default_value='can_left',
        description='CAN port for the left arm'
    )
    piper_can_right_port_arg = DeclareLaunchArgument(
        'piper_can_right_port',
        default_value='can_right',
        description='CAN port for the right arm'
    )
    auto_enable_arg = DeclareLaunchArgument(
        'auto_enable',
        default_value='true',
        description='Enable robot arm automatically'
    )
    display_xacro_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(piper_description_path)
    )

    dual_piper_state_get_node = Node(
        package='DualPiper_Collect_Pkg',
        executable='DualPiper_StateGet_Node',
        name='DualPiper_StateGet_Node',
        output='screen',
        parameters=[
            {'left_can_port': LaunchConfiguration('piper_can_left_port')},
            {'right_can_port': LaunchConfiguration('piper_can_right_port')},
            {'auto_enable': LaunchConfiguration('auto_enable')},
        ],
        # remappings=[('right/master_joint_states','joint_states')]    # ������ӳ��
    )

    return LaunchDescription([
        piper_can_left_port_arg,
        piper_can_right_port_arg,
        auto_enable_arg,
        display_xacro_launch,
        dual_piper_state_get_node,
        # front_cam_node,
        # wrist_cam_node,
        top_cam_node,
        left_wrist_cam_node
    ])