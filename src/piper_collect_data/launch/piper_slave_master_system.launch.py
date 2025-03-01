from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
import os
from ament_index_python import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

def generate_launch_description():
    # piper机器人ROS接口初始化
    # Get the path to the piper_description package
    piper_description_path = os.path.join(
        get_package_share_directory('piper_description'),
        'launch',
        'display_xacro.launch.py'
    )
    master_can_port_arg = DeclareLaunchArgument(
        'master_can_port',
        default_value='can_master',
        description='CAN port for the master arm'
    )
    slave_can_port_arg = DeclareLaunchArgument(
        'slave_can_port',
        default_value='can_slave',
        description='CAN port for the slave arm'
    )
    auto_enable_arg = DeclareLaunchArgument(
        'auto_enable',
        default_value='true',
        description='Enable robot arm automatically'
    )

    # Include display_xacro.launch.py
    display_xacro_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(piper_description_path)
    )

    rviz_ctrl_flag_arg = DeclareLaunchArgument(
        'rviz_ctrl_flag',
        default_value='true',
        description='Start rviz flag.'
    )
    gripper_exist_arg = DeclareLaunchArgument(
        'gripper_exist',
        default_value='true',
        description='Gripper existence flag'
    )

    piper_ctrl_node = Node(
        package='piper',
        executable='piper_slave_master_ctrl',
        name='piper_ctrl_slave_master_node',
        output='screen',
        parameters=[
            {'master_can_port': LaunchConfiguration('master_can_port')},
            {'slave_can_port': LaunchConfiguration('slave_can_port')},
            {'auto_enable': LaunchConfiguration('auto_enable')},
            {'rviz_ctrl_flag': LaunchConfiguration('rviz_ctrl_flag')},
            {'gripper_exist': LaunchConfiguration('gripper_exist')}
        ],
        remappings=[('slave_joint_states','joint_states')]
        # remappings=[
        #     ('joint_ctrl_single', '/joint_states')
        # ]
    )

    #启动realsense2_camera
    front_realsense_num = "148522072680"
    wrist_realsense_num = "241222073777"

    front_cam_node =Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        namespace='front_camera',
        name=f'front_camera',
        output='screen',
        parameters=[{'serial_no': front_realsense_num},
                     {"enable_color": True},
                    {"enable_depth": True},
                    {"enable_infra1": False},
                    {"enable_infra2": False},
                    {"enable_pointcloud": False},
                    {"rgb_camera_fps": 30}
                    ]
    )
    wrist_cam_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        namespace='wrist_camera',
        name=f'wrist_camera',
        output='screen',
        parameters=[{'serial_no': wrist_realsense_num},
                    {"enable_color": True},
                    {"enable_depth": True},
                    {"enable_infra1": False},
                    {"enable_infra2": False},
                    {"enable_pointcloud": False},
                    {"rgb_camera_fps": 30}
                    ]
    )

    ##遥操作控制节点
    slave_arm_control_node =Node(
        package='piper_collect_data',
        executable ='slave_arm_control_node',
        output='screen'
    )
    #初始化摄像头与机械臂的相对坐标信息
    # eyehand_publish_launch_file = os.path.join(get_package_share_directory('easy_handeye2'),
    #                                            'launch','publish.launch.py')
    return LaunchDescription([
        wrist_cam_node,
        front_cam_node,
        master_can_port_arg,
        slave_can_port_arg,
        auto_enable_arg,
        display_xacro_launch,
        rviz_ctrl_flag_arg,
        gripper_exist_arg,
        piper_ctrl_node,
        slave_arm_control_node
    ])