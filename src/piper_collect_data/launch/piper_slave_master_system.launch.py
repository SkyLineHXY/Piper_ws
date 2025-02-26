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

    return LaunchDescription([
        # realsense_launch,
        # eyehand_publish_launch,
        # piper_grasp_predict_Node,
        master_can_port_arg,
        slave_can_port_arg,
        auto_enable_arg,
        display_xacro_launch,
        rviz_ctrl_flag_arg,
        gripper_exist_arg,
        piper_ctrl_node,
        # grasp_link6_transformPub_Node
    ])