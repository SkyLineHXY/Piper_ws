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
    can_port_arg = DeclareLaunchArgument(
        'can_port',
        default_value='can0',
        description='CAN port for the robot arm'
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
        executable='piper_single_ctrl',
        name='piper_ctrl_single_node',
        output='screen',
        parameters=[
            {'can_port': LaunchConfiguration('can_port')},
            {'auto_enable': LaunchConfiguration('auto_enable')},
            {'rviz_ctrl_flag': LaunchConfiguration('rviz_ctrl_flag')},
            {'gripper_exist': LaunchConfiguration('gripper_exist')}
        ],
        remappings=[('joint_states_single','joint_states')]
        # remappings=[
        #     ('joint_ctrl_single', '/joint_states')
        # ]
    )

    # realsense摄像头初始化
    realsense_launch_file = os.path.join(
        get_package_share_directory('realsense2_camera'),  # 替换为RealSense包的名称
        'launch',
        'rs_launch.py'
    )
    realsense_pointcloud_enable = LaunchConfiguration('pointcloud.enable', default='true')
    realsense_color_width = LaunchConfiguration('color.width', default='640')  # 设置图像宽度
    realsense_color_height = LaunchConfiguration('color.height', default='480')  # 设置图像高度
    realsense_color_fps = LaunchConfiguration('color.fps', default='30')  # 设置图像帧率
    realsense_depth_width = LaunchConfiguration('depth.width',default='640')
    realsense_depth_height = LaunchConfiguration('depth.height',default='480')
    realsense_depth_fps = LaunchConfiguration('depth.fps', default='30')

    realsense_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(realsense_launch_file),
            launch_arguments={'pointcloud.enable': realsense_pointcloud_enable,
                              'color.width': realsense_color_width,
                              'color.height': realsense_color_height,
                              'color.fps': realsense_color_fps,
                              'depth.width': realsense_depth_width,
                              'depth.height': realsense_depth_height,
                              'depth.fps': realsense_depth_fps}.items())
    #初始化摄像头与机械臂的相对坐标信息
    eyehand_publish_launch_file = os.path.join(get_package_share_directory('easy_handeye2'),
                                               'launch','publish.launch.py')
    eyehand_publish_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(eyehand_publish_launch_file))
    piper_grasp_predict_Node =Node(
        package = 'piper_graspNet',
        executable = 'Grasp_pose_generator_Node',
        output = "screen"
    )
    grasp_link6_transformPub_Node = Node(
        package = 'piper_graspNet',
        executable='Grasplink_tf_pub_Node',
        output = "screen"
    )
    return LaunchDescription([
        realsense_launch,
        eyehand_publish_launch,
        piper_grasp_predict_Node,
        can_port_arg,
        auto_enable_arg,
        display_xacro_launch,
        rviz_ctrl_flag_arg,
        gripper_exist_arg,
        piper_ctrl_node,
        grasp_link6_transformPub_Node
    ])
