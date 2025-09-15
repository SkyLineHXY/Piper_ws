from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
import os
from ament_index_python import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node


def generate_launch_description():
    
    DualPiper_System_Launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('DualPiper_Collect_Pkg'), 'launch', 'DualPiper_TeleoperationSystem.launch.py')
        )
    )
    DualArm_DataToHDF5_Node = Node(
        package='DualPiper_Collect_Pkg',
        executable='DualArm_DataToHDF5_Node',
        name='DualArm_DataToHDF5_Node',
        output='screen',
        )
    return LaunchDescription([
        DualPiper_System_Launch,
        DualArm_DataToHDF5_Node
    ])