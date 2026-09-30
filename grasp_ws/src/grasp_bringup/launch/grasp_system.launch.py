from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    start_arm_control = LaunchConfiguration('start_arm_control')
    start_coordinator = LaunchConfiguration('start_coordinator')

    return LaunchDescription([
        DeclareLaunchArgument('start_arm_control', default_value='false'),
        DeclareLaunchArgument('start_coordinator', default_value='false'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('grasp_bringup'),
                    'launch',
                    'perception.launch.py',
                ])
            )
        ),
        Node(
            package='arm_control',
            executable='arm_control_node',
            name='arm_control_node',
            output='screen',
            condition=IfCondition(start_arm_control),
        ),
        Node(
            package='grasp_coordinator',
            executable='grasp_coordinator_node',
            name='grasp_coordinator_node',
            output='screen',
            condition=IfCondition(start_coordinator),
        ),
    ])
