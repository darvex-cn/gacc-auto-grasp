from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        Node(
            package='pose_detect',
            executable='pose_detect_node',
            name='pose_detect_node',
            output='screen',
        ),
    ])
