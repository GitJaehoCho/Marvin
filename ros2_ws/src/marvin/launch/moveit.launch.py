import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder(
            robot_name="marvin", package_name="marvin_moveit"
        )
        .robot_description(file_path="config/marvin.urdf.xacro")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .moveit_cpp(
            file_path=get_package_share_directory("marvin")
            + "/config/moveit.yaml"
        )
        .to_moveit_configs()
    )

    moveit_node = Node(
        name="moveit_node",
        package="marvin",
        executable="moveit",
        output="both",
        parameters=[moveit_config.to_dict()],
    )

    return LaunchDescription(
        [
            moveit_node
        ]
    )