from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    args = [
        # Frames
        DeclareLaunchArgument("map_frame",   default_value="map"),
        DeclareLaunchArgument("base_frame",  default_value="camera"),

        # Entrada
        DeclareLaunchArgument("use_scan",    default_value="false"),
        DeclareLaunchArgument("input_topic", default_value="/cloud_in"),
        DeclareLaunchArgument("scan_topic",  default_value="/scan"),

        # GICP
        DeclareLaunchArgument("num_threads",             default_value="16"),
        DeclareLaunchArgument("max_iterations",          default_value="64"),
        DeclareLaunchArgument("max_correspondence_dist", default_value="2.0"),

        # Downsampling
        DeclareLaunchArgument("input_leaf_size", default_value="2.0"),
        DeclareLaunchArgument("map_leaf_size",   default_value="10.0"),

        # Keyframes
        DeclareLaunchArgument("keyframe_delta_trans", default_value="0.3"),
        DeclareLaunchArgument("keyframe_delta_angle", default_value="0.15"),
        DeclareLaunchArgument("local_map_window",     default_value="5"),

        # Salida
        DeclareLaunchArgument("map_topic",   default_value="/gicp_map"),
        DeclareLaunchArgument("pose_topic",  default_value="/gicp_pose"),
        DeclareLaunchArgument("path_topic",  default_value="/gicp_path"),
        DeclareLaunchArgument("publish_tf",  default_value="true"),
        DeclareLaunchArgument("map_period",  default_value="2.0"),

        # Guardado
        DeclareLaunchArgument("save_path",   default_value="/tmp/gicp_map.pcd"),
    ]

    node = Node(
        package="gicp_mapper",
        executable="gicp_mapper_node",
        name="gicp_mapper",
        output="screen",
        parameters=[{
            "map_frame":              LaunchConfiguration("map_frame"),
            "base_frame":             LaunchConfiguration("base_frame"),
            "use_scan":               LaunchConfiguration("use_scan"),
            "input_topic":            LaunchConfiguration("input_topic"),
            "scan_topic":             LaunchConfiguration("scan_topic"),
            "num_threads":            LaunchConfiguration("num_threads"),
            "max_iterations":         LaunchConfiguration("max_iterations"),
            "max_correspondence_dist":LaunchConfiguration("max_correspondence_dist"),
            "input_leaf_size":        LaunchConfiguration("input_leaf_size"),
            "map_leaf_size":          LaunchConfiguration("map_leaf_size"),
            "keyframe_delta_trans":   LaunchConfiguration("keyframe_delta_trans"),
            "keyframe_delta_angle":   LaunchConfiguration("keyframe_delta_angle"),
            "local_map_window":       LaunchConfiguration("local_map_window"),
            "map_topic":              LaunchConfiguration("map_topic"),
            "pose_topic":             LaunchConfiguration("pose_topic"),
            "path_topic":             LaunchConfiguration("path_topic"),
            "publish_tf":             LaunchConfiguration("publish_tf"),
            "map_period":             LaunchConfiguration("map_period"),
            "save_path":              LaunchConfiguration("save_path"),
        }],
    )

    return LaunchDescription(args + [node])