from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    gicp_mapper = Node(
        package='gicp_mapper',
        executable='voxelized_mapper',
        name='gicp_mapper',
        output='screen',
        parameters=[
            {
                # "use_vgicp": False,
                # "leaf_size_input": 0.5,
                # "leaf_size_map": 0.02,
                # "max_range": 60.0,
                # "min_range": 0.2,
                # "max_iterations": 64,
                # "transformation_epsilon": 1e-5,
                # "max_correspondence_dist": 100.0,
                # "num_threads": 14,
                # "vgicp_resolution": 1,
                # "map_max_frames": 200,
                # "keyframe_dist": 0.5,
                # "keyframe_angle": 0.087,  # ~5°
                # "force_accept": True,
                # "score_warn_threshold": 2.0,
                # "map_save_path": "/home/ubuntu/map",
                # "map_save_format": "pcd"
                "use_vgicp": True,

                "leaf_size_input": 0.10,
                "leaf_size_local_target": 0.25,
                "leaf_size_map": 0.25,

                "min_range": 0.1,
                "max_range": 100.0,

                "max_iterations": 128,
                "transformation_epsilon": 1e-4,
                "rotation_epsilon": 0.05,
                "max_correspondence_dist": 0.2,

                "num_threads": 14,
                "vgicp_resolution": 0.5,

                "map_max_frames": -1,
                "local_window_size": 6,

                "keyframe_dist": 0.4,
                "keyframe_angle": 0.087,

                "score_warn_threshold": 0.3,
                "force_accept": True,

                "save_voxel_size": 0.05,
                "map_save_path": "/home/ubuntu/map",
                "map_save_format": "pcd"

            }
        ]
    )

    return LaunchDescription([
        gicp_mapper
    ])