from launch import LaunchDescription
from launch_ros.actions import Node 
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config = os.path.join('/home/teama/dev_ws/src/yolov4', 'config', 'cam2image.yaml')

    return LaunchDescription([
        Node(
            package='image_tools',
            node_executable='cam2image',
            parameters=[config]
        ),
        Node(
            package='yolov5',
            node_executable='yolov5_node',
	    output='screen'
        ),
	Node(
            package='rqt_image_view',
            node_executable='rqt_image_view',
	    output='screen'
        )
    ])
