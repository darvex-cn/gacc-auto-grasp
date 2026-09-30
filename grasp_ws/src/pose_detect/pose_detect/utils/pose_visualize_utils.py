"""Create RViz messages for GraspNet predictions."""

import numpy as np
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

from .transforms import rotation_matrix_to_quaternion


def _point(x: float, y: float, z: float = 0.0) -> Point:
    """Create one geometry point."""
    point = Point()
    point.x = float(x)
    point.y = float(y)
    point.z = float(z)
    return point


def make_grasp_markers(
    best,
    stamp,
    camera_frame: str,
    line_width: float,
) -> MarkerArray:
    """Convert the best grasp into one RViz parallel-gripper marker."""
    marker_array = MarkerArray()
    clear_marker = Marker()
    clear_marker.action = Marker.DELETEALL
    marker_array.markers.append(clear_marker)

    if best is None:
        return marker_array

    finger_base_x = -0.024
    tail_end_x = -0.064
    half_width = best.width / 2.0

    marker = Marker()
    marker.header.stamp = stamp
    marker.header.frame_id = camera_frame
    marker.ns = 'best_grasp'
    marker.id = 0
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.scale.x = line_width * 1.5

    marker.pose.position.x = float(best.translation[0])
    marker.pose.position.y = float(best.translation[1])
    marker.pose.position.z = float(best.translation[2])
    qx, qy, qz, qw = rotation_matrix_to_quaternion(best.rotation_matrix)
    marker.pose.orientation.x = qx
    marker.pose.orientation.y = qy
    marker.pose.orientation.z = qz
    marker.pose.orientation.w = qw

    marker.points = [
        _point(finger_base_x, -half_width),
        _point(best.depth, -half_width),
        _point(finger_base_x, half_width),
        _point(best.depth, half_width),
        _point(finger_base_x, -half_width),
        _point(finger_base_x, half_width),
        _point(tail_end_x, 0.0),
        _point(finger_base_x, 0.0),
    ]
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker_array.markers.append(marker)
    return marker_array


def make_scene_cloud(
    points,
    stamp,
    camera_frame: str,
    max_points: int,
) -> PointCloud2:
    """Convert GraspNet scene points into an RViz point cloud."""
    if max_points > 0 and len(points) > max_points:
        indices = np.linspace(
            0, len(points) - 1, max_points, dtype=np.int64
        )
        points = points[indices]
    header = Header()
    header.stamp = stamp
    header.frame_id = camera_frame
    return point_cloud2.create_cloud_xyz32(header, points)
