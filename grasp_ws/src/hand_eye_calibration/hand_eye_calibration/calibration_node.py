"""使用固定 ChArUco 标定板的 ROS 2 眼在手上手眼标定节点."""

from pathlib import Path
from collections import deque
import threading

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger
import yaml

from .calibration import (
    CalibrationSample,
    CharucoPoseDetector,
    calibrate_eye_in_hand,
    make_transform,
    quaternion_to_rotation,
    rotation_to_quaternion,
)

# 单位缩放因子，用于把 PoseStamped 中的平移量转换为米。
POSE_UNITS = {
    'm': 1.0,
    'cm': 0.01,
    'mm': 0.001,
}


class HandEyeCalibrationNode(Node):
    """同步采集机械臂和标定板位姿，并求解末端到相机的固定外参."""

    def __init__(self) -> None:
        super().__init__('hand_eye_calibration_node')
        self.init_parameters()
        # 图像回调与服务回调可能并发访问检测结果和样本，使用锁保护共享数据。
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.camera_matrix = None
        self.distortion = None
        self.latest_detection = None
        # 缓存厂家发布的 T_base_gripper，供采样时按图像时间戳匹配。
        self.robot_poses = deque(maxlen=20)
        self.samples: list[CalibrationSample] = []
        self.result = None

        # 创建与实体 ChArUco 板规格完全一致的检测器。
        self.detector = CharucoPoseDetector(
            squares_x=self.get_parameter('squares_x').value,
            squares_y=self.get_parameter('squares_y').value,
            square_length=self.get_parameter('square_length').value,
            marker_length=self.get_parameter('marker_length').value,
            dictionary_name=self.get_parameter('dictionary').value,
            minimum_corners=self.get_parameter('minimum_corners').value,
        )
        # CameraInfo 提供 K 内参矩阵和镜头畸变系数。
        self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info_topic').value,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )
        # 彩色图像用于检测 ChArUco Marker 和角点。
        self.create_subscription(
            Image,
            self.get_parameter('image_topic').value,
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PoseStamped,
            self.get_parameter('robot_pose_topic').value,
            self.robot_pose_callback,
            qos_profile_sensor_data,
        )
        self.debug_publisher = self.create_publisher(
            Image, '/charuco_debug_image', 10
        )
        # 标定由服务手动控制，确保机械臂停止稳定后再采样。
        self.create_service(Trigger, '/capture_sample', self.capture_sample)
        self.create_service(Trigger, '/calibrate', self.calibrate)
        self.create_service(Trigger, '/reset', self.reset)
        self.create_service(Trigger, '/save', self.save)
        self.get_logger().info(
            'Eye-in-hand calibration ready. Keep the ChArUco board fixed, move '
            'the robot, then call /capture_sample at each stable pose.'
        )

    def init_parameters(self) -> None:
        """声明相机、机械臂位姿、标定板、采样质量和输出相关参数."""
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('robot_pose_topic', '/rokae_driver7/tool_pose')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('gripper_frame', 'tool0')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('pose_units', 'm')
        self.declare_parameter('squares_x', 11)
        self.declare_parameter('squares_y', 8)
        self.declare_parameter('square_length', 0.020)
        self.declare_parameter('marker_length', 0.015)
        self.declare_parameter('dictionary', 'DICT_5X5_100')
        self.declare_parameter('minimum_corners', 20)
        self.declare_parameter('minimum_samples', 15)
        self.declare_parameter('minimum_translation_delta', 0.01)
        self.declare_parameter('minimum_rotation_delta_deg', 5.0)
        self.declare_parameter('max_detection_age', 3.0)
        self.declare_parameter('max_pose_time_difference', 0.3)
        self.declare_parameter('method', 'PARK')
        self.declare_parameter('output_file', '/home/daron/Projects/gacc_grasp/grasp_ws/src/hand_eye_calibration/outcome/' \
                                'hand_eye_calibration.yaml')

    def camera_info_callback(self, message: CameraInfo) -> None:
        """缓存相机内参；CameraInfo 通常只需更新一次."""
        camera_matrix = np.asarray(message.k, dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(message.d, dtype=np.float64)
        with self.lock:
            self.camera_matrix = camera_matrix
            self.distortion = distortion

    def robot_pose_callback(self, message: PoseStamped) -> None:
        """缓存厂家发布的 T_base_gripper 位姿消息."""
        expected_frame = self.get_parameter('base_frame').value
        if message.header.frame_id and message.header.frame_id != expected_frame:
            self.get_logger().warning(
                f'Ignoring robot pose in frame {message.header.frame_id}; '
                f'expected {expected_frame}',
                throttle_duration_sec=5.0,
            )
            return
        with self.lock:
            self.robot_poses.append(message)

    def image_callback(self, message: Image) -> None:
        """检测最新图像中的标定板，并发布带标记的调试图像."""
        with self.lock:
            if self.camera_matrix is None:
                return
            camera_matrix = self.camera_matrix.copy()
            distortion = self.distortion.copy()
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
            camera_to_target, annotated, corner_count = self.detector.detect(
                image, camera_matrix, distortion
            )
            debug_message = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            debug_message.header = message.header
            self.debug_publisher.publish(debug_message)
            # 检测结果连同图像时间戳保存，采样时匹配机械臂位姿话题。
            with self.lock:
                self.latest_detection = (
                    camera_to_target,
                    message.header.stamp,
                    message.header.frame_id,
                    corner_count
                )
        except Exception as error:
            self.get_logger().warning(f'ChArUco detection failed: {error}')

    def capture_sample(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """保存一组同步的 T_base_gripper 和 T_camera_target."""
        del request
        with self.lock:
            detection = self.latest_detection
        if detection is None or detection[0] is None:
            response.success = False
            response.message = 'No valid ChArUco pose is available'
            return response

        camera_to_target, stamp_message, detected_frame, corner_count = detection
        # 拒绝陈旧检测，避免使用机械臂已经移动前的标定板位姿。
        stamp = Time.from_msg(stamp_message)
        age = (self.get_clock().now() - stamp).nanoseconds / 1e9
        if age > self.get_parameter('max_detection_age').value:
            response.success = False
            response.message = f'Latest board detection is too old ({age:.2f} s)'
            return response

        expected_camera_frame = self.get_parameter('camera_frame').value
        if detected_frame and detected_frame != expected_camera_frame:
            self.get_logger().warning(
                f'Image frame is {detected_frame}, configured camera frame is '
                f'{expected_camera_frame}'
            )
        with self.lock:
            robot_poses = list(self.robot_poses)
        if not robot_poses:
            response.success = False
            response.message = (
                'No robot pose received on '
                f"{self.get_parameter('robot_pose_topic').value}"
            )
            return response

        # 使用与标定板图像时间最接近的 T_base_gripper，避免机械臂运动时错配。
        robot_pose = min(
            robot_poses,
            key=lambda pose: abs(
                (Time.from_msg(pose.header.stamp) - stamp).nanoseconds
            ),
        )
        time_difference = abs(
            (Time.from_msg(robot_pose.header.stamp) - stamp).nanoseconds
        ) / 1e9
        maximum_difference = self.get_parameter(
            'max_pose_time_difference'
        ).value
        if time_difference > maximum_difference:
            response.success = False
            response.message = (
                'Closest robot pose differs from image by '
                f'{time_difference:.3f} s (maximum {maximum_difference:.3f} s)'
            )
            return response

        try:
            base_to_gripper = self.pose_message_to_matrix(self.get_parameter('pose_units').value, robot_pose)
        except ValueError as error:
            response.success = False
            response.message = f'Invalid robot pose: {error}'
            return response
        with self.lock:
            previous_sample = self.samples[-1] if self.samples else None
        if previous_sample is not None:
            # 连续姿态过于接近会使 AX=XB 求解退化，因此拒绝重复采样。
            translation_delta = np.linalg.norm(
                base_to_gripper[:3, 3] - previous_sample.base_to_gripper[:3, 3]
            )
            rotation_delta = (
                previous_sample.base_to_gripper[:3, :3].T
                @ base_to_gripper[:3, :3]
            )
            rotation_angle = np.degrees(
                np.arccos(
                    np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0)
                )
            )
            if (
                translation_delta
                < self.get_parameter('minimum_translation_delta').value
                and rotation_angle
                < self.get_parameter('minimum_rotation_delta_deg').value
            ):
                response.success = False
                response.message = (
                    'Pose is too similar to the previous sample: '
                    f'{translation_delta * 1000:.1f} mm, {rotation_angle:.2f} deg'
                )
                return response
        sample = CalibrationSample(
            base_to_gripper=base_to_gripper,
            camera_to_target=camera_to_target.copy(),
        )
        with self.lock:
            self.samples.append(sample)
            count = len(self.samples)
        response.success = True
        response.message = f'Captured sample {count} using {corner_count} corners'
        self.get_logger().info(response.message)
        return response

    def calibrate(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """使用累计样本求解 T_gripper_camera，并保存到 YAML."""
        del request
        with self.lock:
            samples = list(self.samples)
        minimum_samples = self.get_parameter('minimum_samples').value
        if len(samples) < minimum_samples:
            response.success = False
            response.message = (
                f'Need at least {minimum_samples} samples; currently {len(samples)}'
            )
            return response
        try:
            result = calibrate_eye_in_hand(
                samples, method=self.get_parameter('method').value
            )
        except Exception as error:
            response.success = False
            response.message = f'Calibration failed: {error}'
            return response
        with self.lock:
            self.result = result
        self.write_result(result)
        response.success = True
        response.message = (
            f'{result.method} solved {result.sample_count} samples; '
            f'translation RMS={result.translation_rms_m * 1000:.2f} mm, '
            f'rotation RMS={result.rotation_rms_deg:.3f} deg'
        )
        self.get_logger().info(response.message)
        return response

    def reset(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """清空全部样本和当前标定结果."""
        del request
        with self.lock:
            count = len(self.samples)
            self.samples.clear()
            self.result = None
        response.success = True
        response.message = f'Cleared {count} samples'
        self.get_logger().info(response.message)
        return response

    def save(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """把最近一次成功的标定结果重新写入 YAML 文件."""
        del request
        with self.lock:
            result = self.result
        if result is None:
            response.success = False
            response.message = 'No calibration result to save'
            return response
        try:
            output_path = self.write_result(result)
        except Exception as error:
            response.success = False
            response.message = f'Cannot save result: {error}'
            return response
        response.success = True
        response.message = f'Saved calibration to {output_path}'
        return response

    @staticmethod
    def pose_message_to_matrix(unit: str, message: PoseStamped) -> np.ndarray:
        """把厂家发布的 PoseStamped 转换为 T_base_gripper 齐次矩阵."""
        if unit not in POSE_UNITS:
            raise ValueError(f'pose_units must be one of {list(POSE_UNITS)}')
        translation = message.pose.position
        scale = POSE_UNITS[unit]
        translation_norm = np.array([translation.x, translation.y, translation.z], dtype=np.float64)* scale
        quaternion = message.pose.orientation
        quaternion_norm = np.linalg.norm(
            [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        )
        if quaternion_norm < 1e-8:
            raise ValueError('orientation quaternion has zero length')
        return make_transform(
            quaternion_to_rotation(
                quaternion.x, quaternion.y, quaternion.z, quaternion.w
            ),
            translation_norm,
        )

    def write_result(self, result) -> Path:
        """保存 frame、误差、平移、四元数和齐次矩阵到 YAML."""
        output_path = Path(
            self.get_parameter('output_file').value
        ).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        transform = result.gripper_to_camera
        quaternion = rotation_to_quaternion(transform[:3, :3])
        data = {
            'calibration_type': 'eye_in_hand',
            'parent_frame': self.get_parameter('gripper_frame').value,
            'child_frame': self.get_parameter('camera_frame').value,
            'method': result.method,
            'sample_count': result.sample_count,
            'translation_rms_m': result.translation_rms_m,
            'rotation_rms_deg': result.rotation_rms_deg,
            'translation': {
                'x': float(transform[0, 3]),
                'y': float(transform[1, 3]),
                'z': float(transform[2, 3]),
            },
            'quaternion': {
                'x': float(quaternion[0]),
                'y': float(quaternion[1]),
                'z': float(quaternion[2]),
                'w': float(quaternion[3]),
            },
            'transform_matrix': transform.tolist(),
        }
        with output_path.open('w', encoding='utf-8') as stream:
            yaml.safe_dump(data, stream, sort_keys=False)
        return output_path


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HandEyeCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
