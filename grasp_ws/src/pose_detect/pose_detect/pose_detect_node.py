"""ROS interfaces, callbacks, and executor for GraspNet pose detection."""

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from grasp_interfaces.srv import DetectGrasp
from graspnet_adapter import CameraIntrinsics, GraspPrediction, GraspCandidate
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from visualization_msgs.msg import MarkerArray

from .object_detect import ObjectDetector, SegmentationResult
from .pose_detect import PoseDetector
from .utils.calibration_io import load_hand_eye_transform
from .utils.image_utils import (
    bgr_to_image_message,
    camera_intrinsics,
    color_message_to_bgr,
    depth_message_to_numpy,
)
from .utils.pose_visualize_utils import make_grasp_markers, make_scene_cloud
from .utils.tools import load_config
from .utils.transforms import (
    base_down_direction_in_camera,
    make_transform,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    transform_base_to_grasp_simple,
)
from .constants import POSE_UNITS, R_GRASP_GRIPPER, POSE_DETECT_NODE_PARAMETERS


@dataclass(frozen=True)
class _VisualTopics:
    '''Publishers enabled by the pose and segmentation visualization options.'''
    marker_publisher: Publisher | None = None
    cloud_publisher: Publisher | None = None
    segment_image_publisher: Publisher | None = None

@dataclass(frozen=True)
class _InputFrame:
    '''Newest synchronized sensor messages claimed for segmentation.'''
    color_msg: Image
    depth_msg: Image
    camera_info_msg: CameraInfo
    t_base_to_gripper: np.ndarray | None
    t_base_to_camera: np.ndarray | None
    source_arrival_ns: int

@dataclass(frozen=True)
class _SegmentedFrame:
    '''One synchronized RGB-D frame and its selected YOLO target mask.'''
    color_rgb: np.ndarray
    depth: np.ndarray
    intrinsics: CameraIntrinsics
    target_mask: np.ndarray
    stamp: Time
    base_to_gripper: np.ndarray | None
    base_to_camera: np.ndarray | None
    source_arrival_ns: int


class PoseDetectNode(Node):
    '''Call the detector and expose its result through ROS 2 interfaces.'''

    def __init__(self) -> None:
        super().__init__('pose_detect_node')
        config_path = (
            Path(get_package_share_directory('pose_detect'))
            / 'config'
            / 'config.yaml'
        )
        config = load_config(
            config_path,
            'pose_detect_node',
            POSE_DETECT_NODE_PARAMETERS,
            path_params={
                'general.hand_eye_calibration_path',
                'graspnet.checkpoint_path',
                'object_detect.object_model_path',
            },
        )
        for name, value in config.items():
            self.declare_parameter(name, value)
        self._validate_parameters()
        self.get_logger().info(f'Loaded configuration: {config_path}')

        self._callback_group = ReentrantCallbackGroup()
        self._image_lock = Lock()
        self._segmentation_result_lock = Lock()
        self._segmentation_inference_lock = Lock()
        self._grasp_inference_lock = Lock()
        self._robot_pose_lock = Lock()
        self._latest_camera_info: CameraInfo | None = None
        self._latest_input_frame: _InputFrame | None = None
        self._latest_segmented_frame: _SegmentedFrame | None = None
        self._t_base_to_gripper = None
        self._t_base_to_camera = None
        # GraspNet 到夹爪的映射，目前只有旋转关系。
        self._t_graspnet_to_gripper = make_transform(R_GRASP_GRIPPER, np.zeros(3))
        try:
            self._t_gripper_to_camera = load_hand_eye_transform(
                self.get_parameter('general.hand_eye_calibration_path').value
            )
            self.get_logger().info('Loaded t_gripper_to_camera from calibration YAML')
        except Exception as error:
            self._t_gripper_to_camera = None
            self.get_logger().error(f'Cannot use hand-eye calibration: {error}')

        self._pose_detector, self._object_detector = self._load_detector()
        self._create_interfaces()
        segmentation_rate = self.get_parameter(
            'object_detect.inference_rate'
        ).value
        self.get_logger().info(
            f'Node ready: continuous segmentation up to {segmentation_rate} '
            'Hz; waiting for /detect_grasp'
        )

    # 加载算法模型检测器
    def _load_detector(self) -> tuple[PoseDetector | None, ObjectDetector | None]:
        """Load GraspNet and Object detector once so realtime service calls can reuse it."""
        try:
            pose_detector = PoseDetector(
                self.get_parameter('graspnet.checkpoint_path').value,
                num_point=self.get_parameter('graspnet.num_point').value,
                num_view=self.get_parameter('graspnet.num_view').value,
                collision_thresh=self.get_parameter('graspnet.collision_thresh').value,
                voxel_size=self.get_parameter('graspnet.voxel_size').value,
                max_gripper_width=self.get_parameter(
                    'graspnet.max_gripper_width'
                ).value,
                device=self.get_parameter('graspnet.device').value,
                logger=self.get_logger()
            )
            object_detector = ObjectDetector(
                self.get_parameter('object_detect.object_model_path').value,
                target_class=self.get_parameter('object_detect.target_class').value,
                selection_policy=self.get_parameter('object_detect.selection_policy').value,
                visual_enable=self.get_parameter(
                    'object_detect.visual_enable'
                ).value,
                confidence_threshold=self.get_parameter('object_detect.confidence_threshold').value,
                iou_threshold=self.get_parameter('object_detect.iou_threshold').value,
                mask_threshold=self.get_parameter('object_detect.mask_threshold').value,
                image_size=self.get_parameter('object_detect.image_size').value,
                device=self.get_parameter('object_detect.device').value,
                quantize=self.get_parameter('object_detect.quantize').value,
                retina_masks=self.get_parameter('object_detect.retina_masks').value,
                logger=self.get_logger()
            )
            return pose_detector, object_detector
        except Exception as error:
            self.get_logger().error(f'Cannot load GraspNet or object detection model: {error}')
            return None, None

    def _create_interfaces(self) -> None:
        """Create all publishers, subscriptions, services, and timers."""
        transient_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        grasp_visual_enabled = self.get_parameter('rviz.grasp_visual_enable').value
        segmentation_visual_enabled = self.get_parameter('object_detect.visual_enable').value
        if grasp_visual_enabled or segmentation_visual_enabled:
            self._visual_topic = _VisualTopics(
                marker_publisher=(
                    self.create_publisher(MarkerArray, '/pose_visual/grasp_candidates', transient_qos) 
                    if grasp_visual_enabled else None),
                cloud_publisher=(
                    self.create_publisher(PointCloud2, '/pose_visual/graspnet_scene_cloud', transient_qos)
                    if grasp_visual_enabled else None),
                segment_image_publisher=(
                    self.create_publisher(Image,'/pose_visual/segmentation_image',transient_qos)
                    if segmentation_visual_enabled else None)
            )
        else:
            self._visual_topic = None

        self.create_subscription(
            PoseStamped,
            self.get_parameter('general.robot_pose_topic').value,
            self._robot_pose_callback,
            1,
            callback_group=self._callback_group,
        )
        self._color_subscriber = Subscriber(
            self,
            Image,
            self.get_parameter('camera.rgb_topic').value,
            qos_profile=qos_profile_sensor_data,
            callback_group=self._callback_group,
        )
        self._depth_subscriber = Subscriber(
            self,
            Image,
            self.get_parameter('camera.depth_topic').value,
            qos_profile=qos_profile_sensor_data,
            callback_group=self._callback_group,
        )
        # 同步器，保证rgb和depth图像的时间戳差值在允许范围内，并同步获取
        self._image_synchronizer = ApproximateTimeSynchronizer(
            [self._color_subscriber, self._depth_subscriber],
            queue_size=5,
            slop=float(
                self.get_parameter(
                    'camera.max_image_time_difference'
                ).value)
        )
        self._image_synchronizer.registerCallback(
            self._synchronized_image_callback
        )
        self.create_subscription(
            CameraInfo,
            self.get_parameter('camera.camera_info_topic').value,
            self._camera_info_callback,
            qos_profile_sensor_data,
            callback_group=self._callback_group,
        )
        # 创建触发监听服务
        self.create_service(
            DetectGrasp,
            '/detect_grasp',
            self._handle_detect_grasp,
            callback_group=self._callback_group,
        )
        # 创建分割定时器，分割实时运行，加快推理速度，减少gpu的warmup时间
        self._segment_timer = self.create_timer(
            1.0 / float(
                self.get_parameter('object_detect.inference_rate').value
            ),
            self._run_realtime_segmentation,
            callback_group=self._callback_group,
        )

    def _synchronized_image_callback(
        self, color: Image, depth: Image
    ) -> None:
        """Assemble and cache the newest synchronized input frame."""
        with self._robot_pose_lock:
            base_to_gripper = (
                None if self._t_base_to_gripper is None
                else self._t_base_to_gripper.copy()
            )
            base_to_camera = (
                None if self._t_base_to_camera is None
                else self._t_base_to_camera.copy()
            )
        with self._image_lock:
            if self._latest_camera_info is None:
                return
            self._latest_input_frame = _InputFrame(
                color_msg=color,
                depth_msg=depth,
                camera_info_msg=self._latest_camera_info,
                t_base_to_gripper=base_to_gripper,
                t_base_to_camera=base_to_camera,
                source_arrival_ns=self.get_clock().now().nanoseconds,
            )

    def _camera_info_callback(self, message: CameraInfo) -> None:
        """Cache the latest depth-camera intrinsics."""
        if self._latest_camera_info is not None:
            return
        with self._image_lock:
            self._latest_camera_info = message # 只需读取一次

    def _robot_pose_callback(self, message: PoseStamped) -> None:
        """把厂家 PoseStamped 缓存为 T_base_gripper."""
        expected_frame = self.get_parameter('general.base_frame').value
        if message.header.frame_id and message.header.frame_id != expected_frame:
            self.get_logger().warning(
                f'Ignoring robot pose in frame {message.header.frame_id}; '
                f'expected {expected_frame}',
                throttle_duration_sec=5.0,
            )
            return
        orientation = message.pose.orientation
        try:
            rotation = quaternion_to_rotation_matrix(
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            )
        except ValueError as error:
            self.get_logger().warning(f'Ignoring invalid robot pose: {error}')
            return
        position = message.pose.position
        unit = self.get_parameter('general.pose_units').value
        if unit not in POSE_UNITS:
            raise ValueError(f'pose_units must be one of {list(POSE_UNITS)}')
        scale = POSE_UNITS[unit]
        translation_norm = np.array([position.x, position.y, position.z], dtype=np.float64) * scale
        base_to_gripper = make_transform(
            rotation, translation_norm
        )
        with self._robot_pose_lock:
            self._t_base_to_gripper = base_to_gripper
            self._t_base_to_camera = base_to_gripper @ self._t_gripper_to_camera

    def _run_realtime_segmentation(self) -> None:
        """Segment the newest RGB-D pair, publish it, and cache its mask."""
        if self._object_detector is None:
            return
        if not self._segmentation_inference_lock.acquire(blocking=False):
            return
        try:
            with self._image_lock:
                input_frame = self._latest_input_frame
                self._latest_input_frame = None
            if input_frame is None:
                return

            color_bgr = color_message_to_bgr(input_frame.color_msg)
            depth = depth_message_to_numpy(input_frame.depth_msg)
            intrinsics = camera_intrinsics(
                input_frame.camera_info_msg,
                depth,
                scale=self.get_parameter('camera.depth_scale').value,
            )

            result = self._detect_target_object(color_bgr, depth)
            segmented_frame = _SegmentedFrame(
                color_rgb=color_bgr[:, :, ::-1].copy(),
                depth=depth,
                intrinsics=intrinsics,
                target_mask=result.target_mask.copy(),
                stamp=input_frame.color_msg.header.stamp,
                base_to_gripper=input_frame.t_base_to_gripper,
                base_to_camera=input_frame.t_base_to_camera,
                source_arrival_ns=input_frame.source_arrival_ns,
            )
            with self._segmentation_result_lock:
                self._latest_segmented_frame = segmented_frame

            segment_image_publisher = self._visual_topic.segment_image_publisher if self._visual_topic is not None else None
            if self._visual_topic is not None and self._visual_topic.segment_image_publisher is not None:
                segment_image_publisher.publish(
                    bgr_to_image_message(result.annotated_image, input_frame.color_msg)
                )
        except Exception as error:
            self.get_logger().error(
                f'Continuous segmentation failed: {error}',
                throttle_duration_sec=3.0,
            )
        finally:
            self._segmentation_inference_lock.release()

    @staticmethod
    def _transform_to_pose_stamped(
        transform: np.ndarray, frame_id: str, stamp
    ) -> PoseStamped:
        """把 4x4 抓取位姿变换矩阵转换为 PoseStamped。"""
        message = PoseStamped()
        message.header.stamp = stamp
        message.header.frame_id = frame_id
        message.pose.position.x = float(transform[0, 3])
        message.pose.position.y = float(transform[1, 3])
        message.pose.position.z = float(transform[2, 3])
        qx, qy, qz, qw = rotation_matrix_to_quaternion(transform[:3, :3])
        message.pose.orientation.x = qx
        message.pose.orientation.y = qy
        message.pose.orientation.z = qz
        message.pose.orientation.w = qw
        return message

    def _filter_and_rank_preferred_grasps(
        self,
        prediction: GraspPrediction,
        base_to_camera: np.ndarray,
        base_to_gripper: np.ndarray,
    ) -> None:
        """Keep safe downward grasps and prefer the least gripper rotation."""
        if len(prediction) == 0:
            return

        rotations = np.asarray(
            prediction.grasps.rotation_matrices, dtype=np.float64
        )
        preferred_direction = base_down_direction_in_camera(base_to_camera)
        approach_alignments = rotations[:, :, 0] @ preferred_direction
        maximum_angle = float(
            self.get_parameter('graspnet.max_approach_angle_deg').value
        )
        minimum_alignment = np.cos(np.deg2rad(maximum_angle))
        downward_mask = approach_alignments >= minimum_alignment

        camera_to_gripper_rotations = rotations @ R_GRASP_GRIPPER
        target_rotations = (
            base_to_camera[:3, :3] @ camera_to_gripper_rotations
        )
        relative_rotations = (
            base_to_gripper[:3, :3].T @ target_rotations
        )
        rotation_similarities = np.trace(
            relative_rotations, axis1=1, axis2=2
        )

        prediction.filter_and_rank(
            downward_mask,
            rotation_similarities,
            approach_alignments
        )

    def _handle_detect_grasp(
        self, request: DetectGrasp.Request, response: DetectGrasp.Response
    ) -> DetectGrasp.Response:
        """Run GraspNet at most twice and return the first valid grasp."""
        del request
        if not self._grasp_inference_lock.acquire(blocking=False):
            return self._assemble_response(response, False, 'Pose detection is already running')
        try:
            frame = self._snapshot_segmented_frame()
            if not np.any(frame.target_mask):
                return self._assemble_response(response, False, 'Latest segmentation contains no target object')
            if frame.base_to_camera is None or frame.base_to_gripper is None:
                return self._assemble_response(response, False, 'Latest segmented frame has no robot pose')
            maximum_attempts = 2
            for attempt in range(1, maximum_attempts + 1):
                prediction = self._pose_detector.predict(
                    frame.color_rgb,
                    frame.depth,
                    frame.target_mask,
                    frame.intrinsics,
                    max_grasps=self.get_parameter('graspnet.max_grasps').value,
                )
                self._filter_and_rank_preferred_grasps(
                    prediction,
                    frame.base_to_camera,
                    frame.base_to_gripper,
                )
                best = prediction.best
                if best is not None:
                    break
                if attempt < maximum_attempts:
                    self.get_logger().warning(f'No valid grasp on attempt {attempt}; retrying')

            if self._visual_topic is not None and self._visual_topic.marker_publisher is not None:
                self._publish_rviz_result(prediction.scene_points, best, frame.stamp)
            response = self._response_pose(response, best, frame.stamp, frame.base_to_camera)
        except Exception as error:
            return self._assemble_response(response, False, f'Grasp detection failed: {error}')
        finally:
            self._grasp_inference_lock.release()
        return response

    def _snapshot_segmented_frame(self) -> _SegmentedFrame:
        """Return a fresh snapshot of the latest completed segmentation."""
        with self._segmentation_result_lock:
            frame = self._latest_segmented_frame
        if frame is None:
            raise RuntimeError('Still waiting for the first segmentation result')
        now_ns = self.get_clock().now().nanoseconds
        maximum_age_ns = int(
            float(self.get_parameter('camera.max_image_age').value) * 1e9
        )
        if now_ns - frame.source_arrival_ns > maximum_age_ns:
            raise RuntimeError('Latest segmentation result is too old')
        return frame

    def _detect_target_object(
        self, color_bgr: np.ndarray, depth: np.ndarray
    ) -> SegmentationResult:
        """Create the mask that will later be supplied by object_detect."""
        if color_bgr.shape[:2] != depth.shape:
            raise ValueError(
                'color/depth shapes differ: '
                f'{color_bgr.shape[:2]} vs {depth.shape}'
            )
        return self._object_detector.segment(color_bgr)

    def _response_pose(
        self,
        response: DetectGrasp.Response,
        best: GraspCandidate | None, 
        stamp: Time, 
        t_base_to_camera: np.ndarray
    ) -> DetectGrasp.Response:
        if best is None:
            self.get_logger().warning('No valid grasp was found')
            return self._assemble_response(
                response, False, 'No valid grasp was found'
            )
        # 实际识别的位姿，是目标物体在graspnet坐标系下，基于相机的位姿
        t_camera_to_graspnet_detect = make_transform(
            best.rotation_matrix, best.translation
        )
        # 将gripper在graspnet下识别的坐标位姿转换为在相机坐标系下的位姿
        t_camera_to_gripper = t_camera_to_graspnet_detect @ self._t_graspnet_to_gripper
        t_base_to_gripper = transform_base_to_grasp_simple(
            t_base_to_camera,
            t_camera_to_gripper
        )
        result_pose = self._transform_to_pose_stamped(t_base_to_gripper, self.get_parameter('general.base_frame').value, stamp)
        return self._assemble_response(response, True, 'Detection completed', result_pose, best.score, best.width)

    def _publish_rviz_result(
        self,
        points,
        best: GraspCandidate | None,
        stamp,
    ) -> None:
        """发布最佳抓取线框和场景点云."""
        visual_topic = self._visual_topic
        if visual_topic is None or visual_topic.marker_publisher is None:
            return
        visual_topic.marker_publisher.publish(
            make_grasp_markers(
                best,
                stamp,
                self.get_parameter('camera.camera_frame').value,
                float(self.get_parameter('rviz.gripper_line_width').value),
            )
        )
        visual_topic.cloud_publisher.publish(
            make_scene_cloud(
                points,
                stamp,
                self.get_parameter('camera.camera_frame').value,
                self.get_parameter('rviz.rviz_cloud_max_points').value,
            )
        )
        topics = '/grasp_candidates and /graspnet_scene_cloud'
        self.get_logger().info(f'Published RViz2 data to {topics}')

    def _validate_parameters(self) -> None:
        """Fail during initialization when configuration is inconsistent."""
        pose_units = str(self.get_parameter('general.pose_units').value)
        if pose_units not in POSE_UNITS:
            raise ValueError(f'pose_units must be one of {list(POSE_UNITS)}')

        string_parameters = (
            'graspnet.checkpoint_path',
            'general.hand_eye_calibration_path',
            'camera.camera_frame',
            'general.base_frame',
            'graspnet.device',
            'camera.rgb_topic',
            'camera.depth_topic',
            'camera.camera_info_topic',
            'general.robot_pose_topic',
        )
        for name in string_parameters:
            if not str(self.get_parameter(name).value).strip():
                raise ValueError(f'{name} must not be empty')

        checkpoint = Path(
            str(self.get_parameter('graspnet.checkpoint_path').value)
        ).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')

        object_model = Path(
            str(self.get_parameter('object_detect.object_model_path').value)
        ).expanduser()
        if not object_model.is_file():
            raise FileNotFoundError(
                f'YOLO segmentation model not found: {object_model}'
            )

        hand_eye_path = Path(
            str(self.get_parameter('general.hand_eye_calibration_path').value)
        ).expanduser()
        if not hand_eye_path.is_file():
            raise FileNotFoundError(
                f'Hand-eye calibration not found: {hand_eye_path}'
            )

        positive_integers = ('graspnet.num_point', 'graspnet.num_view', 'graspnet.max_grasps')
        for name in positive_integers:
            value = self.get_parameter(name).value
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')

        maximum_points = self.get_parameter('rviz.rviz_cloud_max_points').value
        if (
            isinstance(maximum_points, bool)
            or not isinstance(maximum_points, int)
            or maximum_points < 0
        ):
            raise ValueError(
                'rviz_cloud_max_points must be a non-negative integer'
            )

        positive_numbers = (
            'camera.depth_scale',
            'camera.max_image_age',
            'object_detect.inference_rate',
            'graspnet.voxel_size',
            'graspnet.max_gripper_width',
            'rviz.gripper_line_width',
        )
        for name in positive_numbers:
            if float(self.get_parameter(name).value) <= 0.0:
                raise ValueError(f'{name} must be greater than zero')

        bounded_numbers = ('graspnet.collision_thresh',)
        for name in bounded_numbers:
            value = float(self.get_parameter(name).value)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must be between zero and one')

        maximum_approach_angle = float(
            self.get_parameter('graspnet.max_approach_angle_deg').value
        )
        if not 0.0 <= maximum_approach_angle <= 90.0:
            raise ValueError(
                'graspnet.max_approach_angle_deg must be between zero and 90'
            )

        time_difference = float(
            self.get_parameter('camera.max_image_time_difference').value
        )
        if time_difference < 0.0:
            raise ValueError(
                'max_image_time_difference must be non-negative'
            )
        for name in (
            'object_detect.visual_enable',
            'rviz.grasp_visual_enable'
        ):
            if not isinstance(self.get_parameter(name).value, bool):
                raise ValueError(f'{name} must be boolean')

    def _assemble_response(
        self,
        response: DetectGrasp.Response,
        success: bool = False,
        message: str = '', 
        base_to_gripper_pose: PoseStamped = None,
        score: float = 0.0,
        width: float = 0.0
    ) -> DetectGrasp.Response:
        '''Create a Trigger response with the given success and message.'''
        self.get_logger().info(f"Response: success={success}, message='{message}'")
        response.success = success
        response.message = message
        if base_to_gripper_pose is not None:
            response.base_to_gripper_pose = base_to_gripper_pose
        response.score = score
        response.width = width
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PoseDetectNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
