"""Publish live YOLO segmentation results as a ROS 2 image."""

from .utils.tools import load_config

from pathlib import Path
from threading import Lock
from time import perf_counter

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from .object_detect import ObjectDetector
from .utils.image_utils import color_message_to_bgr, bgr_to_image_message
from .constants import SEGMENTATION_TEST_CONFIG_PARAMETERS

class SegmentationTestNode(Node):
    """Run YOLO segmentation on the latest camera frame for RViz."""

    def __init__(self) -> None:
        super().__init__('segmentation_test_node')
        config_path = (
            Path(get_package_share_directory('pose_detect'))
            / 'config'
            / 'config.yaml'
        )
        config = load_config(
            config_path,
            'segmentation_test_node',
            SEGMENTATION_TEST_CONFIG_PARAMETERS,
            path_params={'object_model_path'},
        )
        for name, value in config.items():
            self.declare_parameter(name, value)
        self._validate_parameters()

        self._callback_group = ReentrantCallbackGroup()
        self._image_lock = Lock()
        self._inference_lock = Lock()
        self._latest_image: Image | None = None

        model_path = self.get_parameter('object_model_path').value
        self.get_logger().info(f'Loading segmentation model: {model_path}')
        self._detector = ObjectDetector(
            model_path,
            target_class=self.get_parameter('target_class').value,
            selection_policy=self.get_parameter('selection_policy').value,
            visual_enable=self.get_parameter('visual_enable').value,
            confidence_threshold=self.get_parameter(
                'confidence_threshold'
            ).value,
            iou_threshold=self.get_parameter('iou_threshold').value,
            mask_threshold=self.get_parameter('mask_threshold').value,
            image_size=self.get_parameter('image_size').value,
            device=self.get_parameter('device').value,
            quantize=self.get_parameter('quantize').value,
            retina_masks=self.get_parameter('retina_masks').value,
            logger=self.get_logger()
        )

        self._publisher = self.create_publisher(
            Image,
            self.get_parameter('output_topic').value,
            1,  # Use reliable QoS to keep RViz image display stable.
        )
        self.create_subscription(
            Image,
            self.get_parameter('rgb_topic').value,
            self._image_callback,
            qos_profile_sensor_data,
            callback_group=self._callback_group,
        )
        self.create_timer(
            1.0 / float(self.get_parameter('inference_rate').value),
            self._process_latest_image,
            callback_group=self._callback_group,
        )
        self.get_logger().info(
            'Segmentation test ready: %s -> %s'
            % (
                self.get_parameter('rgb_topic').value,
                self.get_parameter('output_topic').value,
            )
        )

    def _image_callback(self, message: Image) -> None:
        """Cache only the newest camera frame."""
        with self._image_lock:
            self._latest_image = message

    def _process_latest_image(self) -> None:
        """Segment a new frame and publish its annotated image."""
        if not self._inference_lock.acquire(blocking=False):
            return
        try:
            with self._image_lock:
                message = self._latest_image
                self._latest_image = None
            if message is None:
                return

            image = color_message_to_bgr(message)
            started_at = perf_counter()
            result = self._detector.segment(image)
            processing_elapsed = perf_counter() - started_at
            if result.annotated_image is not None:
                output = bgr_to_image_message(
                    result.annotated_image,
                    message
                )
                self._publisher.publish(output)
                action = 'Published'
            else:
                action = 'Processed'
            model_speed = ', '.join(
                f'{name}={duration:.1f} ms'
                for name, duration in result.speed.items()
            )
            self.get_logger().info(
                f'{action} segmentation with {len(result.instances)} '
                f'motor-part instance(s); model [{model_speed}], '
                f'processing={processing_elapsed * 1000:.1f} ms, '
                f'FPS={1.0 / (processing_elapsed or 1e-6):.1f}',
                throttle_duration_sec=3.0,
            )
        except Exception as error:
            self.get_logger().error(
                f'Segmentation inference failed: {error}',
                throttle_duration_sec=3.0,
            )
        finally:
            self._inference_lock.release()

    def _validate_parameters(self) -> None:
        """Reject invalid settings before loading the model."""
        for name in (
            'object_model_path',
            'selection_policy',
            'device',
            'rgb_topic',
            'output_topic',
        ):
            if not str(self.get_parameter(name).value).strip():
                raise ValueError(f'{name} must not be empty')
        target_classes = self.get_parameter('target_class').value
        if not isinstance(target_classes, list) or not target_classes:
            raise ValueError('target_class must be a non-empty string list')
        if any(
            not isinstance(name, str) or not name.strip()
            for name in target_classes
        ):
            raise ValueError(
                'Every target_class entry must be a non-empty string'
            )
        model_path = Path(
            str(self.get_parameter('object_model_path').value)
        ).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f'YOLO model not found: {model_path}')
        if self.get_parameter('selection_policy').value not in (
            'highest_confidence',
            'largest_area',
            'all',
        ):
            raise ValueError('selection_policy is invalid')
        if not isinstance(self.get_parameter('visual_enable').value, bool):
            raise ValueError('visual_enable must be boolean')
        for name in (
            'confidence_threshold',
            'iou_threshold',
            'mask_threshold',
        ):
            value = float(self.get_parameter(name).value)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must be between zero and one')
        if int(self.get_parameter('image_size').value) <= 0:
            raise ValueError('image_size must be greater than zero')
        if float(self.get_parameter('inference_rate').value) <= 0.0:
            raise ValueError('inference_rate must be greater than zero')
        quantize = self.get_parameter('quantize').value
        if quantize not in (16, 32):
            raise ValueError('quantize must be 16 (FP16) or 32 (FP32)')
        if not isinstance(self.get_parameter('retina_masks').value, bool):
            raise ValueError('retina_masks must be boolean')


def main(args: list[str] | None = None) -> None:
    """Run the live segmentation test node."""
    rclpy.init(args=args)
    node = SegmentationTestNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
