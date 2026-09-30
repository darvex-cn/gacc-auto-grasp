"""YOLO26 instance segmentation for selecting a grasp target."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal, Sequence, get_args

import cv2
import numpy as np
from ultralytics import YOLO


SelectionPolicy = Literal['highest_confidence', 'largest_area']
TargetClasses = str | Sequence[str]
_VALID_SELECTION_POLICIES = get_args(SelectionPolicy)


@dataclass(frozen=True)
class SegmentationInstance:
    """One segmented object instance."""

    class_id: int
    class_name: str
    confidence: float
    bounding_box: tuple[float, float, float, float]
    mask: np.ndarray

    @property
    def center(self) -> tuple[float, float]:
        """Return the bounding-box center in image pixels."""
        x_min, y_min, x_max, y_max = self.bounding_box
        return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)

    @property
    def area(self) -> int:
        """Return the number of foreground pixels in the mask."""
        return int(np.count_nonzero(self.mask))


@dataclass(frozen=True)
class SegmentationResult:
    """Instance predictions and the selected GraspNet target mask."""

    instances: tuple[SegmentationInstance, ...]
    target_mask: np.ndarray
    annotated_image: np.ndarray | None
    # 用于记录模型推理时间的字典，键为阶段名称，值为耗时（毫秒）
    speed: dict[str, float]


class ObjectDetector:
    """Segment and select configured motor-part classes with YOLO26m-seg."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        target_class: TargetClasses = (
            'motor_part_type1',
            'motor_part_type2',
        ),
        selection_policy: SelectionPolicy = 'highest_confidence',
        visual_enable: bool = False,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        mask_threshold: float = 0.5,
        image_size: int = 640,
        device: str = 'cuda:0',
        quantize: int | None = None,
        retina_masks: bool = True,
        logger: object | None = None,
    ) -> None:
        self._logger = logger
        """Load the segmentation model and validate inference settings."""
        self._logger.info(f'Loading Object detection model: {model_path}')
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(
                f'YOLO segmentation model not found: {model_path}'
            )
        target_classes = self._normalize_target_classes(target_class)
        if selection_policy not in _VALID_SELECTION_POLICIES:
            raise ValueError(
                'selection_policy must be one of '
                f'{_VALID_SELECTION_POLICIES}'
            )
        if not isinstance(visual_enable, bool):
            raise ValueError('visual_enable must be boolean')
        for name, value in (
            ('confidence_threshold', confidence_threshold),
            ('iou_threshold', iou_threshold),
            ('mask_threshold', mask_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must be between zero and one')
        if image_size <= 0:
            raise ValueError('image_size must be greater than zero')
        if quantize not in (None, 16, 32):
            raise ValueError('quantize must be null, 16 (FP16), or 32 (FP32)')
        if not isinstance(retina_masks, bool):
            raise ValueError('retina_masks must be boolean')

        self._target_classes = target_classes
        self._selection_policy = selection_policy
        self._visual_enable = visual_enable
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._mask_threshold = mask_threshold
        self._image_size = image_size
        self._device = device
        self._quantize = quantize
        self._retina_masks = retina_masks
        self._model = YOLO(str(model_path))
        self._target_class_ids = self._class_ids(self._target_classes)
        self._logger.info(f'Object detection model Loaded successfully!')
        self._warm_up()

    def _warm_up(self) -> None:
        """Warm up the complete segmentation path with black and white images."""
        started_at = perf_counter()
        image_shape = (self._image_size, self._image_size, 3)
        for pixel_value in (0, 255):
            image = np.full(image_shape, pixel_value, dtype=np.uint8)
            self.segment(image)
        self._logger.info(
            'Object detection warm-up completed in '
            f'{(perf_counter() - started_at) * 1000:.2f} ms.'
        )

    def segment(
        self,
        image: str | Path | np.ndarray,
    ) -> SegmentationResult:
        """
        Segment the target class and select a mask for GraspNet.

        Numpy input images must use OpenCV BGR channel order. Target classes,
        selection and visualization behavior are fixed during initialization.
        """
        source = self._validate_image(image)
        start_time = perf_counter()
        results = self._model.predict(
            source=source,
            conf=self._confidence_threshold,
            iou=self._iou_threshold,
            imgsz=self._image_size,
            device=self._device,
            classes=self._target_class_ids,
            quantize=self._quantize,
            retina_masks=self._retina_masks,
            verbose=False,
        )
        if not results:
            raise RuntimeError('YOLO returned no result for the input image')
        self._logger.info('Object detection cost: 'f'{(perf_counter() - start_time) * 1000:.2f} ms.',throttle_duration_sec=5.0)
        result = results[0]
        image_height, image_width = result.orig_img.shape[:2]
        instances = self._instances_from_result(
            result, image_height, image_width
        )
        target_instances = self._select_instances(
            instances, self._selection_policy
        )
        target_mask = self._merge_masks(
            target_instances,
            (image_height, image_width),
        )
        if self._visual_enable:
            annotated_image = self._draw_instances(
                result.orig_img, instances, target_instances
            )
        else:
            annotated_image = None
        return SegmentationResult(
            instances=instances,
            target_mask=target_mask,
            annotated_image=annotated_image,
            speed={
                str(name): float(value)
                for name, value in result.speed.items()
            },
        )

    @staticmethod
    def _draw_instances(
        image: np.ndarray,
        instances: tuple[SegmentationInstance, ...],
        target_instances: tuple[SegmentationInstance, ...],
    ) -> np.ndarray:
        """Draw every instance and highlight selected targets in red."""
        annotated = image.copy()
        colors = (
            (0, 200, 255),
            (255, 120, 0),
            (80, 220, 80),
            (220, 80, 220),
        )
        target_color = (0, 0, 255)
        target_ids = {id(instance) for instance in target_instances}
        # Draw targets last so their red masks remain visible if masks overlap.
        ordered_instances = tuple(
            instance for instance in instances
            if id(instance) not in target_ids
        ) + target_instances

        # Blend each already-resized boolean mask directly into the image.
        for instance in ordered_instances:
            instance_color = (
                target_color
                if id(instance) in target_ids
                else colors[instance.class_id % len(colors)]
            )
            color = np.asarray(
                instance_color, dtype=np.float32
            )
            foreground = annotated[instance.mask].astype(np.float32)
            annotated[instance.mask] = (
                foreground * 0.55 + color * 0.45
            ).astype(np.uint8)

        image_height, image_width = annotated.shape[:2]
        for instance in ordered_instances:
            color = (
                target_color
                if id(instance) in target_ids
                else colors[instance.class_id % len(colors)]
            )
            x_min, y_min, x_max, y_max = instance.bounding_box
            top_left = (
                int(np.clip(round(x_min), 0, image_width - 1)),
                int(np.clip(round(y_min), 0, image_height - 1)),
            )
            bottom_right = (
                int(np.clip(round(x_max), 0, image_width - 1)),
                int(np.clip(round(y_max), 0, image_height - 1)),
            )
            cv2.rectangle(annotated, top_left, bottom_right, color, 2)
            label = f'{instance.class_name} {instance.confidence:.2f}'
            text_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            text_x = top_left[0]
            text_y = max(top_left[1], text_size[1] + baseline + 2)
            background_top = max(0, text_y - text_size[1] - baseline - 2)
            background_right = min(
                image_width - 1, text_x + text_size[0] + 4
            )
            cv2.rectangle(
                annotated,
                (text_x, background_top),
                (background_right, text_y + 2),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                annotated,
                label,
                (text_x + 2, text_y - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        return annotated

    def _instances_from_result(
        self, result, image_height: int, image_width: int
    ) -> tuple[SegmentationInstance, ...]:
        """Convert one Ultralytics result into immutable numpy objects."""
        if result.boxes is None or result.masks is None:
            return ()

        boxes = result.boxes.cpu()
        masks = result.masks.data.cpu().numpy()
        if len(boxes) != len(masks):
            raise RuntimeError('YOLO returned different box and mask counts')

        instances = []
        for class_id, confidence, coordinates, raw_mask in zip(
            boxes.cls.tolist(),
            boxes.conf.tolist(),
            boxes.xyxy.tolist(),
            masks,
        ):
            numeric_id = int(class_id)
            mask = self._resize_mask(raw_mask, image_height, image_width)
            instances.append(
                SegmentationInstance(
                    class_id=numeric_id,
                    class_name=str(result.names[numeric_id]),
                    confidence=float(confidence),
                    bounding_box=tuple(float(value) for value in coordinates),
                    mask=mask,
                )
            )
        return tuple(instances)

    def _resize_mask(
        self, mask: np.ndarray, image_height: int, image_width: int
    ) -> np.ndarray:
        """Return an original-resolution boolean instance mask."""
        if mask.shape != (image_height, image_width):
            mask = cv2.resize(
                mask.astype(np.float32),
                (image_width, image_height),
                interpolation=cv2.INTER_NEAREST,
            )
        return np.asarray(mask >= self._mask_threshold, dtype=bool)

    @staticmethod
    def _select_instances(
        instances: tuple[SegmentationInstance, ...],
        selection_policy: SelectionPolicy,
    ) -> tuple[SegmentationInstance, ...]:
        """Select the instances included in the GraspNet target mask."""
        if selection_policy not in _VALID_SELECTION_POLICIES:
            raise ValueError(
                f'Unsupported selection policy: {selection_policy}'
            )
        if not instances:
            return ()
        if selection_policy == 'highest_confidence':
            selected = max(
                instances, key=lambda instance: instance.confidence
            )
        elif selection_policy == 'largest_area':
            selected = max(instances, key=lambda instance: instance.area)
        else:
            raise AssertionError('unreachable selection policy')
        # Keep the return type consistent when only one target is selected.
        return (selected,)

    @staticmethod
    def _merge_masks(
        instances: tuple[SegmentationInstance, ...],
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        """Return one boolean mask containing all selected instances."""
        if not instances:
            return np.zeros(image_shape, dtype=bool)
        if len(instances) == 1:
            return instances[0].mask.copy()
        return np.logical_or.reduce(
            [instance.mask for instance in instances]
        )

    def _class_ids(self, class_names: tuple[str, ...]) -> list[int]:
        """Translate class names to the numeric IDs expected by YOLO."""
        model_names = self._model.names
        name_to_id = {
            str(name): int(index) for index, name in model_names.items()
        }
        missing = [name for name in class_names if name not in name_to_id]
        if missing:
            raise ValueError(
                f'Classes {missing!r} are not present in this model. '
                f'Available classes: {", ".join(name_to_id)}'
            )
        return [name_to_id[name] for name in class_names]

    '''
    防御性编码：将 target_classes 参数规范化为一个经过验证的、无重复的类名元组。
    @param target_classes: 目标类名，字符串或字符串列表
    @return: 经过验证的、无重复的类名元组
    '''
    @staticmethod
    def _normalize_target_classes(
        target_classes: TargetClasses,
    ) -> tuple[str, ...]:
        """Return a validated, duplicate-free tuple of class names."""
        if isinstance(target_classes, str):
            target_classes = (target_classes,)
        elif not isinstance(target_classes, Sequence):
            raise TypeError('target_class must be a string or a list of strings')
        normalized = tuple(target_classes)
        if not normalized:
            raise ValueError('target_class must not be empty')
        if any(not isinstance(name, str) or not name.strip() for name in normalized):
            raise ValueError('Every target_class entry must be a non-empty string')
        return tuple(dict.fromkeys(normalized))

    @staticmethod
    def _validate_image(
        image: str | Path | np.ndarray,
    ) -> str | np.ndarray:
        """Validate an image path or OpenCV BGR numpy image."""
        if isinstance(image, (str, Path)):
            image_path = Path(image).expanduser().resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f'Input image not found: {image_path}')
            return str(image_path)
        if not isinstance(image, np.ndarray):
            raise TypeError('image must be a path or numpy array')
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('image must be an HxWx3 BGR array')
        return image
