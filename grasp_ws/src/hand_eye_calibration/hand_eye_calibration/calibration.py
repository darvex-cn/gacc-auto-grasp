"""与 ROS 无关的 ChArUco 检测及眼在手上手眼标定算法."""

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class CalibrationSample:
    """一组同步采样数据，两个矩阵均采用 T_父坐标系_子坐标系 记法."""

    # 机械臂末端在基座坐标系中的位姿 T_base_gripper。
    base_to_gripper: np.ndarray
    # 标定板在相机坐标系中的位姿 T_camera_target。
    camera_to_target: np.ndarray


@dataclass(frozen=True)
class CalibrationResult:
    """手眼标定结果以及固定标定板的一致性误差."""

    # 最终需要发布的外参：相机在末端坐标系中的位姿 T_gripper_camera。
    gripper_to_camera: np.ndarray
    method: str
    translation_rms_m: float
    rotation_rms_deg: float
    sample_count: int


class CharucoPoseDetector:
    """从彩色图像估计 ChArUco 标定板在相机坐标系中的位姿."""

    def __init__(
        self,
        squares_x: int = 11,
        squares_y: int = 8,
        square_length: float = 0.020,
        marker_length: float = 0.015,
        dictionary_name: str = 'DICT_5X5_100',
        minimum_corners: int = 6,
    ) -> None:
        if squares_x < 2 or squares_y < 2:
            raise ValueError('ChArUco board must contain at least 2x2 squares')
        if not 0 < marker_length < square_length:
            raise ValueError('marker_length must be positive and smaller than square_length')
        if not hasattr(cv2.aruco, dictionary_name):
            raise ValueError(f'unknown ArUco dictionary: {dictionary_name}')
        # dictionary 只是编码字典，不是标定板图片路径。
        dictionary_id = getattr(cv2.aruco, dictionary_name)
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        # 在内存中建立与实体标定板一致的几何模型，不会生成或保存图片。
        self.board = cv2.aruco.CharucoBoard_create(
            squares_x,
            squares_y,
            square_length,
            marker_length,
            self.dictionary,
        )
        self.parameters = cv2.aruco.DetectorParameters_create()
        self.minimum_corners = minimum_corners

    def detect(
        self,
        image_bgr: np.ndarray,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray, int]:
        """返回 T_camera_target、带检测标记的图像和 ChArUco 角点数量."""
        annotated = image_bgr.copy()
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # 第一步：在图像中检测独立的 ArUco Marker。
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
            gray,
            self.dictionary,
            parameters=self.parameters,
            cameraMatrix=camera_matrix,
            distCoeff=distortion,
        )
        if marker_ids is None:
            return None, annotated, 0

        cv2.aruco.drawDetectedMarkers(annotated, marker_corners, marker_ids)
        # 第二步：利用 Marker ID 和棋盘结构插值出精度更高的 ChArUco 角点。
        corner_count, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            self.board,
            cameraMatrix=camera_matrix,
            distCoeffs=distortion,
        )
        count = int(corner_count or 0)
        if charuco_ids is None or count < self.minimum_corners:
            return None, annotated, count

        cv2.aruco.drawDetectedCornersCharuco(
            annotated, charuco_corners, charuco_ids
        )
        # 第三步：结合相机内参和畸变参数，通过 PnP 估计标定板位姿。
        valid, rotation_vector, translation = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners,
            charuco_ids,
            self.board,
            camera_matrix,
            distortion,
            None,
            None,
        )
        if not valid:
            return None, annotated, count

        # OpenCV 返回的是 target -> camera，即 T_camera_target。
        rotation, _ = cv2.Rodrigues(rotation_vector)
        transform = make_transform(rotation, translation.reshape(3))
        cv2.drawFrameAxes(
            annotated,
            camera_matrix,
            distortion,
            rotation_vector,
            translation,
            2.0 * self.board.getSquareLength(),
        )
        return transform, annotated, count


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """由 3x3 旋转矩阵和平移向量构造齐次变换矩阵."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def quaternion_to_rotation(x: float, y: float, z: float, w: float) -> np.ndarray:
    """把 ROS 顺序 (x, y, z, w) 的四元数转换为旋转矩阵."""
    quaternion = np.array([x, y, z, w], dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm == 0.0:
        raise ValueError('zero-length quaternion')
    x, y, z, w = quaternion / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """把旋转矩阵转换成 ROS 顺序 (x, y, z, w) 的单位四元数."""
    matrix = np.asarray(rotation, dtype=np.float64)
    quaternion = np.empty(4, dtype=np.float64)
    trace = np.trace(matrix)
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion[:] = [
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        ]
    else:
        index = int(np.argmax(np.diag(matrix)))
        next_index = (index + 1) % 3
        last_index = (index + 2) % 3
        scale = np.sqrt(
            1 + matrix[index, index] - matrix[next_index, next_index]
            - matrix[last_index, last_index]
        ) * 2.0
        quaternion[index] = 0.25 * scale
        quaternion[3] = (
            matrix[last_index, next_index] - matrix[next_index, last_index]
        ) / scale
        quaternion[next_index] = (
            matrix[next_index, index] + matrix[index, next_index]
        ) / scale
        quaternion[last_index] = (
            matrix[last_index, index] + matrix[index, last_index]
        ) / scale
    return quaternion / np.linalg.norm(quaternion)


# OpenCV 支持的手眼标定求解器名称与枚举值。
HAND_EYE_METHODS = {
    'TSAI': cv2.CALIB_HAND_EYE_TSAI,
    'PARK': cv2.CALIB_HAND_EYE_PARK,
    'HORAUD': cv2.CALIB_HAND_EYE_HORAUD,
    'ANDREFF': cv2.CALIB_HAND_EYE_ANDREFF,
    'DANIILIDIS': cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def calibrate_eye_in_hand(
    samples: Sequence[CalibrationSample], method: str = 'PARK'
) -> CalibrationResult:
    """根据多组 T_base_gripper 和 T_camera_target 求解 T_gripper_camera."""
    method_name = method.upper()
    if method_name not in HAND_EYE_METHODS:
        raise ValueError(f'unsupported method {method}; choose {list(HAND_EYE_METHODS)}')
    if len(samples) < 3:
        raise ValueError('OpenCV hand-eye calibration requires at least 3 samples')

    # OpenCV 参数名中的 gripper2base 表示把末端坐标转换到基座坐标，
    # 正好对应厂家 PoseStamped 话题给出的 T_base_gripper。
    rotations_gripper_to_base = [sample.base_to_gripper[:3, :3] for sample in samples]
    translations_gripper_to_base = [sample.base_to_gripper[:3, 3] for sample in samples]
    rotations_target_to_camera = [sample.camera_to_target[:3, :3] for sample in samples]
    translations_target_to_camera = [sample.camera_to_target[:3, 3] for sample in samples]
    # 返回 camera -> gripper，即相机在机械臂末端坐标系中的固定外参。
    rotation_camera_to_gripper, translation_camera_to_gripper = cv2.calibrateHandEye(
        rotations_gripper_to_base,
        translations_gripper_to_base,
        rotations_target_to_camera,
        translations_target_to_camera,
        method=HAND_EYE_METHODS[method_name],
    )
    gripper_to_camera = make_transform(
        rotation_camera_to_gripper, translation_camera_to_gripper
    )
    # 标定板实际固定不动，因此可用它在 base 下的漂移量评价标定质量。
    translation_rms, rotation_rms = evaluate_target_consistency(
        samples, gripper_to_camera
    )
    return CalibrationResult(
        gripper_to_camera=gripper_to_camera,
        method=method_name,
        translation_rms_m=translation_rms,
        rotation_rms_deg=rotation_rms,
        sample_count=len(samples),
    )


def evaluate_target_consistency(
    samples: Sequence[CalibrationSample], gripper_to_camera: np.ndarray
) -> tuple[float, float]:
    """计算固定标定板在机器人基座坐标系中的平移和旋转 RMS 漂移."""
    # T_base_target = T_base_gripper * T_gripper_camera * T_camera_target。
    base_to_targets = [
        sample.base_to_gripper @ gripper_to_camera @ sample.camera_to_target
        for sample in samples
    ]
    translations = np.stack([transform[:3, 3] for transform in base_to_targets])
    translation_mean = translations.mean(axis=0)
    translation_rms = float(
        np.sqrt(np.mean(np.sum((translations - translation_mean) ** 2, axis=1)))
    )

    # 旋转矩阵不能直接逐元素平均；先平均，再用 SVD 投影回合法 SO(3)。
    rotation_mean_raw = np.mean(
        np.stack([transform[:3, :3] for transform in base_to_targets]), axis=0
    )
    u_matrix, _, v_transpose = np.linalg.svd(rotation_mean_raw)
    rotation_mean = u_matrix @ v_transpose
    if np.linalg.det(rotation_mean) < 0:
        u_matrix[:, -1] *= -1
        rotation_mean = u_matrix @ v_transpose
    angles = []
    for transform in base_to_targets:
        delta = rotation_mean.T @ transform[:3, :3]
        cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cosine)))
    rotation_rms = float(np.sqrt(np.mean(np.square(angles))))
    return translation_rms, rotation_rms
