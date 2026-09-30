"""GraspNet inference wrapper used by the ROS 2 pose-detection node."""

from pathlib import Path
from time import perf_counter

import numpy as np

from graspnet_adapter import (
    CameraIntrinsics,
    GraspNetInference,
    GraspPrediction
)


'''
位姿检测器类，负责加载GraspNet模型并运行检测以及矩阵转换
@param checkpoint_path: GraspNet模型的检查点路径
@param num_point: 输入点云的点数
@param num_view: 视图数量
@param collision_thresh: 碰撞阈值
@param voxel_size: 体素大小
@param device: 运行设备（如 'cuda:0' 或 'cpu'）
'''
class PoseDetector:
    """Own a persistent GraspNet model and run RGB-D grasp inference."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        num_point: int = 20000,
        num_view: int = 300,
        collision_thresh: float = 0.01,
        voxel_size: float = 0.01,
        max_gripper_width: float | None = None,
        device: str = 'cuda:0',
        logger: object | None = None,
    ) -> None:
        self._logger = logger
        self._logger.info(f'Loading GraspNet checkpoint: {checkpoint_path}')
        self._inference = GraspNetInference(
            checkpoint_path,
            num_point=num_point,
            num_view=num_view,
            collision_thresh=collision_thresh,
            voxel_size=voxel_size,
            max_gripper_width=max_gripper_width,
            device=device,
        )
        self._logger.info("GraspNet checkpoint Loaded successfully!")

    def predict(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        intrinsics: CameraIntrinsics,
        *,
        max_grasps: int = 5,
    ) -> GraspPrediction:
        """Run grasp inference on one synchronized live RGB-D frame."""
        if color.shape[:2] != depth.shape:
            raise ValueError(
                f'color and depth shapes differ: {color.shape[:2]} '
                f'versus {depth.shape}'
            )
        if mask.shape != depth.shape:
            raise ValueError(
                f'mask and depth shapes differ: {mask.shape} '
                f'versus {depth.shape}'
            )
        
        if isinstance(max_grasps, bool) or not isinstance(max_grasps, int):
            raise TypeError('max_grasps must be an integer')
        if max_grasps <= 0:
            raise ValueError('max_grasps must be greater than zero')
        start_time = perf_counter()
        prediction = self._inference.predict(
            color,
            depth,
            intrinsics,
            mask,
            max_grasps=max_grasps,
        )
        self._logger.info(f"Grasp prediction cost: {(perf_counter()-start_time)*1000:.2f} ms.")
        return prediction
