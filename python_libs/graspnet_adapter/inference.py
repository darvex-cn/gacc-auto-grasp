"""Reusable adapter around the original GraspNet baseline inference code."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np


DEFAULT_COLLISION_WORKSPACE_MARGIN_M = 0.04


def default_baseline_root() -> Path:
    """Return the configured or adjacent GraspNet baseline source directory."""
    configured_root = os.environ.get('GRASPNET_BASELINE_ROOT')
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[2]
    return project_root / 'third_party' / 'graspnet-baseline'


def _add_legacy_import_paths(baseline_root: Path) -> None:
    """Make the original script-style GraspNet sources importable."""
    paths = (
        baseline_root,
        baseline_root / 'models',
        baseline_root / 'dataset',
        baseline_root / 'utils',
        baseline_root / 'pointnet2',
        baseline_root / 'knn',
    )
    for path in reversed(paths):
        path_string = str(path)
        if path_string not in sys.path:
            sys.path.insert(0, path_string)


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole parameters; depth_scale converts raw depth values to metres."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 1000.0

    def validate_for(self, depth: np.ndarray) -> None:
        if depth.ndim != 2:
            raise ValueError(f'depth must be HxW, got shape {depth.shape}')
        if depth.shape != (self.height, self.width):
            raise ValueError(
                f'depth shape {depth.shape} does not match '
                f'{(self.height, self.width)}'
            )
        if self.fx <= 0 or self.fy <= 0 or self.depth_scale <= 0:
            raise ValueError('fx, fy and depth_scale must be positive')


@dataclass(frozen=True)
class GraspCandidate:
    """One grasp pose expressed in the input camera optical frame."""

    score: float
    translation: np.ndarray
    rotation_matrix: np.ndarray
    width: float
    height: float
    depth: float

    @property
    def pose_matrix(self) -> np.ndarray:
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = self.rotation_matrix
        pose[:3, 3] = self.translation
        return pose


@dataclass
class GraspPrediction:
    """Predicted grasps and the filtered scene used for collision checks."""

    grasps: Any
    scene_points: np.ndarray
    scene_colors: np.ndarray

    def __len__(self) -> int:
        return len(self.grasps)

    def filter_and_rank(
        self,
        valid_mask: np.ndarray,
        *rank_scores: np.ndarray,
    ) -> None:
        '''过滤无效抓取，然后根据多个分数对剩余抓取进行降序排列。

        valid_mask 用于过滤无效抓取；rank_scores 按优先级从高到低
        传入，每项分数都是数值越大、排序越靠前。
        '''
        selected_indices = np.flatnonzero(valid_mask)
        if selected_indices.size > 0 and rank_scores:
            # lexsort 将最后一个键作为最高优先级，因此需要反转调用方传入的
            # 优先级顺序；分数取负后即可实现降序排列。
            keys = tuple(
                -score[selected_indices]
                for score in reversed(rank_scores)
            )
            selected_indices = selected_indices[np.lexsort(keys)]

        self.grasps = self.grasps[selected_indices]

    def rank_by_approach_direction(
        self, preferred_direction: np.ndarray
    ) -> None:
        """Rank grasps by alignment of their approach axes to a direction."""
        direction = np.asarray(preferred_direction, dtype=np.float64)
        if direction.shape != (3,):
            raise ValueError(
                'preferred_direction must have shape (3,), got '
                f'{direction.shape}'
            )
        if not np.all(np.isfinite(direction)):
            raise ValueError('preferred_direction contains non-finite values')
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-8:
            raise ValueError('preferred_direction must not be zero')
        if len(self.grasps) == 0:
            return

        direction /= direction_norm
        approach_axes = np.asarray(
            self.grasps.rotation_matrices, dtype=np.float64
        )[:, :, 0]
        approach_norms = np.linalg.norm(approach_axes, axis=1)
        if np.any(approach_norms < 1e-8):
            raise ValueError('grasp rotation contains a zero approach axis')
        alignments = (
            approach_axes @ direction / approach_norms
        )
        order = np.argsort(-alignments, kind='stable')
        self.grasps = self.grasps[order]

    @property
    def best(self) -> Optional[GraspCandidate]:
        if len(self.grasps) == 0:
            return None
        return GraspCandidate(
            score=float(self.grasps.scores[0]),
            translation=np.asarray(self.grasps.translations[0]).copy(),
            rotation_matrix=np.asarray(self.grasps.rotation_matrices[0]).copy(),
            width=float(self.grasps.widths[0]),
            height=float(self.grasps.heights[0]),
            depth=float(self.grasps.depths[0]),
        )


class GraspNetInference:
    """Load the network once and run repeated RGB-D inference calls."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        num_point: int = 20000,
        num_view: int = 300,
        collision_thresh: float = 0.01,
        voxel_size: float = 0.01,
        max_gripper_width: Optional[float] = None,
        approach_distance: float = 0.05,
        device: Optional[str] = None,
        random_seed: Optional[int] = None,
        baseline_root: str | Path | None = None,
    ) -> None:
        if num_point <= 0 or num_view <= 0:
            raise ValueError('num_point and num_view must be positive')
        if voxel_size <= 0:
            raise ValueError('voxel_size must be positive')
        if max_gripper_width is not None and max_gripper_width <= 0:
            raise ValueError('max_gripper_width must be positive or None')
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f'checkpoint not found: {checkpoint}')

        self.baseline_root = Path(
            baseline_root if baseline_root is not None else default_baseline_root()
        ).expanduser().resolve()
        if not (self.baseline_root / 'models' / 'graspnet.py').is_file():
            raise FileNotFoundError(
                f'GraspNet baseline source not found: {self.baseline_root}'
            )

        _add_legacy_import_paths(self.baseline_root)
        try:
            import torch
            from graspnet import GraspNet, pred_decode
            from graspnetAPI import GraspGroup
        except ImportError as error:
            raise RuntimeError(
                'GraspNet runtime is incomplete. Install PyTorch, graspnetAPI, '
                'Open3D and compile pointnet2/knn in the ROS Python environment.'
            ) from error

        self._torch = torch
        self._pred_decode = pred_decode
        self._grasp_group_type = GraspGroup
        self.num_point = int(num_point)
        self.collision_thresh = float(collision_thresh)
        self.voxel_size = float(voxel_size)
        self.max_gripper_width = (
            None
            if max_gripper_width is None
            else float(max_gripper_width)
        )
        self.approach_distance = float(approach_distance)
        self._rng = np.random.default_rng(random_seed)
        default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(
            device if device is not None else default_device
        )
        if self.device.type != 'cuda':
            raise RuntimeError(
                'The bundled GraspNet PointNet++ extension does not support CPU '
                'inference. Run this module in an environment with CUDA available.'
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                f'CUDA device {self.device} was requested, but PyTorch cannot '
                'access CUDA'
            )
        self.net = GraspNet(
            input_feature_dim=0,
            num_view=num_view,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
        ).to(self.device)
        loaded_checkpoint = self._load_checkpoint(checkpoint)
        self.net.load_state_dict(loaded_checkpoint['model_state_dict'])
        self.checkpoint_epoch = loaded_checkpoint.get('epoch')
        self.net.eval()

    def _load_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        try:
            return self._torch.load(
                str(checkpoint_path), map_location=self.device, weights_only=False
            )
        except TypeError:
            return self._torch.load(str(checkpoint_path), map_location=self.device)

    @staticmethod
    def create_point_cloud(
        depth: np.ndarray, intrinsics: CameraIntrinsics
    ) -> np.ndarray:
        depth_array = np.asarray(depth)
        intrinsics.validate_for(depth_array)
        x_map, y_map = np.meshgrid(
            np.arange(intrinsics.width, dtype=np.float32),
            np.arange(intrinsics.height, dtype=np.float32),
        )
        z = depth_array.astype(np.float32) / np.float32(intrinsics.depth_scale)
        x = (x_map - np.float32(intrinsics.cx)) * z / np.float32(intrinsics.fx)
        y = (y_map - np.float32(intrinsics.cy)) * z / np.float32(intrinsics.fy)
        return np.stack((x, y, z), axis=-1)

    def prepare_input(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
        mask: Optional[np.ndarray] = None,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        color_array = np.asarray(color)
        depth_array = np.asarray(depth)
        intrinsics.validate_for(depth_array)
        if color_array.shape != (intrinsics.height, intrinsics.width, 3):
            raise ValueError(f'color must be HxWx3, got {color_array.shape}')

        if np.issubdtype(color_array.dtype, np.integer):
            normalized_color = color_array.astype(np.float32) / 255.0
        else:
            normalized_color = color_array.astype(np.float32)
            if normalized_color.size and normalized_color.max() > 1.0:
                normalized_color /= 255.0

        depth_valid_mask = np.isfinite(depth_array) & (depth_array > 0)
        object_mask = depth_valid_mask.copy()
        if mask is not None:
            mask_array = np.asarray(mask)
            if mask_array.shape != depth_array.shape:
                raise ValueError('mask must have the same HxW shape as depth')
            object_mask &= mask_array.astype(bool)

        cloud = self.create_point_cloud(depth_array, intrinsics)
        object_points = cloud[object_mask].astype(np.float32, copy=False)
        object_colors = normalized_color[object_mask].astype(
            np.float32, copy=False
        )
        if len(object_points) == 0:
            raise ValueError('no valid depth points remain after applying the mask')

        # GraspNet only receives the segmented object. Collision detection also
        # receives nearby table and obstacle points from the full depth cloud.
        margin = np.float32(DEFAULT_COLLISION_WORKSPACE_MARGIN_M)
        workspace_minimum = object_points.min(axis=0) - margin
        workspace_maximum = object_points.max(axis=0) + margin
        collision_mask = depth_valid_mask & np.all(
            (cloud >= workspace_minimum) & (cloud <= workspace_maximum),
            axis=-1,
        )
        collision_points = cloud[collision_mask].astype(
            np.float32, copy=False
        )
        collision_colors = normalized_color[collision_mask].astype(
            np.float32, copy=False
        )

        sample_indices = self._rng.choice(
            len(object_points),
            size=self.num_point,
            replace=len(object_points) < self.num_point,
        )
        sampled_points = object_points[sample_indices]
        end_points = {
            'point_clouds': self._torch.from_numpy(
                sampled_points[np.newaxis]
            ).to(self.device),
            'cloud_colors': object_colors[sample_indices],
        }
        return end_points, collision_points, collision_colors

    def predict(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
        mask: Optional[np.ndarray] = None,
        *,
        collision_detection: bool = True,
        nms: bool = True,
        sort_by_score: bool = True,
        max_grasps: Optional[int] = None,
    ) -> GraspPrediction:
        end_points, collision_points, collision_colors = self.prepare_input(
            color, depth, intrinsics, mask
        )
        with self._torch.inference_mode():
            network_output = self.net(end_points)
            prediction = self._pred_decode(network_output)[0]
        grasps = self._grasp_group_type(prediction.detach().cpu().numpy())

        if self.max_gripper_width is not None and len(grasps) > 0:
            grasps = grasps[grasps.widths <= self.max_gripper_width]

        if collision_detection and self.collision_thresh > 0 and len(grasps) > 0:
            from collision_detector import ModelFreeCollisionDetector

            detector = ModelFreeCollisionDetector(
                collision_points, voxel_size=self.voxel_size
            )
            collision_mask = detector.detect(
                grasps,
                approach_dist=self.approach_distance,
                collision_thresh=self.collision_thresh,
            )
            grasps = grasps[~collision_mask]
        if nms and len(grasps) > 0:
            grasps.nms()
        if sort_by_score and len(grasps) > 0:
            grasps.sort_by_score()
        if max_grasps is not None:
            if max_grasps < 0:
                raise ValueError('max_grasps must be non-negative or None')
            grasps = grasps[:max_grasps]
        return GraspPrediction(
            grasps, collision_points, collision_colors
        )
