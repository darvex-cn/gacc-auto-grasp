import numpy as np
from geometry_msgs.msg import TransformStamped

'''
检查 4x4 齐次变换矩阵的有效性
@ transform: 4x4 齐次变换矩阵
@ maximum_translation_m: 平移向量的最大允许长度（米）
@return: None
@ raises ValueError: 如果矩阵无效或平移过大
'''
def validate_transform(
    transform: np.ndarray,
    maximum_translation_m: float = 2.0,
) -> None:
    if transform.shape != (4, 4):
        raise ValueError(f'transform must be 4x4, got {transform.shape}')
    if not np.all(np.isfinite(transform)):
        raise ValueError('transform contains non-finite values')
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError('transform has an invalid last row')

    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
        raise ValueError('transform rotation is not orthonormal')
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
        raise ValueError('transform rotation determinant is not 1')

    translation_norm = float(np.linalg.norm(transform[:3, 3]))
    if translation_norm > maximum_translation_m:
        raise ValueError(
            'transform translation is implausibly large: '
            f'{translation_norm:.3f} m (maximum {maximum_translation_m:.3f} m)'
        )


'''
将旋转矩阵和平移向量组合成 4x4 齐次变换矩阵
@param rotation: 3x3 旋转矩阵
@param translation: 3 元素平移向量
@return: 4x4 齐次变换矩阵
'''
def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """由旋转矩阵和平移向量组成 4x4 齐次变换矩阵."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


'''
计算 T_base_grasp = T_base_gripper * T_gripper_camera * T_camera_grasp
@param base_to_gripper: 4x4 齐次变换矩阵 T_base_gripper
@param gripper_to_camera: 4x4 齐次变换矩阵 T_gripper_camera
@param camera_to_grasp: 4x4 齐次变换矩阵 T_camera_grasp
@return: 4x4 齐次变换矩阵 T_base_grasp
'''
def transform_base_to_grasp(
    base_to_gripper: np.ndarray,
    gripper_to_camera: np.ndarray,
    camera_to_grasp: np.ndarray,
) -> np.ndarray:
    if base_to_gripper is None or gripper_to_camera is None or camera_to_grasp is None:
        return None
    """计算 T_base_grasp = T_base_gripper * T_gripper_camera * T_camera_grasp."""
    return (
        transform_base_to_camera(base_to_gripper, gripper_to_camera)
        @ np.asarray(camera_to_grasp, dtype=np.float64)
    )


'''
已知 T_base_gripper 和 T_gripper_camera，计算 T_base_camera
@param base_to_gripper: 4x4 齐次变换矩阵 T_base_gripper
@param gripper_to_camera: 4x4 齐次变换矩阵 T_gripper_camera
@return: 4x4 齐次变换矩阵 T_base_camera
'''
def transform_base_to_grasp_simple(
    base_to_camera: np.ndarray,
    camera_to_grasp: np.ndarray,
) -> np.ndarray:
    if base_to_camera is None or camera_to_grasp is None:
        return None
    """计算 T_base_grasp = T_base_camera * T_camera_grasp."""
    return (
        np.asarray(base_to_camera, dtype=np.float64)
        @ np.asarray(camera_to_grasp, dtype=np.float64)
    )


'''
计算 T_base_camera = T_base_gripper * T_gripper_camera
@param base_to_gripper: 4x4 齐次变换矩阵 T_base_gripper
@param gripper_to_camera: 4x4 齐次变换矩阵 T_gripper_camera
@return: 4x4 齐次变换矩阵 T_base_camera
'''
def transform_base_to_camera(
    base_to_gripper: np.ndarray,
    gripper_to_camera: np.ndarray,
) -> np.ndarray:
    if base_to_gripper is None or gripper_to_camera is None:
        return None
    """计算 T_base_camera = T_base_gripper * T_gripper_camera."""
    return (
        np.asarray(base_to_gripper, dtype=np.float64)
        @ np.asarray(gripper_to_camera, dtype=np.float64)
    )


'''
将 ROS 顺序四元数 (x, y, z, w) 转换成 3x3 旋转矩阵
@param x: 四元数的 x 分量
@param y: 四元数的 y 分量
@param z: 四元数的 z 分量
@param w: 四元数的 w 分量
@return: 3x3 旋转矩阵
'''
def quaternion_to_rotation_matrix(
    x: float, y: float, z: float, w: float
) -> np.ndarray:
    """把 ROS 顺序四元数 (x, y, z, w) 转换成 3x3 旋转矩阵."""
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        raise ValueError('orientation quaternion has zero length')
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


'''
将旋转矩阵转换为 ROS 顺序四元数 (x, y, z, w)
@param matrix: 3x3 旋转矩阵
@return: ROS 顺序四元数 (x, y, z, w)
'''
def rotation_matrix_to_quaternion(
    matrix: np.ndarray,
) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to ROS order (x, y, z, w)."""
    rotation = np.asarray(matrix, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f'rotation matrix must be 3x3, got {rotation.shape}')
    quaternion = np.empty(4, dtype=np.float64)
    trace = np.trace(rotation)
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion[3] = 0.25 * scale
        quaternion[0] = (rotation[2, 1] - rotation[1, 2]) / scale
        quaternion[1] = (rotation[0, 2] - rotation[2, 0]) / scale
        quaternion[2] = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        # 当旋转接近180度时，scale会接近于0,会造成计算错误，所以需要改成先算x、y、z中绝对值最大的一个计算
        index = int(np.argmax(np.diag(rotation)))
        next_index = (index + 1) % 3
        last_index = (index + 2) % 3
        scale = np.sqrt(
            1.0 + rotation[index, index]
            - rotation[next_index, next_index]
            - rotation[last_index, last_index]
        ) * 2.0
        quaternion[index] = 0.25 * scale
        quaternion[3] = (
            rotation[last_index, next_index] - rotation[next_index, last_index]
        ) / scale
        quaternion[next_index] = (
            rotation[next_index, index] + rotation[index, next_index]
        ) / scale
        quaternion[last_index] = (
            rotation[last_index, index] + rotation[index, last_index]
        ) / scale
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


'''
计算基座标下的 -Z 方向在相机坐标系中的表示
@param base_to_camera: 4x4 齐次变换矩阵 T_base_camera
@return: 3 元素向量，表示基座 -Z 方向在相机坐标系中的表示
'''
def base_down_direction_in_camera(
    t_base_to_camera: np.ndarray
) -> np.ndarray:
    """Return the base-frame -Z direction expressed in camera axes."""
    if t_base_to_camera is None:
        raise RuntimeError(
            'Cannot select a downward grasp without base_to_camera transform'
        )
    base_down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return t_base_to_camera[:3, :3].T @ base_down
