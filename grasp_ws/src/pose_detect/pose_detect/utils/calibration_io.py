import numpy as np
import yaml
from pathlib import Path
from .transforms import validate_transform

'''
读取并验证手眼标定 YAML 文件中的 T_gripper_camera
@param yaml_path: 手眼标定 YAML 文件路径
@param maximum_translation_m: 平移向量的最大允许长度（米）
@return: 4x4 齐次变换矩阵 T_gripper_camera
@ raises FileNotFoundError: 如果 YAML 文件不存在
@ raises ValueError: 如果矩阵无效或平移过大
@ return: 4x4 齐次变换矩阵 T_gripper_camera
'''
def load_hand_eye_transform(
    yaml_path: str | Path,
    *,
    maximum_translation_m: float = 2.0,
) -> np.ndarray:
    """从标定 YAML 读取并校验 T_gripper_camera."""
    path = Path(yaml_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'hand-eye calibration file not found: {path}')
    with path.open(encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or 'transform_matrix' not in data:
        raise ValueError('calibration YAML has no transform_matrix')

    t_gripper_to_camera = np.asarray(data['transform_matrix'], dtype=np.float64)
    validate_transform(t_gripper_to_camera, maximum_translation_m)

    return t_gripper_to_camera

