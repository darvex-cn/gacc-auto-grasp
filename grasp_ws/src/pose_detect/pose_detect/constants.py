import numpy as np

# 单位缩放因子，用于把 PoseStamped 中的平移量转换为米。
POSE_UNITS = {
    'm': 1.0,
    'cm': 0.01,
    'mm': 0.001,
}


# graspnet的旋转矩阵与夹爪的旋转映射
R_GRASP_GRIPPER = np.array([
    [ 0.0, 0.0, 1.0],
    [ 0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
])


POSE_DETECT_NODE_PARAMETERS = {
    'general.hand_eye_calibration_path',
    'general.robot_pose_topic',
    'general.pose_units',
    'general.base_frame',
    'graspnet.checkpoint_path',
    'graspnet.device',
    'graspnet.num_point',
    'graspnet.num_view',
    'graspnet.collision_thresh',
    'graspnet.voxel_size',
    'graspnet.max_gripper_width',
    'graspnet.max_grasps',
    'graspnet.max_approach_angle_deg',
    'object_detect.object_model_path',
    'object_detect.target_class',
    'object_detect.selection_policy',
    'object_detect.visual_enable',
    'object_detect.confidence_threshold',
    'object_detect.iou_threshold',
    'object_detect.mask_threshold',
    'object_detect.image_size',
    'object_detect.inference_rate',
    'object_detect.device',
    'object_detect.quantize',
    'object_detect.retina_masks',
    'camera.rgb_topic',
    'camera.depth_topic',
    'camera.camera_info_topic',
    'camera.depth_scale',
    'camera.max_image_age',
    'camera.max_image_time_difference',
    'camera.camera_frame',
    'rviz.grasp_visual_enable',
    'rviz.rviz_cloud_max_points',
    'rviz.gripper_line_width',
}

SEGMENTATION_TEST_CONFIG_PARAMETERS = {
    'object_model_path',
    'target_class',
    'selection_policy',
    'visual_enable',
    'confidence_threshold',
    'iou_threshold',
    'mask_threshold',
    'image_size',
    'device',
    'quantize',
    'retina_masks',
    'rgb_topic',
    'output_topic',
    'inference_rate',
}


