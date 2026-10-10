<div align="center">

<h1>GACC Auto Grasp</h1>

<p>
基于 ROS 2 Humble 的视觉抓取工程<br>
集成 YOLO 实例分割、GraspNet 抓取位姿估计与 Eye-in-Hand 手眼标定<br>
将抓取位姿从相机坐标系转换到机器人基座坐标系
</p>

<p>
<a href="#1-下载工程与初始化子模块">下载与初始化</a> ·
<a href="#2-新建-conda-环境部署依赖与编译工程">环境部署与编译</a> ·
<a href="#3-运行与使用">运行与使用</a>
</p>

<p>
<img src="https://img.shields.io/badge/Ubuntu-22.04-E95420?logo=ubuntu&logoColor=white" alt="Ubuntu 22.04">
<img src="https://img.shields.io/badge/ROS_2-Humble-22314E?logo=ros&logoColor=white" alt="ROS 2 Humble">
<img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white" alt="Python 3.10">
<a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
<a href="https://github.com/darvex-cn/gacc-auto-grasp/stargazers"><img src="https://img.shields.io/github/stars/darvex-cn/gacc-auto-grasp?style=flat" alt="GitHub Stars"></a>
</p>

</div>

当前感知流程为：相机发布对齐后的 RGB-D 数据，YOLO 持续分割目标；调用 `/detect_grasp` 服务时，GraspNet 根据最新分割结果生成抓取候选，并结合手眼标定结果和机械臂末端位姿，将最佳抓取位姿转换到机器人基座坐标系。

`arm_control` 和 `grasp_coordinator` 当前只有接口与配置，尚未接入具体机械臂 SDK，也没有可执行节点。

## 1. 下载工程与初始化子模块

### 1.1 克隆工程

工程使用 Git submodule 管理 `graspnet-baseline` 和 `graspnetAPI`：

~~~bash
git clone --recurse-submodules https://github.com/darvex-cn/gacc-auto-grasp.git
cd gacc-auto-grasp
~~~

后续命令除特别说明外，均在项目根目录执行。

如果克隆时没有使用 `--recurse-submodules`，执行：

~~~bash
git submodule update --init --recursive
~~~

拉取项目更新后，同步子模块：

~~~bash
git pull
git submodule update --init --recursive
~~~

### 1.2 项目结构

~~~text
gacc-auto-grasp/
├── requirements.txt
├── grasp_ws/
│   └── src/
│       ├── grasp_interfaces/       # ROS 2 srv/action 接口
│       ├── pose_detect/            # YOLO、GraspNet 与坐标转换
│       ├── hand_eye_calibration/   # ChArUco 手眼标定
│       ├── arm_control/            # 机械臂控制接口边界
│       ├── grasp_coordinator/      # 抓取任务编排接口边界
│       └── grasp_bringup/          # launch 与系统集成
├── python_libs/
│   └── graspnet_adapter/
├── third_party/
│   ├── graspnet-baseline/          # Git submodule
│   └── graspnetAPI/                # Git submodule
├── models/                         # 本地模型，不提交到 Git
├── datasets/                       # 本地数据集，不提交到 Git
└── training/yolo/                  # YOLO 数据转换与训练脚本
~~~

## 2. 新建 Conda 环境、部署依赖与编译工程

推荐使用 Ubuntu 22.04、ROS 2 Humble、Python 3.10，以及支持当前 PyTorch 版本的 NVIDIA 驱动和 CUDA 环境。

### 2.1 创建 Conda 环境

~~~bash
conda create -n graspnet python=3.10 -y
conda activate graspnet
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
~~~

`requirements.txt` 会安装 PyTorch、Ultralytics、Open3D、`graspnetAPI`、`graspnet_adapter` 等 Python 依赖。

### 2.2 编译 GraspNet CUDA 扩展

PointNet++ 和 KNN 的构建脚本需要使用当前 Conda 环境中已经安装的 PyTorch，因此必须关闭 pip 的隔离构建：

~~~bash
pip install --no-build-isolation -v ./third_party/graspnet-baseline/pointnet2
pip install --no-build-isolation -v ./third_party/graspnet-baseline/knn
~~~

如需加快编译速度，可以提前执行 `pip install ninja`。

### 2.3 安装 ROS 2 依赖并编译

~~~bash
source /opt/ros/humble/setup.bash
conda activate graspnet
rosdep install --from-paths grasp_ws/src --ignore-src -r -y

cd grasp_ws
colcon build
source install/setup.bash
cd ..
~~~

`pose_detect/setup.py` 会显式指定 Conda 环境中的 Python。如果 Conda 安装位置或环境名称不同，需要将其中的 `build_scripts.executable` 改为当前环境的 Python 路径。

修改 `grasp_ws/src/pose_detect/config/` 下的配置后，需要重新构建：

~~~bash
cd grasp_ws
colcon build --packages-select pose_detect
source install/setup.bash
cd ..
~~~

### 2.4 准备模型与配置路径

模型文件不会随 Git 仓库下载，运行完整流程前至少需要准备：

~~~text
models/graspnet/checkpoint-rs.tar
models/yolo/trained/motor_parts_yolo26s_seg/best.pt
~~~

`grasp_ws/src/pose_detect/config/config.yaml` 中的文件路径均相对于项目根目录：

~~~yaml
pose_detect_node:
  general:
    hand_eye_calibration_path: grasp_ws/src/pose_detect/config/hand_eye_calibration.yaml
  graspnet:
    checkpoint_path: models/graspnet/checkpoint-rs.tar
  object_detect:
    object_model_path: models/yolo/trained/motor_parts_yolo26s_seg/best.pt

segmentation_test_node:
  object_model_path: models/yolo/trained/motor_parts_yolo26s_seg/best.pt
~~~

### 2.5 新终端加载环境

在项目根目录执行：

~~~bash
source /opt/ros/humble/setup.bash
conda activate graspnet
source grasp_ws/install/setup.bash
~~~

## 3. 运行与使用

### 3.1 输入数据

完整感知流程默认使用以下输入：

| 数据 | 类型 | 默认话题 |
| --- | --- | --- |
| RGB 图像 | `sensor_msgs/msg/Image` | `/camera/color/image_raw` |
| 深度图像 | `sensor_msgs/msg/Image` | `/camera/depth/image_raw` |
| 深度相机内参 | `sensor_msgs/msg/CameraInfo` | `/camera/depth/camera_info` |
| 机械臂末端位姿 | `geometry_msgs/msg/PoseStamped` | `/rokae_driver7/tool_pose` |

机械臂位姿必须表示 `T_base_gripper`，位置单位与 `general.pose_units` 一致，`header.frame_id` 与 `general.base_frame` 一致。

### 3.2 手眼标定

默认采用 eye-in-hand：相机固定在夹爪上，ChArUco 板固定在工作台。

启动标定节点：

~~~bash
ros2 run hand_eye_calibration calibration_node --ros-args \
  -p image_topic:=/camera/color/image_raw \
  -p camera_info_topic:=/camera/color/camera_info \
  -p robot_pose_topic:=/rokae_driver7/tool_pose \
  -p base_frame:=base_link \
  -p gripper_frame:=tool0 \
  -p camera_frame:=camera_color_optical_frame \
  -p pose_units:=m \
  -p output_file:="$(pwd)/grasp_ws/src/pose_detect/config/hand_eye_calibration.yaml"
~~~

查看标定板检测图像：

~~~bash
ros2 run rqt_image_view rqt_image_view /charuco_debug_image
~~~

机械臂每次移动到新的稳定姿态后采样：

~~~bash
ros2 service call /capture_sample std_srvs/srv/Trigger '{}'
~~~

建议采集 20～30 组不同位置和旋转姿态，然后求解并保存：

~~~bash
ros2 service call /calibrate std_srvs/srv/Trigger '{}'
ros2 service call /save std_srvs/srv/Trigger '{}'
~~~

保存标定文件后，重新构建 `pose_detect`，使安装空间中的配置同步更新。

### 3.3 YOLO 实例分割训练

训练流程为：Labelme 标注、转换 YOLO segmentation 数据集、训练模型、部署 `best.pt`。

#### 3.3.1 准备数据

将图片和同名 Labelme JSON 放入：

~~~text
datasets/motor_parts/
├── motor_parts_001.png
├── motor_parts_001.json
├── motor_parts_002.png
└── motor_parts_002.json
~~~

标注支持 `polygon` 和 `rectangle`，标签名称必须存在于 `training/yolo/config.yaml` 的 `conversion.class_names` 中。

#### 3.3.2 转换数据集

主要配置：

~~~yaml
conversion:
  source_directory: datasets/motor_parts
  output_directory: datasets/motor_parts_yolo
  class_names:
    - motor_part_type1
    - motor_part_type2
  train_ratio: 0.9
  val_ratio: 0.1
  test_ratio: 0
  random_seed: 42
  overwrite: false
~~~

输出目录已经存在并确认可以覆盖时，将 `overwrite` 临时设为 `true`。执行：

~~~bash
python training/yolo/convert_labelme_to_yolo_seg.py
~~~

#### 3.3.3 训练模型

主要配置：

~~~yaml
training:
  model_path: models/yolo/pretrained/yolo26s-seg.pt
  data_yaml: datasets/motor_parts_yolo/data.yaml
  epochs: 200
  image_size: 640
  batch_size: 16
  device: cuda:0
  workers: 8
  patience: 40
  resume: false
  project_directory: training/yolo/runs/yolo
  run_name: motor_parts_yolo26s_seg
~~~

执行训练：

~~~bash
python training/yolo/train_yolo26_seg.py
~~~

训练权重默认保存在：

~~~text
training/yolo/runs/yolo/motor_parts_yolo26s_seg/weights/
├── best.pt
└── last.pt
~~~

断点续训时，将 `model_path` 指向 `last.pt`，并将 `resume` 设为 `true`。

#### 3.3.4 部署模型

~~~bash
mkdir -p models/yolo/trained/motor_parts_yolo26s_seg
cp training/yolo/runs/yolo/motor_parts_yolo26s_seg/weights/best.pt \
  models/yolo/trained/motor_parts_yolo26s_seg/best.pt
~~~

确认 `config.yaml` 中的 `object_model_path` 指向该文件。只替换同一路径的权重时无需重新构建，但必须重启节点。

### 3.4 单独运行 YOLO 分割

启动相机后运行：

~~~bash
ros2 run pose_detect segmentation_test_node
~~~

默认输出话题为 `/motor_part/segmentation_image`。该节点只运行 YOLO，不执行 GraspNet。

### 3.5 运行 Pose Detect

先启动相机和机械臂位姿发布节点，再启动感知：

~~~bash
ros2 launch grasp_bringup perception.launch.py
~~~

等待节点获得分割结果后，触发一次 GraspNet 推理：

~~~bash
ros2 service call /detect_grasp grasp_interfaces/srv/DetectGrasp '{}'
~~~

服务返回：

- `success` 和 `message`
- 基座坐标系下的 `base_to_gripper_pose`
- GraspNet 抓取分数 `score`
- 夹爪宽度 `width`

坐标变换关系：

~~~text
T_base_camera = T_base_gripper × T_gripper_camera
T_base_gripper_target = T_base_camera × T_camera_gripper_target
~~~

### 3.6 RViz2 可视化

在 `grasp_ws/src/pose_detect/config/config.yaml` 中启用：

~~~yaml
pose_detect_node:
  object_detect:
    visual_enable: true
  rviz:
    grasp_visual_enable: true
~~~

将 RViz2 的 Fixed Frame 设置为 `camera.camera_frame`，然后添加：

- Image：`/pose_visual/segmentation_image`
- MarkerArray：`/pose_visual/grasp_candidates`
- PointCloud2：`/pose_visual/graspnet_scene_cloud`

抓取候选和场景点云仅在 `/detect_grasp` 推理成功后更新。

### 3.7 当前限制与安全说明

- 相机驱动必须先完成彩色图与深度图对齐。
- 机械臂位姿与 RGB-D 当前没有严格时间同步。
- `arm_control` 和 `grasp_coordinator` 尚未实现，相关 Action 只有接口定义。
- TCP 偏置、夹爪坐标轴、工作空间和碰撞边界必须结合真实硬件验证。
- 完成坐标系、限位、急停和低速测试前，不要把视觉输出直接下发给机械臂。
