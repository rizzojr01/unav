# Floor Depth Analyzer

基于DA3深度估计和SAM3语义分割的Floor点云重建和2D Floor Map生成。

## 快速开始（推荐）

使用 UNav 统一接口运行：

```bash
# 单个场景
python -m unav.run_floor_map \
    /mnt/data/UNav-IO/temp \
    /mnt/data/UNav-IO/data \
    New_York_City \
    LightHouse \
    3_floor \
    10

# 批量处理（编辑 run_floor_map.sh 配置后运行）
./run_floor_map.sh
```

**输入：** SLAM 输出的关键帧和轨迹（`stella_vslam_dense/keyframes` 和 `eval_logs/keyframe_trajectory.txt`）

**输出：**
- `floor_map/floor_pointcloud.glb` - Floor点云
- `floor_map/floor_map/floor_map.png` - 2D Floor Map

## 流程概述

```
全景图 (equirectangular)
    ↓ 切片 (多个yaw和pitch角度)
透视图切片
    ↓ DA3深度推理
深度图 + 置信度
    ↓ SAM3 "floor" mask推理
每张切片的floor mask
    ↓ 用mask过滤深度
只保留floor区域的深度
    ↓ 生成3D点云
Floor点云 (GLB)
    ↓ 投影到2D
Floor Map (占用地图)
```

## 快速开始

### 完整Pipeline（推荐）

一键运行完整流程：

```bash
python unav/floor_depth_analyzer/scripts/run_pipeline.py \
    --keyframe_dir /path/to/keyframes \
    --trajectory_file /path/to/trajectory.txt \
    --output_dir /tmp/floor_output \
    --num_images 10
```

### 分步运行

#### 1. Floor重建（DA3深度 + SAM3 floor mask）

```bash
python unav/floor_depth_analyzer/scripts/run_reconstruction.py \
    --keyframe_dir /path/to/keyframes \
    --trajectory_file /path/to/trajectory.txt \
    --output_dir /tmp/floor_output \
    --num_images 10
```

**输出：**
- `floor_pointcloud.glb` - Floor点云（只包含floor区域）
- `floor_points.npy` - Floor点云数据
- `floor_masks.npy` - SAM3生成的floor mask
- `all_depths.npy` - 深度数据
- `slices/` - 切片图像

#### 2. 生成2D Floor Map

```bash
python unav/floor_depth_analyzer/scripts/generate_floor_map.py \
    --points_file /tmp/floor_output/floor_points.npy \
    --output_dir /tmp/floor_output/floor_map \
    --resolution 0.02
```

**输出：**
- `floor_map.png` - 占用地图
- `floor_map_visualization.png` - 可视化

### 3. 可视化GLB（可选）

```bash
python unav/floor_depth_analyzer/scripts/visualize_glb.py \
    --glb /tmp/floor_output/floor_pointcloud.glb
```

## 目录结构

```
floor_depth_analyzer/
├── README.md
├── __init__.py
├── modules/              # 核心模块
│   ├── preprocessing/    # 数据加载、切片
│   ├── depth_anything_v3/# DA3模型
│   ├── sam3/             # SAM3 floor mask
│   ├── pointcloud/       # 点云处理
│   ├── floor_map/        # Floor map生成
│   └── visualization/    # 可视化
├── scripts/              # 入口脚本
│   ├── run_reconstruction.py  # Floor重建
│   ├── generate_floor_map.py  # 生成Floor Map
│   ├── run_pipeline.py        # 完整Pipeline
│   └── visualize_glb.py       # GLB可视化
├── pano_depth/           # 全景深度估计研究 (Depth-Anywhere)
│   ├── pano_to_3d.py     # 全景深度→3D点云
│   ├── compare_*.py      # GT对比脚本
│   └── README.md         # 详细文档
└── utils/                # 工具函数
```

## 参数说明

### run_reconstruction.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--keyframe_dir` | 必需 | 关键帧目录 |
| `--trajectory_file` | 必需 | 相机轨迹文件 |
| `--output_dir` | 必需 | 输出目录 |
| `--num_images` | 10 | 处理的全景图数量 |
| `--yaw_angles` | 0,45,90...315 | Yaw角度列表 |
| `--pitch_angles` | 0,-20 | Pitch角度列表 |
| `--fov` | 90 | 视场角 |
| `--conf_thresh` | 1.5 | 深度置信度阈值 |

### generate_floor_map.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--points_file` | 必需 | Floor点云文件(.npy) |
| `--output_dir` | 必需 | 输出目录 |
| `--resolution` | 0.02 | 地图分辨率(m/pixel) |

## 技术细节

### 切片策略

- **Yaw角度**: [0, 45, 90, 135, 180, 225, 270, 315]
- **Pitch角度**: [0, -20]（向下看以获取更多地面）
- **FOV**: 90度
- **每张全景图**: 16个切片

### Floor检测

使用SAM3的文本提示"floor"来检测每张切片中的地面区域，只保留floor区域的深度用于生成点云。

### 深度过滤

- 使用DA3的置信度过滤低质量深度
- 使用SAM3的floor mask只保留地面区域

## 依赖

- Python 3.10+
- PyTorch
- Depth Anything V3 (DA3)
- SAM3
- trimesh
- opencv-python
- numpy
- matplotlib
- tqdm
