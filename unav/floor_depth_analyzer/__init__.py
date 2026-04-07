"""
Floor Depth Analyzer

使用Depth Anything V3和SAM3生成floor map的工具包

模块结构:
- modules/preprocessing: 数据预处理（图片加载、切片）
- modules/depth_anything_v3: DA3 3D重建（纯DA3功能）
- modules/sam3: SAM3 floor mask生成
- modules/pointcloud: 点云处理
- modules/floor_map: Floor map生成
- modules/visualization: 可视化工具
- utils: 工具函数（相机参数计算等）
- scripts: 可执行脚本（组装各模块）
"""

__version__ = "0.2.0"

# 导出主要功能
from .modules.preprocessing import load_keyframes_and_trajectory, slice_equirectangular_images
from .modules.depth_anything_v3 import load_da3_model, run_da3_inference
from .modules.sam3 import load_sam3_model, generate_floor_masks
from .modules.pointcloud import depth_to_pointcloud, depth_to_pointcloud_with_mask
from .modules.floor_map import generate_floor_map, save_floor_map
from .modules.visualization import visualize_glb
from .utils import compute_slice_camera_params

__all__ = [
    # Preprocessing
    'load_keyframes_and_trajectory',
    'slice_equirectangular_images',
    # DA3
    'load_da3_model',
    'run_da3_inference',
    # SAM3
    'load_sam3_model',
    'generate_floor_masks',
    # Point cloud
    'depth_to_pointcloud',
    'depth_to_pointcloud_with_mask',
    # Floor map
    'generate_floor_map',
    'save_floor_map',
    # Visualization
    'visualize_glb',
    # Utils
    'compute_slice_camera_params',
]
