#!/usr/bin/env python3
"""
Floor Map Generation

从3D点云生成2D floor map
"""

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def generate_floor_map(
    depths,
    confs,
    slice_paths,
    extrinsics,
    intrinsics,
    floor_masks=None,
    resolution=0.02,
    conf_thresh=1.5,
    z_low=10,
    z_high=30,
):
    """
    从深度数据生成floor map

    Args:
        depths: 深度数组 (N, H, W)
        confs: 置信度数组 (N, H, W)
        slice_paths: 切片路径列表
        extrinsics: 相机外参 (N, 4, 4)
        intrinsics: 相机内参 (N, 3, 3)
        floor_masks: Floor mask字典（可选）
        resolution: 地图分辨率(m/pixel)
        conf_thresh: 置信度阈值
        z_low: Z值下百分位
        z_high: Z值上百分位

    Returns:
        floor_map: 2D floor map
        map_info: 地图信息字典
    """
    from ..pointcloud.processor import depth_to_pointcloud, depth_to_pointcloud_with_mask

    print("="*80)
    print("生成Floor Map")
    print("="*80)
    print(f"\n配置:")
    print(f"  分辨率: {resolution} m/pixel")
    print(f"  置信度阈值: {conf_thresh}")
    print(f"  Z值过滤: {z_low}% - {z_high}%")
    print(f"  使用floor mask: {'是' if floor_masks else '否'}")
    print()

    # Step 1: 收集所有点云
    print("Step 1: 生成点云...")
    all_points = []

    for i in tqdm(range(len(depths)), desc="处理深度图"):
        depth = depths[i]
        conf = confs[i]
        extrinsic = extrinsics[i]
        intrinsic = intrinsics[i]

        # 如果有floor mask，使用它
        if floor_masks is not None:
            slice_name = Path(slice_paths[i]).name
            if slice_name in floor_masks:
                floor_mask = floor_masks[slice_name]
                points = depth_to_pointcloud_with_mask(
                    depth, conf, floor_mask, intrinsic, extrinsic, conf_thresh
                )
            else:
                points = depth_to_pointcloud(
                    depth, conf, intrinsic, extrinsic, conf_thresh
                )
        else:
            points = depth_to_pointcloud(
                depth, conf, intrinsic, extrinsic, conf_thresh
            )

        if len(points) > 0:
            all_points.append(points)

    if len(all_points) == 0:
        raise ValueError("没有生成任何点云！")

    all_points = np.vstack(all_points)
    print(f"\n✅ 总点数: {len(all_points):,}")

    # Step 2: Z值过滤
    print(f"\nStep 2: Z值过滤 ({z_low}% - {z_high}%)...")
    z_values = all_points[:, 2]
    z_min = np.percentile(z_values, z_low)
    z_max = np.percentile(z_values, z_high)

    print(f"  Z范围: [{z_values.min():.2f}, {z_values.max():.2f}]")
    print(f"  过滤范围: [{z_min:.2f}, {z_max:.2f}]")

    mask = (z_values >= z_min) & (z_values <= z_max)
    filtered_points = all_points[mask]

    print(f"  过滤后点数: {len(filtered_points):,} ({len(filtered_points)/len(all_points)*100:.1f}%)")

    # Step 3: 生成2D floor map
    print("\nStep 3: 生成2D floor map...")

    x = filtered_points[:, 0]
    y = filtered_points[:, 1]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    print(f"  X范围: [{x_min:.2f}, {x_max:.2f}]")
    print(f"  Y范围: [{y_min:.2f}, {y_max:.2f}]")

    # 计算地图尺寸
    map_width = int((x_max - x_min) / resolution) + 1
    map_height = int((y_max - y_min) / resolution) + 1

    print(f"  地图尺寸: {map_width} x {map_height}")

    # 创建occupancy grid
    floor_map = np.zeros((map_height, map_width), dtype=np.uint8)

    # 将点投影到2D网格
    grid_x = ((x - x_min) / resolution).astype(int)
    grid_y = ((y - y_min) / resolution).astype(int)

    # 限制在地图范围内
    grid_x = np.clip(grid_x, 0, map_width - 1)
    grid_y = np.clip(grid_y, 0, map_height - 1)

    # 标记占据的格子
    floor_map[grid_y, grid_x] = 255

    occupied_cells = np.sum(floor_map > 0)
    print(f"  占据格子数: {occupied_cells:,} ({occupied_cells/(map_width*map_height)*100:.1f}%)")

    # 地图信息
    map_info = {
        'resolution': resolution,
        'width': map_width,
        'height': map_height,
        'origin_x': x_min,
        'origin_y': y_min,
        'x_range': (x_min, x_max),
        'y_range': (y_min, y_max),
        'total_points': len(all_points),
        'filtered_points': len(filtered_points),
        'occupied_cells': occupied_cells,
    }

    print("\n✅ Floor map生成完成")

    return floor_map, map_info


def save_floor_map(floor_map, map_info, output_dir):
    """
    保存floor map和可视化

    Args:
        floor_map: 2D floor map
        map_info: 地图信息
        output_dir: 输出目录

    Returns:
        output_files: 保存的文件路径字典
    """
    import matplotlib.pyplot as plt

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n保存floor map...")

    # 保存原始数据
    map_file = output_path / "floor_map.npy"
    np.save(map_file, floor_map)
    print(f"  ✅ 地图数据: {map_file}")

    # 保存地图信息
    info_file = output_path / "map_info.npy"
    np.save(info_file, map_info)
    print(f"  ✅ 地图信息: {info_file}")

    # 保存PNG
    png_file = output_path / "floor_map.png"
    cv2.imwrite(str(png_file), floor_map)
    print(f"  ✅ PNG图像: {png_file}")

    # 创建可视化
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 原始地图
    axes[0].imshow(floor_map, cmap='gray', origin='lower')
    axes[0].set_title('Floor Map (原始)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel(f'X (pixels, {map_info["resolution"]}m/pixel)')
    axes[0].set_ylabel(f'Y (pixels, {map_info["resolution"]}m/pixel)')
    axes[0].grid(True, alpha=0.3)

    # 反转颜色（黑色=占据，白色=空闲）
    floor_map_inv = 255 - floor_map
    axes[1].imshow(floor_map_inv, cmap='gray', origin='lower')
    axes[1].set_title('Floor Map (反转)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel(f'X (pixels, {map_info["resolution"]}m/pixel)')
    axes[1].set_ylabel(f'Y (pixels, {map_info["resolution"]}m/pixel)')
    axes[1].grid(True, alpha=0.3)

    # 添加统计信息
    stats_text = f"""地图统计:
尺寸: {map_info['width']} x {map_info['height']}
分辨率: {map_info['resolution']} m/pixel
X范围: [{map_info['x_range'][0]:.2f}, {map_info['x_range'][1]:.2f}] m
Y范围: [{map_info['y_range'][0]:.2f}, {map_info['y_range'][1]:.2f}] m
总点数: {map_info['total_points']:,}
过滤后: {map_info['filtered_points']:,}
占据格子: {map_info['occupied_cells']:,}"""

    plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.1, 1, 1])

    vis_file = output_path / "floor_map_visualization.png"
    plt.savefig(vis_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 可视化: {vis_file}")

    return {
        'map_file': map_file,
        'info_file': info_file,
        'png_file': png_file,
        'vis_file': vis_file,
    }
