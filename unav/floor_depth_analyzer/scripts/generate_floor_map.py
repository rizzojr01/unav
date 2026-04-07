#!/usr/bin/env python3
"""
Generate Floor Map Script

从floor点云生成2D floor map
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/unav/Desktop/unav")


def generate_floor_map_from_pointcloud(
    points,
    resolution=0.02,
    output_dir=None,
):
    """
    从floor点云生成2D floor map

    Args:
        points: floor点云 (N, 3)
        resolution: 地图分辨率(m/pixel)
        output_dir: 输出目录

    Returns:
        floor_map: 2D floor map
        map_info: 地图信息
    """
    print("="*80)
    print("从Floor点云生成2D Floor Map")
    print("="*80)
    print(f"\n配置:")
    print(f"  点云点数: {len(points):,}")
    print(f"  分辨率: {resolution} m/pixel")
    print()

    # 投影到XY平面
    x = points[:, 0]
    y = points[:, 1]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    print(f"点云范围:")
    print(f"  X: [{x_min:.2f}, {x_max:.2f}] ({x_max - x_min:.2f} m)")
    print(f"  Y: [{y_min:.2f}, {y_max:.2f}] ({y_max - y_min:.2f} m)")

    # 计算地图尺寸
    map_width = int((x_max - x_min) / resolution) + 1
    map_height = int((y_max - y_min) / resolution) + 1

    print(f"\n地图尺寸: {map_width} x {map_height} 像素")

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
    print(f"占据格子: {occupied_cells:,} ({occupied_cells/(map_width*map_height)*100:.1f}%)")

    # 地图信息
    map_info = {
        'resolution': resolution,
        'width': map_width,
        'height': map_height,
        'origin_x': x_min,
        'origin_y': y_min,
        'x_range': (x_min, x_max),
        'y_range': (y_min, y_max),
        'total_points': len(points),
        'occupied_cells': occupied_cells,
    }

    # 保存结果
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存PNG
        cv2.imwrite(str(output_path / "floor_map.png"), floor_map)
        print(f"\n✅ Floor map: {output_path / 'floor_map.png'}")

        # 保存数据
        np.save(output_path / "floor_map.npy", floor_map)
        np.save(output_path / "map_info.npy", map_info)

        # 创建可视化
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # 原始地图
        axes[0].imshow(floor_map, cmap='gray', origin='lower')
        axes[0].set_title('Floor Map', fontsize=14, fontweight='bold')
        axes[0].set_xlabel(f'X (pixels, {resolution}m/pixel)')
        axes[0].set_ylabel(f'Y (pixels, {resolution}m/pixel)')
        axes[0].grid(True, alpha=0.3)

        # 反转颜色
        floor_map_inv = 255 - floor_map
        axes[1].imshow(floor_map_inv, cmap='gray', origin='lower')
        axes[1].set_title('Floor Map (反转)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel(f'X (pixels, {resolution}m/pixel)')
        axes[1].set_ylabel(f'Y (pixels, {resolution}m/pixel)')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        vis_path = output_path / "floor_map_visualization.png"
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 可视化: {vis_path}")

    print("\n✅ Floor map生成完成")

    return floor_map, map_info


def main():
    parser = argparse.ArgumentParser(
        description='从Floor点云生成2D Floor Map',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/generate_floor_map.py \\
      --points_file /tmp/floor_output/floor_points.npy \\
      --output_dir /tmp/floor_map \\
      --resolution 0.02
        """
    )

    parser.add_argument('--points_file', type=str, required=True,
                        help='Floor点云文件(.npy)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--resolution', type=float, default=0.02,
                        help='地图分辨率(m/pixel)（默认0.02）')

    args = parser.parse_args()

    # 加载点云
    print("加载Floor点云...")
    points = np.load(args.points_file)
    print(f"✅ 加载了 {len(points):,} 个点")

    # 生成floor map
    floor_map, map_info = generate_floor_map_from_pointcloud(
        points=points,
        resolution=args.resolution,
        output_dir=args.output_dir,
    )

    print("\n" + "="*80)
    print("完成！")
    print("="*80)
    print(f"\n查看结果:")
    print(f"  eog {Path(args.output_dir) / 'floor_map_visualization.png'}")
    print()

    sys.exit(0)


if __name__ == "__main__":
    main()
