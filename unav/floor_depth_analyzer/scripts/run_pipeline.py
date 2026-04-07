#!/usr/bin/env python3
"""
Complete Pipeline Script

完整的floor map生成pipeline:
1. run_reconstruction.py - 生成floor点云
2. generate_floor_map.py - 生成2D floor map
"""

import sys
import argparse
from pathlib import Path
import subprocess


def run_complete_pipeline(
    keyframe_dir: str,
    trajectory_file: str,
    output_dir: str,
    num_images: int = 10,
    resolution: float = 0.02,
    conf_thresh: float = 1.5,
):
    """
    运行完整的floor map生成pipeline

    Args:
        keyframe_dir: 关键帧目录
        trajectory_file: 轨迹文件
        output_dir: 输出目录
        num_images: 处理的图片数量
        resolution: 地图分辨率(m/pixel)
        conf_thresh: 置信度阈值
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Floor Map生成 - 完整Pipeline")
    print("="*80)
    print("\n配置:")
    print(f"  关键帧目录: {keyframe_dir}")
    print(f"  轨迹文件: {trajectory_file}")
    print(f"  输出目录: {output_dir}")
    print(f"  图片数量: {num_images}")
    print(f"  地图分辨率: {resolution} m/pixel")
    print()

    # Step 1: Floor重建（DA3深度 + SAM3 floor mask）
    print("="*80)
    print("Step 1: Floor重建（DA3深度 + SAM3 floor mask）")
    print("="*80)

    cmd1 = [
        "python", "unav/floor_depth_analyzer/scripts/run_reconstruction.py",
        "--keyframe_dir", keyframe_dir,
        "--trajectory_file", trajectory_file,
        "--output_dir", output_dir,
        "--num_images", str(num_images),
        "--conf_thresh", str(conf_thresh),
    ]

    print(f"\n运行命令: {' '.join(cmd1)}\n")
    result1 = subprocess.run(cmd1, cwd="/home/unav/Desktop/unav")

    if result1.returncode != 0:
        print("\n[错误] Floor重建失败")
        return False

    print("\n[完成] Floor重建完成")

    # Step 2: 生成Floor Map
    print("\n" + "="*80)
    print("Step 2: 生成Floor Map")
    print("="*80)

    floor_map_dir = output_path / "floor_map"
    floor_points_file = output_path / "floor_points.npy"

    cmd2 = [
        "python", "unav/floor_depth_analyzer/scripts/generate_floor_map.py",
        "--points_file", str(floor_points_file),
        "--output_dir", str(floor_map_dir),
        "--resolution", str(resolution),
    ]

    print(f"\n运行命令: {' '.join(cmd2)}\n")
    result2 = subprocess.run(cmd2, cwd="/home/unav/Desktop/unav")

    if result2.returncode != 0:
        print("\n[错误] Floor Map生成失败")
        return False

    print("\n[完成] Floor Map生成完成")

    # 总结
    print("\n" + "="*80)
    print("Pipeline完成！")
    print("="*80)

    print("\n查看结果:")
    print(f"  Floor点云: {output_path / 'floor_pointcloud.glb'}")
    print(f"  Floor Map: {floor_map_dir / 'floor_map.png'}")
    print(f"  可视化: {floor_map_dir / 'floor_map_visualization.png'}")
    print()
    print("可视化命令:")
    print(f"  eog {floor_map_dir / 'floor_map_visualization.png'}")
    print()
    print("在线查看3D模型:")
    print("  https://3dviewer.net/")
    print(f"  上传: {output_path / 'floor_pointcloud.glb'}")
    print()

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Floor Map生成 - 完整Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/run_pipeline.py \\
      --keyframe_dir /path/to/keyframes \\
      --trajectory_file /path/to/trajectory.txt \\
      --output_dir /tmp/floor_output \\
      --num_images 10
        """
    )

    parser.add_argument('--keyframe_dir', type=str, required=True,
                        help='关键帧目录')
    parser.add_argument('--trajectory_file', type=str, required=True,
                        help='相机轨迹文件')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--num_images', type=int, default=10,
                        help='处理的图片数量（默认10）')
    parser.add_argument('--resolution', type=float, default=0.02,
                        help='地图分辨率(m/pixel)（默认0.02）')
    parser.add_argument('--conf_thresh', type=float, default=1.5,
                        help='置信度阈值（默认1.5）')

    args = parser.parse_args()

    success = run_complete_pipeline(
        keyframe_dir=args.keyframe_dir,
        trajectory_file=args.trajectory_file,
        output_dir=args.output_dir,
        num_images=args.num_images,
        resolution=args.resolution,
        conf_thresh=args.conf_thresh,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
