#!/usr/bin/env python3
"""
Visualize GLB Script

可视化GLB 3D点云文件
"""

import sys
import argparse

sys.path.insert(0, "/home/unav/Desktop/unav")

from unav.floor_depth_analyzer.modules.visualization import visualize_glb


def main():
    parser = argparse.ArgumentParser(description='可视化GLB 3D点云')
    parser.add_argument('--glb', type=str,
                        default="/tmp/da3_3d_reconstruction/scene.glb",
                        help='GLB文件路径')
    parser.add_argument('--output', type=str,
                        default=None,
                        help='输出目录（默认与GLB同目录）')

    args = parser.parse_args()

    visualize_glb(args.glb, args.output)

    sys.exit(0)


if __name__ == "__main__":
    main()
