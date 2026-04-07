#!/usr/bin/env python3
"""
Compare floor plan extraction methods:
1. Point cloud-based (original SLAM dense points)
2. Depth-based (DA² depth + camera poses)
"""

import sys
sys.path.insert(0, "/home/unav/Desktop/unav/unav/mapper_floorplan")

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def load_ply_points(ply_path):
    """Load points from PLY file."""
    points = []
    with open(ply_path, 'r') as f:
        in_header = True
        vertex_count = 0
        for line in f:
            if in_header:
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.strip() == 'end_header':
                    in_header = False
            else:
                parts = line.strip().split()
                if len(parts) >= 3:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    points.append([x, y, z])
    return np.array(points)


def create_topdown_image(points, resolution=0.05, padding=0.5):
    """Create top-down image from points."""
    if len(points) == 0:
        return np.zeros((100, 100), dtype=np.uint8)

    x = points[:, 0]
    z = points[:, 2]

    x_min, x_max = x.min() - padding, x.max() + padding
    z_min, z_max = z.min() - padding, z.max() + padding

    width = int((x_max - x_min) / resolution)
    height = int((z_max - z_min) / resolution)

    img = np.zeros((height, width), dtype=np.uint8)

    px = ((x - x_min) / resolution).astype(int)
    pz = ((z_max - z) / resolution).astype(int)

    px = np.clip(px, 0, width - 1)
    pz = np.clip(pz, 0, height - 1)

    img[pz, px] = 255

    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return img, (x_min, x_max, z_min, z_max)


def main():
    output_dir = Path('/home/unav/Desktop/unav/unav/tmp/depth_comparison/unav_test')

    # Load depth-based points
    depth_ply = output_dir / 'output_depth' / 'floor_points_depth.ply'
    pc_ply = output_dir / 'output_pointcloud' / 'floor_points_pc.ply'

    if not depth_ply.exists():
        print(f"Depth PLY not found: {depth_ply}")
        return

    if not pc_ply.exists():
        print(f"Point cloud PLY not found: {pc_ply}")
        return

    print("Loading point clouds...")
    depth_points = load_ply_points(depth_ply)
    pc_points = load_ply_points(pc_ply)

    print(f"  Depth-based points: {len(depth_points)}")
    print(f"  Point cloud-based points: {len(pc_points)}")

    # Create top-down images with same bounds
    all_points = np.vstack([depth_points, pc_points]) if len(pc_points) > 0 else depth_points
    x_min, x_max = all_points[:, 0].min() - 0.5, all_points[:, 0].max() + 0.5
    z_min, z_max = all_points[:, 2].min() - 0.5, all_points[:, 2].max() + 0.5

    resolution = 0.05
    width = int((x_max - x_min) / resolution)
    height = int((z_max - z_min) / resolution)

    # Create depth-based image
    depth_img = np.zeros((height, width), dtype=np.uint8)
    if len(depth_points) > 0:
        px = ((depth_points[:, 0] - x_min) / resolution).astype(int)
        pz = ((z_max - depth_points[:, 2]) / resolution).astype(int)
        px = np.clip(px, 0, width - 1)
        pz = np.clip(pz, 0, height - 1)
        depth_img[pz, px] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        depth_img = cv2.morphologyEx(depth_img, cv2.MORPH_CLOSE, kernel)

    # Create point cloud-based image
    pc_img = np.zeros((height, width), dtype=np.uint8)
    if len(pc_points) > 0:
        px = ((pc_points[:, 0] - x_min) / resolution).astype(int)
        pz = ((z_max - pc_points[:, 2]) / resolution).astype(int)
        px = np.clip(px, 0, width - 1)
        pz = np.clip(pz, 0, height - 1)
        pc_img[pz, px] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        pc_img = cv2.morphologyEx(pc_img, cv2.MORPH_CLOSE, kernel)

    # Create comparison visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(depth_img, cmap='gray')
    axes[0].set_title(f'DA² Depth-based\n({len(depth_points)} points)', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(pc_img, cmap='gray')
    axes[1].set_title(f'SLAM Point Cloud-based\n({len(pc_points)} points)', fontsize=12)
    axes[1].axis('off')

    # Overlay comparison
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    overlay[:, :, 0] = depth_img  # Red = depth-based
    overlay[:, :, 1] = pc_img      # Green = point cloud-based
    # Yellow = overlap

    axes[2].imshow(overlay)
    axes[2].set_title('Overlay\n(Red=Depth, Green=PC, Yellow=Both)', fontsize=12)
    axes[2].axis('off')

    plt.suptitle('Floor Plan Extraction: DA² Depth vs SLAM Point Cloud', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'floorplan_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved comparison to {output_path}")

    # Print statistics
    print("\n" + "=" * 50)
    print("COMPARISON STATISTICS")
    print("=" * 50)
    print(f"Depth-based (DA²):")
    print(f"  - Total points: {len(depth_points)}")
    if len(depth_points) > 0:
        print(f"  - X range: {depth_points[:, 0].min():.2f} - {depth_points[:, 0].max():.2f} m")
        print(f"  - Z range: {depth_points[:, 2].min():.2f} - {depth_points[:, 2].max():.2f} m")
        print(f"  - Coverage: {depth_img.sum() / 255:.0f} grid cells")

    print(f"\nPoint cloud-based (SLAM):")
    print(f"  - Total points: {len(pc_points)}")
    if len(pc_points) > 0:
        print(f"  - X range: {pc_points[:, 0].min():.2f} - {pc_points[:, 0].max():.2f} m")
        print(f"  - Z range: {pc_points[:, 2].min():.2f} - {pc_points[:, 2].max():.2f} m")
        print(f"  - Coverage: {pc_img.sum() / 255:.0f} grid cells")
    print("=" * 50)


if __name__ == '__main__':
    main()
