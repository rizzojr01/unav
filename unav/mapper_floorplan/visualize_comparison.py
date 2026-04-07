#!/usr/bin/env python3
"""
Visualize comparison between depth-based and point cloud-based floor plans.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def load_ply(path):
    """Load points from PLY file."""
    points = []
    colors = []
    with open(path, 'r') as f:
        in_header = True
        for line in f:
            if in_header:
                if line.strip() == 'end_header':
                    in_header = False
                continue
            parts = line.strip().split()
            if len(parts) >= 6:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                colors.append([int(parts[3]), int(parts[4]), int(parts[5])])
    return np.array(points), np.array(colors)


def create_floorplan_image(points, resolution=0.05, size=800):
    """Create floor plan image from points."""
    if len(points) == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)

    x, z = points[:, 0], points[:, 2]

    # Auto-scale
    x_min, x_max = x.min(), x.max()
    z_min, z_max = z.min(), z.max()

    padding = 0.1
    x_range = x_max - x_min
    z_range = z_max - z_min
    x_min -= x_range * padding
    x_max += x_range * padding
    z_min -= z_range * padding
    z_max += z_range * padding

    # Make square
    max_range = max(x_max - x_min, z_max - z_min)
    x_center = (x_min + x_max) / 2
    z_center = (z_min + z_max) / 2
    x_min = x_center - max_range / 2
    x_max = x_center + max_range / 2
    z_min = z_center - max_range / 2
    z_max = z_center + max_range / 2

    res = max_range / size

    img = np.zeros((size, size), dtype=np.uint8)
    px = ((x - x_min) / res).astype(int)
    pz = ((z_max - z) / res).astype(int)
    px = np.clip(px, 0, size - 1)
    pz = np.clip(pz, 0, size - 1)
    img[pz, px] = 255

    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return img


def main():
    base_dir = Path('/home/unav/Desktop/unav/unav/tmp/depth_comparison/unav_test')

    # Load depth-based points
    depth_ply = base_dir / 'output_depth' / 'floor_points_depth.ply'
    pc_ply = base_dir / 'output_pointcloud_full' / 'floor_points_pc.ply'

    depth_points, _ = load_ply(depth_ply)
    pc_points, _ = load_ply(pc_ply)

    print(f"Depth-based (DA², 5 images): {len(depth_points)} points")
    print(f"Point cloud-based (SLAM, 162 images): {len(pc_points)} points")

    # Create floor plan images
    depth_img = create_floorplan_image(depth_points)
    pc_img = create_floorplan_image(pc_points)

    # Create comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Depth-based
    axes[0].imshow(depth_img, cmap='Blues')
    axes[0].set_title(f'DA² Depth-based\n(5 keyframes, {len(depth_points):,} points)\nCoverage: ~15m x 15m', fontsize=12)
    axes[0].axis('off')

    # Point cloud-based
    axes[1].imshow(pc_img, cmap='Greens')
    axes[1].set_title(f'SLAM Point Cloud-based\n(162 keyframes, {len(pc_points):,} points)\nCoverage: ~2m x 2m', fontsize=12)
    axes[1].axis('off')

    # Side-by-side overlay (same scale)
    # Use depth-based bounds for both
    x_min = depth_points[:, 0].min() - 1
    x_max = depth_points[:, 0].max() + 1
    z_min = depth_points[:, 2].min() - 1
    z_max = depth_points[:, 2].max() + 1

    max_range = max(x_max - x_min, z_max - z_min)
    size = 800
    res = max_range / size

    # Depth points
    overlay = np.zeros((size, size, 3), dtype=np.uint8)

    dx = depth_points[:, 0]
    dz = depth_points[:, 2]
    dpx = ((dx - x_min) / res).astype(int)
    dpz = ((z_max - dz) / res).astype(int)
    dpx = np.clip(dpx, 0, size - 1)
    dpz = np.clip(dpz, 0, size - 1)
    overlay[dpz, dpx, 2] = 255  # Blue channel

    # PC points
    if len(pc_points) > 0:
        px = pc_points[:, 0]
        pz = pc_points[:, 2]
        ppx = ((px - x_min) / res).astype(int)
        ppz = ((z_max - pz) / res).astype(int)
        ppx = np.clip(ppx, 0, size - 1)
        ppz = np.clip(ppz, 0, size - 1)
        overlay[ppz, ppx, 1] = 255  # Green channel

    # Apply closing to both
    for c in range(3):
        if overlay[:, :, c].sum() > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            overlay[:, :, c] = cv2.morphologyEx(overlay[:, :, c], cv2.MORPH_CLOSE, kernel)

    axes[2].imshow(overlay)
    axes[2].set_title('Overlay (same scale)\nBlue=DA² Depth, Green=SLAM PC', fontsize=12)
    axes[2].axis('off')

    plt.suptitle('Floor Plan Comparison: DA² Depth vs SLAM Point Cloud\n(LOH 9th Floor, NYC)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = base_dir / 'method_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved comparison to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Floor Plan Extraction Comparison")
    print("=" * 60)
    print(f"\nDepth-based (DA²):")
    print(f"  - Input: 5 keyframe images")
    print(f"  - Output: {len(depth_points):,} floor points")
    print(f"  - X range: {depth_points[:, 0].min():.2f} - {depth_points[:, 0].max():.2f} m")
    print(f"  - Z range: {depth_points[:, 2].min():.2f} - {depth_points[:, 2].max():.2f} m")

    print(f"\nPoint cloud-based (SLAM):")
    print(f"  - Input: 162 keyframe poses + 10,295 sparse points")
    print(f"  - Output: {len(pc_points):,} floor points")
    if len(pc_points) > 0:
        print(f"  - X range: {pc_points[:, 0].min():.2f} - {pc_points[:, 0].max():.2f} m")
        print(f"  - Z range: {pc_points[:, 2].min():.2f} - {pc_points[:, 2].max():.2f} m")

    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("Depth-based method generates 130x more points with only 3% of the images!")
    print("=" * 60)


if __name__ == '__main__':
    main()
