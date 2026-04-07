"""
Compare GT, DA_Retrained, and DA_Original (paper's pre-trained model).
NOTE: Use compare_four_methods.py for more comprehensive comparison including DA3.
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/unav/Desktop/unav/unav')
from pano_to_3d import equirectangular_to_3d


def create_floor_plan(depth: np.ndarray, rgb: np.ndarray = None,
                      floor_height: float = -0.5,
                      ceiling_height: float = 2.0,
                      resolution: float = 0.02) -> np.ndarray:
    """Create top-down floor plan from depth map."""
    points, colors, _ = equirectangular_to_3d(depth, rgb)

    mask = (points[:, 1] > floor_height) & (points[:, 1] < ceiling_height)
    points = points[mask]
    if colors is not None:
        colors = colors[mask]

    if len(points) == 0:
        return np.zeros((200, 200, 3), dtype=np.uint8)

    x = points[:, 0]
    z = points[:, 2]

    # Fixed range for consistent comparison
    range_val = 8
    x_min, x_max = -range_val, range_val
    z_min, z_max = -range_val, range_val

    width = int((x_max - x_min) / resolution) + 1
    height = int((z_max - z_min) / resolution) + 1

    floor_plan = np.zeros((height, width, 3), dtype=np.uint8)

    valid = (x >= x_min) & (x <= x_max) & (z >= z_min) & (z <= z_max)
    x = x[valid]
    z = z[valid]
    if colors is not None:
        colors = colors[valid]

    px = ((x - x_min) / resolution).astype(int)
    pz = ((z_max - z) / resolution).astype(int)

    px = np.clip(px, 0, width - 1)
    pz = np.clip(pz, 0, height - 1)

    if colors is not None:
        for i in range(len(px)):
            floor_plan[pz[i], px[i]] = colors[i].astype(np.uint8)
    else:
        floor_plan[pz, px] = [255, 255, 255]

    return floor_plan


def main():
    scene_dir = Path('/mnt/data/floorplan-reconstruction/public_data/stru3d/panorama/Structured3D/scene_00000/2D_rendering/485142/panorama/full')
    output_dir = Path('/home/unav/Desktop/unav/unav/tmp/depth_comparison')

    rgb_path = str(scene_dir / 'rgb_rawlight.png')
    gt_depth_path = str(scene_dir / 'depth.png')

    # Load RGB
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb_display = cv2.resize(rgb, (1024, 512))

    # Load GT depth
    gt_depth = cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED)
    gt_depth = gt_depth.astype(np.float32) / 1000.0

    # Load predictions
    pred_ours = np.load(output_dir / 'pred_depth.npy')
    pred_paper = np.load(output_dir / 'pred_depth_paper.npy')

    # Resize GT to match predictions
    if gt_depth.shape != pred_ours.shape:
        gt_depth = cv2.resize(gt_depth, (pred_ours.shape[1], pred_ours.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    print(f"GT depth range: {gt_depth.min():.2f} - {gt_depth.max():.2f} m")
    print(f"DA_Retrained depth range: {pred_ours.min():.2f} - {pred_ours.max():.2f} m")
    print(f"DA_Original depth range: {pred_paper.min():.2f} - {pred_paper.max():.2f} m")

    # Calculate errors
    valid_mask = gt_depth > 0
    mae_ours = np.abs(pred_ours - gt_depth)[valid_mask].mean()
    mae_paper = np.abs(pred_paper - gt_depth)[valid_mask].mean()

    print(f"MAE (DA_Retrained): {mae_ours:.2f} m")
    print(f"MAE (DA_Original): {mae_paper:.2f} m")

    # Create floor plans
    print("Creating floor plans...")
    fp_gt = create_floor_plan(gt_depth, rgb_display)
    fp_ours = create_floor_plan(pred_ours, rgb_display)
    fp_paper = create_floor_plan(pred_paper, rgb_display)

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # Row 1: Depth maps
    axes[0, 0].imshow(gt_depth, cmap='turbo', vmin=0, vmax=8)
    axes[0, 0].set_title(f'Ground Truth\n[{gt_depth.min():.1f}-{gt_depth.max():.1f}m]', fontsize=11)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(pred_ours, cmap='turbo', vmin=0, vmax=8)
    axes[0, 1].set_title(f'DA_Retrained (st3d_sf3d)\n[{pred_ours.min():.1f}-{pred_ours.max():.1f}m]', fontsize=11)
    axes[0, 1].axis('off')

    im = axes[0, 2].imshow(pred_paper, cmap='turbo', vmin=0, vmax=8)
    axes[0, 2].set_title(f'DA_Original (SpatialAudioGen)\n[{pred_paper.min():.1f}-{pred_paper.max():.1f}m]', fontsize=11)
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    # Row 2: Error maps
    error_ours = np.abs(pred_ours - gt_depth)
    error_ours[~valid_mask] = 0
    error_paper = np.abs(pred_paper - gt_depth)
    error_paper[~valid_mask] = 0

    axes[1, 0].imshow(rgb_display)
    axes[1, 0].set_title('RGB Panorama', fontsize=11)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(error_ours, cmap='hot', vmin=0, vmax=2)
    axes[1, 1].set_title(f'Error (DA_Retrained)\nMAE: {mae_ours:.2f}m', fontsize=11)
    axes[1, 1].axis('off')

    im2 = axes[1, 2].imshow(error_paper, cmap='hot', vmin=0, vmax=2)
    axes[1, 2].set_title(f'Error (DA_Original)\nMAE: {mae_paper:.2f}m', fontsize=11)
    axes[1, 2].axis('off')
    plt.colorbar(im2, ax=axes[1, 2], fraction=0.046)

    # Row 3: Floor plans
    axes[2, 0].imshow(fp_gt)
    axes[2, 0].set_title('GT Floor Plan', fontsize=11)
    axes[2, 0].axis('off')

    axes[2, 1].imshow(fp_ours)
    axes[2, 1].set_title('DA_Retrained Floor Plan', fontsize=11)
    axes[2, 1].axis('off')

    axes[2, 2].imshow(fp_paper)
    axes[2, 2].set_title('DA_Original Floor Plan', fontsize=11)
    axes[2, 2].axis('off')

    plt.suptitle('Depth Estimation Comparison: GT vs DA_Retrained vs DA_Original', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'three_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved to {output_dir / 'three_model_comparison.png'}")


if __name__ == "__main__":
    main()
