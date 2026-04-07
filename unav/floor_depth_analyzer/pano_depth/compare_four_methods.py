"""
Compare 4 depth estimation methods for floor plan reconstruction:
1. Ground Truth (Structured3D)
2. DA_Retrained (Depth-Anywhere UniFuse retrained on st3d+sf3d)
3. DA_Original (Depth-Anywhere paper checkpoint: SpatialAudioGen)
4. DA3 (Depth Anything V3 via cubemap stitching)
"""

import sys
sys.path.insert(0, "/home/unav/Desktop/unav")

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import torch


def equirectangular_to_3d(depth: np.ndarray, rgb: np.ndarray = None):
    """Convert equirectangular depth to 3D points."""
    H, W = depth.shape[:2]

    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)

    theta = (u / W - 0.5) * 2 * np.pi
    phi = (0.5 - v / H) * np.pi

    x = depth * np.cos(phi) * np.sin(theta)
    y = depth * np.sin(phi)
    z = -depth * np.cos(phi) * np.cos(theta)

    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    colors = None
    if rgb is not None:
        if len(rgb.shape) == 3:
            colors = rgb.reshape(-1, 3)

    valid = depth.flatten() > 0.01
    points = points[valid]
    if colors is not None:
        colors = colors[valid]

    return points, colors


def create_floor_plan(depth: np.ndarray, rgb: np.ndarray = None,
                      floor_height: float = -0.5,
                      ceiling_height: float = 2.0,
                      resolution: float = 0.02,
                      range_val: float = 8.0) -> np.ndarray:
    """Create top-down floor plan from depth map."""
    points, colors = equirectangular_to_3d(depth, rgb)

    mask = (points[:, 1] > floor_height) & (points[:, 1] < ceiling_height)
    points = points[mask]
    if colors is not None:
        colors = colors[mask]

    if len(points) == 0:
        return np.zeros((400, 400, 3), dtype=np.uint8)

    x = points[:, 0]
    z = points[:, 2]

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


def scale_align_depth(pred_depth: np.ndarray, gt_depth: np.ndarray) -> np.ndarray:
    """Scale align predicted depth to ground truth using median scaling."""
    valid_mask = (gt_depth > 0.01) & (pred_depth > 0.01)
    if valid_mask.sum() < 100:
        return pred_depth

    scale = np.median(gt_depth[valid_mask]) / np.median(pred_depth[valid_mask])
    return pred_depth * scale


def run_da3_inference(rgb_path: str, output_size: tuple = (512, 1024)):
    """Run DA3 via cubemap stitching."""
    from unav.floor_depth_analyzer.scripts.equirect_depth_via_cube_dav3 import (
        load_da3_model, equirect_to_cube_depth_dav3
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_da3_model(device)

    equirect_depth, _, _, _ = equirect_to_cube_depth_dav3(
        rgb_path, model, cube_size=512, device=device
    )

    # Resize to match other methods
    if equirect_depth.shape != output_size:
        equirect_depth = cv2.resize(equirect_depth, (output_size[1], output_size[0]))

    return equirect_depth


def main():
    # Paths
    scene_dir = Path('/mnt/data/floorplan-reconstruction/public_data/stru3d/panorama/Structured3D/scene_00000/2D_rendering/485142/panorama/full')
    output_dir = Path('/home/unav/Desktop/unav/unav/tmp/depth_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = str(scene_dir / 'rgb_rawlight.png')
    gt_depth_path = str(scene_dir / 'depth.png')

    # Load RGB
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb_512 = cv2.resize(rgb, (1024, 512))

    # 1. Load GT depth
    print("Loading Ground Truth depth...")
    gt_depth = cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED)
    gt_depth = gt_depth.astype(np.float32) / 1000.0  # mm to m
    gt_depth_512 = cv2.resize(gt_depth, (1024, 512), interpolation=cv2.INTER_NEAREST)
    print(f"  GT range: {gt_depth_512.min():.2f} - {gt_depth_512.max():.2f} m")

    # 2. Load Depth-Anywhere (retrained on st3d+sf3d)
    print("Loading Depth-Anywhere (retrained)...")
    pred_retrained = np.load(output_dir / 'pred_depth.npy')
    print(f"  DA_Retrained range: {pred_retrained.min():.2f} - {pred_retrained.max():.2f} m")

    # 3. Load Depth-Anywhere (original paper checkpoint)
    print("Loading Depth-Anywhere (original)...")
    pred_original = np.load(output_dir / 'pred_depth_paper.npy')
    print(f"  DA_Original range: {pred_original.min():.2f} - {pred_original.max():.2f} m")

    # 4. Run DA3 inference
    print("Running DA3 inference (cubemap stitching)...")
    pred_da3_raw = run_da3_inference(rgb_path, output_size=(512, 1024))
    print(f"  DA3 raw range: {pred_da3_raw.min():.4f} - {pred_da3_raw.max():.4f} (normalized)")

    # Scale align DA3 to GT
    pred_da3 = scale_align_depth(pred_da3_raw, gt_depth_512)
    print(f"  DA3 aligned range: {pred_da3.min():.2f} - {pred_da3.max():.2f} m")

    # Calculate MAE for each method
    valid_mask = gt_depth_512 > 0.01
    mae_retrained = np.abs(pred_retrained - gt_depth_512)[valid_mask].mean()
    mae_original = np.abs(pred_original - gt_depth_512)[valid_mask].mean()
    mae_da3 = np.abs(pred_da3 - gt_depth_512)[valid_mask].mean()

    print(f"\nMAE comparison:")
    print(f"  DA_Retrained (st3d_sf3d):    {mae_retrained:.3f} m")
    print(f"  DA_Original (SpatialAudioGen): {mae_original:.3f} m")
    print(f"  DA3 (cubemap stitch):          {mae_da3:.3f} m")

    # Create floor plans
    print("\nCreating floor plans...")
    fp_gt = create_floor_plan(gt_depth_512, rgb_512)
    fp_retrained = create_floor_plan(pred_retrained, rgb_512)
    fp_original = create_floor_plan(pred_original, rgb_512)
    fp_da3 = create_floor_plan(pred_da3, rgb_512)

    # Create visualization
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    methods = ['Ground Truth', 'DA_Retrained (st3d_sf3d)', 'DA_Original (SpatialAudioGen)', 'DA3 (cubemap)']
    depths = [gt_depth_512, pred_retrained, pred_original, pred_da3]
    floor_plans = [fp_gt, fp_retrained, fp_original, fp_da3]
    maes = [0, mae_retrained, mae_original, mae_da3]

    # Row 1: Depth maps
    for i, (method, depth, mae) in enumerate(zip(methods, depths, maes)):
        im = axes[0, i].imshow(depth, cmap='turbo', vmin=0, vmax=8)
        title = f'{method}\n[{depth.min():.1f}-{depth.max():.1f}m]'
        if mae > 0:
            title += f'\nMAE: {mae:.2f}m'
        axes[0, i].set_title(title, fontsize=10)
        axes[0, i].axis('off')
    plt.colorbar(im, ax=axes[0, -1], fraction=0.046)

    # Row 2: Error maps
    axes[1, 0].imshow(rgb_512)
    axes[1, 0].set_title('RGB Panorama', fontsize=10)
    axes[1, 0].axis('off')

    for i, (method, depth) in enumerate(zip(methods[1:], depths[1:]), 1):
        error = np.abs(depth - gt_depth_512)
        error[~valid_mask] = 0
        im = axes[1, i].imshow(error, cmap='hot', vmin=0, vmax=2)
        axes[1, i].set_title(f'Error: {method.split()[0]}', fontsize=10)
        axes[1, i].axis('off')
    plt.colorbar(im, ax=axes[1, -1], fraction=0.046)

    # Row 3: Floor plans
    for i, (method, fp) in enumerate(zip(methods, floor_plans)):
        axes[2, i].imshow(fp)
        axes[2, i].set_title(f'{method.split()[0]} Floor Plan', fontsize=10)
        axes[2, i].axis('off')

    plt.suptitle('Depth Estimation Comparison: GT vs DA_Retrained vs DA_Original vs DA3\n(Structured3D scene_00000)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'four_methods_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved to {output_path}")

    # Save DA3 depth for future use
    np.save(output_dir / 'pred_depth_da3.npy', pred_da3)
    print(f"Saved DA3 depth to {output_dir / 'pred_depth_da3.npy'}")


if __name__ == "__main__":
    main()
