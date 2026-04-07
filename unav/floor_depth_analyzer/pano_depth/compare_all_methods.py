"""
Compare ALL depth estimation methods for floor plan reconstruction:
1. Ground Truth (Structured3D)
2. DA-UniFuse (Depth-Anywhere NeurIPS'24 with UniFuse backbone)
3. DA-BiFuseV2 (Depth-Anywhere with BiFuseV2 backbone)
4. EGFormer (ICCV'23)
5. DA² (arXiv'25, SOTA)
"""

import sys
sys.path.insert(0, "/home/unav/Desktop/unav")

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt


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
    colors = rgb.reshape(-1, 3) if rgb is not None and len(rgb.shape) == 3 else None

    valid = depth.flatten() > 0.01
    points = points[valid]
    if colors is not None:
        colors = colors[valid]

    return points, colors


def create_floor_plan(depth: np.ndarray, rgb: np.ndarray = None,
                      floor_height: float = -0.5, ceiling_height: float = 2.0,
                      resolution: float = 0.02, range_val: float = 8.0) -> np.ndarray:
    """Create top-down floor plan from depth map."""
    points, colors = equirectangular_to_3d(depth, rgb)
    mask = (points[:, 1] > floor_height) & (points[:, 1] < ceiling_height)
    points = points[mask]
    if colors is not None:
        colors = colors[mask]

    if len(points) == 0:
        return np.zeros((400, 400, 3), dtype=np.uint8)

    x, z = points[:, 0], points[:, 2]
    x_min, x_max = -range_val, range_val
    z_min, z_max = -range_val, range_val

    width = int((x_max - x_min) / resolution) + 1
    height = int((z_max - z_min) / resolution) + 1
    floor_plan = np.zeros((height, width, 3), dtype=np.uint8)

    valid = (x >= x_min) & (x <= x_max) & (z >= z_min) & (z <= z_max)
    x, z = x[valid], z[valid]
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


def scale_align_depth(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Scale align predicted depth to GT using median scaling."""
    valid = (gt > 0.01) & (pred > 0.01)
    if valid.sum() < 100:
        return pred
    scale = np.median(gt[valid]) / np.median(pred[valid])
    return pred * scale


def main():
    # Paths
    scene_dir = Path('/mnt/data/floorplan-reconstruction/public_data/stru3d/panorama/Structured3D/scene_00000/2D_rendering/485142/panorama/full')
    output_dir = Path('/home/unav/Desktop/unav/unav/tmp/depth_comparison')
    models_dir = output_dir / 'all_models'

    rgb_path = str(scene_dir / 'rgb_rawlight.png')
    gt_depth_path = str(scene_dir / 'depth.png')

    # Load RGB
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb_512 = cv2.resize(rgb, (1024, 512))

    # 1. GT depth
    print("Loading Ground Truth...")
    gt_depth = cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    gt_depth_512 = cv2.resize(gt_depth, (1024, 512), interpolation=cv2.INTER_NEAREST)

    # Load all model predictions
    methods = {}

    # DA-UniFuse (Depth-Anywhere with UniFuse backbone)
    print("Loading DA-UniFuse...")
    unifuse = np.load(models_dir / 'depth_unifuse.npy')
    if unifuse.shape != (512, 1024):
        unifuse = cv2.resize(unifuse, (1024, 512))
    unifuse = np.clip(unifuse, 0.01, 20)
    methods['DA-UniFuse'] = unifuse

    # DA-BiFuseV2 (Depth-Anywhere with BiFuseV2 backbone)
    print("Loading DA-BiFuseV2...")
    bifuse = np.load(models_dir / 'depth_bifusev2.npy')
    if bifuse.shape != (512, 1024):
        bifuse = cv2.resize(bifuse, (1024, 512))
    bifuse = np.clip(bifuse, 0.01, 20)
    methods['DA-BiFuseV2'] = bifuse

    # EGFormer
    print("Loading EGFormer...")
    egformer = np.load(models_dir / 'depth_egformer.npy')
    if egformer.shape != (512, 1024):
        egformer = cv2.resize(egformer, (1024, 512))
    egformer = np.clip(egformer, 0.01, 20)
    methods['EGFormer'] = egformer

    # DA3 (if available)
    da3_path = output_dir / 'pred_depth_da3.npy'
    if da3_path.exists():
        print("Loading DA3...")
        da3 = np.load(da3_path)
        if da3.shape != (512, 1024):
            da3 = cv2.resize(da3, (1024, 512))
        da3 = np.clip(da3, 0.01, 20)
        methods['DA3'] = da3

    # DA² (if available)
    da2_path = models_dir / 'depth_da2.npy'
    if da2_path.exists():
        print("Loading DA²...")
        da2 = np.load(da2_path)
        if da2.shape != (512, 1024):
            da2 = cv2.resize(da2, (1024, 512))
        # DA² outputs normalized depth, need scale alignment
        da2 = scale_align_depth(da2, gt_depth_512)
        da2 = np.clip(da2, 0.01, 20)
        methods['DA²'] = da2

    # Print raw depth ranges
    print("\n=== Raw Depth Ranges ===")
    print(f"GT:       {gt_depth_512.min():.2f} - {gt_depth_512.max():.2f} m")
    for name, depth in methods.items():
        print(f"{name:10s}: {depth.min():.2f} - {depth.max():.2f} m")

    # Calculate MAE
    valid_mask = gt_depth_512 > 0.01
    print("\n=== MAE (meters) ===")
    maes = {}
    for name, depth in methods.items():
        mae = np.abs(depth - gt_depth_512)[valid_mask].mean()
        maes[name] = mae
        print(f"{name:10s}: {mae:.3f} m")

    # Create floor plans
    print("\nCreating floor plans...")
    floor_plans = {'GT': create_floor_plan(gt_depth_512, rgb_512)}
    for name, depth in methods.items():
        floor_plans[name] = create_floor_plan(depth, rgb_512)

    # Create visualization
    n_methods = len(methods) + 1  # +1 for GT
    fig, axes = plt.subplots(3, n_methods, figsize=(4 * n_methods, 12))

    all_names = ['GT'] + list(methods.keys())
    all_depths = [gt_depth_512] + list(methods.values())
    all_maes = [0] + [maes[n] for n in methods.keys()]
    all_fps = [floor_plans['GT']] + [floor_plans[n] for n in methods.keys()]

    # Row 1: Depth maps
    for i, (name, depth, mae) in enumerate(zip(all_names, all_depths, all_maes)):
        im = axes[0, i].imshow(depth, cmap='turbo', vmin=0, vmax=8)
        title = f'{name}\n[{depth.min():.1f}-{depth.max():.1f}m]'
        if mae > 0:
            title += f'\nMAE: {mae:.2f}m'
        axes[0, i].set_title(title, fontsize=10)
        axes[0, i].axis('off')
    plt.colorbar(im, ax=axes[0, -1], fraction=0.046)

    # Row 2: Error maps
    for i, (name, depth) in enumerate(zip(all_names, all_depths)):
        if i == 0:  # GT - show RGB
            axes[1, i].imshow(rgb_512)
            axes[1, i].set_title('RGB Panorama', fontsize=10)
        else:
            error = np.abs(depth - gt_depth_512)
            error[~valid_mask] = 0
            im = axes[1, i].imshow(error, cmap='hot', vmin=0, vmax=2)
            axes[1, i].set_title(f'Error: {name}', fontsize=10)
        axes[1, i].axis('off')
    plt.colorbar(im, ax=axes[1, -1], fraction=0.046)

    # Row 3: Floor plans
    for i, (name, fp) in enumerate(zip(all_names, all_fps)):
        axes[2, i].imshow(fp)
        axes[2, i].set_title(f'{name} Floor Plan', fontsize=10)
        axes[2, i].axis('off')

    plt.suptitle('Panoramic Depth Estimation Methods Comparison\n(Structured3D scene_00000)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'all_methods_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved to {output_path}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<15} {'Depth Range':<20} {'MAE (m)':<10}")
    print("-" * 60)
    print(f"{'GT':<15} {f'{gt_depth_512.min():.1f} - {gt_depth_512.max():.1f}m':<20} {'-':<10}")
    for name in methods.keys():
        d = methods[name]
        print(f"{name:<15} {f'{d.min():.1f} - {d.max():.1f}m':<20} {maes[name]:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
