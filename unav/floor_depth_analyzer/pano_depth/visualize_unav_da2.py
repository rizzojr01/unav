"""
Visualize DA² depth estimation results on UNav data.
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
                      floor_height: float = -1.0, ceiling_height: float = 2.5,
                      output_size: int = 800, padding: float = 0.1) -> np.ndarray:
    """Create top-down floor plan from depth map with auto-scaling."""
    points, colors = equirectangular_to_3d(depth, rgb)
    mask = (points[:, 1] > floor_height) & (points[:, 1] < ceiling_height)
    points = points[mask]
    if colors is not None:
        colors = colors[mask]

    if len(points) == 0:
        return np.zeros((output_size, output_size, 3), dtype=np.uint8)

    x, z = points[:, 0], points[:, 2]

    # Auto-scale based on actual data extent
    x_min, x_max = x.min(), x.max()
    z_min, z_max = z.min(), z.max()

    # Add padding
    x_range = x_max - x_min
    z_range = z_max - z_min
    x_min -= x_range * padding
    x_max += x_range * padding
    z_min -= z_range * padding
    z_max += z_range * padding

    # Make square and compute resolution
    max_range = max(x_max - x_min, z_max - z_min)
    x_center = (x_min + x_max) / 2
    z_center = (z_min + z_max) / 2
    x_min = x_center - max_range / 2
    x_max = x_center + max_range / 2
    z_min = z_center - max_range / 2
    z_max = z_center + max_range / 2

    resolution = max_range / output_size

    floor_plan = np.zeros((output_size, output_size, 3), dtype=np.uint8)

    px = ((x - x_min) / resolution).astype(int)
    pz = ((z_max - z) / resolution).astype(int)
    px = np.clip(px, 0, output_size - 1)
    pz = np.clip(pz, 0, output_size - 1)

    if colors is not None:
        for i in range(len(px)):
            floor_plan[pz[i], px[i]] = colors[i].astype(np.uint8)
    else:
        floor_plan[pz, px] = [255, 255, 255]

    return floor_plan


def main():
    # Paths
    da2_output_dir = Path('/home/unav/Desktop/unav/unav/tmp/pano_depth_methods/DA-2/output/infer_unav_20260204_214032')
    unav_images_dir = Path('/home/unav/Desktop/unav/unav/tmp/pano_depth_methods/DA-2/assets/unav_demos')
    output_dir = Path('/home/unav/Desktop/unav/unav/tmp/depth_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get image names
    image_names = sorted([f.stem for f in (da2_output_dir / 'depth').glob('*.npy')])
    print(f"Found {len(image_names)} images: {image_names}")

    # Create visualization
    n_images = len(image_names)
    fig, axes = plt.subplots(3, n_images, figsize=(5 * n_images, 12))

    for i, name in enumerate(image_names):
        # Load RGB
        rgb_path = unav_images_dir / f'{name}.png'
        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Load depth (DA² output is normalized, need to understand scale)
        depth_path = da2_output_dir / 'depth' / f'{name}.npy'
        depth = np.load(depth_path)
        print(f"{name}: depth shape={depth.shape}, range=[{depth.min():.3f}, {depth.max():.3f}]")

        # Resize RGB to match depth
        if rgb.shape[:2] != depth.shape[:2]:
            rgb_resized = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))
        else:
            rgb_resized = rgb

        # Create floor plan
        floor_plan = create_floor_plan(depth, rgb_resized)

        # Row 1: RGB
        axes[0, i].imshow(rgb)
        axes[0, i].set_title(f'{name}', fontsize=12)
        axes[0, i].axis('off')

        # Row 2: Depth map
        im = axes[1, i].imshow(depth, cmap='turbo')
        axes[1, i].set_title(f'Depth [{depth.min():.2f}-{depth.max():.2f}]', fontsize=10)
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i], fraction=0.046)

        # Row 3: Floor plan
        axes[2, i].imshow(floor_plan)
        axes[2, i].set_title('Floor Plan', fontsize=10)
        axes[2, i].axis('off')

    plt.suptitle('DA² Depth Estimation on UNav Data\n(LOH 9th Floor)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'unav_da2_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved visualization to {output_path}")

    # Also copy the original DA² visualization
    import shutil
    vis_all_path = da2_output_dir / 'vis_all.png'
    if vis_all_path.exists():
        shutil.copy(vis_all_path, output_dir / 'unav_da2_vis_all.png')
        print(f"Copied DA² visualization to {output_dir / 'unav_da2_vis_all.png'}")


if __name__ == "__main__":
    main()
