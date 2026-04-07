"""
Compare different depth estimation methods by their top-down floor plan reconstruction.
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from pano_to_3d import equirectangular_to_3d


def create_floor_plan(depth: np.ndarray, rgb: np.ndarray = None,
                      floor_height: float = -0.5,
                      ceiling_height: float = 2.0,
                      resolution: float = 0.02) -> np.ndarray:
    """
    Create top-down floor plan from depth map.

    Args:
        depth: (H, W) equirectangular depth map
        rgb: (H, W, 3) optional RGB for coloring
        floor_height: min Y to include
        ceiling_height: max Y to include
        resolution: meters per pixel

    Returns:
        floor_plan: (H, W, 3) top-down view image
    """
    # Convert to 3D points
    points, colors, _ = equirectangular_to_3d(depth, rgb)

    # Filter by height (Y axis)
    mask = (points[:, 1] > floor_height) & (points[:, 1] < ceiling_height)
    points = points[mask]
    if colors is not None:
        colors = colors[mask]

    if len(points) == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    # Project to XZ plane (top-down)
    x = points[:, 0]
    z = points[:, 2]

    # Determine bounds with some padding
    x_min, x_max = x.min() - 0.5, x.max() + 0.5
    z_min, z_max = z.min() - 0.5, z.max() + 0.5

    # Make it square-ish for better comparison
    x_range = x_max - x_min
    z_range = z_max - z_min
    max_range = max(x_range, z_range)

    x_center = (x_min + x_max) / 2
    z_center = (z_min + z_max) / 2

    x_min = x_center - max_range / 2
    x_max = x_center + max_range / 2
    z_min = z_center - max_range / 2
    z_max = z_center + max_range / 2

    width = int((x_max - x_min) / resolution) + 1
    height = int((z_max - z_min) / resolution) + 1

    # Create image
    floor_plan = np.zeros((height, width, 3), dtype=np.uint8)

    # Convert to pixel coordinates
    px = ((x - x_min) / resolution).astype(int)
    pz = ((z_max - z) / resolution).astype(int)

    # Clip to bounds
    px = np.clip(px, 0, width - 1)
    pz = np.clip(pz, 0, height - 1)

    if colors is not None:
        for i in range(len(px)):
            floor_plan[pz[i], px[i]] = colors[i].astype(np.uint8)
    else:
        floor_plan[pz, px] = [255, 255, 255]

    return floor_plan


def compare_depth_methods(rgb_path: str, depth_paths: dict, output_path: str,
                          gt_depth_path: str = None):
    """
    Compare multiple depth estimation methods.

    Args:
        rgb_path: path to RGB panorama
        depth_paths: dict of {method_name: depth_path}
        output_path: output comparison image path
        gt_depth_path: optional ground truth depth path
    """
    # Load RGB
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # Resize RGB to match depth maps
    rgb = cv2.resize(rgb, (1024, 512))

    n_methods = len(depth_paths)
    if gt_depth_path:
        n_methods += 1

    # Create figure
    fig, axes = plt.subplots(2, n_methods + 1, figsize=(4 * (n_methods + 1), 8))

    # Show RGB
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('RGB Panorama', fontsize=12)
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')

    col = 1

    # Ground truth if available
    if gt_depth_path:
        gt_depth = np.load(gt_depth_path) if gt_depth_path.endswith('.npy') else \
                   cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000

        # Depth visualization
        axes[0, col].imshow(gt_depth, cmap='turbo')
        axes[0, col].set_title('Ground Truth Depth', fontsize=12)
        axes[0, col].axis('off')

        # Floor plan
        gt_floor_plan = create_floor_plan(gt_depth, rgb)
        axes[1, col].imshow(gt_floor_plan)
        axes[1, col].set_title('GT Floor Plan', fontsize=12)
        axes[1, col].axis('off')

        col += 1

    # Each method
    for method_name, depth_path in depth_paths.items():
        # Load depth
        if depth_path.endswith('.npy'):
            depth = np.load(depth_path)
        else:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000

        # Resize if needed
        if depth.shape != (512, 1024):
            depth = cv2.resize(depth, (1024, 512), interpolation=cv2.INTER_NEAREST)

        # Depth visualization
        axes[0, col].imshow(depth, cmap='turbo', vmin=0, vmax=10)
        axes[0, col].set_title(f'{method_name}\n[{depth.min():.1f}-{depth.max():.1f}m]', fontsize=10)
        axes[0, col].axis('off')

        # Floor plan
        floor_plan = create_floor_plan(depth, rgb)
        axes[1, col].imshow(floor_plan)
        axes[1, col].set_title(f'{method_name} Floor Plan', fontsize=10)
        axes[1, col].axis('off')

        col += 1

    plt.suptitle('Depth Estimation Methods Comparison (Top: Depth, Bottom: Floor Plan)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved comparison to {output_path}")


def main():
    # Paths
    rgb_path = "/home/unav/Desktop/unav/unav/floor_depth_analyzer/output/image0_rgb.png"
    output_dir = Path("/home/unav/Desktop/unav/unav/tmp/depth_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Different depth estimation methods
    depth_paths = {
        "DA_Independent": "/home/unav/Desktop/unav/unav/floor_depth_analyzer/output/equi_depth_independent.npy",
        "DA_Multiview": "/home/unav/Desktop/unav/unav/floor_depth_analyzer/output/equi_depth_multiview.npy",
        "Depth_Anywhere": "/home/unav/Desktop/unav/unav/floor_depth_analyzer/output/image0_depth.npy",
    }

    # Run comparison
    compare_depth_methods(
        rgb_path=rgb_path,
        depth_paths=depth_paths,
        output_path=str(output_dir / "depth_methods_comparison.png")
    )


if __name__ == "__main__":
    main()
