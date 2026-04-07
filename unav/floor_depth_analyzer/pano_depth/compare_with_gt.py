"""
Compare depth estimation with ground truth using Structured3D data.
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
    range_val = 8  # meters
    x_min, x_max = -range_val, range_val
    z_min, z_max = -range_val, range_val

    width = int((x_max - x_min) / resolution) + 1
    height = int((z_max - z_min) / resolution) + 1

    floor_plan = np.zeros((height, width, 3), dtype=np.uint8)

    # Filter points within range
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


def run_depth_anywhere(rgb_path: str) -> np.ndarray:
    """Run Depth-Anywhere model on RGB panorama."""
    import torch
    sys.path.insert(0, '/home/unav/Desktop/unav/unav/tmp/Depth-Anywhere')

    from baseline_models.UniFuse.networks.unifuse import UniFuse
    from utils.Projection import py360_E2C
    from torchvision import transforms

    # Fix numpy compatibility
    np.bool = np.bool_
    np.float = np.float32

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UniFuse(
        num_layers=18,
        equi_h=512,
        equi_w=1024,
        pretrained=False,
        max_depth=10.0,
        fusion_type='cee',
        se_in_fusion=True
    )

    # Load checkpoint
    ckpt_path = '/home/unav/Desktop/unav/unav/tmp/Depth-Anywhere/ckpts/UniFuse/unifuse_st3d_sf3d/ckpt_100.pth'
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model = model.to(device)
    model.eval()

    # Load and preprocess image
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (1024, 512))

    # Create cube projection
    e2c = py360_E2C(equ_h=512, equ_w=1024, face_w=256)
    cube_rgb = e2c.run(rgb)  # (256*6, 256, 3)

    # Normalize
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def normalize(img):
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
        return img.transpose(2, 0, 1)  # HWC -> CHW

    rgb_norm = normalize(rgb)
    cube_rgb_norm = normalize(cube_rgb)

    rgb_tensor = torch.from_numpy(rgb_norm).unsqueeze(0).float().to(device)
    cube_tensor = torch.from_numpy(cube_rgb_norm).unsqueeze(0).float().to(device)

    # Inference
    with torch.no_grad():
        outputs = model(rgb_tensor, cube_tensor)
        pred_depth = outputs['pred_depth'].squeeze().cpu().numpy()

    return pred_depth


def main():
    # Use Structured3D sample with GT
    scene_dir = Path('/mnt/data/floorplan-reconstruction/public_data/stru3d/panorama/Structured3D/scene_00000/2D_rendering/485142/panorama/full')

    rgb_path = str(scene_dir / 'rgb_rawlight.png')
    gt_depth_path = str(scene_dir / 'depth.png')

    output_dir = Path('/home/unav/Desktop/unav/unav/tmp/depth_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load RGB
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (1024, 512))

    # Load GT depth (Structured3D: uint16 in mm)
    gt_depth = cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED)
    gt_depth = gt_depth.astype(np.float32) / 1000.0  # mm to meters

    print(f"GT depth range: {gt_depth.min():.2f} - {gt_depth.max():.2f} m")

    # Load pre-computed Depth-Anywhere prediction
    print("Loading Depth-Anywhere prediction...")
    pred_depth_path = output_dir / 'pred_depth.npy'
    if pred_depth_path.exists():
        pred_depth = np.load(pred_depth_path)
        print(f"Pred depth range: {pred_depth.min():.2f} - {pred_depth.max():.2f} m")
    else:
        print(f"Pre-computed depth not found at {pred_depth_path}")
        print("Run: cd /home/unav/Desktop/unav/unav/tmp/Depth-Anywhere && python run_unifuse_inference.py ...")
        return

    # Resize GT depth to match prediction if needed
    if gt_depth.shape != pred_depth.shape:
        print(f"Resizing GT depth from {gt_depth.shape} to {pred_depth.shape}")
        gt_depth = cv2.resize(gt_depth, (pred_depth.shape[1], pred_depth.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    # Create floor plans
    print("Creating floor plans...")
    gt_floor_plan = create_floor_plan(gt_depth, rgb)
    pred_floor_plan = create_floor_plan(pred_depth, rgb)

    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: RGB, GT Depth, Pred Depth
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('RGB Panorama', fontsize=12)
    axes[0, 0].axis('off')

    im1 = axes[0, 1].imshow(gt_depth, cmap='turbo', vmin=0, vmax=8)
    axes[0, 1].set_title(f'Ground Truth Depth\n[{gt_depth.min():.1f}-{gt_depth.max():.1f}m]', fontsize=12)
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(pred_depth, cmap='turbo', vmin=0, vmax=8)
    axes[0, 2].set_title(f'Predicted Depth\n[{pred_depth.min():.1f}-{pred_depth.max():.1f}m]', fontsize=12)
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # Row 2: Error map, GT Floor Plan, Pred Floor Plan
    error = np.abs(pred_depth - gt_depth)
    error[gt_depth == 0] = 0  # Mask invalid regions

    im3 = axes[1, 0].imshow(error, cmap='hot', vmin=0, vmax=2)
    axes[1, 0].set_title(f'Absolute Error\nMAE: {error[gt_depth > 0].mean():.2f}m', fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].imshow(gt_floor_plan)
    axes[1, 1].set_title('GT Floor Plan (Top-Down)', fontsize=12)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(pred_floor_plan)
    axes[1, 2].set_title('Predicted Floor Plan (Top-Down)', fontsize=12)
    axes[1, 2].axis('off')

    plt.suptitle('Depth-Anywhere vs Ground Truth (Structured3D)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'gt_vs_pred_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved to {output_dir / 'gt_vs_pred_comparison.png'}")


if __name__ == "__main__":
    main()
