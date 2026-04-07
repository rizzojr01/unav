#!/usr/bin/env python3
"""
Extract floor plan from depth maps using camera poses.

Workflow:
1. Load camera poses from SLAM database
2. Load DA² depth maps for each keyframe
3. Load floor masks (from SAM3)
4. For each keyframe:
   - Reproject depth to 3D points (camera-relative)
   - Filter by floor mask
   - Save relative floor points + visualization
5. Stitch all keyframes using absolute poses to create final floor plan

Usage:
    python extract_floor_from_depth.py <sqlite3_db> <depth_dir> <mask_dir> <output_dir>
"""

import sys
sys.path.insert(0, "/home/unav/Desktop/unav/unav/mapper_floorplan")

import os
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from utils.database import load_keyframe_poses
from utils.pointcloud import save_ply
from utils.visualization import generate_topdown_grid_map


def depth_to_3d_camera_frame(depth: np.ndarray, mask: np.ndarray = None,
                              subsample: int = 4, image: np.ndarray = None):
    """
    Convert equirectangular depth map to 3D points in CAMERA frame.
    Optionally sample per-point colors from the original keyframe image.

    Args:
        depth: HxW depth map (meters)
        mask: HxW binary mask (optional, only use pixels where mask > 0)
        subsample: subsample factor to reduce point count

    Returns:
        points_c: Nx3 points in camera coordinates
        colors_c: Nx3 colors in BGR uint8 (or None if image not provided)
    """
    H, W = depth.shape[:2]

    # Create pixel grid (subsampled)
    v_indices = np.arange(0, H, subsample)
    u_indices = np.arange(0, W, subsample)
    u_grid, v_grid = np.meshgrid(u_indices, v_indices)

    # Flatten
    u_flat = u_grid.flatten()
    v_flat = v_grid.flatten()

    # Get depth values at subsampled locations
    depth_flat = depth[v_flat, u_flat]

    # Apply mask if provided
    if mask is not None:
        # Resize mask to match depth if needed
        if mask.shape[:2] != depth.shape[:2]:
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask_flat = mask[v_flat, u_flat]
        valid = (depth_flat > 0.1) & (depth_flat < 50) & (mask_flat > 0)
    else:
        valid = (depth_flat > 0.1) & (depth_flat < 50)

    u_valid = u_flat[valid]
    v_valid = v_flat[valid]
    depth_valid = depth_flat[valid]

    if len(depth_valid) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    # Convert to spherical coordinates
    longitude = (u_valid / W - 0.5) * 2 * np.pi  # [-pi, pi]
    latitude = (0.5 - v_valid / H) * np.pi        # [-pi/2, pi/2]

    # Spherical to Cartesian (camera frame)
    x_c = depth_valid * np.cos(latitude) * np.sin(longitude)
    y_c = -depth_valid * np.sin(latitude)
    z_c = depth_valid * np.cos(latitude) * np.cos(longitude)

    points_c = np.stack([x_c, y_c, z_c], axis=1)

    colors_c = None
    if image is not None:
        if image.shape[:2] != (H, W):
            image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
        colors_c = image[v_valid, u_valid].astype(np.uint8)
    else:
        colors_c = np.ones((len(points_c), 3), dtype=np.uint8) * 255

    return points_c, colors_c


def transform_to_world(points_c: np.ndarray, pose_cw: np.ndarray) -> np.ndarray:
    """
    Transform points from camera frame to world frame.

    Args:
        points_c: Nx3 points in camera coordinates
        pose_cw: 4x4 camera-to-world pose matrix

    Returns:
        points_w: Nx3 points in world coordinates
    """
    if len(points_c) == 0:
        return points_c

    R_cw = pose_cw[:3, :3]
    t_cw = pose_cw[:3, 3]

    # pose_cw transforms world to camera: p_c = R_cw @ p_w + t_cw
    # Inverse: p_w = R_cw.T @ (p_c - t_cw) = R_cw.T @ p_c - R_cw.T @ t_cw
    R_wc = R_cw.T
    t_wc = -R_wc @ t_cw

    points_w = (R_wc @ points_c.T).T + t_wc

    return points_w


def get_camera_world_pose(pose_cw: np.ndarray):
    """
    Get camera position and forward direction in world coordinates.
    """
    R_cw = pose_cw[:3, :3]
    t_cw = pose_cw[:3, 3]
    R_wc = R_cw.T
    position = -R_wc @ t_cw
    # Forward direction (camera +Z axis in world)
    forward = R_wc @ np.array([0, 0, 1])
    return position, forward


def draw_camera_indicator(img: np.ndarray, cam_px: int, cam_pz: int,
                          cam_dir_xz: np.ndarray, color=(0, 0, 255), size=15):
    """Draw a red triangle camera indicator on the image."""
    dx, dz = cam_dir_xz[0], cam_dir_xz[1]
    length = np.sqrt(dx*dx + dz*dz)
    if length > 0:
        dx, dz = dx / length, dz / length
    else:
        dx, dz = 0, -1

    # Triangle vertices
    front_x = cam_px + int(dx * size)
    front_z = cam_pz - int(dz * size)
    perp_x, perp_z = -dz, -dx
    back_left_x = cam_px - int(dx * size * 0.5) + int(perp_x * size * 0.4)
    back_left_z = cam_pz + int(dz * size * 0.5) - int(perp_z * size * 0.4)
    back_right_x = cam_px - int(dx * size * 0.5) - int(perp_x * size * 0.4)
    back_right_z = cam_pz + int(dz * size * 0.5) + int(perp_z * size * 0.4)

    pts = np.array([[front_x, front_z], [back_left_x, back_left_z],
                    [back_right_x, back_right_z]], np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], color)
    cv2.polylines(img, [pts], True, (255, 255, 255), 1)
    cv2.circle(img, (cam_px, cam_pz), 3, (255, 255, 255), -1)


def create_topdown_view(points: np.ndarray, colors: np.ndarray, output_path: str,
                        size: int = 400, padding: float = 0.1,
                        bounds: dict = None, pose_cw: np.ndarray = None):
    """
    Create and save a top-down view of points in fixed world coordinates.

    Args:
        points: Nx3 points (in world coordinates)
        colors: Nx3 colors (BGR)
        output_path: Path to save the image
        size: Output image size
        padding: Padding ratio
        bounds: Fixed world bounds {x_min, x_max, z_min, z_max}
        pose_cw: Camera pose for drawing indicator
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)

    if bounds is not None:
        x_min, x_max = bounds['x_min'], bounds['x_max']
        z_min, z_max = bounds['z_min'], bounds['z_max']
    elif len(points) > 0:
        x, z = points[:, 0], points[:, 2]
        x_min, x_max = x.min(), x.max()
        z_min, z_max = z.min(), z.max()
    else:
        cv2.imwrite(output_path, img)
        return

    # Add padding
    x_range = max(x_max - x_min, 0.1)
    z_range = max(z_max - z_min, 0.1)
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

    # Draw points
    if len(points) > 0:
        x, z = points[:, 0], points[:, 2]
        px = ((x - x_min) / res).astype(int)
        pz = ((z_max - z) / res).astype(int)
        px = np.clip(px, 0, size - 1)
        pz = np.clip(pz, 0, size - 1)
        if colors is None or len(colors) != len(points):
            img[pz, px] = 255
        else:
            img[pz, px] = colors

    # Draw camera indicator
    if pose_cw is not None:
        cam_pos, cam_forward = get_camera_world_pose(pose_cw)
        cam_px = int((cam_pos[0] - x_min) / res)
        cam_pz = int((z_max - cam_pos[2]) / res)
        cam_dir_xz = np.array([cam_forward[0], cam_forward[2]])
        draw_camera_indicator(img, cam_px, cam_pz, cam_dir_xz, color=(0, 0, 255), size=20)

    cv2.imwrite(output_path, img)


def main():
    parser = argparse.ArgumentParser(description='Extract floor plan from depth maps')
    parser.add_argument('sqlite3_db', type=str, help='Path to SQLite3 database (for camera poses)')
    parser.add_argument('depth_dir', type=str, help='Directory containing depth maps (.npy files)')
    parser.add_argument('mask_dir', type=str, help='Directory containing floor masks')
    parser.add_argument('output_dir', type=str, help='Output directory')
    parser.add_argument('--depth-pattern', type=str, default='image{}.npy',
                        help='Depth filename pattern (default: image{}.npy)')
    parser.add_argument('--mask-pattern', type=str, default='image{}_floor_mask.png',
                        help='Mask filename pattern (default: image{}_floor_mask.png)')
    parser.add_argument('--subsample', type=int, default=4,
                        help='Subsample factor for depth map (default: 4)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scale factor for depth values (default: 1.0)')
    parser.add_argument('--keyframes', type=str, default=None,
                        help='Comma-separated list of keyframe IDs to process (default: all available)')
    parser.add_argument('--skip-map', action='store_true',
                        help='Skip generating top-down grid map (avoids sklearn dependency)')

    args = parser.parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    keyframe_dir = output_dir / 'keyframe_floors'
    vis_dir = output_dir / 'keyframe_vis'
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Load camera poses
    print(f"Loading keyframe poses from {args.sqlite3_db}...")
    keyframes = load_keyframe_poses(args.sqlite3_db)
    print(f"  Loaded {len(keyframes)} keyframes")

    # Determine which keyframes to process
    depth_dir = Path(args.depth_dir)
    mask_dir = Path(args.mask_dir)

    if args.keyframes:
        kf_ids = [int(x) for x in args.keyframes.split(',')]
    else:
        # Find all available depth files
        kf_ids = []
        for f in depth_dir.glob('*.npy'):
            name = f.stem
            if name.startswith('image'):
                try:
                    kf_id = int(name.replace('image', ''))
                    if kf_id in keyframes:
                        kf_ids.append(kf_id)
                except ValueError:
                    continue
        kf_ids = sorted(kf_ids)

    print(f"Processing {len(kf_ids)} keyframes")

    # ═══════════════════════════════════════════════════════════════════════════
    # Pass 1: Process each keyframe, compute world points and bounds
    # ═══════════════════════════════════════════════════════════════════════════

    keyframe_data = {}  # Store data for final stitching
    all_world_points = []  # Accumulate for final output
    all_world_colors = []  # Accumulate per-point colors
    processed_kf_data = []  # Store data for second pass

    for kf_id in tqdm(kf_ids, desc="Pass 1: Computing world points"):
        # Load depth
        depth_path = depth_dir / args.depth_pattern.format(kf_id)
        if not depth_path.exists():
            continue

        depth = np.load(depth_path)
        # Handle DA² output shape (1, H, W) or (C, H, W) -> (H, W)
        if depth.ndim == 3:
            depth = depth.squeeze(0)  # Remove batch/channel dimension
        depth = depth * args.scale

        # Load mask
        mask_path = mask_dir / args.mask_pattern.format(kf_id)
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # Use lower half as approximate floor region
            mask = np.zeros(depth.shape[:2], dtype=np.uint8)
            h = depth.shape[0]
            mask[h//2:, :] = 255

        # Get camera pose
        pose_cw = keyframes[kf_id]['pose_cw']

        # Load keyframe image for color sampling (optional)
        keyframe_img_path = Path(args.sqlite3_db).parent / "keyframes" / f"image{kf_id}.png"
        if not keyframe_img_path.exists():
            keyframe_img_path = Path(args.sqlite3_db).parent / "keyframes" / f"image{kf_id}.jpg"
        keyframe_img = cv2.imread(str(keyframe_img_path)) if keyframe_img_path.exists() else None

        # Reproject depth to 3D points (camera-relative)
        points_c, colors_c = depth_to_3d_camera_frame(depth, mask, subsample=args.subsample, image=keyframe_img)

        if len(points_c) == 0:
            continue

        # Transform to world
        points_w = transform_to_world(points_c, pose_cw)
        all_world_points.append(points_w)
        all_world_colors.append(colors_c)

        # Store for second pass
        processed_kf_data.append({
            'kf_id': kf_id,
            'points_c': points_c,
            'points_w': points_w,
            'colors_c': colors_c,
            'pose_cw': pose_cw
        })

        keyframe_data[kf_id] = {
            'pose_cw': pose_cw,
            'num_points': len(points_c)
        }

    # Compute world bounds from all points
    if len(all_world_points) == 0:
        print("No floor points extracted!")
        return

    all_pts = np.vstack(all_world_points)
    y_coords = all_pts[:, 1]
    y_median = np.median(y_coords)
    y_std = np.std(y_coords)
    floor_mask_tmp = (y_coords > y_median - 2*y_std) & (y_coords < y_median + 2*y_std)
    floor_pts_tmp = all_pts[floor_mask_tmp]

    world_bounds = {
        'x_min': float(floor_pts_tmp[:, 0].min()),
        'x_max': float(floor_pts_tmp[:, 0].max()),
        'z_min': float(floor_pts_tmp[:, 2].min()),
        'z_max': float(floor_pts_tmp[:, 2].max()),
    }
    print(f"\nWorld bounds: X=[{world_bounds['x_min']:.1f}, {world_bounds['x_max']:.1f}], "
          f"Z=[{world_bounds['z_min']:.1f}, {world_bounds['z_max']:.1f}]")

    # ═══════════════════════════════════════════════════════════════════════════
    # Pass 2: Generate visualizations with fixed world coordinates
    # ═══════════════════════════════════════════════════════════════════════════

    accumulated_world_pts = []
    accumulated_world_colors = []

    for data in tqdm(processed_kf_data, desc="Pass 2: Generating visualizations"):
        kf_id = data['kf_id']
        points_c = data['points_c']
        points_w = data['points_w']
        colors_c = data['colors_c']
        pose_cw = data['pose_cw']

        # Save per-keyframe data
        kf_output = {
            'kf_id': kf_id,
            'points_camera': points_c.tolist(),
            'pose_cw': pose_cw.tolist(),
            'num_points': len(points_c)
        }
        kf_json_path = keyframe_dir / f'keyframe_{kf_id}.json'
        with open(kf_json_path, 'w') as f:
            json.dump(kf_output, f)

        # Save per-keyframe PLY (camera-relative)
        kf_ply_path = keyframe_dir / f'keyframe_{kf_id}_floor.ply'
        save_ply(str(kf_ply_path), points_c.tolist(), colors_c.tolist())

        # Accumulate world points for progressive visualization
        accumulated_world_pts.append(points_w)
        accumulated_world_colors.append(colors_c)

        acc_pts = np.vstack(accumulated_world_pts)
        acc_colors = np.vstack(accumulated_world_colors)

        # Filter by floor height
        y_c = acc_pts[:, 1]
        y_med = np.median(y_c)
        y_s = np.std(y_c) if len(y_c) > 1 else 1.0
        fm = (y_c > y_med - 2*y_s) & (y_c < y_med + 2*y_s)
        acc_pts_filtered = acc_pts[fm]
        acc_colors_filtered = acc_colors[fm]

        # Generate top-down view in fixed world coordinates with camera indicator
        vis_path = vis_dir / f'keyframe_{kf_id}_topdown.png'
        create_topdown_view(acc_pts_filtered, acc_colors_filtered, str(vis_path),
                           bounds=world_bounds, pose_cw=pose_cw)

    print(f"\nProcessed {len(keyframe_data)} keyframes with floor points")

    # ═══════════════════════════════════════════════════════════════════════════
    # Step 5: Stitch all keyframes using absolute poses
    # ═══════════════════════════════════════════════════════════════════════════

    if len(all_world_points) == 0:
        print("No floor points extracted!")
        return

    print("\nStitching floor points from all keyframes...")

    all_world_points = np.vstack(all_world_points)
    all_world_colors = np.vstack(all_world_colors)
    print(f"  Total floor points: {len(all_world_points)}")

    # Filter by height (keep floor-level points)
    y_coords = all_world_points[:, 1]
    y_median = np.median(y_coords)
    y_std = np.std(y_coords)
    print(f"  Y coordinates: median={y_median:.2f}m, std={y_std:.2f}m")

    floor_mask = (y_coords > y_median - 2*y_std) & (y_coords < y_median + 2*y_std)
    floor_points = all_world_points[floor_mask]
    floor_colors = all_world_colors[floor_mask]
    print(f"  Floor points after height filter: {len(floor_points)}")

    # Save stitched floor points
    output_ply = output_dir / 'floor_points_depth.ply'
    save_ply(str(output_ply), floor_points.tolist(), floor_colors.tolist())
    print(f"\nSaved stitched floor points to {output_ply}")

    # Generate final floor plan visualization
    if not args.skip_map:
        print("\nGenerating final floor plan...")
        map_output = output_dir / 'floor_map_depth.png'
        try:
            generate_topdown_grid_map(
                floor_points.tolist(),
                floor_colors.tolist(),
                {k: keyframes[k] for k in keyframe_data.keys()},
                None,
                str(map_output),
                grid_resolution=0.05,
                draw_doors=False
            )
        except Exception as e:
            print(f"Warning: failed to generate floor map: {e}")

    # Save metadata
    metadata = {
        'total_keyframes': len(keyframe_data),
        'total_points': len(floor_points),
        'keyframes': {str(k): v['num_points'] for k, v in keyframe_data.items()},
        'bounds': {
            'x_min': float(floor_points[:, 0].min()),
            'x_max': float(floor_points[:, 0].max()),
            'y_min': float(floor_points[:, 1].min()),
            'y_max': float(floor_points[:, 1].max()),
            'z_min': float(floor_points[:, 2].min()),
            'z_max': float(floor_points[:, 2].max()),
        }
    }
    with open(output_dir / 'floor_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Keyframes processed: {len(keyframe_data)}")
    print(f"Total floor points:  {len(floor_points)}")
    print(f"X range: {floor_points[:, 0].min():.2f} - {floor_points[:, 0].max():.2f} m")
    print(f"Z range: {floor_points[:, 2].min():.2f} - {floor_points[:, 2].max():.2f} m")
    print("=" * 60)
    print("\nOutput files:")
    print(f"  Per-keyframe data: {keyframe_dir}/")
    print(f"  Per-keyframe vis:  {vis_dir}/")
    print(f"  Stitched PLY:      {output_ply}")
    if not args.skip_map:
        print(f"  Floor map:         {map_output}")
    print(f"  Metadata:          {output_dir / 'floor_metadata.json'}")


if __name__ == '__main__':
    main()
