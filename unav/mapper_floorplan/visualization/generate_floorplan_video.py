#!/usr/bin/env python3
"""
Generate a 2x2 grid video showing:
- Top-left: Original keyframe image (2:1 aspect ratio)
- Top-right: Top-down floor plan with camera indicator
- Bottom-left: Floor mask (2:1 aspect ratio)
- Bottom-right: Depth map visualization (2:1 aspect ratio)

The top-down view uses a fixed world coordinate system (first frame as reference),
with a red triangle showing the current camera position and orientation.
Each frame only shows the current keyframe's floor points (not accumulated).
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, "/home/unav/Desktop/unav/unav/mapper_floorplan")
from utils.database import load_keyframe_poses


def colorize_depth(depth: np.ndarray, min_depth: float = 0.1, max_depth: float = 10.0) -> np.ndarray:
    """Convert depth map to colorized visualization."""
    # Clip and normalize
    depth_clipped = np.clip(depth, min_depth, max_depth)
    depth_norm = (depth_clipped - min_depth) / (max_depth - min_depth)

    # Apply colormap (turbo for better visualization)
    depth_colored = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

    # Set invalid regions to black
    depth_colored[depth < min_depth] = 0

    return depth_colored


def get_camera_position_and_direction(pose_cw: np.ndarray):
    """
    Extract camera position and forward direction in world coordinates.

    Args:
        pose_cw: 4x4 camera-to-world transformation matrix

    Returns:
        position: (x, y, z) camera position in world
        forward: (x, y, z) forward direction vector in world
    """
    R_cw = pose_cw[:3, :3]
    t_cw = pose_cw[:3, 3]

    # Camera position in world: p_w = -R_cw.T @ t_cw
    R_wc = R_cw.T
    position = -R_wc @ t_cw

    # Camera forward direction (z-axis in camera frame) in world
    # In camera frame, forward is +Z
    forward_cam = np.array([0, 0, 1])
    forward_world = R_wc @ forward_cam

    return position, forward_world


def draw_camera_indicator(img: np.ndarray,
                          cam_pos_px: tuple,
                          cam_dir: np.ndarray,
                          color: tuple = (0, 0, 255),  # Red in BGR
                          size: int = 15,
                          thickness: int = 2):
    """
    Draw a camera indicator (triangle/frustum) on the image.

    Args:
        img: Image to draw on
        cam_pos_px: (px, pz) camera position in pixel coordinates
        cam_dir: (dx, dz) camera forward direction (normalized, in world XZ plane)
        color: BGR color tuple
        size: Size of the triangle
        thickness: Line thickness
    """
    px, pz = int(cam_pos_px[0]), int(cam_pos_px[1])

    # Normalize direction
    dx, dz = cam_dir[0], cam_dir[1]
    length = np.sqrt(dx*dx + dz*dz)
    if length > 0:
        dx, dz = dx / length, dz / length
    else:
        dx, dz = 0, -1  # Default to pointing up

    # Triangle vertices (pointing in camera direction)
    # Front vertex
    front_x = px + int(dx * size)
    front_z = pz - int(dz * size)  # Note: z is inverted in image coords

    # Back vertices (perpendicular to direction)
    perp_x, perp_z = -dz, -dx  # Perpendicular in XZ plane
    back_left_x = px - int(dx * size * 0.5) + int(perp_x * size * 0.4)
    back_left_z = pz + int(dz * size * 0.5) - int(perp_z * size * 0.4)
    back_right_x = px - int(dx * size * 0.5) - int(perp_x * size * 0.4)
    back_right_z = pz + int(dz * size * 0.5) + int(perp_z * size * 0.4)

    # Draw filled triangle
    pts = np.array([[front_x, front_z],
                    [back_left_x, back_left_z],
                    [back_right_x, back_right_z]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], color)
    cv2.polylines(img, [pts], True, (255, 255, 255), 1)  # White border

    # Draw camera center point
    cv2.circle(img, (px, pz), 3, (255, 255, 255), -1)


def create_topdown_view_world(all_points: np.ndarray,
                               all_colors: np.ndarray,
                               current_pose: np.ndarray,
                               bounds: dict,
                               size: int = 400,
                               cam_trail: list = None) -> np.ndarray:
    """
    Create a top-down view in fixed world coordinates with camera indicator.

    Args:
        all_points: Nx3 accumulated world points
        all_colors: Nx3 colors (BGR)
        current_pose: 4x4 current camera pose
        bounds: Dict with x_min, x_max, z_min, z_max
        size: Output image size
        cam_trail: List of previous camera positions for trail visualization

    Returns:
        Top-down view image
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)

    x_min, x_max = bounds['x_min'], bounds['x_max']
    z_min, z_max = bounds['z_min'], bounds['z_max']

    # Add padding
    x_range = x_max - x_min
    z_range = z_max - z_min
    padding = 0.1
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

    # Draw accumulated floor points
    if len(all_points) > 0:
        x, z = all_points[:, 0], all_points[:, 2]
        px = ((x - x_min) / res).astype(int)
        pz = ((z_max - z) / res).astype(int)
        px = np.clip(px, 0, size - 1)
        pz = np.clip(pz, 0, size - 1)

        if all_colors is not None and len(all_colors) == len(all_points):
            img[pz, px] = all_colors
        else:
            img[pz, px] = 255

    # Draw camera trail
    if cam_trail and len(cam_trail) > 1:
        for i in range(1, len(cam_trail)):
            p1 = cam_trail[i-1]
            p2 = cam_trail[i]
            px1 = int((p1[0] - x_min) / res)
            pz1 = int((z_max - p1[2]) / res)
            px2 = int((p2[0] - x_min) / res)
            pz2 = int((z_max - p2[2]) / res)
            cv2.line(img, (px1, pz1), (px2, pz2), (100, 100, 100), 1)

    # Draw current camera indicator
    if current_pose is not None:
        cam_pos, cam_forward = get_camera_position_and_direction(current_pose)

        # Convert to pixel coordinates
        cam_px = int((cam_pos[0] - x_min) / res)
        cam_pz = int((z_max - cam_pos[2]) / res)

        # Camera direction in XZ plane (note: z is inverted for image)
        cam_dir = np.array([cam_forward[0], cam_forward[2]])

        draw_camera_indicator(img, (cam_px, cam_pz), cam_dir,
                             color=(0, 0, 255), size=20, thickness=2)

    return img


def generate_video(floor_map_dir: str,
                   slam_dir: str,
                   output_path: str,
                   fps: int = 10,
                   cell_height: int = 360):
    """
    Generate 2x2 grid video.

    Layout:
    - Top-left: Keyframe (2:1 aspect ratio)
    - Top-right: Floor map (square, padded to 2:1)
    - Bottom-left: Mask (2:1 aspect ratio)
    - Bottom-right: Depth (2:1 aspect ratio)

    Args:
        floor_map_dir: Directory containing floor_map outputs (masks, depth, etc.)
        slam_dir: Directory containing SLAM data (keyframes, database)
        output_path: Output video path
        fps: Frames per second
        cell_height: Height of each cell (width = 2 * height for 2:1 ratio)
    """
    cell_width = cell_height * 2  # 2:1 aspect ratio
    floor_map_dir = Path(floor_map_dir)
    slam_dir = Path(slam_dir)

    # Load keyframe poses
    db_path = slam_dir / "final_map.msg"
    print(f"Loading keyframe poses from {db_path}...")
    keyframes = load_keyframe_poses(str(db_path))
    print(f"  Loaded {len(keyframes)} keyframes")

    # Find available keyframes (those with depth files)
    depth_dir = floor_map_dir / "depth"
    mask_dir = floor_map_dir / "masks"
    keyframes_dir = slam_dir / "keyframes"

    kf_ids = []
    for f in sorted(depth_dir.glob("image*.npy")):
        name = f.stem
        try:
            kf_id = int(name.replace("image", ""))
            if kf_id in keyframes:
                kf_ids.append(kf_id)
        except ValueError:
            continue
    kf_ids = sorted(kf_ids)
    print(f"Found {len(kf_ids)} keyframes with depth data")

    if len(kf_ids) == 0:
        print("No keyframes found!")
        return

    # Get first frame pose as reference (for coordinate system)
    first_pose = keyframes[kf_ids[0]]['pose_cw']

    # First pass: compute world bounds and accumulate all points
    print("\nFirst pass: computing world bounds...")
    all_world_points = []
    all_world_colors = []

    for kf_id in tqdm(kf_ids, desc="Loading points"):
        # Load depth
        depth_path = depth_dir / f"image{kf_id}.npy"
        depth = np.load(depth_path)
        if depth.ndim == 3:
            depth = depth.squeeze(0)
        depth = depth * 5.0  # Scale factor

        # Load mask
        mask_path = mask_dir / f"image{kf_id}_floor_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None

        # Load image for colors
        img_path = keyframes_dir / f"image{kf_id}.png"
        if not img_path.exists():
            img_path = keyframes_dir / f"image{kf_id}.jpg"
        keyframe_img = cv2.imread(str(img_path)) if img_path.exists() else None

        # Get camera pose
        pose_cw = keyframes[kf_id]['pose_cw']

        # Reproject to 3D (simplified, subsample heavily for speed)
        H, W = depth.shape[:2]
        subsample = 16
        v_indices = np.arange(0, H, subsample)
        u_indices = np.arange(0, W, subsample)
        u_grid, v_grid = np.meshgrid(u_indices, v_indices)
        u_flat, v_flat = u_grid.flatten(), v_grid.flatten()
        depth_flat = depth[v_flat, u_flat]

        if mask is not None:
            if mask.shape[:2] != depth.shape[:2]:
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            mask_flat = mask[v_flat, u_flat]
            valid = (depth_flat > 0.1) & (depth_flat < 50) & (mask_flat > 0)
        else:
            valid = (depth_flat > 0.1) & (depth_flat < 50)

        u_valid, v_valid = u_flat[valid], v_flat[valid]
        depth_valid = depth_flat[valid]

        if len(depth_valid) == 0:
            continue

        # Spherical to Cartesian
        longitude = (u_valid / W - 0.5) * 2 * np.pi
        latitude = (0.5 - v_valid / H) * np.pi
        x_c = depth_valid * np.cos(latitude) * np.sin(longitude)
        y_c = -depth_valid * np.sin(latitude)
        z_c = depth_valid * np.cos(latitude) * np.cos(longitude)
        points_c = np.stack([x_c, y_c, z_c], axis=1)

        # Transform to world
        R_cw = pose_cw[:3, :3]
        t_cw = pose_cw[:3, 3]
        R_wc = R_cw.T
        t_wc = -R_wc @ t_cw
        points_w = (R_wc @ points_c.T).T + t_wc

        # Get colors
        if keyframe_img is not None:
            if keyframe_img.shape[:2] != (H, W):
                keyframe_img = cv2.resize(keyframe_img, (W, H))
            colors = keyframe_img[v_valid, u_valid]
        else:
            colors = np.ones((len(points_w), 3), dtype=np.uint8) * 255

        all_world_points.append(points_w)
        all_world_colors.append(colors)

    # Compute bounds
    all_world_points_np = np.vstack(all_world_points)
    all_world_colors_np = np.vstack(all_world_colors)

    # Filter by height
    y_coords = all_world_points_np[:, 1]
    y_median = np.median(y_coords)
    y_std = np.std(y_coords)
    floor_mask = (y_coords > y_median - 2*y_std) & (y_coords < y_median + 2*y_std)
    all_world_points_np = all_world_points_np[floor_mask]
    all_world_colors_np = all_world_colors_np[floor_mask]

    bounds = {
        'x_min': float(all_world_points_np[:, 0].min()),
        'x_max': float(all_world_points_np[:, 0].max()),
        'z_min': float(all_world_points_np[:, 2].min()),
        'z_max': float(all_world_points_np[:, 2].max()),
    }
    print(f"  World bounds: X=[{bounds['x_min']:.1f}, {bounds['x_max']:.1f}], Z=[{bounds['z_min']:.1f}, {bounds['z_max']:.1f}]")

    # Setup video writer (2x2 grid, each cell is cell_width x cell_height)
    video_width = cell_width * 2
    video_height = cell_height * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height))

    # Second pass: generate frames
    print("\nGenerating video frames...")
    cam_trail = []

    for idx, kf_id in enumerate(tqdm(kf_ids, desc="Generating frames")):
        # Load keyframe image (2:1 aspect ratio)
        img_path = keyframes_dir / f"image{kf_id}.png"
        if not img_path.exists():
            img_path = keyframes_dir / f"image{kf_id}.jpg"
        keyframe_img = cv2.imread(str(img_path))
        if keyframe_img is None:
            keyframe_img = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
        else:
            keyframe_img = cv2.resize(keyframe_img, (cell_width, cell_height))

        # Load mask (2:1 aspect ratio)
        mask_path = mask_dir / f"image{kf_id}_floor_mask.png"
        if mask_path.exists():
            mask_img = cv2.imread(str(mask_path))
            mask_img = cv2.resize(mask_img, (cell_width, cell_height))
        else:
            mask_img = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)

        # Load and colorize depth (2:1 aspect ratio)
        depth_path = depth_dir / f"image{kf_id}.npy"
        depth = np.load(depth_path)
        if depth.ndim == 3:
            depth = depth.squeeze(0)
        depth = depth * 5.0
        depth_vis = colorize_depth(depth, min_depth=0.5, max_depth=15.0)
        depth_vis = cv2.resize(depth_vis, (cell_width, cell_height))

        # Get current frame's points only (not accumulated)
        current_pts = all_world_points[idx]
        current_cols = all_world_colors[idx]

        # Filter by height
        y_c = current_pts[:, 1]
        y_med = np.median(y_c) if len(y_c) > 0 else 0
        y_s = np.std(y_c) if len(y_c) > 1 else 1.0
        fm = (y_c > y_med - 2*y_s) & (y_c < y_med + 2*y_s)
        current_pts = current_pts[fm]
        current_cols = current_cols[fm]

        # Get current camera pose and add to trail
        pose_cw = keyframes[kf_id]['pose_cw']
        cam_pos, _ = get_camera_position_and_direction(pose_cw)
        cam_trail.append(cam_pos)

        # Create top-down view (square, then pad to 2:1)
        topdown_square = create_topdown_view_world(current_pts, current_cols, pose_cw, bounds,
                                                   size=cell_height, cam_trail=cam_trail)
        # Pad square to 2:1 (add black bars on left and right)
        pad_width = (cell_width - cell_height) // 2
        topdown = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
        topdown[:, pad_width:pad_width+cell_height] = topdown_square

        # Add labels
        cv2.putText(keyframe_img, f"Keyframe {kf_id}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(topdown, f"Floor Map ({idx+1}/{len(kf_ids)})", (pad_width + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(mask_img, "Floor Mask", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(depth_vis, "Depth", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Compose 2x2 grid: keyframe(TL), floor_map(TR), mask(BL), depth(BR)
        top_row = np.hstack([keyframe_img, topdown])
        bottom_row = np.hstack([mask_img, depth_vis])
        frame = np.vstack([top_row, bottom_row])

        out.write(frame)

    out.release()
    print(f"\nVideo saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate floor plan visualization video')
    parser.add_argument('floor_map_dir', type=str,
                       help='Directory containing floor_map outputs')
    parser.add_argument('slam_dir', type=str,
                       help='Directory containing SLAM data (keyframes, database)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output video path (default: floor_map_dir/floorplan_video.mp4)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second (default: 10)')
    parser.add_argument('--height', type=int, default=360,
                       help='Height of each cell (width = 2*height for 2:1 ratio, default: 360)')

    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        output_path = str(Path(args.floor_map_dir) / "floorplan_video.mp4")

    generate_video(args.floor_map_dir, args.slam_dir, output_path,
                  fps=args.fps, cell_height=args.height)


if __name__ == '__main__':
    main()
