"""
Visualization utilities for point clouds and maps.
"""

import numpy as np


def draw_camera_icon(ax, position, direction, size=0.5, color='red', alpha=0.8):
    """
    Draw a camera icon as an isosceles triangle.
    The apex of the triangle points towards the camera direction.

    Args:
        ax: matplotlib axis
        position: camera position (x, y) in 2D
        direction: camera forward direction (x, y) in 2D
        size: size of the triangle
        color: color of the triangle
        alpha: transparency
    """
    from matplotlib.patches import Polygon

    # Normalize direction
    direction = np.array(direction)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        direction = np.array([0, 1])
    else:
        direction = direction / norm

    # Perpendicular direction (for the base of triangle)
    perp = np.array([-direction[1], direction[0]])

    # Triangle vertices: apex points forward, base is perpendicular
    apex = position + direction * size * 1.5  # Apex in front
    base_left = position - perp * size * 0.5
    base_right = position + perp * size * 0.5

    # Create triangle
    triangle = Polygon([apex, base_left, base_right],
                      closed=True,
                      facecolor=color,
                      edgecolor='black',
                      alpha=alpha,
                      linewidth=1.5)
    ax.add_patch(triangle)

    return triangle


def add_scale_bar(ax, x_min, x_max, z_min, z_max, length=1.0):
    """
    Add a scale bar to the map.

    Args:
        ax: matplotlib axis
        x_min, x_max: x-axis bounds
        z_min, z_max: z-axis bounds
        length: scale bar length in meters
    """
    from matplotlib.lines import Line2D

    # Position scale bar at bottom left
    bar_x = x_min + (x_max - x_min) * 0.05
    bar_z = z_min + (z_max - z_min) * 0.05

    # Draw scale bar
    line = Line2D([bar_x, bar_x + length], [bar_z, bar_z],
                  linewidth=3, color='black')
    ax.add_line(line)

    # Add text
    ax.text(bar_x + length/2, bar_z - (z_max - z_min) * 0.02,
            f'{length}m', ha='center', va='top', fontsize=10, fontweight='bold')


def visualize_floor_points_with_cameras(points, colors, keyframes, output_path, max_points=50000):
    """
    Create three views of floor points with camera poses in top view.
    Saves combined view to output_path.

    Args:
        points: list of 3D points (N x 3)
        colors: list of colors (N x 3, BGR)
        keyframes: dict of keyframe data with poses
        output_path: path to save the visualization
        max_points: maximum points to render (downsampled if exceeded)
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from .geometry import get_camera_world_pose

    if len(points) == 0:
        print("No points to visualize")
        return

    points = np.array(points)
    colors = np.array(colors) / 255.0  # Normalize to [0, 1]

    # Convert BGR to RGB for matplotlib
    colors_rgb = colors[:, [2, 1, 0]]

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 6))

    # Downsample points for faster rendering if too many
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points_vis = points[indices]
        colors_vis = colors_rgb[indices]
    else:
        points_vis = points
        colors_vis = colors_rgb

    # 1. Top view (looking down, -Y axis)
    ax1 = fig.add_subplot(131)
    ax1.scatter(points_vis[:, 0], points_vis[:, 2], c=colors_vis, s=1, alpha=0.5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Z (m)')
    ax1.set_title('Top View (Looking Down)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Add camera poses to top view
    camera_positions = []
    camera_directions = []

    for kf_id, kf_data in keyframes.items():
        cam_pos, cam_forward = get_camera_world_pose(kf_data['pose_cw'])
        camera_positions.append(cam_pos)
        camera_directions.append(cam_forward)

    # Draw camera icons (every Nth camera to avoid clutter)
    step = max(1, len(camera_positions) // 50)  # Show at most 50 cameras
    for i in range(0, len(camera_positions), step):
        pos = camera_positions[i]
        direction = camera_directions[i]

        # Project to XZ plane (top view)
        pos_2d = np.array([pos[0], pos[2]])
        dir_2d = np.array([direction[0], direction[2]])

        # Determine size based on point cloud extent
        extent = np.max(points[:, [0, 2]].max(axis=0) - points[:, [0, 2]].min(axis=0))
        cam_size = extent * 0.02  # 2% of scene size

        draw_camera_icon(ax1, pos_2d, dir_2d, size=cam_size, color='red', alpha=0.7)

    # 2. Front view (looking along +Z axis)
    ax2 = fig.add_subplot(132)
    ax2.scatter(points_vis[:, 0], points_vis[:, 1], c=colors_vis, s=1, alpha=0.5)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Front View')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # Invert Y axis (up is positive in world frame)

    # 3. Side view (looking along +X axis)
    ax3 = fig.add_subplot(133)
    ax3.scatter(points_vis[:, 2], points_vis[:, 1], c=colors_vis, s=1, alpha=0.5)
    ax3.set_xlabel('Z (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('Side View')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()  # Invert Y axis

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()


def remove_outliers_percentile(points, colors, percentile=2):
    """
    Remove outliers based on percentile for each dimension.

    Args:
        points: Nx3 array of points
        colors: Nx3 array of colors
        percentile: percentile to cut from each end (default 2%)

    Returns:
        filtered points and colors
    """
    mask = np.ones(len(points), dtype=bool)

    for dim in range(3):
        low = np.percentile(points[:, dim], percentile)
        high = np.percentile(points[:, dim], 100 - percentile)
        mask &= (points[:, dim] >= low) & (points[:, dim] <= high)

    return points[mask], colors[mask]


def generate_separate_views(points, colors, keyframes, output_dir, max_points=50000, point_keyframe_ids=None):
    """
    Generate separate view images for interactive web display.

    Args:
        points: list of 3D points (N x 3)
        colors: list of colors (N x 3, BGR)
        keyframes: dict of keyframe data with poses
        output_dir: directory to save the view images
        max_points: maximum points to render
        point_keyframe_ids: list of keyframe IDs for each point (for highlighting)

    Returns:
        dict: Camera data for web interface {kf_id: {x, z, dir_x, dir_z}}
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os
    import json
    from .geometry import get_camera_world_pose

    os.makedirs(output_dir, exist_ok=True)

    if len(points) == 0:
        print("No points to visualize")
        return {}

    points = np.array(points)
    colors = np.array(colors) / 255.0
    colors_rgb = colors[:, [2, 1, 0]]

    # Downsample points
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points_vis = points[indices]
        colors_vis = colors_rgb[indices]
        if point_keyframe_ids is not None:
            point_kf_ids_vis = [point_keyframe_ids[i] for i in indices]
        else:
            point_kf_ids_vis = None
    else:
        points_vis = points
        colors_vis = colors_rgb
        point_kf_ids_vis = point_keyframe_ids

    # Remove outliers for front/side views (more aggressive filtering)
    points_filtered, colors_filtered = remove_outliers_percentile(points_vis, colors_vis, percentile=5)

    # Get bounds for consistent scaling
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()

    # Prepare camera data with point indices
    camera_data = {}
    keyframe_points = {}  # kf_id -> list of point indices

    if point_kf_ids_vis is not None:
        for idx, kf_id in enumerate(point_kf_ids_vis):
            if kf_id not in keyframe_points:
                keyframe_points[kf_id] = []
            keyframe_points[kf_id].append(idx)

    for kf_id, kf_info in keyframes.items():
        cam_pos, cam_forward = get_camera_world_pose(kf_info['pose_cw'])
        camera_data[kf_id] = {
            'x': float(cam_pos[0]),
            'y': float(cam_pos[1]),
            'z': float(cam_pos[2]),
            'dir_x': float(cam_forward[0]),
            'dir_y': float(cam_forward[1]),
            'dir_z': float(cam_forward[2])
        }

    # 1. Top view (main view, no axis labels, no title)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(points_vis[:, 0], points_vis[:, 2], c=colors_vis, s=1, alpha=0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(output_dir, 'view_top.png'), dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # 2. Front view (thumbnail, filtered, no axis, no title, tighter bounds)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(points_filtered[:, 0], points_filtered[:, 1], c=colors_filtered, s=0.5, alpha=0.6)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(output_dir, 'view_front.png'), dpi=100, bbox_inches='tight', pad_inches=0.05)
    plt.close()

    # 3. Side view (thumbnail, filtered, no axis, no title, tighter bounds)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(points_filtered[:, 2], points_filtered[:, 1], c=colors_filtered, s=0.5, alpha=0.6)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(output_dir, 'view_side.png'), dpi=100, bbox_inches='tight', pad_inches=0.05)
    plt.close()

    # Prepare point positions for each keyframe (in normalized image coordinates)
    # to enable highlighting in the frontend
    keyframe_point_positions = {}
    if point_kf_ids_vis is not None:
        for kf_id, point_indices in keyframe_points.items():
            if len(point_indices) > 0:
                # Limit points per keyframe to avoid huge JSON
                sampled_indices = point_indices[:200] if len(point_indices) > 200 else point_indices
                positions = []
                for idx in sampled_indices:
                    pt = points_vis[idx]
                    # Normalized coordinates (0-1) for top view
                    nx = (pt[0] - x_min) / (x_max - x_min) if x_max > x_min else 0.5
                    nz = (pt[2] - z_min) / (z_max - z_min) if z_max > z_min else 0.5
                    positions.append([round(nx, 4), round(nz, 4)])
                keyframe_point_positions[str(kf_id)] = positions

    # Save camera data and bounds as JSON
    metadata = {
        'cameras': camera_data,
        'bounds': {
            'x_min': float(x_min), 'x_max': float(x_max),
            'y_min': float(y_min), 'y_max': float(y_max),
            'z_min': float(z_min), 'z_max': float(z_max)
        },
        'keyframe_points': keyframe_point_positions
    }
    with open(os.path.join(output_dir, 'view_metadata.json'), 'w') as f:
        json.dump(metadata, f)

    print(f"Saved separate views to {output_dir}")
    print(f"  - view_top.png (main view)")
    print(f"  - view_front.png (thumbnail)")
    print(f"  - view_side.png (thumbnail)")
    print(f"  - view_metadata.json ({len(camera_data)} cameras, {len(keyframe_point_positions)} with points)")

    return camera_data


def generate_topdown_grid_map(points, colors, keyframes, door_masks_dir, output_path,
                               grid_resolution=0.05, draw_doors=False):
    """
    Generate top-down floor maps (both RGB and white versions).

    Args:
        points: list of 3D points (N x 3) in WORLD coordinates
        colors: list of colors (N x 3, BGR)
        keyframes: dict of keyframe data with poses
        door_masks_dir: directory containing door masks (or None)
        output_path: path to save the map (will also generate _white.png version)
        grid_resolution: grid cell size in meters
        draw_doors: whether to draw door markers (default: False)
    """
    import cv2
    import os
    from pathlib import Path

    if len(points) == 0:
        print("No points to generate map")
        return

    points = np.array(points)
    colors = np.array(colors) if colors is not None else None

    # Get bounds
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()

    # Add padding
    padding = 0.5  # meters
    x_min -= padding
    x_max += padding
    z_min -= padding
    z_max += padding

    # Create grid dimensions
    width = x_max - x_min
    height = z_max - z_min
    grid_w = int(width / grid_resolution)
    grid_h = int(height / grid_resolution)

    print(f"  Grid size: {grid_w} x {grid_h} (resolution: {grid_resolution}m)")

    # Initialize grids
    # grid_rgb: accumulate colors for RGB version
    # grid_count: count points per cell for averaging
    # grid_white: binary occupancy for white version
    grid_rgb = np.zeros((grid_h, grid_w, 3), dtype=np.float64)
    grid_count = np.zeros((grid_h, grid_w), dtype=np.int32)
    grid_white = np.zeros((grid_h, grid_w), dtype=np.uint8)

    # Accumulate points in grid
    for i, point in enumerate(points):
        x, z = point[0], point[2]
        grid_x = int((x - x_min) / grid_resolution)
        grid_z = int((z - z_min) / grid_resolution)

        if 0 <= grid_x < grid_w and 0 <= grid_z < grid_h:
            grid_white[grid_z, grid_x] = 255
            grid_count[grid_z, grid_x] += 1
            if colors is not None and i < len(colors):
                grid_rgb[grid_z, grid_x] += colors[i]

    # Average colors where we have points
    mask = grid_count > 0
    grid_rgb[mask] = grid_rgb[mask] / grid_count[mask, np.newaxis]
    grid_rgb = grid_rgb.astype(np.uint8)

    # Apply morphological closing to fill small gaps (for white version)
    kernel_size = int(0.2 / grid_resolution)  # 0.2m kernel
    if kernel_size % 2 == 0:
        kernel_size += 1  # Must be odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    grid_white_closed = cv2.morphologyEx(grid_white, cv2.MORPH_CLOSE, kernel)

    # Apply opening to remove small noise
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grid_white_final = cv2.morphologyEx(grid_white_closed, cv2.MORPH_OPEN, small_kernel)

    # Create white version (3-channel)
    grid_white_rgb = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    grid_white_rgb[grid_white_final == 255] = [255, 255, 255]

    # For RGB version, also apply morphological operations to fill gaps
    # Dilate RGB to fill small gaps, then mask with white version
    rgb_dilated = cv2.dilate(grid_rgb, small_kernel, iterations=1)
    # Use closed white mask to determine where to keep colors
    grid_rgb_final = np.zeros_like(grid_rgb)
    grid_rgb_final[grid_white_closed > 0] = rgb_dilated[grid_white_closed > 0]

    # Extract and cluster door locations if requested
    if draw_doors and door_masks_dir and os.path.isdir(door_masks_dir):
        try:
            from .door_detection import extract_door_points_from_masks
            from sklearn.cluster import DBSCAN

            door_points_list = extract_door_points_from_masks(keyframes, door_masks_dir, points)
            if len(door_points_list) > 0:
                door_points_array = np.array(door_points_list)

                clustering = DBSCAN(eps=1.0, min_samples=1).fit(door_points_array[:, [0, 2]])
                unique_labels = set(clustering.labels_)

                print(f"  Detected {len(door_points_list)} door observations, clustered into {len(unique_labels)} unique doors")

                for label in unique_labels:
                    if label == -1:
                        continue

                    cluster_mask = clustering.labels_ == label
                    cluster_points = door_points_array[cluster_mask]
                    centroid = cluster_points.mean(axis=0)
                    x, z = centroid[0], centroid[2]

                    grid_x = int((x - x_min) / grid_resolution)
                    grid_z = int((z - z_min) / grid_resolution)

                    if 0 <= grid_x < grid_w and 0 <= grid_z < grid_h:
                        door_width_cells = int(0.9 / grid_resolution)
                        door_height_cells = int(0.3 / grid_resolution)

                        x1 = max(0, grid_x - door_width_cells // 2)
                        x2 = min(grid_w, grid_x + door_width_cells // 2)
                        z1 = max(0, grid_z - door_height_cells // 2)
                        z2 = min(grid_h, grid_z + door_height_cells // 2)

                        grid_rgb_final[z1:z2, x1:x2] = [0, 255, 0]
                        grid_white_rgb[z1:z2, x1:x2] = [0, 255, 0]
        except Exception as e:
            print(f"  Warning: door detection failed: {e}")

    # Flip vertically for correct orientation (origin at bottom-left)
    grid_rgb_final = np.flipud(grid_rgb_final)
    grid_white_rgb = np.flipud(grid_white_rgb)

    # Save RGB version
    cv2.imwrite(output_path, grid_rgb_final)
    print(f"Saved RGB floor map to {output_path}")

    # Save white version
    output_path_obj = Path(output_path)
    white_path = output_path_obj.parent / f"{output_path_obj.stem}_white{output_path_obj.suffix}"
    cv2.imwrite(str(white_path), grid_white_rgb)
    print(f"Saved white floor map to {white_path}")

    print(f"  Map dimensions: {grid_w} x {grid_h} pixels")
    print(f"  Real-world size: {width:.2f}m x {height:.2f}m")
