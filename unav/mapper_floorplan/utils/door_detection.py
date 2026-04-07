"""
Door detection utilities.
"""

import numpy as np
from pathlib import Path


def extract_door_points_from_masks(keyframes, door_masks_dir, floor_points):
    """
    Extract door locations from door masks by finding intersection with floor points.

    Args:
        keyframes: dict of keyframe data with poses
        door_masks_dir: directory containing door masks
        floor_points: array of floor points (N x 3) to find door positions

    Returns:
        list: List of 3D door positions
    """
    import cv2
    from scipy.spatial import cKDTree
    from .geometry import get_camera_world_pose, pixel_to_bearing_equirectangular

    door_points = []

    # Build KD-tree for fast nearest neighbor search
    if len(floor_points) == 0:
        return door_points

    floor_points = np.array(floor_points)
    floor_tree = cKDTree(floor_points)

    for kf_id, kf_data in keyframes.items():
        # Look for door mask file
        mask_path = Path(door_masks_dir) / f"image{kf_id}_door_mask.png"
        if not mask_path.exists():
            continue

        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # Find door regions (white pixels)
        door_pixels = np.where(mask > 128)
        if len(door_pixels[0]) == 0:
            continue

        # Get mask dimensions
        mask_h, mask_w = mask.shape

        # Calculate center of door region
        center_v = int(np.mean(door_pixels[0]))
        center_u = int(np.mean(door_pixels[1]))

        # Get camera pose
        pose_cw = kf_data['pose_cw']
        cam_pos, _ = get_camera_world_pose(pose_cw)

        # Get rotation matrix for world coordinates
        R_cw = pose_cw[:3, :3]
        R_wc = R_cw.T

        # Convert pixel to bearing in camera frame
        bearing_camera = pixel_to_bearing_equirectangular(center_u, center_v, mask_w, mask_h)

        # Transform to world frame
        bearing_world = R_wc @ bearing_camera

        # Ray-cast to find door position on floor boundary
        # Sample points along the ray and find the one closest to floor points
        best_distance = float('inf')
        best_point = None

        for distance in np.linspace(0.5, 10.0, 20):  # Sample from 0.5m to 10m
            ray_point = cam_pos + bearing_world * distance

            # Find nearest floor point
            dist, idx = floor_tree.query(ray_point)

            # Look for the point where we're close to floor (within 0.5m)
            if dist < 0.5 and dist < best_distance:
                best_distance = dist
                best_point = floor_points[idx]

        if best_point is not None:
            door_points.append(best_point)

    return door_points
