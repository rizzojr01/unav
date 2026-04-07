"""
Geometry utilities for 3D projection and coordinate transformations.
"""

import numpy as np


def project_to_equirectangular(pos_w, pose_cw, cols, rows):
    """
    Project 3D world point to equirectangular image coordinates.

    Args:
        pos_w: 3D point in world coordinates (3,)
        pose_cw: camera-to-world pose matrix (4x4)
        cols: image width
        rows: image height

    Returns:
        tuple: (u, v) pixel coordinates or None if invalid
    """
    # Transform to camera coordinates
    R_cw = pose_cw[:3, :3]
    t_cw = pose_cw[:3, 3]

    pos_c = R_cw @ pos_w + t_cw

    # Normalize to get bearing vector
    norm = np.linalg.norm(pos_c)
    if norm < 1e-10:
        return None
    bearing = pos_c / norm

    # Convert to spherical coordinates
    # bearing = [x, y, z] where y is up
    latitude = -np.arcsin(bearing[1])  # range [-pi/2, pi/2]
    longitude = np.arctan2(bearing[0], bearing[2])  # range [-pi, pi]

    # Convert to pixel coordinates
    u = cols * (0.5 + longitude / (2.0 * np.pi))
    v = rows * (0.5 - latitude / np.pi)

    return (u, v)


def check_in_mask(u, v, mask):
    """
    Check if pixel (u, v) is in the mask.

    Args:
        u: x coordinate (float)
        v: y coordinate (float)
        mask: 2D numpy array (grayscale mask)

    Returns:
        bool: True if pixel is in mask (non-zero value)
    """
    h, w = mask.shape[:2]

    # Round to nearest pixel
    px = int(round(u))
    py = int(round(v))

    # Boundary check
    if px < 0 or px >= w or py < 0 or py >= h:
        return False

    # Check mask value (non-zero = in mask)
    return mask[py, px] > 0


def get_camera_world_pose(pose_cw):
    """
    Extract camera position and forward direction in world coordinates.

    Args:
        pose_cw: camera-to-world pose matrix (4x4)

    Returns:
        tuple: (position, forward_direction) both as ndarray(3,)
    """
    R_cw = pose_cw[:3, :3]
    t_cw = pose_cw[:3, 3]

    # Camera position in world frame
    R_wc = R_cw.T
    t_wc = -R_wc @ t_cw

    # Camera forward direction (Z-axis of camera in world frame)
    cam_forward_camera = np.array([0, 0, 1])
    cam_forward_world = R_wc @ cam_forward_camera

    return t_wc, cam_forward_world


def pixel_to_bearing_equirectangular(u, v, width, height):
    """
    Convert equirectangular pixel coordinates to bearing vector.

    Args:
        u: x coordinate
        v: y coordinate
        width: image width
        height: image height

    Returns:
        ndarray: bearing vector (3,) in camera frame
    """
    longitude = (u / width - 0.5) * 2.0 * np.pi
    latitude = -(v / height - 0.5) * np.pi

    bearing = np.array([
        np.cos(latitude) * np.sin(longitude),
        -np.sin(latitude),
        np.cos(latitude) * np.cos(longitude)
    ])

    return bearing
