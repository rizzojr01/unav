"""
Database utilities for loading SLAM data from SQLite.
"""

import sqlite3
import numpy as np


def load_keyframe_poses(db_path):
    """
    Load keyframe poses from SQLite database.

    Args:
        db_path: Path to SQLite3 database file

    Returns:
        dict: Dictionary mapping keyframe ID to pose data
              {kf_id: {'pose_cw': 4x4 matrix, 'cols': int, 'rows': int}}
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get camera info (cols, rows)
    cam_info = cursor.execute("SELECT cols, rows FROM cameras LIMIT 1").fetchone()
    cam_cols, cam_rows = cam_info[0], cam_info[1]
    print(f"  Camera resolution: {cam_cols}x{cam_rows}")

    keyframes = {}
    rows = cursor.execute("SELECT id, pose_cw FROM keyframes").fetchall()

    for row in rows:
        kf_id = row[0]
        pose_cw = np.frombuffer(row[1], dtype=np.float64).reshape(4, 4).T
        keyframes[kf_id] = {
            'pose_cw': pose_cw,
            'cols': cam_cols,
            'rows': cam_rows
        }

    conn.close()
    return keyframes


def load_dense_points(db_path):
    """
    Load dense points from SQLite database.

    Args:
        db_path: Path to SQLite3 database file

    Returns:
        list: List of point dictionaries
              [{'id': int, 'pos_w': ndarray(3,), 'color': ndarray(3,), 'ref_keyfrm_id': int}, ...]
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    points = []
    rows = cursor.execute("SELECT id, pos_w, color, ref_keyfrm FROM dense_points").fetchall()

    for row in rows:
        point_id = row[0]
        pos_w = np.frombuffer(row[1], dtype=np.float64)
        color = np.frombuffer(row[2], dtype=np.uint8)
        ref_keyfrm_id = row[3]

        points.append({
            'id': point_id,
            'pos_w': pos_w,
            'color': color,  # BGR format
            'ref_keyfrm_id': ref_keyfrm_id
        })

    conn.close()
    return points
