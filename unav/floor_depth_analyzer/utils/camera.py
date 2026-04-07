#!/usr/bin/env python3
"""
Camera Parameter Utilities

相机参数计算工具
"""

import numpy as np


def compute_slice_camera_params(T_cw, yaw, pitch, fov, width, height):
    """
    计算切片的相机外参和内参

    Args:
        T_cw: 全景相机的外参矩阵 (4x4)
        yaw: Yaw角度（度）
        pitch: Pitch角度（度）
        fov: 视场角（度）
        width: 切片宽度
        height: 切片高度

    Returns:
        extrinsic: 切片相机外参 (4x4)
        intrinsic: 切片相机内参 (3x3)
    """
    # 计算旋转矩阵
    yaw_rad = np.deg2rad(yaw)
    pitch_rad = np.deg2rad(pitch)

    # Yaw旋转（绕Z轴）
    R_yaw = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])

    # Pitch旋转（绕Y轴）
    R_pitch = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])

    # 组合旋转
    R_slice = R_yaw @ R_pitch

    # 计算切片相机的外参
    R_cw = T_cw[:3, :3]
    t_cw = T_cw[:3, 3]

    R_slice_world = R_cw @ R_slice
    t_slice_world = t_cw

    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R_slice_world
    extrinsic[:3, 3] = t_slice_world

    # 计算内参
    focal_length = width / (2 * np.tan(np.deg2rad(fov) / 2))

    intrinsic = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ])

    return extrinsic, intrinsic
