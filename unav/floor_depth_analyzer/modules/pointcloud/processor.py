#!/usr/bin/env python3
"""
Point Cloud Processing

深度图转点云的处理函数
"""

import numpy as np
import cv2


def depth_to_pointcloud(depth, conf, intrinsic, extrinsic, conf_thresh=1.0):
    """
    将深度图转换为3D点云（不使用floor mask）

    Args:
        depth: 深度图
        conf: 置信度图
        intrinsic: 相机内参矩阵
        extrinsic: 相机外参矩阵
        conf_thresh: 置信度阈值

    Returns:
        points_world: 世界坐标系中的3D点 (N, 3)
    """
    h, w = depth.shape

    # 过滤低置信度点
    mask = conf > conf_thresh

    # 生成像素坐标
    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # 应用mask
    u = u[mask]
    v = v[mask]
    d = depth[mask]

    if len(u) == 0:
        return np.array([]).reshape(0, 3)

    # 反投影到相机坐标系
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    x_cam = (u - cx) * d / fx
    y_cam = (v - cy) * d / fy
    z_cam = d

    # 相机坐标系中的点
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

    # 转换到世界坐标系
    T_cw = np.linalg.inv(extrinsic)
    R = T_cw[:3, :3]
    t = T_cw[:3, 3]

    points_world = (R @ points_cam.T).T + t

    return points_world


def depth_to_pointcloud_with_mask(depth, conf, floor_mask, intrinsic, extrinsic, conf_thresh=1.0):
    """
    将深度图转换为3D点云（使用floor mask过滤）

    Args:
        depth: 深度图
        conf: 置信度图
        floor_mask: Floor区域mask
        intrinsic: 相机内参矩阵
        extrinsic: 相机外参矩阵
        conf_thresh: 置信度阈值

    Returns:
        points_world: 世界坐标系中的3D点 (N, 3)
    """
    h, w = depth.shape

    # 组合mask：置信度 + floor mask
    # 需要将floor_mask调整到depth的尺寸
    # 处理mask可能是多维的情况
    if floor_mask.ndim > 2:
        floor_mask = floor_mask.squeeze()

    if floor_mask.shape != depth.shape:
        floor_mask_resized = cv2.resize(
            floor_mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    else:
        floor_mask_resized = floor_mask

    # 组合mask
    combined_mask = (conf > conf_thresh) & floor_mask_resized

    # 生成像素坐标
    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # 应用mask
    u = u[combined_mask]
    v = v[combined_mask]
    d = depth[combined_mask]

    if len(u) == 0:
        return np.array([]).reshape(0, 3)

    # 反投影到相机坐标系
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    x_cam = (u - cx) * d / fx
    y_cam = (v - cy) * d / fy
    z_cam = d

    # 相机坐标系中的点
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

    # 转换到世界坐标系
    T_cw = np.linalg.inv(extrinsic)
    R = T_cw[:3, :3]
    t = T_cw[:3, 3]

    points_world = (R @ points_cam.T).T + t

    return points_world


def save_pointcloud_glb(points, output_path, colors=None):
    """
    保存点云为GLB文件

    Args:
        points: 点云数组 (N, 3)
        output_path: 输出路径
        colors: 颜色数组 (N, 3) 或 (N, 4)，可选
    """
    import trimesh

    if colors is None:
        # 默认使用蓝色
        colors = np.ones((len(points), 4), dtype=np.uint8)
        colors[:, 0] = 100  # R
        colors[:, 1] = 150  # G
        colors[:, 2] = 255  # B
        colors[:, 3] = 255  # A

    # 创建点云
    cloud = trimesh.PointCloud(vertices=points, colors=colors)

    # 导出为GLB
    cloud.export(str(output_path), file_type='glb')
    print(f"✅ 点云已保存: {output_path}")

    return output_path
