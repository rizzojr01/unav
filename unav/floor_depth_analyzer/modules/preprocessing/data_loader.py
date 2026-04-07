#!/usr/bin/env python3
"""
Data Loader

读取关键帧图片和轨迹数据
"""

import sys
sys.path.insert(0, "/home/unav/Desktop/unav")

from unav.mapper.tools.slam.read_data import get_keyframe_image_list, read_keyframe_trajectory


def load_keyframes_and_trajectory(keyframe_dir, trajectory_file, num_images=None):
    """
    读取关键帧图片列表和轨迹

    Args:
        keyframe_dir: 关键帧目录
        trajectory_file: 轨迹文件
        num_images: 读取的图片数量（None表示全部）

    Returns:
        data: 数据字典
            - image_list: 图片文件名列表
            - poses: 相机位姿字典 {image_name: T_cw (4x4)}
    """
    print("="*80)
    print("读取关键帧和轨迹")
    print("="*80)
    print(f"\n关键帧目录: {keyframe_dir}")
    print(f"轨迹文件: {trajectory_file}")

    # 读取图片列表
    image_list = get_keyframe_image_list(keyframe_dir)
    if num_images is not None:
        image_list = image_list[:num_images]

    print(f"\n选择 {len(image_list)} 张全景图")

    # 读取位姿
    poses = read_keyframe_trajectory(trajectory_file, image_list)
    print(f"读取了 {len(poses)} 个相机位姿")

    return {
        'image_list': image_list,
        'poses': poses,
    }
