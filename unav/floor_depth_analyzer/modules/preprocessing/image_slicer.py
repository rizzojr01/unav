#!/usr/bin/env python3
"""
Image Slicer

全景图切片功能
"""

import sys
sys.path.insert(0, "/home/unav/Desktop/unav")

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

from unav.mapper.slicer import equirectangular_to_perspective


def slice_equirectangular_images(
    image_list,
    keyframe_dir,
    poses,
    output_dir,
    yaw_angles=[0, 45, 90, 135, 180, 225, 270, 315],
    pitch_angles=[0, -20],
    fov=90,
    patch_size=14,
):
    """
    切片全景图

    Args:
        image_list: 图片文件名列表
        keyframe_dir: 关键帧目录
        poses: 相机位姿字典 {image_name: T_cw}
        output_dir: 输出目录
        yaw_angles: Yaw角度列表
        pitch_angles: Pitch角度列表
        fov: 视场角
        patch_size: 切片尺寸必须是patch_size的倍数

    Returns:
        slice_info: 切片信息字典
            - slice_paths: 切片路径列表
            - camera_poses: 对应的相机位姿列表
            - slice_params: 切片参数列表 [(yaw, pitch, fov), ...]
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("切片全景图")
    print("="*80)
    print(f"\n配置:")
    print(f"  图片数: {len(image_list)}")
    print(f"  Yaw角度: {yaw_angles}")
    print(f"  Pitch角度: {pitch_angles}")
    print(f"  每张图切片数: {len(yaw_angles) * len(pitch_angles)}")
    print(f"  总切片数: {len(image_list) * len(yaw_angles) * len(pitch_angles)}")
    print()

    # 计算切片尺寸
    first_img = cv2.imread(str(Path(keyframe_dir) / image_list[0]))
    h_eq, w_eq = first_img.shape[:2]

    out_w = int((fov / 360.0) * w_eq)
    out_h = int(out_w * 9 / 16)
    out_w = (out_w // patch_size) * patch_size
    out_h = (out_h // patch_size) * patch_size

    print(f"全景图尺寸: {w_eq} x {h_eq}")
    print(f"切片尺寸: {out_w} x {out_h}")
    print()

    slice_paths = []
    camera_poses = []
    slice_params = []

    for img_name in tqdm(image_list, desc="切片全景图"):
        img_path = Path(keyframe_dir) / img_name

        equirect_img = cv2.imread(str(img_path))
        if equirect_img is None:
            continue

        equirect_img = cv2.cvtColor(equirect_img, cv2.COLOR_BGR2RGB)
        T_cw = poses[img_name]

        # 对每个pitch角度生成切片
        for pitch in pitch_angles:
            for yaw in yaw_angles:
                slice_img = equirectangular_to_perspective(
                    equirect_img,
                    fov_deg=fov,
                    yaw_deg=yaw,
                    pitch_deg=pitch,
                    width=out_w,
                    height=out_h
                )

                stem = Path(img_name).stem
                slice_name = f"{stem}_yaw{yaw:03d}_pitch{int(pitch):+03d}.png"
                slice_path = output_path / slice_name

                slice_img_bgr = cv2.cvtColor(slice_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(slice_path), slice_img_bgr)

                slice_paths.append(str(slice_path))
                camera_poses.append(T_cw)
                slice_params.append((yaw, pitch, fov, out_w, out_h))

    print(f"\n✅ 生成了 {len(slice_paths)} 个切片")

    return {
        'slice_paths': slice_paths,
        'camera_poses': camera_poses,
        'slice_params': slice_params,
        'output_dir': str(output_path),
    }
