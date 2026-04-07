#!/usr/bin/env python3
"""
Floor Reconstruction Pipeline

完整的floor点云重建流程:
1. 切片全景图
2. DA3深度推理
3. SAM3 floor mask推理
4. 用mask过滤深度，只保留floor区域
5. 生成floor点云并导出GLB
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, "/home/unav/Desktop/unav")

from unav.floor_depth_analyzer.modules.preprocessing import (
    load_keyframes_and_trajectory,
    slice_equirectangular_images
)
from unav.floor_depth_analyzer.modules.depth_anything_v3 import (
    load_da3_model,
    run_da3_inference
)
from unav.floor_depth_analyzer.modules.sam3 import (
    load_sam3_model,
    generate_floor_masks
)
from unav.floor_depth_analyzer.modules.pointcloud import (
    depth_to_pointcloud_with_mask,
    save_pointcloud_glb
)
from unav.floor_depth_analyzer.utils import compute_slice_camera_params


def generate_floor_pointcloud(
    depths,
    confs,
    floor_masks,
    slice_paths,
    extrinsics,
    intrinsics,
    conf_thresh=1.5,
):
    """
    从深度和floor mask生成floor点云

    Args:
        depths: 深度数组 (N, H, W)
        confs: 置信度数组 (N, H, W)
        floor_masks: floor mask字典
        slice_paths: 切片路径列表
        extrinsics: 相机外参
        intrinsics: 相机内参
        conf_thresh: 置信度阈值

    Returns:
        all_points: floor点云 (M, 3)
    """
    print("\n" + "="*80)
    print("生成Floor点云")
    print("="*80)

    all_points = []
    valid_slices = 0

    for i in tqdm(range(len(depths)), desc="处理切片"):
        slice_name = Path(slice_paths[i]).name

        if slice_name not in floor_masks:
            continue

        floor_mask = floor_masks[slice_name]

        # 检查mask是否有floor区域
        if not floor_mask.any():
            continue

        # 使用floor mask过滤深度生成点云
        points = depth_to_pointcloud_with_mask(
            depth=depths[i],
            conf=confs[i],
            floor_mask=floor_mask,
            intrinsic=intrinsics[i],
            extrinsic=extrinsics[i],
            conf_thresh=conf_thresh,
        )

        if len(points) > 0:
            all_points.append(points)
            valid_slices += 1

    if len(all_points) == 0:
        raise ValueError("没有生成任何floor点云！请检查SAM3 floor mask。")

    all_points = np.vstack(all_points)

    print(f"\n✅ Floor点云生成完成:")
    print(f"  有效切片数: {valid_slices}/{len(depths)}")
    print(f"  总点数: {len(all_points):,}")

    return all_points


def main():
    parser = argparse.ArgumentParser(
        description='Floor重建Pipeline（DA3深度 + SAM3 floor mask）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/run_reconstruction.py \\
      --keyframe_dir /path/to/keyframes \\
      --trajectory_file /path/to/trajectory.txt \\
      --output_dir /tmp/floor_output \\
      --num_images 10
        """
    )

    parser.add_argument('--keyframe_dir', type=str, required=True,
                        help='关键帧目录')
    parser.add_argument('--trajectory_file', type=str, required=True,
                        help='相机轨迹文件')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--num_images', type=int, default=10,
                        help='处理的图片数量（默认10）')
    parser.add_argument('--yaw_angles', type=int, nargs='+',
                        default=[0, 45, 90, 135, 180, 225, 270, 315],
                        help='Yaw角度列表')
    parser.add_argument('--pitch_angles', type=int, nargs='+',
                        default=[0, -20],
                        help='Pitch角度列表')
    parser.add_argument('--fov', type=float, default=90,
                        help='视场角（默认90）')
    parser.add_argument('--conf_thresh', type=float, default=1.5,
                        help='深度置信度阈值（默认1.5）')

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*80)
    print("Floor重建Pipeline")
    print("="*80)
    print(f"\n配置:")
    print(f"  关键帧目录: {args.keyframe_dir}")
    print(f"  轨迹文件: {args.trajectory_file}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  图片数: {args.num_images}")
    print(f"  Yaw角度: {args.yaw_angles}")
    print(f"  Pitch角度: {args.pitch_angles}")
    print(f"  设备: {device}")
    print()

    # ========================================================================
    # Step 1: 读取数据
    # ========================================================================
    print("="*80)
    print("Step 1: 读取关键帧和轨迹")
    print("="*80)

    data = load_keyframes_and_trajectory(
        args.keyframe_dir,
        args.trajectory_file,
        num_images=args.num_images
    )

    # ========================================================================
    # Step 2: 切片全景图
    # ========================================================================
    print("\n" + "="*80)
    print("Step 2: 切片全景图")
    print("="*80)

    slices_dir = output_path / "slices"
    slice_info = slice_equirectangular_images(
        image_list=data['image_list'],
        keyframe_dir=args.keyframe_dir,
        poses=data['poses'],
        output_dir=slices_dir,
        yaw_angles=args.yaw_angles,
        pitch_angles=args.pitch_angles,
        fov=args.fov,
    )

    # ========================================================================
    # Step 3: 计算相机参数
    # ========================================================================
    print("\n" + "="*80)
    print("Step 3: 计算相机参数")
    print("="*80)

    extrinsics = []
    intrinsics = []

    for T_cw, (yaw, pitch, fov, width, height) in zip(
        slice_info['camera_poses'],
        slice_info['slice_params']
    ):
        extrinsic, intrinsic = compute_slice_camera_params(
            T_cw, yaw, pitch, fov, width, height
        )
        extrinsics.append(extrinsic)
        intrinsics.append(intrinsic)

    extrinsics = np.stack(extrinsics, axis=0)
    intrinsics = np.stack(intrinsics, axis=0)

    print(f"\n✅ 计算了 {len(extrinsics)} 个相机参数")

    # ========================================================================
    # Step 4: DA3深度推理
    # ========================================================================
    print("\n" + "="*80)
    print("Step 4: DA3深度推理")
    print("="*80)

    da3_model = load_da3_model(device=device)

    # 不导出GLB，只获取深度
    prediction = run_da3_inference(
        model=da3_model,
        image_paths=slice_info['slice_paths'],
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        export_dir=None,  # 不导出
        export_format=None,
    )

    # 获取深度和置信度
    depths = prediction.depth
    if torch.is_tensor(depths):
        depths = depths.cpu().numpy()

    confs = prediction.conf
    if torch.is_tensor(confs):
        confs = confs.cpu().numpy()

    print(f"\n深度数据:")
    print(f"  形状: {depths.shape}")
    print(f"  范围: [{depths.min():.4f}, {depths.max():.4f}]")

    # 释放DA3模型显存
    del da3_model
    torch.cuda.empty_cache()

    # ========================================================================
    # Step 5: SAM3 Floor Mask推理
    # ========================================================================
    print("\n" + "="*80)
    print("Step 5: SAM3 Floor Mask推理")
    print("="*80)

    sam3_model, sam3_processor = load_sam3_model(device=device)
    floor_masks = generate_floor_masks(
        slice_info['slice_paths'],
        sam3_model,
        sam3_processor,
        device
    )

    # 释放SAM3模型显存
    del sam3_model, sam3_processor
    torch.cuda.empty_cache()

    # ========================================================================
    # Step 6: 生成Floor点云
    # ========================================================================
    floor_points = generate_floor_pointcloud(
        depths=depths,
        confs=confs,
        floor_masks=floor_masks,
        slice_paths=slice_info['slice_paths'],
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        conf_thresh=args.conf_thresh,
    )

    # ========================================================================
    # Step 7: 保存结果
    # ========================================================================
    print("\n" + "="*80)
    print("Step 7: 保存结果")
    print("="*80)

    # 保存floor点云为GLB
    glb_path = output_path / "floor_pointcloud.glb"
    save_pointcloud_glb(floor_points, glb_path)

    # 保存点云数据
    np.save(output_path / "floor_points.npy", floor_points)
    print(f"✅ 点云数据: {output_path / 'floor_points.npy'}")

    # 保存深度和相机参数（便于后续处理）
    np.save(output_path / "all_depths.npy", depths)
    np.save(output_path / "all_confs.npy", confs)
    np.save(output_path / "slice_paths.npy", np.array(slice_info['slice_paths']))
    np.save(output_path / "extrinsics.npy", extrinsics)
    np.save(output_path / "intrinsics.npy", intrinsics)

    # 保存floor masks
    np.save(output_path / "floor_masks.npy", floor_masks)

    print(f"✅ 其他数据已保存到: {output_path}")

    # ========================================================================
    # 统计
    # ========================================================================
    print("\n" + "="*80)
    print("重建结果统计")
    print("="*80)

    print(f"\n输入:")
    print(f"  全景图数: {args.num_images}")
    print(f"  切片数: {len(slice_info['slice_paths'])}")

    print(f"\n深度:")
    print(f"  范围: [{depths.min():.4f}, {depths.max():.4f}]")
    print(f"  均值: {depths.mean():.4f}")

    print(f"\nFloor点云:")
    print(f"  总点数: {len(floor_points):,}")
    print(f"  X范围: [{floor_points[:, 0].min():.2f}, {floor_points[:, 0].max():.2f}]")
    print(f"  Y范围: [{floor_points[:, 1].min():.2f}, {floor_points[:, 1].max():.2f}]")
    print(f"  Z范围: [{floor_points[:, 2].min():.2f}, {floor_points[:, 2].max():.2f}]")

    print("\n" + "="*80)
    print("完成！")
    print("="*80)
    print(f"\n查看结果:")
    print(f"  Floor点云 (GLB): {glb_path}")
    print(f"\n在线查看GLB:")
    print(f"  https://3dviewer.net/")
    print()

    sys.exit(0)


if __name__ == "__main__":
    main()
