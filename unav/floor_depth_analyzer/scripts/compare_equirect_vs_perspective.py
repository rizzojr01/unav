#!/usr/bin/env python3
"""
Depth Anything V3 Equirectangular vs Perspective Comparison

比较DA3在以下两种情况下的深度估计效果:
1. 直接对equirectangular图像估计深度，然后切片
2. 先切片成perspective图像，再分别估计深度

用于分析DA3在equirectangular图像上的表现
"""

import sys
sys.path.insert(0, "/home/unav/Desktop/unav")

import numpy as np
import cv2
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3
from unav.mapper.slicer import equirectangular_to_perspective


def load_da3_model(device="cuda"):
    """加载DA3模型"""
    print("加载 Depth Anything V3...")
    model = DepthAnything3.from_pretrained("depth-anything/da3-base")
    model = model.to(device=device)
    model.eval()
    print("✅ 模型加载完成")
    return model


def run_depth_estimation(model, image_paths):
    """
    运行DA3深度估计

    Args:
        model: DA3模型
        image_paths: 图像路径列表

    Returns:
        depths: 深度图列表 (N, H, W)
    """
    with torch.no_grad():
        prediction = model.inference(
            image_paths,
            extrinsics=None,
            intrinsics=None,
        )
    return prediction.depth


def slice_image_to_perspectives(
    equirect_img,
    yaw_angles,
    pitch_angle,
    fov,
    patch_size=14,
):
    """
    将equirectangular图像切成perspective切片

    Args:
        equirect_img: equirectangular图像 (H, W, 3)
        yaw_angles: yaw角度列表
        pitch_angle: pitch角度
        fov: 视场角
        patch_size: 切片尺寸必须是patch_size的倍数

    Returns:
        slices: perspective切片列表
        slice_params: (out_w, out_h)
    """
    h_eq, w_eq = equirect_img.shape[:2]

    # 计算输出尺寸
    out_w = int((fov / 360.0) * w_eq)
    out_h = int(out_w * 9 / 16)
    out_w = (out_w // patch_size) * patch_size
    out_h = (out_h // patch_size) * patch_size

    slices = []
    for yaw in yaw_angles:
        slice_img = equirectangular_to_perspective(
            equirect_img,
            fov_deg=fov,
            yaw_deg=yaw,
            pitch_deg=pitch_angle,
            width=out_w,
            height=out_h
        )
        slices.append(slice_img)

    return slices, (out_w, out_h)


def slice_depth_to_perspectives(
    equirect_depth,
    yaw_angles,
    pitch_angle,
    fov,
    out_w,
    out_h,
):
    """
    将equirectangular深度图切成perspective切片

    Args:
        equirect_depth: equirectangular深度图 (H, W)
        yaw_angles: yaw角度列表
        pitch_angle: pitch角度
        fov: 视场角
        out_w, out_h: 输出尺寸

    Returns:
        depth_slices: perspective深度切片列表
    """
    # 将深度图扩展为3通道以便使用相同的切片函数
    depth_3ch = np.stack([equirect_depth] * 3, axis=-1)

    depth_slices = []
    for yaw in yaw_angles:
        slice_depth = equirectangular_to_perspective(
            depth_3ch.astype(np.float32),
            fov_deg=fov,
            yaw_deg=yaw,
            pitch_deg=pitch_angle,
            width=out_w,
            height=out_h
        )
        # 取第一个通道
        depth_slices.append(slice_depth[:, :, 0])

    return depth_slices


def compute_depth_metrics(depth1, depth2, name1="Depth1", name2="Depth2"):
    """
    计算两个深度图之间的差异指标

    Args:
        depth1, depth2: 两个深度图
        name1, name2: 名称

    Returns:
        metrics: 指标字典
    """
    # 确保形状一致
    if depth1.shape != depth2.shape:
        # 调整大小
        depth2 = cv2.resize(depth2, (depth1.shape[1], depth1.shape[0]),
                           interpolation=cv2.INTER_LINEAR)

    # 归一化到相同范围进行比较
    d1_norm = (depth1 - depth1.min()) / (depth1.max() - depth1.min() + 1e-8)
    d2_norm = (depth2 - depth2.min()) / (depth2.max() - depth2.min() + 1e-8)

    # 计算差异
    diff = np.abs(d1_norm - d2_norm)

    metrics = {
        'mae': np.mean(diff),
        'rmse': np.sqrt(np.mean(diff ** 2)),
        'max_diff': np.max(diff),
        'median_diff': np.median(diff),
        'diff_map': diff,
    }

    return metrics


def visualize_comparison(
    equirect_img,
    equirect_depth,
    sliced_depths_from_whole,
    sliced_depths_from_persp,
    perspective_imgs,
    yaw_angles,
    output_dir,
):
    """
    可视化比较结果
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_slices = len(yaw_angles)

    # 1. 保存equirectangular原图和深度
    fig, axes = plt.subplots(2, 1, figsize=(20, 10))

    axes[0].imshow(equirect_img)
    axes[0].set_title("Equirectangular Image")
    axes[0].axis('off')

    im = axes[1].imshow(equirect_depth, cmap='turbo')
    axes[1].set_title("Depth from Equirectangular (DA3)")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.02)

    plt.tight_layout()
    plt.savefig(output_path / "01_equirectangular_overview.png", dpi=150)
    plt.close()

    # 2. 比较每个perspective切片
    fig, axes = plt.subplots(n_slices, 5, figsize=(25, 5 * n_slices))

    for i, yaw in enumerate(yaw_angles):
        # Perspective image
        axes[i, 0].imshow(perspective_imgs[i])
        axes[i, 0].set_title(f"Perspective (yaw={yaw}°)")
        axes[i, 0].axis('off')

        # Depth from perspective
        im1 = axes[i, 1].imshow(sliced_depths_from_persp[i], cmap='turbo')
        axes[i, 1].set_title("Depth from Perspective")
        axes[i, 1].axis('off')

        # Depth sliced from whole
        im2 = axes[i, 2].imshow(sliced_depths_from_whole[i], cmap='turbo')
        axes[i, 2].set_title("Depth Sliced from Whole")
        axes[i, 2].axis('off')

        # Compute metrics and show difference
        metrics = compute_depth_metrics(
            sliced_depths_from_persp[i],
            sliced_depths_from_whole[i]
        )

        im3 = axes[i, 3].imshow(metrics['diff_map'], cmap='hot', vmin=0, vmax=0.5)
        axes[i, 3].set_title(f"Difference (MAE={metrics['mae']:.4f})")
        axes[i, 3].axis('off')

        # Metrics text
        axes[i, 4].text(0.1, 0.7, f"MAE: {metrics['mae']:.4f}", fontsize=12)
        axes[i, 4].text(0.1, 0.5, f"RMSE: {metrics['rmse']:.4f}", fontsize=12)
        axes[i, 4].text(0.1, 0.3, f"Max Diff: {metrics['max_diff']:.4f}", fontsize=12)
        axes[i, 4].text(0.1, 0.1, f"Median Diff: {metrics['median_diff']:.4f}", fontsize=12)
        axes[i, 4].set_xlim(0, 1)
        axes[i, 4].set_ylim(0, 1)
        axes[i, 4].axis('off')
        axes[i, 4].set_title("Metrics")

    plt.tight_layout()
    plt.savefig(output_path / "02_perspective_comparison.png", dpi=150)
    plt.close()

    # 3. 汇总统计
    all_metrics = []
    for i, yaw in enumerate(yaw_angles):
        metrics = compute_depth_metrics(
            sliced_depths_from_persp[i],
            sliced_depths_from_whole[i]
        )
        all_metrics.append({
            'yaw': yaw,
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'max_diff': metrics['max_diff'],
            'median_diff': metrics['median_diff'],
        })

    # 绘制汇总图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    yaws = [m['yaw'] for m in all_metrics]
    maes = [m['mae'] for m in all_metrics]
    rmses = [m['rmse'] for m in all_metrics]
    max_diffs = [m['max_diff'] for m in all_metrics]
    median_diffs = [m['median_diff'] for m in all_metrics]

    axes[0, 0].bar(yaws, maes, width=20)
    axes[0, 0].set_xlabel("Yaw Angle (°)")
    axes[0, 0].set_ylabel("MAE")
    axes[0, 0].set_title("Mean Absolute Error by Yaw")

    axes[0, 1].bar(yaws, rmses, width=20)
    axes[0, 1].set_xlabel("Yaw Angle (°)")
    axes[0, 1].set_ylabel("RMSE")
    axes[0, 1].set_title("RMSE by Yaw")

    axes[1, 0].bar(yaws, max_diffs, width=20)
    axes[1, 0].set_xlabel("Yaw Angle (°)")
    axes[1, 0].set_ylabel("Max Diff")
    axes[1, 0].set_title("Max Difference by Yaw")

    axes[1, 1].bar(yaws, median_diffs, width=20)
    axes[1, 1].set_xlabel("Yaw Angle (°)")
    axes[1, 1].set_ylabel("Median Diff")
    axes[1, 1].set_title("Median Difference by Yaw")

    plt.suptitle("Comparison: Depth from Perspective vs Sliced from Whole", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / "03_summary_statistics.png", dpi=150)
    plt.close()

    # 打印汇总
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    print(f"Average MAE: {np.mean(maes):.4f}")
    print(f"Average RMSE: {np.mean(rmses):.4f}")
    print(f"Average Max Diff: {np.mean(max_diffs):.4f}")
    print(f"Average Median Diff: {np.mean(median_diffs):.4f}")
    print()

    return all_metrics


def main():
    """主函数"""
    # 配置
    input_image = "/mnt/data/UNav-IO/temp/New_York_University/Tandon/4_floor/stella_vslam_dense/keyframes/image0.png"
    output_dir = "/home/unav/Desktop/unav/unav/floor_depth_analyzer/output/equirect_vs_persp_comparison"

    # 切片参数
    yaw_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    pitch_angle = 0
    fov = 90

    print("="*80)
    print("DA3 Equirectangular vs Perspective Comparison")
    print("="*80)
    print(f"\nInput image: {input_image}")
    print(f"Output dir: {output_dir}")
    print(f"Yaw angles: {yaw_angles}")
    print(f"Pitch angle: {pitch_angle}")
    print(f"FOV: {fov}")
    print()

    # 1. 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_da3_model(device)

    # 2. 加载equirectangular图像
    print("\n加载equirectangular图像...")
    equirect_img = cv2.imread(input_image)
    equirect_img = cv2.cvtColor(equirect_img, cv2.COLOR_BGR2RGB)
    print(f"  图像尺寸: {equirect_img.shape}")

    # 3. 对equirectangular图像直接运行DA3
    print("\n[Step 1] 对equirectangular图像运行DA3...")
    equirect_depths = run_depth_estimation(model, [input_image])
    equirect_depth = equirect_depths[0]  # (H, W)
    print(f"  深度图尺寸: {equirect_depth.shape}")
    print(f"  深度范围: [{equirect_depth.min():.4f}, {equirect_depth.max():.4f}]")

    # 4. 切片equirectangular图像
    print("\n[Step 2] 切片equirectangular图像...")
    perspective_imgs, (out_w, out_h) = slice_image_to_perspectives(
        equirect_img, yaw_angles, pitch_angle, fov
    )
    print(f"  生成 {len(perspective_imgs)} 个perspective切片")
    print(f"  切片尺寸: {out_w} x {out_h}")

    # 5. 保存perspective切片临时文件
    print("\n[Step 3] 保存perspective切片临时文件...")
    temp_dir = Path(output_dir) / "temp_slices"
    temp_dir.mkdir(parents=True, exist_ok=True)

    slice_paths = []
    for i, (img, yaw) in enumerate(zip(perspective_imgs, yaw_angles)):
        path = temp_dir / f"slice_yaw{yaw:03d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        slice_paths.append(str(path))

    # 6. 对perspective切片运行DA3
    print("\n[Step 4] 对perspective切片运行DA3...")
    persp_depths = run_depth_estimation(model, slice_paths)
    print(f"  获得 {len(persp_depths)} 个深度图")

    # 7. 将equirectangular深度切片
    print("\n[Step 5] 将equirectangular深度切片...")
    sliced_depths_from_whole = slice_depth_to_perspectives(
        equirect_depth, yaw_angles, pitch_angle, fov, out_w, out_h
    )
    print(f"  切片得到 {len(sliced_depths_from_whole)} 个深度图")

    # 8. 可视化比较
    print("\n[Step 6] 可视化比较...")
    metrics = visualize_comparison(
        equirect_img,
        equirect_depth,
        sliced_depths_from_whole,
        list(persp_depths),
        perspective_imgs,
        yaw_angles,
        output_dir,
    )

    print(f"\n✅ 比较完成！结果保存在: {output_dir}")

    return metrics


if __name__ == "__main__":
    main()
