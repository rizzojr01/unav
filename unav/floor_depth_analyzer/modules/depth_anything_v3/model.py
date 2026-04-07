#!/usr/bin/env python3
"""
Depth Anything V3 Model

DA3模型加载和推理（纯DA3功能）
"""

import torch
from depth_anything_3.api import DepthAnything3


def load_da3_model(model_name="depth-anything/da3-base", device="cuda"):
    """
    加载DA3模型

    Args:
        model_name: 模型名称
        device: 设备(cuda/cpu)

    Returns:
        model: DA3模型
    """
    print(f"加载 Depth Anything V3 (device={device})...")
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device=device)
    model.eval()
    print("✅ 模型加载完成")
    return model


def run_da3_inference(
    model,
    image_paths,
    extrinsics,
    intrinsics,
    export_dir=None,
    export_format="glb",
    conf_thresh_percentile=40.0,
    num_max_points=5_000_000,
    show_cameras=True,
):
    """
    运行DA3推理

    Args:
        model: DA3模型
        image_paths: 图片路径列表
        extrinsics: 相机外参 (N, 4, 4)
        intrinsics: 相机内参 (N, 3, 3)
        export_dir: 导出目录
        export_format: 导出格式
        conf_thresh_percentile: 置信度阈值百分位
        num_max_points: 最大点数
        show_cameras: 是否显示相机

    Returns:
        prediction: DA3预测结果
    """
    print(f"\n开始DA3推理...")
    print(f"  输入图片数: {len(image_paths)}")
    print(f"  相机位姿: {extrinsics.shape}")
    print(f"  相机内参: {intrinsics.shape}")
    if export_dir:
        print(f"  导出目录: {export_dir}")
        print(f"  导出格式: {export_format}")
    print()

    with torch.no_grad():
        # 如果不需要导出，export_format 必须是空字符串而不是 None
        actual_export_format = export_format if export_dir else ""
        prediction = model.inference(
            image_paths,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            infer_gs=False,
            export_dir=str(export_dir) if export_dir else None,
            export_format=actual_export_format,
            conf_thresh_percentile=conf_thresh_percentile,
            num_max_points=num_max_points,
            show_cameras=show_cameras,
        )

    print("\n✅ DA3推理完成！")

    return prediction
