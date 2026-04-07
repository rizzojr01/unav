#!/usr/bin/env python3
"""
SAM3 Floor Mask Generation

使用SAM3模型生成floor区域的mask
"""

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image


def load_sam3_model(device="cuda"):
    """
    加载SAM3模型

    Args:
        device: 设备(cuda/cpu)

    Returns:
        sam3_model: SAM3模型
        sam3_processor: SAM3处理器
    """
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print("加载SAM3模型...")
    sam3_model = build_sam3_image_model(
        device=device,
        eval_mode=True
    )
    sam3_processor = Sam3Processor(sam3_model)
    print("✅ SAM3模型加载完成")

    return sam3_model, sam3_processor


def generate_floor_masks(slice_paths, sam3_model, sam3_processor, device):
    """
    为所有切片生成floor mask

    Args:
        slice_paths: 切片图像路径列表
        sam3_model: SAM3模型
        sam3_processor: SAM3处理器
        device: 设备

    Returns:
        masks: 字典，key为文件名，value为mask数组
    """
    print("\n生成floor mask...")
    masks = {}

    for slice_path in tqdm(slice_paths, desc="生成floor mask"):
        # 读取图像
        try:
            img = Image.open(slice_path).convert("RGB")
        except Exception as e:
            print(f"  ⚠️  无法读取: {slice_path}, {e}")
            continue

        # SAM3推理
        with torch.no_grad():
            # Set image
            inference_state = sam3_processor.set_image(img)

            # Set text prompt
            output = sam3_processor.set_text_prompt(
                state=inference_state,
                prompt="floor"
            )

            # 获取masks
            pred_masks = output.get("masks")

            if pred_masks is not None and len(pred_masks) > 0:
                # 合并所有floor masks（如果有多个）
                mask = pred_masks[0].cpu().numpy()
                for m in pred_masks[1:]:
                    mask = np.logical_or(mask, m.cpu().numpy())
            else:
                # 没有检测到floor，创建空mask
                mask = np.zeros((img.height, img.width), dtype=bool)

        # 保存mask（使用文件名作为key）
        masks[Path(slice_path).name] = mask

    print(f"✅ 生成了 {len(masks)} 个floor mask")

    return masks
