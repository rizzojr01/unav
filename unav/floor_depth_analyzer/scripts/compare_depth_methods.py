"""
对比 Depth Anywhere 与 DA V3 cubemap 方法的深度估计结果
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path("/home/unav/Desktop/unav/unav/floor_depth_analyzer/output")

# 加载 Depth Anywhere 结果
da_rgb = cv2.imread(str(output_dir / "image0_rgb.png"))
da_rgb = cv2.cvtColor(da_rgb, cv2.COLOR_BGR2RGB)
da_depth = np.load(str(output_dir / "image0_depth.npy"))
da_depth_color = cv2.imread(str(output_dir / "image0_depth_color.png"))
da_depth_color = cv2.cvtColor(da_depth_color, cv2.COLOR_BGR2RGB)

# 加载原始 equirectangular 图像
orig_img_path = "/mnt/data/UNav-IO/temp/New_York_University/Tandon/4_floor/stella_vslam_dense/keyframes/image0.png"
orig_img = cv2.imread(orig_img_path)
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
orig_img_resized = cv2.resize(orig_img, (1024, 512))

# 创建可视化
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# 原始图像
axes[0].imshow(orig_img_resized)
axes[0].set_title(f"Original Equirectangular Image (resized to 1024x512)", fontsize=14)
axes[0].axis('off')

# Depth Anywhere 深度图
im1 = axes[1].imshow(da_depth, cmap='inferno')
axes[1].set_title(f"Depth Anywhere (UniFuse + Distillation)\nDepth range: {da_depth.min():.3f} - {da_depth.max():.3f} m", fontsize=14)
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='Depth (m)')

# 彩色深度图
axes[2].imshow(da_depth_color)
axes[2].set_title("Depth Anywhere - Color Visualization (INFERNO colormap)", fontsize=14)
axes[2].axis('off')

plt.tight_layout()
plt.savefig(output_dir / "depth_anywhere_result.png", dpi=150, bbox_inches='tight')
plt.close()

print("=" * 60)
print("Depth Anywhere 推理结果")
print("=" * 60)
print(f"输入图像: {orig_img_path}")
print(f"原始尺寸: {orig_img.shape}")
print(f"模型输入尺寸: 1024x512")
print(f"深度范围: {da_depth.min():.3f} - {da_depth.max():.3f} m")
print(f"深度均值: {da_depth.mean():.3f} m")
print(f"深度标准差: {da_depth.std():.3f} m")
print()
print(f"结果保存至: {output_dir / 'depth_anywhere_result.png'}")

# 创建详细统计可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 深度图
im1 = axes[0, 0].imshow(da_depth, cmap='inferno')
axes[0, 0].set_title("Depth Map (meters)", fontsize=12)
axes[0, 0].axis('off')
plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

# 深度直方图
axes[0, 1].hist(da_depth.flatten(), bins=100, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Depth (m)')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Depth Distribution', fontsize=12)
axes[0, 1].axvline(da_depth.mean(), color='red', linestyle='--', label=f'Mean: {da_depth.mean():.2f}m')
axes[0, 1].legend()

# 逆深度图 (disparity)
disparity = 1.0 / (da_depth + 1e-6)
im2 = axes[1, 0].imshow(disparity, cmap='magma')
axes[1, 0].set_title("Disparity (1/depth)", fontsize=12)
axes[1, 0].axis('off')
plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

# 原图叠加深度
axes[1, 1].imshow(da_rgb)
depth_overlay = axes[1, 1].imshow(da_depth, cmap='inferno', alpha=0.5)
axes[1, 1].set_title("RGB + Depth Overlay", fontsize=12)
axes[1, 1].axis('off')
plt.colorbar(depth_overlay, ax=axes[1, 1], fraction=0.046, pad=0.04)

plt.suptitle("Depth Anywhere Analysis", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / "depth_anywhere_analysis.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"分析结果保存至: {output_dir / 'depth_anywhere_analysis.png'}")
