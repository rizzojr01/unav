#!/usr/bin/env python3
"""
Equirectangular Depth Estimation via Cubemap + Depth Anything V3

仿照 Depth Anywhere 的方法：
1. 将 equirectangular 图像转换为 6-face cubemap
2. 对每个 cube face 使用 Depth Anything V3 估计深度
3. 将 cube 深度融合回 equirectangular 格式

这样可以避免 DA V3 直接处理 equirectangular 时的畸变问题
"""

import sys
sys.path.insert(0, "/home/unav/Desktop/unav")

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from depth_anything_3.api import DepthAnything3


# ==================== Equirec2Cube (from Depth Anywhere) ====================
class Equirec2Cube(nn.Module):
    """将 equirectangular 图像转换为 6-face cubemap"""
    def __init__(self, cube_dim, equ_h, FoV=90.0):
        super().__init__()
        self.cube_dim = cube_dim
        self.equ_h = equ_h
        self.equ_w = equ_h * 2
        self.FoV = FoV / 180.0 * np.pi
        self.r_lst = np.array([
            [0, -180.0, 0],
            [90.0, 0, 0],
            [0, 0, 0],
            [0, 90, 0],
            [0, -90, 0],
            [-90, 0, 0]
        ], np.float32) / 180.0 * np.pi
        self.R_lst = [cv2.Rodrigues(x)[0] for x in self.r_lst]
        grids = self._getCubeGrid()

        for i, grid in enumerate(grids):
            self.register_buffer('grid_%d'%i, grid)

    def _getCubeGrid(self):
        f = 0.5 * self.cube_dim / np.tan(0.5 * self.FoV)
        cx = (self.cube_dim - 1) / 2
        cy = cx
        x = np.tile(np.arange(self.cube_dim)[None, ..., None], [self.cube_dim, 1, 1])
        y = np.tile(np.arange(self.cube_dim)[..., None, None], [1, self.cube_dim, 1])
        ones = np.ones_like(x)
        xyz = np.concatenate([x, y, ones], axis=-1)
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ], np.float32)
        xyz = xyz @ np.linalg.inv(K).T
        xyz /= np.linalg.norm(xyz, axis=-1, keepdims=True)
        grids = []
        for _, R in enumerate(self.R_lst):
            tmp = xyz @ R
            lon = np.arctan2(tmp[..., 0:1], tmp[..., 2:]) / np.pi
            lat = np.arcsin(tmp[..., 1:2]) / (0.5 * np.pi)
            lonlat = np.concatenate([lon, lat], axis=-1)
            grids.append(torch.FloatTensor(lonlat[None, ...]))
        return grids

    def forward(self, batch, mode='bilinear'):
        [_, _, h, w] = batch.shape
        assert h == self.equ_h and w == self.equ_w
        assert mode in ['nearest', 'bilinear']

        out = []
        for i in range(6):
            grid = getattr(self, 'grid_%d'%i)
            grid = grid.repeat(batch.shape[0], 1, 1, 1)
            sample = F.grid_sample(batch, grid, mode=mode, align_corners=True)
            out.append(sample)
        out = torch.cat(out, dim=0)
        final_out = []
        for i in range(batch.shape[0]):
            final_out.append(out[i::batch.shape[0], ...])
        final_out = torch.cat(final_out, dim=0)
        return final_out


# ==================== Cube2Equirec (from Depth Anywhere) ====================
class Cube2Equirec(nn.Module):
    """将 6-face cubemap 转换为 equirectangular 图像"""
    def __init__(self, cube_length, equ_h):
        super().__init__()
        self.cube_length = cube_length
        self.equ_h = equ_h
        equ_w = equ_h * 2
        self.equ_w = equ_w
        theta = (np.arange(equ_w) / (equ_w-1) - 0.5) * 2 * np.pi
        phi = (np.arange(equ_h) / (equ_h-1) - 0.5) * np.pi

        theta, phi = np.meshgrid(theta, phi)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(phi)
        z = np.cos(theta) * np.cos(phi)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)

        planes = np.asarray([
            [0, 0, 1,  1],  # z = -1
            [0, 1, 0, -1],  # y =  1
            [0, 0, 1, -1],  # z =  1
            [1, 0, 0,  1],  # x = -1
            [1, 0, 0, -1],  # x =  1
            [0, 1, 0,  1]   # y = -1
        ])
        r_lst = np.array([
            [0, 1, 0],
            [0.5, 0, 0],
            [0, 0, 0],
            [0, 0.5, 0],
            [0, -0.5, 0],
            [-0.5, 0, 0]
        ]) * np.pi
        f = cube_length / 2.0
        self.K = np.array([
            [f, 0, (cube_length-1)/2.0],
            [0, f, (cube_length-1)/2.0],
            [0, 0, 1]
        ])
        self.R_lst = [cv2.Rodrigues(x)[0] for x in r_lst]

        masks, XYs = self._intersection(xyz, planes)

        for i in range(6):
            self.register_buffer('mask_%d'%i, masks[i])
            self.register_buffer('XY_%d'%i, XYs[i])

    def forward(self, x, mode='bilinear'):
        assert mode in ['nearest', 'bilinear']
        assert x.shape[0] % 6 == 0
        equ_count = x.shape[0] // 6
        equi = torch.zeros(equ_count, x.shape[1], self.equ_h, self.equ_w, device=x.device)
        for i in range(6):
            now = x[i::6, ...]
            mask = getattr(self, 'mask_%d'%i)
            mask = mask[None, ...].repeat(equ_count, x.shape[1], 1, 1)

            XY = (getattr(self, 'XY_%d'%i)[None, None, :, :].repeat(equ_count, 1, 1, 1) / (self.cube_length-1) - 0.5) * 2
            sample = F.grid_sample(now, XY, mode=mode, align_corners=True)[..., 0, :]
            equi[mask] = sample.view(-1)

        return equi

    def _intersection(self, xyz, planes):
        abc = planes[:, :-1]

        depth = -planes[:, 3][None, None, ...] / np.dot(xyz, abc.T)
        depth[depth < 0] = np.inf
        arg = np.argmin(depth, axis=-1)
        depth = np.min(depth, axis=-1)

        pts = depth[..., None] * xyz

        mask_lst = []
        mapping_XY = []
        for i in range(6):
            mask = arg == i
            mask = np.tile(mask[..., None], [1, 1, 3])

            XY = np.dot(np.dot(pts[mask].reshape([-1, 3]), self.R_lst[i].T), self.K.T)
            XY = np.clip(XY[..., :2].copy() / XY[..., 2:], 0, self.cube_length-1)
            mask_lst.append(mask[..., 0])
            mapping_XY.append(XY)
        mask_lst = [torch.BoolTensor(x) for x in mask_lst]
        mapping_XY = [torch.FloatTensor(x) for x in mapping_XY]

        return mask_lst, mapping_XY


def load_da3_model(device="cuda"):
    """加载 DA V3 模型"""
    print("加载 Depth Anything V3...")
    model = DepthAnything3.from_pretrained("depth-anything/da3-base")
    model = model.to(device=device)
    model.eval()
    print("模型加载完成")
    return model


def equirect_to_cube_depth_dav3(
    equirect_img_path: str,
    model,
    cube_size: int = 512,
    device: str = "cuda",
):
    """
    使用 6-face cubemap + DA V3 估计 equirectangular 图像的深度

    Args:
        equirect_img_path: equirectangular 图像路径
        model: DA V3 模型
        cube_size: cube face 的尺寸
        device: 设备

    Returns:
        equirect_depth: equirectangular 格式的深度图
        cube_depths: 6 个 cube face 的深度图
        cube_images: 6 个 cube face 的 RGB 图像
    """
    # 1. 读取 equirectangular 图像
    equirect_img = cv2.imread(equirect_img_path)
    equirect_img = cv2.cvtColor(equirect_img, cv2.COLOR_BGR2RGB)
    h, w = equirect_img.shape[:2]
    print(f"Equirectangular 图像尺寸: {w} x {h}")

    # 2. 创建 Equirec2Cube 和 Cube2Equirec 转换器
    e2c = Equirec2Cube(cube_dim=cube_size, equ_h=h, FoV=90.0).to(device)
    c2e = Cube2Equirec(cube_length=cube_size, equ_h=h).to(device)

    # 3. 将 equirectangular 图像转换为 6-face cubemap
    equirect_tensor = torch.from_numpy(equirect_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    equirect_tensor = equirect_tensor.to(device)

    with torch.no_grad():
        cube_faces = e2c(equirect_tensor, mode='bilinear')  # (6, 3, cube_size, cube_size)

    # cube_faces: B, D, F, L, R, U (back, down, front, left, right, up)
    face_names = ['Back', 'Down', 'Front', 'Left', 'Right', 'Up']

    # 4. 保存 cube faces 为临时文件，供 DA V3 处理
    temp_dir = Path("/tmp/cube_faces_temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    cube_paths = []
    cube_images = []
    for i in range(6):
        face_img = cube_faces[i].cpu().numpy().transpose(1, 2, 0)
        face_img = (face_img * 255).astype(np.uint8)
        cube_images.append(face_img)

        face_path = temp_dir / f"face_{i}_{face_names[i]}.png"
        cv2.imwrite(str(face_path), cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
        cube_paths.append(str(face_path))

    # 5. 使用 DA V3 对每个 cube face 估计深度
    print("对 6 个 cube face 进行深度估计...")
    with torch.no_grad():
        prediction = model.inference(
            cube_paths,
            extrinsics=None,
            intrinsics=None,
        )
    cube_depths_raw = prediction.depth  # (6, H, W)
    print(f"Cube 深度图尺寸: {cube_depths_raw.shape}")

    # 6. 调整深度图尺寸到 cube_size
    cube_depths_list = []
    for i in range(6):
        depth = cube_depths_raw[i]
        if depth.shape[0] != cube_size or depth.shape[1] != cube_size:
            depth = cv2.resize(depth, (cube_size, cube_size), interpolation=cv2.INTER_LINEAR)
        cube_depths_list.append(depth)

    cube_depths = np.stack(cube_depths_list, axis=0)  # (6, cube_size, cube_size)

    # 7. 归一化深度（每个 face 单独归一化）
    for i in range(6):
        d = cube_depths[i]
        d_min, d_max = d.min(), d.max()
        if d_max > d_min:
            cube_depths[i] = (d - d_min) / (d_max - d_min)

    # 8. 将 cube 深度转换为 equirectangular 深度
    # 将深度转为 tensor 并添加 channel 维度
    cube_depths_tensor = torch.from_numpy(cube_depths).unsqueeze(1).float().to(device)  # (6, 1, H, W)

    with torch.no_grad():
        equirect_depth_tensor = c2e(cube_depths_tensor, mode='bilinear')  # (1, 1, H, W)

    equirect_depth = equirect_depth_tensor[0, 0].cpu().numpy()

    return equirect_depth, cube_depths, cube_images, face_names


def visualize_results(
    equirect_img_path: str,
    equirect_depth: np.ndarray,
    cube_depths: np.ndarray,
    cube_images: list,
    face_names: list,
    output_dir: str,
    direct_equirect_depth: np.ndarray = None,
):
    """可视化结果"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 读取原始 equirectangular 图像
    equirect_img = cv2.imread(equirect_img_path)
    equirect_img = cv2.cvtColor(equirect_img, cv2.COLOR_BGR2RGB)

    # 1. Equirectangular 概览
    fig, axes = plt.subplots(2, 1, figsize=(20, 10))

    axes[0].imshow(equirect_img)
    axes[0].set_title("Equirectangular Image", fontsize=14)
    axes[0].axis('off')

    im = axes[1].imshow(equirect_depth, cmap='turbo')
    axes[1].set_title("Depth via Cubemap + DA V3", fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.02)

    plt.tight_layout()
    plt.savefig(output_path / "01_equirect_depth_via_cube.png", dpi=150)
    plt.close()

    # 2. 6 个 cube faces
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))

    for i in range(6):
        axes[0, i].imshow(cube_images[i])
        axes[0, i].set_title(f"{face_names[i]}", fontsize=12)
        axes[0, i].axis('off')

        im = axes[1, i].imshow(cube_depths[i], cmap='turbo')
        axes[1, i].set_title(f"{face_names[i]} Depth", fontsize=12)
        axes[1, i].axis('off')

    plt.suptitle("6-Face Cubemap and Depth", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path / "02_cube_faces_and_depth.png", dpi=150)
    plt.close()

    # 3. 如果有直接估计的 equirectangular 深度，进行比较
    if direct_equirect_depth is not None:
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))

        # 调整 direct_equirect_depth 尺寸
        if direct_equirect_depth.shape != equirect_depth.shape:
            direct_equirect_depth = cv2.resize(
                direct_equirect_depth,
                (equirect_depth.shape[1], equirect_depth.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        im1 = axes[0, 0].imshow(direct_equirect_depth, cmap='turbo')
        axes[0, 0].set_title("Direct Equirectangular Depth (DA V3)", fontsize=12)
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.02)

        im2 = axes[0, 1].imshow(equirect_depth, cmap='turbo')
        axes[0, 1].set_title("Depth via Cubemap + DA V3", fontsize=12)
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.02)

        # 归一化后计算差异
        d1_norm = (direct_equirect_depth - direct_equirect_depth.min()) / (direct_equirect_depth.max() - direct_equirect_depth.min() + 1e-8)
        d2_norm = (equirect_depth - equirect_depth.min()) / (equirect_depth.max() - equirect_depth.min() + 1e-8)
        diff = np.abs(d1_norm - d2_norm)

        im3 = axes[1, 0].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
        axes[1, 0].set_title(f"Difference (MAE={diff.mean():.4f})", fontsize=12)
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.02)

        # 直方图
        axes[1, 1].hist(diff.flatten(), bins=100, alpha=0.7)
        axes[1, 1].set_xlabel("Absolute Difference")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title("Difference Distribution")

        plt.suptitle("Comparison: Direct vs Cubemap-based Depth Estimation", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path / "03_comparison.png", dpi=150)
        plt.close()

        print(f"\n比较指标:")
        print(f"  MAE: {diff.mean():.4f}")
        print(f"  RMSE: {np.sqrt((diff**2).mean()):.4f}")
        print(f"  Max Diff: {diff.max():.4f}")

    print(f"\n结果保存在: {output_path}")


def main():
    """主函数"""
    # 配置
    input_image = "/mnt/data/UNav-IO/temp/New_York_University/Tandon/4_floor/stella_vslam_dense/keyframes/image0.png"
    output_dir = "/home/unav/Desktop/unav/unav/floor_depth_analyzer/output/equirect_depth_via_cube_dav3"
    cube_size = 512

    print("=" * 80)
    print("Equirectangular Depth Estimation via Cubemap + DA V3")
    print("=" * 80)
    print(f"\nInput: {input_image}")
    print(f"Output: {output_dir}")
    print(f"Cube size: {cube_size}")
    print()

    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_da3_model(device)

    # 方法1: 通过 cubemap 估计深度
    print("\n[方法1] 通过 Cubemap + DA V3 估计深度...")
    equirect_depth, cube_depths, cube_images, face_names = equirect_to_cube_depth_dav3(
        input_image, model, cube_size, device
    )
    print(f"Equirectangular 深度图尺寸: {equirect_depth.shape}")
    print(f"深度范围: [{equirect_depth.min():.4f}, {equirect_depth.max():.4f}]")

    # 方法2: 直接对 equirectangular 估计深度（用于比较）
    print("\n[方法2] 直接对 Equirectangular 运行 DA V3（用于比较）...")
    with torch.no_grad():
        prediction = model.inference([input_image], extrinsics=None, intrinsics=None)
    direct_equirect_depth = prediction.depth[0]
    print(f"直接深度图尺寸: {direct_equirect_depth.shape}")

    # 可视化
    print("\n生成可视化...")
    visualize_results(
        input_image,
        equirect_depth,
        cube_depths,
        cube_images,
        face_names,
        output_dir,
        direct_equirect_depth,
    )

    print("\n完成!")


if __name__ == "__main__":
    main()
