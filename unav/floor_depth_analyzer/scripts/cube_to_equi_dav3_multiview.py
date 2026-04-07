"""
使用 DA V3 多视角重建能力处理 cubemap，然后融合回 equirectangular

DA V3 支持多图输入并统一度量，这样可以避免独立估计时的尺度不一致问题
"""

import sys
sys.path.insert(0, "/home/unav/Desktop/unav")

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy.ndimage import map_coordinates
import tempfile
import os

from depth_anything_3.api import DepthAnything3


# ==================== Equirec2Cube ====================
class Equirec2Cube:
    """Equirectangular to Cubemap 转换"""
    def __init__(self, equ_h, equ_w, face_w):
        self.equ_h = equ_h
        self.equ_w = equ_w
        self.face_w = face_w
        self._xyzcube()
        self._xyz2coor()

    def _xyzcube(self):
        self.xyz = np.zeros((self.face_w, self.face_w * 6, 3), np.float32)
        rng = np.linspace(-0.5, 0.5, num=self.face_w, dtype=np.float32)
        self.grid = np.stack(np.meshgrid(rng, -rng), -1)

        # Front, Right, Back, Left, Up, Down
        self.xyz[:, 0*self.face_w:1*self.face_w, [0, 1]] = self.grid
        self.xyz[:, 0*self.face_w:1*self.face_w, 2] = 0.5

        self.xyz[:, 1*self.face_w:2*self.face_w, [2, 1]] = self.grid[:, ::-1]
        self.xyz[:, 1*self.face_w:2*self.face_w, 0] = 0.5

        self.xyz[:, 2*self.face_w:3*self.face_w, [0, 1]] = self.grid[:, ::-1]
        self.xyz[:, 2*self.face_w:3*self.face_w, 2] = -0.5

        self.xyz[:, 3*self.face_w:4*self.face_w, [2, 1]] = self.grid
        self.xyz[:, 3*self.face_w:4*self.face_w, 0] = -0.5

        self.xyz[:, 4*self.face_w:5*self.face_w, [0, 2]] = self.grid[::-1, :]
        self.xyz[:, 4*self.face_w:5*self.face_w, 1] = 0.5

        self.xyz[:, 5*self.face_w:6*self.face_w, [0, 2]] = self.grid
        self.xyz[:, 5*self.face_w:6*self.face_w, 1] = -0.5

    def _xyz2coor(self):
        x, y, z = np.split(self.xyz, 3, axis=-1)
        lon = np.arctan2(x, z)
        c = np.sqrt(x ** 2 + z ** 2)
        lat = np.arctan2(y, c)
        self.coor_x = (lon / (2 * np.pi) + 0.5) * self.equ_w - 0.5
        self.coor_y = (-lat / np.pi + 0.5) * self.equ_h - 0.5

    def sample_equirec(self, e_img, order=1):
        pad_u = np.roll(e_img[[0]], self.equ_w // 2, 1)
        pad_d = np.roll(e_img[[-1]], self.equ_w // 2, 1)
        e_img = np.concatenate([e_img, pad_d, pad_u], 0)
        return map_coordinates(e_img, [self.coor_y, self.coor_x], order=order, mode='wrap')[..., 0]

    def run(self, equ_img):
        h, w = equ_img.shape[:2]
        if h != self.equ_h or w != self.equ_w:
            equ_img = cv2.resize(equ_img, (self.equ_w, self.equ_h))
        cube_img = np.stack([self.sample_equirec(equ_img[..., i], order=1)
                             for i in range(equ_img.shape[2])], axis=-1)
        return cube_img

    def get_faces(self, equ_img):
        """返回 6 个独立的 cube faces"""
        cube = self.run(equ_img)
        faces = []
        for i in range(6):
            face = cube[:, i*self.face_w:(i+1)*self.face_w, :]
            faces.append(face)
        return faces


# ==================== Cube2Equirec ====================
class Cube2Equirec(nn.Module):
    """Cubemap to Equirectangular 转换"""
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
            [0, 0, 1,  1],   # Back   (z = -1)
            [0, 1, 0, -1],   # Up     (y =  1)
            [0, 0, 1, -1],   # Front  (z =  1)
            [1, 0, 0,  1],   # Left   (x = -1)
            [1, 0, 0, -1],   # Right  (x =  1)
            [0, 1, 0,  1]    # Down   (y = -1)
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


def get_cube_extrinsics():
    """获取 cubemap 6 个面的相机外参 (相对于中心)"""
    # 6 faces: Front, Right, Back, Left, Up, Down
    # 每个面的 4x4 变换矩阵 (从世界坐标到相机坐标)
    extrinsics = []

    # Front (看向 +Z)
    R_front = np.eye(3)
    t_front = np.zeros(3)
    E_front = np.eye(4)
    E_front[:3, :3] = R_front
    E_front[:3, 3] = t_front
    extrinsics.append(E_front)

    # Right (看向 +X)
    R_right = cv2.Rodrigues(np.array([0, -np.pi/2, 0]))[0]
    E_right = np.eye(4)
    E_right[:3, :3] = R_right
    extrinsics.append(E_right)

    # Back (看向 -Z)
    R_back = cv2.Rodrigues(np.array([0, np.pi, 0]))[0]
    E_back = np.eye(4)
    E_back[:3, :3] = R_back
    extrinsics.append(E_back)

    # Left (看向 -X)
    R_left = cv2.Rodrigues(np.array([0, np.pi/2, 0]))[0]
    E_left = np.eye(4)
    E_left[:3, :3] = R_left
    extrinsics.append(E_left)

    # Up (看向 +Y)
    R_up = cv2.Rodrigues(np.array([np.pi/2, 0, 0]))[0]
    E_up = np.eye(4)
    E_up[:3, :3] = R_up
    extrinsics.append(E_up)

    # Down (看向 -Y)
    R_down = cv2.Rodrigues(np.array([-np.pi/2, 0, 0]))[0]
    E_down = np.eye(4)
    E_down[:3, :3] = R_down
    extrinsics.append(E_down)

    return np.stack(extrinsics, axis=0)  # (6, 4, 4)


def get_cube_intrinsics(face_w, fov=90):
    """获取 cubemap 的相机内参 (90度 FOV)"""
    f = face_w / (2 * np.tan(np.radians(fov) / 2))
    cx = face_w / 2
    cy = face_w / 2

    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ])
    # 返回 6 个相同的内参
    return np.stack([K] * 6, axis=0)  # (6, 3, 3)


def main():
    output_dir = Path("/home/unav/Desktop/unav/unav/floor_depth_analyzer/output")
    output_dir.mkdir(exist_ok=True)

    # 加载测试图像
    img_path = "/mnt/data/UNav-IO/temp/New_York_University/Tandon/4_floor/stella_vslam_dense/keyframes/image0.png"
    print(f"Loading image: {img_path}")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 参数设置
    equ_h, equ_w = 512, 1024
    face_w = 256

    # 提取 cubemap faces
    e2c = Equirec2Cube(equ_h, equ_w, face_w)
    img_resized = cv2.resize(img, (equ_w, equ_h))
    faces = e2c.get_faces(img_resized)
    face_names = ['Front', 'Right', 'Back', 'Left', 'Up', 'Down']

    print(f"Extracted {len(faces)} cube faces, each {face_w}x{face_w}")

    # 保存 cube faces 到临时文件
    temp_dir = tempfile.mkdtemp()
    cube_paths = []
    for i, (face, name) in enumerate(zip(faces, face_names)):
        path = os.path.join(temp_dir, f'face_{i}_{name}.png')
        Image.fromarray(face).save(path)
        cube_paths.append(path)
        print(f"  Saved {name} face to {path}")

    # 加载 DA V3
    print("\nLoading Depth Anything V3...")
    model = DepthAnything3.from_pretrained("depth-anything/da3-base")
    model = model.cuda().eval()
    print("DA V3 loaded!")

    # 准备相机参数
    extrinsics = get_cube_extrinsics()
    intrinsics = get_cube_intrinsics(face_w)
    print(f"\nExtrinsics shape: {extrinsics.shape}")
    print(f"Intrinsics shape: {intrinsics.shape}")

    # 方法1: 独立处理每个 face
    print("\n=== Method 1: Independent depth estimation ===")
    depths_independent = []
    with torch.no_grad():
        for i, path in enumerate(cube_paths):
            pred = model.inference([path], extrinsics=None, intrinsics=None)
            depth = pred.depth[0]
            depths_independent.append(depth)
            print(f"  {face_names[i]}: {depth.min():.3f} - {depth.max():.3f} m")

    # 方法2: 批量处理所有 faces (无相机参数，让 DA V3 独立估计但一次处理)
    print("\n=== Method 2: Batch depth estimation (no camera params) ===")
    with torch.no_grad():
        pred_batch = model.inference(
            cube_paths,
            extrinsics=None,
            intrinsics=None,
        )
    depths_multiview = pred_batch.depth
    print(f"Batch output shape: {depths_multiview.shape}")
    for i, name in enumerate(face_names):
        d = depths_multiview[i]
        print(f"  {name}: {d.min():.3f} - {d.max():.3f} m")

    # 融合回 equirectangular
    print("\n=== Fusing to Equirectangular ===")
    c2e = Cube2Equirec(face_w, equ_h)

    # 方法1的融合
    depths_ind_tensor = torch.stack([torch.from_numpy(d).float() for d in depths_independent], dim=0)
    depths_ind_tensor = depths_ind_tensor.unsqueeze(1)  # (6, 1, H, W)
    # 调整顺序: [Front, Right, Back, Left, Up, Down] -> [Back, Up, Front, Left, Right, Down]
    # Cube2Equirec 期望的顺序
    order_mapping = [2, 4, 0, 3, 1, 5]  # Front->2, Right->4, Back->0, Left->3, Up->1, Down->5
    depths_ind_reordered = depths_ind_tensor[order_mapping]
    equi_depth_ind = c2e(depths_ind_reordered).squeeze().numpy()

    # 方法2的融合
    depths_mv_tensor = torch.from_numpy(depths_multiview).float().unsqueeze(1)  # (6, 1, H, W)
    depths_mv_reordered = depths_mv_tensor[order_mapping]
    equi_depth_mv = c2e(depths_mv_reordered).squeeze().numpy()

    print(f"Independent fusion - Depth range: {equi_depth_ind.min():.3f} - {equi_depth_ind.max():.3f} m")
    print(f"Multi-view fusion - Depth range: {equi_depth_mv.min():.3f} - {equi_depth_mv.max():.3f} m")

    # 可视化对比
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    # 原图
    axes[0, 0].imshow(img_resized)
    axes[0, 0].set_title('Original Equirectangular', fontsize=12)
    axes[0, 0].axis('off')

    # Cube faces 展示
    cube_display = np.concatenate(faces[:4], axis=1)  # Front, Right, Back, Left
    axes[0, 1].imshow(cube_display)
    axes[0, 1].set_title('Cube Faces (Front, Right, Back, Left)', fontsize=12)
    axes[0, 1].axis('off')

    # 独立估计融合结果
    im1 = axes[1, 0].imshow(equi_depth_ind, cmap='inferno')
    axes[1, 0].set_title(f'Independent Estimation Fusion\n[{equi_depth_ind.min():.2f} - {equi_depth_ind.max():.2f} m]', fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, label='Depth (m)')

    # 多视角估计融合结果
    im2 = axes[1, 1].imshow(equi_depth_mv, cmap='inferno')
    axes[1, 1].set_title(f'Multi-view Estimation Fusion\n[{equi_depth_mv.min():.2f} - {equi_depth_mv.max():.2f} m]', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, label='Depth (m)')

    # 差异图
    diff = np.abs(equi_depth_ind - equi_depth_mv)
    im3 = axes[2, 0].imshow(diff, cmap='hot')
    axes[2, 0].set_title(f'Absolute Difference\n[Mean: {diff.mean():.3f} m, Max: {diff.max():.3f} m]', fontsize=12)
    axes[2, 0].axis('off')
    plt.colorbar(im3, ax=axes[2, 0], fraction=0.046, label='Diff (m)')

    # RGB + Depth overlay
    axes[2, 1].imshow(img_resized)
    im4 = axes[2, 1].imshow(equi_depth_mv, cmap='inferno', alpha=0.5)
    axes[2, 1].set_title('RGB + Multi-view Depth Overlay', fontsize=12)
    axes[2, 1].axis('off')
    plt.colorbar(im4, ax=axes[2, 1], fraction=0.046, label='Depth (m)')

    plt.suptitle('DA V3 Cubemap to Equirectangular Depth Fusion\n(Independent vs Multi-view)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'dav3_cube_to_equi_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nResults saved to: {output_dir / 'dav3_cube_to_equi_comparison.png'}")

    # 保存深度数据
    np.save(output_dir / 'equi_depth_independent.npy', equi_depth_ind)
    np.save(output_dir / 'equi_depth_multiview.npy', equi_depth_mv)
    print(f"Depth arrays saved to output directory")

    # 清理临时文件
    for path in cube_paths:
        os.unlink(path)
    os.rmdir(temp_dir)

    # 统计对比
    print("\n" + "=" * 60)
    print("Statistics Comparison")
    print("=" * 60)
    print(f"{'Metric':<25} {'Independent':<15} {'Multi-view':<15}")
    print("-" * 60)
    print(f"{'Min depth (m)':<25} {equi_depth_ind.min():<15.3f} {equi_depth_mv.min():<15.3f}")
    print(f"{'Max depth (m)':<25} {equi_depth_ind.max():<15.3f} {equi_depth_mv.max():<15.3f}")
    print(f"{'Mean depth (m)':<25} {equi_depth_ind.mean():<15.3f} {equi_depth_mv.mean():<15.3f}")
    print(f"{'Std depth (m)':<25} {equi_depth_ind.std():<15.3f} {equi_depth_mv.std():<15.3f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
