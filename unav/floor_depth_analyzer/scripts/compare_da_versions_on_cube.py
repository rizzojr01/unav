"""
对比 Depth Anything V1, V2, V3 在 cubemap faces 上的深度估计效果
"""

import sys
sys.path.insert(0, "/home/unav/Desktop/unav")

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy.ndimage import map_coordinates

# ==================== Equirec2Cube ====================
class Equirec2Cube:
    """基于 py360convert 的 equirectangular 到 cubemap 转换"""
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


def load_da_v2():
    """加载 Depth Anything V2"""
    print("Loading Depth Anything V2...")
    sys.path.insert(0, "/home/unav/Desktop/unav/unav/tmp/Depth-Anything-V2")
    from depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    encoder = 'vitl'
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'/home/unav/.cache/depth_anything_v2/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.cuda().eval()
    print("DA V2 loaded!")
    return model


def load_da_v3():
    """加载 Depth Anything V3"""
    print("Loading Depth Anything V3...")
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained("depth-anything/da3-base")
    model = model.cuda().eval()
    print("DA V3 loaded!")
    return model


def predict_v2(model, image_np):
    """DA V2 预测"""
    with torch.no_grad():
        depth = model.infer_image(image_np)
    return depth


def predict_v3(model, image_np):
    """DA V3 预测 (metric depth)"""
    import tempfile
    import os

    # DA V3 需要文件路径作为输入
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_path = f.name
        Image.fromarray(image_np).save(temp_path)

    try:
        with torch.no_grad():
            prediction = model.inference(
                [temp_path],
                extrinsics=None,
                intrinsics=None,
            )
        depth = prediction.depth[0]  # Already numpy array (H, W)
        if hasattr(depth, 'cpu'):
            depth = depth.cpu().numpy()
    finally:
        os.unlink(temp_path)

    return depth


def main():
    output_dir = Path("/home/unav/Desktop/unav/unav/floor_depth_analyzer/output")
    output_dir.mkdir(exist_ok=True)

    # 加载测试图像
    img_path = "/mnt/data/UNav-IO/temp/New_York_University/Tandon/4_floor/stella_vslam_dense/keyframes/image0.png"
    print(f"Loading image: {img_path}")
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 提取 cubemap faces
    equ_h, equ_w = 512, 1024
    face_w = 256
    e2c = Equirec2Cube(equ_h, equ_w, face_w)

    img_resized = cv2.resize(img, (equ_w, equ_h))
    faces = e2c.get_faces(img_resized)
    face_names = ['Front', 'Right', 'Back', 'Left', 'Up', 'Down']

    print(f"Extracted {len(faces)} cube faces, each {face_w}x{face_w}")

    # 加载模型
    model_v2 = load_da_v2()
    model_v3 = load_da_v3()

    # 对每个 face 进行深度估计
    results = {'v2': [], 'v3': []}

    for i, (face, name) in enumerate(zip(faces, face_names)):
        print(f"\nProcessing {name} face...")

        # V2 (relative depth)
        depth_v2 = predict_v2(model_v2, face)
        results['v2'].append(depth_v2)

        # V3 (metric depth)
        depth_v3 = predict_v3(model_v3, face)
        results['v3'].append(depth_v3)

        print(f"  V2 range: {depth_v2.min():.3f} - {depth_v2.max():.3f} (relative)")
        print(f"  V3 range: {depth_v3.min():.3f} - {depth_v3.max():.3f} m (metric)")

    # 可视化对比
    fig, axes = plt.subplots(6, 3, figsize=(14, 24))

    for i, name in enumerate(face_names):
        # RGB
        axes[i, 0].imshow(faces[i])
        axes[i, 0].set_title(f'{name} - RGB')
        axes[i, 0].axis('off')

        # V2 (relative)
        im2 = axes[i, 1].imshow(results['v2'][i], cmap='inferno')
        axes[i, 1].set_title(f'DA V2 (relative)\n[{results["v2"][i].min():.1f}-{results["v2"][i].max():.1f}]')
        axes[i, 1].axis('off')
        plt.colorbar(im2, ax=axes[i, 1], fraction=0.046)

        # V3 (metric)
        im3 = axes[i, 2].imshow(results['v3'][i], cmap='inferno')
        axes[i, 2].set_title(f'DA V3 (metric)\n[{results["v3"][i].min():.2f}-{results["v3"][i].max():.2f}m]')
        axes[i, 2].axis('off')
        plt.colorbar(im3, ax=axes[i, 2], fraction=0.046)

    plt.suptitle('Depth Anything V2 vs V3 on Cubemap Faces\n(V2: relative depth, V3: metric depth in meters)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'da_v2_v3_cube_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nComparison saved to: {output_dir / 'da_v2_v3_cube_comparison.png'}")

    # 统计对比
    print("\n" + "=" * 70)
    print("Depth Statistics Comparison (V2: relative, V3: metric in meters)")
    print("=" * 70)
    print(f"{'Face':<10} {'Version':<8} {'Min':<10} {'Max':<10} {'Mean':<10} {'Std':<10}")
    print("-" * 70)

    for i, name in enumerate(face_names):
        for ver, depths in results.items():
            d = depths[i]
            unit = "m" if ver == "v3" else ""
            print(f"{name:<10} {ver.upper():<8} {d.min():<10.3f} {d.max():<10.3f} {d.mean():<10.3f} {d.std():<10.3f} {unit}")
        print("-" * 70)

    # 创建简化对比图 (只看 Front 和 Down)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for row, idx in enumerate([0, 5]):  # Front and Down
        name = face_names[idx]
        axes[row, 0].imshow(faces[idx])
        axes[row, 0].set_title(f'{name} - RGB', fontsize=12)
        axes[row, 0].axis('off')

        for col, (ver, label) in enumerate([('v2', 'V2 (relative)'), ('v3', 'V3 (metric)')]):
            depth = results[ver][idx]
            im = axes[row, col+1].imshow(depth, cmap='inferno')
            unit = 'm' if ver == 'v3' else ''
            axes[row, col+1].set_title(f'DA {label}\n[{depth.min():.2f} - {depth.max():.2f}{unit}]', fontsize=11)
            axes[row, col+1].axis('off')
            plt.colorbar(im, ax=axes[row, col+1], fraction=0.046)

    plt.suptitle('Depth Anything V2 vs V3 (Front & Down faces)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'da_v2_v3_front_down.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Simplified comparison saved to: {output_dir / 'da_v2_v3_front_down.png'}")


if __name__ == '__main__':
    main()
