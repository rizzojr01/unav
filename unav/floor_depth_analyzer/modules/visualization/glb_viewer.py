#!/usr/bin/env python3
"""
GLB 3D Point Cloud Visualization

可视化DA3生成的GLB文件
"""

import numpy as np
import trimesh
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_glb(glb_path: str, output_dir: str = None):
    """
    可视化GLB文件

    Args:
        glb_path: GLB文件路径
        output_dir: 输出目录（可选）

    Returns:
        vertices: 点云顶点
        colors: 点云颜色
    """
    print("="*80)
    print("GLB 3D点云可视化")
    print("="*80)
    print(f"\n加载: {glb_path}")

    # 加载GLB文件
    scene = trimesh.load(glb_path)

    print(f"\n场景类型: {type(scene)}")

    # 如果是Scene对象，获取所有几何体
    if isinstance(scene, trimesh.Scene):
        print(f"场景包含 {len(scene.geometry)} 个几何体")

        # 合并所有几何体
        meshes = []
        for name, geom in scene.geometry.items():
            print(f"  - {name}: {type(geom)}")
            if isinstance(geom, trimesh.PointCloud):
                print(f"    点数: {len(geom.vertices):,}")
                meshes.append(geom)
            elif isinstance(geom, trimesh.Trimesh):
                print(f"    顶点数: {len(geom.vertices):,}")
                print(f"    面数: {len(geom.faces):,}")
                meshes.append(geom)

        if len(meshes) == 1:
            mesh = meshes[0]
        else:
            # 合并多个几何体
            vertices = []
            colors = []
            for m in meshes:
                vertices.append(m.vertices)
                if hasattr(m, 'colors') and m.colors is not None:
                    colors.append(m.colors[:, :3])
                elif hasattr(m, 'visual') and hasattr(m.visual, 'vertex_colors'):
                    colors.append(m.visual.vertex_colors[:, :3])
                else:
                    colors.append(np.ones((len(m.vertices), 3)) * 128)

            vertices = np.vstack(vertices)
            colors = np.vstack(colors)
            mesh = trimesh.PointCloud(vertices=vertices, colors=colors)

    else:
        mesh = scene

    # 获取点云数据
    vertices = mesh.vertices
    if hasattr(mesh, 'colors') and mesh.colors is not None:
        colors = mesh.colors[:, :3] / 255.0
    elif hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
        colors = mesh.visual.vertex_colors[:, :3] / 255.0
    else:
        colors = np.ones((len(vertices), 3)) * 0.5

    print(f"\n点云统计:")
    print(f"  总点数: {len(vertices):,}")
    print(f"  X范围: [{vertices[:, 0].min():.2f}, {vertices[:, 0].max():.2f}]")
    print(f"  Y范围: [{vertices[:, 1].min():.2f}, {vertices[:, 1].max():.2f}]")
    print(f"  Z范围: [{vertices[:, 2].min():.2f}, {vertices[:, 2].max():.2f}]")

    # 创建可视化
    fig = plt.figure(figsize=(20, 15))

    # 1. XY平面（俯视图）
    ax1 = fig.add_subplot(2, 3, 1)
    scatter1 = ax1.scatter(vertices[:, 0], vertices[:, 1], c=colors, s=0.1, alpha=0.5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('俯视图 (XY平面)', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # 2. XZ平面（侧视图）
    ax2 = fig.add_subplot(2, 3, 2)
    scatter2 = ax2.scatter(vertices[:, 0], vertices[:, 2], c=colors, s=0.1, alpha=0.5)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('侧视图 (XZ平面)', fontsize=14, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # 3. YZ平面（正视图）
    ax3 = fig.add_subplot(2, 3, 3)
    scatter3 = ax3.scatter(vertices[:, 1], vertices[:, 2], c=colors, s=0.1, alpha=0.5)
    ax3.set_xlabel('Y (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('正视图 (YZ平面)', fontsize=14, fontweight='bold')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    # 4. 高度分布
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.hist(vertices[:, 2], bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax4.set_xlabel('Z (m)')
    ax4.set_ylabel('点数')
    ax4.set_title('高度分布', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. 按高度着色的俯视图
    ax5 = fig.add_subplot(2, 3, 5)
    scatter5 = ax5.scatter(vertices[:, 0], vertices[:, 1], c=vertices[:, 2],
                          cmap='viridis', s=0.1, alpha=0.5)
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')
    ax5.set_title('俯视图（按高度着色）', fontsize=14, fontweight='bold')
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter5, ax=ax5, label='Z (m)')

    # 6. 点密度统计
    ax6 = fig.add_subplot(2, 3, 6)
    # 计算2D网格密度
    H, xedges, yedges = np.histogram2d(vertices[:, 0], vertices[:, 1], bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax6.imshow(H.T, extent=extent, origin='lower', cmap='hot', aspect='auto')
    ax6.set_xlabel('X (m)')
    ax6.set_ylabel('Y (m)')
    ax6.set_title('点密度热图', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax6, label='点数')

    plt.suptitle(f'3D点云可视化: {Path(glb_path).name}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    # 保存
    if output_dir:
        output_path = Path(output_dir)
        vis_path = output_path / "pointcloud_visualization.png"
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ 可视化保存在: {vis_path}")
    else:
        vis_path = Path(glb_path).parent / "pointcloud_visualization.png"
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ 可视化保存在: {vis_path}")

    plt.close()

    print("\n" + "="*80)
    print("完成！")
    print("="*80)
    print(f"\n查看可视化:")
    print(f"  eog {vis_path}")
    print()

    return vertices, colors
