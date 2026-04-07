"""
Visualize 3D point cloud and mesh from panorama depth reconstruction.
"""

import numpy as np
import open3d as o3d
import argparse
from pathlib import Path


def visualize_ply(ply_path: str, window_name: str = "3D Viewer"):
    """Visualize PLY file using Open3D."""
    ply_path = Path(ply_path)

    if not ply_path.exists():
        print(f"File not found: {ply_path}")
        return

    print(f"Loading {ply_path}...")

    # Try loading as mesh first
    mesh = o3d.io.read_triangle_mesh(str(ply_path))

    if len(mesh.triangles) > 0:
        print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh], window_name=window_name)
    else:
        # Load as point cloud
        pcd = o3d.io.read_point_cloud(str(ply_path))
        print(f"Loaded point cloud: {len(pcd.points)} points")
        o3d.visualization.draw_geometries([pcd], window_name=window_name)


def visualize_both(pointcloud_path: str, mesh_path: str):
    """Visualize both point cloud and mesh side by side."""
    pcd = o3d.io.read_point_cloud(pointcloud_path)
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    # Translate mesh to the right for side-by-side view
    mesh.translate([10, 0, 0])

    print(f"Point cloud: {len(pcd.points)} points")
    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

    o3d.visualization.draw_geometries([pcd, mesh], window_name="Point Cloud (left) vs Mesh (right)")


def create_floor_plan_from_pointcloud(ply_path: str, output_path: str = None,
                                       floor_height: float = -0.5,
                                       ceiling_height: float = 2.0,
                                       resolution: float = 0.05):
    """
    Create a 2D floor plan from 3D point cloud by projecting points.

    Args:
        ply_path: path to point cloud PLY
        output_path: output image path
        floor_height: minimum Y to consider (below this = floor)
        ceiling_height: maximum Y to consider (above this = ceiling)
        resolution: meters per pixel
    """
    import cv2

    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    # Filter by height (Y axis)
    mask = (points[:, 1] > floor_height) & (points[:, 1] < ceiling_height)
    points = points[mask]
    if colors is not None:
        colors = colors[mask]

    # Project to XZ plane (top-down view)
    x = points[:, 0]
    z = points[:, 2]

    # Determine image size
    x_min, x_max = x.min(), x.max()
    z_min, z_max = z.min(), z.max()

    width = int((x_max - x_min) / resolution) + 1
    height = int((z_max - z_min) / resolution) + 1

    print(f"Floor plan size: {width} x {height} pixels")

    # Create image
    floor_plan = np.zeros((height, width, 3), dtype=np.uint8)

    # Convert to pixel coordinates
    px = ((x - x_min) / resolution).astype(int)
    pz = ((z_max - z) / resolution).astype(int)  # Flip Z for image coordinates

    # Clip to image bounds
    px = np.clip(px, 0, width - 1)
    pz = np.clip(pz, 0, height - 1)

    if colors is not None:
        # Use actual colors
        for i in range(len(px)):
            floor_plan[pz[i], px[i]] = (colors[i] * 255).astype(np.uint8)
    else:
        # Binary occupancy
        floor_plan[pz, px] = 255

    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(floor_plan, cv2.COLOR_RGB2BGR))
        print(f"Saved floor plan to {output_path}")

    return floor_plan


def main():
    parser = argparse.ArgumentParser(description="Visualize 3D reconstruction")
    parser.add_argument("--ply", type=str, help="Path to PLY file")
    parser.add_argument("--pointcloud", type=str, help="Path to point cloud PLY")
    parser.add_argument("--mesh", type=str, help="Path to mesh PLY")
    parser.add_argument("--floor-plan", action="store_true", help="Generate floor plan")
    parser.add_argument("--output", type=str, default="floor_plan.png", help="Floor plan output path")

    args = parser.parse_args()

    if args.ply:
        visualize_ply(args.ply)
    elif args.pointcloud and args.mesh:
        visualize_both(args.pointcloud, args.mesh)
    elif args.floor_plan and args.pointcloud:
        create_floor_plan_from_pointcloud(args.pointcloud, args.output)
    else:
        print("Please provide --ply or both --pointcloud and --mesh")


if __name__ == "__main__":
    main()
