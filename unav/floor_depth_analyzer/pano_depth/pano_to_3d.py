"""
Panorama Depth to 3D Structure Reconstruction

Convert equirectangular depth map to 3D point cloud and mesh.
"""

import numpy as np
import cv2
from pathlib import Path
import argparse


def equirectangular_to_3d(depth: np.ndarray, rgb: np.ndarray = None) -> tuple:
    """
    Convert equirectangular depth map to 3D point cloud.

    Args:
        depth: (H, W) depth map in meters
        rgb: (H, W, 3) optional RGB image for coloring points

    Returns:
        points: (N, 3) 3D points
        colors: (N, 3) RGB colors (if rgb provided)
    """
    H, W = depth.shape

    # Create pixel coordinates
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)

    # Convert to spherical coordinates
    # θ (longitude): -π to π (left to right)
    # φ (latitude): π/2 to -π/2 (top to bottom)
    theta = (u / W - 0.5) * 2 * np.pi  # longitude
    phi = (0.5 - v / H) * np.pi         # latitude

    # Convert to 3D Cartesian coordinates
    # Convention: Y up, -Z forward (OpenGL style)
    x = depth * np.cos(phi) * np.sin(theta)
    y = depth * np.sin(phi)
    z = -depth * np.cos(phi) * np.cos(theta)

    # Create valid mask (depth > 0 and not too far)
    valid_mask = (depth > 0) & (depth < 100) & ~np.isnan(depth)

    # Stack points
    points = np.stack([x, y, z], axis=-1)  # (H, W, 3)
    points = points[valid_mask]  # (N, 3)

    colors = None
    if rgb is not None:
        colors = rgb[valid_mask]  # (N, 3)

    return points, colors, valid_mask


def create_mesh_from_depth(depth: np.ndarray, rgb: np.ndarray = None,
                           step: int = 1) -> tuple:
    """
    Create mesh (vertices + faces) from equirectangular depth map.

    Args:
        depth: (H, W) depth map
        rgb: (H, W, 3) optional RGB image
        step: downsampling step for mesh creation

    Returns:
        vertices: (N, 3) mesh vertices
        faces: (M, 3) triangle faces (vertex indices)
        colors: (N, 3) vertex colors
    """
    H, W = depth.shape

    # Downsample for efficiency
    depth_ds = depth[::step, ::step]
    H_ds, W_ds = depth_ds.shape

    if rgb is not None:
        rgb_ds = rgb[::step, ::step]

    # Create pixel coordinates
    u = np.arange(W_ds)
    v = np.arange(H_ds)
    u, v = np.meshgrid(u, v)

    # Convert to spherical coordinates
    theta = (u / W_ds - 0.5) * 2 * np.pi
    phi = (0.5 - v / H_ds) * np.pi

    # Convert to 3D
    x = depth_ds * np.cos(phi) * np.sin(theta)
    y = depth_ds * np.sin(phi)
    z = -depth_ds * np.cos(phi) * np.cos(theta)

    # Stack vertices
    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Create faces (triangles)
    faces = []
    valid_depth = (depth_ds > 0) & (depth_ds < 100) & ~np.isnan(depth_ds)

    for i in range(H_ds - 1):
        for j in range(W_ds - 1):
            # Check if all 4 corners are valid
            idx00 = i * W_ds + j
            idx01 = i * W_ds + (j + 1) % W_ds  # wrap around for panorama
            idx10 = (i + 1) * W_ds + j
            idx11 = (i + 1) * W_ds + (j + 1) % W_ds

            if valid_depth[i, j] and valid_depth[i, (j+1) % W_ds] and \
               valid_depth[i+1, j] and valid_depth[i+1, (j+1) % W_ds]:
                # Check depth discontinuity
                d00, d01 = depth_ds[i, j], depth_ds[i, (j+1) % W_ds]
                d10, d11 = depth_ds[i+1, j], depth_ds[i+1, (j+1) % W_ds]

                max_d = max(d00, d01, d10, d11)
                min_d = min(d00, d01, d10, d11)

                # Skip faces with large depth discontinuity
                if max_d / (min_d + 1e-6) < 2.0:
                    faces.append([idx00, idx10, idx01])
                    faces.append([idx01, idx10, idx11])

    faces = np.array(faces, dtype=np.int32)

    colors = None
    if rgb is not None:
        colors = rgb_ds.reshape(-1, 3)

    return vertices, faces, colors


def save_ply(filename: str, points: np.ndarray, colors: np.ndarray = None):
    """Save point cloud to PLY file."""
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(len(points)):
            if colors is not None:
                f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                       f"{int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}\n")
            else:
                f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}\n")


def save_mesh_ply(filename: str, vertices: np.ndarray, faces: np.ndarray,
                  colors: np.ndarray = None):
    """Save mesh to PLY file."""
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for i in range(len(vertices)):
            if colors is not None:
                f.write(f"{vertices[i, 0]:.6f} {vertices[i, 1]:.6f} {vertices[i, 2]:.6f} "
                       f"{int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}\n")
            else:
                f.write(f"{vertices[i, 0]:.6f} {vertices[i, 1]:.6f} {vertices[i, 2]:.6f}\n")

        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def load_depth(depth_path: str) -> np.ndarray:
    """Load depth map from various formats."""
    depth_path = Path(depth_path)

    if depth_path.suffix == '.npy':
        depth = np.load(depth_path)
    elif depth_path.suffix == '.png':
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth.dtype == np.uint16:
            # Structured3D format: mm to meters
            depth = depth.astype(np.float32) / 1000.0
        elif 'matterport' in str(depth_path).lower():
            # Matterport format: divide by 4000
            depth = depth.astype(np.float32) / 4000.0
        elif 'stanford' in str(depth_path).lower():
            # Stanford2D3D format: divide by 512
            depth = depth.astype(np.float32) / 512.0
    elif depth_path.suffix == '.dpt':
        # 360MonoDepth format
        with open(depth_path, 'rb') as f:
            # Read header
            header = f.read(80).decode('utf-8')
            # Parse dimensions from header
            import re
            match = re.search(r'(\d+)\s+(\d+)', header)
            if match:
                width, height = int(match.group(1)), int(match.group(2))
            # Read depth data
            depth = np.frombuffer(f.read(), dtype=np.float32).reshape(height, width)
    else:
        raise ValueError(f"Unsupported depth format: {depth_path.suffix}")

    return depth


def pano_depth_to_3d(rgb_path: str, depth_path: str, output_dir: str,
                     create_mesh: bool = True, mesh_step: int = 2):
    """
    Main function to convert panorama depth to 3D.

    Args:
        rgb_path: path to RGB panorama
        depth_path: path to depth map
        output_dir: output directory
        create_mesh: whether to create mesh (slower but better visualization)
        mesh_step: downsampling step for mesh
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading RGB: {rgb_path}")
    rgb = cv2.imread(str(rgb_path))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    print(f"Loading depth: {depth_path}")
    depth = load_depth(depth_path)

    # Resize if needed
    if rgb.shape[:2] != depth.shape:
        print(f"Resizing RGB from {rgb.shape[:2]} to {depth.shape}")
        rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))

    print(f"RGB shape: {rgb.shape}, Depth shape: {depth.shape}")
    print(f"Depth range: {depth.min():.2f} - {depth.max():.2f} meters")

    # Convert to point cloud
    print("Converting to 3D point cloud...")
    points, colors, valid_mask = equirectangular_to_3d(depth, rgb)
    print(f"Generated {len(points)} points")

    # Save point cloud
    pc_path = output_dir / "pointcloud.ply"
    print(f"Saving point cloud to {pc_path}")
    save_ply(str(pc_path), points, colors)

    # Create mesh
    if create_mesh:
        print(f"Creating mesh (step={mesh_step})...")
        vertices, faces, vertex_colors = create_mesh_from_depth(depth, rgb, step=mesh_step)
        print(f"Generated {len(vertices)} vertices, {len(faces)} faces")

        mesh_path = output_dir / "mesh.ply"
        print(f"Saving mesh to {mesh_path}")
        save_mesh_ply(str(mesh_path), vertices, faces, vertex_colors)

    print("Done!")
    return points, colors


def main():
    parser = argparse.ArgumentParser(description="Convert panorama depth to 3D")
    parser.add_argument("--rgb", type=str, required=True, help="Path to RGB panorama")
    parser.add_argument("--depth", type=str, required=True, help="Path to depth map")
    parser.add_argument("--output", type=str, default="output_3d", help="Output directory")
    parser.add_argument("--no-mesh", action="store_true", help="Skip mesh creation")
    parser.add_argument("--mesh-step", type=int, default=2, help="Mesh downsampling step")

    args = parser.parse_args()

    pano_depth_to_3d(
        rgb_path=args.rgb,
        depth_path=args.depth,
        output_dir=args.output,
        create_mesh=not args.no_mesh,
        mesh_step=args.mesh_step
    )


if __name__ == "__main__":
    main()
