"""
Point cloud I/O utilities.
"""


def save_ply(output_path, points, colors):
    """
    Save point cloud to PLY file (ASCII format).

    Args:
        output_path: Output file path
        points: List or array of 3D points (N x 3)
        colors: List or array of colors (N x 3, BGR format)
    """
    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(len(points)):
            p = points[i]
            c = colors[i]
            # Color is BGR, convert to RGB
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[2]} {c[1]} {c[0]}\n")


def save_ply_binary(output_path, points, colors):
    """
    Save point cloud to PLY file (binary format, faster for large clouds).

    Args:
        output_path: Output file path
        points: List or array of 3D points (N x 3)
        colors: List or array of colors (N x 3, BGR format)
    """
    import numpy as np
    import struct

    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)

    # Convert BGR to RGB
    colors_rgb = colors[:, [2, 1, 0]]

    with open(output_path, 'wb') as f:
        # Write header
        header = f"""ply
format binary_little_endian 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        f.write(header.encode('ascii'))

        # Write binary data
        for i in range(len(points)):
            f.write(struct.pack('<fff', *points[i]))
            f.write(struct.pack('<BBB', *colors_rgb[i]))
