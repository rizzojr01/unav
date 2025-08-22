import os
import logging
from pathlib import Path
import h5py
import numpy as np
from typing import List, Dict, Any

from unav.core.colmap.database import COLMAPDatabase
from unav.core.colmap.read_write_model import CAMERA_MODEL_NAMES
from unav.core.colmap.utils_pose import load_colmap_images_file_qt

def write_colmap_camera_and_images(
    perspective_data: List[Dict[str, Any]],
    output_w: int,
    output_h: int,
    fov: float,
    camera_file: str,
    image_file: str
) -> None:
    """
    Write COLMAP camera and image files for a perspective dataset.

    Args:
        perspective_data (List[Dict]): List of perspective slice dicts.
        output_w (int): Output image width.
        output_h (int): Output image height.
        fov (float): Field of view in degrees.
        camera_file (str): Output path for 'cameras.txt'.
        image_file (str): Output path for 'images.txt'.
    """
    fov_rad = np.radians(fov)
    fx = fy = (output_w / 2) / np.tan(fov_rad / 2)
    cx = output_w / 2
    cy = output_h / 2

    # Write camera intrinsics to file
    with open(camera_file, 'w') as fc:
        fc.write("# Camera list\n")
        fc.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS...\n")
        fc.write(f"1 PINHOLE {output_w} {output_h} {fx} {fy} {cx} {cy}\n")

    # Write image poses to file
    with open(image_file, 'w') as fi:
        fi.write("# Image list\n")
        fi.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for i, d in enumerate(perspective_data):
            qw, qx, qy, qz = d["q_wxyz"]
            tx, ty, tz = d["t_c"]
            name = d["image_name"]
            fi.write(f"{i+1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {name}\n\n")

def load_colmap_cameras_file(cameras_file: str) -> Dict[int, dict]:
    """
    Load a COLMAP-formatted cameras.txt file.

    Args:
        cameras_file (str): Path to cameras.txt.

    Returns:
        dict: Mapping {camera_id: {'model_id', 'width', 'height', 'params'}}
    """
    cameras = {}
    with open(cameras_file, 'r') as f:
        for line in f:
            if line.strip() == '' or line.startswith('#'):
                continue
            tokens = line.strip().split()
            if len(tokens) < 5:
                continue

            camera_id = int(tokens[0])
            model_name = tokens[1]  # e.g. "PINHOLE"
            model_id  = CAMERA_MODEL_NAMES[model_name].model_id
            width     = int(tokens[2])
            height    = int(tokens[3])
            params    = list(map(float, tokens[4:]))

            cameras[camera_id] = {
                'model_id': model_id,
                'width': width,
                'height': height,
                'params': params
            }
    logging.info(f"[UNav] Loaded {len(cameras)} cameras from {cameras_file}")
    return cameras

def create_colmap_database_with_known_poses(
    database_path: Path,
    local_feature_file: Path,
    matches_file: Path,
    cameras_txt: Path,
    images_txt: Path,
    pairs_txt: Path,
    overwrite: bool = True
) -> None:
    """
    Create and populate a COLMAP SQLite database using known poses and feature matches.

    Args:
        database_path (Path): Path to COLMAP database file to create.
        local_feature_file (Path): HDF5 file of local features.
        matches_file (Path): HDF5 file of keypoint matches.
        cameras_txt (Path): COLMAP-formatted cameras.txt file.
        images_txt (Path): COLMAP-formatted images.txt file.
        pairs_txt (Path): List of matched image pairs.
    """
    logging.info(f"[UNav] Creating COLMAP database with known poses from TXT files...")
    if overwrite and os.path.exists(database_path):
        os.remove(database_path)
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    # --- Import camera intrinsics ---
    cameras = load_colmap_cameras_file(str(cameras_txt))
    for cid, cam in cameras.items():
        db.add_camera(
            cam['model_id'],
            cam['width'],
            cam['height'],
            cam['params'],
            camera_id=cid,
            prior_focal_length=True
        )

    # --- Import images (known poses) ---
    poses = load_colmap_images_file_qt(str(images_txt))
    image_id_map = {}
    for name, data in poses.items():
        db.add_image(
            name,
            camera_id=1,
            prior_q=data['qvec'],
            prior_t=data['tvec'],
            image_id=data['image_id']
        )
        image_id_map[name] = data['image_id']

    # --- Import keypoints ---
    with h5py.File(local_feature_file, 'r') as f_feat:
        for name in poses.keys():
            kpts = f_feat[name]['keypoints'][:]
            kpts = kpts.astype(np.float32) + 0.5  # COLMAP pixel-center origin
            db.add_keypoints(image_id_map[name], kpts)

    # --- Import matches ---
    with open(pairs_txt, 'r') as f:
        pairs = [line.strip().split() for line in f.readlines() if line.strip()]

    with h5py.File(matches_file, 'r') as f_match:
        for img1_name, img2_name in pairs:
            if img1_name not in image_id_map or img2_name not in image_id_map:
                logging.error(f"[UNav] Unknown image in pairs.txt: {img1_name} or {img2_name}")
                continue

            img1_id = image_id_map[img1_name]
            img2_id = image_id_map[img2_name]

            key = f"{img1_name}_{img2_name}"
            if key not in f_match:
                key = f"{img2_name}_{img1_name}"
                if key not in f_match:
                    logging.error(f"[UNav] Could not find matches for pair {img1_name} {img2_name}")
                    continue

            matches = f_match[key]['matches0'][:]
            db.add_matches(img1_id, img2_id, matches)
            db.add_two_view_geometry(
                img1_id,
                img2_id,
                matches,
                F=np.eye(3),
                E=np.eye(3),
                H=np.eye(3),
                config=2
            )
    db.commit()
    db.close()
    logging.info(f"[UNav] COLMAP database created and populated.")

def create_colmap_database_without_poses(
    database_path: Path,
    local_feature_file: Path,
    matches_file: Path,
    pairs_txt: Path,
    fov_deg: float = 70.0,
    camera_model: str = "OPENCV",
    overwrite: bool = True
) -> None:
    """
    Prepare a COLMAP database in segment mode (no prior pose, no external cameras.txt/images.txt):
    - Automatically infer camera intrinsics from feature images.
    - Add a single camera (SIMPLE_RADIAL, fx, cx, cy, k) based on FOV and feature h5.
    - Add images (no pose) and import keypoints.
    - Import matches between images and write to both 'matches' and 'two_view_geometries' tables.
    All fields are consistent with official COLMAP database format.
    """
    logging.info(f"[UNav] Creating COLMAP segment database: {database_path}")

    if overwrite and database_path.exists():
        database_path.unlink()
    db = COLMAPDatabase.connect(str(database_path))
    db.create_tables()

    # --- Infer camera intrinsics from first available image in local_feature_file ---
    with h5py.File(local_feature_file, 'r') as f_feat:
        img_names = list(f_feat.keys())
        for name in img_names:
            if 'image_size' in f_feat[name]:
                h, w = f_feat[name]['image_size'][:]
                break
        else:
            raise RuntimeError("[UNav] No image_size field found in local_feature_file.")
        # Calculate fx, fy, cx, cy for OPENCV
        fov_rad = np.radians(fov_deg)
        fx = fy = (w / 2) / np.tan(fov_rad / 2)
        cx = w / 2
        cy = h / 2
        # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
        cam_params = np.array([fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        model_id = CAMERA_MODEL_NAMES[camera_model].model_id

    # --- Add camera (camera_id=1) ---
    camera_id = db.add_camera(
        model_id,
        width=int(w),
        height=int(h),
        params=cam_params,
        prior_focal_length=True
    )

    # --- Add images (one per key) ---
    image_id_map = {}
    for idx, name in enumerate(img_names):
        image_id = db.add_image(
            name=name,
            camera_id=camera_id  # All images share the same camera
        )
        image_id_map[name] = image_id

    # --- Add keypoints ---
    with h5py.File(local_feature_file, 'r') as f_feat:
        for name in img_names:
            kpts = f_feat[name]['keypoints'][:]
            kpts = kpts.astype(np.float32) + 0.5  # COLMAP uses pixel-center origin
            db.add_keypoints(image_id_map[name], kpts)

    # --- Add matches and two_view_geometries ---
    with open(pairs_txt, 'r') as f:
        pairs = [line.strip().split() for line in f.readlines() if line.strip()]

    with h5py.File(matches_file, 'r') as f_match:
        for img1_name, img2_name in pairs:
            if img1_name not in image_id_map or img2_name not in image_id_map:
                logging.error(f"[UNav] Unknown image in pairs.txt: {img1_name} or {img2_name}")
                continue
            img1_id = image_id_map[img1_name]
            img2_id = image_id_map[img2_name]
            key = f"{img1_name}_{img2_name}"
            if key not in f_match:
                key = f"{img2_name}_{img1_name}"
                if key not in f_match:
                    logging.error(f"[UNav] Could not find matches for pair {img1_name} {img2_name}")
                    continue
            matches = f_match[key]['matches0'][:]
            db.add_matches(img1_id, img2_id, matches)
            db.add_two_view_geometry(
                img1_id,
                img2_id,
                matches,
                F=np.eye(3),  # Placeholder
                E=np.eye(3),  # Placeholder
                H=np.eye(3),  # Placeholder
                config=2      # 2 = confirmed match
            )

    db.commit()
    db.close()
    logging.info(f"[UNav] Segment COLMAP database created at {database_path}")
