#!/usr/bin/env python3
"""
UNav Floor Map Generation Pipeline

This script generates floor point clouds and 2D floor maps from
equirectangular keyframes using DA3 depth estimation and SAM3 floor segmentation.

Pipeline stages:
    1. Slice equirectangular keyframes into perspective images
    2. DA3 depth inference
    3. SAM3 floor mask inference
    4. Generate floor-only point cloud
    5. Generate 2D floor map

Typical usage:
    python -m unav.run_floor_map <data_temp_root> <data_final_root> <place> <building> <floor> [num_images]
"""

import sys
import time
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def parse_args() -> tuple:
    """
    Parse command-line arguments.

    Returns:
        Tuple containing config parameters
    """
    if len(sys.argv) < 6:
        print(
            f"Usage: python {sys.argv[0]} "
            "<data_temp_root> <data_final_root> <place> <building> <floor> [num_images]"
        )
        sys.exit(1)

    data_temp_root = sys.argv[1]
    data_final_root = sys.argv[2]
    place = sys.argv[3]
    building = sys.argv[4]
    floor = sys.argv[5]
    num_images = int(sys.argv[6]) if len(sys.argv) > 6 else 10

    return data_temp_root, data_final_root, place, building, floor, num_images


def main():
    """
    Main function for running the Floor Map generation pipeline.
    """
    # ------------------- Configuration Section -------------------
    (
        data_temp_root,
        data_final_root,
        place,
        building,
        floor,
        num_images
    ) = parse_args()

    from unav.config import UNavFloorMapConfig

    config = UNavFloorMapConfig(
        data_temp_root=data_temp_root,
        data_final_root=data_final_root,
        place=place,
        building=building,
        floor=floor,
        num_images=num_images,
    )

    print("="*80)
    print("UNav Floor Map Generation Pipeline")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Place: {config.place}")
    print(f"  Building: {config.building}")
    print(f"  Floor: {config.floor}")
    print(f"  Num images: {config.num_images}")
    print(f"  Keyframe dir: {config.keyframe_dir}")
    print(f"  Output dir: {config.output_dir}")
    print()

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline_start = time.time()

    # ------------------- Stage 1: Load Data -------------------
    print("="*80)
    print("Stage 1: Load keyframes and trajectory")
    print("="*80)
    t0 = time.time()

    from unav.floor_depth_analyzer.modules.preprocessing import (
        load_keyframes_and_trajectory,
        slice_equirectangular_images
    )

    data = load_keyframes_and_trajectory(
        config.keyframe_dir,
        config.trajectory_file,
        num_images=config.num_images
    )
    print(f"[Stage 1] Completed in {time.time() - t0:.2f} seconds.")

    # ------------------- Stage 2: Slice Panoramas -------------------
    print("\n" + "="*80)
    print("Stage 2: Slice equirectangular images")
    print("="*80)
    t1 = time.time()

    slices_dir = Path(config.output_dir) / "slices"
    slice_info = slice_equirectangular_images(
        image_list=data['image_list'],
        keyframe_dir=config.keyframe_dir,
        poses=data['poses'],
        output_dir=slices_dir,
        yaw_angles=config.yaw_angles,
        pitch_angles=config.pitch_angles,
        fov=config.fov,
    )
    print(f"[Stage 2] Completed in {time.time() - t1:.2f} seconds.")

    # ------------------- Stage 3: Compute Camera Parameters -------------------
    print("\n" + "="*80)
    print("Stage 3: Compute camera parameters")
    print("="*80)
    t2 = time.time()

    from unav.floor_depth_analyzer.utils import compute_slice_camera_params

    extrinsics = []
    intrinsics = []

    for T_cw, (yaw, pitch, fov, width, height) in zip(
        slice_info['camera_poses'],
        slice_info['slice_params']
    ):
        extrinsic, intrinsic = compute_slice_camera_params(
            T_cw, yaw, pitch, fov, width, height
        )
        extrinsics.append(extrinsic)
        intrinsics.append(intrinsic)

    extrinsics = np.stack(extrinsics, axis=0)
    intrinsics = np.stack(intrinsics, axis=0)

    print(f"  Computed {len(extrinsics)} camera parameters")
    print(f"[Stage 3] Completed in {time.time() - t2:.2f} seconds.")

    # ------------------- Stage 4: DA3 Depth Inference -------------------
    print("\n" + "="*80)
    print("Stage 4: DA3 depth inference")
    print("="*80)
    t3 = time.time()

    from unav.floor_depth_analyzer.modules.depth_anything_v3 import (
        load_da3_model,
        run_da3_inference
    )

    da3_model = load_da3_model(device=device)

    prediction = run_da3_inference(
        model=da3_model,
        image_paths=slice_info['slice_paths'],
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        export_dir=None,
        export_format=None,
    )

    depths = prediction.depth
    if torch.is_tensor(depths):
        depths = depths.cpu().numpy()

    confs = prediction.conf
    if torch.is_tensor(confs):
        confs = confs.cpu().numpy()

    # Free GPU memory
    del da3_model
    torch.cuda.empty_cache()

    print(f"  Depth shape: {depths.shape}")
    print(f"[Stage 4] Completed in {time.time() - t3:.2f} seconds.")

    # ------------------- Stage 5: SAM3 Floor Mask Inference -------------------
    print("\n" + "="*80)
    print("Stage 5: SAM3 floor mask inference")
    print("="*80)
    t4 = time.time()

    from unav.floor_depth_analyzer.modules.sam3 import (
        load_sam3_model,
        generate_floor_masks
    )

    sam3_model, sam3_processor = load_sam3_model(device=device)
    floor_masks = generate_floor_masks(
        slice_info['slice_paths'],
        sam3_model,
        sam3_processor,
        device
    )

    # Free GPU memory
    del sam3_model, sam3_processor
    torch.cuda.empty_cache()

    print(f"[Stage 5] Completed in {time.time() - t4:.2f} seconds.")

    # ------------------- Stage 6: Generate Floor Point Cloud -------------------
    print("\n" + "="*80)
    print("Stage 6: Generate floor point cloud")
    print("="*80)
    t5 = time.time()

    from unav.floor_depth_analyzer.modules.pointcloud import (
        depth_to_pointcloud_with_mask,
        save_pointcloud_glb
    )

    all_points = []
    valid_slices = 0

    for i in tqdm(range(len(depths)), desc="Processing slices"):
        slice_name = Path(slice_info['slice_paths'][i]).name

        if slice_name not in floor_masks:
            continue

        floor_mask = floor_masks[slice_name]
        if not floor_mask.any():
            continue

        points = depth_to_pointcloud_with_mask(
            depth=depths[i],
            conf=confs[i],
            floor_mask=floor_mask,
            intrinsic=intrinsics[i],
            extrinsic=extrinsics[i],
            conf_thresh=config.conf_thresh,
        )

        if len(points) > 0:
            all_points.append(points)
            valid_slices += 1

    if len(all_points) == 0:
        print("[Error] No floor points generated!")
        sys.exit(1)

    floor_points = np.vstack(all_points)
    print(f"  Valid slices: {valid_slices}/{len(depths)}")
    print(f"  Total floor points: {len(floor_points):,}")

    # Save point cloud
    save_pointcloud_glb(floor_points, config.floor_pointcloud_glb)
    np.save(config.floor_points_npy, floor_points)

    print(f"[Stage 6] Completed in {time.time() - t5:.2f} seconds.")

    # ------------------- Stage 7: Generate 2D Floor Map -------------------
    print("\n" + "="*80)
    print("Stage 7: Generate 2D floor map")
    print("="*80)
    t6 = time.time()

    from unav.floor_depth_analyzer.scripts.generate_floor_map import (
        generate_floor_map_from_pointcloud
    )

    floor_map, map_info = generate_floor_map_from_pointcloud(
        points=floor_points,
        resolution=config.resolution,
        output_dir=config.floor_map_dir,
    )

    print(f"[Stage 7] Completed in {time.time() - t6:.2f} seconds.")

    # ------------------- Save Additional Data -------------------
    print("\n" + "="*80)
    print("Saving additional data")
    print("="*80)

    np.save(Path(config.output_dir) / "all_depths.npy", depths)
    np.save(Path(config.output_dir) / "all_confs.npy", confs)
    np.save(Path(config.output_dir) / "slice_paths.npy", np.array(slice_info['slice_paths']))
    np.save(Path(config.output_dir) / "extrinsics.npy", extrinsics)
    np.save(Path(config.output_dir) / "intrinsics.npy", intrinsics)
    np.save(Path(config.output_dir) / "floor_masks.npy", floor_masks)

    print(f"  Data saved to: {config.output_dir}")

    # ------------------- Summary -------------------
    total_time = time.time() - pipeline_start
    print("\n" + "="*80)
    print("Pipeline Complete!")
    print("="*80)
    print(f"\nTotal time: {total_time:.2f} seconds")
    print(f"\nOutput files:")
    print(f"  Floor point cloud: {config.floor_pointcloud_glb}")
    print(f"  Floor map: {config.floor_map_dir}/floor_map.png")
    print(f"\nView results:")
    print(f"  eog {config.floor_map_dir}/floor_map_visualization.png")
    print(f"  Online 3D viewer: https://3dviewer.net/")
    print()


if __name__ == "__main__":
    main()
