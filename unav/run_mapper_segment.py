#!/usr/bin/env python3
"""
UNav Segment Mapper Pipeline (Perspective Image Version)

This script orchestrates the mapping pipeline for the UNav system
with perspective videos as input. Pipeline stages:

    1. Extract frames from perspective videos
    2. Feature extraction (local & global)
    3. Feature matching with geometric verification
    4. Build global map with GloMap

Typical usage:
    python run_mapper_segment.py <data_temp_root> <data_final_root> <feature_model> <place> <building> <floor>
"""

import sys
import time
from unav.config import UNavConfig

def parse_args() -> tuple:
    """
    Parse command-line arguments.

    Returns:
        Tuple containing:
            data_temp_root (str)
            data_final_root (str)
            feature_model (str)
            place (str)
            building (str)
            floor (str)
    """
    if len(sys.argv) != 7:
        print(
            f"Usage: python {sys.argv[0]} "
            "<data_temp_root> <data_final_root> <feature_model> <place> <building> <floor>"
        )
        sys.exit(1)
    return tuple(sys.argv[1:])

def main():
    """
    Main function for running the UNav segment mapper pipeline.
    """
    # ------------------- Configuration Section -------------------
    (
        data_temp_root,
        data_final_root,
        feature_model,
        place,
        building,
        floor
    ) = parse_args()

    # Initialize global config, get mapping config
    config = UNavConfig(
        data_temp_root=data_temp_root,
        data_final_root=data_final_root,
        mapping_place=place,
        mapping_building=building,
        mapping_floor=floor,
        global_descriptor_model=feature_model,
        mapping_mode="segment"
    )
    mapper_config = config.mapping_config

    pipeline_start = time.time()
    print("Starting UNav segment mapping pipeline (perspective images)...")

    # 1. Extract frames from each video
    # t0 = time.time()
    # from unav.mapper.frame_extractor import extract_frames_from_videos
    # extract_frames_from_videos(mapper_config)
    # print(f"[Stage 1] Frame extraction completed in {time.time() - t0:.2f} seconds.")

    # 2. Feature extraction
    # t1 = time.time()
    # from unav.mapper.feature_extractor import extract_features_from_dir
    # extract_features_from_dir(mapper_config)
    # print(f"[Stage 2] Feature extraction completed in {time.time() - t1:.2f} seconds.")

    # # 3. Feature matching with geometric verification
    # t2 = time.time()
    # from unav.mapper.matcher import generate_and_stream_colmap
    # generate_and_stream_colmap(mapper_config)
    # print(f"[Stage 3] Feature matching and verification completed in {time.time() - t2:.2f} seconds.")

    # 4. Build global map with GloMap
    t3 = time.time()
    from unav.mapper.glomap_runner import run_glomap_segment_pipeline
    run_glomap_segment_pipeline(mapper_config)
    print(f"[Stage 4] GloMap triangulation completed in {time.time() - t3:.2f} seconds.")

    # total_time = time.time() - pipeline_start
    # print(f"UNav segment mapping pipeline finished in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()
