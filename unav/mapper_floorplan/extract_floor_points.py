#!/usr/bin/env python3
"""
Extract floor points from dense point cloud using floor masks.

Usage:
    python extract_floor_points.py <sqlite3_db> <mask_dir> <output_ply>

Example:
    python extract_floor_points.py map.db ./floor_masks/ floor_points.ply
"""

import os
import argparse
import cv2

from utils.database import load_keyframe_poses, load_dense_points
from utils.geometry import project_to_equirectangular, check_in_mask
from utils.pointcloud import save_ply
from utils.visualization import visualize_floor_points_with_cameras, generate_topdown_grid_map, generate_separate_views


def main():
    parser = argparse.ArgumentParser(description='Extract floor points from dense point cloud')
    parser.add_argument('sqlite3_db', type=str, help='Path to SQLite3 database')
    parser.add_argument('mask_dir', type=str, help='Directory containing floor masks (mask0.png, mask1.png, ...)')
    parser.add_argument('output_ply', type=str, help='Output PLY file path')
    parser.add_argument('--mask-pattern', type=str, default='mask{}.png',
                        help='Mask filename pattern (default: mask{}.png)')
    parser.add_argument('--invert-mask', action='store_true',
                        help='Invert mask (use if floor is black instead of white)')
    parser.add_argument('--visualize', action='store_true',
                        help='Save visualization of projected points on masks')
    parser.add_argument('--save-views', action='store_true',
                        help='Save three views of floor points with camera poses')
    parser.add_argument('--save-map', action='store_true',
                        help='Save top-down occupancy grid map')
    parser.add_argument('--door-masks-dir', type=str, default=None,
                        help='Directory containing door masks (optional)')
    parser.add_argument('--draw-doors', action='store_true',
                        help='Draw door markers on the map (default: False)')
    parser.add_argument('--grid-resolution', type=float, default=0.05,
                        help='Grid resolution in meters (default: 0.05)')

    args = parser.parse_args()

    print(f"Loading keyframes from {args.sqlite3_db}...")
    keyframes = load_keyframe_poses(args.sqlite3_db)
    print(f"  Loaded {len(keyframes)} keyframes")

    print(f"Loading dense points from {args.sqlite3_db}...")
    dense_points = load_dense_points(args.sqlite3_db)
    print(f"  Loaded {len(dense_points)} dense points")

    # Load masks
    print(f"Loading masks from {args.mask_dir}...")
    masks = {}
    for kf_id in keyframes.keys():
        mask_path = os.path.join(args.mask_dir, args.mask_pattern.format(kf_id))
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if args.invert_mask:
                mask = 255 - mask
            masks[kf_id] = mask
    print(f"  Loaded {len(masks)} masks")

    if len(masks) == 0:
        print("ERROR: No masks found! Check mask directory and pattern.")
        return

    # Filter points
    print("Filtering points by floor mask...")
    floor_points = []
    floor_colors = []

    stats = {'total': 0, 'no_keyframe': 0, 'no_mask': 0, 'outside_mask': 0, 'floor': 0}

    for point in dense_points:
        stats['total'] += 1

        ref_kf_id = point['ref_keyfrm_id']

        # Skip if reference keyframe not found
        if ref_kf_id not in keyframes:
            stats['no_keyframe'] += 1
            continue

        # Skip if no mask for this keyframe
        if ref_kf_id not in masks:
            stats['no_mask'] += 1
            continue

        kf = keyframes[ref_kf_id]
        mask = masks[ref_kf_id]

        # Get mask dimensions (may differ from original image)
        mask_h, mask_w = mask.shape[:2]

        # Project point to image
        uv = project_to_equirectangular(
            point['pos_w'],
            kf['pose_cw'],
            mask_w,  # Use mask dimensions
            mask_h
        )

        if uv is None:
            stats['outside_mask'] += 1
            continue

        u, v = uv

        # Check if in floor mask
        if check_in_mask(u, v, mask):
            floor_points.append(point['pos_w'])
            floor_colors.append(point['color'])
            stats['floor'] += 1
        else:
            stats['outside_mask'] += 1

    print(f"\nStatistics:")
    print(f"  Total points:        {stats['total']}")
    print(f"  No keyframe:         {stats['no_keyframe']}")
    print(f"  No mask:             {stats['no_mask']}")
    print(f"  Outside floor mask:  {stats['outside_mask']}")
    print(f"  Floor points:        {stats['floor']} ({100*stats['floor']/max(1,stats['total']):.1f}%)")

    # Save output
    if len(floor_points) > 0:
        print(f"\nSaving {len(floor_points)} floor points to {args.output_ply}...")
        save_ply(args.output_ply, floor_points, floor_colors)
        print("Done!")

        # Save three views visualization if requested
        if args.save_views:
            print("\nGenerating three views visualization...")
            views_output = args.output_ply.replace('.ply', '_views.png')
            visualize_floor_points_with_cameras(floor_points, floor_colors, keyframes, views_output)

            # Also generate separate views for web interface
            print("\nGenerating separate views for web interface...")
            views_dir = os.path.dirname(args.output_ply)
            generate_separate_views(floor_points, floor_colors, keyframes, views_dir)

        # Save top-down grid map if requested
        if args.save_map:
            print("\nGenerating top-down grid map...")
            map_output = args.output_ply.replace('.ply', '_map.png')
            generate_topdown_grid_map(floor_points, floor_colors, keyframes,
                                     args.door_masks_dir, map_output, args.grid_resolution, args.draw_doors)
    else:
        print("\nNo floor points found!")


if __name__ == '__main__':
    main()
