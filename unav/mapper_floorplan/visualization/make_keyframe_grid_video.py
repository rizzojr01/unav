#!/usr/bin/env python3
"""
Create a 2x2 grid video per keyframe:
[ keyframe | mask ]
[ depth    | keyframe_vis ]

Assumes outputs follow mapper_floorplan config.yaml.
"""

import argparse
import os
import re
import sys
import subprocess
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

# Ensure project root is on sys.path for utils imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_keyframe_id(filename: str):
    m = re.match(r"image(\d+)\.(png|jpg|jpeg)$", filename)
    if not m:
        return None
    return int(m.group(1))


def load_mask(mask_path: Path, target_size):
    if not mask_path.exists():
        return make_placeholder(target_size, "mask missing")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return make_placeholder(target_size, "mask read error")
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask_rgb


def load_depth(depth_path: Path, target_size):
    if not depth_path.exists():
        return make_placeholder(target_size, "depth missing")
    depth = np.load(str(depth_path))
    if depth.ndim == 3:
        depth = depth.squeeze(0)
    depth = depth.astype(np.float32)

    valid = np.isfinite(depth)
    if valid.any():
        d = depth[valid]
        lo = np.percentile(d, 5)
        hi = np.percentile(d, 95)
        if hi <= lo:
            hi = lo + 1e-6
        depth_norm = np.clip((depth - lo) / (hi - lo), 0, 1)
    else:
        depth_norm = np.zeros_like(depth, dtype=np.float32)

    depth_u8 = (depth_norm * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
    depth_color = cv2.resize(depth_color, target_size, interpolation=cv2.INTER_AREA)
    return depth_color


def load_keyframe_vis(vis_path: Path, target_size):
    if not vis_path.exists():
        return make_placeholder(target_size, "keyframe_vis missing")
    img = cv2.imread(str(vis_path))
    if img is None:
        return make_placeholder(target_size, "keyframe_vis read error")
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img


def make_placeholder(target_size, text):
    w, h = target_size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, text, (10, max(30, h // 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)
    return img


def main():
    parser = argparse.ArgumentParser(description="Create 2x2 grid keyframe video")
    parser.add_argument("--place", type=str, default=None)
    parser.add_argument("--building", type=str, default=None)
    parser.add_argument("--floor", type=str, default=None)
    parser.add_argument("--config", type=str, default="/home/unav/Desktop/unav/unav/mapper_floorplan/config.yaml")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--align-camera-up", action="store_true",
                        help="(Deprecated) No-op; kept for backward compatibility")
    parser.add_argument("--max-grid-width", type=int, default=1920,
                        help="Max output video width (grid); tiles will be downscaled to fit")
    parser.add_argument("--max-grid-height", type=int, default=1080,
                        help="Max output video height (grid); tiles will be downscaled to fit")
    parser.add_argument("--use-ffmpeg", action="store_true",
                        help="Force using ffmpeg to assemble video (writes PNG frames first)")

    args = parser.parse_args()

    config = load_config(Path(args.config))

    data_root = Path(config["data_root"])
    place = args.place or config.get("default_place", "")
    building = args.building or config.get("default_building", "")
    floor = args.floor or config.get("default_floor", "")

    if not place or not building or not floor:
        raise SystemExit("Missing place/building/floor. Provide args or set defaults in config.yaml")

    slam_dir = config["dir_names"]["slam"]
    keyframes_dir = config["dir_names"].get("keyframes", "keyframes")
    floor_map_dir = config["dir_names"]["floor_map"]
    mask_dir_name = config["dir_names"]["mask"]
    depth_dir_name = config["dir_names"]["depth"]
    keyframe_vis_dir_name = config["dir_names"]["keyframe_vis"]

    mask_pattern = config.get("extraction", {}).get("mask_pattern", "image{}_floor_mask.png")
    depth_pattern = config.get("da2", {}).get("depth_pattern", "image{}.npy")
    db_filename = config["file_names"]["database"]

    base_floor = data_root / place / building / floor
    keyframes_path = base_floor / slam_dir / keyframes_dir
    outputs_root = base_floor / floor_map_dir
    masks_path = outputs_root / mask_dir_name
    depth_path = outputs_root / depth_dir_name
    keyframe_vis_path = outputs_root / keyframe_vis_dir_name
    db_path = base_floor / slam_dir / db_filename

    if not keyframes_path.exists():
        raise SystemExit(f"Keyframes directory not found: {keyframes_path}")

    keyframe_files = sorted([p for p in keyframes_path.iterdir() if p.is_file()])
    keyframe_ids = []
    for p in keyframe_files:
        kf_id = parse_keyframe_id(p.name)
        if kf_id is not None:
            keyframe_ids.append(kf_id)

    keyframe_ids = sorted(keyframe_ids)
    if args.max_frames:
        keyframe_ids = keyframe_ids[: args.max_frames]

    if not keyframe_ids:
        raise SystemExit("No keyframe images found in keyframes directory")

    # Read first keyframe to define tile size
    first_kf_path = keyframes_path / f"image{keyframe_ids[0]}.png"
    first_img = cv2.imread(str(first_kf_path))
    if first_img is None:
        raise SystemExit(f"Failed to read keyframe: {first_kf_path}")

    tile_h, tile_w = first_img.shape[:2]
    grid_w = tile_w * 2
    grid_h = tile_h * 2
    scale = min(
        1.0,
        args.max_grid_width / grid_w if grid_w > 0 else 1.0,
        args.max_grid_height / grid_h if grid_h > 0 else 1.0,
    )
    tile_w = int(tile_w * scale)
    tile_h = int(tile_h * scale)
    tile_size = (tile_w, tile_h)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        candidates = [(str(output_path), "mp4v")]
    else:
        output_path = outputs_root / f"{place}_{building}_{floor}_keyframe_grid.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        candidates = [
            (str(output_path), "mp4v"),
            (str(output_path.with_suffix(".avi")), "XVID"),
            (str(output_path.with_suffix(".avi")), "MJPG"),
        ]

    writer = None
    if not args.use_ffmpeg:
        for path, codec in candidates:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(path, fourcc, args.fps, (tile_w * 2, tile_h * 2))
            if writer.isOpened():
                output_path = Path(path)
                break

    use_ffmpeg = args.use_ffmpeg or (writer is None) or (not writer.isOpened())
    if use_ffmpeg:
        output_path = output_path.with_suffix(".mp4")
        frames_dir = outputs_root / "keyframe_grid_frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing video to: {output_path}")
    print(f"Frames: {len(keyframe_ids)}, FPS: {args.fps}")
    if use_ffmpeg:
        print(f"Using ffmpeg. Frames dir: {frames_dir}")

    # Rotation disabled by request; keep database load out to avoid unnecessary work.
    keyframes_pose = None

    for idx, kf_id in enumerate(tqdm(keyframe_ids, desc="Rendering frames")):
        keyframe_file = keyframes_path / f"image{kf_id}.png"
        if not keyframe_file.exists():
            keyframe_file = keyframes_path / f"image{kf_id}.jpg"

        keyframe_img = cv2.imread(str(keyframe_file))
        if keyframe_img is None:
            keyframe_img = make_placeholder(tile_size, f"keyframe {kf_id} missing")
        else:
            keyframe_img = cv2.resize(keyframe_img, tile_size, interpolation=cv2.INTER_AREA)

        mask_file = masks_path / mask_pattern.format(kf_id)
        depth_file = depth_path / depth_pattern.format(kf_id)
        keyframe_vis_file = keyframe_vis_path / f"keyframe_{kf_id}_topdown.png"

        mask_img = load_mask(mask_file, tile_size)
        depth_img = load_depth(depth_file, tile_size)
        keyframe_vis_img = load_keyframe_vis(keyframe_vis_file, tile_size)

        top = np.hstack([keyframe_img, mask_img])
        bottom = np.hstack([depth_img, keyframe_vis_img])
        frame = np.vstack([top, bottom])

        if use_ffmpeg:
            frame_path = frames_dir / f"frame_{idx:06d}.png"
            cv2.imwrite(str(frame_path), frame)
        else:
            writer.write(frame)

    if not use_ffmpeg:
        writer.release()
        print("Done.")
        return

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found. Install ffmpeg or run without --use-ffmpeg.")

    cmd = [
        ffmpeg,
        "-y",
        "-framerate", str(args.fps),
        "-i", str(frames_dir / "frame_%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("ffmpeg failed to assemble video")

    print("Done.")


if __name__ == "__main__":
    main()
