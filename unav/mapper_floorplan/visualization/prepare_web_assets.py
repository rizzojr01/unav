#!/usr/bin/env python3
"""
Prepare lightweight web assets from floor_map/keyframe_floors for fast UI loading.
Outputs:
  floor_map/web/keyframe_points_world.json
  floor_map/web/web_meta.json
"""
import argparse
import json
import os
from pathlib import Path
import random
import numpy as np


def sample_points(points, max_points):
    if max_points <= 0 or len(points) <= max_points:
        return points
    step = max(1, len(points) // max_points)
    return points[::step][:max_points]


def compute_world_points(points_c, pose_cw):
    pose = np.asarray(pose_cw, dtype=np.float64)
    Rcw = pose[:3, :3]
    tcw = pose[:3, 3]
    Rwc = Rcw.T
    t_wc = -(Rwc @ tcw)
    pts = np.asarray(points_c, dtype=np.float64)
    pts_w = (Rwc @ pts.T).T + t_wc
    return pts_w, t_wc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("floor_map_dir", type=str, help=".../floor_map")
    parser.add_argument("--max-points", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    floor_map_dir = Path(args.floor_map_dir)
    keyframe_dir = floor_map_dir / "keyframe_floors"
    if not keyframe_dir.is_dir():
        raise SystemExit(f"[ERROR] keyframe_floors not found: {keyframe_dir}")

    out_dir = floor_map_dir / "web"
    out_dir.mkdir(parents=True, exist_ok=True)

    keyframes_out = {}
    x_min = float("inf")
    x_max = float("-inf")
    z_min = float("inf")
    z_max = float("-inf")

    json_files = sorted(keyframe_dir.glob("keyframe_*.json"))
    for jf in json_files:
        try:
            kf_id = int(jf.stem.split("_")[1])
        except Exception:
            continue
        with open(jf, "r") as f:
            data = json.load(f)

        pose_cw = data.get("pose_cw")
        points_c = data.get("points_camera", [])
        if pose_cw is None or not points_c:
            continue

        points_c = sample_points(points_c, args.max_points)
        pts_w, cam_pos = compute_world_points(points_c, pose_cw)
        pts_w_xz = [[float(p[0]), float(p[2])] for p in pts_w]

        for x, z in pts_w_xz:
            if x < x_min: x_min = x
            if x > x_max: x_max = x
            if z < z_min: z_min = z
            if z > z_max: z_max = z

        keyframes_out[str(kf_id)] = {
            "pose_cw": pose_cw,
            "cam_pos": [float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])],
            "points_w": pts_w_xz,
        }

    if not keyframes_out:
        raise SystemExit("[ERROR] No keyframes processed.")

    if not np.isfinite([x_min, x_max, z_min, z_max]).all():
        x_min, x_max, z_min, z_max = 0.0, 1.0, 0.0, 1.0

    payload = {
        "precomputed_world": True,
        "bounds": {"x_min": x_min, "x_max": x_max, "z_min": z_min, "z_max": z_max},
        "keyframes": keyframes_out,
    }
    out_path = out_dir / "keyframe_points_world.json"
    with open(out_path, "w") as f:
        json.dump(payload, f)

    meta = {
        "keyframe_count": len(keyframes_out),
        "source_keyframe_json": len(json_files),
        "max_points": args.max_points,
    }
    with open(out_dir / "web_meta.json", "w") as f:
        json.dump(meta, f)

    print(f"[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()
