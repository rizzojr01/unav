import torch
import numpy as np
from unav.core.feature_filter import match_query_to_database, ransac_filter

from typing import Dict, Any, Callable, List, Tuple

def batch_local_matching_and_ransac(
    query_local_feat: Dict[str, Any],
    candidates_data: Dict[str, Dict[str, Any]],
    matcher: Callable,
    device: torch.device,
    feature_score_threshold: float = 0.09,
    min_inliers: int = 6
) -> Tuple[Tuple[str, str, str], Dict[str, np.ndarray], List[Dict[str, Any]]]:
    """
    Perform batch local matching and geometric verification (RANSAC) for VPR candidates.
    Returns only the best map region's 2D-3D matches for robust downstream PnP/refinement.

    Args:
        query_local_feat (Dict[str, Any]): Query local features (keypoints, descriptors, etc.).
        candidates_data (Dict[str, Dict[str, Any]]): 
            Dictionary of candidates, keyed by ref_image_name.
            Each value must contain:
                - 'frame': COLMAP frame dictionary.
                - 'local_feat': Local feature dictionary.
                - 'map_key': (place, building, floor) tuple.
                - 'score': Retrieval similarity score (float).
        matcher (Callable): Local feature matcher function/class.
        device (torch.device): Torch device for computation.
        feature_score_threshold (float): Keypoint score threshold for filtering (default: 0.09).
        min_inliers (int): Minimum required RANSAC inliers (default: 6).

    Returns:
        Tuple[Tuple[str, str, str], Dict[str, np.ndarray], List[Dict[str, Any]]]:
            - best_map_key: The selected best (place, building, floor) tuple after geometric verification.
            - pnp_pairs: Dict containing 'kpts2d', 'pts3d', and other arrays for the best region, for downstream PnP/refinement.
            - results: List of matching info for all candidates in the best region.
    """

    # 1. Local feature matching
    ref_img_names = list(candidates_data.keys())
    db_feats = [
        {
            "descriptors": torch.tensor(candidates_data[name]["local_feat"]["descriptors"], device=device),
            "keypoints": torch.tensor(candidates_data[name]["local_feat"]["keypoints"], device=device),
            "scores": torch.tensor(candidates_data[name]["local_feat"]["scores"], device=device),
            "image_size": torch.tensor(candidates_data[name]["local_feat"]["image_size"], device=device),
        }
        for name in ref_img_names
    ]
    feat1 = {
        "descriptors": torch.tensor(query_local_feat["descriptors"], device=device),
        "keypoints": torch.tensor(query_local_feat["keypoints"], device=device),
        "scores": torch.tensor(query_local_feat["scores"], device=device),
        "image_size": torch.tensor(query_local_feat["image_size"], device=device),
    }

    # 2. Batch matching and RANSAC
    names2, p0_idx, p1_idx, scores_list = match_query_to_database(
        feat1, db_feats, ref_img_names,
        local_feature_matcher=matcher,
        device=device,
        feature_score_threshold=feature_score_threshold
    )

    if not names2:
        return None, {"image_points": np.zeros((0, 2)), "object_points": np.zeros((0, 3))}, []

    k0_np = np.array(query_local_feat["keypoints"])
    k0_list = [k0_np] * len(names2)
    k1_list = [
        np.array(candidates_data[name]["local_feat"]["keypoints"]) for name in names2
    ]

    valid_mask, in0, in1, inlier_scores = ransac_filter(
        p0_idx, p1_idx, k0_list, k1_list, scores_list, device, threshold=min_inliers
    )

    # 3. Group by map_key for robust region selection
    grouped = dict()
    for idx, name in enumerate(names2):
        if not valid_mask[idx] or in0[idx] is None or len(in0[idx]) < min_inliers:
            continue

        candidate = candidates_data[name]
        map_key = candidate["map_key"]
        ref_frame = candidate["frame"]
        query_idxs = in0[idx]
        ref_idxs = in1[idx]

        # Only keep valid 3D-2D correspondences
        valid_pairs = [
            (qi, ri)
            for qi, ri in zip(query_idxs, ref_idxs)
            if ref_frame["points3D_xyz"][ri] is not None
        ]
        if len(valid_pairs) < min_inliers:
            continue

        query_valid_idx, ref_valid_idx = zip(*valid_pairs)
        image_points = k0_np[list(query_valid_idx)]
        object_points = np.array([ref_frame["points3D_xyz"][ri] for ri in ref_valid_idx])

        # Group by map_key
        if map_key not in grouped:
            grouped[map_key] = {
                "all_image_points": [],
                "all_object_points": [],
                "results": [],
                "total_inliers": 0,
            }
        grouped[map_key]["all_image_points"].append(image_points)
        grouped[map_key]["all_object_points"].append(object_points)
        grouped[map_key]["total_inliers"] += len(object_points)
        grouped[map_key]["results"].append({
            "ref_image_name": name,
            "map_key": map_key,
            "score": candidate["score"],
            "inliers": len(object_points),
            "query_idxs": list(query_valid_idx),
            "ref_idxs": list(ref_valid_idx),
            "object_points": object_points,
            "image_points": image_points,
            "debug": {"raw_inliers": len(query_idxs)}
        })

    if not grouped:
        return None, {"image_points": np.zeros((0, 2)), "object_points": np.zeros((0, 3))}, []

    # 4. Choose best_map_key by max inliers
    best_map_key = max(grouped, key=lambda k: grouped[k]["total_inliers"])
    block = grouped[best_map_key]

    if block["all_image_points"] and block["all_object_points"]:
        pnp_image_points = np.concatenate(block["all_image_points"], axis=0)
        pnp_object_points = np.concatenate(block["all_object_points"], axis=0)
    else:
        pnp_image_points = np.zeros((0, 2))
        pnp_object_points = np.zeros((0, 3))

    pnp_pairs = {
        "image_points": pnp_image_points,
        "object_points": pnp_object_points
    }
    return best_map_key, pnp_pairs, block["results"]


def mast3r_matching_and_pnp(
    query_img_path: str,
    candidates_data,
    mast3r_matcher,
    colmap_models,
    max_nn_dist: float = 20.0,
    min_inliers: int = 6,
    max_candidates: int = 10,
    early_stop_inliers: int = 50,
):
    """
    MASt3R dense matching replacement for batch_local_matching_and_ransac().

    For each candidate:
      1. MASt3R dense match (query, DB image) → (query_2d, db_2d)
      2. NN lookup: db_2d → colmap points2D_xy → world points3D_xyz
      3. Collect (image_points, object_points) for PnP

    Returns same signature as batch_local_matching_and_ransac():
      (best_map_key, pnp_pairs, results)
    """
    from scipy.spatial import cKDTree
    import os

    ref_img_names = list(candidates_data.keys())[:max_candidates]
    grouped = {}

    # Build DB image paths for batch matching
    db_paths = []
    db_names = []
    for name in ref_img_names:
        candidate = candidates_data[name]
        place, building, floor = candidate["map_key"]
        db_img_path = f'/mnt/data/UNav-IO/temp/{place}/{building}/{floor}/perspectives/{name}'
        db_paths.append(db_img_path)
        db_names.append(name)

    # Batch MASt3R inference (all pairs at once)
    if hasattr(mast3r_matcher, 'match_batch') and len(db_paths) > 1:
        batch_results = mast3r_matcher.match_batch(query_img_path, db_paths)
    else:
        batch_results = [mast3r_matcher.match_pair(query_img_path, p) for p in db_paths]

    for idx, name in enumerate(db_names):
        candidate = candidates_data[name]
        map_key = candidate["map_key"]
        ref_frame = candidate["frame"]

        result = batch_results[idx]
        if result is None:
            query_2d, db_2d, conf = None, None, None
        else:
            query_2d, db_2d, conf = result
        if query_2d is None or len(query_2d) < min_inliers:
            continue

        # Colmap 3D lookup via NN
        colmap_2d = ref_frame['points2D_xy']
        colmap_3d = ref_frame['points3D_xyz']
        valid_mask = np.array([p is not None for p in colmap_3d])
        valid_idx = np.where(valid_mask)[0]

        if len(valid_idx) < min_inliers:
            continue

        valid_2d = colmap_2d[valid_idx]
        valid_3d = np.array([colmap_3d[i] for i in valid_idx])

        tree = cKDTree(valid_2d)
        dists, nn_idx = tree.query(db_2d, k=1)
        close_mask = dists < max_nn_dist

        if close_mask.sum() < min_inliers:
            continue

        image_points = query_2d[close_mask]
        object_points = valid_3d[nn_idx[close_mask]]
        n_inliers = len(image_points)

        if map_key not in grouped:
            grouped[map_key] = {
                "all_image_points": [],
                "all_object_points": [],
                "results": [],
                "total_inliers": 0,
            }
        grouped[map_key]["all_image_points"].append(image_points)
        grouped[map_key]["all_object_points"].append(object_points)
        grouped[map_key]["total_inliers"] += n_inliers
        grouped[map_key]["results"].append({
            "ref_image_name": name,
            "map_key": map_key,
            "score": candidate.get("score", 0),
            "inliers": n_inliers,
            "object_points": object_points,
            "image_points": image_points,
        })

        # Early stopping: if we have enough inliers, stop matching more candidates
        if grouped[map_key]["total_inliers"] >= early_stop_inliers:
            break

    if not grouped:
        return None, {"image_points": np.zeros((0, 2)), "object_points": np.zeros((0, 3))}, []

    best_map_key = max(grouped, key=lambda k: grouped[k]["total_inliers"])
    block = grouped[best_map_key]

    pnp_image_points = np.concatenate(block["all_image_points"], axis=0)
    pnp_object_points = np.concatenate(block["all_object_points"], axis=0)

    return best_map_key, {
        "image_points": pnp_image_points,
        "object_points": pnp_object_points,
    }, block["results"]


def mast3r_relpose_localization(
    query_img_path: str,
    candidates_data,
    mast3r_matcher,
    colmap_models,
    transform_matrices,
    min_inliers: int = 10,
    max_candidates: int = 5,
):
    """
    Map-free localization: MASt3R 3D pointmap + poselib in local frame + compose with ref world pose.
    No colmap 3D points needed — only ref images with known poses.

    Returns same signature as batch_local_matching_and_ransac():
      (best_map_key, pnp_pairs, results)
    """
    import os, poselib
    from scipy.spatial.transform import Rotation as R

    ref_img_names = list(candidates_data.keys())[:max_candidates]
    estimates = []

    for name in ref_img_names:
        candidate = candidates_data[name]
        map_key = candidate["map_key"]
        ref_frame = candidate["frame"]

        place, building, floor = map_key
        db_img_path = None
        for base in ['/mnt/data/UNav-IO/temp', '/mnt/data/UNav-IO/data', '/data']:
            p = f'{base}/{place}/{building}/{floor}/perspectives/{name}'
            if os.path.exists(p):
                db_img_path = p
                break
        if db_img_path is None:
            continue

        # MASt3R matching + get 3D pointmap
        result = mast3r_matcher.match_pair_with_pts3d(query_img_path, db_img_path)
        if result is None:
            continue

        query_2d, pts3d_matched, n_matches = result
        if n_matches < min_inliers:
            continue

        # PnP in MASt3R local frame (no intrinsics needed)
        import cv2 as _cv2
        q_img = _cv2.imread(query_img_path)
        qh, qw = q_img.shape[:2]
        pp = np.array([qw / 2.0, qh / 2.0])

        try:
            pose, info = poselib.estimate_1D_radial_absolute_pose(
                query_2d - pp, pts3d_matched.astype(np.float64),
                {"max_reproj_error": 12.0, "max_iterations": 10000}
            )
        except Exception:
            continue

        n_inliers = int(np.sum(info['inliers'])) if 'inliers' in info else 0
        if n_inliers < min_inliers:
            continue

        # Query camera center in MASt3R local frame
        q_rot = R.from_quat([pose.q[1], pose.q[2], pose.q[3], pose.q[0]])
        q_center_local = -q_rot.as_matrix().T @ np.array(pose.t)

        # Transform to world via ref's colmap pose
        ref_qvec = ref_frame['qvec']
        ref_tvec = ref_frame['tvec']
        ref_quat_xyzw = [ref_qvec[1], ref_qvec[2], ref_qvec[3], ref_qvec[0]]
        ref_rot = R.from_quat(ref_quat_xyzw)
        ref_rmat = ref_rot.as_matrix()
        ref_center = -ref_rmat.T @ ref_tvec

        q_center_world = ref_center + ref_rmat.T @ q_center_local

        # Compose rotation for heading
        q_rmat_world = ref_rmat.T @ q_rot.as_matrix()
        q_rot_world = R.from_matrix(q_rmat_world)
        q_quat_xyzw = q_rot_world.as_quat()
        q_qvec_wxyz = np.array([q_quat_xyzw[3], q_quat_xyzw[0], q_quat_xyzw[1], q_quat_xyzw[2]])
        q_tvec = -q_rmat_world @ q_center_world

        estimates.append({
            "world_pos": q_center_world,
            "qvec": q_qvec_wxyz,
            "tvec": q_tvec,
            "n_inliers": n_inliers,
            "map_key": map_key,
            "ref_image_name": name,
            "score": candidate.get("score", 0),
        })

    if not estimates:
        return None, {"image_points": np.zeros((0, 2)), "object_points": np.zeros((0, 3))}, []

    # Weighted average position
    weights = np.array([e['n_inliers'] for e in estimates], dtype=float)
    weights /= weights.sum()
    avg_pos = sum(w * e['world_pos'] for w, e in zip(weights, estimates))

    # Heading: outlier rejection + weighted circular mean
    import math
    headings = []
    heading_weights = []
    for est in estimates:
        cam_fwd = R.from_quat([est['qvec'][1], est['qvec'][2], est['qvec'][3], est['qvec'][0]]).apply(np.array([0, 0, 1]))
        fwd_world = est['world_pos'] + cam_fwd
        tm = transform_matrices.get(est['map_key'])
        if tm is None:
            continue
        xy_s = tm @ np.append(est['world_pos'], 1.0)
        xy_f = tm @ np.append(fwd_world, 1.0)
        vec = xy_f - xy_s
        h = float(np.degrees(np.arctan2(vec[1], vec[0])) % 360)
        headings.append(h)
        heading_weights.append(est['n_inliers'])

    # Best map key from majority
    from collections import Counter
    key_counts = Counter(tuple(e['map_key']) for e in estimates)
    best_map_key = key_counts.most_common(1)[0][0]

    # Build dummy pnp_pairs (for compatibility with downstream)
    pnp_pairs = {"image_points": np.zeros((0, 2)), "object_points": np.zeros((0, 3))}

    results = [{
        "ref_image_name": e["ref_image_name"],
        "map_key": e["map_key"],
        "score": e["score"],
        "inliers": e["n_inliers"],
    } for e in estimates]

    # Store pose for floorplan transform
    best_est = max(estimates, key=lambda e: e['n_inliers'])
    pnp_pairs["_relpose_qvec"] = best_est["qvec"]
    pnp_pairs["_relpose_tvec"] = best_est["tvec"]
    pnp_pairs["_relpose_avg_pos"] = avg_pos
    pnp_pairs["_relpose_heading"] = headings
    pnp_pairs["_relpose_heading_weights"] = heading_weights

    return best_map_key, pnp_pairs, results
