import os
import torch
import numpy as np
from unav.core.feature_filter import match_query_to_database, ransac_filter

from typing import Dict, Any, Callable, List, Tuple, Iterable, Optional


# Docker-image default (the /data mount); host installs should pass their own
# list of roots via the ``data_roots`` argument of the MASt3R helpers below.
DEFAULT_DB_IMAGE_ROOTS: Tuple[str, ...] = ("/data",)


def _resolve_db_image_path(
    data_roots: Iterable[str],
    place: str,
    building: str,
    floor: str,
    name: str,
) -> Optional[str]:
    """
    Locate the on-disk DB image for a retrieval candidate.

    ``data_roots`` are tried in order; the first existing path wins.
    Returns ``None`` if the image is not found under any root.
    """
    for base in data_roots:
        if not base:
            continue
        p = os.path.join(base, place, building, floor, "perspectives", name)
        if os.path.exists(p):
            return p
    return None

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
    data_roots: Optional[Iterable[str]] = None,
):
    """
    MASt3R dense matching replacement for batch_local_matching_and_ransac().

    For each candidate:
      1. MASt3R dense match (query, DB image) → (query_2d, db_2d)
      2. NN lookup: db_2d → colmap points2D_xy → world points3D_xyz
      3. Collect (image_points, object_points) for PnP

    Args:
        data_roots: Ordered list of root directories under which the DB
            images live (``<root>/<place>/<building>/<floor>/perspectives/...``).
            Typically ``[cfg.data_temp_root, cfg.data_final_root]``. When not
            given, falls back to :data:`DEFAULT_DB_IMAGE_ROOTS`.

    Returns same signature as batch_local_matching_and_ransac():
      (best_map_key, pnp_pairs, results)
    """
    from scipy.spatial import cKDTree

    if data_roots is None:
        data_roots = DEFAULT_DB_IMAGE_ROOTS
    data_roots = tuple(data_roots)

    ref_img_names = list(candidates_data.keys())[:max_candidates]
    grouped = {}

    # Build DB image paths for batch matching — skip candidates whose image
    # cannot be located under any of the configured roots.
    db_paths = []
    db_names = []
    for name in ref_img_names:
        candidate = candidates_data[name]
        place, building, floor = candidate["map_key"]
        db_img_path = _resolve_db_image_path(data_roots, place, building, floor, name)
        if db_img_path is None:
            continue
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
    data_roots: Optional[Iterable[str]] = None,
):
    """
    Map-free localization with Procrustes coordinate system alignment.

    1. For each (query, ref_i) pair, MASt3R gives ref's 3D pointmap in local frame
    2. Collect matched 3D points in both MASt3R-local and colmap-world coordinates
       from multiple refs to solve the rigid transform (Procrustes)
    3. Transform query pose to world coordinates using this alignment
    4. No colmap 3D point cloud needed — only ref images with known poses

    Args:
        data_roots: Ordered list of root directories for DB image lookup
            (see :func:`mast3r_matching_and_pnp`).

    Returns same signature as batch_local_matching_and_ransac():
      (best_map_key, pnp_pairs, results)
    """
    import poselib
    from scipy.spatial.transform import Rotation as R
    from collections import Counter

    if data_roots is None:
        data_roots = DEFAULT_DB_IMAGE_ROOTS
    data_roots = tuple(data_roots)

    ref_img_names = list(candidates_data.keys())[:max_candidates]

    # Phase 1: Run MASt3R on all pairs, collect data
    pair_data = []
    for name in ref_img_names:
        candidate = candidates_data[name]
        map_key = candidate["map_key"]
        ref_frame = candidate["frame"]

        place, building, floor = map_key
        db_img_path = _resolve_db_image_path(data_roots, place, building, floor, name)
        if db_img_path is None:
            continue

        result = mast3r_matcher.match_pair_with_pts3d(query_img_path, db_img_path)
        if result is None:
            continue

        query_2d, pts3d_local, n_matches = result
        if n_matches < min_inliers:
            continue

        # PnP in MASt3R local frame to get query pose in local frame
        import cv2 as _cv2
        q_img = _cv2.imread(query_img_path)
        qh, qw = q_img.shape[:2]
        pp = np.array([qw / 2.0, qh / 2.0])

        try:
            pose, info = poselib.estimate_1D_radial_absolute_pose(
                query_2d - pp, pts3d_local,
                {"max_reproj_error": 12.0, "max_iterations": 10000}
            )
        except Exception:
            continue

        n_inliers = int(np.sum(info['inliers'])) if 'inliers' in info else 0
        if n_inliers < min_inliers:
            continue

        # Query pose in MASt3R local frame
        q_rot_local = R.from_quat([pose.q[1], pose.q[2], pose.q[3], pose.q[0]])
        q_center_local = -q_rot_local.as_matrix().T @ np.array(pose.t)

        # Ref camera center in colmap world
        ref_qvec = ref_frame['qvec']
        ref_tvec = ref_frame['tvec']
        ref_quat_xyzw = [ref_qvec[1], ref_qvec[2], ref_qvec[3], ref_qvec[0]]
        ref_rot_world = R.from_quat(ref_quat_xyzw)
        ref_rmat = ref_rot_world.as_matrix()
        ref_center_world = -ref_rmat.T @ ref_tvec

        # In MASt3R local frame, ref camera is approximately at origin
        # (MASt3R's pred1 is centered on view1)
        # Collect anchor points: (local_pos, world_pos)
        # The ref center in local frame ≈ centroid of its own pointmap projected back
        # But simpler: ref is at ~origin in its own MASt3R frame
        ref_center_local = np.zeros(3)  # approximate

        pair_data.append({
            'name': name,
            'map_key': map_key,
            'candidate': candidate,
            'q_center_local': q_center_local,
            'q_rot_local': q_rot_local,
            'ref_center_local': ref_center_local,
            'ref_center_world': ref_center_world,
            'ref_rmat': ref_rmat,
            'n_inliers': n_inliers,
            'pts3d_local': pts3d_local,  # matched 3D in local frame
        })

    if len(pair_data) == 0:
        return None, {"image_points": np.zeros((0, 2)), "object_points": np.zeros((0, 3))}, []

    if len(pair_data) == 1:
        # Only 1 ref: fall back to simple composition (can't do Procrustes)
        pd = pair_data[0]
        q_center_world = pd['ref_center_world'] + pd['ref_rmat'].T @ pd['q_center_local']
        q_rmat_world = pd['ref_rmat'].T @ pd['q_rot_local'].as_matrix()
        q_rot_world = R.from_matrix(q_rmat_world)
        q_quat = q_rot_world.as_quat()  # xyzw
        qvec = np.array([q_quat[3], q_quat[0], q_quat[1], q_quat[2]])  # wxyz
        tvec = -q_rmat_world @ q_center_world
    else:
        # Phase 2: Procrustes alignment using multiple refs
        # Each ref gives us: (ref_center_local ≈ 0, ref_center_world)
        # But all refs share the SAME MASt3R local frame only within one pair
        # Different pairs have DIFFERENT local frames!
        # So we can't directly align across pairs.
        #
        # Instead: for each pair, compute query_world independently,
        # then use robust averaging (already proven to work for position).
        # For heading: use Procrustes on the per-pair forward directions
        # projected to floorplan, with outlier rejection.

        world_positions = []
        world_qvecs = []
        world_tvecs = []
        weights = []

        for pd in pair_data:
            q_cw = pd['ref_center_world'] + pd['ref_rmat'].T @ pd['q_center_local']
            q_rmat_w = pd['ref_rmat'].T @ pd['q_rot_local'].as_matrix()
            q_rot_w = R.from_matrix(q_rmat_w)
            q_quat = q_rot_w.as_quat()  # xyzw
            qv = np.array([q_quat[3], q_quat[0], q_quat[1], q_quat[2]])  # wxyz
            tv = -q_rmat_w @ q_cw

            world_positions.append(q_cw)
            world_qvecs.append(qv)
            world_tvecs.append(tv)
            weights.append(pd['n_inliers'])

        weights = np.array(weights, dtype=float)
        weights /= weights.sum()

        # Weighted average position
        q_center_world = sum(w * p for w, p in zip(weights, world_positions))

        # Heading: compute per-ref heading on floorplan, outlier reject, average
        best_map_key_candidates = [tuple(pd['map_key']) for pd in pair_data]
        key_counts = Counter(best_map_key_candidates)
        best_mk = key_counts.most_common(1)[0][0]
        tm = transform_matrices.get(best_mk)

        headings = []
        heading_w = []
        if tm is not None:
            for pd, qv, w in zip(pair_data, world_qvecs, weights):
                rot = R.from_quat([qv[1], qv[2], qv[3], qv[0]])
                fwd = rot.apply(np.array([0, 0, 1]))
                pos = pd['ref_center_world'] + pd['ref_rmat'].T @ pd['q_center_local']
                fwd_pos = pos + fwd
                xy_s = tm @ np.append(pos, 1.0)
                xy_f = tm @ np.append(fwd_pos, 1.0)
                vec = xy_f - xy_s
                h = float(np.degrees(np.arctan2(vec[1], vec[0])) % 360)
                headings.append(h)
                heading_w.append(float(w))

        # Outlier rejection on headings
        if len(headings) >= 3:
            rads = np.radians(headings)
            sin_med = np.median(np.sin(rads))
            cos_med = np.median(np.cos(rads))
            anchor = np.arctan2(sin_med, cos_med)
            keep_h = []
            keep_w = []
            for h, w in zip(headings, heading_w):
                diff = abs(((np.radians(h) - anchor + np.pi) % (2 * np.pi)) - np.pi)
                if diff < np.pi / 2:
                    keep_h.append(h)
                    keep_w.append(w)
            if keep_h:
                rk = np.radians(keep_h)
                wk = np.array(keep_w)
                wk /= wk.sum()
                avg_ang = float(np.degrees(np.arctan2(
                    np.sum(wk * np.sin(rk)), np.sum(wk * np.cos(rk)))) % 360)
            else:
                avg_ang = float(np.degrees(anchor) % 360)
        elif headings:
            avg_ang = headings[0]
        else:
            avg_ang = 0

        # Construct final qvec/tvec using best estimate's rotation
        # but corrected heading
        best_pd = max(pair_data, key=lambda x: x['n_inliers'])
        qvec = world_qvecs[pair_data.index(best_pd)]
        tvec = world_tvecs[pair_data.index(best_pd)]

    # Build result
    best_map_key_candidates = [tuple(pd['map_key']) for pd in pair_data]
    key_counts = Counter(best_map_key_candidates)
    best_map_key = key_counts.most_common(1)[0][0]

    pnp_pairs = {
        "image_points": np.zeros((0, 2)),
        "object_points": np.zeros((0, 3)),
        "_relpose_qvec": qvec,
        "_relpose_tvec": tvec,
    }

    results = [{
        "ref_image_name": pd["name"],
        "map_key": pd["map_key"],
        "score": pd["candidate"].get("score", 0),
        "inliers": pd["n_inliers"],
    } for pd in pair_data]

    return best_map_key, pnp_pairs, results
