import pyimplicitdist
import poselib
import numpy as np
from typing import Dict, Any, Tuple
from scipy.spatial.transform import Rotation as R

def refine_pose_from_queue(
    current_pairs: Dict[str, np.ndarray],
    current_img_shape: Tuple[int, int, int],
    refinement_queue: Dict[str, list],
    max_history: int = 5,
    pp: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Multi-frame pose refinement using a sliding window of recent frames (with implicit pose optimization).

    Args:
        current_pairs (dict): 2D-3D correspondences of the current frame.
            {
                "image_points": np.ndarray [N, 2],
                "object_points": np.ndarray [N, 3]
            }
        current_img_shape (tuple): Shape of input image (H, W, C).
        refinement_queue (dict): Historical info for pose refinement, with keys:
            - "pairs": list of 2D-3D pairs (dict)
            - "initial_poses": list of CameraPose objects (from pyimplicitdist)
            - "pps": list of np.ndarray (principal points)
        max_history (int): Sliding window size for multi-frame optimization.

    Returns:
        dict: {
            "success": bool,
            "qvec": np.ndarray (4,),    # Refined quaternion
            "tvec": np.ndarray (3,),    # Refined translation
            "n_frames": int,            # Number of frames in optimization
            "new_refinement_queue": dict (updated history)
        }
    """
    H, W = current_img_shape[:2]
    if pp is None:
        pp = np.array([W / 2, H / 2])

    # --- Require enough matches
    if len(current_pairs["image_points"]) < 6:
        return {"success": False, "reason": "Not enough 2D-3D correspondences for pose estimation."}

    # --- 1. Coarse pose: 1D radial PnP + nonlinear refinement
    try:
        pts2d_centered = current_pairs["image_points"] - pp
        poselib_pose, info = poselib.estimate_1D_radial_absolute_pose(
            pts2d_centered, current_pairs["object_points"], {"max_reproj_error": 6.0}
        )
        p2d_inlier = current_pairs["image_points"][info["inliers"]]
        p3d_inlier = current_pairs["object_points"][info["inliers"]]

        initial_pose = pyimplicitdist.CameraPose()
        initial_pose.q_vec = poselib_pose.q
        initial_pose.t = poselib_pose.t

        refine_opt = pyimplicitdist.PoseRefinement1DRadialOptions()
        refined = pyimplicitdist.pose_refinement_1D_radial(
            p2d_inlier, p3d_inlier, initial_pose, pp, refine_opt
        )
        refined_pose = refined["pose"]
        refined_pp = refined["pp"]
    except Exception as e:
        return {"success": False, "reason": f"Coarse pose estimation failed: {e}"}

    # --- 2. Update multi-frame queue (sliding window)
    pairs_new = refinement_queue.get("pairs", []).copy()
    initial_poses_new = refinement_queue.get("initial_poses", []).copy()
    pps_new = refinement_queue.get("pps", []).copy()
    pairs_new.append({"image_points": p2d_inlier, "object_points": p3d_inlier})
    initial_poses_new.append(refined_pose)
    pps_new.append(refined_pp)
    if len(pairs_new) > max_history:
        pairs_new = pairs_new[-max_history:]
        initial_poses_new = initial_poses_new[-max_history:]
        pps_new = pps_new[-max_history:]

    # --- 3. Multi-frame pose refinement (implicit joint optimization)
    list_2d = [item["image_points"] for item in pairs_new]
    list_3d = [item["object_points"] for item in pairs_new]
    mean_pp = np.mean(pps_new, axis=0)
    cm_opt = pyimplicitdist.CostMatrixOptions()
    refinement_opt = pyimplicitdist.PoseRefinementOptions()
    cost_matrix = pyimplicitdist.build_cost_matrix_multi(list_2d, cm_opt, mean_pp)
    refined_poses = pyimplicitdist.pose_refinement_multi(
        list_2d, list_3d, cost_matrix, mean_pp, initial_poses_new, refinement_opt
    )

    # Output pose of current frame is the last in the list
    pose_obj = refined_poses[-1]
    qvec, tvec = pose_obj.q_vec, pose_obj.t

    return {
        "success": True,
        "qvec": qvec,
        "tvec": tvec,
        "n_frames": len(list_2d),
        "new_refinement_queue": {
            "pairs": pairs_new,
            "initial_poses": initial_poses_new,
            "pps": pps_new
        }
    }

def colmap2world(tvec: np.ndarray, qvec: np.ndarray) -> Tuple[np.ndarray, R]:
    """
    Convert COLMAP tvec/qvec to camera center in world coordinates and heading rotation.

    Args:
        tvec (np.ndarray): COLMAP tvec (3,)
        qvec (np.ndarray): COLMAP quaternion (4,) [w, x, y, z]

    Returns:
        cam_center (np.ndarray): Camera position in world (3,)
        heading_rot (scipy Rotation): Rotation object (world orientation)
    """
    quat_xyzw = [qvec[1], qvec[2], qvec[3], qvec[0]]  # Convert to [x, y, z, w]
    r = R.from_quat(quat_xyzw)
    rmat = r.as_matrix()
    cam_center = -rmat.T @ tvec
    r_world = R.from_matrix(rmat.T)
    return cam_center, r_world

def transform_pose_to_floorplan(
    qvec: np.ndarray,
    tvec: np.ndarray,
    transform_matrix: np.ndarray
) -> dict:
    """
    Project COLMAP camera pose to floorplan coordinates and heading.

    Args:
        qvec (np.ndarray): COLMAP quaternion [w, x, y, z]
        tvec (np.ndarray): COLMAP translation
        transform_matrix (np.ndarray): Floorplan 2D affine transform [2, 4]

    Returns:
        dict: {
            "xy": np.ndarray (2,),   # (x, y) in floorplan
            "ang": float             # heading in degrees [0, 360)
        }
    """
    if qvec is None or tvec is None or transform_matrix is None:
        return {"xy": None, "ang": None}

    cam_center, r_world = colmap2world(tvec, qvec)
    xyz1 = np.append(cam_center, 1.0)       # Homogeneous [4,]
    xy_fp = transform_matrix @ xyz1          # Floorplan 2D [2,]

    cam_forward = r_world.apply(np.array([0, 0, 1]))  # +Z in world
    forward_xyz = cam_center + cam_forward
    forward_xyz1 = np.append(forward_xyz, 1.0)
    xy_fp_fwd = transform_matrix @ forward_xyz1

    vec = xy_fp_fwd - xy_fp
    ang = np.degrees(np.arctan2(vec[1], vec[0])) % 360

    return {
        "xy": xy_fp,
        "ang": ang
    }
