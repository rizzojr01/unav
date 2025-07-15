import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict

from unav.config import UNavMappingConfig
from unav.core.feature.local_extractor import Local_extractor
from unav.core.colmap.utils_pose import load_colmap_images_file
from unav.core.feature_filter import match_query_to_database, ransac_filter
from unav.mapper.tools.matcher.geometry import fast_find_adjacent_and_pose_pairs

def compute_similarity_and_generate_topk_pairs(
    descriptor_h5_path: str,
    top_k: int = 50,
    batch_size: int = 500
) -> List[Tuple[str, str]]:
    """
    Compute Top-K most similar image pairs based on global descriptor similarity.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with h5py.File(descriptor_h5_path, 'r') as f:
        img_names = list(f.keys())
        descriptors = torch.stack([torch.tensor(f[name][:]) for name in img_names], dim=0).to(device)
    descriptors = torch.nn.functional.normalize(descriptors, dim=1)
    N = descriptors.shape[0]
    sim_matrix = torch.full((N, N), -float('inf'), device='cpu')

    # Compute similarity in batches for efficiency
    for i in range(0, N, batch_size):
        desc_batch = descriptors[i:i+batch_size]
        sim_batch = torch.einsum('bd,nd->bn', desc_batch, descriptors).cpu()
        for j in range(sim_batch.shape[0]):
            sim_batch[j, i + j] = -float('inf')  # Prevent self-matching
        sim_matrix[i:i+batch_size] = sim_batch

    pairs = [
        (img_names[i], img_names[j])
        for i in range(N)
        for j in sim_matrix[i].numpy().argsort()[::-1][:top_k]
    ]
    return pairs

def insert_segment_dir(base_path: str, segment_name: str) -> str:
    """
    Insert a segment_name subdirectory before the file name of base_path.
    For example:
        base_path = "/a/b/c/colmap_sfm/matches.h5"
        segment_name = "20250711_134901"
        => "/a/b/c/colmap_sfm/20250711_134901/matches.h5"
    """
    folder, fname = os.path.split(base_path)
    return os.path.join(folder, segment_name, fname)

def generate_and_stream_colmap(
    config: UNavMappingConfig
) -> List[Tuple[str, str]]:
    """
    Full pipeline to generate COLMAP-compatible matching files from extracted features.

    For segment mode, each segment outputs its own matches.h5 and pairs.txt to
    .../colmap_sfm/<segment_name>/matches.h5 and .../colmap_sfm/<segment_name>/pairs.txt.

    For whole mode, outputs to .../colmap_sfm/matches.h5 and .../colmap_sfm/pairs.txt.

    Returns:
        List[Tuple[str, str]]: List of all verified image pairs.
    """
    slicer_config = config.slicer_config
    feat_cfg = config.feature_extraction_config
    matcher_config = config.matcher_config
    colmap_config = config.colmap_config
    mapping_mode = getattr(config, "mapping_mode", "whole")
    image_file = colmap_config["image_file"]

    all_verified_pairs: List[Tuple[str, str]] = []

    def match_and_write_for_h5(
        local_feat_h5: str,
        global_feat_h5: str,
        pairs_txt_path: str,
        matches_h5_path: str,
        use_direct_pairs: bool = True
    ) -> List[Tuple[str, str]]:
        """
        Perform matching and geometric verification, write COLMAP-compatible files.
        """
        all_matches_dict: Dict[str, Dict[str, np.ndarray]] = {}

        # Step 1: Candidate pairs
        if use_direct_pairs:
            poses = load_colmap_images_file(image_file)
        with h5py.File(local_feat_h5, 'r') as f_local:
            img_names = list(f_local.keys())
        if use_direct_pairs:
            direct_pairs = fast_find_adjacent_and_pose_pairs(
                img_names,
                poses,
                gv_threshold_pos=matcher_config["gv_threshold_pos"],
                gv_threshold_angle_deg=matcher_config["gv_threshold_angle_deg"],
                gv_fov_deg=slicer_config["fov"]
            )
        else:
            direct_pairs = []
        retrieval_pairs = compute_similarity_and_generate_topk_pairs(
            global_feat_h5,
            top_k=matcher_config["top_k_matches"]
        )
        all_pairs = list({(min(a, b), max(a, b)) for a, b in set(direct_pairs) | set(retrieval_pairs)})

        # Step 2: Group pairs by img1 for batch matching
        img1_groups: Dict[str, List[str]] = {}
        for a, b in all_pairs:
            img1_groups.setdefault(a, []).append(b)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        matcher = Local_extractor(
            feat_cfg['local_extractor_config']
        ).matcher().to(device)
        verified_pairs: List[Tuple[str, str]] = []

        # Step 3: Matching and RANSAC verification
        with h5py.File(local_feat_h5, 'r') as f_local:
            for img1, img2_list in tqdm(img1_groups.items(), desc=f"Matching/Verifying [{os.path.basename(local_feat_h5)}]"):
                grp1 = f_local[img1]
                feat1 = {
                    'descriptors': torch.tensor(grp1['descriptors'][:], device=device),
                    'keypoints': torch.tensor(grp1['keypoints'][:], device=device),
                    'scores': torch.tensor(grp1['scores'][:], device=device),
                    'image_size': torch.tensor(grp1['image_size'][:], device=device)
                }
                db_feats: List[dict] = []
                for img2 in img2_list:
                    grp2 = f_local[img2]
                    db_feats.append({
                        'descriptors': torch.tensor(grp2['descriptors'][:], device=device),
                        'keypoints': torch.tensor(grp2['keypoints'][:], device=device),
                        'scores': torch.tensor(grp2['scores'][:], device=device),
                        'image_size': torch.tensor(grp2['image_size'][:], device=device)
                    })
                names2, p0_idx, p1_idx, scores_list = match_query_to_database(
                    feat1, db_feats, img2_list, matcher, device,
                    feature_score_threshold=matcher_config["feature_score_threshold"],
                    threshold=matcher_config["min_keypoints"]
                )
                if not names2:
                    continue
                k0_np = feat1['keypoints'].cpu().numpy()
                k0_list = [k0_np] * len(names2)
                k1_list = [f_local[name]['keypoints'][:] for name in names2]
                valid_mask, in0, in1, inlier_scores = ransac_filter(
                    p0_idx, p1_idx, k0_list, k1_list, scores_list, device,
                    threshold=matcher_config["min_keypoints"]
                )
                for img2, flag, idx0, idx1, scores in zip(names2, valid_mask, in0, in1, inlier_scores):
                    if not flag:
                        continue
                    verified_pairs.append((img1, img2))
                    match_key = f"{img1}_{img2}"
                    all_matches_dict[match_key] = {
                        'matches0': np.stack([idx0, idx1], axis=1),
                        'matching_scores0': scores
                    }

        # Step 4: Write pairs.txt and matches.h5
        os.makedirs(os.path.dirname(pairs_txt_path), exist_ok=True)
        os.makedirs(os.path.dirname(matches_h5_path), exist_ok=True)
        with open(pairs_txt_path, 'w') as f_pairs:
            for a, b in sorted(set((min(a, b), max(a, b)) for a, b in verified_pairs)):
                f_pairs.write(f"{a} {b}\n")
        with h5py.File(matches_h5_path, 'w') as f_matches:
            for key, match_dict in all_matches_dict.items():
                group = f_matches.create_group(key)
                group.create_dataset('matches0', data=match_dict['matches0'])
                group.create_dataset('matching_scores0', data=match_dict['matching_scores0'])

        print(f"[✓] pairs.txt written to: {pairs_txt_path}")
        print(f"[✓] matches.h5 written to: {matches_h5_path}")
        print(f"[✓] Total verified pairs: {len(verified_pairs)}")
        return verified_pairs

    # ======= Main Logic =======
    if mapping_mode == "segment":
        feat_dir = feat_cfg["output_feature_dir"]
        local_h5_list = sorted([
            os.path.join(feat_dir, f)
            for f in os.listdir(feat_dir)
            if f.startswith("local_features_") and f.endswith(".h5")
        ])
        global_h5_list = sorted([
            os.path.join(feat_dir, f)
            for f in os.listdir(feat_dir)
            if f.startswith("global_features_") and f.endswith(".h5")
        ])
        assert len(local_h5_list) == len(global_h5_list), \
            "Mismatch: local and global feature HDF5 count not equal in segment mode."
        for local_h5, global_h5 in zip(local_h5_list, global_h5_list):
            segment_name = os.path.splitext(os.path.basename(local_h5))[0].replace("local_features_", "")
            pairs_txt_path = insert_segment_dir(colmap_config['pairs_txt'], segment_name)
            matches_h5_path = insert_segment_dir(colmap_config['match_file'], segment_name)
            segment_verified_pairs = match_and_write_for_h5(
                local_h5, global_h5, pairs_txt_path, matches_h5_path, use_direct_pairs=False
            )
            all_verified_pairs.extend(segment_verified_pairs)
    else:
        # Standard (whole) mode: use config colmap_config as output
        local_feat_h5 = feat_cfg['local_feat_save_path']
        global_feat_h5 = feat_cfg['global_feat_save_path']
        pairs_txt_path = colmap_config['pairs_txt']
        matches_h5_path = colmap_config['match_file']
        segment_verified_pairs = match_and_write_for_h5(
            local_feat_h5, global_feat_h5, pairs_txt_path, matches_h5_path, use_direct_pairs=True
        )
        all_verified_pairs.extend(segment_verified_pairs)

    return all_verified_pairs
