import os
import h5py
import cv2
import torch
from tqdm import tqdm
from torch.nn.functional import normalize
from unav.core.feature.Global_Extractors import GlobalExtractors
from unav.core.feature.local_extractor import Local_extractor
from unav.config import UNavMappingConfig
from typing import Dict, Any

def extract_features_from_dir(config: UNavMappingConfig) -> None:
    """
    Extract local and global features from all images in the specified directory.

    The config object must include a 'feature_extraction_config' dictionary containing:
        - input_perspective_dir: Directory containing images.
        - local_feat_save_path: Path to HDF5 file for saving local features.
        - global_feat_save_path: Path to HDF5 file for saving global descriptors.
        - output_feature_dir: Directory for storing any extracted feature outputs.
        - local_feature_model, local_extractor_config: Local feature extractor settings.
        - global_descriptor_model, global_descriptor_config: Global descriptor model settings.
        - parameters_root: Root directory for model weights/config.

    Args:
        config (UNavMappingConfig): Configuration object.
    """
    feat_cfg = config.feature_extraction_config
    mapping_mode = getattr(config, "mapping_mode", "whole")  # default: perspective

    def extract_from_folder(img_dir, local_feat_path, global_feat_path):
        os.makedirs(feat_cfg["output_feature_dir"], exist_ok=True)
        # List all images (png or jpg)
        img_list = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith('.png') or f.lower().endswith('.jpg')
        ])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Loading models: Local -> {feat_cfg['local_feature_model']} | Global -> {feat_cfg['global_descriptor_model']}")
        local_extractor = Local_extractor(feat_cfg["local_extractor_config"]).extractor()
        global_extractor = GlobalExtractors(
            feat_cfg['parameters_root'],
            {feat_cfg["global_descriptor_model"]: feat_cfg["global_descriptor_config"]},
            data_parallel=False
        )
        global_extractor.set_train(False)

        with h5py.File(local_feat_path, "w") as local_h5, \
             h5py.File(global_feat_path, "w") as global_h5:

            for img_name in tqdm(img_list, desc=f"Extracting Features in {os.path.basename(img_dir)}"):
                img_path = os.path.join(img_dir, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[Warning] Cannot read {img_path}, skip.")
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Extract and save global descriptor if not already present
                if img_name not in global_h5:
                    tensor_img = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                    feat = global_extractor(feat_cfg["global_descriptor_model"], tensor_img)
                    if isinstance(feat, tuple):
                        feat = feat[1]
                    feat = normalize(feat, dim=-1).squeeze(0).detach().cpu().numpy()
                    global_h5.create_dataset(img_name, data=feat, compression="gzip")

                # Extract and save local features if not already present
                if img_name not in local_h5:
                    feat_dict: Dict[str, Any] = local_extractor(img_rgb)
                    feat_grp = local_h5.create_group(img_name)
                    for k, v in feat_dict.items():
                        feat_grp.create_dataset(k, data=v, compression="gzip")
        print(f"✅ Features saved: Local -> {local_feat_path} | Global -> {global_feat_path}")

    # ------------------------ Main Logic ------------------------
    if mapping_mode == "segment":
        # Each subfolder in input_perspective_dir is a segment to process
        parent_dir = feat_cfg["input_perspective_dir"]
        for segment in sorted(os.listdir(parent_dir)):
            segment_path = os.path.join(parent_dir, segment)
            if not os.path.isdir(segment_path):
                continue
            # Output features per segment
            local_feat_path = os.path.join(
                feat_cfg["output_feature_dir"], f"local_features_{segment}.h5"
            )
            global_feat_path = os.path.join(
                feat_cfg["output_feature_dir"], f"global_features_{feat_cfg['global_descriptor_model']}_{segment}.h5"
            )
            extract_from_folder(segment_path, local_feat_path, global_feat_path)
    else:
        # Single folder (old pipeline)
        extract_from_folder(
            feat_cfg["input_perspective_dir"],
            feat_cfg["local_feat_save_path"],
            feat_cfg["global_feat_save_path"]
        )