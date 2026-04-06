import torch
import numpy as np
from torch.nn.functional import normalize

def extract_query_features(
    query_img: np.ndarray,
    global_extractor,
    local_extractor,
    global_model_name: str,
    device: torch.device
) -> tuple:
    """
    Extract both global and local features for a query image.

    Args:
        query_img (np.ndarray): RGB image (H, W, 3), dtype=np.uint8 or float32 in [0,255].
        global_extractor: Callable global extractor, e.g., GlobalExtractors instance.
        local_extractor: Callable local extractor, e.g., Local_extractor().extractor().
        global_model_name (str): Name of global feature model (e.g., 'DinoV2Salad').
        device (torch.device): Device to run extraction ('cuda' or 'cpu').

    Returns:
        Tuple:
            - global_feat (np.ndarray): (D, ), normalized.
            - local_feat_dict (Dict[str, np.ndarray]): keys include 'keypoints', 'descriptors', 'scores', 'image_size'.
    """
    # --- Prepare tensor image for global model ---
    if query_img.dtype != np.float32:
        img = query_img.astype(np.float32)
    else:
        img = query_img.copy()
    tensor_img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    # --- Global feature extraction ---
    global_feat = global_extractor(global_model_name, tensor_img)
    if isinstance(global_feat, tuple):
        global_feat = global_feat[1]   # For some models, returns (logits, feat)
    global_feat = normalize(global_feat, dim=-1).squeeze(0).detach().cpu().numpy()

    # --- Local feature extraction ---
    # MASt3R mode: local_extractor is None (joint extraction+matching done later)
    if local_extractor is not None:
        local_feat_dict = local_extractor(query_img)
    else:
        local_feat_dict = None

    return global_feat, local_feat_dict
