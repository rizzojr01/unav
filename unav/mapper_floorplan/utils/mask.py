"""
Mask loading and processing utilities.
"""

import os
import cv2


def load_masks(mask_dir, keyframe_ids, pattern='mask{}.png', invert=False):
    """
    Load masks for keyframes from a directory.

    Args:
        mask_dir: Directory containing mask files
        keyframe_ids: List or iterable of keyframe IDs
        pattern: Filename pattern with {} placeholder for keyframe ID
        invert: If True, invert the mask (255 - mask)

    Returns:
        dict: Dictionary mapping keyframe ID to mask array
    """
    masks = {}
    for kf_id in keyframe_ids:
        mask_path = os.path.join(mask_dir, pattern.format(kf_id))
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if invert:
                mask = 255 - mask
            masks[kf_id] = mask
    return masks


def load_masks_with_suffix(mask_dir, keyframe_ids, suffix='_floor_mask.png', invert=False):
    """
    Load masks using image name with suffix pattern.

    Args:
        mask_dir: Directory containing mask files
        keyframe_ids: List or iterable of keyframe IDs
        suffix: Suffix to append to image name (e.g., '_floor_mask.png')
        invert: If True, invert the mask (255 - mask)

    Returns:
        dict: Dictionary mapping keyframe ID to mask array
    """
    masks = {}
    for kf_id in keyframe_ids:
        mask_path = os.path.join(mask_dir, f"image{kf_id}{suffix}")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if invert:
                mask = 255 - mask
            masks[kf_id] = mask
    return masks


def save_mask(mask, output_path):
    """
    Save a mask to file.

    Args:
        mask: 2D numpy array (uint8 or bool)
        output_path: Output file path
    """
    import numpy as np

    if mask.dtype == bool:
        mask = (mask.astype(np.uint8) * 255)

    if mask.ndim > 2:
        mask = mask.squeeze()

    cv2.imwrite(str(output_path), mask)
