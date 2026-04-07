#!/usr/bin/env python3
"""
Generate floor masks using SAM3 (Segment Anything Model 3).

Usage:
    python generate_floor_masks_sam3.py <keyframes_dir> <output_mask_dir>

Example:
    python generate_floor_masks_sam3.py ./keyframes/ ./keyframes_mask/
"""

import sys
sys.path.insert(0, "/home/unav/Desktop/unav")

import os
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from utils.mask import save_mask


def generate_floor_masks_sam3(keyframes_dir, output_dir, prompt="floor", device=None):
    """
    Generate floor masks using SAM3.

    Args:
        keyframes_dir: Directory containing keyframe images
        output_dir: Output directory for masks
        prompt: SAM3 text prompt
        device: Computing device (cuda/cpu)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all image*.png files
    image_files = sorted(Path(keyframes_dir).glob("image*.png"))
    if len(image_files) == 0:
        print(f"Warning: No image*.png files found in {keyframes_dir}")
        return

    print(f"Found {len(image_files)} images")

    # Load SAM3
    print("\nLoading SAM3...")
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as e:
        print(f"Error: Cannot import SAM3. Make sure it's installed.")
        print(f"  {e}")
        sys.exit(1)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    sam3_model = build_sam3_image_model(device=device, eval_mode=True)
    sam3_processor = Sam3Processor(sam3_model)
    print("SAM3 loaded!")

    # Process each image
    print(f"\nExtracting floor masks with prompt: '{prompt}'...")
    success_count = 0
    empty_count = 0

    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Read image
            img = Image.open(img_path).convert("RGB")

            # SAM3 inference
            with torch.no_grad():
                inference_state = sam3_processor.set_image(img)
                output = sam3_processor.set_text_prompt(state=inference_state, prompt=prompt)
                pred_masks = output.get("masks")

                if pred_masks is not None and len(pred_masks) > 0:
                    # Merge all floor masks
                    mask = pred_masks[0].cpu().numpy()
                    for m in pred_masks[1:]:
                        mask = np.logical_or(mask, m.cpu().numpy())
                    success_count += 1
                else:
                    # No floor detected
                    mask = np.zeros((img.height, img.width), dtype=bool)
                    empty_count += 1

            # Save mask (white=detected region, black=background)
            out_path = Path(output_dir) / f"{img_path.stem}_{prompt}_mask.png"
            save_mask(mask, out_path)

        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"Done! Masks saved to: {output_dir}")
    print(f"Total images: {len(image_files)}")
    print(f"Successful detections: {success_count}")
    print(f"Empty masks: {empty_count}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Generate floor masks using SAM3')
    parser.add_argument('keyframes_dir', type=str, help='Directory containing keyframe images (image*.png)')
    parser.add_argument('output_dir', type=str, help='Output directory for floor masks')
    parser.add_argument('--prompt', type=str, default='floor', help='Text prompt for SAM3 (default: floor)')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'],
                        help='Device to use (default: auto-detect)')

    args = parser.parse_args()

    # Check input directory
    if not os.path.isdir(args.keyframes_dir):
        print(f"Error: Keyframes directory not found: {args.keyframes_dir}")
        sys.exit(1)

    # Generate masks
    generate_floor_masks_sam3(
        keyframes_dir=args.keyframes_dir,
        output_dir=args.output_dir,
        prompt=args.prompt,
        device=args.device
    )


if __name__ == '__main__':
    main()
