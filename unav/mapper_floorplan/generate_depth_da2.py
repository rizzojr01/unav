#!/usr/bin/env python3
"""
Generate depth maps using DA² (Depth Anything in Any Direction).

Usage:
    python generate_depth_da2.py <keyframes_dir> <output_dir>

Example:
    python generate_depth_da2.py ./keyframes/ ./keyframes_depth/

Note: DA² requires Python 3.12+ environment (conda activate da2_py312)
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm


def generate_depth_da2(keyframes_dir, output_dir, device=None, scale=5.0):
    """
    Generate depth maps using DA².

    Args:
        keyframes_dir: Directory containing keyframe images
        output_dir: Output directory for depth maps
        device: Computing device (cuda/cpu)
        scale: Scale factor for depth values (DA² outputs normalized depth)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all image*.png files
    image_files = sorted(Path(keyframes_dir).glob("image*.png"))
    if len(image_files) == 0:
        print(f"Warning: No image*.png files found in {keyframes_dir}")
        return

    print(f"Found {len(image_files)} images")

    # Load DA² model
    print("\nLoading DA²...")
    try:
        import torch
        from sam3.model_builder import build_sam3_image_model  # Check if we can import

        # DA² specific imports
        da2_path = Path("/home/unav/Desktop/unav/unav/tmp/pano_depth_methods/DA-2")
        sys.path.insert(0, str(da2_path))

        from spherevit.model import SphereViT
        from spherevit.data_loader import load_image_for_inference

    except ImportError as e:
        print(f"Error: Cannot import DA² components.")
        print(f"  {e}")
        print("\nDA² requires a specific environment. Please run:")
        print("  conda activate da2_py312")
        print("  python generate_depth_da2.py ...")
        sys.exit(1)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Load model
    try:
        model = SphereViT.from_pretrained("EnVision-Research/DA-2").to(device)
        model.eval()
        print("DA² loaded!")
    except Exception as e:
        print(f"Error loading DA² model: {e}")
        sys.exit(1)

    # Process each image
    print(f"\nGenerating depth maps (scale={scale})...")
    success_count = 0

    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Load and preprocess image
            img = Image.open(img_path).convert("RGB")

            # DA² inference
            with torch.no_grad():
                # Prepare input
                img_tensor = load_image_for_inference(img).to(device)

                # Run inference
                depth = model(img_tensor)

                # Post-process
                depth = depth.squeeze().cpu().numpy()

                # Apply scale factor
                depth = depth * scale

            # Save depth as .npy
            out_path = Path(output_dir) / f"{img_path.stem}_depth.npy"
            np.save(out_path, depth.astype(np.float32))

            # Also save visualization
            vis_path = Path(output_dir) / f"{img_path.stem}_depth_vis.png"
            depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255
            Image.fromarray(depth_vis.astype(np.uint8)).save(vis_path)

            success_count += 1

        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"Done! Depth maps saved to: {output_dir}")
    print(f"Total images: {len(image_files)}")
    print(f"Successful: {success_count}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Generate depth maps using DA²')
    parser.add_argument('keyframes_dir', type=str, help='Directory containing keyframe images')
    parser.add_argument('output_dir', type=str, help='Output directory for depth maps')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'],
                        help='Device to use (default: auto-detect)')
    parser.add_argument('--scale', type=float, default=5.0,
                        help='Scale factor for depth values (default: 5.0)')

    args = parser.parse_args()

    # Check input directory
    if not os.path.isdir(args.keyframes_dir):
        print(f"Error: Keyframes directory not found: {args.keyframes_dir}")
        sys.exit(1)

    # Generate depth maps
    generate_depth_da2(
        keyframes_dir=args.keyframes_dir,
        output_dir=args.output_dir,
        device=args.device,
        scale=args.scale
    )


if __name__ == '__main__':
    main()
