#!/usr/bin/env python3
"""
Simplified DA² inference script - only saves depth maps, skips 3D point cloud generation.
This is much faster than the original infer.py which generates 3D point clouds.
"""
import os
import sys
import numpy as np
import torch
from contextlib import nullcontext
from tqdm import tqdm

# Add DA² to path
DA2_PATH = "/home/unav/Desktop/unav/unav/tmp/pano_depth_methods/DA-2"
sys.path.insert(0, DA2_PATH)

from da2 import (
    prepare_to_run,
    load_model,
    load_infer_data,
)


def infer_depth_only(model, config, accelerator, output_dir):
    """Run inference and save only depth maps (.npy files)."""
    model.eval()

    if accelerator.is_main_process:
        # Create depth output directory
        depth_dir = os.path.join(output_dir, 'depth')
        os.makedirs(depth_dir, exist_ok=True)

        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(accelerator.device.type)

        with autocast_ctx, torch.no_grad():
            infer_data = load_infer_data(config, accelerator.device)

            # Predict and save depth for each image
            for i in tqdm(range(infer_data['size']), desc='Predicting and saving depth'):
                # Predict depth
                distances = model(infer_data['images']['torch'][i])
                depth_np = distances.cpu().numpy()

                # Save depth as .npy
                filename = infer_data['filenames'][i]
                depth_path = os.path.join(depth_dir, f'{filename}.npy')
                np.save(depth_path, depth_np)

            print(f"\n✓ Saved {infer_data['size']} depth maps to {depth_dir}")


if __name__ == '__main__':
    os.chdir(DA2_PATH)
    config, accelerator, output_dir = prepare_to_run()
    model = load_model(config, accelerator)
    infer_depth_only(model, config, accelerator, output_dir)
