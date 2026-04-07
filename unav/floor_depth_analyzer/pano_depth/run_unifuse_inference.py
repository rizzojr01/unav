"""
Run UniFuse inference only (no HoHoNet import issues).
"""
import os
import sys
import cv2
import numpy as np
import torch
from torchvision import transforms

# Fix numpy compatibility
np.bool = np.bool_
np.float = np.float32

from baseline_models.UniFuse.networks import UniFuse
from utils.Projection import py360_E2C

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def run_inference(rgb_path: str, ckpt_path: str, output_path: str):
    """Run UniFuse inference and save raw depth as .npy."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("Loading UniFuse model...")
    model = UniFuse(
        num_layers=18,
        equi_h=512,
        equi_w=1024,
        pretrained=False,
        max_depth=10.0,
        fusion_type='cee',
        se_in_fusion=True
    )

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    model.eval()

    # Load and preprocess image
    print(f"Loading image: {rgb_path}")
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (1024, 512))

    # Create cube projection
    e2c = py360_E2C(equ_h=512, equ_w=1024, face_w=256)
    cube_rgb = e2c.run(rgb)

    # Normalize
    totensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=MEAN, std=STD)

    rgb_tensor = normalize(totensor(rgb)).unsqueeze(0).to(device)
    cube_tensor = normalize(totensor(cube_rgb)).unsqueeze(0).to(device)

    # Inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(rgb_tensor, cube_tensor)
        pred_depth = outputs['pred_depth'].squeeze().cpu().numpy()

    print(f"Predicted depth range: {pred_depth.min():.2f} - {pred_depth.max():.2f} m")

    # Save raw depth
    np.save(output_path, pred_depth)
    print(f"Saved to {output_path}")

    return pred_depth


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="ckpts/UniFuse/unifuse_st3d_sf3d/ckpt_100.pth")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    run_inference(args.rgb, args.ckpt, args.output)
