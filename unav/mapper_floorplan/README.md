# Floor Point Cloud Extraction Pipeline

## Overview

Automated floor point cloud extraction pipeline supporting **two methods**:
1. **Depth Method** (Recommended): DA² depth estimation + camera pose reprojection
2. **Point Cloud Method**: Filter SLAM sparse points using SAM3 floor masks

## Methods Comparison

| Method | Input | Output Points | Density | Best For |
|--------|-------|---------------|---------|----------|
| **Depth (DA²)** | Few images | ~86,000 | High | Production use |
| Point Cloud | All images | ~600 | Sparse | Quick preview |

**Recommendation**: Use `depth` method for better floor plan quality.

## Quick Start

```bash
cd /home/unav/Desktop/unav/unav/mapper_floorplan

# Use depth method (default, recommended)
./extract_floor_auto.sh -m depth New_York_City LOH 9_floor

# Use point cloud method
./extract_floor_auto.sh -m pointcloud New_York_City LOH 9_floor

# Show help
./extract_floor_auto.sh -h
```

## Workflow

### Depth Method (Default)
```
Keyframe Images (keyframes/)
    |
[Step 1] SAM3 Generate Floor Masks
    |
[Step 2] DA² Generate Depth Maps (requires da2_py312 env)
    |
[Step 3] Reproject Depth to 3D Floor Points
    |
Output: floor_points_depth.ply + views + grid map
```

### Point Cloud Method
```
Keyframe Images (keyframes/)
    |
[Step 1] SAM3 Generate Floor Masks
    |
[Step 2] Filter SLAM Sparse Points by Masks
    |
Output: floor_points_extracted.ply + views + grid map
```

## Directory Structure

```
mapper_floorplan/
├── config.yaml                    # Global configuration
├── extract_floor_auto.sh          # Automated extraction script
├── extract_floor_points.py        # Point cloud method
├── extract_floor_from_depth.py    # Depth method
├── generate_floor_masks_sam3.py   # SAM3 mask generation
├── generate_depth_da2.py          # DA² depth generation
├── run_da2_inference.sh           # DA² wrapper script
├── utils/                         # Utility modules
│   ├── database.py               # SQLite database loading
│   ├── geometry.py               # Projection & transforms
│   ├── pointcloud.py             # Point cloud I/O
│   ├── visualization.py          # Views & map generation
│   ├── door_detection.py         # Door detection
│   └── mask.py                   # Mask utilities
└── web/                          # Web visualization app
    ├── app.py
    └── templates/index.html

Data Structure:
/mnt/data/UNav-IO/temp/{PLACE}/{BUILDING}/{FLOOR}/stella_vslam_dense/
├── final_map.msg                  # SQLite database
├── keyframes/                     # Keyframe images
├── keyframes_mask/                # SAM3-generated floor masks
├── keyframes_depth/               # DA² depth maps (.npy)
├── floor_points_depth.ply         # Depth method output
├── floor_points_extracted.ply     # Point cloud method output
└── floor_map_depth.png            # Top-down grid map
```

## Configuration

Edit `config.yaml`:

```yaml
# Extraction method: "depth" or "pointcloud"
extraction:
  method: "depth"
  mask_pattern: "image{}_floor_mask.png"
  grid_resolution: 0.05
  depth_subsample: 8

# DA² settings
da2:
  conda_env: "da2_py312"
  da2_path: "/path/to/DA-2"
  depth_scale: 5.0
  depth_pattern: "image{}.npy"

# SAM3 settings
sam3:
  floor_prompt: "floor"
  device: null  # auto-detect
```

## Script Reference

### extract_floor_auto.sh

Main automated script supporting both methods.

```bash
# Depth method (default)
./extract_floor_auto.sh -m depth [PLACE] [BUILDING] [FLOOR]

# Point cloud method
./extract_floor_auto.sh -m pointcloud [PLACE] [BUILDING] [FLOOR]
```

### extract_floor_from_depth.py

Extract floor points from depth maps.

```bash
python3 extract_floor_from_depth.py \
    <sqlite3_db> \
    <depth_dir> \
    <mask_dir> \
    <output_dir> \
    --depth-pattern "image{}.npy" \
    --mask-pattern "image{}_floor_mask.png" \
    --subsample 8 \
    --scale 5.0
```

### extract_floor_points.py

Extract floor points from SLAM point cloud.

```bash
python3 extract_floor_points.py \
    <sqlite3_db> \
    <mask_dir> \
    <output_ply> \
    --mask-pattern "image{}_floor_mask.png" \
    --save-views --save-map
```

### generate_floor_masks_sam3.py

Generate floor masks using SAM3.

```bash
python3 generate_floor_masks_sam3.py \
    <keyframes_dir> \
    <output_dir> \
    --prompt "floor"
```

## Environment Setup

### Main Environment (SAM3, Point Cloud)
```bash
conda activate unav
```

### DA² Environment (Depth Method)
```bash
# DA² requires Python 3.12
conda create -n da2_py312 python=3.12
conda activate da2_py312
pip install torch torchvision accelerate transformers
```

## Output Files

| File | Method | Description |
|------|--------|-------------|
| `floor_points_depth.ply` | Depth | Dense floor point cloud |
| `floor_points_extracted.ply` | Point Cloud | Sparse floor point cloud |
| `floor_map_depth.png` | Depth | Top-down grid map |
| `floor_points_extracted_map.png` | Point Cloud | Top-down grid map |
| `view_top.png` | Both | Top view for web |
| `view_metadata.json` | Both | Camera positions |

## Troubleshooting

### DA² Environment Issues

```bash
# Make sure to use Python 3.12
conda activate da2_py312
python --version  # Should be 3.12.x
```

### SAM3 Import Error

```bash
# Add SAM3 to path
export PYTHONPATH="/path/to/sam3:$PYTHONPATH"
```

### Disk Space

DA² generates large depth maps. Ensure sufficient space:
- Depth maps: ~30MB per image
- Total: ~5GB per floor (162 images)

## License

See the main repository LICENSE file.
