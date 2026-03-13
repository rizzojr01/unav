# Getting Started

## Hardware Prerequisites

- 360 camera (5K recommended; e.g., GoPro MAX / Insta360 class)
- Linux workstation with NVIDIA GPU (recommended for feature extraction/matching)
- SSD storage with enough free space (mapping can generate many GB per floor)

## Software Prerequisites

- Python 3.8+
- Docker (for `stella_vslam_dense`)
- COLMAP on `PATH` (for triangulation)
- `labelme` (for map annotation)

Install Python deps:

```bash
pip install -r requirements.txt
```

## Build SLAM Docker Image

```bash
git clone https://github.com/RoblabWh/stella_vslam_dense.git
cd stella_vslam_dense/docker
docker build -t stella_vslam_dense .
```

## Critical Runtime Files

Before running mapping, make sure these exist:

- `<DATA_TEMP_ROOT>/orb_vocab.fbow`
- `<DATA_TEMP_ROOT>/<place>/<building>/<floor>/<floor>.mp4`
- `<DATA_FINAL_ROOT>/<place>/<building>/<floor>/floorplan.png` (needed before aligner step)
- `<DATA_FINAL_ROOT>/scale.json` (needed before aligner step)

## Install and Launch Documentation Site Locally

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

Then open `http://127.0.0.1:8000`.
