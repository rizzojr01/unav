# Getting Started

## Hardware Prerequisites

- Insta360-class 360 camera, 5K or higher recommended
- Phone with the camera app installed for preview and export setup
- Selfie stick or monopod that positions the camera slightly above head height
- Linux workstation with NVIDIA GPU, recommended for feature extraction and matching
- SSD storage with enough free space; mapping can generate many GB per floor
- Floorplan image for each floor that will be mapped

GoPro MAX can be used as an alternative if it exports a spherical/equirectangular MP4. The rest of this guide assumes the primary Insta360 workflow.

## Software Prerequisites

- Python 3.10
- Poetry for Python dependency installation and lockfile reproducibility
- Docker for `stella_vslam_dense`
- COLMAP on `PATH` for triangulation
- `labelme` for map annotation

Install the mapping runner dependencies from the dedicated `unav-run` repo:

```bash
git clone https://github.com/endeleze/unav-run.git
cd unav-run
poetry install
poetry run pip install --no-deps git+https://github.com/ai4ce/unav.git
```

Dependency files:

- [`unav-run/pyproject.toml`](https://github.com/endeleze/unav-run/blob/main/pyproject.toml) lists direct mapping runner dependencies.
- [`unav-run/poetry.lock`](https://github.com/endeleze/unav-run/blob/main/poetry.lock) locks nested dependency versions.
- Legacy UNav core requirements live at [`ai4ce/unav/requirements.txt`](https://github.com/ai4ce/unav/blob/main/requirements.txt); use this only when installing the full UNav core repo directly.

## Build SLAM Docker Image

```bash
git clone https://github.com/RoblabWh/stella_vslam_dense.git
cd stella_vslam_dense/docker
docker build -t stella_vslam_dense .
```

## End-to-End Inputs

Before running mapping, make sure these exist:

- `<DATA_TEMP_ROOT>/orb_vocab.fbow`
- `<DATA_TEMP_ROOT>/<place>/<building>/<floor>/<floor>.mp4`
- `<DATA_FINAL_ROOT>/<place>/<building>/<floor>/floorplan.png`
- `<DATA_FINAL_ROOT>/scale.json`

The MP4 should be the exported spherical Insta360 video for that floor. The floorplan and `scale.json` are needed before the aligner step, but preparing them before mapping avoids mismatched folders later.

## Recommended First Run

1. Read [Data Collection SOP](data-collection-sop.md) before going on site.
2. Export one spherical MP4 per floor.
3. Place files according to [Input Output Contract](io-contract.md).
4. Run [Mapping Workflow](mapping-workflow.md).
5. Run [Floorplan Alignment](aligner.md).
6. Create `boundaries.json` and labels using [Map Labeling & Multi-language](map-labeling-multilang.md).

## Install and Launch Documentation Site Locally

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

Then open `http://127.0.0.1:8000`.
