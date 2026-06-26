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

## Feature Models

UNav uses a global descriptor for retrieval and a local feature extractor for matching.

- **Global (retrieval):** `DinoV2Salad`. The checkpoint must exist at
  `<DATA_FINAL_ROOT>/parameters/DinoV2Salad/ckpts/dino_salad.ckpt`.
- **Local — mapping:** `superpoint+lightglue` (default used by the mapping pipeline).
- **Local — localization server:** `mast3r`. Set `LOCAL_FEATURE_MODEL = "mast3r"` in the
  server config. The weights (`naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`,
  ~2.75 GB) download automatically from HuggingFace on first run — the machine needs
  internet access the first time the server starts. No manual download or checkpoint
  placement is required.

## Build SLAM Docker Image

The mapper launches SLAM via `docker run stella_vslam_dense`, so this image must exist
locally. Build it from the **repo root** using the headless (socket) Dockerfile:

```bash
git clone https://github.com/RoblabWh/stella_vslam_dense.git
cd stella_vslam_dense
docker build -t stella_vslam_dense -f Dockerfile.socket . --build-arg NUM_THREADS=$(nproc)
```

> The image **must** be tagged exactly `stella_vslam_dense` — the mapper calls it by that
> name. If the image is missing, the SLAM step fails with a Docker
> `pull access denied / repository does not exist` error (which can look like a
> permissions problem but is really just a missing local image).

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
