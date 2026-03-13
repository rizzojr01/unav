# Mapping Workflow

## Step 1: Prepare Paths and IDs

Example values:

```bash
DATA_TEMP_ROOT=/mnt/data/UNav-IO/temp
DATA_FINAL_ROOT=/mnt/data/UNav-IO/data
FEATURE_MODEL=DinoV2Salad
PLACE=New_York_City
BUILDING=LightHouse
FLOOR=4_floor
```

## Step 2: Verify Input File Contract

```bash
ls -lh "$DATA_TEMP_ROOT/orb_vocab.fbow"
ls -lh "$DATA_TEMP_ROOT/$PLACE/$BUILDING/$FLOOR/$FLOOR.mp4"
```

## Step 3: Run Mapping Pipeline

```bash
python -m unav.run_mapper \
  "$DATA_TEMP_ROOT" \
  "$DATA_FINAL_ROOT" \
  "$FEATURE_MODEL" \
  "$PLACE" \
  "$BUILDING" \
  "$FLOOR"
```

Backward-compatible alias:

```bash
python -m unav.run_mapping \
  "$DATA_TEMP_ROOT" \
  "$DATA_FINAL_ROOT" \
  "$FEATURE_MODEL" \
  "$PLACE" \
  "$BUILDING" \
  "$FLOOR"
```

## What Happens Internally

1. `stella_vslam_dense` runs in Docker on `<floor>.mp4`
2. SLAM keyframes are sliced into perspective images
3. Local/global features are extracted
4. Image pairs are matched and geometrically verified
5. COLMAP triangulates 3D map with known camera poses

## Per-Stage Output Checks

### Stage 1 (SLAM)

Check:

```bash
ls "$DATA_TEMP_ROOT/$PLACE/$BUILDING/$FLOOR/stella_vslam_dense"
```

Expected key files:

- `final_map.msg`
- `keyframes/`
- `eval_logs/keyframe_trajectory.txt`

### Stage 2 (Slicing)

Check:

```bash
ls "$DATA_TEMP_ROOT/$PLACE/$BUILDING/$FLOOR/perspectives" | head
```

Expected filenames:

- `000123_pitch00_yaw05.png` style pattern

### Stage 3 (Feature Extraction)

Check:

```bash
ls "$DATA_FINAL_ROOT/$PLACE/$BUILDING/$FLOOR/features"
```

Expected files:

- `local_features.h5`
- `global_features_<feature_model>.h5`

### Stage 4/5 (Matching + COLMAP)

Check:

```bash
ls "$DATA_TEMP_ROOT/$PLACE/$BUILDING/$FLOOR/colmap_sfm"
ls "$DATA_FINAL_ROOT/$PLACE/$BUILDING/$FLOOR/colmap_map"
```

Expected files:

- `pairs.txt`, `matches.h5`, `database.db`
- triangulation outputs under `colmap_map`

## Typical Runtime Failure Points

- Docker image `stella_vslam_dense` not built
- Missing `orb_vocab.fbow`
- Missing `<floor>.mp4` or name mismatch
- `colmap` command not found
- Feature model checkpoint files not available under `data_final_root/parameters/...`
