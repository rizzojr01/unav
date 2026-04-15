# Mapping Workflow

This page starts after the field team has captured and exported the spherical Insta360 MP4. It explains how to turn the raw map video into UNav map artifacts.

## Step 1: Prepare Paths and IDs

Choose stable identifiers before processing. These identifiers become folder names and deployment IDs.

Example values:

```bash
DATA_TEMP_ROOT=/mnt/data/UNav-IO/temp
DATA_FINAL_ROOT=/mnt/data/UNav-IO/data
FEATURE_MODEL=DinoV2Salad
PLACE=New_York_City
BUILDING=LightHouse
FLOOR=4_floor
```

Meaning:

- `DATA_TEMP_ROOT`: raw video and intermediate artifacts.
- `DATA_FINAL_ROOT`: final navigation map package.
- `FEATURE_MODEL`: global descriptor model used for retrieval.
- `PLACE`, `BUILDING`, `FLOOR`: namespace for this mapped location.

## Step 2: Place Raw Inputs

Create the expected folders:

```bash
mkdir -p $DATA_TEMP_ROOT/$PLACE/$BUILDING/$FLOOR
mkdir -p $DATA_FINAL_ROOT/$PLACE/$BUILDING/$FLOOR
```

Copy the exported 360 video to:

```text
$DATA_TEMP_ROOT/$PLACE/$BUILDING/$FLOOR/$FLOOR.mp4
```

Copy the floorplan to:

```text
$DATA_FINAL_ROOT/$PLACE/$BUILDING/$FLOOR/floorplan.png
```

Make sure `scale.json` exists at:

```text
$DATA_FINAL_ROOT/scale.json
```

## Step 3: Verify Input File Contract

```bash
ls -lh $DATA_TEMP_ROOT/orb_vocab.fbow
ls -lh $DATA_TEMP_ROOT/$PLACE/$BUILDING/$FLOOR/$FLOOR.mp4
ls -lh $DATA_FINAL_ROOT/$PLACE/$BUILDING/$FLOOR/floorplan.png
ls -lh $DATA_FINAL_ROOT/scale.json
```

The video must be an exported spherical/equirectangular MP4 from the Insta360 workflow. The filename must exactly match the `FLOOR` value.

## Step 4: Run Mapping Pipeline

```bash
python -m unav.run_mapper   $DATA_TEMP_ROOT   $DATA_FINAL_ROOT   $FEATURE_MODEL   $PLACE   $BUILDING   $FLOOR
```

Backward-compatible alias:

```bash
python -m unav.run_mapping   $DATA_TEMP_ROOT   $DATA_FINAL_ROOT   $FEATURE_MODEL   $PLACE   $BUILDING   $FLOOR
```

The older internal SOP referred to manual `step1` and `step2` scripts. The current external workflow wraps those stages behind `python -m unav.run_mapper` so the user does not need to know individual script line numbers.

## What Happens Internally

1. SLAM runs on `<floor>.mp4` and estimates keyframes/trajectory.
2. Keyframes are sliced into perspective images.
3. Local and global visual features are extracted.
4. Image pairs are matched and geometrically verified.
5. COLMAP triangulates a 3D map with known camera poses.

## Per-Stage Output Checks

### Stage 1: SLAM

Check:

```bash
ls $DATA_TEMP_ROOT/$PLACE/$BUILDING/$FLOOR/stella_vslam_dense
```

Expected key files:

- `final_map.msg`
- `keyframes/`
- `eval_logs/keyframe_trajectory.txt`

### Stage 2: Perspective Images

Check:

```bash
ls $DATA_TEMP_ROOT/$PLACE/$BUILDING/$FLOOR/perspectives | head
```

Expected filenames:

- `000123_pitch00_yaw05.png` style pattern

### Stage 3: Feature Extraction

Check:

```bash
ls $DATA_FINAL_ROOT/$PLACE/$BUILDING/$FLOOR/features
```

Expected files:

- `local_features.h5`
- `global_features_<feature_model>.h5`

### Stage 4/5: Matching and COLMAP

Check:

```bash
ls $DATA_TEMP_ROOT/$PLACE/$BUILDING/$FLOOR/colmap_sfm
ls $DATA_FINAL_ROOT/$PLACE/$BUILDING/$FLOOR/colmap_map
```

Expected files:

- `pairs.txt`
- `matches.h5`
- `database.db`
- triangulation outputs under `colmap_map`

## Step 5: Align to Floorplan

After mapping finishes, run the aligner:

```bash
python -m unav.run_aligner   $DATA_TEMP_ROOT   $DATA_FINAL_ROOT   $PLACE   $BUILDING   $FLOOR
```

The aligner creates:

```text
$DATA_FINAL_ROOT/$PLACE/$BUILDING/$FLOOR/transform_matrix.npy
```

See [Floorplan Alignment](aligner.md) for GUI details.

## Step 6: Mark Boundaries and Destinations

Open the floorplan in `labelme`, draw walkable regions, obstacles, doors, waypoints, connectors, and destinations, then save:

```text
$DATA_FINAL_ROOT/$PLACE/$BUILDING/$FLOOR/boundaries.json
```

See [Map Labeling & Multi-language](map-labeling-multilang.md) for exact `group_id` rules.

## Step 7: Add Multilingual Labels

If the deployment needs multiple languages, run the translation GUI and populate:

```text
$DATA_FINAL_ROOT/_i18n/labels.json
```

## Step 8: Prepare Upload Package

The upload or backend handoff should include the final `DATA_FINAL_ROOT` files, not only the raw video. At minimum, each floor needs:

- `floorplan.png`
- `transform_matrix.npy`
- `boundaries.json`
- `features/`
- `colmap_map/`

## Typical Runtime Failure Points

- Input video is a reframed flat MP4 rather than a spherical 360 MP4.
- Video filename does not exactly match `<floor>.mp4`.
- `orb_vocab.fbow` is missing from `DATA_TEMP_ROOT`.
- Docker image or GPU runtime for SLAM is not available.
- `colmap` command is not installed or not on `PATH`.
- Feature model checkpoint files are missing under `DATA_FINAL_ROOT/parameters/...`.
- Floorplan and video are from different floors or different building versions.
