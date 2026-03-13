# Floorplan Alignment

This step creates `transform_matrix.npy`, required for mapping floorplan pixels and SLAM coordinates.

## Required Inputs

- `<DATA_TEMP_ROOT>/<place>/<building>/<floor>/stella_vslam_dense/final_map.msg`
- `<DATA_TEMP_ROOT>/<place>/<building>/<floor>/stella_vslam_dense/keyframes/`
- `<DATA_TEMP_ROOT>/<place>/<building>/<floor>/stella_vslam_dense/eval_logs/keyframe_trajectory.txt`
- `<DATA_FINAL_ROOT>/<place>/<building>/<floor>/floorplan.png`
- `<DATA_FINAL_ROOT>/scale.json`

## Run Command

```bash
python -m unav.run_aligner <data_temp_root> <data_final_root> <place> <building> <floor>
```

Example:

```bash
python -m unav.run_aligner \
  /mnt/data/UNav-IO/temp \
  /mnt/data/UNav-IO/data \
  New_York_City LightHouse 4_floor
```

## GUI Operation Steps

1. Select a keyframe from the left panel.
2. Double-click keyframe panel and choose a stable visual feature.
3. Double-click floorplan panel and place the corresponding 2D point.
4. Repeat for multiple correspondences across floor (corners, junctions, doors).
5. Keep adding correspondences until transform error is stable.
6. Click `Save Matrix`.

## Minimum Correspondences

At least 4 2D↔3D correspondences are required.

## Output

- `<DATA_FINAL_ROOT>/<place>/<building>/<floor>/transform_matrix.npy`

## Quality Checks

- Correspondence error in GUI table should be low and consistent.
- Projected trajectory should align with corridor layout on floorplan.
- Re-open GUI and verify matrix reloads without visible drift.
