# Troubleshooting

## `python -m unav.run_mapper ...` fails immediately

### Symptom

Usage printed or missing argument error.

### Fix

Provide all 6 positional args:

```bash
python -m unav.run_mapper <data_temp_root> <data_final_root> <feature_model> <place> <building> <floor>
```

## SLAM stage fails in Docker

### Symptom

`stella_vslam_dense` container exits or no SLAM outputs.

### Fix

- Build image `stella_vslam_dense`
- Verify Docker GPU runtime works (`docker run --gpus ...`)
- Check `<DATA_TEMP_ROOT>/orb_vocab.fbow`
- Check `<DATA_TEMP_ROOT>/<place>/<building>/<floor>/<floor>.mp4`

## COLMAP stage fails

### Symptom

`colmap: command not found` or triangulation error.

### Fix

- Install COLMAP and ensure it is on `PATH`
- Confirm `pairs.txt`, `matches.h5`, `database.db` exist under `colmap_sfm`
- Re-run mapping from start if feature files are incomplete

## Feature model checkpoint not found

### Symptom

Error loading global descriptor checkpoint.

### Fix

The model loader uses `data_final_root` as parameter root. Verify checkpoint path exists under:

```text
<DATA_FINAL_ROOT>/parameters/... 
```

Example for DinoV2Salad:

```text
<DATA_FINAL_ROOT>/parameters/DinoV2Salad/ckpts/dino_salad.ckpt
```

## Aligner cannot compute matrix

### Symptom

Transform computation error.

### Fix

- Add at least 4 valid correspondences
- Ensure selected points are spread across map (not clustered)
- Verify `floorplan.png` and trajectory/keyframes are from same floor

## Translation GUI shows empty tree

### Symptom

No place/building/floor/destination listed.

### Fix

- With `--use-nav`: ensure `boundaries.json` exists per floor
- Without `--use-nav`: provide `destinations.json` per floor
- Check `<DATA_FINAL_ROOT>` path passed to GUI

## Port already in use for translation GUI

### Symptom

Cannot bind to requested port.

### Fix

- Use another port (`-p 5002`)
- `run_translator.sh` already auto-fallbacks to next free port
