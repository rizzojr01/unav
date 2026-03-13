# UNav Self-Serve Mapping Documentation

This site is for third-party partners onboarding a new indoor map into UNav.

Goal: start from raw 360 video and end with a navigation-ready map package.

## End-to-End Flow

1. Prepare hardware/software prerequisites.
2. Collect 360 mapping video in the target space.
3. Place raw input files in the required folder structure.
4. Run mapping pipeline (`python -m unav.run_mapper ...`).
5. Align SLAM map to floorplan (`python -m unav.run_aligner ...`).
6. Label map entities with `labelme` (`boundaries.json`).
7. Add multilingual labels with UNav translation GUI.
8. Deliver final map artifacts under `DATA_FINAL_ROOT`.

## Who Should Use This

- Integrators onboarding a building/floor for the first time
- Teams operating mapping as a repeatable field workflow
- Engineers validating map readiness before API integration

## Read in This Order

1. [Getting Started](getting-started.md)
2. [Data Collection SOP](data-collection-sop.md)
3. [Input Output Contract](io-contract.md)
4. [Mapping Workflow](mapping-workflow.md)
5. [Floorplan Alignment](aligner.md)
6. [Map Labeling & Multi-language](map-labeling-multilang.md)
7. [Troubleshooting](troubleshooting.md)

## Command Summary

```bash
# Mapping
python -m unav.run_mapper <data_temp_root> <data_final_root> <feature_model> <place> <building> <floor>

# Backward-compatible alias
python -m unav.run_mapping <data_temp_root> <data_final_root> <feature_model> <place> <building> <floor>

# Alignment
python -m unav.run_aligner <data_temp_root> <data_final_root> <place> <building> <floor>
```
