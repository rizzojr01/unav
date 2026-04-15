# UNav: Unified Visual Navigation System

UNav is a modular visual navigation framework for indoor mapping, localization, and navigation across multi-floor buildings.

## Install

For the full UNav core package:

```sh
pip install git+https://github.com/ai4ce/unav.git
```

For self-serve mapping runner dependencies, use the dedicated `unav-run` lockfiles:

- [`unav-run/pyproject.toml`](https://github.com/endeleze/unav-run/blob/main/pyproject.toml)
- [`unav-run/poetry.lock`](https://github.com/endeleze/unav-run/blob/main/poetry.lock)

Legacy full-core requirements are explicitly located at [`ai4ce/unav/requirements.txt`](https://github.com/ai4ce/unav/blob/main/requirements.txt).

## Self-Serve Mapping Documentation (External Partner)

If you are onboarding a new building/floor, use the full step-by-step guide:

- [docs/index.md](docs/index.md)

This guide covers:

- Hardware and software prerequisites
- How to walk a new indoor space with a 360 camera
- Exact input/output folder contract
- Mapping and alignment commands
- Map labeling (`labelme`) and multi-language workflow
- Required map deliverables and troubleshooting

Preview docs locally:

```sh
pip install mkdocs mkdocs-material
mkdocs serve
```

## Quick Commands

### 1. Mapping Pipeline

Canonical command:

```sh
python -m unav.run_mapper <data_temp_root> <data_final_root> <feature_model> <place> <building> <floor>
```

Backward-compatible alias:

```sh
python -m unav.run_mapping <data_temp_root> <data_final_root> <feature_model> <place> <building> <floor>
```

Arguments:

- `data_temp_root`: root for raw map input files and intermediate artifacts
- `data_final_root`: root for final map outputs used by localization/navigation
- `feature_model`: one of `DinoV2Salad`, `MixVPR`, `CricaVPR`, `NetVlad`, `AnyLoc`
- `place`, `building`, `floor`: dataset identifiers used in output paths

Example:

```sh
python -m unav.run_mapper /mnt/data/UNav-IO/temp /mnt/data/UNav-IO/data DinoV2Salad New_York_City LightHouse 4_floor
```

### 2. Floorplan Alignment (Required)

```sh
python -m unav.run_aligner <data_temp_root> <data_final_root> <place> <building> <floor>
```

Required before launching aligner:

- `<data_final_root>/<place>/<building>/<floor>/floorplan.png`
- `<data_final_root>/scale.json` (meters per pixel)

Output:

- `<data_final_root>/<place>/<building>/<floor>/transform_matrix.npy`

### 3. Map Labeling + Multi-language

Label map with `labelme` and save:

- `<data_final_root>/<place>/<building>/<floor>/boundaries.json`

Run translation GUI:

```sh
python -m unav.mapper.tools.i18n_label_web --data-final-root <data_final_root> --use-nav --port 5001
```

## Other Modules

- Mapping module docs: `unav/mapper/README.md`
- Localization module docs: `unav/localizer/README.md`
- Navigation module docs: `unav/navigator/README.md`
