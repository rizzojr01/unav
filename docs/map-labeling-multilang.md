# Map Labeling & Multi-language

This page covers:

1. How to annotate `boundaries.json` with `labelme`
2. How to run UNav translation GUI for multilingual labels

Reference examples:

- `https://github.com/ai4ce/UNav_Navigation/tree/main/example_data`

## Part A: Map Labeling with Labelme

### Output File

Save annotation file as:

```text
<DATA_FINAL_ROOT>/<place>/<building>/<floor>/boundaries.json
```

### Group and Shape Rules (Code-Aligned)

The parser in `unav/navigator/pathfinder.py` expects the following:

| group_id | shape_type | meaning | required fields |
|---|---|---|---|
| 0 | polygon or rectangle | walkable room/region | `label` optional room name |
| 1 | polygon or rectangle | obstacle (non-walkable) | - |
| 2 | polygon or rectangle | door area (added to walkable union) | `label` optional door name |
| 3 | point | normal navigation waypoint | recommended `label=waypoint` |
| 4 | point | inter-floor waypoint (stairs/elevator connector) | `label` connector ID, `description` should be `staircase` or `elevator` |
| 5 | point | destination | `label` destination name, `description` orientation hint (`left/right/center/up/...`) |
| 6 | line | companion line for group 4 waypoint | `label` must match group 4 label |

### Labelme Authoring Procedure

1. Open floorplan image in `labelme`.
2. Draw walkable polygons (`group_id=0`).
3. Draw obstacles (`group_id=1`).
4. Draw doors (`group_id=2`).
5. Place navigation waypoints (`group_id=3`).
6. Place stair/elevator transfer points (`group_id=4`).
7. Place destination points (`group_id=5`).
8. Draw companion lines for each group 4 connector (`group_id=6`, same label).
9. Save JSON as `boundaries.json` in floor folder.

### Inter-floor Connector Rule

To connect floors, `group_id=4` labels must match across floors/buildings.

Example:

- `LH-e1` on floor 3 and `LH-e1` on floor 4 means same elevator shaft
- `description=elevator` or `description=staircase` controls cross-floor penalty logic

### Minimal JSON Example

```json
{
  "shapes": [
    {"label": "corridor", "group_id": 0, "shape_type": "polygon", "points": [[0,0],[100,0],[100,20],[0,20]]},
    {"label": "pillar", "group_id": 1, "shape_type": "polygon", "points": [[40,5],[50,5],[50,15],[40,15]]},
    {"label": "door_A", "group_id": 2, "shape_type": "rectangle", "points": [[98,8],[104,12]]},
    {"label": "waypoint", "group_id": 3, "shape_type": "point", "points": [[20,10]]},
    {"label": "LH-e1", "group_id": 4, "shape_type": "point", "description": "elevator", "points": [[80,10]]},
    {"label": "Main Elevator", "group_id": 5, "shape_type": "point", "description": "up", "points": [[82,10]]},
    {"label": "LH-e1", "group_id": 6, "shape_type": "line", "points": [[80,6],[80,14]]}
  ]
}
```

## Part B: Multi-language Labels with UNav GUI

UNav translation editor writes labels to:

```text
<DATA_FINAL_ROOT>/_i18n/labels.json
```

### Option 1: Use your wrapper in `/home/unav/Desktop/unav-run`

```bash
cd /home/unav/Desktop/unav-run
./run_translator.sh -r <DATA_FINAL_ROOT> -H 127.0.0.1 -p 5001
```

Then open:

```text
http://127.0.0.1:5001
```

### Option 2: Run module directly from this repo

```bash
python -m unav.mapper.tools.i18n_label_web \
  --data-final-root <DATA_FINAL_ROOT> \
  --use-nav \
  --host 127.0.0.1 \
  --port 5001
```

### `--use-nav` vs file mode

- `--use-nav`: derives Place/Building/Floor/Destination tree from navigation assets (`boundaries.json`)
- no `--use-nav`: falls back to scanning floor folders and optional `destinations.json`

Fallback `destinations.json` format:

```json
[
  {"id": "101", "name": "Reception"},
  {"id": "102", "name": "Elevator"}
]
```

### `labels.json` Structure

```json
{
  "places": {
    "New_York_City": {"en": "New York City", "zh-Hans": "纽约"}
  },
  "buildings": {
    "New_York_City/LightHouse": {"en": "LightHouse", "zh-Hans": "灯塔楼"}
  },
  "floors": {
    "New_York_City/LightHouse/4_floor": {"en": "4F", "zh-Hans": "四层"}
  },
  "destinations": {
    "New_York_City/LightHouse/4_floor/79": {"en": "Reception", "zh-Hans": "接待处"}
  },
  "aliases": {
    "zh-Hans": {
      "接待处": "New_York_City/LightHouse/4_floor/79"
    }
  }
}
```

## Final Validation Checklist

- `boundaries.json` exists and loads without parse errors
- All required `group_id` categories are present
- `group_id=4` labels match across floors where transitions should exist
- `_i18n/labels.json` created and populated
- Target language labels resolve correctly in app/localization flow
