#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_extract_floor_from_depth.sh [PLACE] [BUILDING] [FLOOR] [SCALE] [SUBSAMPLE]
# Defaults use config.yaml for place/building/floor and 2.0/8 for scale/subsample.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.yaml"

read_yaml() {
    python3 -c "
import yaml
with open('${CONFIG_FILE}', 'r') as f:
    config = yaml.safe_load(f)
keys = '$1'.split('.')
value = config
for key in keys:
    if value is None:
        break
    value = value.get(key)
print(value if value is not None else '')
"
}

DATA_ROOT=$(read_yaml "data_root")
DEFAULT_PLACE=$(read_yaml "default_place")
DEFAULT_BUILDING=$(read_yaml "default_building")
DEFAULT_FLOOR=$(read_yaml "default_floor")

SLAM_DIR_NAME=$(read_yaml "dir_names.slam")
KEYFRAMES_DIR_NAME=$(read_yaml "dir_names.keyframes")
FLOOR_MAP_DIR_NAME=$(read_yaml "dir_names.floor_map")
MASK_DIR_NAME=$(read_yaml "dir_names.mask")
DEPTH_DIR_NAME=$(read_yaml "dir_names.depth")

DB_FILENAME=$(read_yaml "file_names.database")

DEPTH_PATTERN=$(read_yaml "da2.depth_pattern")
MASK_PATTERN=$(read_yaml "extraction.mask_pattern")

PLACE="${1:-$DEFAULT_PLACE}"
BUILDING="${2:-$DEFAULT_BUILDING}"
FLOOR="${3:-$DEFAULT_FLOOR}"
SCALE="${4:-2.0}"
SUBSAMPLE="${5:-8}"

if [[ -z "$PLACE" || -z "$BUILDING" || -z "$FLOOR" ]]; then
  echo "Missing place/building/floor. Provide args or set defaults in config.yaml" >&2
  exit 1
fi

SLAM_DIR="${DATA_ROOT}/${PLACE}/${BUILDING}/${FLOOR}/${SLAM_DIR_NAME}"
FLOOR_MAP_DIR="${DATA_ROOT}/${PLACE}/${BUILDING}/${FLOOR}/${FLOOR_MAP_DIR_NAME}"

SQLITE_DB="${SLAM_DIR}/${DB_FILENAME}"
DEPTH_DIR="${FLOOR_MAP_DIR}/${DEPTH_DIR_NAME}"
MASK_DIR="${FLOOR_MAP_DIR}/${MASK_DIR_NAME}"

python3 "${SCRIPT_DIR}/extract_floor_from_depth.py" \
  "${SQLITE_DB}" \
  "${DEPTH_DIR}" \
  "${MASK_DIR}" \
  "${FLOOR_MAP_DIR}" \
  --depth-pattern "${DEPTH_PATTERN}" \
  --mask-pattern "${MASK_PATTERN}" \
  --subsample "${SUBSAMPLE}" \
  --scale "${SCALE}"
