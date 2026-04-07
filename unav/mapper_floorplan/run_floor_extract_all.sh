#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_floor_extract_all.sh [DATA_ROOT]
# Default DATA_ROOT: /mnt/data/UNav-IO/temp

DATA_ROOT="${1:-/mnt/data/UNav-IO/temp}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRACT_SCRIPT="${SCRIPT_DIR}/extract_floor_auto.sh"

if [[ ! -x "$EXTRACT_SCRIPT" ]]; then
  echo "[ERROR] extract_floor_auto.sh not found or not executable: $EXTRACT_SCRIPT" >&2
  exit 1
fi

# Build candidate list: has materials but missing floor_map outputs
CANDIDATES=$(
  DATA_ROOT="$DATA_ROOT" python3 - <<'PY'
import os
root = os.environ.get('DATA_ROOT', '/mnt/data/UNav-IO/temp')
if not os.path.isdir(root):
    print("")
    raise SystemExit(0)

out = []
for place in sorted(os.listdir(root)):
    place_dir = os.path.join(root, place)
    if not os.path.isdir(place_dir):
        continue
    for building in sorted(os.listdir(place_dir)):
        building_dir = os.path.join(place_dir, building)
        if not os.path.isdir(building_dir):
            continue
        for floor in sorted(os.listdir(building_dir)):
            floor_dir = os.path.join(building_dir, floor)
            if not os.path.isdir(floor_dir):
                continue

            slam_dir = os.path.join(floor_dir, 'stella_vslam_dense')
            if not os.path.isdir(slam_dir):
                continue

            if not os.path.isfile(os.path.join(slam_dir, 'final_map.msg')):
                continue

            keyframes_dir = os.path.join(slam_dir, 'keyframes')
            if not os.path.isdir(keyframes_dir):
                continue

            has_images = any(name.startswith('image') and name.endswith('.png') for name in os.listdir(keyframes_dir))
            if not has_images:
                continue

            floor_map_dir = os.path.join(floor_dir, 'floor_map')
            if os.path.isdir(floor_map_dir):
                has_output = (
                    os.path.exists(os.path.join(floor_map_dir, 'floor_metadata.json')) or
                    os.path.exists(os.path.join(floor_map_dir, 'floor_points_depth.ply')) or
                    os.path.exists(os.path.join(floor_map_dir, 'floor_map_depth.png'))
                )
                if has_output:
                    continue

            out.append((place, building, floor))

print("\n".join(["|".join(x) for x in out]))
PY
)

if [[ -z "${CANDIDATES// }" ]]; then
  echo "[INFO] No pending floors found under $DATA_ROOT"
  exit 0
fi

echo "[INFO] Pending floors:" 
echo "$CANDIDATES" | sed 's/^/  - /'

# Run extraction for each pending floor
while IFS='|' read -r PLACE BUILDING FLOOR; do
  [[ -z "$PLACE" ]] && continue
  echo ""
  echo "[RUN] $PLACE / $BUILDING / $FLOOR"
  # Auto-answer "n" for any re-generate prompts
  yes n | "$EXTRACT_SCRIPT" -m depth "$PLACE" "$BUILDING" "$FLOOR" || echo "[WARN] Failed: $PLACE / $BUILDING / $FLOOR"
  echo "[DONE] $PLACE / $BUILDING / $FLOOR"
  echo "------------------------------------------------------------"
done <<< "$CANDIDATES"

