#!/bin/bash
# UNav Floor Map Analyzer Batch Runner Script
#
# This script batch-processes floor depth analysis for multiple places,
# buildings, and floors. It generates floor point clouds and 2D floor maps
# using DA3 depth estimation and SAM3 floor segmentation.
#
# Usage:
#   ./run_floor_map.sh
#
# Variables below can be modified for your own jobs.


# ------------- User-Configurable Section -------------

DATA_TEMP_ROOT="/mnt/data/UNav-IO/temp"
DATA_FINAL_ROOT="/mnt/data/UNav-IO/data"

PLACES=("New_York_City")
BUILDINGS=("LightHouse")
FLOORS=("3_floor")

# Number of keyframes to process (more = better coverage, but slower)
NUM_IMAGES=10

# ------------- Main Batch Processing Loop ------------

for place in "${PLACES[@]}"; do
  for building in "${BUILDINGS[@]}"; do
    for floor in "${FLOORS[@]}"; do
      echo "---------------------------------------------"
      echo ">> Floor Map: Place=$place | Building=$building | Floor=$floor"
      echo "---------------------------------------------"
      python -m unav.run_floor_map \
        "$DATA_TEMP_ROOT" \
        "$DATA_FINAL_ROOT" \
        "$place" \
        "$building" \
        "$floor" \
        "$NUM_IMAGES"
      echo ""
    done
  done
done

echo "All floor depth jobs finished."
