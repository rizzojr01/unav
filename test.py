"""
test.py
Test script for the UNav Segment Mapper Pipeline.

This script sets up test parameters and directly calls the main() function of run_mapper_segment.py,
simulating command-line execution without subprocess.

Usage:
    python test.py
"""

import sys
from unav import run_mapper_segment

# ================== EDIT YOUR TEST PARAMETERS BELOW ==================
DATA_TEMP_ROOT = "/mnt/data/UNav-IO/temp"
DATA_FINAL_ROOT = "/mnt/data/UNav-IO/data"
FEATURE_MODEL = "DinoV2Salad"
PLACE = "New_York_City"
BUILDING = "LightHouse"
FLOOR = "gerrard-hall"
# =====================================================================

def main():
    # Simulate command-line arguments
    sys.argv = [
        "run_mapper_segment.py",
        DATA_TEMP_ROOT,
        DATA_FINAL_ROOT,
        FEATURE_MODEL,
        PLACE,
        BUILDING,
        FLOOR
    ]

    # Import and invoke the pipeline main function
    run_mapper_segment.main()

if __name__ == "__main__":
    main()
