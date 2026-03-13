# Data Collection SOP

This section answers: "I am in a new indoor space. What exactly should I do with the camera?"

## Before Entering the Space

1. Charge camera batteries to full.
2. Verify enough storage for full-floor recording.
3. Clean all camera lenses.
4. Prepare a floorplan image (for later alignment and labeling).
5. Assign at least one assistant for door handling.

## Camera Settings

- Resolution: highest available (5K recommended)
- Stabilization: on
- Orientation: keep camera forward along walking direction
- Motion: steady walk, avoid sudden spins

## Walking Procedure (Per Floor)

Run at least 3 loops.

### Loop 1: Main Corridors (Loop Closure)

1. Walk major corridors and primary circulation paths.
2. Keep movement continuous and form a closed loop if possible.
3. Assistant opens doors ahead of time.

### Loop 2: Rooms and Side Areas (Coverage)

1. Enter each accessible room/branch area.
2. Make a short pass inside, then return to corridor.
3. Cover blind corners and dead ends.

### Loop 3: Door Interaction (Scene Diversity)

1. Start with doors closed where possible.
2. Open/close doors while entering/leaving.
3. Revisit key paths to capture appearance variation.

## Coverage Rules

- Every walkable public area must appear in video.
- Slow down at narrow passages and door transitions.
- Repeat problematic segments (dark, reflective, crowded).

## File Naming and Placement

Final mapping input must be placed as:

```text
<DATA_TEMP_ROOT>/<place>/<building>/<floor>/<floor>.mp4
```

Example:

```text
/mnt/data/UNav-IO/temp/New_York_City/LightHouse/4_floor/4_floor.mp4
```

If you recorded multiple loops separately, merge into one floor video before mapping, or keep one high-quality representative video named exactly `<floor>.mp4`.

## Scale Measurement for Alignment

You need meters-per-pixel for each floor.

1. Mark at least 3 point pairs on floorplan.
2. Measure real-world distances between those points.
3. Compute average `meters_per_pixel` and store in `scale.json`.

Example:

```json
{
  "New_York_City": {
    "LightHouse": {
      "4_floor": 0.0124
    }
  }
}
```

## Field QA Checklist (Before Leaving Site)

- Video file exists and can be played fully
- Major corridors are covered
- Rooms and side spaces are covered
- Key doors captured at least once
- Data copied to two storage locations
