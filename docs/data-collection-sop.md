# Data Collection SOP

This section answers: I am in a new indoor space. What exactly should I do with the camera?

The field capture workflow is based on the internal `Insta360/map gen UNav Mapping SOP`. Use an Insta360-class 360 camera where possible. GoPro MAX can be used as an alternative if it exports a spherical MP4 with comparable quality.

## Capture Goal

For each floor, produce one continuous 360 video that sees every walkable area that should be supported by UNav navigation. The mapping pipeline uses this video to recover camera motion, slice perspective images, extract visual features, and build the map artifacts used later for localization.

## Hardware Checklist

- Insta360 360 camera, charged and with enough storage for a full-floor recording
- Phone with the Insta360 app installed, used to connect to the camera over Wi-Fi
- Selfie stick or monopod that can hold the camera slightly above head height
- Extra batteries or charging cable for long captures
- A floorplan image for the same floor, saved for the alignment and labeling steps
- Optional assistant for opening doors, turning lights on, and taking route notes

## Before Recording

1. Turn on the Insta360 camera.
2. Connect the phone to the camera over Wi-Fi and verify the live preview.
3. Mount the camera on the selfie stick slightly above head height.
4. Turn on lights in all spaces that will be mapped.
5. Pre-open doors where possible, especially doors to rooms that should be navigable.
6. Remove or avoid moving objects where possible.
7. Confirm the floorplan image matches the floor being recorded.
8. Start a simple route log: note the floor, start point, route order, and any skipped rooms.

## Camera Settings

- Record in 360 video mode.
- Use the highest practical resolution; 5K or higher is recommended.
- Keep stabilization enabled if available.
- Hold the selfie stick steady and keep the camera above head height.
- Walk at normal speed in open areas and slow down at doors, corners, narrow halls, and visually repetitive spaces.
- Avoid sharp turns; use large round turns.
- Keep walking continuously where possible. Do not stop and rotate in place unless needed for coverage.

## Walking Procedure Per Floor

Record at least three loops when time permits. If only one video can be delivered, keep the loops in the same continuous video.

### Loop 1: Main Closed Loop

1. Start from a clear landmark such as an elevator lobby, reception desk, stairwell, or main entrance.
2. Walk the main corridors and primary circulation paths.
3. Make the route a closed loop where possible: return to the original start point or an already visited location.
4. Keep the camera moving smoothly and avoid abrupt turns.

### Loop 2: Rooms and Side Areas

1. Enter each accessible office, classroom, conference room, clinic room, or branch area that should be mapped.
2. For a small office, enter the room, make a small loop, and return to the entrance.
3. For a large office, classroom, or conference room, walk around the outer perimeter or as close to the outer perimeter as possible.
4. Return to the corridor after each side area so the route reconnects to known visual context.

### Loop 3: Occlusions, Doors, and Scene Variation

1. Revisit important corridors and intersections.
2. If there are columns, partitions, furniture clusters, or other barriers that block the view, loop around them so all sides are visible.
3. Capture door transitions slowly. If doors may be open or closed during real use, capture representative door states when possible.
4. Revisit visually similar areas from more than one direction to improve loop closure and localization robustness.

## Route Logging

Keep a lightweight note while recording. This helps the map builder debug failures and align the video to the floorplan.

Recommended route log fields:

```text
place: New_York_City
building: LightHouse
floor: 4_floor
camera: Insta360 X3
video_file: 4_floor.mp4
start_point: elevator lobby
route_order: elevator lobby -> main corridor -> reception -> exam rooms -> conference room -> elevator lobby
skipped_areas: storage closet 402 locked
notes: north stair door closed; west corridor crowded during first pass
```

## Coverage Rules

- Every public walkable area that should be navigable must appear in the video.
- Every destination area should be visible from the corridor and near the final destination point.
- Slow down at doors, turns, elevators, stairs, and visually repetitive hallways.
- Avoid people walking close to the camera where possible.
- Avoid mirrors, glass, and reflective surfaces when a small route adjustment can reduce glare.
- Repeat problematic segments if they are dark, crowded, reflective, or blocked.
- End the recording at the original start point or a previously visited location when possible.

## Export Requirements

The mapping input should be a spherical/equirectangular MP4 file.

For Insta360 cameras, export the 360 video from Insta360 Studio or the mobile app without reframing. Do not export a normal flat reframed video, because the mapping pipeline expects a 360 video.

For GoPro MAX `.360` files, export through GoPro Player as a spherical MP4. Choose 4K or higher if available. Auxiliary `.LRV` and `.THM` preview files are not mapping inputs.

## File Naming and Placement

Final mapping input must be named exactly `<floor>.mp4` and placed as:

```text
<DATA_TEMP_ROOT>/<place>/<building>/<floor>/<floor>.mp4
```

Example:

```text
/mnt/data/UNav-IO/temp/New_York_City/LightHouse/4_floor/4_floor.mp4
```

If you recorded multiple loops as separate clips, either merge them into one floor video before mapping or choose the best complete continuous clip and name it exactly `<floor>.mp4`.

## Floorplan and Scale Inputs

Before alignment, prepare:

- `<DATA_FINAL_ROOT>/<place>/<building>/<floor>/floorplan.png`
- `<DATA_FINAL_ROOT>/scale.json`

You need meters-per-pixel for each floor.

1. Mark at least 3 point pairs on the floorplan.
2. Measure real-world distances between those points.
3. Compute average `meters_per_pixel` and store it in `scale.json`.

Example:

```json
{
  New_York_City: {
    LightHouse: {
      4_floor: 0.0124
    }
  }
}
```

## Field QA Checklist Before Leaving Site

- The video file exists and plays fully.
- The video is spherical 360 MP4, not a reframed flat video.
- The camera was held slightly above head height.
- Main corridors form at least one closed loop.
- Rooms and side spaces are covered.
- Doors, elevators, stairs, and intersections are captured slowly.
- Columns or occluding structures are captured from multiple sides.
- The route log records start point, route order, skipped areas, and unusual conditions.
- The floorplan image for the same floor is available.
- Data is copied to at least two storage locations.
