# navigation_commands.py

import math
from typing import List, Dict, Any, Tuple, Literal, Union
from shapely.geometry import LineString
from unav.navigator.nav_text import nav_text, unit_text

import math
from typing import List, Dict, Any, Tuple, Literal, Union
from shapely.geometry import LineString
from unav.navigator.nav_text import nav_text, unit_text

def normalize_angle(angle: float) -> float:
    """
    Normalize an angle to the range [-180, 180] degrees.

    Args:
        angle: Angle in degrees.

    Returns:
        Normalized angle in [-180, 180].
    """
    return (angle + 180) % 360 - 180

def convert_distance_meta(meters: float, unit: Literal["meter", "feet"], lang: str) -> Tuple[str, float, str]:
    """
    Convert a distance in meters to a localized string and return the numeric value and unit.

    Args:
        meters: Distance in meters.
        unit: "meter" or "feet".
        lang: Language code.

    Returns:
        Tuple of (localized string, numeric value, unit).
    """
    if unit == "feet":
        value = meters * 3.28084
        localized = unit_text(value, "feet", lang)
        return localized, value, "feet"
    elif unit == "meter":
        localized = unit_text(meters, "meter", lang)
        return localized, meters, "meter"
    else:
        raise ValueError("Unit must be 'meter' or 'feet'.")

# --- Add this helper just below normalize_angle()/convert_distance_meta() ---

def angle_to_clock_hour(turn_deg: float) -> int:
    """
    Map a signed relative turn angle (in degrees) to a 12-hour clock index.
    Right turns are mapped to hours 1..6, left turns to 11..7.
    """
    raw = -turn_deg  # positive => right, negative => left (match existing code logic)
    clock_n = int(round(raw / 30.0)) % 12
    hour = 12 if clock_n == 0 else clock_n
    return hour

def classify_turn_sector(turn_deg: float) -> Tuple[str, str]:
    """
    Classify the relative turn into fine-grained sectors based on your clock ranges.

    Bins (absolute raw angle = |-turn|):
      0°–15°                 -> 'ahead'            (11:30–12:30)
      15°–45°                -> 'very_slight'
      45°–75°                -> 'slight'
      75°–105°               -> 'turn'
      105°–135°              -> 'sharp'
      135°–165°              -> 'very_sharp'
      >=165° (up to 180°)    -> 'u_turn'           (5:30–6:30)

    Returns:
      qual: one of {'ahead','very_slight','slight','turn','sharp','very_sharp','u_turn'}
      direction: 'left' | 'right'  (for 'ahead'/'u_turn' the direction is still set
                                    by sign, but you may ignore it downstream)
    """
    # raw angle: positive => right, negative => left (consistent with hour mapping)
    raw = -turn_deg
    abs_raw = abs(raw)

    direction = 'right' if raw > 0 else 'left'

    if abs_raw <= 15.0:
        return 'ahead', direction
    elif abs_raw <= 45.0:
        return 'very_slight', direction
    elif abs_raw <= 75.0:
        return 'slight', direction
    elif abs_raw <= 105.0:
        return 'turn', direction
    elif abs_raw <= 135.0:
        return 'sharp', direction
    elif abs_raw <= 165.0:
        return 'very_sharp', direction
    else:
        return 'u_turn', direction

def commands_from_result(
    navigator,
    path_result: Dict[str, Any],
    initial_heading: float,
    unit: Literal["meter", "feet"] = "meter",
    language: str = "en"
) -> List[Dict[str, Any]]:
    """
    Generate step-by-step navigation instructions with semantic tags.

    Each command is represented as a dictionary containing:
    - tag: semantic label of the command (e.g., 'turn', 'forward', 'arrive')
    - text: localized instruction string
    - meta: metadata for further processing (e.g., distance and unit split)

    Args:
        navigator: FacilityNavigator instance providing scale and floorplan.
        path_result: Path planning result with keys, coordinates, and descriptions.
        initial_heading: Starting heading in degrees.
        unit: Distance unit, either 'meter' or 'feet'.
        language: Language code for localization.

    Returns:
        List of dictionaries, each representing a navigation command.
    """
    if 'error' in path_result:
        raise ValueError(f"Cannot generate commands: {path_result['error']}")

    coords = path_result["path_coords"]
    labels = path_result["path_labels"]
    keys = path_result["path_keys"]
    descriptions = path_result["path_descriptions"]

    commands: List[Dict[str, Any]] = []

    # --- Announce starting location ---
    if len(keys) > 1 and keys[1] != "VIRT":
        if isinstance(keys[1], tuple) and len(keys[1]) == 4:
            floor_key = keys[1][:3]
            place, building, floor = floor_key
            pf0 = navigator.pf_map[floor_key]
            room = pf0.get_current_room(coords[0]) if hasattr(pf0, 'get_current_room') else ""
            commands.append({
                "tag": "start_in",
                "text": nav_text("start_in", language, room=room, floor=floor, building=building, place=place),
                "meta": {"room": room, "floor": floor, "building": building, "place": place}
            })
        else:
            commands.append({
                "tag": "start_nav",
                "text": nav_text("start_nav", language),
                "meta": {}
            })
    else:
        commands.append({
            "tag": "start_nav",
            "text": nav_text("start_nav", language),
            "meta": {}
        })

    heading = initial_heading
    i = 0
    straight_distance = 0.0
    door_events = []

    # --- Main navigation loop ---
    while i < len(coords) - 1:
        key0, key1 = keys[i], keys[i + 1]
        p0, p1 = coords[i], coords[i + 1]
        desc1 = descriptions[i + 1].lower()
        dx, dy = p1[0] - p0[0], p0[1] - p1[1]
        segment_dist = math.hypot(dx, dy)

        # Get map scale
        scale = navigator.scales.get(key1[:3], 1.0) if isinstance(key1, tuple) and len(key1) == 4 else 1.0

        # --- Handle transitions ---
        if isinstance(key0, tuple) and isinstance(key1, tuple) and len(key0) == 4 and len(key1) == 4:
            place0, building0, floor0, _ = key0
            place1, building1, floor1, _ = key1

            # Transition: place/building/floor
            if place0 != place1 or building0 != building1 or floor0 != floor1:
                if straight_distance > 0:
                    dist_text, dist_val, dist_unit = convert_distance_meta(straight_distance * scale, unit, language)
                    if door_events:
                        door_pos = min(door_events, key=lambda d: d["dist"])
                        door_text, door_val, door_unit = convert_distance_meta(door_pos["dist"] * scale, unit, language)
                        commands.append({
                            "tag": "forward_door",
                            "text": nav_text("forward_door", language, dist=dist_text, door_dist=door_text),
                            "meta": {
                                "distance": dist_val,
                                "unit": dist_unit,
                                "door_distance": door_val,
                                "door_unit": door_unit
                            }
                        })
                        door_events.clear()
                    else:
                        commands.append({
                            "tag": "forward",
                            "text": nav_text("forward", language, dist=dist_text),
                            "meta": {
                                "distance": dist_val,
                                "unit": dist_unit
                            }
                        })
                    straight_distance = 0.0

                if place0 != place1:
                    commands.append({
                        "tag": "transition_place",
                        "text": nav_text("transition_place", language, place=place1),
                        "meta": {"place": place1}
                    })
                    commands.append({
                        "tag": "proceed_to",
                        "text": nav_text("proceed_to", language, floor=floor1, building=building1, place=place1),
                        "meta": {"floor": floor1, "building": building1, "place": place1}
                    })

                elif building0 != building1:
                    commands.append({
                        "tag": "transition_building",
                        "text": nav_text("transition_place", language, place=building1),
                        "meta": {"building": building1}
                    })
                    commands.append({
                        "tag": "proceed_to_floor",
                        "text": nav_text("proceed_to_floor", language, floor=floor1, building=building1),
                        "meta": {"floor": floor1, "building": building1}
                    })

                elif floor0 != floor1:
                    if "staircase" in desc1:
                        commands.append({
                            "tag": "approaching_stair",
                            "text": nav_text("approaching_stair", language),
                            "meta": {}
                        })
                    elif "elevator" in desc1:
                        commands.append({
                            "tag": "approaching_elevator",
                            "text": nav_text("approaching_elevator", language),
                            "meta": {}
                        })
                    elif "escalator" in desc1:
                        commands.append({
                            "tag": "approaching_escalator",
                            "text": nav_text("approaching_escalator", language),
                            "meta": {}
                        })
                    else:
                        commands.append({
                            "tag": "proceed_to_floor",
                            "text": nav_text("proceed_to_floor", language, floor=floor1, building=building1),
                            "meta": {"floor": floor1, "building": building1}
                        })

                    try:
                        floor_num_0 = int(''.join(filter(str.isdigit, floor0)))
                        floor_num_1 = int(''.join(filter(str.isdigit, floor1)))
                        direction = "up" if floor_num_1 > floor_num_0 else "down"
                    except Exception:
                        direction = "up" if floor1 > floor0 else "down"

                    if "staircase" in desc1:
                        tag = "go_up_stair"
                    elif "elevator" in desc1:
                        tag = "go_up_elevator"
                    elif "escalator" in desc1:
                        tag = "go_up_escalator"
                    else:
                        tag = "proceed_to_floor"

                    commands.append({
                        "tag": tag,
                        "text": nav_text(tag, language, direction=direction, floor=floor1, building=building1),
                        "meta": {"direction": direction, "floor": floor1, "building": building1}
                    })

                heading = initial_heading
                i += 1
                continue

        # --- Turn detection ---
        bearing = math.degrees(math.atan2(dy, dx))
        turn = normalize_angle(bearing - heading)  # signed delta, left>0, right<0

        # Fine-grained sector classification
        qual, direction_word = classify_turn_sector(turn)
        hour = angle_to_clock_hour(turn)

        # A "turn event" happens on any bin except 'ahead'
        is_turn = (qual != 'ahead')

        if is_turn:
            # Flush any straight distance accumulated before turning
            if straight_distance > 0:
                dist_text, dist_val, dist_unit = convert_distance_meta(straight_distance * scale, unit, language)
                if door_events:
                    door_pos = min(door_events, key=lambda d: d["dist"])
                    door_text, door_val, door_unit = convert_distance_meta(door_pos["dist"] * scale, unit, language)
                    commands.append({
                        "tag": "forward_door",
                        "text": nav_text("forward_door", language, dist=dist_text, door_dist=door_text),
                        "meta": {
                            "distance": dist_val,
                            "unit": dist_unit,
                            "door_distance": door_val,
                            "door_unit": door_unit
                        }
                    })
                    door_events.clear()
                else:
                    commands.append({
                        "tag": "forward",
                        "text": nav_text("forward", language, dist=dist_text),
                        "meta": {"distance": dist_val, "unit": dist_unit}
                    })
                straight_distance = 0.0

            # Emit the appropriate turning command
            if qual == 'u_turn':
                commands.append({
                    "tag": "u_turn",
                    "text": nav_text("u_turn", language),
                    "meta": {}
                })
            else:
                # Note: make sure your nav_text('turn', ...) supports the new 'qual' values:
                # {'very_slight','slight','turn','sharp','very_sharp'}
                commands.append({
                    "tag": "turn",
                    "text": nav_text("turn", language, qual=qual, direction=direction_word, hour=hour),
                    "meta": {"qual": qual, "direction": direction_word, "hour": hour}
                })

            # Update heading after a turn
            heading = bearing

        # Accumulate straight distance along this segment
        straight_distance += segment_dist

        # --- Door detection ---
        if key0 != "VIRT" and hasattr(navigator, "pf_map"):
            pf = navigator.pf_map.get(key0[:3], None)
            if pf and hasattr(pf, "door_polygons"):
                line = LineString([p0, p1])
                for door_poly, _ in pf.door_polygons:
                    if line.crosses(door_poly):
                        proj_px = line.project(door_poly.centroid)
                        door_events.append({"dist": proj_px, "idx": i})
                        break

        # If approaching next turn or end, emit forward
        is_last = (i == len(coords) - 2)
        next_turn = False
        if not is_last:
            p2 = coords[i + 2]
            dx2, dy2 = p2[0] - p1[0], p1[1] - p2[1]
            bearing2 = math.degrees(math.atan2(dy2, dx2))
            next_turn = abs(normalize_angle(bearing2 - heading)) >= 25

        if is_last or next_turn:
            if straight_distance > 0:
                dist_text, dist_val, dist_unit = convert_distance_meta(straight_distance * scale, unit, language)
                if door_events:
                    door_pos = min(door_events, key=lambda d: d["dist"])
                    door_text, door_val, door_unit = convert_distance_meta(door_pos["dist"] * scale, unit, language)
                    commands.append({
                        "tag": "forward_door",
                        "text": nav_text("forward_door", language, dist=dist_text, door_dist=door_text),
                        "meta": {
                            "distance": dist_val,
                            "unit": dist_unit,
                            "door_distance": door_val,
                            "door_unit": door_unit
                        }
                    })
                    door_events.clear()
                else:
                    commands.append({
                        "tag": "forward",
                        "text": nav_text("forward", language, dist=dist_text),
                        "meta": {
                            "distance": dist_val,
                            "unit": dist_unit
                        }
                    })
                straight_distance = 0.0

        i += 1

    # --- Final arrival instruction (sector-based) ---
    final_label = labels[-1]

    # Optional orientation hint from description (fallback to current heading)
    desc = descriptions[-1].lower()
    desc_to_bearing = {'up': 90.0, 'right': 0.0, 'down': -90.0, 'left': 180.0}
    orientation_bearing = desc_to_bearing.get(desc, heading)

    # Relative angle from current heading to the desired arrival facing
    turn = normalize_angle(orientation_bearing - heading)

    # Sectorize using the same clock-face rules as turning
    qual, direction_word = classify_turn_sector(turn)        # returns e.g., ('slight', 'right') or ('ahead', 'right') or ('u_turn', 'left')
    hour = angle_to_clock_hour(turn)                         # 1..12

    # Emit arrival text; nav_text('arrive', ...) will localize qual/direction to dir_word
    commands.append({
        "tag": "arrive",
        "text": nav_text(
            "arrive",
            language,
            label=final_label,
            qual=qual,
            direction=direction_word,
            hour=hour
        ),
        "meta": {
            "label": final_label,
            "hour": hour,
            "qual": qual,
            "direction": direction_word
        }
    })

    return commands


def split_path_by_floor(
    path_keys: List[Union[str, Tuple[str, str, str, int]]],
    path_coords: List[Tuple[float, float]]
) -> Dict[Tuple[str, str, str], List[Tuple[float, float]]]:
    """
    Split a global path into floor-specific segments using unique (place, building, floor) tuple as key.

    Args:
        path_keys: List of node keys (may include "VIRT" as a string, others as (place, building, floor, node_id)).
        path_coords: List of coordinates.

    Returns:
        Dict mapping (place, building, floor) to list of coordinates on that floor.
    """
    floor_segs: Dict[Tuple[str, str, str], List[Tuple[float, float]]] = {}
    start_coord = None
    start_inserted = False

    for key, coord in zip(path_keys, path_coords):
        if key == "VIRT":
            start_coord = coord
            continue

        # key should be tuple (place, building, floor, node_id)
        floor_key = key[:3]  # (place, building, floor)

        if floor_key not in floor_segs:
            floor_segs[floor_key] = []
            if start_coord is not None and not start_inserted:
                floor_segs[floor_key].append(start_coord)
                start_inserted = True

        floor_segs[floor_key].append(coord)

    return floor_segs