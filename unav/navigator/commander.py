# navigation_commands.py
# -*- coding: utf-8 -*-

"""
Step-by-step navigation command generator with i18n labels and rich debug logs.

Key features:
- Localized text via nav_text()/unit_text() (from unav.navigator.nav_text).
- Optional I18NLabels integration to fetch human labels for place/building/floor/destination.
- Robust turn classification (qualitative + clock) and optional 15° quantized degree mode.
- Door announcement support (using shapely if available).
- Correct unit scaling (pixels -> meters) with per-floor scale from navigator.
- Detailed debug logging (toggle by debug=True).

Author: UNav Team
"""

from __future__ import annotations

import math
import logging
from typing import List, Dict, Any, Tuple, Literal, Union, Optional

from unav.navigator.nav_text import nav_text, unit_text

# Optional: I18NLabels helper (new in earlier refactor)
try:
    from unav.navigator.nav_text import I18NLabels  # type: ignore
except Exception:
    I18NLabels = None  # type: ignore

# Optional shapely door detection
try:
    from shapely.geometry import LineString  # type: ignore
    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False

logger = logging.getLogger("unav.navigation")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


# ------------------------------ Angle helpers ------------------------------ #

def normalize_angle(angle: float) -> float:
    """Normalize any angle to [-180, 180] degrees."""
    return (angle + 180.0) % 360.0 - 180.0


def angle_to_clock_hour(turn_deg: float) -> int:
    """
    Map a signed relative turn to a 12-hour clock index.

    Convention: turn = bearing - heading  (left > 0, right < 0).
    We invert sign so positive raw => right.
    """
    raw = -turn_deg
    clock_n = int(round(raw / 30.0)) % 12
    return 12 if clock_n == 0 else clock_n


def classify_turn_sector(turn_deg: float) -> Tuple[str, str]:
    """
    Classify the relative turn into sectors using clock-like ranges.
    Return: (qual, direction) where qual in
      {'ahead','very_slight','slight','turn','sharp','very_sharp','u_turn'},
      direction in {'left','right'} (ignored for 'ahead'/'u_turn').
    """
    raw = -turn_deg  # positive => right
    abs_raw = abs(raw)
    direction = "right" if raw > 0 else "left"

    if abs_raw <= 15.0:
        return "ahead", direction
    elif abs_raw <= 45.0:
        return "very_slight", direction
    elif abs_raw <= 75.0:
        return "slight", direction
    elif abs_raw <= 105.0:
        return "turn", direction
    elif abs_raw <= 135.0:
        return "sharp", direction
    elif abs_raw <= 165.0:
        return "very_sharp", direction
    else:
        return "u_turn", direction


def quantize_degrees_15(turn_deg: float) -> int:
    """Quantize |turn_deg| to nearest 15° multiple in [0, 180]."""
    q = int(round(abs(turn_deg) / 15.0)) * 15
    return max(0, min(180, q))


# ------------------------------ Units & i18n ------------------------------- #

def convert_distance_meta(meters: float, unit: Literal["meter", "feet"], lang: str) -> Tuple[str, float, str]:
    """
    Convert meters to localized text + numeric value + final unit.
    Return: (localized_str, numeric_value, 'meter'|'feet')
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


def _ensure_labels(labels: Optional["I18NLabels"], data_final_root: Optional[str], debug: bool=False) -> Optional["I18NLabels"]:
    """Create or reuse I18NLabels helper."""
    if labels is not None:
        if debug: logger.debug("Using provided I18NLabels instance")
        return labels
    if data_final_root and I18NLabels is not None:
        try:
            if debug: logger.debug("Creating I18NLabels from data_final_root=%s", data_final_root)
            return I18NLabels(data_final_root, default_lang="en")
        except Exception as e:
            logger.warning("I18NLabels init failed: %s", e)
            return None
    if debug: logger.debug("No I18NLabels and no data_final_root; fallbacks only")
    return None


def _label_entity(labels: Optional["I18NLabels"], section: str, key: str, lang: str, fallback: str, debug: bool=False) -> str:
    """
    Resolve label with fallback: target lang -> en -> fallback.
    """
    if labels is None:
        if debug: logger.debug("[label] %s/%s -> (no labels) fallback=%s", section, key, fallback)
        return fallback
    try:
        v = labels.label(section, key, lang, fallback)
        if debug:
            src = "target_or_en" if v != fallback else "fallback"
            logger.debug("[label] %s/%s lang=%s -> %s (%s)", section, key, lang, v, src)
        return v
    except Exception as e:
        if debug: logger.debug("[label] %s/%s exception=%s -> fallback=%s", section, key, e, fallback)
        return fallback


# ------------------------------ Door helpers ------------------------------- #

def _append_door_event(door_events: List[Dict[str, Any]], line: "LineString", door_poly: Any, scale: float, seg_idx: int, debug: bool=False) -> None:
    """
    Compute door distance along segment in METERS (project px * scale) and store it.
    """
    proj_px = line.project(door_poly.centroid)
    dist_m = proj_px * scale  # ⭐ convert to meters here
    door_events.append({"dist": dist_m, "idx": seg_idx})
    if debug:
        logger.debug("door@seg%d: proj=%.3f px -> %.3f m (scale=%.4f)", seg_idx, proj_px, dist_m, scale)


def _flush_forward(
    out: List[Dict[str, Any]],
    meters: float,
    unit: Literal["meter", "feet"],
    lang: str,
    door_events: List[Dict[str, Any]],
    debug: bool=False
) -> None:
    """
    Emit either 'forward' or 'forward_door' based on door_events (already meters).
    """
    dist_text, dist_val, dist_unit = convert_distance_meta(meters, unit, lang)
    if door_events:
        door_pos = min(door_events, key=lambda d: d["dist"])
        door_text, door_val, door_unit = convert_distance_meta(door_pos["dist"], unit, lang)
        if debug:
            logger.debug("flush forward_door: dist=%.3f %s door=%.3f %s", dist_val, dist_unit, door_val, door_unit)
        out.append({
            "tag": "forward_door",
            "text": nav_text("forward_door", lang, dist=dist_text, door_dist=door_text),
            "meta": {
                "distance": dist_val, "unit": dist_unit,
                "door_distance": door_val, "door_unit": door_unit
            }
        })
    else:
        if debug:
            logger.debug("flush forward: dist=%.3f %s", dist_val, dist_unit)
        out.append({
            "tag": "forward",
            "text": nav_text("forward", lang, dist=dist_text),
            "meta": {"distance": dist_val, "unit": dist_unit}
        })


# ------------------------------ Arrival helpers ---------------------------- #

def _bearing_from_desc(desc: str, default: float) -> float:
    """
    Parse a simple facing description to a bearing (deg). Fallback to `default`.
    Allowed tokens (case-insensitive): up/right/down/left.
    """
    if not desc:
        return default
    s = str(desc).strip().lower()
    if "up" in s:
        return 90.0
    if "right" in s:
        return 0.0
    if "down" in s:
        return -90.0
    if "left" in s:
        return 180.0
    return default


# ------------------------------ Main generator ----------------------------- #

def commands_from_result(
    navigator: Any,
    path_result: Dict[str, Any],
    initial_heading: float,
    unit: Literal["meter", "feet"] = "meter",
    language: str = "en",
    turn_mode: Literal["default", "deg15"] = "default",
    *,
    labels: Optional["I18NLabels"] = None,
    data_final_root: Optional[str] = None,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate step-by-step navigation instructions with semantic tags.
    Returns: list of {"tag","text","meta"}.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("=== commands_from_result BEGIN ===")
        logger.debug("lang=%s unit=%s turn_mode=%s", language, unit, turn_mode)

    labels = _ensure_labels(labels, data_final_root, debug=debug)

    if "error" in path_result:
        raise ValueError(f"Cannot generate commands: {path_result['error']}")

    coords = path_result["path_coords"]
    keys = path_result["path_keys"]
    labels_seq = path_result["path_labels"]
    descriptions = path_result["path_descriptions"]

    if debug:
        logger.debug("steps=%d (coords=%d)", max(0, len(coords)-1), len(coords))
        logger.debug("shapely=%s", _HAS_SHAPELY)

    commands: List[Dict[str, Any]] = []
    heading = float(initial_heading)

    # --- Start announcement ---
    if len(keys) > 1 and keys[1] != "VIRT" and isinstance(keys[1], tuple) and len(keys[1]) == 4:
        floor_key = keys[1][:3]
        place, building, floor = floor_key
        pf0 = navigator.pf_map.get(floor_key) if hasattr(navigator, "pf_map") else None
        room = pf0.get_current_room(coords[0]) if (pf0 and hasattr(pf0, "get_current_room")) else ""

        place_name = _label_entity(labels, "places", place, language, place, debug=debug)
        building_name = _label_entity(labels, "buildings", f"{place}/{building}", language, building, debug=debug)
        floor_name = _label_entity(labels, "floors", f"{place}/{building}/{floor}", language, floor, debug=debug)

        commands.append({
            "tag": "start_in",
            "text": nav_text("start_in", language, room=room, floor=floor_name, building=building_name, place=place_name),
            "meta": {"room": room, "floor": floor, "building": building, "place": place}
        })
    else:
        commands.append({"tag": "start_nav", "text": nav_text("start_nav", language), "meta": {}})

    i = 0
    straight_distance_px = 0.0  # accumulate in pixels then multiply by scale when flushing
    door_events: List[Dict[str, Any]] = []  # each item: {"dist": meters, "idx": seg_idx}

    # --- Main loop over path segments ---
    while i < len(coords) - 1:
        key0, key1 = keys[i], keys[i + 1]
        p0, p1 = coords[i], coords[i + 1]
        desc1 = str(path_result["path_descriptions"][i + 1]).lower() if i + 1 < len(path_result["path_descriptions"]) else ""

        # per-floor scale (fallback 1.0)
        if isinstance(key1, tuple) and len(key1) == 4 and hasattr(navigator, "scales"):
            scale = float(navigator.scales.get(key1[:3], 1.0))
        else:
            scale = 1.0

        dx, dy = p1[0] - p0[0], p0[1] - p1[1]
        seg_len_px = math.hypot(dx, dy)

        if debug:
            logger.debug("seg %d: key0=%s key1=%s scale=%.4f len_px=%.3f", i, str(key0), str(key1), scale, seg_len_px)

        # Transitions across place/building/floor
        if isinstance(key0, tuple) and isinstance(key1, tuple) and len(key0) == 4 and len(key1) == 4:
            place0, building0, floor0, _ = key0
            place1, building1, floor1, _ = key1

            if (place0, building0, floor0) != (place1, building1, floor1):
                # flush straight before transition (convert accumulated px to meters using *previous* scale)
                if straight_distance_px > 0:
                    _flush_forward(commands, straight_distance_px * scale, unit, language, door_events, debug=debug)
                    straight_distance_px = 0.0
                    door_events.clear()

                # place / building / floor transitions
                if place0 != place1:
                    place_name = _label_entity(labels, "places", place1, language, place1, debug=debug)
                    commands.append({"tag": "transition_place", "text": nav_text("transition_place", language, place=place_name), "meta": {"place": place1}})
                    bname = _label_entity(labels, "buildings", f"{place1}/{building1}", language, building1, debug=debug)
                    fname = _label_entity(labels, "floors", f"{place1}/{building1}/{floor1}", language, floor1, debug=debug)
                    commands.append({"tag": "proceed_to", "text": nav_text("proceed_to", language, floor=fname, building=bname, place=place_name), "meta": {"floor": floor1, "building": building1, "place": place1}})
                elif building0 != building1:
                    bname = _label_entity(labels, "buildings", f"{place1}/{building1}", language, building1, debug=debug)
                    commands.append({"tag": "transition_building", "text": nav_text("transition_place", language, place=bname), "meta": {"building": building1}})
                    fname = _label_entity(labels, "floors", f"{place1}/{building1}/{floor1}", language, floor1, debug=debug)
                    commands.append({"tag": "proceed_to_floor", "text": nav_text("proceed_to_floor", language, floor=fname, building=bname), "meta": {"floor": floor1, "building": building1}})
                elif floor0 != floor1:
                    # approach hint
                    if "staircase" in desc1:
                        commands.append({"tag": "approaching_stair", "text": nav_text("approaching_stair", language), "meta": {}})
                    elif "elevator" in desc1:
                        commands.append({"tag": "approaching_elevator", "text": nav_text("approaching_elevator", language), "meta": {}})
                    elif "escalator" in desc1:
                        commands.append({"tag": "approaching_escalator", "text": nav_text("approaching_escalator", language), "meta": {}})
                    # proceed
                    bname = _label_entity(labels, "buildings", f"{place1}/{building1}", language, building1, debug=debug)
                    fname = _label_entity(labels, "floors", f"{place1}/{building1}/{floor1}", language, floor1, debug=debug)
                    commands.append({"tag": "proceed_to_floor", "text": nav_text("proceed_to_floor", language, floor=fname, building=bname), "meta": {"floor": floor1, "building": building1}})

                # Reset heading after big transitions (policy choice)
                heading = float(initial_heading)
                i += 1
                continue

        # Turn detection on current segment
        bearing = math.degrees(math.atan2(dy, dx))
        turn = normalize_angle(bearing - heading)
        qual, direction_word = classify_turn_sector(turn)
        hour = angle_to_clock_hour(turn)
        deg15 = quantize_degrees_15(turn)
        is_turn_event = (qual != "ahead")

        if debug:
            logger.debug("turn: bearing=%.2f heading=%.2f turn=%.2f qual=%s dir=%s hour=%d deg15=%d",
                         bearing, heading, turn, qual, direction_word, hour, deg15)

        if is_turn_event:
            # flush forward before a turn (convert px to meters)
            if straight_distance_px > 0:
                _flush_forward(commands, straight_distance_px * scale, unit, language, door_events, debug=debug)
                straight_distance_px = 0.0
                door_events.clear()

            # Emit turn command
            if turn_mode == "deg15":
                if qual == "u_turn":
                    commands.append({"tag": "turn", "text": nav_text("turn", language, qual="u_turn"), "meta": {"qual": "u_turn", "direction": direction_word}})
                else:
                    commands.append({"tag": "turn", "text": nav_text("turn_deg", language, direction=direction_word, deg=deg15), "meta": {"qual": qual, "direction": direction_word, "hour": hour, "deg15": deg15}})
            else:
                commands.append({"tag": "turn", "text": nav_text("turn", language, qual=qual, direction=direction_word, hour=hour), "meta": {"qual": qual, "direction": direction_word, "hour": hour, "deg15": deg15}})

            # Update heading
            heading = bearing

        # Accumulate forward distance (in pixels)
        straight_distance_px += seg_len_px

        # Door detection (store meters)
        if _HAS_SHAPELY and key0 != "VIRT" and hasattr(navigator, "pf_map") and isinstance(key0, tuple):
            pf = navigator.pf_map.get(key0[:3], None)
            if pf and hasattr(pf, "door_polygons") and pf.door_polygons:
                try:
                    line = LineString([p0, p1])
                    for door_poly, _ in pf.door_polygons:
                        if line.crosses(door_poly):
                            _append_door_event(door_events, line, door_poly, scale, seg_idx=i, debug=debug)
                            break
                except Exception as e:
                    if debug:
                        logger.debug("door check exception: %s", e)

        # Decide whether to flush forward before next turn/end
        is_last = (i == len(coords) - 2)
        next_turn = False
        if not is_last:
            p2 = coords[i + 2]
            dx2, dy2 = p2[0] - p1[0], p1[1] - p2[1]
            bearing2 = math.degrees(math.atan2(dy2, dx2))
            next_turn = abs(normalize_angle(bearing2 - heading)) >= 25.0

        if is_last or next_turn:
            if straight_distance_px > 0:
                _flush_forward(commands, straight_distance_px * scale, unit, language, door_events, debug=debug)
                straight_distance_px = 0.0
                door_events.clear()

        i += 1

    # --- Final arrival instruction ---
    final_label_raw = labels_seq[-1] if labels_seq else ""
    if debug:
        logger.debug("arrive: final_label_raw=%s desc_last=%s", final_label_raw, (descriptions[-1] if descriptions else ""))

    # If final label is a destination ID, you may enhance it via I18NLabels if你在调用侧传入dest key
    # 这里保留调用者的 label 列表值
    final_label = final_label_raw

    orientation_bearing = _bearing_from_desc(descriptions[-1] if descriptions else "", default=heading)
    turn_final = normalize_angle(orientation_bearing - heading)
    qual_final, direction_final = classify_turn_sector(turn_final)
    hour_final = angle_to_clock_hour(turn_final)

    if debug:
        logger.debug("arrive: heading=%.2f orient=%.2f turn=%.2f qual=%s dir=%s hour=%d",
                     heading, orientation_bearing, turn_final, qual_final, direction_final, hour_final)

    commands.append({
        "tag": "arrive",
        "text": nav_text("arrive", language, label=final_label, qual=qual_final, direction=direction_final, hour=hour_final),
        "meta": {"label": final_label, "hour": hour_final, "qual": qual_final, "direction": direction_final}
    })

    if debug:
        logger.debug("=== commands_from_result END (%d cmds) ===", len(commands))

    return commands


# ------------------------------ Utilities ---------------------------------- #

def split_path_by_floor(
    path_keys: List[Union[str, Tuple[str, str, str, int]]],
    path_coords: List[Tuple[float, float]]
) -> Dict[Tuple[str, str, str], List[Tuple[float, float]]]:
    """
    Split a global path into floor-specific segments using (place, building, floor) as key.
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
