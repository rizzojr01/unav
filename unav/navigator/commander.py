# navigation_commands.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Literal, Union, Optional

try:
    from shapely.geometry import LineString
    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False

from unav.navigator.nav_text import nav_text, unit_text


# -------------------------- Angle & distance helpers --------------------------

def normalize_angle(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def angle_to_clock_hour(turn_deg: float) -> int:
    raw = -turn_deg  # positive => right, negative => left
    clock_n = int(round(raw / 30.0)) % 12
    return 12 if clock_n == 0 else clock_n


def quantize_degrees_15(turn_deg: float) -> int:
    q = int(round(abs(turn_deg) / 15.0)) * 15
    return max(0, min(180, q))


def classify_turn_sector(turn_deg: float) -> Tuple[str, str]:
    raw = -turn_deg  # positive => right, negative => left
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


def convert_distance_meta(meters: float, unit: Literal["meter", "feet"], lang: str) -> Tuple[str, float, str]:
    if unit == "feet":
        value = meters * 3.28084
        localized = unit_text(value, "feet", lang)
        return localized, value, "feet"
    elif unit == "meter":
        localized = unit_text(meters, "meter", lang)
        return localized, meters, "meter"
    else:
        raise ValueError("Unit must be 'meter' or 'feet'.")


def _bearing_from_desc(desc: str, default: float) -> float:
    """
    将描述字符串里的方向词（up/right/down/left）转换为一个目标朝向角度。
    若无匹配则返回 default。
    """
    d = (desc or "").strip().lower()
    if "up" in d:
        return 90.0
    if "right" in d:
        return 0.0
    if "down" in d:
        return -90.0
    if "left" in d:
        return 180.0
    return default


# -------------------------- Label access (optional) --------------------------

class I18NLabels:
    """
    简单 labels 访问器，对应 <DATA_FINAL_ROOT>/_i18n/labels.json
    结构：
    {
      "places": {...},
      "buildings": {...},
      "floors": {...},
      "destinations": {...},
      "aliases": {...}
    }
    """
    def __init__(self, payload: Optional[Dict[str, Any]] = None):
        self.data = payload or {
            "places": {}, "buildings": {}, "floors": {}, "destinations": {}, "aliases": {}
        }

    @classmethod
    def load_from_root(cls, root: Union[str, Path]) -> "I18NLabels":
        p = Path(root) / "_i18n" / "labels.json"
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                raw = {}
        else:
            raw = {}
        data = {
            "places": raw.get("places", {}) if isinstance(raw.get("places", {}), dict) else {},
            "buildings": raw.get("buildings", {}) if isinstance(raw.get("buildings", {}), dict) else {},
            "floors": raw.get("floors", {}) if isinstance(raw.get("floors", {}), dict) else {},
            "destinations": raw.get("destinations", {}) if isinstance(raw.get("destinations", {}), dict) else {},
            "aliases": raw.get("aliases", {}) if isinstance(raw.get("aliases", {}), dict) else {},
        }
        return cls(data)

    def get(self, section: str, key: str, lang: str, fallback: str = "") -> str:
        entry = (self.data.get(section, {}) or {}).get(key, {}) or {}
        # 先取目标语言，再回退到英文，最后回退到 fallback
        return (entry.get(lang) or (entry.get("en") if lang != "en" else "") or fallback or "").strip()


def _ensure_labels(labels: Optional[I18NLabels],
                   data_final_root: Optional[Union[str, Path]]) -> I18NLabels:
    if labels:
        return labels
    if data_final_root:
        return I18NLabels.load_from_root(data_final_root)
    return I18NLabels()


def _label_entity(labels: I18NLabels, section: str, key: str, lang: str, fallback: str) -> str:
    return labels.get(section, key, lang, fallback)


# -------------------------- Forward flush & door detection --------------------------

def _flush_forward_if_needed(commands: List[Dict[str, Any]],
                             straight_px: float,
                             scale_m_per_px: float,
                             unit: Literal["meter", "feet"],
                             lang: str,
                             door_events: List[Dict[str, Any]]) -> float:
    """
    将累计的直行距离输出成 forward/forward_door 指令，并清空累计值和门事件。
    返回清零后的累计距离（通常为 0）。
    """
    if straight_px <= 0:
        return 0.0

    dist_m = straight_px * scale_m_per_px
    dist_text, dist_val, dist_unit = convert_distance_meta(dist_m, unit, lang)

    if door_events:
        door_first = min(door_events, key=lambda d: d["dist_px"])
        door_text, door_val, door_unit = convert_distance_meta(door_first["dist_px"] * scale_m_per_px, unit, lang)
        commands.append({
            "tag": "forward_door",
            "text": nav_text("forward_door", lang, dist=dist_text, door_dist=door_text),
            "meta": {"distance": dist_val, "unit": dist_unit, "door_distance": door_val, "door_unit": door_unit}
        })
        door_events.clear()
    else:
        commands.append({
            "tag": "forward",
            "text": nav_text("forward", lang, dist=dist_text),
            "meta": {"distance": dist_val, "unit": dist_unit}
        })

    return 0.0


def _append_door_event_if_any(navigator: Any,
                              key_tuple: Any,
                              p0: Tuple[float, float],
                              p1: Tuple[float, float],
                              i_idx: int) -> Optional[Dict[str, Any]]:
    """
    如果 shapely 可用且当前楼层含有 door 多边形，检测线段是否穿过一扇门。
    返回 {'dist_px': float, 'idx': i_idx} 或 None。
    """
    if not _HAS_SHAPELY:
        return None
    if not isinstance(key_tuple, tuple) or len(key_tuple) != 4:
        return None
    floor_key = key_tuple[:3]
    pf = getattr(navigator, "pf_map", {}).get(floor_key)
    if not pf or not hasattr(pf, "door_polygons"):
        return None

    seg = LineString([p0, p1])
    for door_poly, _ in pf.door_polygons:
        if seg.crosses(door_poly):
            proj_px = seg.project(door_poly.centroid)
            return {"dist_px": float(proj_px), "idx": i_idx}
    return None


# -------------------------- Main command generator --------------------------

def commands_from_result(
    navigator: Any,
    path_result: Dict[str, Any],
    initial_heading: float,
    unit: Literal["meter", "feet"] = "meter",
    language: str = "en",
    turn_mode: Literal["default", "deg15"] = "default",
    *,
    labels: Optional[I18NLabels] = None,
    data_final_root: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """
    生成逐步导航指令（无调试输出）。
    返回: List[{"tag","text","meta"}]
    """
    labels = _ensure_labels(labels, data_final_root)

    if "error" in path_result:
        raise ValueError(f"Cannot generate commands: {path_result['error']}")

    coords: List[Tuple[float, float]] = path_result.get("path_coords") or []
    keys: List[Union[str, Tuple[str, str, str, int]]] = path_result.get("path_keys") or []
    labels_seq: List[str] = path_result.get("path_labels") or []
    descriptions: List[str] = path_result.get("path_descriptions") or []

    commands: List[Dict[str, Any]] = []
    heading = float(initial_heading)

    # ---------- Start announcement ----------
    if len(keys) > 1 and keys[1] != "VIRT" and isinstance(keys[1], tuple) and len(keys[1]) == 4:
        floor_key = keys[1][:3]
        place, building, floor = floor_key
        pf0 = getattr(navigator, "pf_map", {}).get(floor_key)
        room = pf0.get_current_room(coords[0]) if (pf0 and hasattr(pf0, "get_current_room")) else ""

        place_name = _label_entity(labels, "places", place, language, place)
        building_name = _label_entity(labels, "buildings", f"{place}/{building}", language, building)
        floor_name = _label_entity(labels, "floors", f"{place}/{building}/{floor}", language, floor)

        commands.append({
            "tag": "start_in",
            "text": nav_text("start_in", language, room=room, floor=floor_name, building=building_name, place=place_name),
            "meta": {"room": room, "floor": floor, "building": building, "place": place}
        })
    else:
        commands.append({"tag": "start_nav", "text": nav_text("start_nav", language), "meta": {}})

    # ---------- Main loop ----------
    i = 0
    straight_px = 0.0
    door_events: List[Dict[str, Any]] = []

    while i < len(coords) - 1:
        key0 = keys[i]
        key1 = keys[i + 1]
        p0, p1 = coords[i], coords[i + 1]
        desc1 = str(descriptions[i + 1]).lower() if i + 1 < len(descriptions) else ""

        # scale（m/px）
        if isinstance(key1, tuple) and len(key1) == 4 and hasattr(navigator, "scales"):
            scale = float(getattr(navigator, "scales", {}).get(key1[:3], 1.0))
        else:
            scale = 1.0

        dx, dy = p1[0] - p0[0], p0[1] - p1[1]
        seg_len_px = math.hypot(dx, dy)

        # ---- 跨 place/building/floor 的处理 ----
        if (isinstance(key0, tuple) and len(key0) == 4 and isinstance(key1, tuple) and len(key1) == 4):
            place0, building0, floor0, _ = key0
            place1, building1, floor1, _ = key1

            if (place0, building0, floor0) != (place1, building1, floor1):
                # flush 累计直行
                straight_px = _flush_forward_if_needed(commands, straight_px, scale, unit, language, door_events)

                # place 变更
                if place0 != place1:
                    place_name = _label_entity(labels, "places", place1, language, place1)
                    commands.append({
                        "tag": "transition_place",
                        "text": nav_text("transition_place", language, place=place_name),
                        "meta": {"place": place1}
                    })
                    building_name = _label_entity(labels, "buildings", f"{place1}/{building1}", language, building1)
                    floor_name = _label_entity(labels, "floors", f"{place1}/{building1}/{floor1}", language, floor1)
                    commands.append({
                        "tag": "proceed_to",
                        "text": nav_text("proceed_to", language, floor=floor_name, building=building_name, place=place_name),
                        "meta": {"floor": floor1, "building": building1, "place": place1}
                    })

                # building 变更
                elif building0 != building1:
                    building_name = _label_entity(labels, "buildings", f"{place1}/{building1}", language, building1)
                    commands.append({
                        "tag": "transition_building",
                        "text": nav_text("transition_place", language, place=building_name),
                        "meta": {"building": building1}
                    })
                    floor_name = _label_entity(labels, "floors", f"{place1}/{building1}/{floor1}", language, floor1)
                    commands.append({
                        "tag": "proceed_to_floor",
                        "text": nav_text("proceed_to_floor", language, floor=floor_name, building=building_name),
                        "meta": {"floor": floor1, "building": building1}
                    })

                # floor 变更
                elif floor0 != floor1:
                    # 接近提示
                    if "stair" in desc1:
                        commands.append({"tag": "approaching_stair", "text": nav_text("approaching_stair", language), "meta": {}})
                    elif "elevator" in desc1:
                        commands.append({"tag": "approaching_elevator", "text": nav_text("approaching_elevator", language), "meta": {}})
                    elif "escalator" in desc1:
                        commands.append({"tag": "approaching_escalator", "text": nav_text("approaching_escalator", language), "meta": {}})
                    else:
                        building_name = _label_entity(labels, "buildings", f"{place1}/{building1}", language, building1)
                        floor_name = _label_entity(labels, "floors", f"{place1}/{building1}/{floor1}", language, floor1)
                        commands.append({
                            "tag": "proceed_to_floor",
                            "text": nav_text("proceed_to_floor", language, floor=floor_name, building=building_name),
                            "meta": {"floor": floor1, "building": building1}
                        })

                    # 上/下判断
                    try:
                        n0 = int("".join(filter(str.isdigit, floor0)))
                        n1 = int("".join(filter(str.isdigit, floor1)))
                        ud = "up" if n1 > n0 else "down"
                    except Exception:
                        ud = "up" if str(floor1) > str(floor0) else "down"

                    tag = (
                        "go_up_stair" if "stair" in desc1 else
                        ("go_up_elevator" if "elevator" in desc1 else
                         ("go_up_escalator" if "escalator" in desc1 else "proceed_to_floor"))
                    )
                    building_name = _label_entity(labels, "buildings", f"{place1}/{building1}", language, building1)
                    floor_name = _label_entity(labels, "floors", f"{place1}/{building1}/{floor1}", language, floor1)
                    commands.append({
                        "tag": tag,
                        "text": nav_text(tag, language, direction=ud, floor=floor_name, building=building_name),
                        "meta": {"direction": ud, "floor": floor1, "building": building1}
                    })

                # 重置朝向策略（重置为初始朝向）
                heading = float(initial_heading)
                i += 1
                continue

        # ---- 转向检测 ----
        bearing = math.degrees(math.atan2(dy, dx))
        turn = normalize_angle(bearing - heading)
        qual, direction_word = classify_turn_sector(turn)
        hour = angle_to_clock_hour(turn)
        deg15 = quantize_degrees_15(turn)

        is_turn_event = (qual != "ahead")
        if is_turn_event:
            # 先 flush 直行段
            straight_px = _flush_forward_if_needed(commands, straight_px, scale, unit, language, door_events)

            if turn_mode == "deg15":
                if qual == "u_turn":
                    commands.append({
                        "tag": "turn",
                        "text": nav_text("turn", language, qual="u_turn"),
                        "meta": {"qual": "u_turn", "direction": direction_word}
                    })
                else:
                    commands.append({
                        "tag": "turn",
                        "text": nav_text("turn_deg", language, direction=direction_word, deg=deg15),
                        "meta": {"qual": qual, "direction": direction_word, "hour": hour, "deg15": deg15}
                    })
            else:
                commands.append({
                    "tag": "turn",
                    "text": nav_text("turn", language, qual=qual, direction=direction_word, hour=hour),
                    "meta": {"qual": qual, "direction": direction_word, "hour": hour, "deg15": deg15}
                })

            # 更新朝向
            heading = bearing

        # 累加直行像素
        straight_px += seg_len_px

        # 门检测
        door_evt = _append_door_event_if_any(navigator, key0, p0, p1, i)
        if door_evt:
            door_events.append(door_evt)

        # 判断是否需要在下一步前 flush（下一个转角较大或即将结束）
        is_last = (i == len(coords) - 2)
        need_flush_now = False
        if not is_last:
            p2 = coords[i + 2]
            dx2, dy2 = p2[0] - p1[0], p1[1] - p2[1]
            bearing2 = math.degrees(math.atan2(dy2, dx2))
            next_turn = abs(normalize_angle(bearing2 - heading)) >= 25.0
            need_flush_now = next_turn
        else:
            need_flush_now = True  # 最后一段必须 flush

        if need_flush_now:
            straight_px = _flush_forward_if_needed(commands, straight_px, scale, unit, language, door_events)

        i += 1

    # ---------- Final arrival ----------
    final_label = labels_seq[-1] if labels_seq else ""
    orientation_bearing = _bearing_from_desc(descriptions[-1] if descriptions else "", default=heading)
    turn_final = normalize_angle(orientation_bearing - heading)
    qual_final, direction_final = classify_turn_sector(turn_final)
    hour_final = angle_to_clock_hour(turn_final)

    commands.append({
        "tag": "arrive",
        "text": nav_text("arrive", language, label=final_label, qual=qual_final, direction=direction_final, hour=hour_final),
        "meta": {"label": final_label, "hour": hour_final, "qual": qual_final, "direction": direction_final}
    })

    return commands


# -------------------------- Utility: split path by floor --------------------------

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

        if not isinstance(key, tuple) or len(key) != 4:
            # unexpected; put into a dummy bucket
            floor_key = ("", "", "")
        else:
            floor_key = key[:3]

        if floor_key not in floor_segs:
            floor_segs[floor_key] = []
            if start_coord is not None and not start_inserted:
                floor_segs[floor_key].append(start_coord)
                start_inserted = True

        floor_segs[floor_key].append(coord)

    return floor_segs
