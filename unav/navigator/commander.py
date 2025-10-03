# navigation_commands.py
# -*- coding: utf-8 -*-
"""
UNav Navigation Commands (rewritten)
-----------------------------------

职责
- 将路径结果（coords/keys/labels/descriptions）转换为逐步导航指令（含本地化文本与元数据）。
- 优先使用 i18n labels（<DATA_FINAL_ROOT>/_i18n/labels.json）渲染 place/building/floor/destination 名称。
- 与 nav_text.py 对接（nav_text / unit_text）。
- 对“门”事件（door）进行可选检测：若可用 shapely 则在直行段内提示“前方多少米有门”。

兼容性
- 入口函数 `commands_from_result` 的参数保持易用：
    commands_from_result(
        navigator,
        path_result: Dict[str, Any],
        initial_heading: float,
        unit: Literal["meter", "feet"] = "meter",
        language: str = "en",
        turn_mode: Literal["default", "deg15"] = "default",
        labels: Optional[I18NLabels] = None,
        data_final_root: Optional[str] = None
    ) -> List[Dict[str, Any]]

  若未传 `labels`，但提供了 `data_final_root`，将自动实例化 I18NLabels；两者都没有则仅用回退值。

路径结果统一字段（由上游规范/适配）：
- path_coords: List[Tuple[x, y]]
- path_labels: List[str]
- path_keys:   List[Union["VIRT", (place, building, floor, node_id)]]
- path_descriptions: List[str]

输出元素结构：
- 每个指令是一个 dict：
  {
    "tag": "forward"|"turn"|...,
    "text": "<localized sentence>",
    "meta": {...}   # 元数据（单位拆分/角度/距离/方向等）
  }

作者：UNav Team
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple, Literal, Optional, Union

# shapely 可选
try:
    from shapely.geometry import LineString  # type: ignore
    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False

from unav.navigator.nav_text import nav_text, unit_text

# 可选的 i18n 标签读取器（若你已放到 unav/navigator/i18n_labels.py）
try:
    from unav.navigator.i18n_labels import I18NLabels  # type: ignore
except Exception:  # 允许没有该模块时依旧可用（仅使用英文/回退）
    I18NLabels = None  # type: ignore


# --------------------------
# 基础角度/距离工具
# --------------------------

def normalize_angle(angle: float) -> float:
    """归一化角度到 [-180, 180] 区间。"""
    return (angle + 180.0) % 360.0 - 180.0


def angle_to_clock_hour(turn_deg: float) -> int:
    """
    将“相对转角（左>0, 右<0）”映射到 12 小时钟刻度。
    我们取 raw=-turn_deg，这样原始正数=向右，方便右侧为正的直觉。
    """
    raw = -turn_deg
    clock_n = int(round(raw / 30.0)) % 12
    return 12 if clock_n == 0 else clock_n


def classify_turn_sector(turn_deg: float) -> Tuple[str, str]:
    """
    将相对转角分桶：
      |turn| <= 15      -> 'ahead'
      15~45             -> 'very_slight'
      45~75             -> 'slight'
      75~105            -> 'turn'
      105~135           -> 'sharp'
      135~165           -> 'very_sharp'
      >=165             -> 'u_turn'
    返回 (qual, direction)，direction∈{'left','right'}（ahead/u_turn 时可忽略）
    """
    raw = -turn_deg  # 右正左负
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
    """将 |角度| 量化到最接近的 15 度，范围 [0, 180]。"""
    q = int(round(abs(turn_deg) / 15.0)) * 15
    return max(0, min(180, q))


def convert_distance_meta(meters: float, unit: Literal["meter", "feet"], lang: str) -> Tuple[str, float, str]:
    """
    将米转换为本地化字符串，并返回 (文本, 数值, 单位字符串)。
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


# --------------------------
# i18n label 辅助
# --------------------------

def _ensure_labels(labels: Optional["I18NLabels"], data_final_root: Optional[str]) -> Optional["I18NLabels"]:
    """
    若未传 labels 且提供了 data_final_root，尝试实例化 I18NLabels。
    若 I18NLabels 不可用，返回 None（调用处将使用回退）。
    """
    if labels is not None:
        return labels
    if data_final_root and I18NLabels is not None:
        try:
            return I18NLabels(data_final_root, default_lang="en")
        except Exception:
            return None
    return None


def _label_entity(
    labels: Optional["I18NLabels"],
    section: str,
    key: str,
    lang: str,
    fallback: str
) -> str:
    """
    获取本地化标签：若 labels 不可用或不存在目标语言，则回退到 'en'，再回退到 fallback。
    """
    if labels is None:
        return fallback
    try:
        return labels.label(section, key, lang, fallback)
    except Exception:
        return fallback


# --------------------------
# 主流程：从路径产出指令
# --------------------------

def commands_from_result(
    navigator: Any,
    path_result: Dict[str, Any],
    initial_heading: float,
    unit: Literal["meter", "feet"] = "meter",
    language: str = "en",
    turn_mode: Literal["default", "deg15"] = "default",
    *,
    labels: Optional["I18NLabels"] = None,
    data_final_root: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    将路径结果转换为逐步导航指令（本地化）。

    Args:
        navigator: FacilityNavigator 实例（需含 pf_map / scales 等字段，若无也可工作但会少部分信息）。
        path_result: 同模块顶部说明。
        initial_heading: 起始朝向（度）。
        unit: "meter" 或 "feet"。
        language: 语言代码（可为 BCP47，如 "zh-Hant"）。
        turn_mode: "default"（钟点+程度）或 "deg15"（度数）。
        labels: 可选的 I18NLabels 实例；若缺省且提供 data_final_root，会自动实例化。
        data_final_root: 若 labels 未提供，用它初始化 I18NLabels。

    Returns:
        指令列表；若 path_result 有 "error"，本函数应由调用方在进入前处理。
    """
    # 保障 labels
    labels = _ensure_labels(labels, data_final_root)

    coords = path_result["path_coords"]
    keys = path_result["path_keys"]
    labels_seq = path_result["path_labels"]
    descriptions = path_result["path_descriptions"]

    commands: List[Dict[str, Any]] = []

    # ---------- 起始提示 ----------
    # 尝试本地化 place/building/floor/room
    if len(keys) > 1 and keys[1] != "VIRT" and isinstance(keys[1], tuple) and len(keys[1]) == 4:
        place, building, floor, _ = keys[1]
        place_name = _label_entity(labels, "places", place, language, place)
        b_key = f"{place}/{building}"
        f_key = f"{place}/{building}/{floor}"
        building_name = _label_entity(labels, "buildings", b_key, language, building)
        floor_name = _label_entity(labels, "floors", f_key, language, floor)

        room = ""
        pf = navigator.pf_map.get((place, building, floor)) if hasattr(navigator, "pf_map") else None
        if pf and hasattr(pf, "get_current_room") and coords:
            try:
                room_id = pf.get_current_room(coords[0])  # 取第一点
                # 若存在房间 id，可尝试目的地标签空间（destinations）中查找
                d_key = f"{place}/{building}/{floor}/{room_id}"
                room = _label_entity(labels, "destinations", d_key, language, str(room_id) if room_id else "")
            except Exception:
                room = ""
        commands.append({
            "tag": "start_in" if room else "start_nav",
            "text": nav_text("start_in" if room else "start_nav", language,
                             room=room, floor=floor_name, building=building_name, place=place_name),
            "meta": {"room": room, "floor": floor_name, "building": building_name, "place": place_name}
        })
    else:
        commands.append({"tag": "start_nav", "text": nav_text("start_nav", language), "meta": {}})

    # ---------- 主循环 ----------
    heading = initial_heading
    i = 0
    straight_distance = 0.0
    door_events: List[Dict[str, Any]] = []  # {"dist": float(px), "idx": i}

    # 帮助：取某个 key 对应的 scale
    def _scale_for_key(k: Union[str, Tuple[str, str, str, int]]) -> float:
        if isinstance(k, tuple) and len(k) == 4 and hasattr(navigator, "scales"):
            return navigator.scales.get(k[:3], 1.0)
        return 1.0

    while i < len(coords) - 1:
        key0, key1 = keys[i], keys[i + 1]
        p0, p1 = coords[i], coords[i + 1]
        desc1 = (descriptions[i + 1] if i + 1 < len(descriptions) else "") or ""
        desc1 = str(desc1).lower()

        dx, dy = p1[0] - p0[0], p0[1] - p1[1]
        segment_dist = math.hypot(dx, dy)
        scale = _scale_for_key(key1)

        # ---------- 跨 place/building/floor 切换 ----------
        if (isinstance(key0, tuple) and len(key0) == 4 and
                isinstance(key1, tuple) and len(key1) == 4):
            place0, building0, floor0, _ = key0
            place1, building1, floor1, _ = key1

            if (place0, building0, floor0) != (place1, building1, floor1):
                # flush 当前直行
                if straight_distance > 0:
                    _flush_forward(commands, straight_distance * scale, unit, language, door_events)
                    straight_distance = 0.0
                    door_events.clear()

                # 目标名称本地化
                place1_name = _label_entity(labels, "places", place1, language, place1)
                b1_key = f"{place1}/{building1}"
                f1_key = f"{place1}/{building1}/{floor1}"
                building1_name = _label_entity(labels, "buildings", b1_key, language, building1)
                floor1_name = _label_entity(labels, "floors", f1_key, language, floor1)

                # place 变化
                if place0 != place1:
                    commands.append({
                        "tag": "transition_place",
                        "text": nav_text("transition_place", language, place=place1_name),
                        "meta": {"place": place1_name}
                    })
                    commands.append({
                        "tag": "proceed_to",
                        "text": nav_text("proceed_to", language, floor=floor1_name, building=building1_name, place=place1_name),
                        "meta": {"floor": floor1_name, "building": building1_name, "place": place1_name}
                    })

                # building 变化（沿用 transition_place 的模板以保持你现有的 nav_text 兼容）
                elif building0 != building1:
                    commands.append({
                        "tag": "transition_building",
                        "text": nav_text("transition_place", language, place=building1_name),
                        "meta": {"building": building1_name}
                    })
                    commands.append({
                        "tag": "proceed_to_floor",
                        "text": nav_text("proceed_to_floor", language, floor=floor1_name, building=building1_name),
                        "meta": {"floor": floor1_name, "building": building1_name}
                    })

                # floor 变化
                elif floor0 != floor1:
                    if "staircase" in desc1:
                        commands.append({"tag": "approaching_stair", "text": nav_text("approaching_stair", language), "meta": {}})
                    elif "elevator" in desc1:
                        commands.append({"tag": "approaching_elevator", "text": nav_text("approaching_elevator", language), "meta": {}})
                    elif "escalator" in desc1:
                        commands.append({"tag": "approaching_escalator", "text": nav_text("approaching_escalator", language), "meta": {}})
                    else:
                        commands.append({
                            "tag": "proceed_to_floor",
                            "text": nav_text("proceed_to_floor", language, floor=floor1_name, building=building1_name),
                            "meta": {"floor": floor1_name, "building": building1_name}
                        })

                    # 上/下 判断（尽力而为）
                    direction_ud = _up_or_down(floor0, floor1)

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
                        "text": nav_text(tag, language, direction=direction_ud, floor=floor1_name, building=building1_name),
                        "meta": {"direction": direction_ud, "floor": floor1_name, "building": building1_name}
                    })

                # 切换后，重置朝向（可按策略调整）
                heading = initial_heading
                i += 1
                continue

        # ---------- 计算转向 ----------
        bearing = math.degrees(math.atan2(dy, dx))
        turn = normalize_angle(bearing - heading)  # 左>0，右<0
        qual, direction_word = classify_turn_sector(turn)
        hour = angle_to_clock_hour(turn)
        deg15 = quantize_degrees_15(turn)
        is_turn_event = (qual != "ahead")

        if is_turn_event:
            # flush 直行
            if straight_distance > 0:
                _flush_forward(commands, straight_distance * scale, unit, language, door_events)
                straight_distance = 0.0
                door_events.clear()

            # 发出转向
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
            heading = bearing

        # 累加直行
        straight_distance += math.hypot(dx, dy)

        # ---------- 门检测（可选） ----------
        if _HAS_SHAPELY and key0 != "VIRT" and hasattr(navigator, "pf_map"):
            pf = navigator.pf_map.get(key0[:3]) if isinstance(key0, tuple) else None
            if pf and hasattr(pf, "door_polygons") and pf.door_polygons:
                try:
                    line = LineString([p0, p1])
                    for door_poly, _ in pf.door_polygons:
                        if line.crosses(door_poly):
                            proj_px = line.project(door_poly.centroid)
                            door_events.append({"dist": proj_px, "idx": i})
                            break
                except Exception:
                    pass

        # ---------- 决定是否在下一个转向/结尾前 flush ----------
        is_last = (i == len(coords) - 2)
        next_turn = False
        if not is_last:
            p2 = coords[i + 2]
            dx2, dy2 = p2[0] - p1[0], p1[1] - p2[1]
            bearing2 = math.degrees(math.atan2(dy2, dx2))
            next_turn = abs(normalize_angle(bearing2 - heading)) >= 25.0

        if is_last or next_turn:
            if straight_distance > 0:
                _flush_forward(commands, straight_distance * scale, unit, language, door_events)
                straight_distance = 0.0
                door_events.clear()

        i += 1

    # ---------- 抵达 ----------
    final_label = labels_seq[-1] if labels_seq else ""
    last_key = keys[-1] if keys else "VIRT"
    if isinstance(last_key, tuple) and len(last_key) == 4:
        place, building, floor, did = last_key
        d_key = f"{place}/{building}/{floor}/{did}"
        final_label = _label_entity(labels, "destinations", d_key, language, final_label or str(did))
    elif isinstance(last_key, tuple) and len(last_key) == 3:
        place, building, floor = last_key
        f_key = f"{place}/{building}/{floor}"
        final_label = _label_entity(labels, "floors", f_key, language, final_label or floor)

    # 期望朝向（如果 descriptions[-1] 给了方向词，就用它，否则用当前 heading）
    orientation_bearing = _bearing_from_desc(descriptions[-1] if descriptions else "", default=heading)
    turn_final = normalize_angle(orientation_bearing - heading)
    qual_final, direction_final = classify_turn_sector(turn_final)
    hour_final = angle_to_clock_hour(turn_final)
    
    commands.append({
        "tag": "arrive",
        "text": nav_text("arrive", language, label=final_label, qual=qual_final, direction=direction_final, hour=hour_final),
        "meta": {"label": final_label, "hour": hour_final, "qual": qual_final, "direction": direction_final}
    })
    print(commands)
    print(language)
    return commands


# --------------------------
# 内部小工具
# --------------------------

def _flush_forward(
    out: List[Dict[str, Any]],
    meters: float,
    unit: Literal["meter", "feet"],
    lang: str,
    door_events: List[Dict[str, Any]]
) -> None:
    """
    把累计的直行距离吐出为一条（或带门提示的一条）指令。
    meters: 已乘以 scale 的真实米数。
    """
    dist_text, dist_val, dist_unit = convert_distance_meta(meters, unit, lang)
    if door_events:
        door_pos = min(door_events, key=lambda d: d["dist"])
        door_text, door_val, door_unit = convert_distance_meta(door_pos["dist"], unit, lang)
        out.append({
            "tag": "forward_door",
            "text": nav_text("forward_door", lang, dist=dist_text, door_dist=door_text),
            "meta": {
                "distance": dist_val, "unit": dist_unit,
                "door_distance": door_val, "door_unit": door_unit
            }
        })
    else:
        out.append({
            "tag": "forward",
            "text": nav_text("forward", lang, dist=dist_text),
            "meta": {"distance": dist_val, "unit": dist_unit}
        })


def _up_or_down(floor0: str, floor1: str) -> str:
    """
    根据楼层字符串估测上下方向。优先按数字比较，否则按字符串比较。
    """
    try:
        n0 = int("".join(filter(str.isdigit, floor0)) or "0")
        n1 = int("".join(filter(str.isdigit, floor1)) or "0")
        return "up" if n1 > n0 else "down"
    except Exception:
        return "up" if floor1 > floor0 else "down"


def _bearing_from_desc(desc: str, default: float) -> float:
    """
    将描述中的 up/right/down/left 映射到绝对朝向（度）；否则返回 default。
    """
    d = (desc or "").lower().strip()
    if "up" in d:
        return 90.0
    if "right" in d:
        return 0.0
    if "down" in d:
        return -90.0
    if "left" in d:
        return 180.0
    return default

def split_path_by_floor(
    path_keys: List[Union[str, Tuple[str, str, str, int]]],
    path_coords: List[Tuple[float, float]]
) -> Dict[Tuple[str, str, str], List[Tuple[float, float]]]:
    """
    Split a global path into floor-specific segments using (place, building, floor) as key.

    Args:
        path_keys: List of node keys (may include "VIRT", or (place, building, floor, node_id)).
        path_coords: List of coordinates.

    Returns:
        Dict mapping (place, building, floor) -> list of coordinates on that floor.
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
