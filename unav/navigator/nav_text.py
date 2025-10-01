# nav_text.py
# -*- coding: utf-8 -*-

"""
Navigation text templates, localization helpers, and TTS-safe rendering.

Key points:
- All public renderers (nav_text, unit_text) return TTS-sanitized strings.
- "turn" supports clock/qual phrasing; "turn_deg" supports degree-based phrasing.
- "arrive" auto-composes a human-friendly direction word from (qual, direction).
"""

from __future__ import annotations

import re
from typing import Dict

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

NAV_TEXT: Dict[str, Dict[str, str]] = {
    "start_in": {
        "en": "You are currently in {room} on {floor} of {building}, {place}.",
        "zh": "您目前在{place}{building}{floor}的{room}。",
        "th": "ขณะนี้คุณอยู่ที่ {room} ชั้น {floor} อาคาร {building} ใน {place}"
    },
    "start_nav": {
        "en": "Starting navigation.",
        "zh": "开始导航。",
        "th": "เริ่มการนำทาง"
    },
    "forward": {
        "en": "Forward {dist}",
        "zh": "直行{dist}",
        "th": "เดินตรงไป {dist}"
    },
    "forward_door": {
        "en": "Forward {dist} and go through a door in {door_dist}",
        "zh": "直行{dist}，{door_dist}后穿过一扇门",
        "th": "เดินตรงไป {dist} แล้วผ่านประตูที่ {door_dist}"
    },
    "transition_place": {
        "en": "You are approaching the transition to {place}.",
        "zh": "您正接近{place}。",
        "th": "คุณกำลังจะถึง {place}"
    },
    "proceed_to": {
        "en": "Proceed to {floor} of {building} in {place}.",
        "zh": "前往{place}{building}{floor}。",
        "th": "ไปที่ชั้น {floor} อาคาร {building} ใน {place}"
    },
    "approaching_stair": {
        "en": "You are approaching the staircase.",
        "zh": "您正接近楼梯。",
        "th": "คุณกำลังจะถึงบันได"
    },
    "approaching_elevator": {
        "en": "You are approaching the elevator.",
        "zh": "您正接近电梯。",
        "th": "คุณกำลังจะถึงลิฟต์"
    },
    "approaching_escalator": {
        "en": "You are approaching the escalator.",
        "zh": "您正接近扶梯。",
        "th": "คุณกำลังจะถึงบันไดเลื่อน"
    },
    "go_up_stair": {
        "en": "Go {direction} to {floor} of {building} via the staircase.",
        "zh": "通过楼梯{direction}到{building}{floor}。",
        "th": "ใช้บันไดไป {direction} ถึงชั้น {floor} อาคาร {building}"
    },
    "go_up_elevator": {
        "en": "Press the {direction} button to {floor} of {building} using the elevator.",
        "zh": "乘电梯{direction}到{building}{floor}。",
        "th": "ใช้ลิฟต์ไป {direction} ถึงชั้น {floor} อาคาร {building}"
    },
    "go_up_escalator": {
        "en": "Take the escalator {direction} to {floor} of {building}.",
        "zh": "乘扶梯{direction}到{building}{floor}。",
        "th": "ใช้บันไดเลื่อนไป {direction} ถึงชั้น {floor} อาคาร {building}"
    },
    "proceed_to_floor": {
        "en": "Proceed to {floor} of {building}.",
        "zh": "前往{building}{floor}。",
        "th": "ไปที่ชั้น {floor} อาคาร {building}"
    },

    # Default turn: qual + direction + hour (U-turn handled specially)
    "turn": {
        "en": "{qual} {direction} to {hour} o'clock",
        "zh": "{hour}点方向{qual}{direction}转弯",
        "th": "{qual} เลี้ยว{direction} ไปทาง {hour} นาฬิกา"
    },

    # Degree-based turn: direction + N degrees
    "turn_deg": {
        "en": "Turn {direction} {deg} degrees",
        "zh": "向{direction}转{deg}度",
        "th": "เลี้ยว{direction} {deg} องศา"
    },

    # Dedicated short phrase for U-turn (used in both modes)
    "u_turn": {
        "en": "Make a U-turn",
        "zh": "掉头",
        "th": "กลับรถ"
    },

    "arrive": {
        "en": "{label} on {hour} o'clock {dir_word}",
        "zh": "{label}在{hour}点方向{dir_word}",
        "th": "{label} ที่ {hour} นาฬิกา {dir_word}"
    }
}

UNIT_TEXT: Dict[str, Dict[str, str]] = {
    "meter":   {"en": "{v} meters", "zh": "{v}米",   "th": "{v} เมตร"},
    "meter_1": {"en": "1 meter",    "zh": "1米",     "th": "1 เมตร"},
    "feet":    {"en": "{v} feet",   "zh": "{v}英尺", "th": "{v} ฟุต"},
    "feet_1":  {"en": "1 foot",     "zh": "1英尺",   "th": "1 ฟุต"}
}

QUAL_TEXT: Dict[str, Dict[str, str]] = {
    "en": {
        "very_slight": "very slight",
        "slight": "slight",
        "turn": "turn",
        "sharp": "sharp",
        "very_sharp": "very sharp",
        "u_turn": "U-turn"
    },
    "zh": {
        "very_slight": "极小幅",
        "slight": "小幅",
        "turn": " ",
        "sharp": "急",
        "very_sharp": "极急",
        "u_turn": "掉头"
    },
    "th": {
        "very_slight": "เอียงเล็กมาก",
        "slight": "เล็กน้อย",
        "turn": "เลี้ยว",
        "sharp": "หัก",
        "very_sharp": "หักมาก",
        "u_turn": "กลับรถ"
    }
}

DIRECTION_TEXT: Dict[str, Dict[str, str]] = {
    "en": {"left": "left", "right": "right"},
    "zh": {"left": "左", "right": "右"},
    "th": {"left": "ซ้าย", "right": "ขวา"}
}

ARRIVE_DIR_TEXT: Dict[str, Dict[str, str]] = {
    "en": {
        "ahead": "ahead",
        "behind": "behind",
        "very_slight_left": "very slight left",
        "slight_left": "slight left",
        "turn_left": "left",
        "sharp_left": "sharp left",
        "very_sharp_left": "very sharp left",
        "very_slight_right": "very slight right",
        "slight_right": "slight right",
        "turn_right": "right",
        "sharp_right": "sharp right",
        "very_sharp_right": "very sharp right",
        "u_turn": "behind"
    },
    "zh": {
        "ahead": "正前方",
        "behind": "正后方",
        "very_slight_left": "极小幅左侧",
        "slight_left": "小幅左侧",
        "turn_left": "左侧",
        "sharp_left": "急左侧",
        "very_sharp_left": "极急左侧",
        "very_slight_right": "极小幅右侧",
        "slight_right": "小幅右侧",
        "turn_right": "右侧",
        "sharp_right": "急右侧",
        "very_sharp_right": "极急右侧",
        "u_turn": "正后方"
    },
    "th": {
        "ahead": "ข้างหน้า",
        "behind": "ข้างหลัง",
        "very_slight_left": "เอียงซ้ายเล็กมาก",
        "slight_left": "ซ้ายน้อย",
        "turn_left": "ซ้าย",
        "sharp_left": "ซ้ายหัก",
        "very_sharp_left": "ซ้ายหักมาก",
        "very_slight_right": "เอียงขวาเล็กมาก",
        "slight_right": "ขวาน้อย",
        "turn_right": "ขวา",
        "sharp_right": "ขวาหัก",
        "very_sharp_right": "ขวาหักมาก",
        "u_turn": "ข้างหลัง"
    }
}

# ---------------------------------------------------------------------------
# TTS sanitization
# ---------------------------------------------------------------------------

# Replace runs of these symbols with a single space to avoid TTS reading them.
_SANITIZE_PATTERN = re.compile(r"[_+\@\#\$\%\^\&\*\=\~\`\|\:\\\/]+")
_MULTI_SPACE = re.compile(r"\s{2,}")

def _sanitize_for_tts(s: str) -> str:
    """
    Make a sentence safe for screen reader / TTS engines.

    Steps:
      1) Replace disallowed symbol runs with a single space.
      2) Collapse multiple spaces.
      3) Trim leading/trailing spaces.
      4) Remove spaces before common punctuation (English and Chinese).
    """
    if not s:
        return s
    s = _SANITIZE_PATTERN.sub(" ", s)
    s = _MULTI_SPACE.sub(" ", s).strip()
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\s+([，。！？；：、])", r"\1", s)
    return s

# ---------------------------------------------------------------------------
# Localization helpers
# ---------------------------------------------------------------------------

def _localize_turn_tokens(lang: str, kwargs: dict) -> dict:
    """Localize 'qual' and 'direction' for the 'turn' template."""
    if "qual" in kwargs:
        qual_map = QUAL_TEXT.get(lang) or QUAL_TEXT["en"]
        kwargs["qual"] = qual_map.get(kwargs["qual"], str(kwargs["qual"])).strip()
    if "direction" in kwargs:
        dir_map = DIRECTION_TEXT.get(lang) or DIRECTION_TEXT["en"]
        kwargs["direction"] = dir_map.get(kwargs["direction"], str(kwargs["direction"]))
    return kwargs

def _localize_direction_only(lang: str, kwargs: dict) -> dict:
    """Localize only the 'direction' token for the 'turn_deg' template."""
    if "direction" in kwargs:
        dir_map = DIRECTION_TEXT.get(lang) or DIRECTION_TEXT["en"]
        kwargs["direction"] = dir_map.get(kwargs["direction"], str(kwargs["direction"]))
    return kwargs

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def nav_text(key: str, lang: str, **kwargs) -> str:
    """
    Render a localized navigation sentence (TTS-safe).

    Special handling:
      - key == "turn" with qual == "u_turn" -> short U-turn phrase.
      - key == "turn_deg" -> localize direction and render '{direction} {deg} degrees'.
      - key == "arrive" -> compute 'dir_word' from qual + direction if not provided.
    """
    # Degree-based turn
    if key == "turn_deg":
        if kwargs.get("qual") == "u_turn":
            text = (NAV_TEXT["u_turn"].get(lang) or NAV_TEXT["u_turn"]["en"])
            return _sanitize_for_tts(text)
        kwargs = _localize_direction_only(lang, kwargs)
        text = (NAV_TEXT["turn_deg"].get(lang) or NAV_TEXT["turn_deg"]["en"]).format(**kwargs)
        return _sanitize_for_tts(text)

    # Default turn
    if key == "turn":
        if kwargs.get("qual") == "u_turn":
            text = (NAV_TEXT["u_turn"].get(lang) or NAV_TEXT["u_turn"]["en"])
            return _sanitize_for_tts(text)
        kwargs = _localize_turn_tokens(lang, kwargs)
        text = (NAV_TEXT["turn"].get(lang) or NAV_TEXT["turn"]["en"]).format(**kwargs)
        return _sanitize_for_tts(text)

    # Arrival
    if key == "arrive":
        if "qual" in kwargs and "direction" in kwargs:
            token = f"{kwargs['qual']}_{kwargs['direction']}"
            dir_map = ARRIVE_DIR_TEXT.get(lang) or ARRIVE_DIR_TEXT["en"]
            kwargs["dir_word"] = dir_map.get(token, dir_map.get("ahead"))
        elif "dir_word" not in kwargs:
            kwargs["dir_word"] = (ARRIVE_DIR_TEXT.get(lang) or ARRIVE_DIR_TEXT["en"]).get("ahead", "ahead")
        text = (NAV_TEXT["arrive"].get(lang) or NAV_TEXT["arrive"]["en"]).format(**kwargs)
        return _sanitize_for_tts(text)

    # Generic
    tpl = NAV_TEXT.get(key, {})
    text = (tpl.get(lang) or tpl.get("en") or "").format(**kwargs)
    return _sanitize_for_tts(text)


def unit_text(value: float, unit: str, lang: str) -> str:
    """
    Get a localized distance string for value/unit/language (TTS-safe).
    Rounds to the nearest integer for clean TTS/UI.
    """
    v_int = int(round(value))
    if unit == "meter":
        text = UNIT_TEXT["meter_1"][lang] if v_int == 1 else UNIT_TEXT["meter"][lang].format(v=v_int)
    elif unit == "feet":
        text = UNIT_TEXT["feet_1"][lang] if v_int == 1 else UNIT_TEXT["feet"][lang].format(v=v_int)
    else:
        raise ValueError("Unit must be 'meter' or 'feet'")
    return _sanitize_for_tts(text)
