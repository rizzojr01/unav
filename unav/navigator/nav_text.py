# nav_text.py

"""
Navigation text templates and simple localization helpers.

- Callers pass semantic tokens (e.g., qual='very_slight', direction='left').
- nav_text() will localize these tokens for the 'turn' template automatically.
- Distance unit strings are provided by unit_text().
"""

NAV_TEXT = {
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
    "turn": {
        # NOTE: {qual} and {direction} are localized automatically in nav_text()
        "en": "{qual} {direction} to {hour} o'clock",
        "zh": "{hour}点方向{qual}{direction}转弯",
        "th": "{qual} เลี้ยว{direction} ไปทาง {hour} นาฬิกา"
    },
    "u_turn": {
        "en": "Make a U-turn (6 o'clock)",
        "zh": "掉头（6点方向）",
        "th": "กลับรถ (6 นาฬิกา)"
    },
    "arrive": {
        "en": "{label} on {hour} o'clock {dir_word}",
        "zh": "{label}在{hour}点方向{dir_word}",
        "th": "{label} ที่ {hour} นาฬิกา {dir_word}"
    }
}

UNIT_TEXT = {
    "meter": {
        "en": "{v} meters",
        "zh": "{v}米",
        "th": "{v} เมตร"
    },
    "meter_1": {
        "en": "1 meter",
        "zh": "1米",
        "th": "1 เมตร"
    },
    "feet": {
        "en": "{v} feet",
        "zh": "{v}英尺",
        "th": "{v} ฟุต"
    },
    "feet_1": {
        "en": "1 foot",
        "zh": "1英尺",
        "th": "1 ฟุต"
    }
}

# ---- New: localization maps for turn qualifiers and directions ----

QUAL_TEXT = {
    # Tokens expected from the planner: 'very_slight','slight','turn','sharp','very_sharp'
    "en": {
        "very_slight": "very slight",
        "slight": "slight",
        "turn": "turn",
        "sharp": "sharp",
        "very_sharp": "very sharp"
    },
    "zh": {
        "very_slight": "极小幅",
        "slight": "小幅",
        "turn": " ",
        "sharp": "急",
        "very_sharp": "极急"
    },
    "th": {
        "very_slight": "เอียงเล็กมาก",
        "slight": "เล็กน้อย",
        "turn": "เลี้ยว",
        "sharp": "หัก",
        "very_sharp": "หักมาก"
    }
}

DIRECTION_TEXT = {
    "en": {"left": "left", "right": "right"},
    "zh": {"left": "左", "right": "右"},
    "th": {"left": "ซ้าย", "right": "ขวา"}
}

# ---- New: localization maps for arrive directions ----

ARRIVE_DIR_TEXT = {
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

def _localize_turn_tokens(lang: str, kwargs: dict) -> dict:
    """
    If this is a 'turn' template, localize 'qual' and 'direction' tokens in-place.
    Callers provide tokens like qual='very_slight', direction='left'.
    """
    if "qual" in kwargs:
        qual_token = kwargs["qual"]
        qual_map = QUAL_TEXT.get(lang) or QUAL_TEXT["en"]
        kwargs["qual"] = qual_map.get(qual_token, str(qual_token)).strip()
    if "direction" in kwargs:
        dir_token = kwargs["direction"]
        dir_map = DIRECTION_TEXT.get(lang) or DIRECTION_TEXT["en"]
        kwargs["direction"] = dir_map.get(dir_token, str(dir_token))
    return kwargs


def nav_text(key: str, lang: str, **kwargs) -> str:
    tpl = NAV_TEXT.get(key, {})
    text = tpl.get(lang) or tpl.get("en") or ""

    if key == "turn":
        kwargs = _localize_turn_tokens(lang, kwargs)

    if key == "arrive":
        # Expect tokens: qual + direction OR precomputed dir_word
        if "qual" in kwargs and "direction" in kwargs:
            token = f"{kwargs['qual']}_{kwargs['direction']}"
            dir_map = ARRIVE_DIR_TEXT.get(lang) or ARRIVE_DIR_TEXT["en"]
            kwargs["dir_word"] = dir_map.get(token, dir_map.get("ahead"))
        elif "dir_word" not in kwargs:
            kwargs["dir_word"] = ARRIVE_DIR_TEXT.get(lang, {}).get("ahead", "ahead")

    return text.format(**kwargs)



def unit_text(value: float, unit: str, lang: str) -> str:
    """Get a localized distance string for value/unit/language."""
    v_int = int(round(value))
    if unit == "meter":
        if v_int == 1:
            return UNIT_TEXT["meter_1"][lang]
        else:
            return UNIT_TEXT["meter"][lang].format(v=v_int)
    elif unit == "feet":
        if v_int == 1:
            return UNIT_TEXT["feet_1"][lang]
        else:
            return UNIT_TEXT["feet"][lang].format(v=v_int)
    else:
        raise ValueError("Unit must be 'meter' or 'feet'")
