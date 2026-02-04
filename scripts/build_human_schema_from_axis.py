# scripts/build_human_schema_from_axis.py

def pick_ambiguous_combo(kmap: dict, threshold=0.45):
    items = sorted(kmap.items(), key=lambda x: -x[1])
    if not items:
        return "neutral"

    if len(items) == 1:
        return items[0][0]

    top, second = items[0], items[1]

    if top[1] < threshold:
        return f"ambiguous-{top[0]}-{second[0]}"
    return top[0]


def build_human_anchors(axis: dict):
    anchors = []

    if axis.get("eye_fox", 0) > 0.25:
        anchors.append(
            "strongly upturned narrow eyes giving a fox-like impression"
        )

    if axis.get("eye_droopy", 0) > 0.25:
        anchors.append(
            "downturned sleepy eyes creating a gentle look"
        )

    if axis.get("face_long", 0) > 0.30:
        anchors.append(
            "elongated vertical face shape"
        )

    if axis.get("face_round", 0) > 0.30:
        anchors.append(
            "rounded face with soft cheek volume"
        )

    if axis.get("vibe_cool", 0) > 0.30:
        anchors.append(
            "cool and calm facial impression"
        )

    if axis.get("vibe_cute", 0) > 0.30:
        anchors.append(
            "cute youthful facial impression"
        )

    return anchors


def build_human_schema_from_axis(axis: dict):
    g = axis.get  # shorthand

    schema = {}

    schema["eye_shape"] = pick_ambiguous_combo({
        "round": g("eye_round", 0),
        "narrow": g("eye_narrow", g("eye_thin", 0)),
        "droopy": g("eye_droopy", 0),
        "fox": g("eye_fox", 0),
    })

    schema["eye_spacing"] = pick_ambiguous_combo({
        "wide": g("eye_wide", 0),
        "close": g("eye_close", 0),
    })

    schema["eye_height"] = pick_ambiguous_combo({
        "high": g("eye_high", 0),
        "low": g("eye_low", 0),
    })

    schema["mouth_size"] = pick_ambiguous_combo({
        "small": g("mouth_small", 0),
        "large": g("mouth_large", 0),
    })

    schema["jaw_shape"] = pick_ambiguous_combo({
        "round": g("jaw_round", 0),
        "angular": g("jaw_angular", 0),
    })

    schema["face_proportion"] = pick_ambiguous_combo({
        "long": g("face_long", 0),
        "short": g("face_short", 0),
    })

    # 🔥 핵심: anchor를 schema에 실제로 넣는다
    schema["anchors"] = build_human_anchors(axis)

    return schema


# backward compatibility
build_schema_from_axis = build_human_schema_from_axis
