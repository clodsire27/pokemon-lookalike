# scripts/schema_to_multiview_prompts.py

def _pick_for_clip(value: str) -> str:
    """
    CLIP 입력용 값 정제:
    - ambiguous-fox-round → fox
    - ambiguous-long-short → long
    - neutral / None → neutral
    """
    if not value:
        return "neutral"

    if value.startswith("ambiguous-"):
        parts = value.split("-")
        if len(parts) >= 2:
            return parts[1]   # 가장 강한 첫 번째 축
        return "neutral"

    return value


def schema_to_multiview_prompts(schema: dict):
    prompts = {}

    # =========================
    # Case 1: FLAT schema (Human)
    # =========================
    if "eye_shape" in schema:
        eye_shape = _pick_for_clip(schema.get("eye_shape"))
        eye_spacing = _pick_for_clip(schema.get("eye_spacing"))
        eye_height = _pick_for_clip(schema.get("eye_height"))

        mouth_size = _pick_for_clip(schema.get("mouth_size"))
        jaw_shape = _pick_for_clip(schema.get("jaw_shape"))

        face_prop = _pick_for_clip(schema.get("face_proportion"))

        prompts["eye_focused"] = (
            f"A face with {eye_shape} eyes, "
            f"{eye_spacing} spacing, "
            f"and {eye_height} positioned eyes."
        )

        prompts["mouth_jaw_focused"] = (
            f"A face featuring a {mouth_size} mouth "
            f"and {jaw_shape} jaw shape."
        )

        prompts["proportion_focused"] = (
            f"A face with {face_prop} facial proportions."
        )

        personality = schema.get("personality", {})

        expression_parts = []

        if personality.get("smile", 0.0) > 0.35:
            expression_parts.append("a broad bright smile")

        if personality.get("mouth_open", 0.0) > 0.35:
            expression_parts.append("an open-mouth smile")

        if personality.get("cheerful", 0.0) > 0.35:
            expression_parts.append("a cheerful happy expression")

        if personality.get("playful", 0.0) > 0.35:
            expression_parts.append("a playful mischievous expression")

        if personality.get("energetic", 0.0) > 0.35:
            expression_parts.append("a lively energetic impression")

        if personality.get("cute", 0.0) > 0.35:
            expression_parts.append("a cute youthful impression")

        prompts["anchor_only"] = (
            f"{eye_shape} eyes, {face_prop} face, "
            "distinct facial impression"
        )

        if expression_parts:
            prompts["expression_focused"] = (
                "A face showing "
                + ", ".join(expression_parts)
                + "."
            )

        return prompts

    # =========================
    # Case 2: RICH schema (Pokémon)
    # =========================
    g = schema.get("global", {})
    e = schema.get("eyes", {})
    o = schema.get("overall", {})
    anchors = schema.get("anchors", [])

    eye_shape = _pick_for_clip(e.get("eye_shape"))
    eye_tilt = _pick_for_clip(e.get("eye_tilt"))
    gaze = _pick_for_clip(e.get("gaze_impression"))

    head_shape = _pick_for_clip(g.get("head_shape"))
    aspect = _pick_for_clip(g.get("aspect_ratio"))

    style = _pick_for_clip(o.get("overall_style"))
    tension = _pick_for_clip(o.get("overall_tension"))

    prompts["eye_focused"] = (
        f"{eye_shape} eyes with {eye_tilt} tilt, "
        f"{gaze} gaze"
    )

    prompts["proportion_focused"] = (
        f"{head_shape} head shape, "
        f"{aspect} facial proportions"
    )

    # 🔥 anchor는 그대로 사용 (이미 확정 표현)
    prompts["anchor_only"] = (
        ", ".join(anchors)
        if anchors else
        "stylized animated creature face"
    )

    prompts["overall_vibe"] = (
        f"{style} style, "
        f"{tension} facial tension"
    )

    return prompts
