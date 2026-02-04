# scripts/schema_to_multiview_prompts.py

def schema_to_multiview_prompts(schema: dict):
    prompts = {}

    # =========================
    # Case 1: FLAT schema (Human)
    # =========================
    if "eye_shape" in schema:
        prompts["eye_focused"] = (
            f"A face with {schema.get('eye_shape', 'neutral')} eyes, "
            f"{schema.get('eye_spacing', 'neutral')} spacing, "
            f"and {schema.get('eye_height', 'neutral')} positioned eyes."
        )

        prompts["mouth_jaw_focused"] = (
            f"A face featuring a {schema.get('mouth_size', 'neutral')} mouth "
            f"and {schema.get('jaw_shape', 'neutral')} jaw shape."
        )

        prompts["proportion_focused"] = (
            f"A face with {schema.get('face_proportion', 'balanced')} facial proportions."
        )

        prompts["anchor_only"] = (
            "neutral balanced facial structure with subtle distinguishing traits"
        )

        return prompts

    # =========================
    # Case 2: RICH schema (Pokémon)
    # =========================
    g = schema.get("global", {})
    e = schema.get("eyes", {})
    o = schema.get("overall", {})
    anchors = schema.get("anchors", [])

    prompts["eye_focused"] = (
        f"{e.get('eye_size','medium')} {e.get('eye_shape','round')} eyes, "
        f"{e.get('eye_tilt','neutral')} tilt, "
        f"{e.get('gaze_impression','gentle')} gaze"
    )

    prompts["proportion_focused"] = (
        f"{g.get('head_shape','round')} head shape, "
        f"{g.get('aspect_ratio','balanced')} proportions"
    )

    prompts["anchor_only"] = (
        ", ".join(anchors)
        if anchors else
        "stylized animated creature face"
    )

    prompts["overall_vibe"] = (
        f"{o.get('overall_style','soft')} style, "
        f"{o.get('overall_tension','neutral')} tension"
    )

    return prompts
