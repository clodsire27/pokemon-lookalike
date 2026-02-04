def schema_to_clip_prompt(schema: dict) -> str:
    """
    Convert Face Archetype Schema (A~F) to fixed CLIP prompt (Template v1)
    """

    g = schema["global"]
    e = schema["eyes"]
    m = schema["mouth"]
    n = schema["nose_proxy"]
    a = schema["anchors"]
    o = schema["overall"]

    lines = []

    # -------------------------
    # 1. Global
    # -------------------------
    lines.append(
        f"Head: {g['head_shape']}, {g['aspect_ratio']}; "
        f"jaw {g['jaw_definition']}, chin {g['chin_shape']}; "
        f"midface {g['midface_fullness']}; "
        f"features {g['feature_density']}."
    )

    # -------------------------
    # 2. Eyes
    # -------------------------
    lines.append(
        f"Eyes: {e['eye_size']} {e['eye_shape']}, {e['eye_tilt']}; "
        f"spacing {e['eye_spacing']}, height {e['eye_height']}; "
        f"lids {e['upper_lid_weight']}; "
        f"iris {e['iris_dominance']}; "
        f"contrast {e['eye_contrast']}; "
        f"gaze {e['gaze_impression']}."
    )

    # -------------------------
    # 3. Mouth
    # -------------------------
    lines.append(
        f"Mouth: {m['mouth_size']}, {m['mouth_position']}; "
        f"curve {m['mouth_curve']}; "
        f"openness {m['mouth_openness']}; "
        f"lips {m['lip_definition']}; "
        f"smile {m['smile_signal']}."
    )

    # -------------------------
    # 4. Nose / snout proxy
    # -------------------------
    lines.append(
        f"Nose/snout: presence {n['nose_presence']}, "
        f"projection {n['midface_projection']}, "
        f"shape {n['snout_shape']}."
    )

    # -------------------------
    # 5. Anchors (2~3 forced)
    # -------------------------
    anchor_text = "; ".join(a)
    lines.append(f"Anchors: {anchor_text}.")

    # -------------------------
    # 6. Overall
    # -------------------------
    lines.append(
        f"Overall: tension {o['overall_tension']}, "
        f"ratio {o['overall_age_ratio']}, "
        f"style {o['overall_style']}."
    )

    return "\n".join(lines)
