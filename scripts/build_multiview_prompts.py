# scripts/build_multiview_prompts.py

def build_anchor_prompts(axis: dict):
    anchors = []

    if axis.get("eye_fox", 0) > 0.35:
        anchors.append(
            "strongly upturned almond-shaped eyes dominating the upper face"
        )

    if axis.get("eye_droopy", 0) > 0.35:
        anchors.append(
            "noticeably downturned sleepy eyes with heavy relaxed eyelids"
        )

    if axis.get("face_long", 0) > 0.4:
        anchors.append(
            "elongated vertical face with tall midface and narrow width"
        )

    if axis.get("face_round", 0) > 0.4:
        anchors.append(
            "rounded facial silhouette with soft cheek volume"
        )

    if axis.get("jaw_angular", 0) > 0.35:
        anchors.append(
            "sharp angular jawline creating a strong lower face structure"
        )

    return anchors


def build_multiview_prompts(schema: dict, axis: dict):
    anchors = build_anchor_prompts(axis)

    prompts = {
        "eye_focused": (
            f"A face with {schema['eye_shape']} eyes, "
            f"{schema['eye_spacing']} spacing, "
            f"and {schema['eye_height']} positioned eyes."
        ),

        "mouth_jaw_focused": (
            f"A face featuring a {schema['mouth_size']} mouth "
            f"and {schema['jaw_shape']} jaw shape."
        ),

        "proportion_focused": (
            f"A face with {schema['face_proportion']} facial proportions."
        ),

        "anchor_only": (
            " ".join(anchors) if anchors else
            "neutral balanced facial structure with no dominant extreme traits"
        )
    }

    return prompts
