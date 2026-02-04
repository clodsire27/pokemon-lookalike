# scripts/build_pokemon_schema_from_axis.py

def build_pokemon_schema_from_axis(axis: dict) -> dict:
    """
    Convert pokemon axis scores into face archetype schema (A~F).
    Pokemon-friendly defaults (mouth/nose simplified).
    """

    def pick(mapping, default):
        if not mapping:
            return default
        return max(mapping.items(), key=lambda x: x[1])[0]

    schema = {
        "global": {
            "head_shape": pick(
                {
                    "round": axis.get("face_round", 0),
                    "long": axis.get("face_long", 0),
                    "triangular": axis.get("face_sharp", 0),
                },
                "round",
            ),
            "aspect_ratio": "balanced",
            "jaw_definition": "soft",
            "chin_shape": "rounded",
            "midface_fullness": "medium",
            "feature_density": "balanced",
        },

        "eyes": {
            "eye_size": "large" if axis.get("eye_round", 0) > 0.35 else "medium",
            "eye_shape": pick(
                {
                    "round": axis.get("eye_round", 0),
                    "narrow": axis.get("eye_sharp", 0),
                    "almond": axis.get("eye_fox", 0),
                },
                "round",
            ),
            "eye_tilt": pick(
                {
                    "upturned": axis.get("eye_fox", 0),
                    "downturned": axis.get("eye_droopy", 0),
                    "neutral": 0.1,
                },
                "neutral",
            ),
            "eye_spacing": "wide",
            "eye_height": "mid-set",
            "upper_lid_weight": "light",
            "iris_dominance": "high",
            "eye_contrast": "low",
            "gaze_impression": pick(
                {
                    "sharp": axis.get("eye_sharp", 0),
                    "gentle": axis.get("eye_round", 0),
                },
                "gentle",
            ),
            "eye_signature": "stylized animated eyes",
        },

        "mouth": {
            "mouth_size": "small",
            "mouth_position": "low",
            "mouth_curve": "neutral",
            "mouth_openness": "closed",
            "lip_definition": "low",
            "smile_signal": "none",
            "mouth_signature": "minimal mouth detail",
        },

        "nose_proxy": {
            "nose_presence": "none",
            "midface_projection": "flat",
            "snout_shape": "none",
            "nostril_hint": "none",
        },

        # 🔥 겹침 방지 핵심 (2개 이상)
        "anchors": [
            "stylized anime-like eyes",
            "simplified facial features",
        ],

        "overall": {
            "overall_tension": pick(
                {
                    "tense": axis.get("eye_sharp", 0),
                    "relaxed": axis.get("vibe_soft", 0),
                    "neutral": 0.1,
                },
                "neutral",
            ),
            "overall_age_ratio": (
                "baby-like" if axis.get("vibe_cute", 0) > 0.35 else "balanced"
            ),
            "overall_style": pick(
                {
                    "soft-lines": axis.get("vibe_soft", 0),
                    "sharp-lines": axis.get("vibe_cool", 0),
                    "minimal-features": 0.1,
                },
                "soft-lines",
            ),
        },
    }

    return schema
