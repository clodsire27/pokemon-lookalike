# scripts/explain_util.py

def build_geometry_explanation(h_geo, p_geo, geo_debug):
    """
    h_geo: human geometry dict
    p_geo: pokemon geometry dict
    geo_debug: geometry_similarity(return_debug=True) 결과
    """

    reasons = []
    bars = []

    # -------------------------
    # Eye spacing
    # -------------------------
    eye_spacing_score = geo_debug.get("eye_spacing", {}).get("score", 0.0)

    if eye_spacing_score > 0.85:
        reasons.append("눈 사이 간격이 매우 비슷함")

    bars.append({
        "key": "eye_spacing_ratio",
        "label": "눈 사이 거리",
        "human": h_geo["eye_spacing_ratio"],
        "pokemon": p_geo["eye_spacing_ratio"],
        "comment": (
            "매우 비슷함" if eye_spacing_score > 0.85 else "차이가 있음"
        )
    })

    # -------------------------
    # Eye height
    # -------------------------
    eye_height_score = geo_debug.get("eye_height", {}).get("score", 0.0)

    if eye_height_score > 0.80:
        reasons.append("눈 위치가 유사함")

    bars.append({
        "key": "eye_height_ratio",
        "label": "눈 위치",
        "human": h_geo["eye_height_ratio"],
        "pokemon": p_geo["eye_height_ratio"],
        "comment": (
            "허용 범위" if eye_height_score > 0.80 else "위치 차이 있음"
        )
    })

    # -------------------------
    # Mouth width
    # -------------------------
    mouth_penalty = geo_debug.get("mouth", {}).get("penalty", 0.0)

    if mouth_penalty > 0.95:
        reasons.append("입 크기가 거의 동일함")

    bars.append({
        "key": "mouth_width_ratio",
        "label": "입 크기",
        "human": h_geo["mouth_width_ratio"],
        "pokemon": p_geo["mouth_width_ratio"],
        "comment": (
            "거의 동일" if mouth_penalty > 0.95 else "차이가 있음"
        )
    })

    # -------------------------
    # Face aspect
    # -------------------------
    face_aspect_score = geo_debug.get("face_aspect", {}).get("score", 0.0)

    if face_aspect_score > 0.85:
        reasons.append("얼굴 비율이 비슷함")

    bars.append({
        "key": "face_aspect_ratio",
        "label": "얼굴 비율",
        "human": h_geo["face_aspect_ratio"],
        "pokemon": p_geo["face_aspect_ratio"],
        "comment": (
            "둥근 얼굴형" if face_aspect_score > 0.85 else "형태 차이 있음"
        )
    })

    return {
        "summary": reasons[:3],   # 상단 요약용 (최대 3개)
        "bars": bars              # 프론트 바 차트용
    }
