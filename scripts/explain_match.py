# scripts/explain_match.py

from scripts.explain_utils import AXIS_EXPLANATION


def explain_match(
    human_axis: dict,
    pokemon_axis: dict,
    top_k: int = 3,
    min_contribution: float = 0.05
):
    """
    human_axis, pokemon_axis:
        - 이미 0~1 범위로 정규화된 axis dict
        - 예: {"eye_spacing": 0.7, "mouth": 0.9, ...}

    return:
        기여도가 높은 축 기준 설명 리스트
    """

    contributions = []

    for axis_key, axis_desc in AXIS_EXPLANATION.items():
        if axis_key not in human_axis or axis_key not in pokemon_axis:
            continue

        contrib = min(human_axis[axis_key], pokemon_axis[axis_key])
        if contrib < min_contribution:
            continue

        contributions.append({
            "type": axis_key,
            "description": axis_desc,
            "contribution": float(contrib)
        })

    contributions.sort(key=lambda x: -x["contribution"])
    return contributions[:top_k]
