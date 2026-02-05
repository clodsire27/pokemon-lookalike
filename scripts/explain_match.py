# scripts/explain_match.py

from scripts.explain_utils import AXIS_EXPLANATION


def explain_match(
    human_axis: dict,
    pokemon_axis: dict,
    top_k: int = 3,
    min_contribution: float = 0.05
):
    """
    human_axis, pokemon_axis: normalized axis dict
    return: list of explanation dicts
    """

    contributions = []

    for k, (etype, desc) in AXIS_EXPLANATION.items():
        if k not in human_axis or k not in pokemon_axis:
            continue

        contrib = min(human_axis[k], pokemon_axis[k])
        if contrib < min_contribution:
            continue

        contributions.append({
            "type": etype,
            "description": desc,
            "contribution": float(contrib)
        })

    contributions.sort(key=lambda x: -x["contribution"])
    return contributions[:top_k]
