# scripts/match_human_to_pokemon_archetype.py
import json
import argparse
from pathlib import Path

from scripts.axis_utils import compress_axis_soft, filter_axis
from scripts.extract_human_axis_clip import extract_human_axis

# =========================
# Config
# =========================
ARCHETYPE_PATH = "data/pokemon_archetypes.json"
TOPK = 5

CORE_KEYS = [
    "eye_fox",
    "eye_sharp", "eye_round", "eye_droopy",
    "vibe_cool", "vibe_cute", "vibe_soft", "vibe_elegant",
]

FACE_KEYS = ["face_round", "face_long", "face_sharp"]
KEEP_KEYS = CORE_KEYS + FACE_KEYS

AXIS_WEIGHT = {
    "eye_fox": 1.5,

    "eye_sharp": 1.3,
    "eye_round": 1.3,
    "eye_droopy": 1.2,

    "vibe_cool": 1.4,
    "vibe_elegant": 1.3,
    "vibe_cute": 1.3,
    "vibe_soft": 1.2,
}

FACE_PENALTY_LAMBDA = 0.35

# =========================
# Scoring
# =========================
def axis_score(h, p):
    score = 0.0
    reason = []

    for k in CORE_KEYS:
        if k in h and k in p:
            w = AXIS_WEIGHT.get(k, 1.0)
            contrib = min(h[k], p[k]) * w
            if contrib > 0.015:   # 🔧 낮춤
                score += contrib
                reason.append((k, float(contrib)))

    reason.sort(key=lambda x: -x[1])
    return score, reason


def face_penalty(h, p):
    diff = 0.0
    for k in FACE_KEYS:
        diff += abs(h.get(k, 0.0) - p.get(k, 0.0))
    return -FACE_PENALTY_LAMBDA * diff


def fallback_score(h, p):
    """
    강한 축이 없을 때 쓰는 '완화 점수'
    (여기서는 순수 intersection만 계산)
    """
    score = 0.0
    for k in CORE_KEYS:
        score += min(h.get(k, 0.0), p.get(k, 0.0))
    return score


# =========================
# Explanation
# =========================
LABEL = {
    "eye_fox": "눈꼬리가 올라간 차분한 여우형 눈매",
    "eye_sharp": "강한 카리스마가 느껴지는 눈매",
    "eye_round": "둥글고 또렷한 눈매",
    "eye_droopy": "부드럽게 처진 눈매",

    "vibe_cool": "차분하고 쿨한 분위기",
    "vibe_cute": "귀엽고 동안 느낌",
    "vibe_soft": "부드럽고 순한 인상",
    "vibe_elegant": "세련되고 우아한 분위기",
}

def explain(reason):
    if not reason:
        return "특정 동물상으로 강하게 치우치지 않은 중립형 인상"

    top = reason[:3]
    return " + ".join(f"{LABEL.get(k,k)}({v:.2f})" for k,v in top)


# =========================
# Normalize
# =========================
def normalize_axis(axis):
    axis = compress_axis_soft(axis, temperature=0.02)
    axis = filter_axis(axis, KEEP_KEYS)
    return axis


# =========================
# Main
# =========================
def main(image_path: str):
    if not Path(ARCHETYPE_PATH).exists():
        raise FileNotFoundError("pokemon_archetypes.json 먼저 생성")

    with open(ARCHETYPE_PATH, "r", encoding="utf-8") as f:
        pokemon_db = json.load(f)

    human_axis = normalize_axis(extract_human_axis(image_path))

    scored = []
    for name, p_raw in pokemon_db.items():
        p_axis = normalize_axis(p_raw)

        base, reason = axis_score(human_axis, p_axis)
        pen = face_penalty(human_axis, p_axis)

        if base < 0.05:
            fb = fallback_score(human_axis, p_axis)
            final = fb * 0.3 + pen * 0.1   # 🔥 fallback 전용 완화
            reason = []
        else:
            final = base + pen

        scored.append((name, final, base, pen, reason))

    scored.sort(key=lambda x: -x[1])

    print("\n[RESULT]")
    for i,(n,f,b,p,r) in enumerate(scored[:TOPK],1):
        print(f"{i}. {n:<12} score={f:.3f} (base={b:.3f}, face_pen={p:.3f})  이유: {explain(r)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    main(args.image)
