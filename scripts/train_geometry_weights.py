import json
import torch
import numpy as np
from sklearn.linear_model import Ridge

# =========================
# Config
# =========================
POKEMON_GEO_DB = "data/pokemon_geometry_axis.json"
POKEMON_EMB_DB = "data/pokemon_multiview_embeddings.pt"
OUT_PATH = "data/geometry_weights.json"

# ✅ 살아있는 geometry 축만 사용
GEOMETRY_KEYS = [
    "eye_spacing_ratio",   # 핵심
    "face_aspect_ratio",   # 핵심
    "eye_height_ratio",    # 보조
]

# =========================
# Geometry feature function
# =========================
def geometry_features(g1, g2):
    """
    두 geometry axis 간 유사도 feature 벡터
    값 범위: [0, 1]
    """
    feats = []
    for k in GEOMETRY_KEYS:
        v = 1.0 - abs(g1[k] - g2[k])
        feats.append(max(0.0, v))
    return np.array(feats, dtype=np.float32)

# =========================
# Dataset builder
# =========================
def build_dataset():
    geo_db = json.load(open(POKEMON_GEO_DB, "r", encoding="utf-8"))
    emb_db = torch.load(POKEMON_EMB_DB, map_location="cpu")

    X, y = [], []
    names = list(geo_db.keys())

    for i, a in enumerate(names):
        if a not in emb_db:
            continue
        for b in names:
            if a == b or b not in emb_db:
                continue

            g1 = geo_db[a]
            g2 = geo_db[b]

            feat = geometry_features(g1, g2)
            clip_sim = float(torch.dot(
                emb_db[a], emb_db[b]
            ))

            X.append(feat)
            y.append(clip_sim)

    X = np.stack(X)
    y = np.array(y)

    return X, y

# =========================
# Train
# =========================
def main():
    X, y = build_dataset()

    print(f"[INFO] Training samples: {len(X)}")
    print(f"[INFO] Geometry dims: {X.shape[1]}")

    # 🔥 안정적인 ridge regression
    model = Ridge(alpha=1.0)
    model.fit(X, y)

    weights = model.coef_

    # -------------------------
    # Normalize weights (중요)
    # -------------------------
    weights = np.maximum(weights, 0.0)
    weights = weights / (weights.sum() + 1e-6)

    out = {
        "eye_spacing": float(weights[0]),
        "face_aspect": float(weights[1]),
        "eye_height":  float(weights[2]),
    }

    print("\n[LEARNED GEOMETRY WEIGHTS]")
    for k, v in out.items():
        print(f"{k:15s}: {v:.3f}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nSaved → {OUT_PATH}")

# =========================
# CLI
# =========================
if __name__ == "__main__":
    main()
