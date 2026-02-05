import os
import torch
import argparse
import open_clip
import json
import numpy as np
from PIL import Image

from scripts.extract_face_attributes import extract_face_crop
from scripts.encode_pokemon_multiview import encode_multiview_clip
from scripts.schema_to_multiview_prompts import schema_to_multiview_prompts
from scripts.extract_human_geometry_axis import extract_human_geometry_axis

# =========================
# Config
# =========================
POKEMON_EMB_DB = "data/pokemon_multiview_embeddings.pt"
POKEMON_GEO_DB = "data/pokemon_geometry_axis.json"
GEOMETRY_WEIGHT_DB = "data/geometry_weights.json"
HUMAN_LIKENESS_DB = "data/pokemon_human_likeness.json"

TOPK = 5
THRESHOLD = 0.25

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

FACE_TEXT = "cute cartoon character face with eyes and mouth"

# =========================
# Geometry statistics (z-score)
# =========================
STDS = {
    "eye_spacing_ratio": 0.169,
    "eye_height_ratio":  0.070,
    "mouth_width_ratio": 0.042,
    "face_aspect_ratio": 0.113,
}

# =========================
# Load weights
# =========================
with open(GEOMETRY_WEIGHT_DB, "r", encoding="utf-8") as f:
    WEIGHTS = json.load(f)

with open(HUMAN_LIKENESS_DB, "r", encoding="utf-8") as f:
    HUMAN_LIKENESS = json.load(f)

# =========================
# Geometry similarity
# =========================
def geometry_similarity(h, p):
    # --------------------------------------------------
    # 1️⃣ mouth_width soft gate (완화 버전)
    # --------------------------------------------------
    delta_mouth = abs(h["mouth_width_ratio"] - p["mouth_width_ratio"])

    if delta_mouth > 0.20:
        return 0.0
    elif delta_mouth > 0.12:
        mouth_penalty = np.exp(-5.0 * (delta_mouth - 0.12))
    else:
        mouth_penalty = 1.0

    # --------------------------------------------------
    # 2️⃣ weighted geometry similarity
    # --------------------------------------------------
    s = 0.0
    w = 0.0

    for key, w_key in [
        ("eye_spacing_ratio", "eye_spacing"),
        ("eye_height_ratio",  "eye_height"),
        ("face_aspect_ratio", "face_aspect"),
    ]:
        if w_key not in WEIGHTS:
            continue

        wk = WEIGHTS[w_key]

        # eye_height 과도 지배 방지
        if w_key == "eye_height":
            wk = min(wk, 0.4)

        if wk <= 0:
            continue

        z = abs(h[key] - p[key]) / STDS[key]

        # 🔥 감쇠 완화 (칼 → 저울)
        score = np.exp(-0.7 * z)
        # 대안: score = 1.0 / (1.0 + z)

        s += wk * score
        w += wk

    geo_sim = s / max(w, 1e-6)

    # --------------------------------------------------
    # 3️⃣ face_aspect 미세 감점 (soft gate)
    # --------------------------------------------------
    z_face = abs(
        h["face_aspect_ratio"] - p["face_aspect_ratio"]
    ) / STDS["face_aspect_ratio"]

    geo_sim *= np.exp(-0.3 * z_face)

    # --------------------------------------------------
    # 4️⃣ mouth penalty 적용
    # --------------------------------------------------
    geo_sim *= mouth_penalty

    return geo_sim


# =========================
# Final score
# =========================
def final_score(clip_sim, geo_sim):
    return 0.65 * geo_sim + 0.35 * clip_sim

# =========================
# Image embedding
# =========================
@torch.no_grad()
def encode_image_clip(model, preprocess, image_path, device):
    face_bgr = extract_face_crop(image_path)
    face_rgb = face_bgr[:, :, ::-1]
    img = Image.fromarray(face_rgb)
    x = preprocess(img).unsqueeze(0).to(device)
    e = model.encode_image(x)
    return e[0] / e.norm(dim=-1)

# ============================================================
# 🔥 Core function (API / server용)
# ============================================================
def run_match(image_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, PRETRAINED
    )
    model = model.to(device).eval()

    # Face concept embedding (ROI)
    with torch.no_grad():
        tokens = open_clip.tokenize([FACE_TEXT]).to(device)
        face_text_emb = model.encode_text(tokens)
        face_text_emb = face_text_emb / face_text_emb.norm(dim=-1, keepdim=True)
        face_text_emb = face_text_emb[0]

    # Load DBs
    pokemon_emb_db = torch.load(POKEMON_EMB_DB, map_location=device)
    with open(POKEMON_GEO_DB, "r", encoding="utf-8") as f:
        pokemon_geo_db = json.load(f)

    pokemon_emb_db.pop("pokemon", None)

    # Human geometry
    human_geo = extract_human_geometry_axis(
        image_path,
        clip_model=model,
        preprocess=preprocess,
        text_emb=face_text_emb,
        device=device
    )

    # Human CLIP embedding (semantic)
    schema = {
        "eye_shape": "neutral",
        "eye_spacing": "neutral",
        "eye_height": "neutral",
        "mouth_size": "neutral",
        "jaw_shape": "neutral",
        "face_proportion": "neutral",
    }

    prompts = schema_to_multiview_prompts(schema)
    text_emb_h = encode_multiview_clip(model, prompts)
    img_emb_h  = encode_image_clip(model, preprocess, image_path, device)

    human_emb = 0.5 * text_emb_h + 0.5 * img_emb_h
    human_emb = human_emb / human_emb.norm()

    # Matching
    results = []

    for name, p_emb in pokemon_emb_db.items():
        if name not in pokemon_geo_db:
            continue

        p_geo = pokemon_geo_db[name]

        clip_sim = float(torch.dot(human_emb, p_emb.to(device)))
        geo_sim  = geometry_similarity(human_geo, p_geo)
        score    = final_score(clip_sim, geo_sim)

        # human likeness gate
        human_like = HUMAN_LIKENESS.get(name, 0.0)
        if human_like < -0.05:
            score *= 0.6
        elif human_like < -0.02:
            score *= 0.85

        results.append({
            "name": name,
            "final": float(score),
            "geo": float(geo_sim),
            "clip": float(clip_sim),
            "human": float(human_like),
        })

    results.sort(key=lambda x: -x["final"])

    return {
        "image": os.path.basename(image_path),
        "results": results[:TOPK],
    }

# ============================================================
# CLI wrapper
# ============================================================
def cli_main(image_path: str):
    result = run_match(image_path)

    print("\n[RESULT]")
    for i, r in enumerate(result["results"], 1):
        if r["final"] < THRESHOLD:
            break

        print(
            f"{i}. {r['name']:<12} "
            f"final={r['final']:.3f}  "
            f"geo={r['geo']:.3f}  "
            f"clip={r['clip']:.3f}  "
            f"human={r['human']:+.3f}"
        )

# =========================
# Entry point
# =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    cli_main(args.image)
