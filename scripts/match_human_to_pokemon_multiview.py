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
from scripts.explain_utils import build_geometry_explanation


# =========================
# Config
# =========================
POKEMON_EMB_DB = "data/pokemon_multiview_embeddings.pt"
POKEMON_GEO_DB = "data/pokemon_geometry_axis.json"
GEOMETRY_WEIGHT_DB = "data/geometry_weights.json"
HUMAN_LIKENESS_DB = "data/pokemon_human_likeness.json"

TOPK = 3
THRESHOLD = 0.25

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
FACE_TEXT = "cute cartoon character face with eyes and mouth"


# =========================
# Geometry statistics
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


# ============================================================
# Geometry similarity + debug
# ============================================================
def geometry_similarity(h, p, return_debug=False):
    debug = {}

    # --------------------------------------------------
    # 1️⃣ mouth width soft gate
    # --------------------------------------------------
    delta_mouth = abs(h["mouth_width_ratio"] - p["mouth_width_ratio"])

    if delta_mouth > 0.20:
        if return_debug:
            return 0.0, {
                "mouth": {"delta": delta_mouth, "penalty": 0.0}
            }
        return 0.0

    if delta_mouth > 0.12:
        mouth_penalty = np.exp(-5.0 * (delta_mouth - 0.12))
    else:
        mouth_penalty = 1.0

    debug["mouth"] = {
        "delta": delta_mouth,
        "penalty": mouth_penalty
    }

    # --------------------------------------------------
    # 2️⃣ weighted geometry axes
    # --------------------------------------------------
    s, w = 0.0, 0.0

    for key, w_key in [
        ("eye_spacing_ratio", "eye_spacing"),
        ("eye_height_ratio",  "eye_height"),
        ("face_aspect_ratio", "face_aspect"),
    ]:
        wk = WEIGHTS.get(w_key, 0.0)

        # eye_height 과도 지배 방지
        if w_key == "eye_height":
            wk = min(wk, 0.4)

        if wk <= 0:
            continue

        z = abs(h[key] - p[key]) / STDS[key]
        score = np.exp(-0.7 * z)

        debug[w_key] = {
            "z": float(z),
            "score": float(score)
        }

        s += wk * score
        w += wk

    geo_sim = s / max(w, 1e-6)

    # --------------------------------------------------
    # 3️⃣ face aspect soft penalty
    # --------------------------------------------------
    z_face = abs(
        h["face_aspect_ratio"] - p["face_aspect_ratio"]
    ) / STDS["face_aspect_ratio"]

    face_penalty = np.exp(-0.3 * z_face)
    geo_sim *= face_penalty

    debug["face_aspect"] = {
        "z": float(z_face),
        "score": float(face_penalty)
    }

    # --------------------------------------------------
    # 4️⃣ mouth penalty 적용
    # --------------------------------------------------
    geo_sim *= mouth_penalty

    if return_debug:
        return geo_sim, debug

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
# 🔥 Core function (API / CLI 공용)
# ============================================================
def run_match(image_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # Load CLIP
    # -------------------------
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, PRETRAINED
    )
    model = model.to(device).eval()

    # face concept embedding (ROI용)
    with torch.no_grad():
        tokens = open_clip.tokenize([FACE_TEXT]).to(device)
        face_text_emb = model.encode_text(tokens)
        face_text_emb = face_text_emb / face_text_emb.norm(dim=-1, keepdim=True)
        face_text_emb = face_text_emb[0]

    # -------------------------
    # Load DBs
    # -------------------------
    pokemon_emb_db = torch.load(POKEMON_EMB_DB, map_location=device)
    with open(POKEMON_GEO_DB, "r", encoding="utf-8") as f:
        pokemon_geo_db = json.load(f)

    pokemon_emb_db.pop("pokemon", None)

    # -------------------------
    # Human geometry
    # -------------------------
    human_geo = extract_human_geometry_axis(
        image_path,
        clip_model=model,
        preprocess=preprocess,
        text_emb=face_text_emb,
        device=device
    )

    # -------------------------
    # Human CLIP embedding (semantic)
    # -------------------------
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

    # -------------------------
    # Matching
    # -------------------------
    results = []

    for name, p_emb in pokemon_emb_db.items():
        if name not in pokemon_geo_db:
            continue

        p_geo = pokemon_geo_db[name]

        clip_sim = float(torch.dot(human_emb, p_emb.to(device)))
        geo_sim, geo_debug = geometry_similarity(
            human_geo, p_geo, return_debug=True
        )

        score = final_score(clip_sim, geo_sim)

        if geo_sim < 0.25 and clip_sim > 0.82:
            perceptual_bonus = 0.06
        else:
            perceptual_bonus = 0.0

        score += perceptual_bonus

        # human likeness gate (종 특성 보정)
        human_like = HUMAN_LIKENESS.get(name, 0.0)
        if human_like < -0.03:
            score *= 0.6
        elif human_like < -0.01:
            score *= 0.85

        explanation = build_geometry_explanation(
            human_geo, p_geo, geo_debug
        )

        results.append({
            "name": name,
            "final": float(score),
            "geo": float(geo_sim),
            "clip": float(clip_sim),
            "human": float(human_like),
            "explanation": explanation
        })

    results.sort(key=lambda x: -x["final"])

    return {
        "image": os.path.basename(image_path),
        "results": results[:TOPK],
    }


# ============================================================
# CLI
# ============================================================
def cli_main(image_path: str):
    result = run_match(image_path)

    print("\n[RESULT]")
    for i, r in enumerate(result["results"], 1):
        if r["final"] < THRESHOLD:
            break

        print(
            f"{i}. {r['name']:<12} "
            f"final={r['final']:.3f} "
            f"geo={r['geo']:.3f} "
            f"clip={r['clip']:.3f} "
            f"human={r['human']:+.3f}"
        )
        for line in r["explanation"]["summary"]:
            print("   -", line)


# =========================
# Entry
# =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    cli_main(args.image)
