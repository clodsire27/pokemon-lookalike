#/home/sea/project/pokemon/scripts/extract_human_archetype.py
import os
import json
import numpy as np
import cv2
from PIL import Image

import torch
import open_clip

from scripts.build_pokemon_attributes_axis import extract_face_crop

# =========================
# Config
# =========================
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HOG_SIZE = 128
TOP_K = 2
TRAIT_THRESHOLD = 0.15

# =========================
# Archetype definition (🔥 수정 핵심)
# =========================
ARCHETYPES = {
    # 여우상 = 눈 + 분위기 (윤곽 제거)
    "fox":  ["sharp_eyes", "cool"],

    "cat":  ["sharp_eyes", "cool"],
    "dog":  ["round_eyes", "cute"],
    "bear": ["round_face", "soft"],
    "duck": ["droopy_eyes", "round_face"],
}

# =========================
# Model load
# =========================
print("[INIT] Loading CLIP...")
model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME, PRETRAINED
)
model = model.to(DEVICE).eval()
print("[INIT] Ready")

# =========================
# Utils
# =========================
def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def normalize_group(traits: dict, keys, temperature=0.07):
    """같은 계열 내부 softmax"""
    vals = np.array([traits[k] for k in keys], dtype=np.float32)
    exps = np.exp(vals / temperature)
    probs = exps / (exps.sum() + 1e-12)
    for k, v in zip(keys, probs):
        traits[k] = float(v)


def extract_top_traits(traits: dict):
    items = sorted(traits.items(), key=lambda x: -x[1])
    return {k: float(v) for k, v in items[:TOP_K] if v >= TRAIT_THRESHOLD}


def archetype_score(traits: dict, keys):
    """
    🔥 평균 ❌ → 최대값 ⭕
    여우상은 핵심 하나만 강해도 성립
    """
    vals = [traits[k] for k in keys if k in traits]
    if not vals:
        return 0.0
    return float(max(vals))


def to_py(o):
    if isinstance(o, dict):
        return {k: to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o

# =========================
# Feature extraction
# =========================
def hog_feature_from_bgr(face_bgr):
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (HOG_SIZE, HOG_SIZE))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 120)

    hog = cv2.HOGDescriptor(
        _winSize=(HOG_SIZE, HOG_SIZE),
        _blockSize=(32, 32),
        _blockStride=(16, 16),
        _cellSize=(16, 16),
        _nbins=9
    )
    feat = hog.compute(edges).reshape(-1)
    return feat / (np.linalg.norm(feat) + 1e-12)


@torch.no_grad()
def clip_image_embed(face_bgr):
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = preprocess(pil).unsqueeze(0).to(DEVICE)
    f = model.encode_image(x)
    f = f / f.norm(dim=-1, keepdim=True)
    return f.cpu().numpy()[0]


@torch.no_grad()
def clip_text_embed(text):
    tokens = open_clip.tokenize([text]).to(DEVICE)
    t = model.encode_text(tokens)
    t = t / t.norm(dim=-1, keepdim=True)
    return t.cpu().numpy()[0]

# =========================
# Trait extraction
# =========================
def extract_traits(q_clip, q_hog):
    traits = {}

    # --- Eye traits (CLIP 핵심)
    eye_texts = {
        "sharp_eyes":  "sharp eyes, narrow eyes, intense gaze",
        "round_eyes":  "round eyes, big eyes, cute eyes",
        "droopy_eyes": "droopy eyes, sleepy eyes",
    }
    for k, txt in eye_texts.items():
        traits[k] = cosine(q_clip, clip_text_embed(txt))

    # --- Face shape (HOG, 참고용)
    thirds = len(q_hog) // 3
    eye_e = np.linalg.norm(q_hog[:thirds])
    mid_e = np.linalg.norm(q_hog[thirds:2*thirds])
    jaw_e = np.linalg.norm(q_hog[2*thirds:])

    total = eye_e + mid_e + jaw_e + 1e-12
    traits["round_face"]   = mid_e / total
    traits["angular_face"] = jaw_e / total

    normalize_group(
        traits,
        ["round_face", "angular_face"],
        temperature=0.2
    )

    # --- Vibe traits (CLIP)
    vibe_texts = {
        "cool": "cool impression, calm, sharp",
        "cute": "cute, friendly, adorable",
        "soft": "soft, gentle, warm",
        "wild": "wild, aggressive, strong",
    }
    for k, txt in vibe_texts.items():
        traits[k] = cosine(q_clip, clip_text_embed(txt))

    return traits

# =========================
# Main API
# =========================
def extract_human_archetype(image_path: str):
    face = extract_face_crop(image_path)

    q_clip = clip_image_embed(face)
    q_hog  = hog_feature_from_bgr(face)

    traits = extract_traits(q_clip, q_hog)

    # 그룹별 정규화
    normalize_group(traits, ["sharp_eyes", "round_eyes", "droopy_eyes"], temperature=0.07)
    normalize_group(traits, ["cool", "cute", "soft", "wild"], temperature=0.07)

    top_traits = extract_top_traits(traits)

    archetypes = {}
    for name, keys in ARCHETYPES.items():
        archetypes[name] = archetype_score(traits, keys)

    archetypes = dict(sorted(archetypes.items(), key=lambda x: -x[1]))

    return {
        "top_traits": top_traits,
        "archetypes": archetypes,
    }

# =========================
# CLI
# =========================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    result = extract_human_archetype(args.image)
    print(json.dumps(to_py(result), indent=2, ensure_ascii=False))
