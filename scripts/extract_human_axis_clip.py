# scripts/extract_human_axis_clip.py

import torch
import open_clip
import numpy as np
import cv2
from PIL import Image
import json

from scripts.axis_utils import compress_axis_soft
from scripts.extract_face_attributes import extract_face_crop

# =========================
# Config
# =========================
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HOG_SIZE = 128

# =========================
# CLIP init (1회 로드)
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
    return float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    )


def softmax_norm(values: dict, temperature=0.07):
    keys = list(values.keys())
    v = np.array([values[k] for k in keys], dtype=np.float32)
    e = np.exp(v / temperature)
    p = e / (e.sum() + 1e-12)
    return {k: float(x) for k, x in zip(keys, p)}


@torch.no_grad()
def clip_image_embed(face_bgr):
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = preprocess(pil).unsqueeze(0).to(DEVICE)
    f = model.encode_image(x)
    f = f / f.norm(dim=-1, keepdim=True)
    return f.cpu().numpy()[0]


@torch.no_grad()
def clip_text_embed(text: str):
    tokens = open_clip.tokenize([text]).to(DEVICE)
    f = model.encode_text(tokens)
    f = f / f.norm(dim=-1, keepdim=True)
    return f.cpu().numpy()[0]


# =========================
# Face shape (HOG)
# =========================
def hog_face_shape(face_bgr):
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

    thirds = len(feat) // 3
    jaw = np.linalg.norm(feat[2 * thirds:])
    mid = np.linalg.norm(feat[thirds:2 * thirds])
    eye = np.linalg.norm(feat[:thirds])

    total = jaw + mid + eye + 1e-12

    return {
        "face_sharp": jaw / total,
        "face_round": mid / total,
        "face_long":  eye / total,
    }

# =========================
# Axis prompts
# =========================
EYE_PROMPTS = {
    # 🔥 핵심: 차분한 여우 눈매
    "eye_fox":    "slanted eyes, upturned eyes, fox-like eyes, elegant narrow eyes",

    # 기존 축
    "eye_sharp":  "sharp eyes, intense gaze, strong eye expression",
    "eye_round":  "round eyes, big eyes, cute impression",
    "eye_droopy": "droopy eyes, sleepy eyes",
}

VIBE_PROMPTS = {
    "vibe_cool":    "cool impression, calm, chic",
    "vibe_cute":    "cute impression, adorable",
    "vibe_soft":    "soft impression, gentle",
    "vibe_elegant": "elegant impression, graceful",
}

# =========================
# Main extraction
# =========================
def extract_human_axis(image_path: str) -> dict:
    """
    사람 얼굴 → raw axis (압축까지만)
    ⚠️ filter_axis는 여기서 절대 사용하지 않음
    """
    face = extract_face_crop(image_path)
    img_emb = clip_image_embed(face)

    # --- Eye axis (CLIP)
    eye_scores = {
        k: cosine(img_emb, clip_text_embed(v))
        for k, v in EYE_PROMPTS.items()
    }
    eye_axis = softmax_norm(eye_scores, temperature=0.07)

    # --- Face axis (HOG)
    face_axis = softmax_norm(
        hog_face_shape(face),
        temperature=0.20
    )

    # --- Vibe axis (CLIP)
    vibe_scores = {
        k: cosine(img_emb, clip_text_embed(v))
        for k, v in VIBE_PROMPTS.items()
    }
    vibe_axis = softmax_norm(vibe_scores, temperature=0.07)

    # --- Merge
    axis = {}
    axis.update(eye_axis)
    axis.update(face_axis)
    axis.update(vibe_axis)

    # 🔥 winner-take-most 압축
    axis = compress_axis_soft(axis, temperature=0.08)

    return axis

# =========================
# CLI test
# =========================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    axis = extract_human_axis(args.image)
    print(json.dumps(axis, indent=2, ensure_ascii=False))
