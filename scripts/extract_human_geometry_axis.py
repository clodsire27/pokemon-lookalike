import cv2
import numpy as np

from scripts.face_roi_selector import select_face_roi
from scripts.geometry_utils import (
    extract_silhouette,
    face_aspect_ratio,
    estimate_eye_features,
    estimate_mouth_and_jaw,
)

# =========================
# Human geometry extraction
# (CLIP + ROI 기반, 최종)
# =========================
def extract_human_geometry_axis(
    img_path: str,
    clip_model,
    preprocess,
    text_emb,
    device,
):
    """
    사람 이미지 → 얼굴 ROI 선택 → geometry axis 추출

    반환값:
    {
        eye_spacing_ratio,
        eye_height_ratio,
        mouth_width_ratio,
        jaw_width_ratio,
        face_aspect_ratio
    }
    """

    # -------------------------
    # Load image
    # -------------------------
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -------------------------
    # Global silhouette
    # -------------------------
    mask = extract_silhouette(gray)

    # -------------------------
    # Face ROI selection (🔥 핵심)
    # -------------------------
    roi = select_face_roi(
        img_bgr=img,
        mask=mask,
        model=clip_model,
        preprocess=preprocess,
        text_emb=text_emb,
        device=device,
        grid=3,
    )

    if roi is None:
        raise ValueError("No face ROI detected (human)")

    x1, y1, x2, y2 = roi

    # -------------------------
    # Crop face region
    # -------------------------
    face = img[y1:y2, x1:x2]
    if face.size < 100:
        raise ValueError("Face ROI too small")

    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    mask_face = extract_silhouette(gray_face)

    # -------------------------
    # Geometry axis extraction
    # -------------------------
    aspect = face_aspect_ratio(mask_face)
    eye_spacing, eye_height = estimate_eye_features(gray_face)
    mouth_width, jaw_width = estimate_mouth_and_jaw(mask_face)

    return {
        "eye_spacing_ratio": float(eye_spacing),
        "eye_height_ratio": float(eye_height),
        "mouth_width_ratio": float(mouth_width),
        "jaw_width_ratio": float(jaw_width),
        "face_aspect_ratio": float(aspect),
    }


# =========================
# CLI debug (optional)
# =========================
if __name__ == "__main__":
    import argparse
    import open_clip
    import torch

    FACE_TEXT = "cute cartoon character face with eyes and mouth"

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", "openai"
    )
    model = model.to(device).eval()

    text_emb = model.encode_text(
        open_clip.tokenize([FACE_TEXT]).to(device)
    )
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb[0]

    geo = extract_human_geometry_axis(
        args.image,
        clip_model=model,
        preprocess=preprocess,
        text_emb=text_emb,
        device=device,
    )

    import json
    print(json.dumps(geo, indent=2, ensure_ascii=False))
