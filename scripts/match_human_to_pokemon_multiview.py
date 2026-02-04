# scripts/match_human_to_pokemon_multiview.py

import torch
import argparse
import open_clip
from PIL import Image

from scripts.extract_human_axis_clip import extract_human_axis
from scripts.build_human_schema_from_axis import build_human_schema_from_axis
from scripts.schema_to_multiview_prompts import schema_to_multiview_prompts
from scripts.encode_pokemon_multiview import encode_multiview_clip

POKEMON_DB = "data/pokemon_multiview_embeddings.pt"
TOPK = 5

# 🔥 이미지 임베딩 비중 (이게 핵심)
IMAGE_WEIGHT = 0.4
TEXT_WEIGHT  = 0.6


@torch.no_grad()
def encode_image_clip(model, preprocess, image_path, device):
    """
    사람 얼굴 이미지를 CLIP image embedding으로 변환
    """
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    e = model.encode_image(x)
    e = e / e.norm(dim=-1, keepdim=True)
    return e[0]


def main(image_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================
    # Load CLIP (text + image)
    # =========================
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", "openai"
    )
    model = model.to(device).eval()

    # =========================
    # Load Pokemon DB
    # =========================
    pokemon_db = torch.load(POKEMON_DB, map_location=device)

    # 🚨 방어 코드: 'pokemon' 엔트리 강제 제거
    if "pokemon" in pokemon_db:
        print("⚠️  WARNING: removing invalid entry 'pokemon' from DB at runtime")
        pokemon_db.pop("pokemon")

    # =========================
    # Human → axis → schema → text prompts
    # =========================
    axis = extract_human_axis(image_path)
    schema = build_human_schema_from_axis(axis)
    prompts = schema_to_multiview_prompts(schema)

    # =========================
    # Human TEXT embedding
    # =========================
    text_emb = encode_multiview_clip(model, prompts)
    text_emb = text_emb / text_emb.norm()

    # =========================
    # Human IMAGE embedding (🔥 결정타)
    # =========================
    image_emb = encode_image_clip(model, preprocess, image_path, device)

    # =========================
    # Final human embedding (text + image fusion)
    # =========================
    human_emb = (
        TEXT_WEIGHT * text_emb +
        IMAGE_WEIGHT * image_emb
    )
    human_emb = human_emb / human_emb.norm()

    # =========================
    # Cosine match
    # =========================
    scores = []
    for name, p_emb in pokemon_db.items():
        p_emb = p_emb.to(device)
        sim = float(torch.dot(human_emb, p_emb))
        scores.append((name, sim))

    scores.sort(key=lambda x: -x[1])

    # =========================
    # Print
    # =========================
    print("\n[RESULT]")
    for i, (name, score) in enumerate(scores[:TOPK], 1):
        print(f"{i}. {name:<12} similarity={score:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    main(args.image)
