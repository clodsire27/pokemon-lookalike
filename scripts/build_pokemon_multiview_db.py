# scripts/build_pokemon_multiview_db.py

import os
import json
import torch
import open_clip
from PIL import Image

from scripts.build_pokemon_schema_from_axis import build_pokemon_schema_from_axis
from scripts.schema_to_multiview_prompts import schema_to_multiview_prompts
from scripts.encode_pokemon_multiview import encode_multiview_clip

# =========================
# Config
# =========================
AXIS_PATH = "data/pokemon_archetypes.json"
IMAGE_ROOT = "data/pokemon_original"   # 포켓몬 이미지 폴더 (이름=포켓몬명)
OUT_PATH  = "data/pokemon_multiview_embeddings.pt"

TEXT_WEIGHT  = 0.6
IMAGE_WEIGHT = 0.4

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"


@torch.no_grad()
def encode_image_clip(model, preprocess, image_path, device):
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    e = model.encode_image(x)
    e = e / e.norm(dim=-1, keepdim=True)
    return e[0]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================
    # Load CLIP
    # =========================
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, PRETRAINED
    )
    model = model.to(device).eval()

    # =========================
    # Load axis DB
    # =========================
    with open(AXIS_PATH, "r", encoding="utf-8") as f:
        axis_db = json.load(f)

    out = {}

    # =========================
    # Build embeddings
    # =========================
    for name, axis in axis_db.items():

        # 🚨 방어: 잘못된 엔트리 제거
        if name.lower() in {"pokemon", "test", "tmp"}:
            continue

        # ---------- TEXT (schema → prompts) ----------
        schema = build_pokemon_schema_from_axis(axis)
        prompts = schema_to_multiview_prompts(schema)

        text_emb = encode_multiview_clip(model, prompts)
        text_emb = text_emb / text_emb.norm()

        # ---------- IMAGE (multi-image mean) ----------
        img_dir = os.path.join(IMAGE_ROOT, name)
        img_embs = []

        if os.path.isdir(img_dir):
            for fn in os.listdir(img_dir):
                if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                    path = os.path.join(img_dir, fn)
                    try:
                        img_embs.append(
                            encode_image_clip(model, preprocess, path, device)
                        )
                    except Exception:
                        continue

        if img_embs:
            image_emb = torch.stack(img_embs).mean(dim=0)
            image_emb = image_emb / image_emb.norm()
        else:
            # 이미지 없으면 text만 사용 (fallback)
            image_emb = None

        # ---------- FUSION ----------
        if image_emb is not None:
            final_emb = (
                TEXT_WEIGHT * text_emb +
                IMAGE_WEIGHT * image_emb
            )
        else:
            final_emb = text_emb

        final_emb = final_emb / final_emb.norm()
        out[name] = final_emb.cpu()

    # =========================
    # Save
    # =========================
    torch.save(out, OUT_PATH)
    print(f"[OK] Saved pokemon multiview embeddings → {OUT_PATH}")
    print(f"[INFO] Total pokemons: {len(out)}")


if __name__ == "__main__":
    main()
