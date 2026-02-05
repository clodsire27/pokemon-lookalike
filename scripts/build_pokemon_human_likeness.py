import torch
import json
import open_clip
from pathlib import Path
from PIL import Image

# =========================
# Config
# =========================
POKEMON_IMAGE_ROOT = "data/pokemon_original"
OUTPUT_JSON = "data/pokemon_human_likeness.json"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

TEXT_HUMAN  = "anime human face, cute character"
TEXT_ANIMAL = "animal face, cartoon creature"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# CLIP utils
# =========================
@torch.no_grad()
def encode_text(model, text):
    tokens = open_clip.tokenize([text]).to(DEVICE)
    emb = model.encode_text(tokens)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0]


@torch.no_grad()
def encode_image(model, preprocess, img_path):
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    emb = model.encode_image(x)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0]


# =========================
# Main
# =========================
def main():
    print("[INIT] Loading CLIP...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, PRETRAINED
    )
    model = model.to(DEVICE).eval()

    print("[INIT] Encoding text prompts...")
    human_text_emb  = encode_text(model, TEXT_HUMAN)
    animal_text_emb = encode_text(model, TEXT_ANIMAL)

    root = Path(POKEMON_IMAGE_ROOT)
    result = {}

    print("[RUN] Computing human_likeness...")
    for pokemon_dir in sorted(root.iterdir()):
        if not pokemon_dir.is_dir():
            continue

        name = pokemon_dir.name
        images = sorted(
            p for p in pokemon_dir.iterdir()
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
        )

        if not images:
            print(f"[SKIP] {name}: no images")
            continue

        img_path = images[0]  # 대표 이미지 1장

        try:
            img_emb = encode_image(model, preprocess, img_path)

            sim_human  = float(torch.dot(img_emb, human_text_emb))
            sim_animal = float(torch.dot(img_emb, animal_text_emb))

            human_likeness = sim_human - sim_animal
            result[name] = human_likeness

            print(
                f"[OK] {name:<12} "
                f"human={sim_human:+.3f}  "
                f"animal={sim_animal:+.3f}  "
                f"Δ={human_likeness:+.3f}"
            )

        except Exception as e:
            print(f"[FAIL] {name}: {e}")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nSaved → {OUTPUT_JSON}")
    print(f"Total Pokémon: {len(result)}")


if __name__ == "__main__":
    main()
