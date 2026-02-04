import os, json, torch, clip
from PIL import Image
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_DIR = "data/pokemon_images"
OUT_PATH = "data/pokemon_archetype_axis.json"

ARCHETYPE_TEXTS = {
    "fox":  "fox-like pokemon, narrow face, sharp eyes, elegant",
    "dog":  "dog-like pokemon, friendly face, round eyes",
    "cat":  "cat-like pokemon, slim face, sharp eyes",
    "bear": "bear-like pokemon, round body, soft appearance",
    "duck": "duck-like pokemon, droopy eyes, beak-like face",
}

def load_clip():
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    return model, preprocess

@torch.no_grad()
def encode_image(model, preprocess, path):
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
    emb = model.encode_image(img)
    return emb / emb.norm(dim=-1, keepdim=True)

@torch.no_grad()
def encode_text(model):
    texts = list(ARCHETYPE_TEXTS.values())
    tokens = clip.tokenize(texts).to(DEVICE)
    emb = model.encode_text(tokens)
    return emb / emb.norm(dim=-1, keepdim=True)

def main():
    model, preprocess = load_clip()
    text_emb = encode_text(model)

    result = {}

    for name in sorted(os.listdir(IMAGE_DIR)):
        folder = os.path.join(IMAGE_DIR, name)
        if not os.path.isdir(folder):
            continue

        scores = []

        for f in os.listdir(folder):
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img_emb = encode_image(model, preprocess, os.path.join(folder, f))
            sim = (img_emb @ text_emb.T).cpu().numpy()[0]
            scores.append(sim)

        if not scores:
            continue

        mean_score = np.mean(scores, axis=0)

        # min-max normalize (중요!)
        minv, maxv = mean_score.min(), mean_score.max()
        norm = (mean_score - minv) / (maxv - minv + 1e-6)

        result[name] = {
            k: float(v)
            for k, v in zip(ARCHETYPE_TEXTS.keys(), norm)
            if v > 0.4  # 🔥 threshold
        }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("[OK] archetype axis saved →", OUT_PATH)

if __name__ == "__main__":
    main()
