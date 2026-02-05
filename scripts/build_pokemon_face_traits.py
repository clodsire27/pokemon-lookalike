import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import clip

# =========================
# Config
# =========================
BASE_DIR = "/home/sea/project/pokemon"
POKEMON_IMG_DIR = os.path.join(BASE_DIR, "images")
OUT_JSON = os.path.join(BASE_DIR, "pokemon_face_traits.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# CLIP
# =========================
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

# =========================
# Main
# =========================
def main():
    traits = {}
    all_embs = []

    # 1️⃣ 포켓몬별 평균 임베딩
    for pokemon in tqdm(sorted(os.listdir(POKEMON_IMG_DIR))):
        p_dir = os.path.join(POKEMON_IMG_DIR, pokemon)
        if not os.path.isdir(p_dir):
            continue

        embs = []
        for fname in os.listdir(p_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img = Image.open(os.path.join(p_dir, fname)).convert("RGB")
            img = preprocess(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                emb = model.encode_image(img).cpu().numpy()[0]
            embs.append(emb)

        if not embs:
            continue

        mean_emb = np.mean(np.stack(embs), axis=0)
        traits[pokemon] = {"_raw": mean_emb}
        all_embs.append(mean_emb)

    # 2️⃣ 전체 기준 z-score
    all_embs = np.stack(all_embs)
    mu = all_embs.mean(axis=0)
    std = all_embs.std(axis=0) + 1e-6

    # 3️⃣ 연속 기하학 proxy 생성
    for p, d in traits.items():
        z = (d["_raw"] - mu) / std

        geom_vector = {
            "face_ratio": float(z[:100].mean()),
            "eye_distance": float(z[100:200].mean()),
            "eye_size": float(z[200:300].mean()),
            "jaw_width": float(z[300:400].mean()),
            "complexity": float(np.var(z)),
        }

        traits[p] = {
            "geom_vector": geom_vector
        }

    # 4️⃣ 저장
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(traits, f, ensure_ascii=False, indent=2)

    print(f"✅ Pokémon face traits with geom_vector saved to {OUT_JSON}")


if __name__ == "__main__":
    main()