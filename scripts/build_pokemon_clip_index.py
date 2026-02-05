import os
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POKEMON_IMG_DIR = os.path.join(BASE_DIR, "images")
OUT_EMB = os.path.join(BASE_DIR, "clip_pokemon_embeddings.npy")
OUT_NAME = os.path.join(BASE_DIR, "clip_pokemon_names.npy")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

embs = []
names = []

for pokemon in tqdm(sorted(os.listdir(POKEMON_IMG_DIR))):
    p_dir = os.path.join(POKEMON_IMG_DIR, pokemon)
    if not os.path.isdir(p_dir):
        continue

    for fname in os.listdir(p_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img = Image.open(os.path.join(p_dir, fname)).convert("RGB")
        img = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            # 🔥 normalize 절대 하지 마라
            emb = model.encode_image(img).cpu().numpy()[0]

        embs.append(emb)
        names.append(pokemon)

# numpy array
embs = np.stack(embs)

# 🔥 최종 단계에서만 normalize
embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

np.save(OUT_EMB, embs)
np.save(OUT_NAME, np.array(names))

print("✅ Pokémon CLIP index built (FIXED)")
