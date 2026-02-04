import os
import sys
import torch
import clip
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMB = np.load(os.path.join(BASE_DIR, "clip_pokemon_mean_embeddings.npy"))
NAMES = np.load(os.path.join(BASE_DIR, "clip_pokemon_mean_names.npy"))

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def encode_image(path):
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img).cpu().numpy()[0]
    return emb / np.linalg.norm(emb)

def dual_match(anchor_path, personal_path, alpha=0.2, top_k=5):
    e_anchor = encode_image(anchor_path)
    e_personal = encode_image(personal_path)

    e_final = e_anchor + alpha * (e_personal - e_anchor)
    e_final = e_final / np.linalg.norm(e_final)

    sims = EMB @ e_final
    idx = np.argsort(-sims)[:top_k]

    print("\n=== DUAL MATCH RESULT ===")
    print(f"alpha = {alpha}")
    for i in idx:
        print(f"{NAMES[i]} (score={sims[i]:.4f})")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python dual_clip_match.py <anchor_img> <personal_img> [alpha]")
        sys.exit(1)

    anchor = sys.argv[1]
    personal = sys.argv[2]
    alpha = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.2

    dual_match(anchor, personal, alpha)
