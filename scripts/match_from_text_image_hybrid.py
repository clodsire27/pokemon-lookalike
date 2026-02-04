import os
import argparse
import numpy as np
from PIL import Image

import torch
import open_clip
from torchvision import transforms

# =========================
# Config
# =========================
BASE_DIR = "/home/sea/project/pokemon"
INDEX_PATH = os.path.join(BASE_DIR, "checkpoints/pokemon_index_image.npz")
CKPT_PATH = os.path.join(BASE_DIR, "checkpoints/pokemon_clip.pt")

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5

# hybrid weight
ALPHA_TEXT = 0.4   # text
ALPHA_IMAGE = 0.6  # image

# =========================
# Utils
# =========================
def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


@torch.no_grad()
def encode_text(model, tokenizer, texts):
    tokens = tokenizer(texts).to(DEVICE)
    feats = model.encode_text(tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


@torch.no_grad()
def encode_image(model, preprocess, image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(DEVICE)
    feats = model.encode_image(image)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0]


def build_query_text_from_file(path):
    """
    Expect slot-based format, e.g.
    eye_distance: wide
    face_length: long
    jaw_width: narrow
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return " | ".join(lines)


# =========================
# Main
# =========================
def main(args):
    print("[INFO] Loading CLIP model")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=MODEL_NAME,
        pretrained=PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    model = model.to(DEVICE)

    # Load fine-tuned weights
    state = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()

    print("[INFO] Loading Pokémon index")
    data = np.load(INDEX_PATH, allow_pickle=True)
    names = data["names"]            # (K,)
    centroids = data["centroids"]    # (K, 512)
    centroids = l2_normalize(centroids)

    # =========================
    # Build query embedding
    # =========================
    embeddings = []

    # ---- text ----
    if args.text_file:
        query_text = build_query_text_from_file(args.text_file)
        print("\n[QUERY TEXT]")
        print(query_text)
        text_emb = encode_text(model, tokenizer, [query_text])[0]
        embeddings.append((ALPHA_TEXT, text_emb))

    # ---- image ----
    if args.image:
        print("\n[QUERY IMAGE]")
        print(args.image)
        image_emb = encode_image(model, preprocess, args.image)
        embeddings.append((ALPHA_IMAGE, image_emb))

    if not embeddings:
        raise ValueError("Provide at least --text_file or --image")

    # ---- hybrid fusion ----
    q = np.zeros_like(embeddings[0][1])
    for w, emb in embeddings:
        q += w * emb
    q = q / np.linalg.norm(q)

    # =========================
    # Similarity + ranking
    # =========================
    sims = centroids @ q

    # z-score normalization (stability)
    mu = sims.mean()
    sigma = sims.std() + 1e-6
    zs = (sims - mu) / sigma

    top_idx = np.argsort(zs)[::-1][:TOP_K]

    print("\nTop-{} Pokémon lookalikes:".format(TOP_K))
    for rank, idx in enumerate(top_idx, 1):
        print(
            f"{rank:>2}. {names[idx]:<10s}  "
            f"score={sims[idx]:.4f}  z={zs[idx]:.2f}"
        )


# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text_file",
        type=str,
        default=None,
        help="Path to slot-based face description text file"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to character or face image"
    )

    args = parser.parse_args()
    main(args)
