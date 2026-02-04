import torch, clip
import numpy as np
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def extract_human_archetype(image_path):
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)

    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    img_emb = model.encode_image(img)
    img_emb /= img_emb.norm(dim=-1, keepdim=True)

    texts = list(HUMAN_ARCHETYPE_TEXTS.values())
    tokens = clip.tokenize(texts).to(DEVICE)
    text_emb = model.encode_text(tokens)
    text_emb /= text_emb.norm(dim=-1, keepdim=True)

    sim = (img_emb @ text_emb.T).cpu().numpy()[0]

    minv, maxv = sim.min(), sim.max()
    norm = (sim - minv) / (maxv - minv + 1e-6)

    return dict(zip(HUMAN_ARCHETYPE_TEXTS.keys(), norm))
