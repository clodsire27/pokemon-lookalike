import torch, clip
import numpy as np
from PIL import Image
from scripts.extract_face_attributes import extract_face_crop

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HUMAN_ARCHETYPE_TEXTS = {
    "fox":  "fox-like face, sharp eyes, cool impression",
    "dog":  "dog-like face, friendly, round eyes",
    "cat":  "cat-like face, slim, sharp eyes",
    "bear": "bear-like face, round, soft impression",
    "duck": "duck-like face, droopy eyes",
}

_model = None
_preprocess = None
_text_emb = None

def _load():
    global _model, _preprocess, _text_emb
    if _model is not None:
        return
    _model, _preprocess = clip.load("ViT-B/32", device=DEVICE)

    texts = list(HUMAN_ARCHETYPE_TEXTS.values())
    tokens = clip.tokenize(texts).to(DEVICE)
    emb = _model.encode_text(tokens)
    _text_emb = emb / emb.norm(dim=-1, keepdim=True)

@torch.no_grad()
def extract_human_archetype(image_path):
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)

    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    img_emb = model.encode_image(img)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

    tokens = clip.tokenize(list(HUMAN_ARCHETYPE_TEXTS.values())).to(DEVICE)
    text_emb = model.encode_text(tokens)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    sim = (img_emb @ text_emb.T).detach().cpu().numpy()[0]

    minv, maxv = sim.min(), sim.max()
    norm = (sim - minv) / (maxv - minv + 1e-6)

    return dict(zip(HUMAN_ARCHETYPE_TEXTS.keys(), norm))

