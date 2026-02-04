import os, json, torch
import numpy as np
from PIL import Image
import open_clip

from scripts.axis_utils import compress_axis_soft, filter_axis

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_DIR = "data/pokemon_original"
OUT_PATH  = "data/pokemon_archetypes.json"

ARCHETYPES = {
    "eye_sharp":   ["sharp eyes", "narrow eyes"],
    "eye_round":   ["round eyes", "big eyes"],
    "eye_droopy":  ["droopy eyes", "sleepy eyes"],

    "face_sharp":  ["sharp face", "V-shaped face", "pointed jaw"],
    "face_round":  ["round face", "soft cheeks"],
    "face_long":   ["long face"],

    "vibe_cool":      ["cool impression", "calm", "chic"],
    "vibe_cute":      ["cute", "adorable"],
    "vibe_elegant":   ["elegant", "graceful"],
    "vibe_soft":      ["soft", "gentle"],
    "vibe_wild":      ["wild", "fierce"],
    "vibe_mysterious":["mysterious"],
}
def to_py(o):
    import numpy as np
    import torch
    if isinstance(o, dict):
        return {k: to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, (np.generic,)):
        return o.item()
    if isinstance(o, torch.Tensor):
        return o.item()
    return o


def main():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", "openai"
    )
    model = model.to(DEVICE).eval()

    # text embeddings
    text_embs = {}
    for k, texts in ARCHETYPES.items():
        tokens = open_clip.tokenize(texts).to(DEVICE)
        with torch.no_grad():
            t = model.encode_text(tokens)
            t = t / t.norm(dim=-1, keepdim=True)
        text_embs[k] = t.mean(dim=0)

    result = {}

    for name in sorted(os.listdir(IMAGE_DIR)):
        folder = os.path.join(IMAGE_DIR, name)
        if not os.path.isdir(folder):
            continue

        img_embs = []
        for f in os.listdir(folder):
            if not f.lower().endswith((".png",".jpg",".jpeg")):
                continue
            img = preprocess(Image.open(os.path.join(folder,f)).convert("RGB"))
            img = img.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                e = model.encode_image(img)
                e = e / e.norm(dim=-1, keepdim=True)
            img_embs.append(e[0])

        if not img_embs:
            continue

        img_mean = torch.stack(img_embs).mean(dim=0)

        scores = {
            k: float((img_mean @ t).cpu())
            for k, t in text_embs.items()
        }

        # group softmax
        def softmax(keys, temp):
            vals = np.array([scores[k] for k in keys])
            exps = np.exp(vals / temp)
            probs = exps / (exps.sum() + 1e-12)
            for k, v in zip(keys, probs):
                scores[k] = float(v)

        softmax(["eye_sharp","eye_round","eye_droopy"], 0.07)
        softmax(["face_sharp","face_round","face_long"], 0.07)
        softmax(
            ["vibe_cool","vibe_cute","vibe_elegant","vibe_soft","vibe_wild","vibe_mysterious"],
            0.08
        )

        # 🔥 핵심: 압축 + 필터
        scores = compress_axis_soft(scores, temperature=0.02)
        scores = filter_axis(scores)

        result[name] = scores

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(to_py(result), f, ensure_ascii=False, indent=2)


    print("[OK] Saved →", OUT_PATH)

if __name__ == "__main__":
    main()
