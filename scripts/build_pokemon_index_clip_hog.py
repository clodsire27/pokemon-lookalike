import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import open_clip

# =========================
# Config
# =========================
BASE_DIR = "/home/sea/project/pokemon"
IMG_DIR = os.path.join(BASE_DIR, "images")
OUT_NPZ = os.path.join(BASE_DIR, "checkpoints/pokemon_index_clip_hog.npz")

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# HOG / Edge params (🔥 튜닝 포인트)
HOG_SIZE = 128
CANNY_LOW = 60
CANNY_HIGH = 120

# =========================
# Utils
# =========================
def l2n(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def alpha_bbox_crop(pil_img: Image.Image, pad=0.05):
    """
    PNG alpha 우선 → 없으면 밝은 배경 heuristic
    실패 시 원본 RGB 반환 (절대 죽지 않음)
    """
    img = np.array(pil_img)

    # grayscale 방어
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    h, w = img.shape[:2]

    try:
        # 1️⃣ alpha channel
        if img.shape[-1] == 4:
            alpha = img[..., 3]
            coords = np.where(alpha > 10)
        else:
            # 2️⃣ 거의 흰 배경 가정
            rgb = img[..., :3]
            mask = np.any(rgb < 245, axis=-1)
            coords = np.where(mask)

        if len(coords) != 2 or len(coords[0]) == 0:
            raise ValueError("invalid mask")

        ys, xs = coords
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        # padding
        dx = int((x2 - x1) * pad)
        dy = int((y2 - y1) * pad)
        x1 = max(0, x1 - dx)
        y1 = max(0, y1 - dy)
        x2 = min(w - 1, x2 + dx)
        y2 = min(h - 1, y2 + dy)

        crop = img[y1:y2 + 1, x1:x2 + 1, :3]
        return Image.fromarray(crop).convert("RGB")

    except Exception:
        return pil_img.convert("RGB")


def hog_feature(pil_img: Image.Image, out_size=HOG_SIZE):
    """
    외곽선 기반 HOG (형태/실루엣 집중)
    """
    img = np.array(pil_img.convert("RGB"))
    img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

    hog = cv2.HOGDescriptor(
        _winSize=(out_size, out_size),
        _blockSize=(32, 32),
        _blockStride=(16, 16),
        _cellSize=(16, 16),
        _nbins=9
    )

    feat = hog.compute(edges).reshape(-1).astype(np.float32)
    feat = feat / (np.linalg.norm(feat) + 1e-12)
    return feat


@torch.no_grad()
def clip_embed(model, preprocess, pil_img: Image.Image):
    x = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    f = model.encode_image(x)
    f = f / f.norm(dim=-1, keepdim=True)
    return f.cpu().numpy()[0]


# =========================
# Main
# =========================
def main():
    print("[INFO] Loading CLIP model")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, PRETRAINED
    )
    model = model.to(DEVICE).eval()

    species = []
    clip_vecs = []
    hog_vecs = []

    print("[INFO] Building Pokémon CLIP + HOG index")

    for name in tqdm(sorted(os.listdir(IMG_DIR))):
        pdir = os.path.join(IMG_DIR, name)
        if not os.path.isdir(pdir):
            continue

        per_clip = []
        per_hog = []

        for fn in os.listdir(pdir):
            if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            pil = Image.open(os.path.join(pdir, fn))
            pil = alpha_bbox_crop(pil, pad=0.08)

            if pil.width < 64 or pil.height < 64:
                continue


            per_clip.append(clip_embed(model, preprocess, pil))
            per_hog.append(hog_feature(pil))

        if not per_clip:
            continue

        # 종 단위 평균
        c = np.mean(np.stack(per_clip), axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)

        h = np.mean(np.stack(per_hog), axis=0)
        h = h / (np.linalg.norm(h) + 1e-12)

        species.append(name)
        clip_vecs.append(c)
        hog_vecs.append(h)

    np.savez(
        OUT_NPZ,
        names=np.array(species),
        clip=l2n(np.stack(clip_vecs)),
        hog=l2n(np.stack(hog_vecs)),
    )

    print(f"✅ Pokémon CLIP+HOG index saved")
    print(f"   path: {OUT_NPZ}")
    print(f"   species count: {len(species)}")


if __name__ == "__main__":
    main()
