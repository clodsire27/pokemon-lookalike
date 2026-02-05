import cv2
import numpy as np
import torch
from PIL import Image

# =========================
# ROI candidates
# =========================
def generate_rois(mask, grid=3):
    h, w = mask.shape
    rois = []
    for gy in range(grid):
        for gx in range(grid):
            y1 = int(h * gy / grid)
            y2 = int(h * (gy + 1) / grid)
            x1 = int(w * gx / grid)
            x2 = int(w * (gx + 1) / grid)
            roi_mask = mask[y1:y2, x1:x2]
            if roi_mask.sum() > 50:
                rois.append((x1, y1, x2, y2))
    return rois


# =========================
# Scores
# =========================
def symmetry_score(mask):
    h, w = mask.shape
    if w < 10:
        return 0.0
    left = mask[:, :w//2]
    right = np.fliplr(mask[:, w//2:])
    minw = min(left.shape[1], right.shape[1])
    diff = np.abs(left[:, :minw] - right[:, :minw])
    return 1.0 - (diff.mean() / 255.0)


def edge_density(gray):
    edges = cv2.Canny(gray, 50, 150)
    return edges.sum() / (edges.size + 1e-6)


@torch.no_grad()
def clip_face_score(model, preprocess, roi_bgr, text_emb, device):
    roi_rgb = roi_bgr[:, :, ::-1]
    img = Image.fromarray(roi_rgb)
    x = preprocess(img).unsqueeze(0).to(device)
    emb = model.encode_image(x)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return float(torch.dot(emb[0], text_emb))


# =========================
# Final selector
# =========================
def select_face_roi(img_bgr, mask, model, preprocess, text_emb, device, grid=3):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    rois = generate_rois(mask, grid)

    best, best_score = None, -1e9

    for (x1, y1, x2, y2) in rois:
        roi_mask = mask[y1:y2, x1:x2]
        roi_gray = gray[y1:y2, x1:x2]
        roi_bgr  = img_bgr[y1:y2, x1:x2]

        if roi_gray.size < 100:
            continue

        s = (
            0.4 * symmetry_score(roi_mask) +
            0.3 * edge_density(roi_gray) +
            0.3 * clip_face_score(model, preprocess, roi_bgr, text_emb, device)
        )

        if s > best_score:
            best_score = s
            best = (x1, y1, x2, y2)

    return best
