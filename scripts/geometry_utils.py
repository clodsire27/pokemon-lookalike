import cv2
import numpy as np

# =========================
# Image utils
# =========================
def load_gray(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def extract_silhouette(gray):
    _, th = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    th = cv2.morphologyEx(
        th, cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8)
    )
    return th


# =========================
# Geometry helpers
# =========================
def face_aspect_ratio(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return 1.0
    h = ys.max() - ys.min()
    w = xs.max() - xs.min()
    return h / (w + 1e-6)


def estimate_eye_features(gray):
    h, w = gray.shape
    roi = gray[:int(h * 0.5), :]

    edges = cv2.Canny(roi, 50, 150)
    cnts, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
    if len(cnts) < 2:
        return 0.35, 0.30  # fallback

    centers = []
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        centers.append((cx, cy))

    if len(centers) < 2:
        return 0.35, 0.30

    (x1, y1), (x2, y2) = centers
    eye_spacing = abs(x1 - x2) / w
    eye_height = ((y1 + y2) / 2) / h

    return eye_spacing, eye_height


def estimate_mouth_and_jaw(mask):
    h, w = mask.shape
    lower = mask[int(h * 0.6):, :]
    ys, xs = np.where(lower > 0)

    if len(xs) < 10:
        return 0.30, 1.0

    mouth_width = (xs.max() - xs.min()) / w
    return mouth_width, 1.0
