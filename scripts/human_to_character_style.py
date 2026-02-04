import os
import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
import mediapipe as mp

# =====================
# PATH SETUP
# =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "temp_character_style")
os.makedirs(OUT_DIR, exist_ok=True)

# =====================
# DEVICE
# =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# =====================
# SD PIPELINE (txt2img)
# =====================
pipe = StableDiffusionPipeline.from_pretrained(
    "gsdf/counterfeit-v2.5",
    safety_checker=None,
    torch_dtype=dtype
).to(device)

# =====================
# FIXED SEED (🔥 핵심)
# =====================
GENERATOR = torch.Generator(device=device).manual_seed(42)

# =====================
# MEDIAPIPE FACE
# =====================
mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

# =====================
# LANDMARK INDEX
# =====================
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
JAW = [234, 454]

# =====================
# FACE PARAM EXTRACTION
# =====================
def extract_face_params(img_bgr):
    h, w, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = mp_face.process(img_rgb)

    if not res.multi_face_landmarks:
        raise RuntimeError("Face not detected")

    lm = res.multi_face_landmarks[0].landmark

    def pt(i):
        return np.array([lm[i].x * w, lm[i].y * h])

    left_eye = (pt(LEFT_EYE[0]) + pt(LEFT_EYE[1])) / 2
    right_eye = (pt(RIGHT_EYE[0]) + pt(RIGHT_EYE[1])) / 2

    eye_dist = np.linalg.norm(left_eye - right_eye) / w
    eye_y = ((left_eye[1] + right_eye[1]) / 2) / h
    jaw_width = np.linalg.norm(pt(JAW[0]) - pt(JAW[1])) / w
    face_ratio = h / w

    return {
        "eye_distance": eye_dist,
        "eye_y": eye_y,
        "jaw_width": jaw_width,
        "face_ratio": face_ratio,
    }

# =====================
# PARAM → PROMPT
# =====================
def face_params_to_prompt(p):
    tokens = []

    if p["eye_distance"] < 0.28:
        tokens.append("close set eyes")
    elif p["eye_distance"] > 0.34:
        tokens.append("wide set eyes")

    if p["eye_y"] < 0.42:
        tokens.append("high eye position")
    elif p["eye_y"] > 0.48:
        tokens.append("low eye position")

    if p["face_ratio"] > 1.35:
        tokens.append("long face")
    elif p["face_ratio"] < 1.15:
        tokens.append("round face")

    if p["jaw_width"] > 0.55:
        tokens.append("wide jaw")
    else:
        tokens.append("narrow jaw")

    return ", ".join(tokens)

# =====================
# CHARACTER GENERATION
# =====================
BASE_PROMPT = (
    "anime character illustration, "
    "2d animation style, "
    "clean lineart, "
    "flat cel shading, "
    "simple facial structure, "
    "front facing, "
    "neutral expression, "
    "single character portrait, "
    "one person only, "
    "same character identity, "
    "no accessories, "
    "no fantasy elements"
)

NEG_PROMPT = (
    "pokemon style, "
    "realistic photo, "
    "soft shading, "
    "airbrush, "
    "oil painting, "
    "blurry face, "
    "deformed face, "
    "chibi"
)

def human_to_character_style(image_path):
    img = cv2.imread(image_path)
    params = extract_face_params(img)
    param_prompt = face_params_to_prompt(params)

    prompt = BASE_PROMPT + ", " + param_prompt

    img = pipe(
        prompt=prompt,
        negative_prompt=NEG_PROMPT,
        guidance_scale=7.0,
        num_inference_steps=30,
        generator=GENERATOR   # 🔥 ID 고정
    ).images[0]

    save_path = os.path.join(OUT_DIR, "character_style_0.png")
    img.save(save_path)

    return [save_path]

# =====================
# CLI
# =====================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python human_to_character_style.py <image_path>")
        exit(1)

    human_to_character_style(sys.argv[1])
