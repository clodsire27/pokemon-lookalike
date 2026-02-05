import cv2, json, torch, open_clip
from pathlib import Path
from scripts.face_roi_selector import select_face_roi
from scripts.geometry_utils import *

FACE_TEXT = "cute cartoon character face with eyes and mouth"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", "openai")
    model = model.to(device).eval()

    text_emb = model.encode_text(
        open_clip.tokenize([FACE_TEXT]).to(device)
    )
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    text_emb = text_emb[0]

    root = Path("data/pokemon_original")
    out = {}

    for d in root.iterdir():
        if not d.is_dir():
            continue

        imgs = list(d.glob("*.jpg"))
        if not imgs:
            continue

        img = cv2.imread(str(imgs[0]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = extract_silhouette(gray)

        roi = select_face_roi(img, mask, model, preprocess, text_emb, device)
        if roi is None:
            continue

        x1,y1,x2,y2 = roi
        face = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        mask = extract_silhouette(gray)

        out[d.name] = {
            "face_aspect_ratio": face_aspect_ratio(mask),
            **dict(zip(
                ["eye_spacing_ratio","eye_height_ratio"],
                estimate_eye_features(gray)
            )),
            **dict(zip(
                ["mouth_width_ratio","jaw_width_ratio"],
                estimate_mouth_and_jaw(mask)
            ))
        }

    json.dump(out, open("data/pokemon_geometry_axis.json","w"), indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
