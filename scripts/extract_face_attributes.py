# scripts/extract_face_attributes.py

import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

def extract_face_crop(image_path):
    img = cv2.imread(image_path)
    faces = app.get(img)

    if not faces:
        raise ValueError("No face detected")

    face = faces[0]
    x1, y1, x2, y2 = map(int, face.bbox)
    crop = img[y1:y2, x1:x2]
    return crop
