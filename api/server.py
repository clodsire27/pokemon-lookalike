from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from scripts.match_from_face_clip_hog import run_match

BASE_DIR = "/home/sea/project/pokemon"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# static image serving (pokemon images)
app.mount(
    "/static",
    StaticFiles(directory=f"{BASE_DIR}/images"),
    name="static"
)

@app.get("/", response_class=HTMLResponse)
def index():
    with open(f"{BASE_DIR}/web/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/match")
async def match(file: UploadFile = File(...)):
    temp_path = os.path.join(BASE_DIR, "temp_upload.jpg")

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🔥 핵심: run_match는 JSON 반환
    return run_match(temp_path)
