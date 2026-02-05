![영케나비](github/영케나비.png)


# 🐾 Pokémon Lookalike Project

A research-oriented fan project that analyzes a human face image and finds the most visually similar Pokémon using **CLIP-based multi-view embeddings**, **axis-based facial attributes**, and **hybrid text–image similarity matching**.

This project emphasizes **interpretable facial feature modeling** rather than simple end-to-end classification.

---

## Overview

Given a human face image, the system performs the following steps:

1. Detects and crops the face  
2. Extracts facial attributes (eyes, face shape, overall vibe)  
3. Converts attributes into structured semantic schemas  
4. Matches the result against a precomputed Pokémon embedding database  
5. Returns the Top-K most similar Pokémon with interpretable explanations  

---

## Core Ideas

- **Axis-based face representation**  
  (eye shape, face proportion, facial vibe, etc.)

- **Winner-take-most soft compression**  
  for stable and dominant attribute selection

- **Multi-view CLIP prompting**  
  (eye-focused / proportion-focused / anchor-only views)

- **Text–Image hybrid embedding fusion**  
  for robust similarity matching

- **Explicit interpretability**  
  (explaining *why* a Pokémon matches)

---

## Tech Stack

- **Backend**: FastAPI  
- **Face Detection**: InsightFace (`buffalo_l`)  
- **Vision-Language Model**: OpenCLIP (ViT-B/32)  
- **Feature Engineering**:
  - CLIP-based semantic similarity
  - HOG-based geometric cues
- **Embedding Fusion**: Weighted text + image similarity  
- **Frameworks**: PyTorch, NumPy, OpenCV  

---

## Project Structure

```bash
pokemon-lookalike/
├── app.py                        # FastAPI entry point
├── scripts/
│   ├── extract_human_axis_clip.py
│   ├── build_human_schema_from_axis.py
│   ├── schema_to_multiview_prompts.py
│   ├── encode_multiview_clip.py
│   ├── match_human_to_pokemon_multiview.py
│   └── ...
├── checkpoints/                  # model weights (optional)
├── schemas/                      # JSON schema validation
├── .gitignore
├── README.md
└── requirements.txt
⚠️ Note
data/ and static/ directories are intentionally excluded from version control
(they contain large assets and generated resources).
