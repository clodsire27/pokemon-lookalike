# scripts/clip_multiview_encoder.py

import torch
import open_clip


def encode_multiview_clip(model, prompts: dict):
    device = next(model.parameters()).device

    weights = {
        "eye_focused": 1.0,
        "mouth_jaw_focused": 0.8,
        "proportion_focused": 0.8,
        "anchor_only": 2.0,  # 🔥 결정타
    }

    embs = []

    for key, text in prompts.items():
        tokens = open_clip.tokenize([text]).to(device)

        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        embs.append(emb[0] * weights.get(key, 1.0))

    final = torch.stack(embs).mean(dim=0)
    return final / final.norm()
