import torch
import open_clip


def encode_multiview_clip(model, prompts: dict) -> torch.Tensor:
    """
    Encode multi-view CLIP prompts with anchor-weighted aggregation.

    prompts: dict[str, str]
        keys must include some of:
        - eye_focused
        - mouth_jaw_focused
        - proportion_focused
        - anchor_only

    return: torch.Tensor (D,)  # normalized
    """

    device = next(model.parameters()).device

    # 🔥 핵심: view별 가중치
    VIEW_WEIGHTS = {
        "eye_focused": 1.0,
        "mouth_jaw_focused": 1.0,
        "proportion_focused": 1.0,
        "anchor_only": 1.8,   # ✅ 구조 분기 핵심
    }

    embs = []
    weights = []

    for view_name, prompt in prompts.items():
        if not prompt.strip():
            continue

        w = VIEW_WEIGHTS.get(view_name, 1.0)

        tokens = open_clip.tokenize([prompt]).to(device)
        with torch.no_grad():
            e = model.encode_text(tokens)
            e = e / e.norm(dim=-1, keepdim=True)

        embs.append(e[0] * w)
        weights.append(w)

    if not embs:
        raise ValueError("No valid prompts provided to encode_multiview_clip")

    # 가중 평균
    emb = torch.stack(embs).sum(dim=0) / (sum(weights) + 1e-8)
    emb = emb / emb.norm()

    return emb
