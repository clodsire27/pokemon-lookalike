# scripts/axis_utils.py
import numpy as np

def compress_axis_soft(axis: dict, temperature: float = 0.02) -> dict:
    """
    axis: {"eye_sharp":0.31,...} 형태의 raw score / prob
    return: softmax-like 확률 분포 (전체 합 1)
    """
    keys = list(axis.keys())
    vals = np.array([float(axis[k]) for k in keys], dtype=np.float32)

    # 안정화
    vals = vals - vals.max()
    exps = np.exp(vals / max(temperature, 1e-6))
    probs = exps / (exps.sum() + 1e-12)

    return {k: float(v) for k, v in zip(keys, probs)}


def filter_axis(axis: dict, keys_to_keep: list) -> dict:
    """
    axis에서 keys_to_keep에 포함된 키만 남김.
    """
    out = {}
    for k in keys_to_keep:
        if k in axis:
            out[k] = float(axis[k])
    return out
