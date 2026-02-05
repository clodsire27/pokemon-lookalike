import numpy as np
from collections import defaultdict
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

embs = np.load(os.path.join(BASE_DIR, "clip_pokemon_embeddings.npy"))
names = np.load(os.path.join(BASE_DIR, "clip_pokemon_names.npy"))

bucket = defaultdict(list)
for e, n in zip(embs, names):
    bucket[n].append(e)

mean_embs = []
mean_names = []

for name, vecs in bucket.items():
    v = np.mean(vecs, axis=0)
    v = v / np.linalg.norm(v)
    mean_embs.append(v)
    mean_names.append(name)

np.save(os.path.join(BASE_DIR, "clip_pokemon_mean_embeddings.npy"),
        np.stack(mean_embs))
np.save(os.path.join(BASE_DIR, "clip_pokemon_mean_names.npy"),
        np.array(mean_names))

print("✅ Pokémon mean CLIP embeddings built")
