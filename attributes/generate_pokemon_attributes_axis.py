import os
import json
import time
from openai import OpenAI

# =========================
# Paths
# =========================
BASE_DIR = "/home/sea/project/pokemon"
MAP_PATH = f"{BASE_DIR}/scripts/pokemon_map.json"
OUT_PATH = f"{BASE_DIR}/attributes/pokemon_attributes_axis.json"

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# =========================
# OpenAI Client
# =========================
client = OpenAI()  # OPENAI_API_KEY 환경변수 사용

# =========================
# Prompts
# =========================
SYSTEM_PROMPT = """
You are an expert in facial morphology.

Map fictional characters into a numeric human facial feature space.
Do NOT average or neutralize features.
Exaggeration is allowed if the character is distinctive.
"""

USER_PROMPT = """
Represent this character as a human face using numeric values.

Return ONLY JSON in this exact format:

{
  "face_vector": [
    face_ratio,      // 0.8 (wide) ~ 1.3 (long)
    eye_spacing,     // 0.30 (close) ~ 0.55 (wide)
    mouth_width,     // 0.30 (small) ~ 0.55 (wide)
    yaw_energy,      // 0.0 ~ 1.0
    pitch_energy,    // 0.0 ~ 1.0
    roll_energy      // 0.0 ~ 1.0
  ]
}

Rules:
- Do NOT normalize toward average.
- Distinctive characters must have distinctive values.
- Return only JSON.
"""



# =========================
# Generate one Pokémon
# =========================
def generate_one(en_name: str):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Pokémon name: {en_name}\n\n{USER_PROMPT}",
            },
        ],
        temperature=0.5,
    )

    data = json.loads(resp.choices[0].message.content)

    # 필수 키 확인
    required = {"face_shape", "eyes", "mouth", "vibe"}
    if not required.issubset(data.keys()):
        raise ValueError(f"Missing keys: {data.keys()}")

    # 🔥 여기서 정규화
    data["face_shape"] = list(data["face_shape"])[:2]
    data["eyes"] = list(data["eyes"])[:3]
    data["mouth"] = list(data["mouth"])[:1]
    data["vibe"] = list(data["vibe"])[:1]

    return data

# =========================
# Main
# =========================
def main():
    with open(MAP_PATH, "r", encoding="utf-8") as f:
        pokemon_map = json.load(f)

    results = {}

    for kr_name, en_name in pokemon_map.items():
        print(f"[GEN] {kr_name} ({en_name})")
        try:
            results[kr_name] = generate_one(en_name)
        except Exception as e:
            print(f"  ❌ error: {e}")

        time.sleep(1.2)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Generated axis attributes: {len(results)} Pokémon")
    print(f"[SAVED] {OUT_PATH}")

if __name__ == "__main__":
    main()
