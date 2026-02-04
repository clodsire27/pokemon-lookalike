import json
from jsonschema import validate, ValidationError

SCHEMA_PATH = "schemas/face_archetype.schema.json"


class FaceArchetypeValidator:
    def __init__(self, schema_path=SCHEMA_PATH):
        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema = json.load(f)

    def validate_schema(self, data: dict):
        try:
            validate(instance=data, schema=self.schema)
        except ValidationError as e:
            raise ValueError(f"[SchemaError] {e.message}")

    def validate_rules(self, data: dict):
        # Rule 1: Anchor 최소 2개
        if len(data["anchors"]) < 2:
            raise ValueError("[RuleError] anchors must be >= 2")

        # Rule 2: Eye system에서 최소 4개 이상 차별화
        eye = data["eyes"]
        distinct = {
            eye["eye_size"],
            eye["eye_shape"],
            eye["eye_tilt"],
            eye["eye_spacing"],
            eye["eye_height"]
        }
        if len(distinct) < 4:
            raise ValueError(
                "[RuleError] Eye system too generic (need >=4 distinct traits)"
            )

    def validate_all(self, data: dict):
        self.validate_schema(data)
        self.validate_rules(data)
        return True
