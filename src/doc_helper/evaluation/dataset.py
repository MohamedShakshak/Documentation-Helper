import json
from pathlib import Path

DATASET_PATH = Path(__file__).parent / "gold_dataset.json"


def load_gold_dataset() -> list[dict]:
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_questions_by_difficulty(difficulty: str) -> list[dict]:
    dataset = load_gold_dataset()
    return [item for item in dataset if item.get("difficulty") == difficulty]
