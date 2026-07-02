import json
from pathlib import Path
from typing import Any

DATASET_PATH = Path(__file__).parent / "generator_dataset.json"


def load_gold_dataset() -> list[dict[str, Any]]:
    with open(DATASET_PATH, encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)
    return data


def get_questions_by_difficulty(difficulty: str) -> list[dict[str, Any]]:
    dataset = load_gold_dataset()
    return [item for item in dataset if item.get("difficulty") == difficulty]
