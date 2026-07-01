import json
from pathlib import Path

DATASET_PATH = Path(__file__).parent / "retrieval_dataset.json"


def load_retrieval_dataset() -> list[dict]:
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError("Retrieval dataset must be a JSON array")

    for i, entry in enumerate(dataset):
        if not isinstance(entry, dict):
            raise ValueError(f"Entry {i} must be a JSON object")
        if "question" not in entry or not isinstance(entry["question"], str):
            raise ValueError(f"Entry {i} missing 'question' (str)")
        urls = entry.get("relevant_urls")
        if not isinstance(urls, list) or len(urls) == 0:
            raise ValueError(f"Entry {i} missing 'relevant_urls' (non-empty list)")
        if not all(isinstance(u, str) for u in urls):
            raise ValueError(f"Entry {i} 'relevant_urls' must be list[str]")
        if "difficulty" not in entry or not isinstance(entry["difficulty"], str):
            raise ValueError(f"Entry {i} missing 'difficulty' (str)")

    return dataset


def get_retrieval_questions_by_difficulty(difficulty: str) -> list[dict]:
    dataset = load_retrieval_dataset()
    return [item for item in dataset if item.get("difficulty") == difficulty]
