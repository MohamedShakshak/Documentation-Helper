import json
from pathlib import Path
from typing import Any

DATASET_PATH = Path(__file__).parent / "generator_dataset.json"

VALID_DIFFICULTIES = {"simple", "multi_hop", "edge_case"}


def load_generator_dataset() -> list[dict[str, Any]]:
    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset: list[dict[str, Any]] = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError("Generator dataset must be a JSON array")

    for i, entry in enumerate(dataset):
        if not isinstance(entry, dict):
            raise ValueError(f"Entry {i} must be a JSON object")

        q = entry.get("question")
        if not isinstance(q, str) or not q:
            raise ValueError(f"Entry {i} missing 'question' (non-empty str)")

        if (
            "reference_answer" not in entry
            or not isinstance(entry["reference_answer"], str)
            or not entry["reference_answer"]
        ):
            raise ValueError(f"Entry {i} missing 'reference_answer' (non-empty str)")

        if "difficulty" not in entry or entry["difficulty"] not in VALID_DIFFICULTIES:
            raise ValueError(f"Entry {i} 'difficulty' must be one of {VALID_DIFFICULTIES}")

        urls = entry.get("relevant_urls")
        if not isinstance(urls, list) or len(urls) == 0:
            raise ValueError(f"Entry {i} missing 'relevant_urls' (non-empty list)")
        if not all(isinstance(u, str) for u in urls):
            raise ValueError(f"Entry {i} 'relevant_urls' must be list[str]")

        facts = entry.get("key_facts")
        if not isinstance(facts, list) or len(facts) == 0:
            raise ValueError(f"Entry {i} missing 'key_facts' (non-empty list)")
        if not all(isinstance(f, str) for f in facts):
            raise ValueError(f"Entry {i} 'key_facts' must be list[str]")

    return dataset


def get_generator_questions_by_difficulty(difficulty: str) -> list[dict[str, Any]]:
    dataset = load_generator_dataset()
    return [item for item in dataset if item.get("difficulty") == difficulty]
