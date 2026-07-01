import argparse
import json
from pathlib import Path

from doc_helper.config.settings import Settings
from doc_helper.evaluation.dataset import load_gold_dataset
from doc_helper.logger import log_header, log_info, log_success
from doc_helper.retrieval.retriever import Retriever
from doc_helper.stores.factory import create_vector_store


def evaluate_retrieval(sample_size: int | None = None) -> dict:
    settings = Settings()
    store = create_vector_store(settings)
    retriever = Retriever(store, settings.retrieval)

    dataset = load_gold_dataset()
    if sample_size:
        dataset = dataset[:sample_size]

    log_header("RETRIEVAL EVALUATION")
    log_info(f"Evaluating {len(dataset)} queries...")

    total_hits = 0
    total_queries = len(dataset)
    reciprocal_ranks: list[float] = []
    results: list[dict] = []

    for i, item in enumerate(dataset):
        question = item["question"]
        expected = item["answer"]
        difficulty = item.get("difficulty", "unknown")

        docs = retriever.retrieve(question)
        retrieved_texts = [d.page_content[:200] for d in docs]

        hit = any(expected[:50] in r for r in retrieved_texts)
        if hit:
            total_hits += 1
            for rank, r in enumerate(retrieved_texts, 1):
                if expected[:50] in r:
                    reciprocal_ranks.append(1.0 / rank)
                    break

        results.append({
            "question": question,
            "difficulty": difficulty,
            "hit": hit,
            "num_retrieved": len(docs),
            "retrieved_snippets": retrieved_texts,
        })

        log_info(
            f"[{i+1}/{total_queries}] {'HIT' if hit else 'MISS'}: "
            f"{question[:50]}..."
        )

    hit_rate = total_hits / total_queries
    mrr = sum(reciprocal_ranks) / total_queries if reciprocal_ranks else 0.0

    report = {
        "total_queries": total_queries,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "results": results,
    }

    output_path = Path("evaluation/retrieval_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log_header("RETRIEVAL RESULTS")
    log_info(f"  Hit Rate: {hit_rate:.2%}")
    log_info(f"  MRR:      {mrr:.4f}")
    log_success(f"Saved to {output_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    evaluate_retrieval(sample_size=args.sample)
