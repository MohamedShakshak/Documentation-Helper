import logging
from collections import defaultdict
from itertools import product
from typing import Any, Literal

from doc_helper.config.settings import RetrievalSettings, Settings
from doc_helper.evaluation.retrieval_metrics import (
    hit_rate,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from doc_helper.retrieval.retriever import Retriever
from doc_helper.stores.base import BaseVectorStore
from doc_helper.stores.factory import create_vector_store

logger = logging.getLogger(__name__)

METRIC_NAMES = ["hit_rate", "mrr", "precision_at_k", "recall_at_k"]


class RetrievalEvaluator:
    def __init__(self, settings: Settings | None = None):
        self._settings = settings or Settings()
        self._store: BaseVectorStore = create_vector_store(self._settings)

    def evaluate(self, dataset: list[dict] | None = None) -> dict[str, Any]:
        from doc_helper.evaluation.retrieval_dataset import load_retrieval_dataset

        if dataset is None:
            dataset = load_retrieval_dataset()

        if not dataset:
            logger.warning("Empty retrieval dataset. Returning empty report.")
            return {"total_queries": 0, "configs": [], "best_config": None}

        config_grid = self._build_config_grid()
        logger.info(f"Running {len(config_grid)} configs across {len(dataset)} queries")

        config_results: list[dict] = []
        for i, config in enumerate(config_grid, start=1):
            logger.info(
                f"Config {i}/{len(config_grid)}: "
                f"search_type={config.search_type}, k={config.search_k}, "
                f"reranker={config.reranker_enabled}"
            )
            result = self._run_config(config, dataset)
            config_entry = {
                "search_type": config.search_type,
                "search_k": config.search_k,
                "reranker_enabled": config.reranker_enabled,
                "metrics": result["metrics"],
                "by_difficulty": result["by_difficulty"],
            }
            config_results.append(config_entry)

        config_results.sort(key=lambda c: c["metrics"]["hit_rate"], reverse=True)
        best = config_results[0] if config_results else None
        best_config: dict | None = None
        if best:
            best_config = {
                "search_type": best["search_type"],
                "search_k": best["search_k"],
                "reranker_enabled": best["reranker_enabled"],
                "metric": "hit_rate",
                "value": best["metrics"]["hit_rate"],
            }

        return {
            "total_queries": len(dataset),
            "configs": config_results,
            "best_config": best_config,
        }

    def _build_config_grid(self) -> list[RetrievalSettings]:
        search_types = ["similarity", "mmr"]
        k_values = [4, 8, 16]
        reranker_options = [False, True]

        reranker_available = True
        try:
            import flashrank  # noqa: F401
        except ImportError:
            reranker_available = False
            logger.warning("flashrank not installed — skipping reranker=True configs")

        if not reranker_available:
            reranker_options = [False]

        grid: list[RetrievalSettings] = []
        base = self._settings.retrieval
        for search_type, k, reranker in product(search_types, k_values, reranker_options):
            config = RetrievalSettings(
                search_type=search_type,
                search_k=k,
                score_threshold=base.score_threshold,
                reranker_enabled=reranker,
            )
            grid.append(config)
        return grid

    def _run_config(self, config: RetrievalSettings, dataset: list[dict]) -> dict[str, Any]:
        retriever = Retriever(self._store, config)
        k = config.search_k

        per_query_scores: list[dict[str, float]] = []
        per_difficulty_scores: dict[str, list[dict[str, float]]] = defaultdict(list)

        for item in dataset:
            question = item["question"]
            relevant_urls = item["relevant_urls"]
            difficulty = item.get("difficulty", "unknown")

            try:
                docs = retriever.retrieve(question)
                retrieved_urls = [
                    doc.metadata.get("source_url", doc.metadata.get("source", "")) for doc in docs
                ]
            except Exception as e:
                logger.warning(f"Retrieval failed for '{question[:50]}': {e}")
                retrieved_urls = []

            scores = {
                "hit_rate": hit_rate(retrieved_urls, relevant_urls, k),
                "mrr": reciprocal_rank(retrieved_urls, relevant_urls, k),
                "precision_at_k": precision_at_k(retrieved_urls, relevant_urls, k),
                "recall_at_k": recall_at_k(retrieved_urls, relevant_urls, k),
            }
            per_query_scores.append(scores)
            per_difficulty_scores[difficulty].append(scores)

        metrics = self._aggregate(per_query_scores)
        by_difficulty = {
            diff: self._aggregate(scores) for diff, scores in per_difficulty_scores.items()
        }

        return {"metrics": metrics, "by_difficulty": by_difficulty}

    @staticmethod
    def _aggregate(scores: list[dict[str, float]]) -> dict[str, float]:
        if not scores:
            return {name: 0.0 for name in METRIC_NAMES}
        result: dict[str, float] = {}
        for name in METRIC_NAMES:
            result[name] = sum(s[name] for s in scores) / len(scores)
        return result
