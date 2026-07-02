import asyncio
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from doc_helper.config.settings import Settings
from doc_helper.evaluation.generator_metrics import METRIC_NAMES, zero_scores

logger = logging.getLogger(__name__)


class GeneratorEvaluator:
    def __init__(self, settings: Settings | None = None):
        self._settings = settings or Settings()
        self._judge: BaseChatModel = self._create_judge_llm(self._settings)

    @staticmethod
    def _create_judge_llm(settings: Settings) -> BaseChatModel:
        from doc_helper.llm.factory import create_chat_model

        judge_cfg = settings.judge_llm
        has_judge_config = judge_cfg.provider is not None and judge_cfg.model is not None

        if has_judge_config:
            return create_chat_model(settings.judge_llm)
        return create_chat_model(settings.llm)

    async def evaluate(self, dataset: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        from doc_helper.evaluation.generator_dataset import load_generator_dataset

        if dataset is None:
            dataset = load_generator_dataset()

        if not dataset:
            logger.warning("Empty generator dataset. Returning empty report.")
            return {
                "total_queries": 0,
                "overall": {},
                "by_difficulty": {},
                "worst_queries": [],
                "per_query": [],
            }

        from doc_helper.agents.rag_agent import create_rag_agent

        agent = create_rag_agent(self._settings)

        judge_desc = self._judge_desc()
        logger.info(
            f"Running generator evaluation: {len(dataset)} queries, 4 metrics, judge: {judge_desc}"
        )

        per_query: list[dict[str, Any]] = []

        for i, item in enumerate(dataset):
            question = item["question"]
            reference_answer = item["reference_answer"]
            key_facts = item["key_facts"]
            difficulty = item.get("difficulty", "unknown")
            relevant_urls = item.get("relevant_urls", [])

            logger.info(f"[{i + 1}/{len(dataset)}] ({difficulty}) {question[:60]}...")

            try:
                result = agent.run(query=question)
                answer = result.get("answer", "")
                context_docs = result.get("context", [])
                sources = result.get("sources", [])
                contexts = [doc.page_content for doc in context_docs] if context_docs else []
            except Exception as e:
                logger.warning(f"Agent failed for query {i + 1}: {e}")
                answer = f"ERROR: {e}"
                contexts = []
                sources = []

            scores = await self._run_metrics(
                judge=self._judge,
                question=question,
                answer=answer,
                contexts=contexts,
                reference_answer=reference_answer,
                key_facts=key_facts,
            )

            total = sum(s["score"] for s in scores.values())

            per_query.append(
                {
                    "question": question,
                    "reference_answer": reference_answer,
                    "answer": answer,
                    "sources": sources,
                    "difficulty": difficulty,
                    "relevant_urls": relevant_urls,
                    "key_facts": key_facts,
                    "scores": scores,
                    "total_score": total,
                }
            )

        overall = self._aggregate_overall(per_query)
        by_difficulty = self._aggregate_by_difficulty(per_query)
        worst_queries = self._worst_queries(per_query, n=3)

        return {
            "total_queries": len(dataset),
            "judge_model": judge_desc,
            "overall": overall,
            "by_difficulty": by_difficulty,
            "worst_queries": worst_queries,
            "per_query": per_query,
        }

    @staticmethod
    async def _run_metrics(
        judge: BaseChatModel,
        question: str,
        answer: str,
        contexts: list[str],
        reference_answer: str,
        key_facts: list[str],
    ) -> dict[str, dict[str, float | str]]:
        from doc_helper.evaluation.generator_metrics import (
            answer_correctness,
            answer_relevancy,
            context_utilization,
            faithfulness,
        )

        if answer.startswith("ERROR"):
            return zero_scores()

        results = await asyncio.gather(
            faithfulness(judge, question, answer, contexts),
            answer_relevancy(judge, question, answer),
            answer_correctness(judge, question, answer, reference_answer, key_facts),
            context_utilization(judge, answer, contexts),
        )

        return {
            "faithfulness": results[0],
            "answer_relevancy": results[1],
            "answer_correctness": results[2],
            "context_utilization": results[3],
        }

    def _judge_desc(self) -> str:
        judge_cfg = self._settings.judge_llm
        if judge_cfg.provider and judge_cfg.model:
            return f"{judge_cfg.provider}/{judge_cfg.model}"
        return f"{self._settings.llm.provider}/{self._settings.llm.model}"

    @staticmethod
    def _aggregate_overall(per_query: list[dict[str, Any]]) -> dict[str, float]:
        if not per_query:
            return {name: 0.0 for name in METRIC_NAMES}
        result: dict[str, float] = {}
        for name in METRIC_NAMES:
            result[name] = sum(q["scores"][name]["score"] for q in per_query) / len(per_query)
        return result

    @staticmethod
    def _aggregate_by_difficulty(per_query: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        from collections import defaultdict

        by_diff: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for q in per_query:
            by_diff[q["difficulty"]].append(q)

        result: dict[str, dict[str, float]] = {}
        for diff, items in sorted(by_diff.items()):
            diff_scores: dict[str, float] = {}
            for name in METRIC_NAMES:
                diff_scores[name] = sum(q["scores"][name]["score"] for q in items) / len(items)
            result[diff] = diff_scores
        return result

    @staticmethod
    def _worst_queries(per_query: list[dict[str, Any]], n: int = 3) -> list[dict[str, Any]]:
        sorted_queries = sorted(per_query, key=lambda q: q["total_score"])
        worst = sorted_queries[:n]
        return [
            {
                "question": q["question"],
                "total_score": q["total_score"],
                "difficulty": q["difficulty"],
                "scores": q["scores"],
            }
            for q in worst
        ]
