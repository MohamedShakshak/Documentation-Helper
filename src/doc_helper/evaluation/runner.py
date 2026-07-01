import asyncio
import json
import logging
from pathlib import Path

import click

from doc_helper.config.settings import Settings
from doc_helper.logger import log_header, log_info, log_success, log_warning

logger = logging.getLogger(__name__)


async def run_evaluation(settings: Settings | None = None, sample_size: int | None = None) -> dict:
    if settings is None:
        settings = Settings()

    log_header("RAGAS EVALUATION")
    log_info("Loading gold dataset...")

    from doc_helper.evaluation.dataset import load_gold_dataset

    dataset = load_gold_dataset()
    if sample_size:
        dataset = dataset[:sample_size]
    log_info(f"Loaded {len(dataset)} Q&A pairs")

    log_info("Initializing RAG agent...")
    from doc_helper.agents.rag_agent import create_rag_agent

    agent = create_rag_agent(settings)

    results: list[dict] = []
    for i, item in enumerate(dataset):
        question = item["question"]
        expected_answer = item["answer"]
        difficulty = item.get("difficulty", "unknown")

        log_info(f"[{i + 1}/{len(dataset)}] ({difficulty}) {question[:60]}...")

        try:
            result = agent.run(query=question)
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            results.append(
                {
                    "question": question,
                    "expected_answer": expected_answer,
                    "answer": answer,
                    "sources": sources,
                    "difficulty": difficulty,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to evaluate question {i + 1}: {e}")
            results.append(
                {
                    "question": question,
                    "expected_answer": expected_answer,
                    "answer": f"ERROR: {e}",
                    "sources": [],
                    "difficulty": difficulty,
                }
            )

    log_info("Running RAGAS metrics...")

    try:
        metrics = await _run_ragas_metrics(results, settings)
    except Exception as e:
        log_warning(f"RAGAS metrics failed (are RAGAS deps installed?): {e}")
        metrics = {"error": str(e)}

    report = {"total_questions": len(dataset), "metrics": metrics, "results": results}

    output_path = Path("evaluation/results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log_success(f"Results saved to {output_path}")

    if "error" not in metrics:
        log_header("METRICS SUMMARY")
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                log_info(f"  {metric_name}: {value:.4f}")
            else:
                log_info(f"  {metric_name}: {value}")

    return report


async def _run_ragas_metrics(results: list[dict], settings: Settings) -> dict:
    from datasets import Dataset as HFDataset
    from ragas import evaluate
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    questions = [r["question"] for r in results]
    answers = [r["answer"] for r in results]
    expected = [r["expected_answer"] for r in results]
    contexts = [[r["expected_answer"]] for r in results]

    hf_dataset = HFDataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "reference": expected,
            "contexts": contexts,
        }
    )

    llm_judge = _get_judge_llm(settings)
    metrics = [
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
    ]

    scores = evaluate(
        dataset=hf_dataset,
        metrics=metrics,
        llm=llm_judge,
    )
    return scores.to_dict() if hasattr(scores, "to_dict") else dict(scores)


def _get_judge_llm(settings: Settings):
    from ragas.llms import LangchainLLMWrapper

    judge_cfg = settings.judge_llm
    provider = judge_cfg.provider or settings.llm.provider
    api_key = judge_cfg.openrouter_api_key or settings.llm.openrouter_api_key
    model = judge_cfg.model or settings.llm.model

    if provider == "openrouter" and api_key:
        from langchain_openai import ChatOpenAI

        judge = ChatOpenAI(
            model=model,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            temperature=0,
        )
    else:
        from langchain_ollama import ChatOllama

        judge = ChatOllama(
            model=model,
            base_url=settings.llm.ollama_base_url,
            temperature=0,
        )

    return LangchainLLMWrapper(langchain_llm=judge)


@click.command()
@click.option("--sample", default=None, type=int, help="Limit to N questions")
@click.option("--output", default="evaluation/results.json", help="Output file path")
def cli(sample, output):
    asyncio.run(run_evaluation(settings=Settings(), sample_size=sample))


if __name__ == "__main__":
    cli()
