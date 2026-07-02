import asyncio
import json
from pathlib import Path

import click

from doc_helper.config.settings import Settings
from doc_helper.evaluation.generator_dataset import load_generator_dataset
from doc_helper.evaluation.generator_evaluator import GeneratorEvaluator
from doc_helper.logger import log_header, log_info, log_success


def _print_report(report: dict) -> None:
    log_header("GENERATOR EVALUATION RESULTS")

    total = report.get("total_queries", 0)
    judge = report.get("judge_model", "unknown")
    log_info(f"{total} queries, 4 metrics, judge: {judge}")

    overall = report.get("overall", {})
    if overall:
        log_info("")
        log_info("Overall:")
        for name, value in overall.items():
            log_info(f"  {name:<22} {value:.4f}")

    by_difficulty = report.get("by_difficulty", {})
    if by_difficulty:
        log_info("")
        log_info("By difficulty:")
        for diff, scores in sorted(by_difficulty.items()):
            parts = "  ".join(
                f"{s['short']}={scores[name]:.2f}" for name, s in _METRIC_SHORT.items()
            )
            log_info(f"  {diff:<12} {parts}")

    worst = report.get("worst_queries", [])
    if worst:
        log_info("")
        log_info("Worst queries:")
        for i, q in enumerate(worst, 1):
            s = q["scores"]
            scores_str = (
                f"faith={s['faithfulness']['score']:.1f} "
                f"rel={s['answer_relevancy']['score']:.1f} "
                f"corr={s['answer_correctness']['score']:.1f} "
                f"util={s['context_utilization']['score']:.1f}"
            )
            q_text = q["question"][:50]
            log_info(f'  {i}. "{q_text}..."  total={q["total_score"]:.2f}/4.0  ({scores_str})')


_METRIC_SHORT = {
    "faithfulness": {"short": "faith"},
    "answer_relevancy": {"short": "rel"},
    "answer_correctness": {"short": "corr"},
    "context_utilization": {"short": "util"},
}


@click.command()
@click.option("--sample", default=None, type=int, help="Limit to N queries")
@click.option(
    "--output",
    default="evaluation/generator_results.json",
    help="Output file path",
)
def cli(sample, output):
    settings = Settings()
    dataset = load_generator_dataset()

    if sample is not None and sample > 0:
        dataset = dataset[:sample]

    log_info(f"Loaded {len(dataset)} generator eval queries")

    evaluator = GeneratorEvaluator(settings)
    report = asyncio.run(evaluator.evaluate(dataset))

    _print_report(report)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log_success(f"Results saved to {output_path}")


if __name__ == "__main__":
    cli()
