import json
from pathlib import Path

import click

from doc_helper.config.settings import Settings
from doc_helper.evaluation.retrieval_dataset import load_retrieval_dataset
from doc_helper.evaluation.retrieval_evaluator import RetrievalEvaluator
from doc_helper.logger import log_header, log_info, log_success


def _format_config_row(config: dict) -> str:
    st = config["search_type"]
    k = config["search_k"]
    r = "T" if config["reranker_enabled"] else "F"
    m = config["metrics"]
    return (
        f"{st:<14} k={k:<3} rr={r}   "
        f"{m['hit_rate']:.4f}   {m['mrr']:.4f}   "
        f"{m['precision_at_k']:.4f}   {m['recall_at_k']:.4f}"
    )


def _print_report(report: dict) -> None:
    log_header("RETRIEVAL EVALUATION RESULTS")

    configs = report.get("configs", [])
    if not configs:
        log_info("No configs evaluated.")
        return

    log_info(f"{'Config':<30} {'Hit':>8}   {'MRR':>8}   {'P@K':>8}   {'R@K':>8}")
    log_info("-" * 75)

    for config in configs:
        marker = ""
        best = report.get("best_config")
        if (
            best
            and config["search_type"] == best["search_type"]
            and config["search_k"] == best["search_k"]
            and config["reranker_enabled"] == best["reranker_enabled"]
        ):
            marker = "  <-- BEST"
        log_info(_format_config_row(config) + marker)

    best = report.get("best_config")
    if best:
        log_info("")
        log_info(
            f"Best: {best['search_type']}, k={best['search_k']}, "
            f"reranker={best['reranker_enabled']} "
            f"(hit_rate={best['value']:.4f})"
        )

        best_config_data = next(
            c
            for c in configs
            if c["search_type"] == best["search_type"]
            and c["search_k"] == best["search_k"]
            and c["reranker_enabled"] == best["reranker_enabled"]
        )
        by_diff = best_config_data.get("by_difficulty", {})
        if by_diff:
            log_info("")
            log_info("By difficulty (best config):")
            for diff, metrics in sorted(by_diff.items()):
                log_info(
                    f"  {diff:<12} "
                    f"hit={metrics['hit_rate']:.4f}  "
                    f"mrr={metrics['mrr']:.4f}  "
                    f"p@k={metrics['precision_at_k']:.4f}  "
                    f"r@k={metrics['recall_at_k']:.4f}"
                )


@click.command()
@click.option("--sample", default=None, type=int, help="Limit to N queries")
@click.option(
    "--output",
    default="evaluation/retrieval_results.json",
    help="Output file path",
)
def cli(sample, output):
    settings = Settings()
    dataset = load_retrieval_dataset()

    if sample is not None and sample > 0:
        dataset = dataset[:sample]

    log_info(f"Loaded {len(dataset)} retrieval eval queries")

    evaluator = RetrievalEvaluator(settings)
    report = evaluator.evaluate(dataset)

    _print_report(report)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log_success(f"Results saved to {output_path}")


if __name__ == "__main__":
    cli()
