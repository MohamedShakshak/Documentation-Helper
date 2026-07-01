# Retrieval Evaluation Module — Design Spec

## Overview

Build a new retrieval evaluation system from scratch that evaluates the RAG retrieval pipeline against a dedicated URL-labeled dataset. The system sweeps across multiple retrieval configurations (search type, k, reranker) and reports core IR metrics (hit rate, MRR, precision@k, recall@k) to identify the best-performing configuration.

Replaces the existing `src/doc_helper/evaluation/retrieval_eval.py` — which uses naive substring matching and no config sweep.

## Decision Summary

| # | Decision | Choice |
|---|----------|--------|
| 1 | Ground truth source | New separate dataset with relevant URLs per query |
| 2 | Label format | URL-level matching (retrieved doc `source_url` vs `relevant_urls`) |
| 3 | Metrics | Hit Rate, MRR (via reciprocal rank), Precision@k, Recall@k |
| 4 | Config sweep | 12 configs: 2 search types x 3 k values x 2 reranker states |
| 5 | Architecture | Approach A — `RetrievalEvaluator` class with separated concerns |
| 6 | Module structure | 4 new files: dataset, metrics, evaluator, runner |
| 7 | Old `retrieval_eval.py` | Deleted entirely |

---

## 1. Dataset

### File

`src/doc_helper/evaluation/retrieval_dataset.json`

### Schema

```json
[
  {
    "question": "What is LangChain?",
    "relevant_urls": ["https://python.langchain.com/docs/concepts/"],
    "difficulty": "simple"
  },
  {
    "question": "How does MMR retrieval work?",
    "relevant_urls": [
      "https://python.langchain.com/docs/how_to/mmr/",
      "https://python.langchain.com/docs/concepts/retrieval/"
    ],
    "difficulty": "multi_hop"
  }
]
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `question` | `str` | The query string sent to the retriever |
| `relevant_urls` | `list[str]` | Source URLs that should be retrieved. A retrieved doc's `source_url` metadata must match one of these (after normalization) to count as a hit. |
| `difficulty` | `str` | `"simple"`, `"multi_hop"`, or `"edge_case"` — same levels as the RAGAS gold dataset |

### Size

~15-20 entries across the three difficulty levels, authored manually with real LangChain doc URLs. Multiple relevant URLs per query are allowed (enables recall@k to be meaningful).

### Loader

`src/doc_helper/evaluation/retrieval_dataset.py`

```python
def load_retrieval_dataset() -> list[dict]:
    ...

def get_retrieval_questions_by_difficulty(difficulty: str) -> list[dict]:
    ...
```

The loader validates that each entry has `question` (str), `relevant_urls` (non-empty list), and `difficulty` (str). Raises `ValueError` on malformed entries.

---

## 2. Metrics

### File

`src/doc_helper/evaluation/retrieval_metrics.py`

### Design

Pure functions. Each takes `(retrieved_urls: list[str], relevant_urls: list[str], k: int)` and returns a `float`. No I/O, no side effects, no dependencies on LangChain or the vector store. Fully unit-testable in isolation.

### URL Normalization

```python
def normalize_url(url: str) -> str:
    """Parse URL, strip trailing slash and fragment, lowercase the domain.
    Query params are preserved."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}{query}"
```

Query params are preserved (LangChain docs use them for some pages like `?from=quickstart`). Trailing slashes and fragments are stripped. Domain is lowercased; path is case-sensitive.

### Functions

```python
def hit_rate(retrieved_urls: list[str], relevant_urls: list[str], k: int) -> float:
    """1.0 if any relevant URL appears in the top-k retrieved, else 0.0.
    k is clamped to len(retrieved_urls). Empty retrieved returns 0.0."""

def reciprocal_rank(retrieved_urls: list[str], relevant_urls: list[str], k: int) -> float:
    """1/rank of first relevant URL in top-k, or 0.0 if none found.
    rank is 1-indexed. k is clamped to len(retrieved_urls)."""

def precision_at_k(retrieved_urls: list[str], relevant_urls: list[str], k: int) -> float:
    """Fraction of top-k retrieved that are relevant.
    Denominator is min(k, len(retrieved_urls)). Empty retrieved returns 0.0."""

def recall_at_k(retrieved_urls: list[str], relevant_urls: list[str], k: int) -> float:
    """Fraction of all relevant URLs that appear in top-k retrieved.
    Returns 0.0 if relevant_urls is empty.
    Numerator is count of distinct relevant URLs found in top-k."""
```

All functions call `normalize_url` on both `retrieved_urls` and `relevant_urls` before comparing.

### Aggregation

Aggregation lives in the evaluator, not in these functions. Per-query scores are computed by calling each metric with that query's retrieved + relevant URLs. The evaluator then computes:
- **Mean** across all queries for each metric
- **Mean** per difficulty level for each metric

---

## 3. Config Sweep

### Axes

| Axis | Values |
|------|--------|
| `search_type` | `"similarity"`, `"mmr"` |
| `search_k` | `4`, `8`, `16` |
| `reranker_enabled` | `false`, `true` |

Total: 2 x 3 x 2 = **12 configurations**.

Not swept: `score_threshold` (requires LLM-scored vectors and adds complexity without clear value for this eval). Kept at `.env` default.

### Sweep Process

For each config:
1. Build a `RetrievalSettings` instance with that combination (all other fields from `.env` defaults)
2. Create a `Retriever` using the shared vector store + that `RetrievalSettings`
3. Run all dataset queries through `Retriever.retrieve()`
4. Extract `source_url` from each retrieved doc's metadata
5. Compute all 4 metrics per query (using `k = config.search_k`)
6. Aggregate: mean across queries + mean per difficulty
7. Record results

### Reranker Availability

If `flashrank` is not importable, `_build_config_grid()` catches `ImportError`, logs a warning, and prunes all `reranker_enabled=True` configs from the sweep (6 configs remain). Runtime reranker failures (model download, OOM) are caught per-query — that query scores 0 on all metrics, error logged, sweep continues.

### Vector Store

The vector store is created once via `create_vector_store(settings)` and shared across all configs. Only `RetrievalSettings` changes per config.

---

## 4. Module Structure

```
src/doc_helper/evaluation/
├── __init__.py
├── dataset.py               # existing — loads RAGAS gold_dataset.json (unchanged)
├── retrieval_dataset.py     # NEW — loads retrieval_dataset.json, validates schema
├── retrieval_metrics.py     # NEW — hit_rate, reciprocal_rank, precision_at_k, recall_at_k, normalize_url
├── retrieval_evaluator.py   # NEW — RetrievalEvaluator: config sweep, orchestration, aggregation
├── retrieval_runner.py      # NEW — Click CLI, console table, JSON output
├── runner.py                 # existing — RAGAS runner (unchanged)
├── retrieval_eval.py         # DELETED
└── gold_dataset.json         # existing — RAGAS Q&A dataset (unchanged)
```

### `retrieval_dataset.py`

```python
DATASET_PATH = Path(__file__).parent / "retrieval_dataset.json"

def load_retrieval_dataset() -> list[dict]:
    """Load and validate the retrieval eval dataset.
    Each entry must have: question (str), relevant_urls (non-empty list), difficulty (str).
    Raises ValueError on malformed entries."""

def get_retrieval_questions_by_difficulty(difficulty: str) -> list[dict]:
    """Filter dataset by difficulty level."""
```

### `retrieval_metrics.py`

4 metric functions + `normalize_url` as described in Section 2. All pure, stateless, no imports beyond stdlib.

### `retrieval_evaluator.py`

```python
class RetrievalEvaluator:
    def __init__(self, settings: Settings | None = None):
        """Creates vector store once, shared across all configs."""

    def evaluate(self, dataset: list[dict] | None = None) -> dict:
        """Run all configs across the dataset. Returns full report dict.
        Picks best config by hit_rate."""

    def _build_config_grid(self) -> list[RetrievalSettings]:
        """Generate all RetrievalSettings combinations. Prunes reranker
        configs if flashrank not importable."""

    def _run_config(
        self, config: RetrievalSettings, dataset: list[dict]
    ) -> dict:
        """Build Retriever, run all queries, compute metrics.
        Returns {metrics: {...}, by_difficulty: {...}}."""
```

### `retrieval_runner.py`

Click-based CLI, consistent with existing `runner.py` and `pipeline.py` patterns.

```python
@click.command()
@click.option("--sample", default=None, type=int, help="Limit to N queries")
@click.option("--output", default="evaluation/retrieval_results.json", help="Output file")
def cli(sample, output):
    ...
```

Registered as `doc-helper-retrieval-eval` entrypoint in `pyproject.toml` `[project.scripts]`.

### Console Output

A summary table sorted by `hit_rate` descending, plus per-difficulty breakdown for the best config:

```
=== RETRIEVAL EVALUATION RESULTS ===

Config                              Hit Rate   MRR    P@K    R@K
similarity, k=4, reranker=False      0.65      0.42   0.53   0.71
similarity, k=8, reranker=False      0.70      0.45   0.40   0.78
mmr, k=8, reranker=True              0.85      0.60   0.55   0.89  <-- BEST
...

Best config: mmr, k=8, reranker=True (hit_rate=0.85)

By difficulty (best config):
  simple:     hit_rate=0.95, mrr=0.80, p@k=0.65, r@k=0.95
  multi_hop:  hit_rate=0.80, mrr=0.55, p@k=0.50, r@k=0.85
  edge_case:  hit_rate=0.80, mrr=0.58, p@k=0.48, r@k=0.80

Results saved to evaluation/retrieval_results.json
```

### JSON Report Structure

```json
{
  "total_queries": 15,
  "configs": [
    {
      "search_type": "similarity",
      "search_k": 4,
      "reranker_enabled": false,
      "metrics": {
        "hit_rate": 0.65,
        "mrr": 0.42,
        "precision_at_k": 0.53,
        "recall_at_k": 0.71
      },
      "by_difficulty": {
        "simple": {"hit_rate": 0.80, "mrr": 0.55, "precision_at_k": 0.65, "recall_at_k": 0.85},
        "multi_hop": {"hit_rate": 0.55, "mrr": 0.35, "precision_at_k": 0.45, "recall_at_k": 0.65},
        "edge_case": {"hit_rate": 0.60, "mrr": 0.38, "precision_at_k": 0.50, "recall_at_k": 0.63}
      }
    }
  ],
  "best_config": {
    "search_type": "mmr",
    "search_k": 8,
    "reranker_enabled": true,
    "metric": "hit_rate",
    "value": 0.85
  }
}
```

---

## 5. Error Handling & Edge Cases

### URL Normalization
- Trailing slashes stripped: `".../docs/"` and `".../docs"` normalize identically
- Fragments stripped: `".../docs#agents"` normalizes to `".../docs"`
- Query params preserved: `".../docs?from=quickstart"` unchanged
- Domain lowercased: `HTTPS://Python...` normalizes to `https://python...`
- Path is case-sensitive (not lowercased)

### Empty States
- `relevant_urls == []` for a query → `recall_at_k`=0.0, `hit_rate`=0.0, query counted in denominator
- `retrieved_urls == []` (store empty or query returns nothing) → all metrics 0.0 for that query
- Dataset has 0 entries → evaluator logs warning, returns empty report, exits without error

### Reranker Availability
- `flashrank` not installed → `_build_config_grid()` catches `ImportError`, logs warning, prunes all `reranker_enabled=True` configs (6 remain)
- Reranker fails at runtime → caught per-query, that query scores 0 on all metrics, error logged, sweep continues

### Vector Store
- Store connection fails → `RetrievalEvaluator.__init__` raises with clear message
- Store has 0 documents → all queries return empty results, all metrics 0.0, report includes `"store_empty": true`

### CLI
- `--sample 0` or negative → treated as "no limit" (uses full dataset)
- `--output` path with no parent directory → auto-created with `mkdir(parents=True)`
- Invalid `retrieval_dataset.json` → `ValueError` propagates from loader, CLI exits with clear message

---

## 6. Testing

### `tests/unit/test_retrieval_metrics.py`

Tests the 4 pure metric functions + `normalize_url`:

| Test | Input | Expected |
|------|-------|----------|
| hit_rate relevant at rank 1 | retrieved=[A,B,C], relevant=[A], k=3 | 1.0 |
| hit_rate relevant at rank 3 (within k) | retrieved=[X,Y,A], relevant=[A], k=3 | 1.0 |
| hit_rate relevant beyond k | retrieved=[A,B,C], relevant=[A], k=2 | 0.0 |
| hit_rate no relevant in retrieved | retrieved=[X,Y,Z], relevant=[A], k=3 | 0.0 |
| hit_rate empty retrieved | retrieved=[], relevant=[A], k=3 | 0.0 |
| reciprocal_rank relevant at rank 1 | retrieved=[A,B,C], relevant=[A], k=3 | 1.0 |
| reciprocal_rank relevant at rank 4 | retrieved=[X,Y,Z,A], relevant=[A], k=4 | 0.25 |
| reciprocal_rank no hit | retrieved=[X,Y,Z], relevant=[A], k=3 | 0.0 |
| precision_at_k all relevant | retrieved=[A,B], relevant=[A,B], k=2 | 1.0 |
| precision_at_k half relevant | retrieved=[A,X], relevant=[A,B], k=2 | 0.5 |
| precision_at_k k > len(retrieved) | retrieved=[A], relevant=[A,B], k=4 | 1.0 (denom=min(4,1)=1) |
| precision_at_k empty retrieved | retrieved=[], relevant=[A], k=3 | 0.0 |
| recall_at_k all relevant retrieved | retrieved=[A,B], relevant=[A,B], k=2 | 1.0 |
| recall_at_k partial relevant retrieved | retrieved=[A], relevant=[A,B], k=2 | 0.5 |
| recall_at_k empty relevant | retrieved=[A,B], relevant=[], k=2 | 0.0 |
| normalize_url strips trailing slash | `".../docs/"` | `".../docs"` |
| normalize_url strips fragment | `".../docs#agents"` | `".../docs"` |
| normalize_url preserves query params | `".../docs?from=x"` | `".../docs?from=x"` |
| normalize_url lowercases domain | `HTTPS://Python.LangChain.COM/Docs` | `https://python.langchain.com/Docs` |

### `tests/unit/test_retrieval_evaluator.py`

Tests the orchestrator with mocked `Retriever`/store:

| Test | What it checks |
|------|----------------|
| config grid generates 12 combos | 2 search_type x 3 k x 2 reranker = 12 RetrievalSettings |
| grid prunes reranker when flashrank missing | 6 configs (reranker=False only) |
| `_run_config` extracts `source_url` from retrieved docs | calls Retriever.retrieve, reads metadata |
| aggregation computes mean across queries | averages per-query scores correctly |
| aggregation computes per-difficulty mean | groups by difficulty field |
| empty dataset returns empty report | 0 queries, no crash, valid structure |
| best config selection by hit_rate | picks config with highest hit_rate |
| normalization applied before comparison | trailing-slash variants in retrieved vs relevant match |

No integration test with a real vector store — requires ingested docs, slow and flaky. Mocked Retriever covers the logic.

---

## 7. pyproject.toml Changes

Add CLI entrypoint:

```toml
[project.scripts]
doc-helper-ingest = "doc_helper.ingestion.pipeline:cli"
doc-helper-evaluate = "doc_helper.evaluation.runner:cli"
doc-helper-retrieval-eval = "doc_helper.evaluation.retrieval_runner:cli"  # NEW
```

No new dependencies required — all metrics use stdlib only.

---

## 8. Files Changed

| Action | File |
|--------|------|
| CREATE | `src/doc_helper/evaluation/retrieval_dataset.json` |
| CREATE | `src/doc_helper/evaluation/retrieval_dataset.py` |
| CREATE | `src/doc_helper/evaluation/retrieval_metrics.py` |
| CREATE | `src/doc_helper/evaluation/retrieval_evaluator.py` |
| CREATE | `src/doc_helper/evaluation/retrieval_runner.py` |
| CREATE | `tests/unit/test_retrieval_metrics.py` |
| CREATE | `tests/unit/test_retrieval_evaluator.py` |
| DELETE | `src/doc_helper/evaluation/retrieval_eval.py` |
| MODIFY | `pyproject.toml` (add entrypoint) |