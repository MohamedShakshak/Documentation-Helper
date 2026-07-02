# Generator Evaluation Design

**Date:** 2026-07-02
**Status:** Approved

## Overview

Rebuild the generator (end-to-end RAG) evaluation from scratch. Measures answer quality after generation — not just retrieval, but the full pipeline: question → RAG agent → answer → judge LLM scoring.

Replaces the existing `evaluation/runner.py` (broken RAGAS import due to langchain-community deprecation). Removes RAGAS dependency entirely.

## Architecture

```
src/doc_helper/evaluation/
├── generator_dataset.json    # 30 queries with rich annotations
├── generator_dataset.py       # Loader with schema validation
├── generator_metrics.py       # 4 LLM-as-judge metric functions (pure async)
├── generator_evaluator.py     # GeneratorEvaluator: agent → metrics → aggregation
├── generator_runner.py        # Click CLI with console table + JSON report
```

Same separation pattern as retrieval eval:
- **Metrics** are pure async functions — no I/O except judge LLM call, no side effects
- **Evaluator** orchestrates: runs RAG agent on each query, collects answers + contexts, runs metrics, aggregates by difficulty
- **Runner** is thin CLI — formats output, saves JSON

## Metrics

Four LLM-as-judge metrics, each scoring 0-1. Each function is async, takes the judge LLM and relevant data, returns `dict[str, float | str]` with `score` and `reason`.

### 1. Faithfulness (0-1)

Measures if the answer is grounded in retrieved contexts (anti-hallucination).

- **Input:** `question`, `answer`, `contexts`
- **Judge prompt:** Identify each factual claim in the answer. For each claim, check if it's supported by the retrieved contexts. Score = supported_claims / total_claims.
- **Output:** `{"score": 0.85, "reason": "3 of 4 claims supported, 1 hallucinated about..."}`

### 2. Answer Relevancy (0-1)

Measures if the answer directly addresses the question asked.

- **Input:** `question`, `answer`
- **Judge prompt:** Rate how relevant the answer is to the question. Does it directly address what was asked? Penalize tangential or evasive answers.
- **Output:** `{"score": 0.9, "reason": "Answer directly addresses the question..."}`

### 3. Answer Correctness (0-1)

Measures factual correctness against reference answer + key facts.

- **Input:** `question`, `answer`, `reference_answer`, `key_facts`
- **Judge prompt:** Compare the generated answer to the reference answer. Check if each key_fact is present and correctly stated. Score = key_facts_present / total_key_facts, adjusted for factual consistency.
- **Output:** `{"score": 0.75, "reason": "3 of 4 key facts found, missing..."}`

### 4. Context Utilization (0-1)

Measures if the answer actually uses information from retrieved contexts (vs answering from training data).

- **Input:** `answer`, `contexts`
- **Judge prompt:** Does the answer draw information from the retrieved contexts? Or does it ignore them and answer from training data? Score based on how much of the answer is derived from contexts.
- **Output:** `{"score": 0.8, "reason": "Answer draws heavily from contexts but adds external knowledge about..."}`

## Evaluator

```
GeneratorEvaluator
├── __init__(settings)     → creates RAG agent + judge LLM from settings
├── evaluate(dataset)      → main entry point
│   ├── for each query (sequential — avoids rate limits):
│   │   ├── agent.run(question) → answer, contexts, sources
│   │   ├── run 4 metrics in parallel (asyncio.gather) with judge LLM
│   │   └── collect per-query: scores + reasons + answer + sources
│   ├── aggregate overall averages
│   ├── aggregate by difficulty
│   └── identify worst 3 queries (lowest total score) for debugging
│
├── _run_metrics(query_answer, contexts, reference, key_facts)
│   └── asyncio.gather all 4 metrics in parallel
│
└── _aggregate(per_query_results) → dict with mean scores
```

**Key behaviors:**
- Metrics run in parallel per query (`asyncio.gather`)
- Agent runs sequentially (one at a time — avoids rate limits)
- Judge LLM uses `JudgeLLMSettings` → falls back to `LLMSettings` if judge not configured
- Failed queries (agent errors) get scored 0 on all metrics
- Report includes per-query breakdown with reasons + worst 3 queries

## Judge LLM Selection

Reuses the existing `JudgeLLMSettings` from config:

```
JUDGE_LLM__PROVIDER=openrouter  → ChatOpenAI with OpenRouter base_url
JUDGE_LLM__PROVIDER=gemini      → ChatGoogleGenerativeAI
JUDGE_LLM__PROVIDER=ollama      → ChatOllama (local, zero keys)
```

If `JudgeLLMSettings` has no provider set, falls back to `LLMSettings`.

## Dataset

30 queries, 10 per difficulty level. Rich annotations per entry:

```json
{
  "question": "What is LCEL?",
  "reference_answer": "LCEL is a declarative way...",
  "difficulty": "simple",
  "relevant_urls": ["https://docs.langchain.com/oss/python/langchain/concepts/lcel"],
  "key_facts": [
    "declarative composition",
    "pipe operator",
    "streaming support",
    "async support"
  ]
}
```

**Fields:**
- `question` (str) — the user query
- `reference_answer` (str) — gold answer for correctness comparison
- `difficulty` (str) — `simple`, `multi_hop`, or `edge_case`
- `relevant_urls` (list[str]) — actual store URLs (cross-reference with retrieval eval)
- `key_facts` (list[str]) — short factual claims a correct answer must contain

**Schema validation** in loader:
- `question` must be non-empty string
- `reference_answer` must be non-empty string
- `difficulty` must be one of `simple`, `multi_hop`, `edge_case`
- `relevant_urls` must be non-empty list of strings
- `key_facts` must be non-empty list of strings

## CLI

```bash
# Full eval (30 queries × 4 metrics)
uv run doc-helper-generator-eval

# Quick test
uv run doc-helper-generator-eval --sample 5

# Custom output
uv run doc-helper-generator-eval --output evaluation/my_gen_results.json
```

**Console output:**
```
GENERATOR EVALUATION RESULTS
30 queries, 4 metrics, judge: openrouter/nvidia/nemotron-3-ultra-550b

Overall:
  faithfulness        0.8500
  answer_relevancy    0.9200
  answer_correctness  0.7800
  context_utilization 0.8100

By difficulty:
  simple       faith=0.90  rel=0.95  corr=0.85  util=0.88
  multi_hop    faith=0.82  rel=0.88  corr=0.72  util=0.78
  edge_case    faith=0.83  rel=0.93  corr=0.77  util=0.77

Worst queries:
  1. "How does MMR work..."       total=2.10/4.0  (faith=0.5, rel=0.6, corr=0.5, util=0.5)
  2. "What is content_hash..."   total=2.30/4.0  ...
  3. "How does SSE streaming..."   total=2.45/4.0  ...

Results saved to evaluation/generator_results.json
```

**JSON report** includes:
- `total_queries`
- `judge_model` — provider/model used
- `overall` — mean scores for all 4 metrics
- `by_difficulty` — per-difficulty mean scores
- `worst_queries` — 3 lowest-scoring queries with reasons
- `per_query` — full breakdown: question, answer, reference_answer, sources, scores + reasons per metric

## Dependencies

- RAGAS removed entirely — removes fragile version dependency
- Judge LLM via LangChain chat models (already installed: `langchain-openai`, `langchain-google-genai`, `langchain-ollama`)
- No new dependencies

## Changes to existing files

- `pyproject.toml` — remove `ragas` dependency, add `doc-helper-generator-eval` entrypoint
- `evaluation/runner.py` — deleted (replaced by `generator_runner.py`)
- `evaluation/dataset.py` — updated to load new `generator_dataset.json` instead of `gold_dataset.json`
- `gold_dataset.json` — deleted (replaced by `generator_dataset.json`)
- Old `runner.py` and `gold_dataset.json` to be removed

## Testing

- `tests/unit/test_generator_metrics.py` — test each metric function with mock judge LLM
- `tests/unit/test_generator_evaluator.py` — test evaluator with mocked agent + judge

## Zero-config compatibility

- Works with zero API keys if Ollama is running (local judge LLM)
- Judge LLM falls back to main LLM settings if judge not configured
- No required env vars beyond what the RAG agent already needs