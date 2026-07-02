# Documentation Helper

A production-grade RAG (Retrieval-Augmented Generation) system for technical documentation, built with LangChain, FastAPI, Streamlit, and ChromaDB.

> Designed as a portfolio-grade engineering project — demonstrates RAG architecture, agent middleware, retrieval evaluation, LLM-as-judge evaluation, and production patterns.

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────────────────┐
│  Streamlit  │────▶│   FastAPI    │────▶│       RAG Agent             │
│  (UI)       │     │   (API)      │     │  ┌───────────────────────┐  │
└─────────────┘     └──────────────┘     │  │  Guardrails (input)   │  │
                                          │  │  Summarization        │  │
                     ┌──────────────┐     │  │  Tool Retry            │  │
                     │  Ingestion   │     │  │  Model Fallback        │  │
                     │  Pipeline    │     │  └───────────────────────┘  │
                     │  (CLI/API)   │     │  ┌───────────────────────┐  │
                     └──────┬───────┘     │  │  retrieve_context     │  │
                            │              │  │  web_search (Tavily)  │  │
                            ▼              │  │  check_links (httpx)  │  │
                     ┌──────────────┐     │  └───────────────────────┘  │
                     │  Vector Store│     └─────────────────────────────┘
                     │  (Chroma)    │              │
                     └──────────────┘              ▼
                                           ┌──────────────┐
                                           │  Telemetry   │
                                           │  LangSmith / │
                                           │  LangFuse    │
                                           └──────────────┘
```

### Layered Metadata Envelope

Every chunk stored in the vector store carries rich metadata for filtering, deduplication, and traceability:

| Field | Source | Description |
|-------|--------|-------------|
| `source_url` | Crawler | Original URL of the document |
| `source_type` | Crawler | `documentation` or `local_file` |
| `doc_title` | Crawler | Extracted from HTML `<title>`, markdown `#`, or URL slug |
| `parent_section` | MarkdownSplitter | Top-level `#` heading |
| `section_heading` | MarkdownSplitter | `##` or `###` sub-heading |
| `chunk_index` | MarkdownSplitter | Position within document |
| `total_chunks` | MarkdownSplitter | Total chunks for the document |
| `ingested_at` | Pipeline | UTC ISO timestamp |
| `content_hash` | Pipeline | SHA-256 fingerprint for deduplication |
| `word_count` | Pipeline | Approximate word count |

---

## Features

### Core RAG
- **Zero-config startup** — runs entirely local with Ollama, BGE embeddings, and ChromaDB. No API keys required.
- **Multi-provider LLMs** — Ollama (local), OpenRouter (cloud), Google Gemini
- **Multi-tool agent** — `retrieve_context`, `web_search` (Tavily), `check_links` (HTTP HEAD)
- **Agent middleware** — Guardrails, summarization, tool retry, model fallback (composable chain)
- **SSE streaming** — Typed server-sent events for real-time agent thought streaming
- **Conversation persistence** — SQLite with WAL mode, full CRUD for conversations and messages

### Ingestion
- **Sitemap crawler** — Discovers documentation URLs via `sitemap.xml`, fetches clean `.md` content
- **Tavily crawler** — AI-powered web search for ad-hoc ingestion
- **Recursive crawler** — LangChain `RecursiveUrlLoader` for legacy sites
- **Local file crawler** — Ingest markdown files from a local directory
- **Markdown-aware chunking** — Two-stage splitter (header hierarchy → recursive character) with section metadata
- **Deduplication** — SHA-256 content hashing skips duplicate chunks on re-ingestion

### Evaluation
- **Retrieval evaluation** — 25-query dataset, 12-config sweep (search type × k × reranker), IR metrics
- **Generator evaluation** — 29-query dataset with reference answers and key facts, 4 LLM-as-judge metrics via structured output (Pydantic-validated)

### Production
- **Observability** — LangSmith and LangFuse tracing (factory pattern, NoOp when disabled)
- **Docker** — Multi-profile compose (dev / full / production)

---

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai/) with a model pulled (e.g., `ollama pull qwen3.5:9b`)

### Setup

```bash
git clone https://github.com/MohamedShakshak/Documentation-Helper.git
cd Documentation-Helper

# Install dependencies
uv sync

# Copy and configure environment
cp .env.example .env
# Edit .env as needed — defaults work locally with Ollama

# Ingest documentation (sitemap crawler fetches 64 LangChain pages)
uv run doc-helper-ingest --crawler sitemap

# Run the API server
uv run uvicorn doc_helper.api.app:app --reload

# In another terminal, run Streamlit
uv run streamlit run src/doc_helper/ui/streamlit_app.py
```

### Ingest Documentation

```bash
# Sitemap crawler (recommended for docs.langchain.com)
uv run doc-helper-ingest --crawler sitemap

# Tavily crawler (requires TAVILY_API_KEY)
uv run doc-helper-ingest --crawler tavily --url https://docs.langchain.com/oss/python/langchain

# Local files
uv run doc-helper-ingest --crawler local

# Via API
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"crawler": "sitemap"}'
```

### Chat

```bash
# Sync
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is LangChain?"}'

# SSE streaming
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is LangChain?"}'
```

---

## Configuration

All configuration is via environment variables or `.env` file. Nested variables use `__` separator.

### Essential

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM__PROVIDER` | `ollama` | LLM provider (`ollama`, `openrouter`, `gemini`) |
| `LLM__MODEL` | `qwen3.5:9b` | Model name |
| `LLM__OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM__OPENROUTER_API_KEY` | — | Required for OpenRouter |
| `LLM__GEMINI_API_KEY` | — | Required for Gemini |
| `EMBEDDING__MODEL` | `bge-small` | Embedding model (`bge-small` or `bge-base`) |
| `INGESTION__CRAWLER` | `tavily` | Default crawler (`tavily`, `recursive`, `local`, `sitemap`) |
| `INGESTION__CRAWL_URL` | `https://docs.langchain.com/oss/python/langchain` | Base URL for sitemap/recursive crawlers |
| `VECTOR_STORE__PROVIDER` | `chroma` | Vector store (`chroma` or `pinecone`) |

### Retrieval

| Variable | Default | Description |
|----------|---------|-------------|
| `RETRIEVAL__SEARCH_TYPE` | `mmr` | Search type (`similarity`, `mmr`, `similarity_score_threshold`) |
| `RETRIEVAL__SEARCH_K` | `16` | Number of documents to retrieve |
| `RETRIEVAL__SCORE_THRESHOLD` | `0.5` | Minimum similarity score (for `similarity_score_threshold`) |
| `RETRIEVAL__RERANKER_ENABLED` | `false` | Enable FlashRank cross-encoder reranker |

### Judge LLM (Evaluation)

| Variable | Default | Description |
|----------|---------|-------------|
| `JUDGE_LLM__PROVIDER` | — | Judge provider (falls back to `LLM__PROVIDER` if unset) |
| `JUDGE_LLM__MODEL` | — | Judge model (falls back to `LLM__MODEL` if unset) |
| `JUDGE_LLM__TEMPERATURE` | `0.0` | Judge temperature (0 for deterministic scoring) |

### Agent Middleware

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT__GUARDRAILS_ENABLED` | `true` | Enable input validation |
| `AGENT__GUARDRAILS_MAX_INPUT_LENGTH` | `2000` | Max input characters |
| `AGENT__SUMMARIZATION_ENABLED` | `true` | Enable conversation summarization |
| `AGENT__SUMMARIZATION_THRESHOLD` | `20` | Messages before summarization triggers |
| `AGENT__MODEL_FALLBACK_ENABLED` | `true` | Enable model fallback on failure |
| `AGENT__FALLBACK_MODEL` | — | Fallback model name |
| `AGENT__MAX_TOOL_RETRIES` | `2` | Tool retry count |

### Observability

| Variable | Default | Description |
|----------|---------|-------------|
| `OBSERVABILITY__ENABLED` | `false` | Enable tracing |
| `OBSERVABILITY__PROVIDER` | `langfuse` | Tracer provider (`langfuse` or `langsmith`) |
| `OBSERVABILITY__LANGFUSE_PUBLIC_KEY` | — | LangFuse public key |
| `OBSERVABILITY__LANGFUSE_SECRET_KEY` | — | LangFuse secret key |
| `OBSERVABILITY__LANGSMITH_API_KEY` | — | LangSmith API key |
| `OBSERVABILITY__LANGSMITH_PROJECT` | `documentation-helper` | LangSmith project name |

See `.env.example` for the full list.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health check with config summary |
| `POST` | `/api/chat` | Sync chat (returns answer + sources) |
| `POST` | `/api/chat/stream` | SSE streaming chat |
| `POST` | `/api/chat/conversations` | Create conversation |
| `GET` | `/api/chat/conversations` | List conversations |
| `GET` | `/api/chat/conversations/{id}` | Get conversation messages |
| `DELETE` | `/api/chat/conversations/{id}` | Delete conversation |
| `POST` | `/api/ingest` | Start async ingestion task |
| `GET` | `/api/ingest/{id}/status` | Get task status |
| `GET` | `/api/ingest` | List ingestion tasks |

### SSE Event Types

| Event | Data | Description |
|-------|------|-------------|
| `agent_thought` | `{content}` | Agent's internal reasoning |
| `tool_call` | `{tool, query}` | Tool being invoked |
| `tool_result` | `{sources, num_docs}` | Tool execution result |
| `answer` | `{content}` | Answer token stream |
| `done` | `{conversation_id, sources}` | Stream complete |
| `error` | `{message}` | Error occurred |

---

## CLI Commands

```bash
# Ingest documentation
uv run doc-helper-ingest --crawler sitemap

# Run retrieval evaluation (12-config sweep, 25 queries)
uv run doc-helper-retrieval-eval

# Run generator evaluation (29 queries, 4 LLM-as-judge metrics)
uv run doc-helper-evaluate

# Start API server
uv run uvicorn doc_helper.api.app:app

# Start Streamlit UI
uv run streamlit run src/doc_helper/ui/streamlit_app.py
```

---

## Evaluation

### Retrieval Evaluation

Measures retrieval pipeline quality **before generation** — can the vector store surface the right documents?

**Dataset:** 25 queries across 3 difficulty levels (simple, multi_hop, edge_case), each labeled with relevant source URLs.

**Config sweep:** 2 search types × 3 k values × 2 reranker states = 12 configurations.

**Metrics:**

| Metric | What it measures |
|--------|-----------------|
| **Hit Rate** | Fraction of queries with at least one relevant doc in top-k |
| **MRR** (Mean Reciprocal Rank) | How high the first relevant doc ranks |
| **Precision@k** | Fraction of retrieved docs that are relevant |
| **Recall@k** | Fraction of relevant docs that were retrieved |

**Full results (25 queries, BGE-base embeddings, 2,356 chunks across 64 pages):**

| Search Type | k | Reranker | Hit Rate | MRR | P@K | R@K |
|-------------|---|----------|----------|-----|-----|-----|
| **mmr** | **16** | **No** | **0.8800** | **0.4029** | **0.1800** | **0.7000** |
| mmr | 16 | Yes | 0.8800 | 0.3850 | 0.1800 | 0.7000 |
| similarity | 16 | No | 0.8400 | 0.4083 | 0.1925 | 0.6600 |
| similarity | 16 | Yes | 0.8400 | 0.3642 | 0.1925 | 0.6600 |
| similarity | 8 | No | 0.7200 | 0.3991 | 0.2350 | 0.5800 |
| similarity | 8 | Yes | 0.7200 | 0.4098 | 0.2350 | 0.5800 |
| mmr | 8 | No | 0.7200 | 0.4127 | 0.1800 | 0.5800 |
| mmr | 8 | Yes | 0.7200 | 0.3940 | 0.1800 | 0.5800 |
| mmr | 4 | No | 0.5200 | 0.3833 | 0.1900 | 0.4000 |
| mmr | 4 | Yes | 0.5200 | 0.3533 | 0.1900 | 0.4000 |
| similarity | 4 | No | 0.4800 | 0.3600 | 0.2000 | 0.4000 |
| similarity | 4 | Yes | 0.4800 | 0.3633 | 0.2000 | 0.4000 |

**Best config: MMR, k=16, no reranker** — 88% hit rate, 70% recall.

**By difficulty (best config):**

| Difficulty | Hit Rate | MRR | P@K | R@K |
|------------|----------|-----|-----|-----|
| simple | 0.8000 | 0.3987 | 0.1313 | 0.6000 |
| multi_hop | 1.0000 | 0.4577 | 0.1812 | 0.8000 |
| edge_case | 0.8000 | 0.3019 | 0.2750 | 0.7000 |

**Key findings:**

- **MMR dominates similarity** — consistently higher hit rate across all k values, especially at k=16
- **k=16 is the sweet spot** — k=4 loses too much recall, k=8 misses relevant docs in the larger store
- **Reranker does not improve hit rate** — it reorders results but cannot find new documents. Slightly hurts MMR by disturbing the diversity ordering.
- **Multi-hop queries retrieve best** — specific, multi-concept queries surface relevant docs effectively

```bash
# Run full retrieval evaluation (25 queries, 12 configs)
uv run doc-helper-retrieval-eval

# Quick test with 5 queries
uv run doc-helper-retrieval-eval --sample 5

# Custom output path
uv run doc-helper-retrieval-eval --output evaluation/my_results.json
```

Results saved to `evaluation/retrieval_results.json`.

---

### Generator Evaluation

Measures end-to-end answer quality — does the RAG agent generate faithful, correct answers from retrieved context?

**Dataset:** 29 queries across 3 difficulty levels, each with a reference answer, key facts, and relevant URLs.

**Judge:** LLM-as-judge using structured output (Pydantic-validated responses via `with_structured_output`). Runs 4 metrics in parallel per query.

**Metrics:**

| Metric | What it measures |
|--------|-----------------|
| **Faithfulness** | Is the answer grounded in retrieved context? (no hallucination) |
| **Answer Relevancy** | Does the answer directly address the question? |
| **Answer Correctness** | Does the answer match reference answer and key facts? |
| **Context Utilization** | Does the answer use retrieved context or rely on training data? |

**Results (29 queries, judge: Gemini/gemma-4-31b-it):**

| Metric | Score |
|--------|-------|
| Faithfulness | 0.69 |
| Answer Relevancy | 1.00 |
| Answer Correctness | 0.76 |
| Context Utilization | 0.71 |

**By difficulty:**

| Difficulty | Faithfulness | Relevancy | Correctness | Utilization |
|------------|-------------|-----------|-------------|-------------|
| simple | 0.50 | 1.00 | 0.82 | 0.53 |
| multi_hop | 0.73 | 1.00 | 0.82 | 0.76 |
| edge_case | 0.88 | 1.00 | 0.59 | 0.88 |

**Key findings:**

- **Answer relevancy is perfect** — the agent always addresses the question asked
- **Edge cases score highest on faithfulness** — specific queries retrieve precise context, reducing hallucination
- **Simple queries score lowest** — broad queries (e.g., "What is LCEL?") reference concepts absent from the new docs.langchain.com, causing the agent to hallucinate from training data
- **Context utilization tracks faithfulness** — when relevant context is retrieved, the agent uses it

```bash
# Run full generator evaluation (29 queries, 4 metrics)
uv run doc-helper-evaluate

# Quick test with 5 queries
uv run doc-helper-evaluate --sample 5

# Custom output path
uv run doc-helper-evaluate --output evaluation/my_results.json
```

Results saved to `evaluation/generator_results.json`.

---

## Project Structure

```
src/doc_helper/
├── agents/           # RAG agent, tools, middleware, SSE events
├── api/              # FastAPI server, routes, dependency injection
├── config/           # Pydantic settings (nested, env-file based)
├── db/               # SQLite connection, migrations, conversation/task managers
├── embeddings/       # BGE-small and BGE-base, factory pattern
├── evaluation/       # Retrieval eval (IR metrics), generator eval (LLM-as-judge), datasets
├── ingestion/        # Crawlers (sitemap, Tavily, recursive, local), splitters, pipeline
├── llm/              # Ollama, OpenRouter, Gemini — factory pattern
├── observability/    # LangFuse, LangSmith, NoOp tracer — factory pattern
├── retrieval/        # Retriever with similarity/MMR/score-threshold + FlashRank reranker
├── stores/           # Chroma and Pinecone vector stores — factory pattern
├── ui/               # Streamlit frontend (thin client calling API)
├── utils.py          # Title extraction helpers
└── logger.py         # Structured logging helpers

tests/
├── unit/             # 234 unit tests (mocked, fast)
└── integration/      # API integration tests with TestClient
```

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **LLM** | Ollama (local), OpenRouter (cloud), Google Gemini |
| **Embeddings** | BGE-small (384d), BGE-base (768d) via sentence-transformers |
| **Vector Store** | ChromaDB (default), Pinecone |
| **Framework** | LangChain, `create_agent` with composable middleware |
| **API** | FastAPI, SSE via sse-starlette |
| **Frontend** | Streamlit (thin client) |
| **Database** | SQLite (WAL mode), aiosqlite |
| **Observability** | LangSmith, LangFuse |
| **Evaluation** | Custom LLM-as-judge with Pydantic structured output, IR metrics (no RAGAS dependency) |
| **Search** | Similarity, MMR, score-threshold, FlashRank reranker |
| **Ingestion** | Sitemap crawler, Tavily, recursive URL, local file |
| **Package** | `uv`, `hatchling` (src layout) |

---

## License

MIT
