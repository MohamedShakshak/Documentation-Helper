# Documentation Helper

A production-grade RAG (Retrieval-Augmented Generation) system for technical documentation, built with LangChain, FastAPI, Streamlit, and friends.

> Built for the AI/LLM Engineer portfolio — focuses on RAG patterns, agent middleware, evaluation, prompt engineering, and production-grade architecture.

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
                           │              │  │  web_search (Tavily) │  │
                           ▼              │  │  check_links (httpx) │  │
                    ┌──────────────┐     │  └───────────────────────┘  │
                    │  Vector Store│     └─────────────────────────────┘
                    │  (Chroma)    │              │
                    └──────────────┘              ▼
                                          ┌──────────────┐
                                          │  Telemetry   │
                                          │  LangFuse/   │
                                          │  LangSmith   │
                                          └──────────────┘
```

### Layered Metadata Envelope

Every chunk carries this metadata:

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

- **Zero API keys by default** — runs entirely local with Ollama + BGE embeddings + Chroma
- **Cloud LLMs** — OpenRouter support for GPT-4o, Claude, etc.
- **Multi-tool agent** — `retrieve_context`, `web_search` (Tavily), `check_links` (HTTP HEAD)
- **Agent middleware** — Guardrails (input validation), summarization (context compression), tool retry, model fallback
- **Markdown-aware chunking** — Two-stage splitter (header hierarchy → recursive character)
- **Deduplication** — SHA-256 content hashing skips duplicate chunks on re-ingestion
- **SSE streaming** — Typed server-sent events (`agent_thought`, `tool_call`, `tool_result`, `answer`, `done`, `error`)
- **Conversation persistence** — SQLite with CRUD for conversations and messages
- **Observability** — LangFuse (default) and LangSmith tracing (factory pattern, NoOp when disabled)
- **Retrieval evaluation** — 25-query URL-labeled dataset, 12-config sweep (search type × k × reranker), IR metrics (hit rate, MRR, precision@k, recall@k)
- **RAGAS evaluation** — 30 Q&A gold dataset (3 difficulty levels), CLI runner, OpenRouter judge
- **Ingestion pipeline** — Tavily, recursive URL, and local file crawlers; CLI and async API
- **Docker** — Multi-profile compose (dev/full/production)

---

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai/) with a model (e.g., `ollama pull qwen3.5:9b`)

### Setup

```bash
git clone https://github.com/MohamedShakshak/Documentation-Helper.git
cd Documentation-Helper

# Create venv and install deps
uv sync

# Copy and configure environment
cp .env.example .env
# Edit .env as needed (defaults work locally with Ollama)

# Run the API server
uv run uvicorn doc_helper.api.app:app --reload

# In another terminal, run Streamlit
uv run streamlit run src/doc_helper/ui/streamlit_app.py
```

### Ingest Documentation

```bash
# Via CLI
uv run python -m doc_helper.ingestion.pipeline --crawler recursive --url https://python.langchain.com/ --depth 1

# Via API
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"crawler": "recursive", "url": "https://python.langchain.com/", "depth": 1}'
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
| `LLM__PROVIDER` | `ollama` | LLM provider (`ollama` or `openrouter`) |
| `LLM__MODEL` | `qwen3.5:9b` | Model name |
| `LLM__OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM__OPENROUTER_API_KEY` | — | Required for OpenRouter |
| `EMBEDDING__MODEL` | `bge-small` | Embedding model (`bge-small` or `bge-base`) |
| `INGESTION__CRAWLER` | `tavily` | Default crawler type |
| `VECTOR_STORE__PROVIDER` | `chroma` | Vector store (`chroma` or `pinecone`) |

### Agent Middleware

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT__GUARDRAILS_ENABLED` | `true` | Enable input validation |
| `AGENT__GUARDRAILS_MAX_INPUT_LENGTH` | `2000` | Max input characters |
| `AGENT__SUMMARIZATION_ENABLED` | `true` | Enable conversation summarization |
| `AGENT__SUMMARIZATION_THRESHOLD` | `20` | Messages before summarization |
| `AGENT__MODEL_FALLBACK_ENABLED` | `true` | Enable model fallback |
| `AGENT__FALLBACK_MODEL` | — | Fallback model name |
| `AGENT__MAX_TOOL_RETRIES` | `2` | Tool retry count |

### Observability

| Variable | Default | Description |
|----------|---------|-------------|
| `OBSERVABILITY__ENABLED` | `false` | Enable tracing |
| `OBSERVABILITY__PROVIDER` | `langfuse` | Tracer provider (`langfuse` or `langsmith`) |
| `OBSERVABILITY__LANGFUSE_PUBLIC_KEY` | — | LangFuse public key |
| `OBSERVABILITY__LANGFUSE_SECRET_KEY` | — | LangFuse secret key |

See `.env.example` for the full list.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health check with config |
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
uv run python -m doc_helper.ingestion.pipeline --url <URL> --depth 2 --crawler recursive

# Run retrieval evaluation (12-config sweep)
uv run doc-helper-retrieval-eval

# Run RAGAS evaluation
uv run python -m doc_helper.evaluation.runner --sample 10

# Start API server
uv run uvicorn doc_helper.api.app:app

# Start Streamlit UI
uv run streamlit run src/doc_helper/ui/streamlit_app.py
```

---

## Docker

```bash
# Dev profile (API only)
docker compose --profile dev up

# Full stack (API + Streamlit + LangFuse + Postgres)
docker compose --profile full up

# Production (API only with external deps)
docker compose --profile production up
```

---

## Evaluation

### Retrieval Evaluation

Measures the retrieval pipeline quality **before generation** — can the vector store surface the right documents?

**Dataset:** 25 queries across 3 difficulty levels (simple, multi_hop, edge_case), labeled with relevant source URLs.

**Config sweep:** 2 search types × 3 k values × 2 reranker states = 12 configurations.

**Metrics:**

| Metric | What it measures |
|--------|-----------------|
| **Hit Rate** | Fraction of queries with ≥1 relevant doc in top-k |
| **MRR** (Mean Reciprocal Rank) | How high the first relevant doc ranks |
| **Precision@k** | Fraction of retrieved docs that are relevant |
| **Recall@k** | Fraction of relevant docs that were retrieved |

**Full results (25 queries, BGE-base embeddings, 899 chunks):**

| Search Type | k | Reranker | Hit Rate | MRR | P@K | R@K |
|-------------|---|----------|----------|-----|-----|-----|
| **mmr** | **8** | **No** | **1.0000** | **0.6008** | **0.3650** | **0.8800** |
| mmr | 8 | Yes | 1.0000 | 0.6920 | 0.3650 | 0.8800 |
| mmr | 16 | No | 1.0000 | 0.5939 | 0.3500 | 0.8800 |
| mmr | 16 | Yes | 1.0000 | 0.6768 | 0.3500 | 0.8800 |
| similarity | 16 | No | 0.9600 | 0.5914 | 0.3600 | 0.8400 |
| similarity | 16 | Yes | 0.9600 | 0.7052 | 0.3600 | 0.8400 |
| mmr | 4 | No | 0.8400 | 0.6000 | 0.3600 | 0.7000 |
| mmr | 4 | Yes | 0.8400 | 0.6633 | 0.3600 | 0.7000 |
| similarity | 8 | No | 0.8000 | 0.5763 | 0.4350 | 0.7200 |
| similarity | 8 | Yes | 0.8000 | 0.6413 | 0.4350 | 0.7200 |
| similarity | 4 | No | 0.7200 | 0.5633 | 0.4400 | 0.6000 |
| similarity | 4 | Yes | 0.7200 | 0.6133 | 0.4400 | 0.6000 |

**Best config: MMR, k=8, no reranker** — 100% hit rate, 88% recall.

**By difficulty (best config):**

| Difficulty | Hit Rate | MRR | P@K | R@K |
|------------|----------|-----|-----|-----|
| simple | 1.0000 | 0.5085 | 0.2875 | 0.8000 |
| multi_hop | 1.0000 | 0.7310 | 0.3625 | 0.9000 |
| edge_case | 1.0000 | 0.5250 | 0.5250 | 1.0000 |

**Key findings:**

- **MMR dominates similarity** — consistently higher hit rate across all k values
- **k=8 is the sweet spot** — k=4 loses recall, k=16 adds noise without improving recall
- **Reranker improves MRR but not hit rate** — it reorders, doesn't find new docs. Useful when ranking quality matters more than coverage.
- **Edge cases are easiest** — narrow, specific queries retrieve well. Simple queries are hardest (broad, ambiguous phrasing).

```bash
# Run full retrieval evaluation (25 queries, 12 configs)
uv run doc-helper-retrieval-eval

# Quick test with 5 queries
uv run doc-helper-retrieval-eval --sample 5

# Custom output path
uv run doc-helper-retrieval-eval --output evaluation/my_results.json
```

Results saved to `evaluation/retrieval_results.json`.

### RAGAS Evaluation (Generation Quality)

Measures end-to-end answer quality — does the LLM generate good answers from retrieved context?

```bash
# Run RAGAS evaluation (uses gold dataset of 30 Q&A pairs)
uv run python -m doc_helper.evaluation.runner

# Limit to 5 questions for quick test
uv run python -m doc_helper.evaluation.runner --sample 5
```

Metrics measured:
- **Faithfulness** — Is the answer grounded in the context?
- **Answer Relevancy** — How relevant is the answer to the question?
- **Context Precision** — Are retrieved documents truly relevant?
- **Context Recall** — Are all needed documents retrieved?

Results saved to `evaluation/results.json`.

---

## Project Structure

```
src/doc_helper/
├── agents/           # RAG agent, tools, middleware, SSE events
├── api/              # FastAPI server, routes, dependency injection
├── config/           # Pydantic settings (nested, env-file based)
├── db/               # SQLite connection, migrations, conversation/task managers
├── embeddings/       # BGE-small and BGE-base, factory pattern
├── evaluation/       # Retrieval eval (IR metrics), RAGAS runner, gold dataset
├── ingestion/        # Crawlers (Tavily, recursive, local), splitters, pipeline
├── llm/              # Ollama and OpenRouter, factory pattern
├── observability/    # LangFuse, LangSmith, NoOp tracer, factory pattern
├── retrieval/        # Retriever with similarity/MMR/score-threshold + reranker
├── stores/           # Chroma and Pinecone vector stores, factory pattern
├── ui/               # Streamlit frontend (thin client calling API)
└── logger.py         # Structured logging helpers

tests/
├── unit/             # 100+ unit tests
└── integration/      # API integration tests with TestClient
```

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **LLM** | Ollama (local), OpenRouter (cloud) |
| **Embeddings** | BGE-small (384d), BGE-base (768d) via sentence-transformers |
| **Vector Store** | Chroma (default), Pinecone |
| **Framework** | LangChain, LangGraph (via `create_agent`) |
| **API** | FastAPI, SSE via sse-starlette |
| **Frontend** | Streamlit (thin client) |
| **Database** | SQLite (WAL mode), aiosqlite |
| **Observability** | LangFuse (default), LangSmith |
| **Evaluation** | RAGAS (faithfulness, relevancy, precision, recall) |
| **Search** | Similarity, MMR, score-threshold, FlashRank reranker |
| **Package** | `uv`, `hatchling` |

---

## License

MIT
