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
├── evaluation/       # Gold dataset, RAGAS runner
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
