# AGENTS.md

## Commands

```bash
uv sync                              # install deps (includes dev group)
uv run pytest                        # run all tests
uv run pytest tests/unit/             # unit tests only
uv run pytest tests/unit/test_config.py  # single test file
uv run ruff check src/ tests/        # lint
uv run ruff format --check src/ tests/  # format check
uv run mypy src/                     # typecheck (strict mode)
```

**Order matters:** lint → typecheck → test.

## Package layout

- `src/doc_helper/` — all application code. Uses hatchling `src/` layout; must be installed (`uv pip install -e .`) or run via `uv run` for imports to resolve.
- `tests/` — `unit/` (mocked, fast) and `integration/` (real components, small fixtures).
- Config is **nested Pydantic Settings** with `__` env-var separator: `LLM__PROVIDER`, `VECTOR_STORE__PROVIDER`, etc. See `.env.example` for all keys.
- Every swappable component (embeddings, LLM, vector store, crawler, observability) uses a **factory pattern** reading from `Settings`.
- `PLAN.md` holds all architecture decisions but is gitignored — read it for context on design rationale.

## Zero-config startup

The app runs with **zero API keys** by default: Ollama (local), Chroma (embedded), BGE-small (local model). API keys only required when switching providers (OpenRouter, Pinecone, Tavily). Don't add required env vars without reason.

## Architecture key points

- **Vector stores** (`stores/`): abstract `BaseVectorStore` with Chroma + Pinecone implementations. Stores embedding model name in vector store metadata and validates at query time — **never mix embedding models on the same index** (384-dim vs 768-dim).
- **RAG agent** (`agents/`): tool-calling agent, instantiated once and reused. Uses SQLite-backed conversation memory. SSE streaming emits typed events (`agents/events.py`: `agent_thought`, `tool_call`, `tool_result`, `answer`, `done`, `error`).
- **Ingestion** (`ingestion/`): pipeline is `crawl → split → store`. CLI via `doc-helper-ingest`. API task tracking via SQLite `tasks` table.
- **Database** (`db/`): SQLite with WAL mode, auto-migrations on connect. Two concerns: conversation history + ingestion task tracking.

## Incomplete modules (still stubs)

- `api/` — FastAPI server (Phase 5)
- `observability/` — LangFuse/LangSmith tracers (Phase 6)
- `evaluation/` — RAGAS runner (Phase 6)
- `docker/` — Dockerfile + compose (Phase 7)

## Conventions

- No comments in code unless explicitly requested.
- `ruff` line-length: 100. mypy: `strict=true`.
- pytest `asyncio_mode = "auto"` — no need for `@pytest.mark.asyncio`.
- Commits: `type: description` (conventional commits). One commit per logical step within a phase.
- **No CI/CD pipeline yet** — don't assume workflows exist.