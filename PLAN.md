# Documentation Helper — Architecture Enhancement Plan

## Overview

Transform the current single-file RAG prototype into a production-grade, modular system that demonstrates AI/LLM engineering depth for a CV.

## Decisions Summary

| # | Decision | Choice |
|---|----------|--------|
| 1 | Vector store | Both Chroma + Pinecone from the start, abstract interface + factory |
| 2 | Embedding models | BGE-small + BGE-base with dimension validation in metadata |
| 3 | LLM providers | Ollama (default, zero keys) + OpenRouter (cloud, production) |
| 4 | Retrieval | Similarity / MMR / score-threshold + optional FlashRank reranker |
| 5 | Agent pattern | Tool-calling RAG agent with conversation memory + middleware |
| 6 | Memory storage | SQLite for conversation persistence |
| 7 | Streaming | SSE with typed events (thought, tool_call, tool_result, answer, done) |
| 8 | Ingestion | CLI + API, SQLite task queue for async tracking |
| 9 | Crawlers | Tavily + RecursiveURLLoader + local files, factory pattern |
| 10 | Config | Nested Pydantic Settings, env-driven |
| 11 | Observability | LangFuse (default) + LangSmith, factory pattern, one active |
| 12 | Evaluation | RAGAS with hybrid gold dataset + OpenRouter as judge |
| 13 | Frontend | Streamlit calls FastAPI (thin frontend, single source of truth) |
| 14 | Project layout | Flat namespace `src/doc_helper/` + shared `db/` module |
| 15 | Docker | Profile-based Compose (dev/full/production), separate FastAPI + Streamlit containers |
| 16 | Testing | Unit + Integration + RAGAS eval |
| 17 | CI/CD | Skipped for now |
| 18 | Error handling | Skipped for now |
| 19 | Agent middleware | Guardrails + Summarization + Tool Retry + Model Fallback (adopted from chat-langchain) |
| 20 | Multi-tool agent | `retrieve_context` (vector store) + `web_search` (Tavily fallback) + `check_links` (URL validation) |
| 21 | Chunking | Markdown-aware splitting (MarkdownHeaderTextSplitter → RecursiveCharacterTextSplitter), chunk_size=800, overlap=100 |

### Inspiration: chat-langchain

Architecture patterns adopted from [langchain-ai/chat-langchain](https://github.com/langchain-ai/chat-langchain):

| Their Pattern | Our Adaptation |
|---------------|----------------|
| Middleware (guardrails, summarization, retry, fallback) | ✅ Add `middleware=[...]` to our `create_agent()` — same LangChain API |
| Multi-tool agent (docs search, support KB, link check) | ✅ 3 tools: `retrieve_context`, `web_search`, `check_links` |
| Model fallback chain (primary → fallback models) | ✅ OpenRouter → Ollama fallback via middleware |
| Prompt management (LangSmith Hub + local) | ✅ Local prompts now, LangFuse prompt management in Phase 6 |

**What we do NOT adopt from chat-langchain:**

| Their Pattern | Why We Skip |
|---------------|------------|
| Mintlify/Pylon API tools | We own our vector store — no external API needed |
| Redis caching with fuzzy matching | ChromaDB handles retrieval caching; overkill for this project |
| Next.js frontend | We keep Streamlit (agreed decision) |
| Raw LangGraph state machine | `create_agent()` already uses LangGraph internally; no benefit to hand-building graphs |

## Target Structure

```
documentation-helper/
├── src/
│   └── doc_helper/
│       ├── __init__.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py              # Nested Pydantic Settings
│       ├── db/
│       │   ├── __init__.py
│       │   ├── connection.py            # SQLite connection + migrations
│       │   ├── conversations.py         # Conversation memory storage
│       │   └── tasks.py                 # Ingestion task tracking
│       ├── embeddings/
│       │   ├── __init__.py
│       │   └── factory.py               # BGE-small + BGE-base, dimension validation
│       ├── llm/
│       │   ├── __init__.py
│       │   └── factory.py               # Ollama + OpenRouter
│       ├── stores/
│       │   ├── __init__.py
│       │   ├── base.py                  # Abstract VectorStore interface
│       │   ├── chroma_store.py          # ChromaDB implementation
│       │   ├── pinecone_store.py        # Pinecone implementation
│       │   └── factory.py               # Config-driven store selection
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── retriever.py             # Similarity/MMR/threshold strategies
│       │   └── reranker.py              # Optional FlashRank reranking
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── rag_agent.py             # Tool-calling agent with memory + middleware
│       │   ├── events.py                # SSE event schema
│       │   ├── tools/
│       │   │   ├── __init__.py
│       │   │   ├── retrieve_context.py  # RAG tool (our vector store)
│       │   │   ├── web_search.py        # Tavily web search fallback
│       │   │   └── check_links.py       # URL validation tool
│       │   └── middleware/
│       │       ├── __init__.py
│       │       ├── guardrails.py        # Block off-topic queries
│       │       ├── summarization.py     # Compress long conversations
│       │       ├── tool_retry.py        # Retry failed tool calls
│       │       └── model_fallback.py    # Fallback LLM on failure
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── crawlers/
│       │   │   ├── __init__.py
│       │   │   ├── base.py              # Abstract Crawler interface
│       │   │   ├── tavily_crawler.py
│       │   │   ├── recursive_crawler.py
│       │   │   └── local_file_crawler.py
│       │   ├── splitters/
│       │   │   ├── __init__.py
│       │   │   └── factory.py           # Markdown + Recursive, chunk_size=800
│       │   └── pipeline.py              # Orchestrate crawl + split + embed + store
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── runner.py                # RAGAS evaluation runner
│       │   └── dataset.py              # Gold Q&A dataset
│       ├── observability/
│       │   ├── __init__.py
│       │   ├── base.py                  # Abstract Tracer interface
│       │   ├── langfuse_tracer.py
│       │   ├── langsmith_tracer.py
│       │   └── factory.py              # LangFuse (default) / LangSmith
│       ├── api/
│       │   ├── __init__.py
│       │   ├── app.py                  # FastAPI application + CORS + lifespan
│       │   ├── deps.py                 # Dependency injection
│       │   └── routes/
│       │       ├── __init__.py
│       │       ├── chat.py             # POST /chat (SSE streaming)
│       │       ├── ingest.py            # POST /ingest + GET /ingest/{task_id}/status
│       │       └── health.py           # GET /health
│       └── ui/
│           ├── __init__.py
│           └── streamlit_app.py         # Thin UI, calls FastAPI
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_embeddings_factory.py
│   │   ├── test_llm_factory.py
│   │   ├── test_stores.py
│   │   ├── test_retriever.py
│   │   ├── test_reranker.py
│   │   ├── test_tools.py
│   │   ├── test_middleware.py
│   │   └── test_splitters.py
│   └── integration/
│       ├── test_rag_agent.py
│       └── test_ingestion_pipeline.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml              # Profiles: dev, full, production
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

## Module Specifications

### 1. Config (`src/doc_helper/config/settings.py`)

Nested Pydantic Settings using `SettingsConfigDict` with `env_file=".env"` and `env_nested_delimiter="__"`.

```python
class LLMSettings(BaseSettings):
    provider: Literal["ollama", "openrouter"] = "ollama"
    model: str = "qwen3.5:9b"
    ollama_base_url: str = "http://localhost:11434"
    openrouter_api_key: str | None = None
    temperature: float = 0.0

class EmbeddingSettings(BaseSettings):
    model: Literal["bge-small", "bge-base"] = "bge-small"
    normalize: bool = True

class VectorStoreSettings(BaseSettings):
    provider: Literal["chroma", "pinecone"] = "chroma"
    chroma_persist_dir: str = "./chroma_db"
    pinecone_api_key: str | None = None
    pinecone_index_name: str = "langchain-docs"

class RetrievalSettings(BaseSettings):
    search_type: Literal["similarity", "mmr", "similarity_score_threshold"] = "similarity"
    search_k: int = 4
    score_threshold: float = 0.5
    reranker_enabled: bool = False

class IngestionSettings(BaseSettings):
    crawler: Literal["tavily", "recursive", "local"] = "tavily"
    tavily_api_key: str | None = None
    crawl_url: str = "https://python.langchain.com/"
    crawl_depth: int = 2
    local_docs_dir: str = "./docs"
    chunk_size: int = 800
    chunk_overlap: int = 100
    batch_size: int = 500

class AgentSettings(BaseSettings):
    guardrails_enabled: bool = True
    guardrails_model: str = "qwen3.5:9b"
    summarization_enabled: bool = True
    summarization_token_threshold: int = 8000
    summarization_keep_tokens: int = 2000
    tool_retry_max_attempts: int = 3
    model_fallback_enabled: bool = True
    model_fallback_provider: str = "ollama"
    model_fallback_model: str = "qwen3.5:9b"
    web_search_enabled: bool = False
    link_check_enabled: bool = True

class ObservabilitySettings(BaseSettings):
    enabled: bool = False
    provider: Literal["langfuse", "langsmith"] = "langfuse"
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3000"
    langsmith_api_key: str | None = None

class DatabaseSettings(BaseSettings):
    url: str = "sqlite:///./data/doc_helper.db"

class Settings(BaseSettings):
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    vector_store: VectorStoreSettings = VectorStoreSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    ingestion: IngestionSettings = IngestionSettings()
    agent: AgentSettings = AgentSettings()
    observability: ObservabilitySettings = ObservabilitySettings()
    database: DatabaseSettings = DatabaseSettings()

    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__")
```

**Key behavior:** The app must start and run with zero environment variables (Ollama + Chroma + BGE-small + no observability). API keys are only required when switching providers.

**Environment file (`.env.example`):**

```env
# LLM Configuration
LLM__PROVIDER=ollama
LLM__MODEL=qwen3.5:9b
LLM__OLLAMA_BASE_URL=http://localhost:11434
LLM__OPENROUTER_API_KEY=

# Embedding Configuration
EMBEDDING__MODEL=bge-small
EMBEDDING__NORMALIZE=true

# Vector Store Configuration
VECTOR_STORE__PROVIDER=chroma
VECTOR_STORE__CHROMA_PERSIST_DIR=./chroma_db
VECTOR_STORE__PINECONE_API_KEY=
VECTOR_STORE__PINECONE_INDEX_NAME=langchain-docs

# Retrieval Configuration
RETRIEVAL__SEARCH_TYPE=similarity
RETRIEVAL__SEARCH_K=4
RETRIEVAL__SCORE_THRESHOLD=0.5
RETRIEVAL__RERANKER_ENABLED=false

# Ingestion Configuration
INGESTION__CRAWLER=tavily
INGESTION__TAVILY_API_KEY=
INGESTION__CRAWL_URL=https://python.langchain.com/
INGESTION__CRAWL_DEPTH=2
INGESTION__LOCAL_DOCS_DIR=./docs
INGESTION__CHUNK_SIZE=800
INGESTION__CHUNK_OVERLAP=100
INGESTION__BATCH_SIZE=500

# Agent Configuration
AGENT__GUARDRAILS_ENABLED=true
AGENT__GUARDRAILS_MODEL=qwen3.5:9b
AGENT__SUMMARIZATION_ENABLED=true
AGENT__SUMMARIZATION_TOKEN_THRESHOLD=8000
AGENT__SUMMARIZATION_KEEP_TOKENS=2000
AGENT__TOOL_RETRY_MAX_ATTEMPTS=3
AGENT__MODEL_FALLBACK_ENABLED=true
AGENT__MODEL_FALLBACK_PROVIDER=ollama
AGENT__MODEL_FALLBACK_MODEL=qwen3.5:9b
AGENT__WEB_SEARCH_ENABLED=false
AGENT__LINK_CHECK_ENABLED=true

# Observability Configuration
OBSERVABILITY__ENABLED=false
OBSERVABILITY__PROVIDER=langfuse
OBSERVABILITY__LANGFUSE_PUBLIC_KEY=
OBSERVABILITY__LANGFUSE_SECRET_KEY=
OBSERVABILITY__LANGFUSE_HOST=http://localhost:3000
OBSERVABILITY__LANGSMITH_API_KEY=

# Database Configuration
DATABASE__URL=sqlite:///./data/doc_helper.db
```

### 2. Embeddings (`src/doc_helper/embeddings/factory.py`)

Factory that returns the correct `HuggingFaceEmbeddings` instance based on config.

```python
EMBEDDING_MODELS = {
    "bge-small": {"model_name": "BAAI/bge-small-en-v1.5", "dim": 384},
    "bge-base": {"model_name": "BAAI/bge-base-en-v1.5", "dim": 768},
}

def create_embeddings(settings: EmbeddingSettings) -> HuggingFaceEmbeddings:
    config = EMBEDDING_MODELS[settings.model]
    return HuggingFaceEmbeddings(
        model_name=config["model_name"],
        encode_kwargs={"normalize_embeddings": settings.normalize},
    )

def get_embedding_dimension(model_key: str) -> int:
    return EMBEDDING_MODELS[model_key]["dim"]
```

**Dimension validation:** When initializing a vector store, store the embedding model name in the collection/index metadata. When querying, validate that the configured model matches the stored model. Raise a clear error on mismatch (e.g., "Store was created with bge-small (384-dim) but bge-base (768-dim) is configured. Re-ingest or switch models.").

### 3. LLM (`src/doc_helper/llm/factory.py`)

Factory that returns the correct chat model based on config.

```python
def create_chat_model(settings: LLMSettings) -> BaseChatModel:
    if settings.provider == "ollama":
        return init_chat_model(
            settings.model,
            model_provider="ollama",
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
        )
    elif settings.provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ConfigError("OPENROUTER_API_KEY required when LLM__PROVIDER=openrouter")
        return init_chat_model(
            settings.model,
            model_provider="openrouter",
            openrouter_api_key=settings.openrouter_api_key,
            temperature=settings.temperature,
        )
```

### 4. Vector Stores (`src/doc_helper/stores/`)

**Abstract interface (`base.py`):**

```python
class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: list[Document], batch_size: int = 500) -> None: ...

    @abstractmethod
    def as_retriever(self, search_type: str, search_kwargs: dict) -> VectorStoreRetriever: ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None: ...

    @abstractmethod
    def get_embedding_model_name(self) -> str: ...

    def validate_embedding_model(self, configured_model: str) -> None:
        stored_model = self.get_embedding_model_name()
        if stored_model and stored_model != configured_model:
            raise StoreError(
                f"Vector store was created with '{stored_model}' but "
                f"'{configured_model}' is configured. Re-ingest or switch models."
            )
```

**Factory (`factory.py`):**

```python
def create_vector_store(settings: Settings) -> BaseVectorStore:
    embeddings = create_embeddings(settings.embedding)
    if settings.vector_store.provider == "chroma":
        return ChromaVectorStore(settings.vector_store, embeddings)
    elif settings.vector_store.provider == "pinecone":
        if not settings.vector_store.pinecone_api_key:
            raise ConfigError("PINECONE_API_KEY required when VECTOR_STORE__PROVIDER=pinecone")
        return PineconeVectorStore(settings.vector_store, embeddings)
```

**ChromaDB implementation (`chroma_store.py`):** Wraps `langchain_chroma.Chroma`, stores embedding model name in collection metadata, validates on query.

**Pinecone implementation (`pinecone_store.py`):** Wraps `langchain_pinecone.PineconeVectorStore`, stores embedding model name in index metadata, validates on query.

### 5. Retrieval (`src/doc_helper/retrieval/`)

**Retriever (`retriever.py`):**

```python
class Retriever:
    def __init__(self, store: BaseVectorStore, settings: RetrievalSettings):
        self.store = store
        self.settings = settings

    def retrieve(self, query: str) -> list[Document]:
        search_kwargs = {"k": self.settings.search_k}
        if self.settings.search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = self.settings.score_threshold
        retriever = self.store.as_retriever(
            search_type=self.settings.search_type,
            search_kwargs=search_kwargs,
        )
        documents = retriever.invoke(query)
        if self.settings.reranker_enabled:
            documents = Reranker().rerank(query, documents)
        return documents
```

**Reranker (`reranker.py`):**

Uses `flashrank` with the `ms-marco-MiniLM-L-12-v2` model. Takes the top-k retrieved documents, re-scores them, and returns the top-k after reranking. Toggled via `RETRIEVAL__RERANKER_ENABLED`.

**Reranker (`reranker.py`):**

Uses `flashrank` with the `ms-marco-MiniLM-L-12-v2` model. Takes the top-k retrieved documents, re-scores them, and returns the top-k after reranking. Toggled via `RETRIEVAL__RERANKER_ENABLED`.

### 6. Agents (`src/doc_helper/agents/`)

**Multi-tool RAG agent with middleware** — adopted from chat-langchain's architecture.

#### 6a. Tools (`agents/tools/`)

The agent has 3 tools, each in its own file:

| Tool | File | Description |
|------|------|-------------|
| `retrieve_context` | `tools/retrieve_context.py` | Searches our vector store (Chroma/Pinecone) for relevant docs. Returns serialized content + raw Document artifacts. This is our primary RAG tool — we own the full retrieval pipeline. |
| `web_search` | `tools/web_search.py` | Falls back to Tavily Search API when vector store results are insufficient. Agent decides: "I checked docs, didn't find a good answer, let me search the web." Requires `TAVILY_API_KEY`. |
| `check_links` | `tools/check_links.py` | Validates URLs in retrieved sources before returning them to the user. Asynchronous HTTP HEAD/GET checks with soft-404 detection for docs.langchain.com domains. Caches results in-memory. |

Each tool is a standalone `@tool` function that can be tested independently.

#### 6b. Middleware (`agents/middleware/`)

Middleware intercepts agent execution at specific hooks. Ordered list passed to `create_agent(middleware=[...])`.

| Middleware | File | What It Does |
|------------|------|-------------|
| Guardrails | `middleware/guardrails.py` | Uses a small LLM to classify queries as ALLOWED/BLOCKED before agent runs. Blocks off-topic queries (non-LangChain questions). Falls back to allowing on classification failure. |
| Summarization | `middleware/summarization.py` | When conversation history exceeds a token threshold (configurable, default 8000), compresses older messages into a summary. Prevents context window overflow in long conversations. |
| Tool Retry | `middleware/tool_retry.py` | Retries failed tool calls up to 3 times with exponential backoff. Handles transient failures (API timeouts, Chroma connection issues). |
| Model Fallback | `middleware/model_fallback.py` | If primary LLM (OpenRouter) fails, falls back to secondary (Ollama). Uses LangChain's `ModelFallbackMiddleware`. |

**Agent creation:**

```python
agent = create_agent(
    model=llm,
    tools=[retrieve_context, web_search, check_links],
    system_prompt=system_prompt,
    middleware=[
        guardrails_middleware,      # Block off-topic queries
        summarization_middleware,    # Compress long conversations
        tool_retry_middleware,      # Retry failed tool calls
        model_fallback_middleware,   # Fall back to Ollama if OpenRouter fails
    ],
)
```

#### 6c. RAG Agent (`agents/rag_agent.py`)

- **Memory:** SQLite-backed conversation history via `ConversationManager`.
- **Agent creation:** Uses `create_agent` with system prompt + middleware. Agent is instantiated once and reused via dependency injection.
- **Streaming:** Uses LangChain's `astream_events` to emit typed SSE events:

```python
# SSE event types emitted during streaming
EventTypes:
  agent_thought   # LLM reasoning tokens
  tool_call       # Tool invoked (includes tool name + query)
  tool_result     # Tool output (includes source list / search results)
  answer          # Final answer tokens
  done            # Stream complete
  error           # Stream failed
```

### 7. Ingestion (`src/doc_helper/ingestion/`)

**Crawler interface (`crawlers/base.py`):**

```python
class BaseCrawler(ABC):
    @abstractmethod
    async def crawl(self) -> list[Document]: ...
```

**Implementations:**

- `tavily_crawler.py` — Uses `TavilyCrawl` (current behavior, extracted from `ingestion.py`)
- `recursive_crawler.py` — Uses LangChain's `RecursiveUrlLoader` (zero API keys needed)
- `local_file_crawler.py` — Recursively loads `.md`, `.txt`, `.pdf` files from a local directory using `DirectoryLoader`

**Splitter factory (`splitters/factory.py`):**

Two-stage chunking strategy selectable via `INGESTION__SPLIT_STRATEGY`:

- **`markdown` (default):** First splits by markdown headers (`MarkdownHeaderTextSplitter` — respects `#`, `##`, `###` boundaries), then applies `RecursiveCharacterTextSplitter` on each section. Preserves logical structure. Degrades gracefully to recursive if content isn't markdown.
- **`recursive`:** Plain `RecursiveCharacterTextSplitter` only.

Defaults updated: `chunk_size=800` (was 4000), `chunk_overlap=100` (was 200). Smaller chunks improve retrieval precision.

**Pipeline (`pipeline.py`):**

Orchestrates: `crawl → split → embed → store`

- Called from CLI (`python -m doc_helper.ingest`) or API (`POST /api/ingest`)
- For API path: creates a task in SQLite `tasks` table, runs async, updates progress
- Supports resumable ingestion (skip already-indexed URLs by checking metadata)

**CLI interface:**

```bash
python -m doc_helper.ingest \
  --url https://python.langchain.com/ \
  --depth 2 \
  --crawler tavily \
  --store chroma \
  --chunk-size 4000 \
  --chunk-overlap 200
```

### 8. Database (`src/doc_helper/db/`)

**SQLite database** stored at `./data/doc_helper.db` (configurable via `DATABASE__URL`).

**Schema:**

```sql
-- Conversations
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,  -- 'user', 'assistant', 'system', 'tool'
    content TEXT NOT NULL,
    sources TEXT,  -- JSON array of source URLs
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

-- Ingestion tasks
CREATE TABLE tasks (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending, running, completed, failed
    progress INTEGER DEFAULT 0,
    crawler TEXT NOT NULL,
    urls_crawled INTEGER DEFAULT 0,
    chunks_created INTEGER DEFAULT 0,
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**`connection.py`:** Manages SQLite connection, runs schema migrations on startup.

**`conversations.py`:** CRUD operations for conversations and messages. Provides LangChain `BaseChatMessageHistory` compatible interface.

**`tasks.py`:** CRUD operations for ingestion tasks. Tracks status, progress, and errors.

### 9. Observability (`src/doc_helper/observability/`)

**Abstract interface (`base.py`):**

```python
class BaseTracer(ABC):
    @abstractmethod
    def trace_agent_run(self, query: str, **kwargs): ...

    @abstractmethod
    def trace_retrieval(self, query: str, documents: list[Document]): ...

    @abstractmethod
    def trace_ingestion(self, documents: list[Document]): ...
```

**LangFuse tracer (`langfuse_tracer.py`):** Wraps `langfuse.LangFuse` with `@observe()` decorator pattern.

**LangSmith tracer (`langsmith_tracer.py`):** Uses LangChain's `LangSmithTracer` callback handler.

**Factory (`factory.py`):** Returns `NoOpTracer` when `OBSERVABILITY__ENABLED=false`, otherwise returns the configured provider tracer. Validates that required API keys are present when a provider is selected.

### 10. Evaluation (`src/doc_helper/evaluation/`)

**Dataset (`dataset.py`):**

JSON file (`evaluation/gold_dataset.json`) containing ~30 question-answer pairs across difficulty levels:

```json
[
  {
    "question": "What is a LangChain agent?",
    "reference_answer": "An agent in LangChain is a system that uses an LLM to decide what actions to take...",
    "reference_contexts": ["https://python.langchain.com/docs/concepts/agents/"],
    "difficulty": "simple"
  },
  {
    "question": "How does a RAG agent decide when to retrieve documents vs answer from memory?",
    "reference_answer": "...",
    "reference_contexts": ["https://python.langchain.com/docs/..."],
    "difficulty": "multi_hop"
  }
]
```

**Runner (`runner.py`):**

- Uses RAGAS metrics: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`
- Uses OpenRouter (configured LLM) as the judge model
- CLI command: `python -m doc_helper.evaluate`
- Outputs structured results to console and saves to `evaluation/results/` as JSON

### 11. FastAPI (`src/doc_helper/api/`)

**`app.py`:**

```python
app = FastAPI(
    title="Documentation Helper API",
    version="0.1.0",
    lifespan=lifespan,  # Initialize store, embeddings, agent on startup
)

# CORS for Streamlit frontend
app.add_middleware(CORALSMiddleware, allow_origins=["*"])
```

**`deps.py`:**

Dependency injection using FastAPI's `Depends()`:

```python
def get_settings() -> Settings: ...
def get_vector_store(settings=Depends(get_settings)) -> BaseVectorStore: ...
def get_agent(settings=Depends(get_settings), store=Depends(get_vector_store)) -> RAGAgent: ...
def get_db() -> Database: ...
```

All factories are called once and cached (singleton pattern via `lru_cache` or FastAPI's default).

**Routes:**

| Route | Method | Description |
|-------|--------|-------------|
| `/api/health` | GET | Health check (store connectivity, LLM availability) |
| `/api/chat` | POST | Send a message, get SSE stream of typed events |
| `/api/chat/{conversation_id}` | GET | Get conversation history |
| `/api/ingest` | POST | Start an ingestion job, return task_id |
| `/api/ingest/{task_id}/status` | GET | Poll ingestion task status |

**SSE streaming format for `/api/chat`:**

```
event: agent_thought
data: {"content": "Let me search for that..."}

event: tool_call
data: {"tool": "retrieve_context", "query": "What are LangChain agents?"}

event: tool_result
data: {"sources": ["https://python.langchain.com/docs/..."], "num_docs": 4}

event: answer
data: {"content": "LangChain agents are"}

event: answer
data: {"content": " systems that use"}

event: done
data: {"conversation_id": "abc-123", "sources": [...]}
```

### 12. Streamlit UI (`src/doc_helper/ui/streamlit_app.py`)

Thin frontend that calls FastAPI endpoints:

1. **On startup:** Checks `/api/health` to confirm backend is running.
2. **Conversation:** Creates a conversation via `/api/chat`, stores `conversation_id` in `st.session_state`.
3. **Sending a message:** POSTs to `/api/chat` with `conversation_id` and `message`. Consumes SSE stream, renders each event type appropriately (thought in a collapsible, sources in an expander, answer as markdown).
4. **No business logic** — all RAG, retrieval, and agent logic lives in FastAPI.

### 13. Docker

**Dockerfile (multi-stage):**

```dockerfile
# Stage 1: Build
FROM python:3.12-slim AS builder
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen --no-dev
COPY src/ src/

# Stage 2: Runtime
FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
ENV PATH="/app/.venv/bin:$PATH"
```

Two entrypoints:
- `python -m doc_helper.api` — FastAPI backend
- `python -m doc_helper.ui` — Streamlit frontend

**docker-compose.yml with profiles:**

```yaml
# Profile: dev (default) — just the API + UI, requires local Ollama
# Profile: full — adds Chroma server + LangFuse + Postgres
# Profile: production — API + UI + Chroma server, uses OpenRouter + Pinecone

services:
  api:
    build: .
    command: python -m doc_helper.api
    profiles: ["dev", "full", "production"]
    ports: ["8000:8000"]
    env_file: .env

  ui:
    build: .
    command: streamlit run src/doc_helper/ui/streamlit_app.py
    profiles: ["dev", "full", "production"]
    ports: ["8501:8501"]
    depends_on: [api]

  ollama:
    image: ollama/ollama:latest
    profiles: ["full"]
    ports: ["11434:11434"]
    volumes: ["ollama_data:/root/.ollama"]

  chroma:
    image: chromadb/chroma:latest
    profiles: ["full", "production"]
    ports: ["8001:8000"]
    volumes: ["chroma_data:/chroma/chroma"]

  langfuse:
    image: langfuse/langfuse:latest
    profiles: ["full"]
    ports: ["3000:3000"]
    depends_on: [postgres]

  postgres:
    image: postgres:16-alpine
    profiles: ["full"]
    environment:
      POSTGRES_DB: langfuse
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: langfuse
    volumes: ["pg_data:/var/lib/postgresql/data"]

volumes:
  ollama_data:
  chroma_data:
  pg_data:
```

## Implementation Phases

### Phase 1: Project Restructure (Days 1-3)

**Goal:** Establish the new project structure, config system, and factories. Fix the vector store mismatch bug.

**Tasks:**
1. Create `src/doc_helper/` package structure with all `__init__.py` files
2. Implement `config/settings.py` with nested Pydantic Settings
3. Create `.env.example` with all documented configuration options
4. Implement `embeddings/factory.py` — BGE-small + BGE-base factory
5. Implement `llm/factory.py` — Ollama + OpenRouter factory
6. Refactor existing `core.py` → temporary bridge code that uses new factories
7. Fix the vector store mismatch (currently ingestion writes to Chroma, core reads from Pinecone)
8. Update `pyproject.toml`:
   - Remove `asyncio`, `torchvision`
   - Add `pydantic-settings`, `fastapi`, `uvicorn`, `flashrank`, `langfuse`, `langsmith`, `ragas`, `click`, `sse-starlette`
   - Add dev dependencies: `pytest`, `pytest-asyncio`, `ruff`, `mypy`
   - Add `[project.scripts]` entrypoints for CLI commands

### Phase 2: Abstract Store Layer (Days 4-6)

**Goal:** Implement the abstract vector store interface with Chroma and Pinecone backends, including dimension validation.

**Tasks:**
1. Implement `stores/base.py` — abstract `BaseVectorStore` interface
2. Implement `stores/chroma_store.py` — ChromaDB backend with metadata embedding model storage
3. Implement `stores/pinecone_store.py` — Pinecone backend with metadata embedding model storage
4. Implement `stores/factory.py` — config-driven store selection with API key validation
5. Add embedding model name to vector store metadata on `add_documents`
6. Add `validate_embedding_model()` check before retrieval queries
7. Write unit tests for:
   - Factory returns correct store type based on config
   - Chroma and Pinecone implementations satisfy the interface
   - Dimension mismatch detection raises clear error
8. Refactor `ingestion.py` to use `BaseVectorStore` instead of direct Chroma

### Phase 3: Retrieval & Agent (Days 7-9) ✅ Done

**Goal:** Build the configurable retriever, optional reranker, RAG agent with memory, SSE events, and SQLite database layer.

**Tasks:**
1. ✅ Implement `retrieval/retriever.py` — configurable search type (similarity/MMR/threshold)
2. ✅ Implement `retrieval/reranker.py` — FlashRank cross-encoder reranker
3. ✅ Implement `db/connection.py` — SQLite connection + migrations (WAL mode)
4. ✅ Implement `db/conversations.py` — conversation CRUD + LangChain message adapter
5. ✅ Implement `db/tasks.py` — ingestion task tracking (pending/running/completed/failed)
6. ✅ Implement `agents/events.py` — SSE event schema (6 event types)
7. ✅ Implement `agents/rag_agent.py` — tool-calling agent with memory + streaming + source extraction
8. ✅ Write unit tests for database, conversations, tasks, events
9. ✅ Write integration test for RAG agent (mock LLM, message building, memory)

### Phase 4: Ingestion Pipeline + Markdown Chunking (Days 10-12)

**Goal:** Build the composable ingestion pipeline with CLI, markdown-aware chunking, and API support.

**Tasks:**
1. ✅ Implement `ingestion/crawlers/base.py` — abstract `BaseCrawler` interface
2. ✅ Implement `ingestion/crawlers/tavily_crawler.py` — extract from current `ingestion.py`
3. ✅ Implement `ingestion/crawlers/recursive_crawler.py` — LangChain `RecursiveUrlLoader`
4. ✅ Implement `ingestion/crawlers/local_file_crawler.py` — LangChain `DirectoryLoader`
5. Update `ingestion/splitters/factory.py` — add `SPLIT_STRATEGY` config (`markdown` | `recursive`):
   - **`markdown` (default):** `MarkdownHeaderTextSplitter` → `RecursiveCharacterTextSplitter` (two-stage)
   - **`recursive`:** plain `RecursiveCharacterTextSplitter`
   - Update defaults: `chunk_size=800`, `chunk_overlap=100`
6. Update `config/settings.py` — add `split_strategy` to `IngestionSettings`
7. Update `.env.example` — document `INGESTION__SPLIT_STRATEGY`
8. ✅ Implement `ingestion/pipeline.py` — orchestrate crawl → split → embed → store
9. ✅ Implement CLI command: `python -m doc_helper.ingest` with Click arguments
10. Write unit tests for:
    - Markdown splitter respects headers
    - Recursive fallback works when content isn't markdown
    - Each crawler returns `list[Document]`
    - Pipeline correctly chains crawl → split → store

### Phase 4.5: Multi-Tool Agent + Middleware (Days 12-13)

**Goal:** Add tools (web_search, check_links) and middleware (guardrails, summarization, retry, fallback) to the RAG agent. Adopted from chat-langchain architecture.

**Tasks:**
1. Create `agents/tools/` directory with `__init__.py`
2. Implement `agents/tools/retrieve_context.py` — extract from current `rag_agent.py`, standalone tool
3. Implement `agents/tools/web_search.py` — Tavily Search API fallback tool (`AGENT__WEB_SEARCH_ENABLED`)
4. Implement `agents/tools/check_links.py` — async URL validation with soft-404 detection (`AGENT__LINK_CHECK_ENABLED`)
5. Create `agents/middleware/` directory with `__init__.py`
6. Implement `agents/middleware/guardrails.py` — LLM-based query classification (ALLOWED/BLOCKED)
7. Implement `agents/middleware/summarization.py` — compress conversations exceeding token threshold
8. Implement `agents/middleware/tool_retry.py` — retry failed tool calls with backoff
9. Implement `agents/middleware/model_fallback.py` — fall back from OpenRouter to Ollama on failure
10. Add `AgentSettings` to `config/settings.py` — all middleware config options
11. Update `.env.example` with `AGENT__*` variables
12. Refactor `agents/rag_agent.py` — wire tools + middleware into `create_agent(middleware=[...])`
13. Write unit tests for:
    - Each tool (retrieve_context, web_search, check_links)
    - Guardrails classification (allowed/blocked queries)
    - Summarization triggers at token threshold
    - Tool retry with backoff
    - Model fallback chain
14. Update `pyproject.toml` dependencies if needed

### Phase 5: FastAPI + Streamlit (Days 13-15)

**Goal:** Build the API server and refactor Streamlit to call it.

**Tasks:**
1. Implement `api/app.py` — FastAPI app with CORS, lifespan (warm up store/embeddings)
2. Implement `api/deps.py` — dependency injection for settings, store, agent, db
3. Implement `api/routes/health.py` — `GET /api/health`
4. Implement `api/routes/chat.py` — `POST /api/chat` (SSE streaming with typed events), `GET /api/chat/{conversation_id}`
5. Implement `api/routes/ingest.py` — `POST /api/ingest`, `GET /api/ingest/{task_id}/status`
6. Implement `db/conversations.py` — conversation CRUD + LangChain `BaseChatMessageHistory` adapter
7. Refactor `ui/streamlit_app.py` — thin frontend calling FastAPI:
   - On startup: health check
   - Conversation management via API
   - SSE stream consumption and rendering (thought → collapsible, sources → expander, answer → markdown)
8. Write integration tests for API endpoints using FastAPI `TestClient`

### Phase 6: Observability & Evaluation (Days 16-18)

**Goal:** Add tracing and RAGAS evaluation.

**Tasks:**
1. Implement `observability/base.py` — abstract `BaseTracer` interface
2. Implement `observability/langfuse_tracer.py` — LangFuse integration
3. Implement `observability/langsmith_tracer.py` — LangSmith integration
4. Implement `observability/factory.py` — returns `NoOpTracer` when disabled, validates API keys when enabled
5. Instrument agent and retrieval code with tracer calls
6. Create `evaluation/gold_dataset.json` — ~30 Q&A pairs about LangChain docs across difficulty levels (simple, multi_hop, edge_case)
7. Implement `evaluation/runner.py` — RAGAS runner with OpenRouter judge
8. Implement CLI command: `python -m doc_helper.evaluate`
9. Write unit tests for tracer factory (returns NoOp when disabled, returns correct tracer when enabled)

### Phase 7: Database, Docker & Polish (Days 19-21)

**Goal:** SQLite infrastructure, containerization, and documentation.

**Tasks:**
1. Implement `db/connection.py` — SQLite connection management, schema migrations on startup
2. Ensure `db/conversations.py` and `db/tasks.py` use shared connection
3. Create `docker/Dockerfile` — multi-stage build with two entrypoints
4. Create `docker/docker-compose.yml` — profiles (dev, full, production)
5. Update `.gitignore` — add `data/`, `chroma_db/`, `*.db`, `evaluation/results/`
6. Update `pyproject.toml` — add CLI entrypoints, fix description, add metadata
7. Write `README.md`:
   - Architecture diagram
   - Setup instructions (zero API keys mode)
   - Configuration reference
   - API endpoint documentation
   - CLI commands
   - Docker instructions
   - Evaluation instructions
8. Final cleanup: type hints everywhere, remove dead code, verify all config defaults work with zero env vars

## Testing Strategy

### Unit Tests
- Config: settings load from env, validation errors on missing required keys
- Embeddings factory: returns correct model, dimension lookup
- LLM factory: returns correct provider, API key validation
- Store factory: returns correct store type, dimension mismatch detection
- Retriever: different search types return documents
- Reranker: reorders results when enabled, passes through when disabled
- Splitters: markdown strategy respects headers, recursive fallback works
- Tools: retrieve_context, web_search, check_links each produce correct output
- Middleware: guardrails classification, summarization threshold, tool retry, model fallback
- Observability factory: NoOp when disabled, correct tracer when enabled

### Integration Tests
- RAG agent: end-to-end query with mock LLM, real Chroma instance, verify tool calls
- Ingestion pipeline: crawl mock data → split (markdown) → store → retrieve, verify round-trip
- Agent middleware: guardrails blocks off-topic, summarization compresses long history
- FastAPI endpoints: health check, chat stream, ingestion task lifecycle

### RAGAS Evaluation
- CLI command: `python -m doc_helper.evaluate`
- Uses gold dataset (`evaluation/gold_dataset.json`)
- Measures: faithfulness, answer_relevancy, context_precision, context_recall
- Uses OpenRouter as judge model
- Manual trigger only (not in CI)

## Key Principles

1. **Zero-config startup** — the app must start with `docker-compose up` in dev profile with zero API keys. Ollama + Chroma + BGE-small work out of the box.
2. **Factory pattern everywhere** — every component (store, embedding, LLM, crawler, tracer) is swappable via config. This is the core CV signal.
3. **Validate early, fail clearly** — mismatched embedding models, missing API keys, wrong search types — all caught at startup with clear error messages.
4. **Separation of concerns** — Streamlit is a thin UI, FastAPI is the API layer, `doc_helper` is the library. No business logic in UI or API routes.
5. **Observable by default** — tracing is built in from day one, not bolted on later.
6. **Measurable quality** — RAGAS evaluation gives you numbers to put on a CV, not just "it works."