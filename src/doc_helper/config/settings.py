from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    provider: Literal["ollama", "openrouter"] = "ollama"
    model: str = "qwen3.5:9b"
    ollama_base_url: str = "http://localhost:11434"
    openrouter_api_key: str | None = None
    temperature: float = 0.0

    model_config = SettingsConfigDict(env_prefix="LLM__")


class EmbeddingSettings(BaseSettings):
    model: Literal["bge-small", "bge-base"] = "bge-small"
    normalize: bool = True

    model_config = SettingsConfigDict(env_prefix="EMBEDDING__")


class VectorStoreSettings(BaseSettings):
    provider: Literal["chroma", "pinecone"] = "chroma"
    chroma_persist_dir: str = "./chroma_db"
    pinecone_api_key: str | None = None
    pinecone_index_name: str = "langchain-docs"

    model_config = SettingsConfigDict(env_prefix="VECTOR_STORE__")


class RetrievalSettings(BaseSettings):
    search_type: Literal["similarity", "mmr", "similarity_score_threshold"] = "similarity"
    search_k: int = 4
    score_threshold: float = 0.5
    reranker_enabled: bool = False

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL__")


class IngestionSettings(BaseSettings):
    crawler: Literal["tavily", "recursive", "local"] = "tavily"
    tavily_api_key: str | None = None
    crawl_url: str = "https://python.langchain.com/docs/"
    crawl_depth: int = 2
    crawl_prevent_outside: bool = True
    crawl_timeout: int = 10
    local_docs_dir: str = "./docs"
    split_strategy: Literal["markdown", "recursive"] = "markdown"
    chunk_size: int = 800
    chunk_overlap: int = 150
    batch_size: int = 500

    model_config = SettingsConfigDict(env_prefix="INGESTION__")


class ObservabilitySettings(BaseSettings):
    enabled: bool = False
    provider: Literal["langfuse", "langsmith"] = "langfuse"
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3000"
    langsmith_api_key: str | None = None

    model_config = SettingsConfigDict(env_prefix="OBSERVABILITY__")


class DatabaseSettings(BaseSettings):
    url: str = "sqlite:///./data/doc_helper.db"

    model_config = SettingsConfigDict(env_prefix="DATABASE__")


class AgentSettings(BaseSettings):
    max_tool_retries: int = 2
    guardrails_enabled: bool = True
    guardrails_max_input_length: int = 2000
    summarization_enabled: bool = True
    summarization_threshold: int = 20
    model_fallback_enabled: bool = True
    fallback_model: str | None = None

    model_config = SettingsConfigDict(env_prefix="AGENT__")


class Settings(BaseSettings):
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    vector_store: VectorStoreSettings = VectorStoreSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    ingestion: IngestionSettings = IngestionSettings()
    observability: ObservabilitySettings = ObservabilitySettings()
    database: DatabaseSettings = DatabaseSettings()
    agent: AgentSettings = AgentSettings()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
    )


def get_settings() -> Settings:
    return Settings()
