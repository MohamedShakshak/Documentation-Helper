from typing import Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseModel):
    provider: Literal["ollama", "openrouter", "gemini"] = "ollama"
    model: str = "qwen3.5:9b"
    ollama_base_url: str = "http://localhost:11434"
    openrouter_api_key: str | None = None
    gemini_api_key: str | None = None
    temperature: float = 0.0


class EmbeddingSettings(BaseModel):
    model: Literal["bge-small", "bge-base"] = "bge-small"
    normalize: bool = True


class VectorStoreSettings(BaseModel):
    provider: Literal["chroma", "pinecone"] = "chroma"
    chroma_persist_dir: str = "./chroma_db"
    pinecone_api_key: str | None = None
    pinecone_index_name: str = "langchain-docs"


class RetrievalSettings(BaseModel):
    search_type: Literal["similarity", "mmr", "similarity_score_threshold"] = "mmr"
    search_k: int = 8
    score_threshold: float = 0.5
    reranker_enabled: bool = True


class IngestionSettings(BaseModel):
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


class ObservabilitySettings(BaseModel):
    enabled: bool = False
    provider: Literal["langfuse", "langsmith"] = "langfuse"
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3000"
    langsmith_api_key: str | None = None
    langsmith_project: str = "documentation-helper"


class DatabaseSettings(BaseModel):
    url: str = "sqlite:///./data/doc_helper.db"


class JudgeLLMSettings(BaseModel):
    provider: Literal["ollama", "openrouter", "gemini"] | None = None
    model: str | None = None
    openrouter_api_key: str | None = None
    gemini_api_key: str | None = None


class AgentSettings(BaseModel):
    max_tool_retries: int = 2
    guardrails_enabled: bool = True
    guardrails_max_input_length: int = 2000
    summarization_enabled: bool = True
    summarization_threshold: int = 20
    model_fallback_enabled: bool = True
    fallback_model: str | None = None


class Settings(BaseSettings):
    llm: LLMSettings = LLMSettings()
    judge_llm: JudgeLLMSettings = JudgeLLMSettings()
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
        extra="ignore",
    )


def get_settings() -> Settings:
    return Settings()
