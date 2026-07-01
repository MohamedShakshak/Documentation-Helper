import os
from unittest.mock import patch

import pytest

from doc_helper.config.settings import (
    DatabaseSettings,
    EmbeddingSettings,
    IngestionSettings,
    LLMSettings,
    ObservabilitySettings,
    RetrievalSettings,
    Settings,
    VectorStoreSettings,
    get_settings,
)


class TestLLMSettings:
    def test_defaults(self):
        settings = LLMSettings()
        assert settings.provider == "ollama"
        assert settings.model == "qwen3.5:9b"
        assert settings.ollama_base_url == "http://localhost:11434"
        assert settings.openrouter_api_key is None
        assert settings.temperature == 0.0

    def test_custom_values(self):
        settings = LLMSettings(provider="openrouter", model="gpt-4o", openrouter_api_key="sk-test")
        assert settings.provider == "openrouter"
        assert settings.model == "gpt-4o"
        assert settings.openrouter_api_key == "sk-test"

    def test_invalid_provider(self):
        with pytest.raises(Exception):
            LLMSettings(provider="invalid")


class TestEmbeddingSettings:
    def test_defaults(self):
        settings = EmbeddingSettings()
        assert settings.model == "bge-small"
        assert settings.normalize is True

    def test_bge_base(self):
        settings = EmbeddingSettings(model="bge-base")
        assert settings.model == "bge-base"

    def test_invalid_model(self):
        with pytest.raises(Exception):
            EmbeddingSettings(model="invalid")


class TestVectorStoreSettings:
    def test_defaults(self):
        settings = VectorStoreSettings()
        assert settings.provider == "chroma"
        assert settings.chroma_persist_dir == "./chroma_db"
        assert settings.pinecone_api_key is None
        assert settings.pinecone_index_name == "langchain-docs"


class TestRetrievalSettings:
    def test_defaults(self):
        settings = RetrievalSettings()
        assert settings.search_type == "similarity"
        assert settings.search_k == 4
        assert settings.score_threshold == 0.5
        assert settings.reranker_enabled is False


class TestIngestionSettings:
    @patch.dict(os.environ, {"INGESTION__CRAWLER": "recursive", "INGESTION__CRAWL_URL": "https://python.langchain.com/docs/"})
    def test_defaults(self):
        settings = IngestionSettings()
        assert settings.crawler == "recursive"
        assert settings.crawl_url == "https://python.langchain.com/docs/"
        assert settings.crawl_depth == 2
        assert settings.crawl_prevent_outside is True
        assert settings.crawl_timeout == 10
        assert settings.chunk_size == 800
        assert settings.chunk_overlap == 150
        assert settings.batch_size == 500


class TestObservabilitySettings:
    def test_defaults(self):
        settings = ObservabilitySettings()
        assert settings.enabled is False
        assert settings.provider == "langfuse"


class TestDatabaseSettings:
    def test_defaults(self):
        settings = DatabaseSettings()
        assert settings.url == "sqlite:///./data/doc_helper.db"


class TestSettings:
    @patch.dict(os.environ, {"INGESTION__CRAWLER": "recursive", "OBSERVABILITY__ENABLED": "false", "LLM__PROVIDER": "ollama"})
    def test_defaults(self):
        settings = Settings()
        assert settings.llm.provider == "ollama"
        assert settings.embedding.model == "bge-small"
        assert settings.vector_store.provider == "chroma"
        assert settings.retrieval.search_type == "similarity"
        assert settings.ingestion.crawler == "recursive"
        assert settings.observability.enabled is False
        assert settings.database.url == "sqlite:///./data/doc_helper.db"
        assert settings.agent.max_tool_retries == 2
        assert settings.agent.guardrails_enabled is True

    def test_nested_override(self):
        settings = Settings(llm=LLMSettings(provider="openrouter", openrouter_api_key="sk-test"))
        assert settings.llm.provider == "openrouter"
        assert settings.embedding.model == "bge-small"

    @patch.dict(os.environ, {"LLM__PROVIDER": "ollama"})
    def test_get_settings(self):
        settings = get_settings()
        assert isinstance(settings, Settings)
        assert settings.llm.provider == "ollama"

    @patch.dict(os.environ, {"LLM__PROVIDER": "openrouter", "LLM__MODEL": "gpt-4o"})
    def test_env_override(self):
        settings = Settings()
        assert settings.llm.provider == "openrouter"
        assert settings.llm.model == "gpt-4o"