import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def disable_langsmith():
    with patch.dict(os.environ, {
        "LANGCHAIN_TRACING_V2": "false",
        "LANGSMITH_API_KEY": "",
    }):
        yield


def test_config_settings_import():
    from doc_helper.config.settings import Settings
    settings = Settings()
    assert settings.llm.provider == "ollama"
    assert settings.embedding.model == "bge-small"
    assert settings.vector_store.provider == "chroma"


def test_embeddings_factory_import():
    from doc_helper.embeddings.factory import EMBEDDING_MODELS, get_embedding_dimension
    assert "bge-small" in EMBEDDING_MODELS
    assert get_embedding_dimension("bge-small") == 384


def test_llm_factory_import():
    from doc_helper.llm.factory import create_chat_model
    assert callable(create_chat_model)


def test_stores_base_import():
    from doc_helper.stores.base import BaseVectorStore
    assert hasattr(BaseVectorStore, "add_documents")
    assert hasattr(BaseVectorStore, "validate_embedding_model")


def test_retriever_import():
    from doc_helper.retrieval.retriever import Retriever
    from doc_helper.retrieval.reranker import Reranker
    assert callable(Retriever)
    assert callable(Reranker)