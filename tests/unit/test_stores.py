from unittest.mock import MagicMock, patch

import pytest

from doc_helper.config.settings import EmbeddingSettings, Settings, VectorStoreSettings
from doc_helper.stores.base import BaseVectorStore
from doc_helper.stores.chroma_store import ChromaVectorStore
from doc_helper.stores.pinecone_store import PineconeVectorStore
from doc_helper.stores.factory import create_vector_store


class TestBaseVectorStore:
    def test_validate_embedding_model_match(self):
        store = MagicMock(spec=BaseVectorStore)
        store.get_embedding_model_name.return_value = "bge-small"
        BaseVectorStore.validate_embedding_model(store, "bge-small")

    def test_validate_embedding_model_mismatch_raises(self):
        store = MagicMock(spec=BaseVectorStore)
        store.get_embedding_model_name.return_value = "bge-small"
        with pytest.raises(ValueError, match="Re-ingest or switch models"):
            BaseVectorStore.validate_embedding_model(store, "bge-base")

    def test_validate_embedding_model_no_stored_model(self):
        store = MagicMock(spec=BaseVectorStore)
        store.get_embedding_model_name.return_value = None
        BaseVectorStore.validate_embedding_model(store, "bge-small")


class TestChromaVectorStore:
    @patch("doc_helper.stores.chroma_store.Chroma")
    def test_add_documents(self, mock_chroma_cls):
        mock_store = MagicMock()
        mock_chroma_cls.return_value = mock_store
        mock_store._collection.metadata = {}

        store = ChromaVectorStore(
            persist_directory="./test_db",
            embedding_function=MagicMock(),
            embedding_model_name="BAAI/bge-small-en-v1.5",
        )

        docs = [MagicMock()]
        store.add_documents(docs, batch_size=500)
        mock_store.add_documents.assert_called()

    @patch("doc_helper.stores.chroma_store.Chroma")
    def test_as_retriever(self, mock_chroma_cls):
        mock_store = MagicMock()
        mock_chroma_cls.return_value = mock_store
        mock_store._collection.metadata = {}

        store = ChromaVectorStore(
            persist_directory="./test_db",
            embedding_function=MagicMock(),
            embedding_model_name="BAAI/bge-small-en-v1.5",
        )

        store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        mock_store.as_retriever.assert_called_once()

    @patch("doc_helper.stores.chroma_store.Chroma")
    def test_get_embedding_model_name_none(self, mock_chroma_cls):
        mock_store = MagicMock()
        mock_chroma_cls.return_value = mock_store
        mock_store._collection.metadata = {}

        store = ChromaVectorStore(
            persist_directory="./test_db",
            embedding_function=MagicMock(),
            embedding_model_name="BAAI/bge-small-en-v1.5",
        )
        assert store.get_embedding_model_name() is None


class TestCreateVectorStore:
    @patch("doc_helper.stores.factory.create_embeddings")
    @patch("doc_helper.stores.chroma_store.Chroma")
    def test_creates_chroma_by_default(self, mock_chroma_cls, mock_create_emb):
        mock_create_emb.return_value = MagicMock()
        mock_chroma_cls.return_value = MagicMock()
        mock_chroma_cls.return_value._collection.metadata = {}

        settings = Settings()
        store = create_vector_store(settings)
        assert isinstance(store, ChromaVectorStore)

    def test_pinecone_without_api_key_raises(self):
        settings = Settings(vector_store=VectorStoreSettings(provider="pinecone", pinecone_api_key=None))
        with pytest.raises(ValueError, match="PINECONE_API_KEY"):
            create_vector_store(settings)

    @patch("doc_helper.stores.factory.create_embeddings")
    @patch("doc_helper.stores.pinecone_store.LangChainPinecone")
    def test_creates_pinecone_with_api_key(self, mock_pinecone_cls, mock_create_emb):
        mock_create_emb.return_value = MagicMock()
        mock_pinecone_cls.return_value = MagicMock()

        settings = Settings(
            vector_store=VectorStoreSettings(provider="pinecone", pinecone_api_key="pk-test")
        )
        store = create_vector_store(settings)
        assert isinstance(store, PineconeVectorStore)