from unittest.mock import MagicMock, patch

import pytest

from doc_helper.config.settings import RetrievalSettings
from doc_helper.retrieval.reranker import Reranker
from doc_helper.retrieval.retriever import Retriever
from doc_helper.stores.base import BaseVectorStore


class TestRetriever:
    def test_retrieve_similarity(self):
        mock_store = MagicMock(spec=BaseVectorStore)
        mock_retriever = MagicMock()
        mock_store.as_retriever.return_value = mock_retriever

        doc = MagicMock()
        doc.page_content = "test"
        doc.metadata = {"source": "test"}
        mock_retriever.invoke.return_value = [doc]

        settings = RetrievalSettings(search_type="similarity", search_k=4)
        retriever = Retriever(mock_store, settings)
        results = retriever.retrieve("test query")

        assert len(results) == 1
        mock_store.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

    def test_retrieve_mmr(self):
        mock_store = MagicMock(spec=BaseVectorStore)
        mock_retriever = MagicMock()
        mock_store.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = []

        settings = RetrievalSettings(search_type="mmr", search_k=6)
        retriever = Retriever(mock_store, settings)
        retriever.retrieve("test")

        mock_store.as_retriever.assert_called_once_with(
            search_type="mmr",
            search_kwargs={"k": 6},
        )

    def test_retrieve_with_score_threshold(self):
        mock_store = MagicMock(spec=BaseVectorStore)
        mock_retriever = MagicMock()
        mock_store.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = []

        settings = RetrievalSettings(
            search_type="similarity_score_threshold",
            search_k=4,
            score_threshold=0.7,
        )
        retriever = Retriever(mock_store, settings)
        retriever.retrieve("test")

        mock_store.as_retriever.assert_called_once_with(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 4, "score_threshold": 0.7},
        )

    def test_retrieve_with_reranker_enabled(self):
        mock_store = MagicMock(spec=BaseVectorStore)
        mock_retriever = MagicMock()
        mock_store.as_retriever.return_value = mock_retriever

        doc1 = MagicMock()
        doc1.page_content = "doc1"
        doc1.metadata = {"source": "a"}
        doc2 = MagicMock()
        doc2.page_content = "doc2"
        doc2.metadata = {"source": "b"}
        mock_retriever.invoke.return_value = [doc1, doc2]

        with patch("doc_helper.retrieval.retriever.Reranker") as mock_reranker_cls:
            mock_reranker_instance = MagicMock()
            mock_reranker_cls.return_value = mock_reranker_instance
            mock_reranker_instance.rerank.return_value = [doc2, doc1]

            settings = RetrievalSettings(reranker_enabled=True)
            retriever = Retriever(mock_store, settings)
            results = retriever.retrieve("test")

            mock_reranker_instance.rerank.assert_called_once()


class TestReranker:
    def test_reranker_default_model(self):
        reranker = Reranker()
        assert reranker._model == "ms-marco-MiniLM-L-12-v2"

    def test_reranker_custom_model(self):
        reranker = Reranker(model="custom-model")
        assert reranker._model == "custom-model"

    def test_reranker_custom_top_k(self):
        reranker = Reranker(top_k=3)
        assert reranker._top_k == 3