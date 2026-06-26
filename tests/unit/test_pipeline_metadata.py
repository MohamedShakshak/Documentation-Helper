from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from doc_helper.ingestion.pipeline import (
    _compute_content_hash,
    _enrich_metadata,
    _filter_duplicates,
)


class TestComputeContentHash:
    def test_hash_is_deterministic(self):
        h1 = _compute_content_hash("hello world")
        h2 = _compute_content_hash("hello world")
        assert h1 == h2

    def test_different_content_different_hash(self):
        h1 = _compute_content_hash("hello")
        h2 = _compute_content_hash("world")
        assert h1 != h2

    def test_hash_prefixed(self):
        h = _compute_content_hash("test")
        assert h.startswith("sha256:")


class TestEnrichMetadata:
    def test_adds_required_fields(self):
        docs = [Document(page_content="Hello world from docs", metadata={})]
        result = _enrich_metadata(docs)
        assert "ingested_at" in result[0].metadata
        assert "content_hash" in result[0].metadata
        assert result[0].metadata["word_count"] == 4

    def test_ingested_at_is_iso(self):
        docs = [Document(page_content="test", metadata={})]
        result = _enrich_metadata(docs)
        parsed = datetime.fromisoformat(result[0].metadata["ingested_at"])
        assert parsed is not None

    def test_preserves_existing_metadata(self):
        docs = [Document(page_content="test", metadata={"source_url": "https://example.com"})]
        result = _enrich_metadata(docs)
        assert result[0].metadata["source_url"] == "https://example.com"
        assert "content_hash" in result[0].metadata

    def test_word_count_simple(self):
        docs = [Document(page_content="one two three four", metadata={})]
        result = _enrich_metadata(docs)
        assert result[0].metadata["word_count"] == 4


class TestFilterDuplicates:
    def test_no_duplicates_returns_all(self):
        store = MagicMock()
        store.as_retriever.return_value.invoke.return_value = []
        docs = [
            Document(page_content="A", metadata={"content_hash": "hash_a"}),
            Document(page_content="B", metadata={"content_hash": "hash_b"}),
        ]
        unique, skipped = _filter_duplicates(docs, store)
        assert len(unique) == 2
        assert skipped == 0

    def test_skips_duplicates(self):
        store = MagicMock()
        store.as_retriever.return_value.invoke.return_value = [
            Document(page_content="A", metadata={"content_hash": "hash_a"}),
        ]
        docs = [
            Document(page_content="A", metadata={"content_hash": "hash_a"}),
            Document(page_content="B", metadata={"content_hash": "hash_b"}),
        ]
        unique, skipped = _filter_duplicates(docs, store)
        assert len(unique) == 1
        assert skipped == 1
        assert unique[0].metadata["content_hash"] == "hash_b"

    def test_handles_store_error_gracefully(self):
        store = MagicMock()
        store.as_retriever.return_value.invoke.side_effect = Exception("store error")
        docs = [
            Document(page_content="A", metadata={"content_hash": "hash_a"}),
            Document(page_content="A", metadata={"content_hash": "hash_a"}),
        ]
        unique, skipped = _filter_duplicates(docs, store)
        assert len(unique) == 1
        assert skipped == 1

    def test_dedup_within_batch(self):
        store = MagicMock()
        store.as_retriever.return_value.invoke.return_value = []
        docs = [
            Document(page_content="A", metadata={"content_hash": "hash_a"}),
            Document(page_content="A", metadata={"content_hash": "hash_a"}),
            Document(page_content="B", metadata={"content_hash": "hash_b"}),
        ]
        unique, skipped = _filter_duplicates(docs, store)
        assert len(unique) == 2
        assert skipped == 1


class TestPipelineIntegration:
    """End-to-end metadata pipeline test (no real vector store)."""

    def test_full_metadata_enrichment_flow(self):
        doc = Document(
            page_content="# Guide\n\n## Section\n\nSome content here.",
            metadata={"source_url": "https://example.com", "source_type": "documentation"},
        )
        enriched = _enrich_metadata([doc])
        assert enriched[0].metadata["content_hash"].startswith("sha256:")
        assert enriched[0].metadata["word_count"] > 0
        assert "ingested_at" in enriched[0].metadata
        assert enriched[0].metadata["source_url"] == "https://example.com"