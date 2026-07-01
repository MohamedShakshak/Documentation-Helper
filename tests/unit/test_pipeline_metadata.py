from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from doc_helper.ingestion.pipeline import (
    _compute_content_hash,
    _enrich_metadata,
    _extract_version,
    _filter_duplicates,
)
from doc_helper.stores.base import BaseVectorStore


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


class TestExtractVersion:
    def test_from_url(self):
        assert _extract_version("content", "https://docs/langchain/v0.2/") == "v0.2"

    def test_from_content(self):
        content = "# LangChain v0.3\nThis is the intro."
        assert _extract_version(content, "https://example.com") == "v0.3"

    def test_from_content_pip_style(self):
        content = "requires langchain>=0.2.15"
        assert _extract_version(content, "https://example.com") == "0.2.15"

    def test_none_when_no_version(self):
        assert _extract_version("no version here", "https://example.com") is None

    def test_url_takes_priority(self):
        content = "v0.5 in content"
        url = "https://docs/v0.2/"
        assert _extract_version(content, url) == "v0.2"


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

    def test_adds_langchain_version_from_url(self):
        docs = [
            Document(
                page_content="Some content about chains.",
                metadata={"source_url": "https://docs/v0.2/intro"},
            )
        ]
        result = _enrich_metadata(docs)
        assert result[0].metadata.get("langchain_version") == "v0.2"

    def test_adds_langchain_version_from_content(self):
        docs = [
            Document(
                page_content="langchain>=0.3.0 documentation here",
                metadata={"source_url": "https://example.com"},
            )
        ]
        result = _enrich_metadata(docs)
        assert result[0].metadata.get("langchain_version") == "0.3.0"


class TestFilterDuplicates:
    def test_no_duplicates_returns_all(self):
        store = MagicMock(spec=BaseVectorStore)
        store.get_existing_hashes.return_value = set()
        docs = [
            Document(page_content="A", metadata={"content_hash": "hash_a"}),
            Document(page_content="B", metadata={"content_hash": "hash_b"}),
        ]
        unique, skipped = _filter_duplicates(docs, store)
        assert len(unique) == 2
        assert skipped == 0

    def test_skips_duplicates(self):
        store = MagicMock(spec=BaseVectorStore)
        store.get_existing_hashes.return_value = {"hash_a"}
        docs = [
            Document(page_content="A", metadata={"content_hash": "hash_a"}),
            Document(page_content="B", metadata={"content_hash": "hash_b"}),
        ]
        unique, skipped = _filter_duplicates(docs, store)
        assert len(unique) == 1
        assert skipped == 1
        assert unique[0].metadata["content_hash"] == "hash_b"

    def test_handles_store_error_gracefully(self):
        store = MagicMock(spec=BaseVectorStore)
        store.get_existing_hashes.side_effect = Exception("store error")
        docs = [
            Document(page_content="A", metadata={"content_hash": "hash_a"}),
            Document(page_content="A", metadata={"content_hash": "hash_a"}),
        ]
        unique, skipped = _filter_duplicates(docs, store)
        assert len(unique) == 1
        assert skipped == 1  # within-batch dedup still works

    def test_dedup_within_batch(self):
        store = MagicMock(spec=BaseVectorStore)
        store.get_existing_hashes.return_value = set()
        docs = [
            Document(page_content="A", metadata={"content_hash": "hash_a"}),
            Document(page_content="A", metadata={"content_hash": "hash_a"}),
            Document(page_content="B", metadata={"content_hash": "hash_b"}),
        ]
        unique, skipped = _filter_duplicates(docs, store)
        assert len(unique) == 2
        assert skipped == 1


class TestPipelineIntegration:
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