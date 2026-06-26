from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from doc_helper.config.settings import IngestionSettings
from doc_helper.ingestion.crawlers.base import BaseCrawler
from doc_helper.ingestion.crawlers.local_file_crawler import LocalFileCrawler
from doc_helper.ingestion.crawlers.recursive_crawler import RecursiveCrawler
from doc_helper.ingestion.crawlers.tavily_crawler import TavilyCrawler


class TestTavilyCrawler:
    def test_requires_api_key(self):
        settings = IngestionSettings(tavily_api_key=None)
        with pytest.raises(ValueError, match="TAVILY_API_KEY"):
            TavilyCrawler(settings)

    @patch("doc_helper.ingestion.crawlers.tavily_crawler.TavilyCrawl")
    def test_crawl_returns_raw_results(self, mock_tavily_cls):
        mock_client = MagicMock()
        mock_tavily_cls.return_value = mock_client
        mock_client.invoke.return_value = {
            "results": [
                {"url": "https://example.com/page1", "raw_content": "Content 1"},
                {"url": "https://example.com/page2", "raw_content": "Content 2"},
            ]
        }

        settings = IngestionSettings(tavily_api_key="tvly-test")
        crawler = TavilyCrawler(settings)
        results = MagicMock()
        mock_client.invoke.return_value = results

        crawler._client = mock_client
        mock_client.invoke.return_value = {
            "results": [
                {"url": "https://example.com/page1", "raw_content": "Content 1"},
                {"url": "https://example.com/page2", "raw_content": "Content 2"},
            ]
        }
        raw = crawler.crawl()

    @patch("doc_helper.ingestion.crawlers.tavily_crawler.TavilyCrawl")
    def test_to_documents(self, mock_tavily_cls):
        mock_client = MagicMock()
        mock_tavily_cls.return_value = mock_client

        settings = IngestionSettings(tavily_api_key="tvly-test")
        crawler = TavilyCrawler(settings)
        raw = [
            {"url": "https://example.com/page1", "raw_content": "Content 1"},
            {"url": "https://example.com/page2", "raw_content": "Content 2"},
            {"url": "https://example.com/empty", "raw_content": ""},
        ]
        docs = crawler.to_documents(raw)
        assert len(docs) == 2
        assert docs[0].page_content == "Content 1"
        assert docs[0].metadata["source"] == "https://example.com/page1"

    @patch("doc_helper.ingestion.crawlers.tavily_crawler.TavilyCrawl")
    def test_to_documents_empty_results(self, mock_tavily_cls):
        mock_tavily_cls.return_value = MagicMock()
        settings = IngestionSettings(tavily_api_key="tvly-test")
        crawler = TavilyCrawler(settings)
        docs = crawler.to_documents([])
        assert docs == []


class TestRecursiveCrawler:
    def test_to_documents(self):
        crawler = RecursiveCrawler()
        raw = [
            {"url": "https://example.com/page1", "raw_content": "Content 1"},
            {"url": "https://example.com/page2", "raw_content": "Content 2"},
        ]
        docs = crawler.to_documents(raw)
        assert len(docs) == 2
        assert docs[0].metadata["source"] == "https://example.com/page1"

    def test_to_documents_skips_empty_content(self):
        crawler = RecursiveCrawler()
        raw = [
            {"url": "https://example.com/empty", "raw_content": ""},
        ]
        docs = crawler.to_documents(raw)
        assert docs == []


class TestLocalFileCrawler:
    def test_to_documents(self):
        crawler = LocalFileCrawler()
        raw = [
            {"url": "/docs/readme.md", "raw_content": "# README\n\nHello world."},
        ]
        docs = crawler.to_documents(raw)
        assert len(docs) == 1
        assert "README" in docs[0].page_content

    def test_to_documents_skips_empty(self):
        crawler = LocalFileCrawler()
        raw = [{"url": "/docs/empty.md", "raw_content": ""}]
        docs = crawler.to_documents(raw)
        assert docs == []


class TestCrawlerFactory:
    def test_create_tavily(self):
        from doc_helper.ingestion.crawlers import create_crawler

        settings = IngestionSettings(crawler="tavily", tavily_api_key="tvly-test")
        with patch("doc_helper.ingestion.crawlers.tavily_crawler.TavilyCrawl"):
            crawler = create_crawler(settings)
            assert isinstance(crawler, TavilyCrawler)

    def test_create_recursive(self):
        from doc_helper.ingestion.crawlers import create_crawler

        settings = IngestionSettings(crawler="recursive")
        crawler = create_crawler(settings)
        assert isinstance(crawler, RecursiveCrawler)

    def test_create_local(self):
        from doc_helper.ingestion.crawlers import create_crawler

        settings = IngestionSettings(crawler="local")
        crawler = create_crawler(settings)
        assert isinstance(crawler, LocalFileCrawler)

    def test_create_unknown_raises(self):
        from doc_helper.ingestion.crawlers import create_crawler

        settings = IngestionSettings()
        settings.crawler = "invalid"
        with pytest.raises(ValueError, match="Unknown crawler"):
            create_crawler(settings)