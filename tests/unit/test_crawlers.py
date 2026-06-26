from unittest.mock import MagicMock, patch

import pytest

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

    @patch("doc_helper.ingestion.crawlers.tavily_crawler.TavilyCrawl")
    def test_to_documents_metadata(self, mock_cls):
        mock_cls.return_value = MagicMock()
        settings = IngestionSettings(tavily_api_key="tvly-test")
        crawler = TavilyCrawler(settings)
        raw = [
            {"url": "https://example.com/awesome-page", "raw_content": "Content 1"},
        ]
        docs = crawler.to_documents(raw)
        assert docs[0].metadata["source_url"] == "https://example.com/awesome-page"
        assert docs[0].metadata["source_type"] == "documentation"
        assert docs[0].metadata["doc_title"] == "Awesome Page"

    @patch("doc_helper.ingestion.crawlers.tavily_crawler.TavilyCrawl")
    def test_to_documents_title_from_html(self, mock_cls):
        mock_cls.return_value = MagicMock()
        settings = IngestionSettings(tavily_api_key="tvly-test")
        crawler = TavilyCrawler(settings)
        raw = [
            {"url": "https://example.com/page", "raw_content": "<title>HTML Title</title>\n<p>content</p>"},
        ]
        docs = crawler.to_documents(raw)
        assert docs[0].metadata["doc_title"] == "HTML Title"

    @patch("doc_helper.ingestion.crawlers.tavily_crawler.TavilyCrawl")
    def test_to_documents_title_from_markdown(self, mock_cls):
        mock_cls.return_value = MagicMock()
        settings = IngestionSettings(tavily_api_key="tvly-test")
        crawler = TavilyCrawler(settings)
        raw = [
            {"url": "https://example.com/page", "raw_content": "# Markdown Title\n\ncontent"},
        ]
        docs = crawler.to_documents(raw)
        assert docs[0].metadata["doc_title"] == "Markdown Title"

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

    def test_to_documents_metadata(self):
        crawler = RecursiveCrawler()
        raw = [
            {"url": "https://example.com/my-great-doc", "raw_content": "Content 1"},
        ]
        docs = crawler.to_documents(raw)
        assert docs[0].metadata["source_url"] == "https://example.com/my-great-doc"
        assert docs[0].metadata["source_type"] == "documentation"
        assert docs[0].metadata["doc_title"] == "My Great Doc"

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

    def test_to_documents_metadata(self):
        crawler = LocalFileCrawler()
        raw = [
            {"url": "/docs/my_awesome_doc.md", "raw_content": "# My Awesome Doc\n\ncontent"},
        ]
        docs = crawler.to_documents(raw)
        assert docs[0].metadata["source_url"] == "/docs/my_awesome_doc.md"
        assert docs[0].metadata["source_type"] == "local_file"
        assert docs[0].metadata["doc_title"] == "My Awesome Doc"

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