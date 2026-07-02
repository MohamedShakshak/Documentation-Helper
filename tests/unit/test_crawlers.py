from unittest.mock import MagicMock, patch

import pytest

from doc_helper.config.settings import IngestionSettings
from doc_helper.ingestion.crawlers.local_file_crawler import LocalFileCrawler
from doc_helper.ingestion.crawlers.recursive_crawler import RecursiveCrawler
from doc_helper.ingestion.crawlers.sitemap_crawler import SitemapCrawler, _parse_sitemap
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

    def test_to_documents_metadata(self):
        settings = IngestionSettings(tavily_api_key="tvly-test")
        crawler = TavilyCrawler(settings)
        raw = [
            {"url": "https://example.com/awesome-page", "raw_content": "Content 1"},
        ]
        docs = crawler.to_documents(raw)
        assert docs[0].metadata["source_url"] == "https://example.com/awesome-page"
        assert docs[0].metadata["source_type"] == "documentation"
        assert docs[0].metadata["doc_title"] == "Awesome Page"

    def test_to_documents_title_from_html(self):
        settings = IngestionSettings(tavily_api_key="tvly-test")
        crawler = TavilyCrawler(settings)
        raw = [
            {
                "url": "https://example.com/page",
                "raw_content": "<title>HTML Title</title>\n<p>content</p>",
            },
        ]
        docs = crawler.to_documents(raw)
        assert docs[0].metadata["doc_title"] == "HTML Title"

    def test_to_documents_title_from_markdown(self):
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
    LONG_CONTENT = "This is a long enough content that passes the minimum length check. " * 5

    def test_to_documents(self):
        crawler = RecursiveCrawler()
        raw = [
            {"url": "https://example.com/page1", "raw_content": self.LONG_CONTENT},
            {"url": "https://example.com/page2", "raw_content": self.LONG_CONTENT},
        ]
        docs = crawler.to_documents(raw)
        assert len(docs) == 2

    def test_to_documents_metadata(self):
        crawler = RecursiveCrawler()
        raw = [
            {"url": "https://example.com/my-great-doc", "raw_content": self.LONG_CONTENT},
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

    def test_to_documents_skips_short_content(self):
        crawler = RecursiveCrawler()
        raw = [
            {"url": "https://example.com/short", "raw_content": "\n\n\n"},
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


class TestSitemapCrawler:
    SITEMAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://docs.langchain.com/oss/python/langchain/overview</loc></url>
  <url><loc>https://docs.langchain.com/oss/python/langchain/agents</loc></url>
  <url><loc>https://docs.langchain.com/oss/python/langchain/concepts</loc></url>
  <url><loc>https://docs.langchain.com/oss/python/langgraph/overview</loc></url>
  <url><loc>https://docs.langchain.com/oss/python/langsmith/intro</loc></url>
</urlset>"""

    def test_parse_sitemap_filters_by_prefix(self):
        urls = _parse_sitemap(self.SITEMAP_XML, "https://docs.langchain.com/oss/python/langchain")
        assert len(urls) == 3
        assert all(u.startswith("https://docs.langchain.com/oss/python/langchain") for u in urls)

    def test_parse_sitemap_handles_trailing_slash(self):
        urls = _parse_sitemap(self.SITEMAP_XML, "https://docs.langchain.com/oss/python/langchain/")
        assert len(urls) == 3

    def test_parse_sitemap_empty_results(self):
        urls = _parse_sitemap(self.SITEMAP_XML, "https://docs.langchain.com/nonexistent")
        assert urls == []

    def test_to_documents(self):
        crawler = SitemapCrawler()
        long_content = "# Overview\n\nLangChain is a framework for building LLM applications. " * 3
        raw = [
            {
                "url": "https://docs.langchain.com/oss/python/langchain/overview",
                "raw_content": long_content,
            },
            {
                "url": "https://docs.langchain.com/oss/python/langchain/agents",
                "raw_content": "# Agents\n\nAgents use tools to accomplish tasks. " * 3,
            },
            {
                "url": "https://docs.langchain.com/oss/python/langchain/empty",
                "raw_content": "",
            },
        ]
        docs = crawler.to_documents(raw)
        assert len(docs) == 2
        assert docs[0].metadata["source_url"] == "https://docs.langchain.com/oss/python/langchain/overview"
        assert docs[0].metadata["source_type"] == "documentation"
        assert docs[0].metadata["doc_title"] == "Overview"

    def test_to_documents_skips_short_content(self):
        crawler = SitemapCrawler()
        raw = [
            {"url": "https://example.com/short", "raw_content": "abc"},
        ]
        docs = crawler.to_documents(raw)
        assert docs == []

    def test_to_documents_title_fallback_to_url(self):
        crawler = SitemapCrawler()
        raw = [
            {
                "url": "https://docs.langchain.com/oss/python/langchain/awesome-page",
                "raw_content": "No heading here but enough text to pass the length check. " * 3,
            },
        ]
        docs = crawler.to_documents(raw)
        assert docs[0].metadata["doc_title"] == "Awesome Page"


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

    def test_create_sitemap(self):
        from doc_helper.ingestion.crawlers import create_crawler

        settings = IngestionSettings(crawler="sitemap")
        crawler = create_crawler(settings)
        assert isinstance(crawler, SitemapCrawler)

    def test_create_unknown_raises(self):
        from doc_helper.ingestion.crawlers import create_crawler

        settings = IngestionSettings()
        settings.crawler = "invalid"
        with pytest.raises(ValueError, match="Unknown crawler"):
            create_crawler(settings)
