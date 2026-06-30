from typing import Any

import html2text
from langchain_core.documents import Document

from doc_helper.config.settings import IngestionSettings
from doc_helper.ingestion.crawlers.base import BaseCrawler
from doc_helper.ingestion.crawlers.tavily_crawler import (
    _extract_title_from_markdown,
    _extract_title_from_url,
)
from doc_helper.logger import log_header, log_info


def _html_to_markdown(html: str) -> str:
    converter = html2text.HTML2Text()
    converter.body_width = 0
    converter.ignore_links = False
    converter.ignore_images = True
    converter.ignore_tables = False
    converter.protect_links = True
    return converter.handle(html)


class RecursiveCrawler(BaseCrawler):
    def __init__(self, settings: IngestionSettings | None = None):
        self._settings = settings or IngestionSettings()

    async def crawl(self) -> list[dict[str, Any]]:
        log_header("RECURSIVE URL CRAWL")
        log_info(f"Crawling {self._settings.crawl_url} (depth={self._settings.crawl_depth})")

        from langchain_community.document_loaders import RecursiveUrlLoader

        loader = RecursiveUrlLoader(
            url=self._settings.crawl_url,
            max_depth=self._settings.crawl_depth,
            extractor=_html_to_markdown,
        )
        docs = loader.load()
        return [
            {"url": doc.metadata.get("source", "Unknown"), "raw_content": doc.page_content}
            for doc in docs
        ]

    def to_documents(self, raw_results: list[dict[str, Any]]) -> list[Document]:
        documents = []
        for item in raw_results:
            url = item.get("url", "Unknown")
            content = item.get("raw_content", "")
            if content:
                doc_title = (
                    _extract_title_from_markdown(content)
                    or _extract_title_from_url(url)
                )
                log_info(f"RecursiveCrawl: Loaded {url}")
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source_url": url,
                            "source_type": "documentation",
                            "doc_title": doc_title,
                        },
                    )
                )
        return documents