from typing import Any

from langchain_core.documents import Document
from langchain_tavily import TavilyCrawl

from doc_helper.config.settings import IngestionSettings
from doc_helper.ingestion.crawlers.base import BaseCrawler
from doc_helper.logger import log_header, log_info
from doc_helper.utils import (
    extract_title_from_html,
    extract_title_from_markdown,
    extract_title_from_url,
)


class TavilyCrawler(BaseCrawler):
    def __init__(self, settings: IngestionSettings | None = None):
        self._settings = settings or IngestionSettings()
        if not self._settings.tavily_api_key:
            raise ValueError("INGESTION__TAVILY_API_KEY is required for Tavily crawler")
        self._client = TavilyCrawl(tavily_api_key=self._settings.tavily_api_key)

    async def crawl(self) -> list[dict[str, Any]]:
        log_header("TAVILY CRAWL")
        log_info(f"Crawling {self._settings.crawl_url} (depth={self._settings.crawl_depth})")

        result = self._client.invoke(
            {
                "url": self._settings.crawl_url,
                "max_depth": self._settings.crawl_depth,
                "extract_depth": "advanced",
            }
        )
        return result.get("results", [])

    def to_documents(self, raw_results: list[dict[str, Any]]) -> list[Document]:
        documents = []
        for item in raw_results:
            url = item.get("url", "Unknown")
            content = item.get("raw_content", "")
            if content:
                doc_title = (
                    extract_title_from_html(content)
                    or extract_title_from_markdown(content)
                    or extract_title_from_url(url)
                )
                log_info(f"TavilyCrawl: Crawled {url}")
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