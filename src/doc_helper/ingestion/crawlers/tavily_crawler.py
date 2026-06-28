import re
from typing import Any
from urllib.parse import urlparse

from langchain_core.documents import Document
from langchain_tavily import TavilyCrawl

from doc_helper.config.settings import IngestionSettings
from doc_helper.ingestion.crawlers.base import BaseCrawler
from doc_helper.logger import log_header, log_info


def _extract_title_from_url(url: str) -> str:
    path = urlparse(url).path
    slug = path.rstrip("/").split("/")[-1]
    slug = re.sub(r"[-_]", " ", slug)
    return slug.title() if slug else url


def _extract_title_from_html(content: str) -> str | None:
    match = re.search(r"<title[^>]*>(.*?)</title>", content, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _extract_title_from_markdown(content: str) -> str | None:
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


class TavilyCrawler(BaseCrawler):
    def __init__(self, settings: IngestionSettings | None = None):
        self._settings = settings or IngestionSettings()
        if not self._settings.tavily_api_key:
            raise ValueError("INGESTION__TAVILY_API_KEY is required for Tavily crawler")
        self._client = TavilyCrawl(api_key=self._settings.tavily_api_key)

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
                    _extract_title_from_html(content)
                    or _extract_title_from_markdown(content)
                    or _extract_title_from_url(url)
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
