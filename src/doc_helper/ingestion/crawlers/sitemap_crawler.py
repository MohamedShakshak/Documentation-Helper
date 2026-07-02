import asyncio
import re
from typing import Any

import httpx
from langchain_core.documents import Document

from doc_helper.config.settings import IngestionSettings
from doc_helper.ingestion.crawlers.base import BaseCrawler
from doc_helper.logger import log_header, log_info, log_success, log_warning
from doc_helper.utils import extract_title_from_markdown, extract_title_from_url

_SITEMAP_URL = "https://docs.langchain.com/sitemap.xml"
_LOC_RE = re.compile(r"<loc>(.*?)</loc>", re.DOTALL)
_MAX_CONCURRENT = 5


def _parse_sitemap(content: str, prefix: str) -> list[str]:
    urls = _LOC_RE.findall(content)
    base = prefix.rstrip("/")
    return [u for u in urls if u.rstrip("/").startswith(base)]


def _strip_doc_index(content: str) -> str:
    if content.startswith("> ## Documentation Index"):
        end = content.find("\n#")
        if end != -1:
            return content[end + 1 :]
    return content


class SitemapCrawler(BaseCrawler):
    def __init__(self, settings: IngestionSettings | None = None):
        self._settings = settings or IngestionSettings()
        self._base_url = self._settings.crawl_url.rstrip("/")
        self._sitemap_url = _SITEMAP_URL

    async def _fetch_page(
        self,
        client: httpx.AsyncClient,
        url: str,
        idx: int,
        total: int,
        sem: asyncio.Semaphore,
    ) -> dict[str, Any] | None:
        md_url = url + ".md"
        async with sem:
            try:
                resp = await client.get(md_url, follow_redirects=True)
                resp.raise_for_status()
                content = _strip_doc_index(resp.text)
                log_info(f"[{idx}/{total}] Fetched {url}")
                return {"url": url, "raw_content": content}
            except Exception as e:
                log_warning(f"[{idx}/{total}] Failed {md_url}: {e}")
                return None

    async def crawl(self) -> list[dict[str, Any]]:
        log_header("SITEMAP CRAWL")
        log_info(f"Fetching sitemap from {self._sitemap_url}")

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(self._sitemap_url)
            resp.raise_for_status()
            urls = _parse_sitemap(resp.text, self._base_url)

        log_info(f"Found {len(urls)} URLs matching {self._base_url}")

        if not urls:
            log_warning("No URLs found. Check crawl_url setting.")
            return []

        sem = asyncio.Semaphore(_MAX_CONCURRENT)
        async with httpx.AsyncClient(timeout=30) as client:
            tasks = [
                self._fetch_page(client, url, i, len(urls), sem)
                for i, url in enumerate(urls, 1)
            ]
            raw_results = await asyncio.gather(*tasks)

        results = [r for r in raw_results if r is not None]
        log_success(f"Crawled {len(results)}/{len(urls)} pages")
        return results

    def to_documents(self, raw_results: list[dict[str, Any]]) -> list[Document]:
        documents = []
        for item in raw_results:
            url = item.get("url", "Unknown")
            content = item.get("raw_content", "")
            if content and len(content.strip()) > 100:
                doc_title = extract_title_from_markdown(content) or extract_title_from_url(url)
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
