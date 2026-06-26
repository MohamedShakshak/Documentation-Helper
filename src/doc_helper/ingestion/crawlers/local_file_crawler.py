from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from doc_helper.ingestion.crawlers.tavily_crawler import _extract_title_from_markdown
from doc_helper.config.settings import IngestionSettings
from doc_helper.ingestion.crawlers.base import BaseCrawler
from doc_helper.logger import log_info, log_header


class LocalFileCrawler(BaseCrawler):
    SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".html", ".rst"}

    def __init__(self, settings: IngestionSettings | None = None):
        self._settings = settings or IngestionSettings()

    async def crawl(self) -> list[dict[str, Any]]:
        log_header("LOCAL FILE CRAWL")
        docs_dir = Path(self._settings.local_docs_dir)
        if not docs_dir.exists():
            raise FileNotFoundError(f"Local docs directory not found: {docs_dir}")

        from langchain_community.document_loaders import DirectoryLoader

        loader = DirectoryLoader(
            str(docs_dir),
            glob="**/*",
            show_progress=True,
        )
        docs = loader.load()
        log_info(f"Loaded {len(docs)} documents from {docs_dir}")
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
                doc_title = _extract_title_from_markdown(content) or Path(url).stem.replace("_", " ").title()
                log_info(f"LocalFile: Loaded {url}")
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source_url": url,
                            "source_type": "local_file",
                            "doc_title": doc_title,
                        },
                    )
                )
        return documents