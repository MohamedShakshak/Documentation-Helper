import asyncio
import hashlib
from datetime import UTC, datetime

import click

from doc_helper.config.settings import Settings
from doc_helper.ingestion.crawlers import create_crawler
from doc_helper.ingestion.splitters.factory import create_text_splitter
from doc_helper.logger import log_header, log_info, log_success, log_warning
from doc_helper.stores.base import BaseVectorStore
from doc_helper.stores.factory import create_vector_store


def _compute_content_hash(content: str) -> str:
    return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"


def _extract_version(content: str, url: str | None = None) -> str | None:
    import re

    if url:
        match = re.search(r"v(\d+\.\d+)", url)
        if match:
            return match.group(0)
    match = re.search(r"langchain[=>]=\s*(\d+\.\d+\.\d+)", content[:500])
    if match:
        return match.group(1)
    match = re.search(r"v(\d+\.\d+)", content[:500])
    if match:
        return match.group(0)
    return None


def _enrich_metadata(documents: list) -> list:
    now = datetime.now(UTC).isoformat()
    for doc in documents:
        doc.metadata["ingested_at"] = now
        doc.metadata["content_hash"] = _compute_content_hash(doc.page_content)
        doc.metadata["word_count"] = len(doc.page_content.split())
        version = _extract_version(doc.page_content, doc.metadata.get("source_url"))
        if version:
            doc.metadata["langchain_version"] = version
    return documents


def _filter_duplicates(documents: list, store: BaseVectorStore) -> tuple[list, int]:
    try:
        existing_hashes = store.get_existing_hashes()
    except Exception:
        existing_hashes = set()
    unique: list = []
    skipped = 0
    for doc in documents:
        h = doc.metadata.get("content_hash")
        if h and h in existing_hashes:
            skipped += 1
        else:
            unique.append(doc)
            if h:
                existing_hashes.add(h)
    return unique, skipped


async def run_ingestion(settings: Settings | None = None) -> None:
    if settings is None:
        settings = Settings()

    log_header("DOCUMENTATION INGESTION PIPELINE")

    crawler = create_crawler(settings.ingestion)
    raw_results = await crawler.crawl()
    documents = crawler.to_documents(raw_results)

    if not documents:
        log_warning("No documents found. Exiting.")
        return

    log_header("DOCUMENT CHUNKING PHASE")
    log_info(f"Processing {len(documents)} documents with "
             f"chunk_size={settings.ingestion.chunk_size}, "
             f"chunk_overlap={settings.ingestion.chunk_overlap}")

    splitter = create_text_splitter(settings.ingestion)
    split_docs = splitter.split_documents(documents)
    log_success(f"Created {len(split_docs)} chunks from {len(documents)} documents")

    log_header("METADATA ENRICHMENT PHASE")
    split_docs = _enrich_metadata(split_docs)
    log_info(f"Enriched {len(split_docs)} chunks with ingested_at, content_hash, word_count")

    log_header("VECTOR STORAGE PHASE")
    store = create_vector_store(settings)

    if isinstance(store, BaseVectorStore):
        unique_docs, skipped = _filter_duplicates(split_docs, store)
        if skipped > 0:
            log_warning(f"Deduplication: skipped {skipped} duplicate chunks")
        if not unique_docs:
            log_warning("All chunks already exist in store. Nothing to add.")
            return
        log_info(f"Adding {len(unique_docs)} chunks to vector store")
        store.add_documents(unique_docs, batch_size=settings.ingestion.batch_size)
    else:
        store.add_documents(split_docs)

    dedup_count = len(split_docs) - len(unique_docs) if isinstance(store, BaseVectorStore) else 0

    log_header("PIPELINE COMPLETE")
    log_success("Documentation ingestion pipeline finished!")
    log_info(f"  Documents extracted: {len(documents)}")
    log_info(f"  Chunks created: {len(split_docs)}")
    log_info(f"  Duplicates skipped: {dedup_count}")


@click.command()
@click.option("--url", default=None, help="URL to crawl (overrides config)")
@click.option("--depth", default=None, type=int, help="Crawl depth (overrides config)")
@click.option(
    "--crawler",
    default=None,
    type=click.Choice(["tavily", "recursive", "local"]),
    help="Crawler type",
)
@click.option(
    "--store",
    default=None,
    type=click.Choice(["chroma", "pinecone"]),
    help="Vector store type",
)
def cli(url, depth, crawler, store):
    settings = Settings()

    if url:
        settings.ingestion.crawl_url = url
    if depth:
        settings.ingestion.crawl_depth = depth
    if crawler:
        settings.ingestion.crawler = crawler
    if store:
        settings.vector_store.provider = store

    asyncio.run(run_ingestion(settings))


if __name__ == "__main__":
    cli()
