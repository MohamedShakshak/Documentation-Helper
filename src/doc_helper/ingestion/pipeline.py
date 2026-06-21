import asyncio

import click

from doc_helper.config.settings import Settings
from doc_helper.embeddings.factory import create_embeddings, get_embedding_model_name
from doc_helper.ingestion.crawlers import create_crawler
from doc_helper.ingestion.splitters.factory import create_text_splitter
from doc_helper.logger import log_header, log_info, log_success, log_warning
from doc_helper.stores.base import BaseVectorStore
from doc_helper.stores.factory import create_vector_store


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

    log_header("VECTOR STORAGE PHASE")
    log_info(f"Adding {len(split_docs)} chunks to vector store")

    store = create_vector_store(settings)
    if isinstance(store, BaseVectorStore):
        store.add_documents(split_docs, batch_size=settings.ingestion.batch_size)
    else:
        store.add_documents(split_docs)

    log_header("PIPELINE COMPLETE")
    log_success("Documentation ingestion pipeline finished!")
    log_info(f"  Documents extracted: {len(documents)}")
    log_info(f"  Chunks created: {len(split_docs)}")


@click.command()
@click.option("--url", default=None, help="URL to crawl (overrides config)")
@click.option("--depth", default=None, type=int, help="Crawl depth (overrides config)")
@click.option("--crawler", default=None, type=click.Choice(["tavily", "recursive", "local"]), help="Crawler type")
@click.option("--store", default=None, type=click.Choice(["chroma", "pinecone"]), help="Vector store type")
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