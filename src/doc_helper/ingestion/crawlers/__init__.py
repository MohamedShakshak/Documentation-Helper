from doc_helper.config.settings import IngestionSettings
from doc_helper.ingestion.crawlers.base import BaseCrawler
from doc_helper.ingestion.crawlers.local_file_crawler import LocalFileCrawler
from doc_helper.ingestion.crawlers.recursive_crawler import RecursiveCrawler
from doc_helper.ingestion.crawlers.tavily_crawler import TavilyCrawler


def create_crawler(settings: IngestionSettings | None = None) -> BaseCrawler:
    if settings is None:
        settings = IngestionSettings()

    if settings.crawler == "tavily":
        return TavilyCrawler(settings)
    elif settings.crawler == "recursive":
        return RecursiveCrawler(settings)
    elif settings.crawler == "local":
        return LocalFileCrawler(settings)

    raise ValueError(
        f"Unknown crawler '{settings.crawler}'. Available: tavily, recursive, local"
    )
