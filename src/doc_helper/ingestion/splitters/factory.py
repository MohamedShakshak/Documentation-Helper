from langchain_text_splitters import RecursiveCharacterTextSplitter

from doc_helper.config.settings import IngestionSettings


def create_text_splitter(settings: IngestionSettings | None = None) -> RecursiveCharacterTextSplitter:
    if settings is None:
        settings = IngestionSettings()

    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )