from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from doc_helper.config.settings import IngestionSettings


def create_text_splitter(settings: IngestionSettings | None = None):
    if settings is None:
        settings = IngestionSettings()

    recursive = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if settings.split_strategy == "recursive":
        return recursive

    return MarkdownSplitter(recursive_splitter=recursive)


class MarkdownSplitter:
    def __init__(
        self,
        recursive_splitter: RecursiveCharacterTextSplitter | None = None,
        headers_to_split_on: list[tuple[str, str]] | None = None,
    ):
        self._recursive = recursive_splitter or RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=100
        )
        self._markdown = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
            or [
                ("#", "header1"),
                ("##", "header2"),
                ("###", "header3"),
            ],
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        result: list[Document] = []
        for doc in documents:
            md_docs = self._markdown.split_text(doc.page_content)
            for md_doc in md_docs:
                metadata = {**doc.metadata, **md_doc.metadata}
                chunks = self._recursive.split_documents(
                    [Document(page_content=md_doc.page_content, metadata=metadata)]
                )
                result.extend(chunks)
        return result

    def split_text(self, text: str) -> list[Document]:
        md_docs = self._markdown.split_text(text)
        result: list[Document] = []
        for md_doc in md_docs:
            chunks = self._recursive.split_documents([md_doc])
            result.extend(chunks)
        return result