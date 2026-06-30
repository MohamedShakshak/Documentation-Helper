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
    HEADER_TO_FIELD = {
        "header1": "parent_section",
        "header2": "section_heading",
        "header3": "subsection",
        "header4": "detail",
    }

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
                ("####", "header4"),
            ],
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        result: list[Document] = []
        for doc in documents:
            md_docs = self._markdown.split_text(doc.page_content)
            chunks: list[Document] = []
            for md_doc in md_docs:
                metadata = {**doc.metadata, **self._remap_headers(md_doc.metadata)}
                split_chunks = self._recursive.split_documents(
                    [Document(page_content=md_doc.page_content, metadata=metadata)]
                )
                chunks.extend(split_chunks)
            total = len(chunks)
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["total_chunks"] = total
            result.extend(chunks)
        return result

    def split_text(self, text: str) -> list[Document]:
        md_docs = self._markdown.split_text(text)
        chunks: list[Document] = []
        for md_doc in md_docs:
            metadata = self._remap_headers(md_doc.metadata)
            split_chunks = self._recursive.split_documents([md_doc])
            for chunk in split_chunks:
                chunk.metadata = {**chunk.metadata, **metadata}
            chunks.extend(split_chunks)
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = total
        return chunks

    def _remap_headers(self, md_metadata: dict) -> dict:
        remapped: dict = {}
        for key, value in md_metadata.items():
            field = self.HEADER_TO_FIELD.get(key, key)
            if field in remapped and not remapped[field]:
                remapped[field] = value
            elif field not in remapped:
                remapped[field] = value
        return remapped