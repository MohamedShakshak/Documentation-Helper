from langchain_core.documents import Document

from doc_helper.config.settings import IngestionSettings
from doc_helper.ingestion.splitters.factory import MarkdownSplitter, create_text_splitter


class TestCreateTextSplitter:
    def test_recursive_strategy(self):
        settings = IngestionSettings(split_strategy="recursive", chunk_size=500, chunk_overlap=50)
        splitter = create_text_splitter(settings)
        assert hasattr(splitter, "split_documents")
        assert hasattr(splitter, "split_text")

    def test_markdown_strategy_default(self):
        settings = IngestionSettings()
        assert settings.split_strategy == "markdown"
        splitter = create_text_splitter(settings)
        assert isinstance(splitter, MarkdownSplitter)

    def test_defaults_chunk_size_800(self):
        settings = IngestionSettings()
        assert settings.chunk_size == 800
        assert settings.chunk_overlap == 100

    def test_none_defaults_to_markdown(self):
        splitter = create_text_splitter(None)
        assert isinstance(splitter, MarkdownSplitter)


class TestMarkdownSplitter:
    def test_split_simple_markdown(self):
        splitter = MarkdownSplitter(
            recursive_splitter=None,
        )
        text = """# Introduction

This is the introduction section about LangChain.

## Getting Started

To get started, install LangChain with pip.

### Installation

Run `pip install langchain` to install.
"""
        docs = splitter.split_text(text)
        assert len(docs) >= 2
        assert any(d.metadata.get("parent_section") == "Introduction" for d in docs)
        assert any(d.metadata.get("section_heading") == "Getting Started" for d in docs)

    def test_split_preserves_source_metadata(self):
        splitter = MarkdownSplitter()
        doc = Document(
            page_content="# Title\n\nSome content here.\n\n## Section\n\nMore content.",
            metadata={"source_url": "https://example.com/docs", "source_type": "documentation"},
        )
        chunks = splitter.split_documents([doc])
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata.get("source_url") == "https://example.com/docs"
            assert chunk.metadata.get("source_type") == "documentation"

    def test_split_non_markdown_content(self):
        splitter = MarkdownSplitter()
        doc = Document(
            page_content="This is plain text without any markdown headers at all.",
            metadata={"source_url": "test"},
        )
        chunks = splitter.split_documents([doc])
        assert len(chunks) >= 1
        assert "plain text" in chunks[0].page_content

    def test_split_empty_content(self):
        splitter = MarkdownSplitter()
        doc = Document(page_content="", metadata={"source_url": "empty"})
        chunks = splitter.split_documents([doc])
        assert len(chunks) == 0

    def test_respects_header_hierarchy(self):
        splitter = MarkdownSplitter()
        text = """# Main Title

Content under main title.

## Subsection A

Content under subsection A.

## Subsection B

Content under subsection B.
"""
        docs = splitter.split_text(text)
        section_values = [d.metadata.get("section_heading") for d in docs if d.metadata.get("section_heading")]
        assert "Subsection A" in section_values
        assert "Subsection B" in section_values

    def test_chunk_size_respected(self):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        recursive = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        splitter = MarkdownSplitter(recursive_splitter=recursive)

        long_section = "# Title\n\n" + "This is a sentence. " * 50
        doc = Document(page_content=long_section, metadata={"source_url": "test"})
        chunks = splitter.split_documents([doc])
        assert all(len(c.page_content) <= 200 for c in chunks)

    def test_chunk_index_and_total_chunks_assigned(self):
        splitter = MarkdownSplitter()
        text = """# Title

Content under title.

## Section A

Some content.

## Section B

More content.
"""
        docs = splitter.split_text(text)
        assert len(docs) >= 2
        total = len(docs)
        for i, doc in enumerate(docs):
            assert doc.metadata["chunk_index"] == i
            assert doc.metadata["total_chunks"] == total

    def test_chunk_index_per_document(self):
        splitter = MarkdownSplitter()
        doc1 = Document(page_content="# A\n\ncontent A", metadata={"source_url": "url1"})
        doc2 = Document(page_content="# B\n\ncontent B", metadata={"source_url": "url2"})
        chunks = splitter.split_documents([doc1, doc2])

        doc1_chunks = [c for c in chunks if c.metadata.get("source_url") == "url1"]
        doc2_chunks = [c for c in chunks if c.metadata.get("source_url") == "url2"]

        for i, c in enumerate(doc1_chunks):
            assert c.metadata["chunk_index"] == i
            assert c.metadata["total_chunks"] == len(doc1_chunks)

        for i, c in enumerate(doc2_chunks):
            assert c.metadata["chunk_index"] == i
            assert c.metadata["total_chunks"] == len(doc2_chunks)