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
        assert any(d.metadata.get("header1") == "Introduction" for d in docs)
        assert any(d.metadata.get("header2") == "Getting Started" for d in docs)

    def test_split_preserves_source_metadata(self):
        splitter = MarkdownSplitter()
        doc = Document(
            page_content="# Title\n\nSome content here.\n\n## Section\n\nMore content.",
            metadata={"source": "https://example.com/docs"},
        )
        chunks = splitter.split_documents([doc])
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata.get("source") == "https://example.com/docs"

    def test_split_non_markdown_content(self):
        splitter = MarkdownSplitter()
        doc = Document(
            page_content="This is plain text without any markdown headers at all.",
            metadata={"source": "test"},
        )
        chunks = splitter.split_documents([doc])
        assert len(chunks) >= 1
        assert "plain text" in chunks[0].page_content

    def test_split_empty_content(self):
        splitter = MarkdownSplitter()
        doc = Document(page_content="", metadata={"source": "empty"})
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
        header2_values = [d.metadata.get("header2") for d in docs if d.metadata.get("header2")]
        assert "Subsection A" in header2_values
        assert "Subsection B" in header2_values

    def test_chunk_size_respected(self):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        recursive = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        splitter = MarkdownSplitter(recursive_splitter=recursive)

        long_section = "# Title\n\n" + "This is a sentence. " * 50
        doc = Document(page_content=long_section, metadata={"source": "test"})
        chunks = splitter.split_documents([doc])
        assert all(len(c.page_content) <= 200 for c in chunks)