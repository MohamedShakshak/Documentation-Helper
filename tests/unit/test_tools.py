import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from doc_helper.config.settings import IngestionSettings
from doc_helper.retrieval.retriever import Retriever
from doc_helper.stores.base import BaseVectorStore


class TestRetrieveContextTool:
    def test_makes_tool_from_retriever(self):
        mock_store = MagicMock(spec=BaseVectorStore)
        retriever = Retriever(mock_store)
        from doc_helper.agents.tools import _make_retrieval_tool

        tool = _make_retrieval_tool(retriever)
        assert tool.name == "retrieve_context"
        assert "content_and_artifact" in tool.response_format

    @patch("doc_helper.retrieval.reranker.Reranker")
    def test_retrieve_context_serializes_docs(self, _mock_reranker):
        from doc_helper.agents.tools import _make_retrieval_tool
        from langchain_core.documents import Document

        mock_store = MagicMock(spec=BaseVectorStore)
        mock_retriever = MagicMock()
        mock_store.as_retriever.return_value = mock_retriever
        doc = Document(
            page_content="LangChain is a framework for building LLM applications.",
            metadata={"source_url": "https://example.com/docs"},
        )
        mock_retriever.invoke.return_value = [doc]

        retriever = Retriever(mock_store)
        tool = _make_retrieval_tool(retriever)
        result = tool.invoke({"query": "what is langchain"})

        if isinstance(result, tuple):
            serialized = result[0]
        else:
            serialized = str(result)
        assert "LangChain is a framework" in serialized
        assert "https://example.com/docs" in serialized

    @patch("doc_helper.retrieval.reranker.Reranker")
    def test_retrieve_context_empty_returns_hint(self, _mock_reranker):
        from doc_helper.agents.tools import _make_retrieval_tool

        mock_store = MagicMock(spec=BaseVectorStore)
        mock_retriever = MagicMock()
        mock_store.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = []

        retriever = Retriever(mock_store)
        tool = _make_retrieval_tool(retriever)
        result = tool.invoke({"query": "nonexistent"})

        if isinstance(result, tuple):
            serialized, artifact = result
        else:
            serialized = str(result)
        assert "No relevant documentation" in serialized
        assert "web_search" in serialized


class TestWebSearchTool:
    def test_requires_api_key(self):
        from doc_helper.agents.tools import _make_web_search_tool

        with pytest.raises(ValueError, match="Tavily API key"):
            _make_web_search_tool(None)

    @patch("doc_helper.agents.tools.TavilySearch")
    def test_web_search_returns_snippets(self, mock_tavily_cls):
        from doc_helper.agents.tools import _make_web_search_tool

        mock_client = MagicMock()
        mock_tavily_cls.return_value = mock_client
        mock_client.invoke.return_value = {
            "results": [
                {"title": "Result 1", "url": "https://example.com/1", "content": "Content 1"},
                {"title": "Result 2", "url": "https://example.com/2", "content": "Content 2"},
            ]
        }

        tool = _make_web_search_tool("tvly-test")
        serialized = tool.invoke({"query": "latest langchain updates"})

        assert isinstance(serialized, str)
        assert "Result 1" in serialized
        assert "https://example.com/1" in serialized

    @patch("doc_helper.agents.tools.TavilySearch")
    def test_web_search_handles_empty_results(self, mock_tavily_cls):
        from doc_helper.agents.tools import _make_web_search_tool

        mock_client = MagicMock()
        mock_tavily_cls.return_value = mock_client
        mock_client.invoke.return_value = {"results": []}

        tool = _make_web_search_tool("tvly-test")
        serialized = tool.invoke({"query": "nonexistent topic"})

        assert isinstance(serialized, str)
        assert "No web results" in serialized


class TestCheckLinksTool:
    def test_handles_empty_input(self):
        from doc_helper.agents.tools import _make_check_links_tool

        tool = _make_check_links_tool()
        result = asyncio.run(tool.ainvoke({"urls": ""}))
        assert "No URLs" in result

    def test_parses_urls_and_reports_status(self):
        from doc_helper.agents.tools import _make_check_links_tool

        tool = _make_check_links_tool()

        async def _run():
            mock_resp_ok = MagicMock()
            mock_resp_ok.status_code = 200
            mock_resp_fail = MagicMock()
            mock_resp_fail.status_code = 404

            mock_client = MagicMock()
            mock_client.head = AsyncMock(
                side_effect=lambda url: mock_resp_ok if "good" in url else mock_resp_fail
            )

            mock_cm = MagicMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)

            with patch("doc_helper.agents.tools.httpx.AsyncClient", return_value=mock_cm):
                result = await tool.ainvoke({"urls": "https://good.com\nhttps://bad.com"})
                assert "200" in result
                assert "404" in result

        asyncio.run(_run())

    def test_handles_comma_separated_urls(self):
        from doc_helper.agents.tools import _make_check_links_tool

        tool = _make_check_links_tool()

        async def _run():
            mock_resp = MagicMock()
            mock_resp.status_code = 200

            mock_client = MagicMock()
            mock_client.head = AsyncMock(return_value=mock_resp)

            mock_cm = MagicMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cm.__aexit__ = AsyncMock(return_value=None)

            with patch("doc_helper.agents.tools.httpx.AsyncClient", return_value=mock_cm):
                result = await tool.ainvoke({"urls": "https://a.com,https://b.com"})
                assert "200" in result

        asyncio.run(_run())


class TestCreateTools:
    def test_creates_retrieve_and_check_links_without_api_key(self):
        mock_store = MagicMock(spec=BaseVectorStore)
        retriever = Retriever(mock_store)

        from doc_helper.agents.tools import create_tools

        settings = IngestionSettings(tavily_api_key=None)
        tools = create_tools(retriever, ingestion_settings=settings, enable_web_search=False)

        tool_names = [t.name for t in tools]
        assert "retrieve_context" in tool_names
        assert "check_links" in tool_names
        assert "web_search" not in tool_names

    @patch("doc_helper.agents.tools.TavilySearch")
    def test_creates_all_tools_with_api_key(self, _mock):
        mock_store = MagicMock(spec=BaseVectorStore)
        retriever = Retriever(mock_store)

        from doc_helper.agents.tools import create_tools

        settings = IngestionSettings(tavily_api_key="tvly-test")
        tools = create_tools(retriever, ingestion_settings=settings, enable_web_search=True)

        tool_names = [t.name for t in tools]
        assert "retrieve_context" in tool_names
        assert "web_search" in tool_names
        assert "check_links" in tool_names