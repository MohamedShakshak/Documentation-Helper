from unittest.mock import MagicMock, patch

import pytest

from doc_helper.agents.rag_agent import RAGAgent, _make_retrieval_tool
from doc_helper.config.settings import RetrievalSettings
from doc_helper.retrieval.retriever import Retriever
from doc_helper.stores.base import BaseVectorStore


class TestRAGAgent:
    def test_make_retrieval_tool(self):
        mock_store = MagicMock(spec=BaseVectorStore)
        mock_retriever_obj = MagicMock()
        doc = MagicMock()
        doc.page_content = "LangChain is a framework"
        doc.metadata = {"source": "https://python.langchain.com/docs"}
        mock_retriever_obj.invoke.return_value = [doc]
        mock_store.as_retriever.return_value = mock_retriever_obj

        retriever = Retriever(mock_store)
        tool = _make_retrieval_tool(retriever)
        assert tool.name == "retrieve_context"

    @patch("doc_helper.agents.rag_agent.create_agent")
    def test_run_basic(self, mock_create_agent):
        mock_llm = MagicMock()
        mock_store = MagicMock(spec=BaseVectorStore)
        mock_retriever = MagicMock(spec=Retriever)

        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.invoke.return_value = {
            "messages": [MagicMock(content="LangChain is a framework")]
        }

        agent = RAGAgent(llm=mock_llm, retriever=mock_retriever)
        result = agent.run("What is LangChain?")
        assert "answer" in result
        assert "context" in result
        assert "sources" in result

    def test_build_messages_without_conversation(self):
        mock_llm = MagicMock()
        mock_store = MagicMock(spec=BaseVectorStore)
        mock_retriever = MagicMock(spec=Retriever)

        with patch("doc_helper.agents.rag_agent.create_agent"):
            agent = RAGAgent(llm=mock_llm, retriever=mock_retriever)

        messages = agent._build_messages("Hello", conversation_id=None)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    def test_build_messages_with_conversation(self):
        mock_llm = MagicMock()
        mock_store = MagicMock(spec=BaseVectorStore)
        mock_retriever = MagicMock(spec=Retriever)
        mock_conv_mgr = MagicMock()
        mock_conv_mgr.get_messages.return_value = [
            {"role": "user", "content": "Hi", "sources": []},
            {"role": "assistant", "content": "Hello!", "sources": []},
        ]

        with patch("doc_helper.agents.rag_agent.create_agent"):
            agent = RAGAgent(
                llm=mock_llm,
                retriever=mock_retriever,
                conversation_manager=mock_conv_mgr,
            )

        messages = agent._build_messages("Follow-up", conversation_id="conv-1")
        assert len(messages) == 3  # 2 history + 1 new
        assert messages[0]["content"] == "Hi"
        assert messages[2]["content"] == "Follow-up"
        mock_conv_mgr.add_message.assert_called_once_with("conv-1", "user", "Follow-up")