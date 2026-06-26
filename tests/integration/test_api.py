from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from doc_helper.api.app import create_app
from doc_helper.api.deps import reset_caches


@pytest.fixture
def client():
    reset_caches()
    app = create_app()
    with TestClient(app) as c:
        yield c
    reset_caches()


class TestHealthEndpoint:
    def test_health_check(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "llm_provider" in data
        assert "embedding_model" in data
        assert "vector_store" in data


class TestChatEndpoints:
    def test_create_conversation(self, client):
        resp = client.post("/api/chat/conversations", json={"title": "Test"})
        assert resp.status_code == 200
        data = resp.json()
        assert "conversation_id" in data
        assert data["title"] == "Test"

    def test_list_conversations(self, client):
        client.post("/api/chat/conversations", json={"title": "Conv 1"})
        client.post("/api/chat/conversations", json={"title": "Conv 2"})

        resp = client.get("/api/chat/conversations")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["conversations"]) >= 2

    def test_get_conversation_messages(self, client):
        create_resp = client.post("/api/chat/conversations", json={"title": "Test"})
        conv_id = create_resp.json()["conversation_id"]

        resp = client.get(f"/api/chat/conversations/{conv_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["conversation_id"] == conv_id
        assert "messages" in data

    def test_get_nonexistent_conversation(self, client):
        resp = client.get("/api/chat/conversations/nonexistent-id")
        assert resp.status_code == 404

    def test_delete_conversation(self, client):
        create_resp = client.post("/api/chat/conversations", json={"title": "ToDelete"})
        conv_id = create_resp.json()["conversation_id"]

        resp = client.delete(f"/api/chat/conversations/{conv_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        get_resp = client.get(f"/api/chat/conversations/{conv_id}")
        assert get_resp.status_code == 404

    @patch("doc_helper.api.deps.get_agent")
    def test_chat_endpoint(self, mock_get_agent, client):
        mock_agent = MagicMock()
        mock_agent.run.return_value = {
            "answer": "LangChain is a framework for LLM apps.",
            "context": [],
            "sources": ["https://example.com/docs"],
        }

        with patch("doc_helper.api.routes.chat.get_agent", return_value=mock_agent):
            resp = client.post("/api/chat", json={"query": "What is LangChain?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "LangChain is a framework" in data["answer"]
        assert "https://example.com/docs" in data["sources"]


class TestIngestEndpoints:
    def test_start_ingestion(self, client):
        with patch("doc_helper.api.routes.ingest._run_ingestion_task") as mock_task:
            import asyncio

            async def _noop(task_id, config):
                pass

            mock_task.side_effect = _noop

            resp = client.post(
                "/api/ingest",
                json={"crawler": "recursive", "url": "https://example.com", "depth": 1},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "task_id" in data
            assert data["status"] == "pending"

    def test_get_task_status(self, client):
        with patch("doc_helper.api.routes.ingest._run_ingestion_task") as mock_task:
            import asyncio

            async def _noop(task_id, config):
                pass

            mock_task.side_effect = _noop

            create_resp = client.post("/api/ingest", json={"crawler": "recursive"})
            task_id = create_resp.json()["task_id"]

            resp = client.get(f"/api/ingest/{task_id}/status")
            assert resp.status_code == 200
            data = resp.json()
            assert data["task_id"] == task_id
            assert data["status"] in ("pending", "running", "completed", "failed")

    def test_get_nonexistent_task(self, client):
        resp = client.get("/api/ingest/nonexistent-id/status")
        assert resp.status_code == 404

    def test_list_tasks(self, client):
        with patch("doc_helper.api.routes.ingest._run_ingestion_task") as mock_task:
            import asyncio

            async def _noop(task_id, config):
                pass

            mock_task.side_effect = _noop

            client.post("/api/ingest", json={"crawler": "recursive"})
            client.post("/api/ingest", json={"crawler": "tavily"})

            resp = client.get("/api/ingest")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) >= 2


class TestSSEStreaming:
    @patch("doc_helper.api.routes.chat.get_agent")
    def test_chat_stream(self, mock_get_agent, client):
        from doc_helper.agents.events import AnswerEvent, DoneEvent

        async def mock_astream(query, conversation_id=None):
            yield AnswerEvent(content="Hello ")
            yield AnswerEvent(content="world!")
            yield DoneEvent(conversation_id=conversation_id or "", sources=["url"])

        mock_agent = MagicMock()
        mock_agent.astream = mock_astream
        mock_get_agent.return_value = mock_agent

        resp = client.post("/api/chat/stream", json={"query": "hi"})
        assert resp.status_code == 200
        body = resp.text
        assert "event: answer" in body
        assert "event: done" in body
        assert "Hello " in body
        assert "world!" in body