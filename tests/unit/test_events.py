from unittest.mock import MagicMock, patch

import pytest

from doc_helper.agents.events import (
    AnswerEvent,
    DoneEvent,
    ErrorEvent,
    EventType,
    ToolCallEvent,
    ToolResultEvent,
)


class TestEventType:
    def test_event_types(self):
        assert EventType.AGENT_THOUGHT.value == "agent_thought"
        assert EventType.TOOL_CALL.value == "tool_call"
        assert EventType.TOOL_RESULT.value == "tool_result"
        assert EventType.ANSWER.value == "answer"
        assert EventType.DONE.value == "done"
        assert EventType.ERROR.value == "error"


class TestAnswerEvent:
    def test_creation(self):
        event = AnswerEvent(content="Hello")
        assert event.event == EventType.ANSWER
        assert event.data["content"] == "Hello"

    def test_to_dict(self):
        event = AnswerEvent(content="Hi")
        d = event.to_dict()
        assert d["event"] == "answer"
        assert d["data"]["content"] == "Hi"

    def test_to_sse(self):
        event = AnswerEvent(content="Test")
        sse = event.to_sse()
        assert sse.startswith("event: answer\n")
        assert '"content": "Test"' in sse


class TestToolCallEvent:
    def test_creation(self):
        event = ToolCallEvent(tool="retrieve_context", query="What is LangChain?")
        assert event.event == EventType.TOOL_CALL
        assert event.data["tool"] == "retrieve_context"
        assert event.data["query"] == "What is LangChain?"


class TestToolResultEvent:
    def test_creation(self):
        event = ToolResultEvent(sources=["url1", "url2"], num_docs=2)
        assert event.event == EventType.TOOL_RESULT
        assert event.data["sources"] == ["url1", "url2"]
        assert event.data["num_docs"] == 2


class TestDoneEvent:
    def test_creation(self):
        event = DoneEvent(conversation_id="abc-123", sources=["url1"])
        assert event.event == EventType.DONE
        assert event.data["conversation_id"] == "abc-123"


class TestErrorEvent:
    def test_creation(self):
        event = ErrorEvent(message="Something went wrong")
        assert event.event == EventType.ERROR
        assert event.data["message"] == "Something went wrong"