from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    AGENT_THOUGHT = "agent_thought"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ANSWER = "answer"
    DONE = "done"
    ERROR = "error"


@dataclass
class SSEEvent:
    event: EventType
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"event": self.event.value, "data": self.data}

    def to_sse(self) -> str:
        import json

        return f"event: {self.event.value}\ndata: {json.dumps(self.data)}\n\n"


@dataclass
class AgentThoughtEvent(SSEEvent):
    def __init__(self, content: str):
        super().__init__(event=EventType.AGENT_THOUGHT, data={"content": content})


@dataclass
class ToolCallEvent(SSEEvent):
    def __init__(self, tool: str, query: str):
        super().__init__(event=EventType.TOOL_CALL, data={"tool": tool, "query": query})


@dataclass
class ToolResultEvent(SSEEvent):
    def __init__(self, sources: list[str], num_docs: int):
        super().__init__(
            event=EventType.TOOL_RESULT,
            data={"sources": sources, "num_docs": num_docs},
        )


@dataclass
class AnswerEvent(SSEEvent):
    def __init__(self, content: str):
        super().__init__(event=EventType.ANSWER, data={"content": content})


@dataclass
class DoneEvent(SSEEvent):
    def __init__(self, conversation_id: str, sources: list[str]):
        super().__init__(
            event=EventType.DONE,
            data={"conversation_id": conversation_id, "sources": sources},
        )


@dataclass
class ErrorEvent(SSEEvent):
    def __init__(self, message: str):
        super().__init__(event=EventType.ERROR, data={"message": message})