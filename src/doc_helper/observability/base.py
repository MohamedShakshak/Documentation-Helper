from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any


class BaseTracer(ABC):
    @abstractmethod
    def trace_retrieval(self, query: str, documents: list[Any]) -> None:
        ...

    @abstractmethod
    def trace_tool_call(self, tool_name: str, tool_input: dict, tool_output: Any) -> None:
        ...

    @abstractmethod
    def trace_llm_call(self, prompt: str, response: str, metadata: dict | None = None) -> None:
        ...

    @abstractmethod
    def trace_agent_run(self, query: str, answer: str, sources: list[str]) -> None:
        ...

    @abstractmethod
    def flush(self) -> None:
        ...


class NoOpTracer(BaseTracer):
    def trace_retrieval(self, query: str, documents: list[Any]) -> None:
        pass

    def trace_tool_call(self, tool_name: str, tool_input: dict, tool_output: Any) -> None:
        pass

    def trace_llm_call(self, prompt: str, response: str, metadata: dict | None = None) -> None:
        pass

    def trace_agent_run(self, query: str, answer: str, sources: list[str]) -> None:
        pass

    def flush(self) -> None:
        pass