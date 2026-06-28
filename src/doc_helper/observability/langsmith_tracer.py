import logging
from typing import Any

from doc_helper.observability.base import BaseTracer

logger = logging.getLogger(__name__)


class LangSmithTracer(BaseTracer):
    def __init__(self, api_key: str, project: str = "documentation-helper"):
        import os

        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = project

        logger.info(f"LangSmith tracer initialized (project={project})")

    def trace_retrieval(self, query: str, documents: list[Any]) -> None:
        logger.debug(f"[LangSmith] retrieval: {query} -> {len(documents)} docs")

    def trace_tool_call(self, tool_name: str, tool_input: dict, tool_output: Any) -> None:
        logger.debug(f"[LangSmith] tool:{tool_name}")

    def trace_llm_call(self, prompt: str, response: str, metadata: dict | None = None) -> None:
        logger.debug(f"[LangSmith] llm_call: {len(prompt)} chars -> {len(response)} chars")

    def trace_agent_run(self, query: str, answer: str, sources: list[str]) -> None:
        logger.info(f"[LangSmith] agent_run: '{query[:50]}...' -> {len(sources)} sources")

    def flush(self) -> None:
        pass
