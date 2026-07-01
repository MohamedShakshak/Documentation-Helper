import logging
import os
from typing import Any

from doc_helper.observability.base import BaseTracer

logger = logging.getLogger(__name__)


class LangSmithTracer(BaseTracer):
    def __init__(self, api_key: str, project: str = "documentation-helper"):
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_PROJECT"] = project

        if os.environ.get("LANGSMITH_ENDPOINT"):
            os.environ.setdefault("LANGCHAIN_ENDPOINT", os.environ["LANGSMITH_ENDPOINT"])

        try:
            from langsmith import Client

            client = Client()
            projects = [p.name for p in client.list_projects()]
            if project not in projects:
                client.create_project(project)
            logger.info(f"LangSmith connected (project={project})")
        except Exception as e:
            logger.warning(f"LangSmith connection failed: {e}. Traces will not be sent.")

    def trace_retrieval(self, query: str, documents: list[Any]) -> None:
        pass

    def trace_tool_call(self, tool_name: str, tool_input: dict, tool_output: Any) -> None:
        pass

    def trace_llm_call(self, prompt: str, response: str, metadata: dict | None = None) -> None:
        pass

    def trace_agent_run(self, query: str, answer: str, sources: list[str]) -> None:
        pass

    def flush(self) -> None:
        try:
            from langsmith import Client

            Client().flush()
        except Exception:
            pass