import logging
from typing import Any

from doc_helper.observability.base import BaseTracer

logger = logging.getLogger(__name__)


class LangFuseTracer(BaseTracer):
    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str = "http://localhost:3000",
    ):
        from langfuse import Langfuse

        self._client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        logger.info(f"LangFuse tracer initialized (host={host})")

    def trace_retrieval(self, query: str, documents: list[Any]) -> None:
        try:
            self._client.span(
                name="retrieval",
                input={"query": query},
                output={
                    "num_docs": len(documents),
                    "sources": [
                        d.metadata.get("source_url", d.metadata.get("source", "Unknown"))
                        if hasattr(d, "metadata")
                        else "Unknown"
                        for d in documents
                    ],
                },
            )
        except Exception as e:
            logger.warning(f"LangFuse trace_retrieval failed: {e}")

    def trace_tool_call(self, tool_name: str, tool_input: dict, tool_output: Any) -> None:
        try:
            self._client.span(
                name=f"tool:{tool_name}",
                input=tool_input,
                output=str(tool_output)[:2000],
            )
        except Exception as e:
            logger.warning(f"LangFuse trace_tool_call failed: {e}")

    def trace_llm_call(self, prompt: str, response: str, metadata: dict | None = None) -> None:
        try:
            self._client.generation(
                name="llm_call",
                input=prompt[:4000],
                output=response[:4000],
                metadata=metadata or {},
            )
        except Exception as e:
            logger.warning(f"LangFuse trace_llm_call failed: {e}")

    def trace_agent_run(self, query: str, answer: str, sources: list[str]) -> None:
        try:
            self._client.trace(
                name="agent_run",
                input={"query": query},
                output={"answer": answer, "sources": sources},
            )
        except Exception as e:
            logger.warning(f"LangFuse trace_agent_run failed: {e}")

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception as e:
            logger.warning(f"LangFuse flush failed: {e}")
