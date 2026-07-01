from collections.abc import AsyncGenerator
from typing import Any

import asyncio

from langchain.agents import create_agent
from langchain.messages import ToolMessage
from langchain_core.language_models import BaseChatModel

from doc_helper.agents.events import (
    AnswerEvent,
    DoneEvent,
    ErrorEvent,
    SSEEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from doc_helper.agents.middleware import (
    guardrails_middleware,
    summarization_middleware,
)
from doc_helper.agents.tools import create_tools
from doc_helper.config.settings import AgentSettings, Settings
from doc_helper.db.conversations import ConversationManager
from doc_helper.observability.base import BaseTracer, NoOpTracer
from doc_helper.retrieval.retriever import Retriever

_SYSTEM_PROMPT = (
    "You are a helpful AI assistant that answers questions about LangChain documentation.\n"
    "You have access to the following tools:\n"
    "1. retrieve_context — Search the local ingested documentation (use this FIRST).\n"
    "2. web_search — Search the web for information not found locally.\n"
    "3. check_links — Verify whether specific URLs are reachable.\n\n"
    "Guidelines:\n"
    "- Always use retrieve_context first.\n"
    "- If retrieve_context returns no results, you MUST call web_search before answering.\n"
    "- Never answer from memory alone. Always retrieve context or search first.\n"
    "- Always cite the source_url of retrieved docs in your answers.\n"
    "- Use check_links only when the user asks about URL validity."
)


class RAGAgent:
    def __init__(
        self,
        llm: BaseChatModel,
        retriever: Retriever,
        conversation_manager: ConversationManager | None = None,
        agent_settings: AgentSettings | None = None,
        ingestion_settings: Any | None = None,
        enable_web_search: bool = True,
        tracer: BaseTracer | None = None,
    ):
        self._llm = llm
        self._retriever = retriever
        self._conversation_manager = conversation_manager
        self._agent_settings = agent_settings or AgentSettings()
        self._tracer = tracer or NoOpTracer()

        self._tools = create_tools(
            retriever=retriever,
            ingestion_settings=ingestion_settings,
            enable_web_search=enable_web_search,
        )
        self._agent = create_agent(
            self._llm, tools=self._tools, system_prompt=_SYSTEM_PROMPT
        )

        self._guardrails = guardrails_middleware(self._agent_settings)
        self._summarize = summarization_middleware(
            conversation_manager, llm, self._agent_settings
        )

    def _maybe_summarize(self, conversation_id: str | None) -> None:
        if not self._summarize:
            return
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(self._summarize(conversation_id))
            else:
                asyncio.run(self._summarize(conversation_id))
        except RuntimeError:
            asyncio.run(self._summarize(conversation_id))

    def run(self, query: str, conversation_id: str | None = None) -> dict[str, Any]:
        guardrail_error = self._guardrails(query)
        if guardrail_error:
            return {"answer": guardrail_error, "context": [], "sources": []}

        self._maybe_summarize(conversation_id)

        messages = self._build_messages(query, conversation_id)
        response = self._agent.invoke({"messages": messages})
        answer, context_docs, sources = self._extract_response(response)

        self._tracer.trace_agent_run(query, answer, sources)
        if context_docs:
            self._tracer.trace_retrieval(query, context_docs)

        self._persist_messages(query, answer, sources, conversation_id)

        return {"answer": answer, "context": context_docs, "sources": sources}

    async def astream(
        self, query: str, conversation_id: str | None = None
    ) -> AsyncGenerator[SSEEvent, None]:
        guardrail_error = self._guardrails(query)
        if guardrail_error:
            yield ErrorEvent(message=guardrail_error)
            return

        if self._summarize:
            await self._summarize(conversation_id)

        try:
            messages = self._build_messages(query, conversation_id)
            collected_answer: list[str] = []
            collected_sources: list[str] = []

            async for event in self._agent.astream_events(
                {"messages": messages}, version="v2"
            ):
                for sse_event in self._process_stream_event(
                    event, collected_answer, collected_sources
                ):
                    yield sse_event

            answer = "".join(collected_answer).strip()
            self._tracer.trace_agent_run(query, answer, collected_sources)

            self._persist_messages(query, answer, collected_sources, conversation_id)
            yield DoneEvent(
                conversation_id=conversation_id or "", sources=collected_sources
            )

        except Exception as e:
            yield ErrorEvent(message=str(e))

    def _process_stream_event(
        self, event: dict, answer_acc: list[str] | None = None, sources_acc: list[str] | None = None
    ) -> Any:
        kind = event["event"]
        data = event.get("data", {})

        if kind == "on_chat_model_stream":
            chunk = data.get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                if answer_acc is not None:
                    answer_acc.append(chunk.content)
                yield AnswerEvent(content=chunk.content)

        elif kind == "on_tool_start":
            tool_input = data.get("input", {})
            tool_name = event.get("name", "unknown")
            self._tracer.trace_tool_call(tool_name, tool_input, None)
            yield ToolCallEvent(
                tool=tool_name,
                query=str(tool_input.get("query", tool_input.get("urls", ""))),
            )

        elif kind == "on_tool_end":
            artifact = data.get("output")
            sources: list[str] = []
            if isinstance(artifact, list):
                for item in artifact:
                    if isinstance(item, str):
                        sources.append(item)
                    elif hasattr(item, "metadata"):
                        sources.append(
                            item.metadata.get("source_url", item.metadata.get("source", "Unknown"))
                        )
            if sources_acc is not None:
                sources_acc.extend(sources)
            yield ToolResultEvent(sources=sources, num_docs=len(sources))

    def _build_messages(self, query: str, conversation_id: str | None = None) -> list[dict]:
        messages = []
        if self._conversation_manager and conversation_id:
            history = self._conversation_manager.get_messages(conversation_id)
            for msg in history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": query})
        return messages

    def _persist_messages(
        self, query: str, answer: str, sources: list[str], conversation_id: str | None
    ) -> None:
        if self._conversation_manager and conversation_id:
            self._conversation_manager.add_message(conversation_id, "user", query)
            self._conversation_manager.add_message(conversation_id, "assistant", answer, sources)

    def _extract_response(self, response: dict) -> tuple[str, list, list[str]]:
        answer = response["messages"][-1].content
        context_docs = []
        sources = []
        for message in response["messages"]:
            if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
                if isinstance(message.artifact, list):
                    for item in message.artifact:
                        if isinstance(item, str):
                            sources.append(item)
                        elif hasattr(item, "metadata"):
                            context_docs.append(item)
                            sources.append(
                                item.metadata.get(
                                    "source_url",
                                    item.metadata.get("source", "Unknown"),
                                )
                            )
        return str(answer), context_docs, sources


def create_rag_agent(
    settings: Settings | None = None,
    conversation_manager: ConversationManager | None = None,
) -> RAGAgent:
    from doc_helper.config.settings import LLMSettings as _LLMSettings
    from doc_helper.config.settings import Settings as _Settings
    from doc_helper.llm.factory import create_chat_model
    from doc_helper.observability.factory import create_tracer as _create_tracer
    from doc_helper.stores.factory import create_vector_store

    if settings is None:
        settings = _Settings()

    primary = create_chat_model(settings.llm)
    store = create_vector_store(settings)
    tracer = _create_tracer(settings.observability)

    from doc_helper.stores.base import BaseVectorStore

    if isinstance(store, BaseVectorStore):
        store.validate_embedding_model(settings.embedding.model)

    agent_cfg = settings.agent
    if agent_cfg.model_fallback_enabled and agent_cfg.fallback_model:
        fallback = create_chat_model(
            _LLMSettings(
                provider="ollama",
                model=agent_cfg.fallback_model,
                temperature=settings.llm.temperature,
            )
        )
        llm = primary.with_fallbacks([fallback])
    else:
        llm = primary

    retriever = Retriever(store, settings.retrieval)

    enable_web_search = bool(settings.ingestion.tavily_api_key)

    return RAGAgent(
        llm=llm,
        retriever=retriever,
        conversation_manager=conversation_manager,
        agent_settings=settings.agent,
        ingestion_settings=settings.ingestion,
        enable_web_search=enable_web_search,
        tracer=tracer,
    )
