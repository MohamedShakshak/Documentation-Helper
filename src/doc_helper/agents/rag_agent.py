from typing import Any

from langchain.agents import create_agent
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_core.language_models import BaseChatModel

from doc_helper.config.settings import Settings
from doc_helper.retrieval.retriever import Retriever


def _make_retrieval_tool(retriever: Retriever):
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve relevant documentation to help answer user queries about LangChain."""
        retrieved_docs = retriever.retrieve(query)
        serialized = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    return retrieve_context


_SYSTEM_PROMPT = (
    "You are a helpful AI assistant that answers questions about LangChain documentation. "
    "You have access to a tool that retrieves relevant documentation. "
    "Use the tool to find relevant information before answering questions. "
    "Always cite the sources you use in your answers. "
    "If you cannot find the answer in the retrieved documentation, say so."
)


class RAGAgent:
    def __init__(self, llm: BaseChatModel, retriever: Retriever):
        self._llm = llm
        self._retriever = retriever
        self._tool = _make_retrieval_tool(retriever)
        self._agent = create_agent(
            self._llm, tools=[self._tool], system_prompt=_SYSTEM_PROMPT
        )

    def run(self, query: str) -> dict[str, Any]:
        response = self._agent.invoke({"messages": [{"role": "user", "content": query}]})
        answer = response["messages"][-1].content
        context_docs = []
        for message in response["messages"]:
            if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
                if isinstance(message.artifact, list):
                    context_docs.extend(message.artifact)
        return {"answer": answer, "context": context_docs}

    async def arun(self, query: str) -> dict[str, Any]:
        response = await self._agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]}
        )
        answer = response["messages"][-1].content
        context_docs = []
        for message in response["messages"]:
            if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
                if isinstance(message.artifact, list):
                    context_docs.extend(message.artifact)
        return {"answer": answer, "context": context_docs}


def create_rag_agent(settings: Settings | None = None) -> RAGAgent:
    from doc_helper.config.settings import Settings as _Settings
    from doc_helper.embeddings.factory import create_embeddings, get_embedding_model_name
    from doc_helper.llm.factory import create_chat_model
    from doc_helper.stores.factory import create_vector_store

    if settings is None:
        settings = _Settings()

    llm = create_chat_model(settings.llm)
    store = create_vector_store(settings)
    embeddings = create_embeddings(settings.embedding)
    model_key = settings.embedding.model

    from doc_helper.stores.base import BaseVectorStore

    if isinstance(store, BaseVectorStore):
        store.validate_embedding_model(model_key)

    retriever = Retriever(store, settings.retrieval)
    return RAGAgent(llm=llm, retriever=retriever)