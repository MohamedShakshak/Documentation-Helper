from langchain_core.documents import Document

from doc_helper.config.settings import RetrievalSettings
from doc_helper.stores.base import BaseVectorStore


class Retriever:
    def __init__(self, store: BaseVectorStore, settings: RetrievalSettings | None = None):
        self._store = store
        self._settings = settings or RetrievalSettings()

    def retrieve(self, query: str) -> list[Document]:
        search_kwargs: dict = {"k": self._settings.search_k}
        if self._settings.search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = self._settings.score_threshold

        retriever = self._store.as_retriever(
            search_type=self._settings.search_type,
            search_kwargs=search_kwargs,
        )
        documents = retriever.invoke(query)

        if self._settings.reranker_enabled:
            from doc_helper.retrieval.reranker import Reranker

            reranker = Reranker()
            documents = reranker.rerank(query, documents)

        return documents
