from langchain_core.documents import Document


class Reranker:
    DEFAULT_MODEL = "ms-marco-MiniLM-L-12-v2"

    def __init__(self, model: str | None = None, top_k: int | None = None):
        self._model = model or self.DEFAULT_MODEL
        self._top_k = top_k

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        try:
            from flashrank import Ranker, RerankRequest
        except ImportError:
            raise ImportError(
                "flashrank is required for reranking. Install it with: pip install flashrank"
            )

        ranker = Ranker(model_name=self._model, cache_dir=".flashrank_cache")
        passages = [
            {"id": i, "text": doc.page_content, "meta": doc.metadata}
            for i, doc in enumerate(documents)
        ]
        rerank_request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(rerank_request)

        top_k = self._top_k or len(documents)
        reranked_docs = []
        for result in results[:top_k]:
            idx = result["id"]
            reranked_docs.append(documents[idx])
        return reranked_docs
