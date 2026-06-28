from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_pinecone import PineconeVectorStore as LangChainPinecone

from doc_helper.logger import log_error, log_success, log_warning
from doc_helper.stores.base import BaseVectorStore


class PineconeVectorStore(BaseVectorStore):
    EMBEDDING_MODEL_METADATA_KEY = "embedding_model"

    def __init__(
        self,
        index_name: str,
        embedding_function,
        embedding_model_name: str,
    ):
        self._index_name = index_name
        self._embedding_function = embedding_function
        self._embedding_model_name = embedding_model_name
        self._store = LangChainPinecone(
            index_name=index_name,
            embedding=embedding_function,
        )

    def add_documents(self, documents: list[Document], batch_size: int = 500) -> None:
        batches = [documents[i : i + batch_size] for i in range(0, len(documents), batch_size)]
        successful = 0
        for i, batch in enumerate(batches, start=1):
            try:
                self._store.add_documents(batch)
                successful += 1
                log_success(f"Pinecone: Added batch {i}/{len(batches)} ({len(batch)} docs)")
            except Exception as e:
                log_error(f"Pinecone: Failed to add batch {i}/{len(batches)} - {e}")

        if successful == len(batches):
            log_success(f"Pinecone: All {len(batches)} batches processed")
        else:
            log_warning(f"Pinecone: Processed {successful}/{len(batches)} batches")

    def as_retriever(
        self, search_type: str = "similarity", search_kwargs: dict | None = None
    ) -> VectorStoreRetriever:
        return self._store.as_retriever(search_type=search_type, search_kwargs=search_kwargs or {})

    def get_embedding_model_name(self) -> str | None:
        try:
            from pinecone import Pinecone

            pc = Pinecone()
            index = pc.Index(self._index_name)
            metadata = index.describe_index_stats().get("metadata_config", {})
            return metadata.get(self.EMBEDDING_MODEL_METADATA_KEY)
        except Exception:
            return None
