import asyncio

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from doc_helper.logger import log_error, log_success, log_warning
from doc_helper.stores.base import BaseVectorStore


class ChromaVectorStore(BaseVectorStore):
    EMBEDDING_MODEL_METADATA_KEY = "embedding_model"

    def __init__(
        self,
        persist_directory: str,
        embedding_function,
        embedding_model_name: str,
    ):
        self._persist_directory = persist_directory
        self._embedding_function = embedding_function
        self._embedding_model_name = embedding_model_name
        self._store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
        )

    def add_documents(self, documents: list[Document], batch_size: int = 500) -> None:
        batches = [documents[i : i + batch_size] for i in range(0, len(documents), batch_size)]
        successful = 0
        for i, batch in enumerate(batches, start=1):
            try:
                self._store.add_documents(batch)
                successful += 1
                log_success(f"Chroma: Added batch {i}/{len(batches)} ({len(batch)} docs)")
            except Exception as e:
                log_error(f"Chroma: Failed to add batch {i}/{len(batches)} - {e}")

        if successful == len(batches):
            log_success(f"Chroma: All {len(batches)} batches processed")
        else:
            log_warning(f"Chroma: Processed {successful}/{len(batches)} batches")

        self._store._collection.modify(
            metadata={self.EMBEDDING_MODEL_METADATA_KEY: self._embedding_model_name}
        )

    async def aadd_documents(self, documents: list[Document], batch_size: int = 500) -> None:
        batches = [documents[i : i + batch_size] for i in range(0, len(documents), batch_size)]

        async def add_batch(batch: list[Document], batch_num: int):
            try:
                await self._store.aadd_documents(batch)
                log_success(f"Chroma: Added batch {batch_num}/{len(batches)} ({len(batch)} docs)")
                return True
            except Exception as e:
                log_error(f"Chroma: Failed to add batch {batch_num} - {e}")
                return False

        tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful = sum(1 for r in results if r is True)

        if successful == len(batches):
            log_success(f"Chroma: All {len(batches)} batches processed")
        else:
            log_warning(f"Chroma: Processed {successful}/{len(batches)} batches")

        self._store._collection.modify(
            metadata={self.EMBEDDING_MODEL_METADATA_KEY: self._embedding_model_name}
        )

    def as_retriever(
        self, search_type: str = "similarity", search_kwargs: dict | None = None
    ) -> VectorStoreRetriever:
        return self._store.as_retriever(search_type=search_type, search_kwargs=search_kwargs or {})

    def get_embedding_model_name(self) -> str | None:
        metadata = self._store._collection.metadata
        if metadata is None:
            return None
        return metadata.get(self.EMBEDDING_MODEL_METADATA_KEY)

    def get_existing_hashes(self) -> set[str]:
        results = self._store._collection.get(include=["metadatas"])
        metadatas = results.get("metadatas") or []
        return {
            m["content_hash"]
            for m in metadatas
            if m and "content_hash" in m
        }
