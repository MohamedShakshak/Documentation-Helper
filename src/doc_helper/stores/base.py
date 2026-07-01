from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever


class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: list[Document], batch_size: int = 500) -> None:
        ...

    @abstractmethod
    def as_retriever(
        self, search_type: str = "similarity", search_kwargs: dict | None = None
    ) -> VectorStoreRetriever:
        ...

    @abstractmethod
    def get_embedding_model_name(self) -> str | None:
        ...

    @abstractmethod
    def get_existing_hashes(self) -> set[str]:
        """Return all content_hash values currently stored in the collection."""
        ...

    def validate_embedding_model(self, configured_model: str) -> None:
        stored_model = self.get_embedding_model_name()
        if stored_model and stored_model != configured_model:
            raise ValueError(
                f"Vector store was created with '{stored_model}' but "
                f"'{configured_model}' is configured. Re-ingest or switch models."
            )
