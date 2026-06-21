from doc_helper.config.settings import Settings, VectorStoreSettings
from doc_helper.embeddings.factory import create_embeddings, get_embedding_model_name
from doc_helper.stores.base import BaseVectorStore
from doc_helper.stores.chroma_store import ChromaVectorStore
from doc_helper.stores.pinecone_store import PineconeVectorStore


def create_vector_store(settings: Settings | None = None) -> BaseVectorStore:
    if settings is None:
        settings = Settings()

    vs_settings = settings.vector_store
    emb_settings = settings.embedding
    embeddings = create_embeddings(emb_settings)
    model_name = get_embedding_model_name(emb_settings.model)

    if vs_settings.provider == "chroma":
        store = ChromaVectorStore(
            persist_directory=vs_settings.chroma_persist_dir,
            embedding_function=embeddings,
            embedding_model_name=model_name,
        )
        store.validate_embedding_model(emb_settings.model)
        return store

    if vs_settings.provider == "pinecone":
        if not vs_settings.pinecone_api_key:
            raise ValueError(
                "VECTOR_STORE__PINECONE_API_KEY is required when VECTOR_STORE__PROVIDER=pinecone"
            )
        store = PineconeVectorStore(
            index_name=vs_settings.pinecone_index_name,
            embedding_function=embeddings,
            embedding_model_name=model_name,
        )
        store.validate_embedding_model(emb_settings.model)
        return store

    raise ValueError(
        f"Unknown vector store provider '{vs_settings.provider}'. "
        f"Available: chroma, pinecone"
    )