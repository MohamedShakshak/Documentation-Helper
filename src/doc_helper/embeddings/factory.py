EMBEDDING_MODELS: dict[str, dict[str, int | str]] = {
    "bge-small": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "dim": 384,
    },
    "bge-base": {
        "model_name": "BAAI/bge-base-en-v1.5",
        "dim": 768,
    },
}


def create_embeddings(settings):
    from langchain_huggingface import HuggingFaceEmbeddings

    from doc_helper.config.settings import EmbeddingSettings

    if not isinstance(settings, EmbeddingSettings):
        settings = EmbeddingSettings()

    config = EMBEDDING_MODELS[settings.model]
    return HuggingFaceEmbeddings(
        model_name=config["model_name"],
        encode_kwargs={"normalize_embeddings": settings.normalize},
    )


def get_embedding_model_name(model_key: str) -> str:
    if model_key not in EMBEDDING_MODELS:
        raise ValueError(
            f"Unknown embedding model '{model_key}'. "
            f"Available: {list(EMBEDDING_MODELS.keys())}"
        )
    return str(EMBEDDING_MODELS[model_key]["model_name"])


def get_embedding_dimension(model_key: str) -> int:
    if model_key not in EMBEDDING_MODELS:
        raise ValueError(
            f"Unknown embedding model '{model_key}'. "
            f"Available: {list(EMBEDDING_MODELS.keys())}"
        )
    return int(EMBEDDING_MODELS[model_key]["dim"])