from fastapi import APIRouter

from doc_helper.api.deps import get_config

router = APIRouter()


@router.get("/health")
async def health_check():
    settings = get_config()
    return {
        "status": "ok",
        "llm_provider": settings.llm.provider,
        "llm_model": settings.llm.model,
        "embedding_model": settings.embedding.model,
        "vector_store": settings.vector_store.provider,
        "crawler": settings.ingestion.crawler,
    }
