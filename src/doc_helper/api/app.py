from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from doc_helper.api.deps import get_db
from doc_helper.api.routes import chat, health, ingest
from doc_helper.logger import log_header, log_info, log_success


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_header("DOCUMENTATION HELPER API")
    db = get_db()
    log_info(f"Database connected: {db._url}")

    try:
        get_agent_ref = app
    except Exception:
        pass

    log_success("API server ready")
    yield

    log_info("Shutting down...")
    from doc_helper.api.deps import reset_caches

    reset_caches()
    log_info("Cleanup complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Documentation Helper API",
        description="RAG-powered documentation assistant with multi-tool agent",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
    app.include_router(ingest.router, prefix="/api/ingest", tags=["ingestion"])

    return app


app = create_app()