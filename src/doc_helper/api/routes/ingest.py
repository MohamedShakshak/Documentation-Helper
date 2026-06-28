import asyncio
import traceback
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from doc_helper.api.deps import get_config, get_task_manager
from doc_helper.ingestion.pipeline import run_ingestion

router = APIRouter()


class IngestRequest(BaseModel):
    crawler: str | None = None
    url: str | None = None
    depth: int | None = None


class IngestResponse(BaseModel):
    task_id: str
    status: str = "pending"


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: int
    urls_crawled: int
    chunks_created: int
    error: str | None = None


@router.post("", response_model=IngestResponse)
async def start_ingestion(request: IngestRequest):
    config = get_config()
    task_mgr = get_task_manager()

    crawler_type = request.crawler or config.ingestion.crawler
    task_id = task_mgr.create_task(crawler_type)

    if request.url:
        config.ingestion.crawl_url = request.url
    if request.depth:
        config.ingestion.crawl_depth = request.depth
    if request.crawler:
        config.ingestion.crawler = request.crawler

    asyncio.create_task(_run_ingestion_task(task_id, config))

    return IngestResponse(task_id=task_id, status="pending")


@router.get("/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    task_mgr = get_task_manager()
    task = task_mgr.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatusResponse(
        task_id=task["id"],
        status=task["status"],
        progress=task["progress"],
        urls_crawled=task["urls_crawled"],
        chunks_created=task["chunks_created"],
        error=task["error"],
    )


@router.get("", response_model=list[dict])
async def list_tasks(limit: int = 20):
    task_mgr = get_task_manager()
    return task_mgr.list_tasks(limit=limit)


async def _run_ingestion_task(task_id: str, config: Any) -> None:
    task_mgr = get_task_manager()
    task_mgr.update_status(task_id, "running")

    try:
        await run_ingestion(config)
        task_mgr.update_status(task_id, "completed")
    except Exception as e:
        task_mgr.set_error(task_id, f"{e}\n{traceback.format_exc()}")
