import json
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from doc_helper.agents.events import ErrorEvent
from doc_helper.api.deps import get_agent, get_conversation_manager

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    conversation_id: str | None = None


class CreateConversationRequest(BaseModel):
    title: str | None = None


class ConversationResponse(BaseModel):
    conversation_id: str
    title: str | None = None


class ConversationListResponse(BaseModel):
    conversations: list[dict]


@router.post("")
async def chat(request: ChatRequest) -> dict[str, Any]:
    agent = get_agent()

    if request.conversation_id:
        conv_mgr = get_conversation_manager()
        messages = conv_mgr.get_messages(request.conversation_id)
        if messages is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

    result = agent.run(query=request.query, conversation_id=request.conversation_id)
    return {
        "answer": result["answer"],
        "sources": result.get("sources", []),
    }


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    from fastapi.responses import StreamingResponse

    agent = get_agent()

    async def event_generator():
        try:
            async for event in agent.astream(
                query=request.query,
                conversation_id=request.conversation_id,
            ):
                yield f"event: {event.event.value}\ndata: {json.dumps(event.data)}\n\n"
        except Exception as e:
            err = ErrorEvent(message=str(e))
            yield f"event: {err.event.value}\ndata: {json.dumps(err.data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(request: CreateConversationRequest):
    conv_mgr = get_conversation_manager()
    conv_id = conv_mgr.create_conversation(title=request.title)
    return ConversationResponse(conversation_id=conv_id, title=request.title)


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations():
    conv_mgr = get_conversation_manager()
    return ConversationListResponse(conversations=conv_mgr.list_conversations())


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    conv_mgr = get_conversation_manager()
    conversations = conv_mgr.list_conversations()
    existing_ids = {c["id"] for c in conversations}
    if conversation_id not in existing_ids:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = conv_mgr.get_messages(conversation_id)
    return {"conversation_id": conversation_id, "messages": messages}


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    conv_mgr = get_conversation_manager()
    conv_mgr.delete_conversation(conversation_id)
    return {"status": "deleted", "conversation_id": conversation_id}
