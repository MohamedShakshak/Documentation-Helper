import json
import os
from typing import Any

import httpx
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000/api")

st.set_page_config(page_title="Documentation Helper", layout="centered")
st.title("Documentation Helper")


def api_get(path: str) -> dict | None:
    try:
        resp = httpx.get(f"{API_BASE}{path}", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, json_body: dict | None = None) -> dict | None:
    try:
        resp = httpx.post(f"{API_BASE}{path}", json=json_body or {}, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_delete(path: str) -> dict | None:
    try:
        resp = httpx.delete(f"{API_BASE}{path}", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def stream_chat(query: str, conversation_id: str | None) -> Any:
    url = f"{API_BASE}/chat/stream"
    body = {"query": query}
    if conversation_id:
        body["conversation_id"] = conversation_id
    with httpx.stream("POST", url, json=body, timeout=120) as resp:
        resp.raise_for_status()
        event_type = None
        for line in resp.iter_lines():
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                yield event_type, data
                event_type = None


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None


def health_check() -> bool:
    result = api_get("/health")
    return result is not None and result.get("status") == "ok"


init_state()

with st.sidebar:
    st.subheader("Connection")
    if st.button("Check API"):
        with st.spinner("Checking..."):
            if health_check():
                st.success("API connected")
            else:
                st.error("API unreachable")

    st.diver()

    st.subheader("Conversation")
    if st.button("New conversation"):
        result = api_post("/chat/conversations", {"title": None})
        if result:
            st.session_state.conversation_id = result["conversation_id"]
            st.session_state.messages = []
            st.rerun()

    if st.session_state.conversation_id:
        st.caption(f"ID: `{st.session_state.conversation_id}`")

    if st.button("Load history") and st.session_state.conversation_id:
        result = api_get(f"/chat/conversations/{st.session_state.conversation_id}")
        if result:
            st.session_state.messages = [
                {"role": m["role"], "content": m["content"], "sources": m.get("sources", [])}
                for m in result["messages"]
            ]
            st.rerun()

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.conversation_id = None
        st.rerun()

    st.diver()

    st.subheader("Ingestion")
    crawl_url = st.text_input("URL to crawl", value="https://python.langchain.com/")
    if st.button("Start ingestion"):
        result = api_post("/ingest", {"url": crawl_url})
        if result:
            st.info(f"Task ID: `{result['task_id']}`")
            st.session_state.ingest_task_id = result["task_id"]

    if "ingest_task_id" in st.session_state:
        if st.button("Check ingestion status"):
            result = api_get(f"/ingest/{st.session_state.ingest_task_id}/status")
            if result:
                st.write(f"Status: **{result['status']}**")
                st.write(f"Progress: {result['progress']}%")
                if result.get("error"):
                    st.error(result["error"])


if not st.session_state.messages:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Ask me anything about LangChain docs. "
                "I'll retrieve context, cite sources, "
                "and can even search the web."
            ),
            "sources": [],
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

prompt = st.chat_input("Ask a question about LangChain...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer_parts: list[str] = []
        sources: list[str] = []

        with st.spinner("Thinking..."):
            try:
                answer_placeholder = st.empty()
                for event_type, data in stream_chat(prompt, st.session_state.conversation_id):
                    if event_type == "answer":
                        answer_parts.append(data.get("content", ""))
                        answer_placeholder.markdown("".join(answer_parts))
                    elif event_type == "tool_call":
                        with st.status(f"Calling {data.get('tool', 'tool')}...", expanded=False):
                            st.write(f"Query: {data.get('query', '')}")
                    elif event_type == "tool_result":
                        sources.extend(data.get("sources", []))
                    elif event_type == "error":
                        st.error(data.get("message", "Unknown error"))
                        break

                answer = "".join(answer_parts).strip() or "(No answer returned.)"
                st.markdown(answer)

                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.markdown(f"- {s}")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )

            except Exception as e:
                st.error("Failed to generate response.")
                st.exception(e)
