import json
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from doc_helper.db.connection import Database


class ConversationManager:
    def __init__(self, db: Database):
        self._db = db

    def create_conversation(self, title: str | None = None) -> str:
        import uuid

        conv_id = str(uuid.uuid4())
        cursor = self._db.connection.cursor()
        cursor.execute(
            "INSERT INTO conversations (id, title) VALUES (?, ?)",
            (conv_id, title),
        )
        self._db.connection.commit()
        return conv_id

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: list[str] | None = None,
    ) -> int:
        cursor = self._db.connection.cursor()
        cursor.execute(
            "INSERT INTO messages (conversation_id, role, content, sources) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, json.dumps(sources) if sources else None),
        )
        self._db.connection.commit()
        return cursor.lastrowid

    def get_messages(self, conversation_id: str) -> list[dict]:
        cursor = self._db.connection.cursor()
        cursor.execute(
            "SELECT role, content, sources FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        )
        rows = cursor.fetchall()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "sources": json.loads(row["sources"]) if row["sources"] else [],
            }
            for row in rows
        ]

    def get_langchain_messages(self, conversation_id: str) -> list[BaseMessage]:
        rows = self.get_messages(conversation_id)
        messages: list[BaseMessage] = []
        for row in rows:
            if row["role"] == "user":
                messages.append(HumanMessage(content=row["content"]))
            elif row["role"] == "assistant":
                messages.append(AIMessage(content=row["content"]))
            elif row["role"] == "system":
                messages.append(SystemMessage(content=row["content"]))
        return messages

    def list_conversations(self) -> list[dict]:
        cursor = self._db.connection.cursor()
        cursor.execute(
            "SELECT id, title, created_at FROM conversations ORDER BY created_at DESC"
        )
        rows = cursor.fetchall()
        return [
            {"id": row["id"], "title": row["title"], "created_at": row["created_at"]}
            for row in rows
        ]

    def delete_conversation(self, conversation_id: str) -> None:
        cursor = self._db.connection.cursor()
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        self._db.connection.commit()

    def get_message_count(self, conversation_id: str) -> int:
        cursor = self._db.connection.cursor()
        cursor.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        )
        return cursor.fetchone()["cnt"]

    def replace_messages(
        self, conversation_id: str, new_messages: list[dict]
    ) -> None:
        cursor = self._db.connection.cursor()
        cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        for msg in new_messages:
            cursor.execute(
                "INSERT INTO messages (conversation_id, role, content, sources) VALUES (?, ?, ?, ?)",
                (
                    conversation_id,
                    msg["role"],
                    msg["content"],
                    json.dumps(msg.get("sources", [])) if msg.get("sources") else None,
                ),
            )
        self._db.connection.commit()