import uuid
from datetime import UTC, datetime

from doc_helper.db.connection import Database


class TaskManager:
    VALID_STATUSES = ("pending", "running", "completed", "failed")

    def __init__(self, db: Database):
        self._db = db

    def create_task(self, crawler: str) -> str:
        task_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        cursor = self._db.connection.cursor()
        cursor.execute(
            "INSERT INTO tasks "
            "(id, status, crawler, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (task_id, "pending", crawler, now, now),
        )
        self._db.connection.commit()
        return task_id

    def get_task(self, task_id: str) -> dict | None:
        cursor = self._db.connection.cursor()
        cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    def update_status(self, task_id: str, status: str) -> None:
        if status not in self.VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'. Must be one of {self.VALID_STATUSES}")
        now = datetime.now(UTC).isoformat()
        cursor = self._db.connection.cursor()
        cursor.execute(
            "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
            (status, now, task_id),
        )
        self._db.connection.commit()

    def update_progress(
        self,
        task_id: str,
        urls_crawled: int | None = None,
        chunks_created: int | None = None,
        progress: int | None = None,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        cursor = self._db.connection.cursor()
        updates = ["updated_at = ?"]
        params: list = [now]
        if urls_crawled is not None:
            updates.append("urls_crawled = ?")
            params.append(urls_crawled)
        if chunks_created is not None:
            updates.append("chunks_created = ?")
            params.append(chunks_created)
        if progress is not None:
            updates.append("progress = ?")
            params.append(progress)
        params.append(task_id)
        cursor.execute(f"UPDATE tasks SET {', '.join(updates)} WHERE id = ?", params)
        self._db.connection.commit()

    def set_error(self, task_id: str, error: str) -> None:
        now = datetime.now(UTC).isoformat()
        cursor = self._db.connection.cursor()
        cursor.execute(
            "UPDATE tasks SET status = ?, error = ?, updated_at = ? WHERE id = ?",
            ("failed", error, now, task_id),
        )
        self._db.connection.commit()

    def list_tasks(self, limit: int = 50) -> list[dict]:
        cursor = self._db.connection.cursor()
        cursor.execute("SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
