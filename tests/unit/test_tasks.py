import os
import tempfile

import pytest

from doc_helper.config.settings import DatabaseSettings
from doc_helper.db.connection import Database
from doc_helper.db.tasks import TaskManager


@pytest.fixture
def task_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        settings = DatabaseSettings(url=f"sqlite:///{db_path}")
        database = Database(settings)
        database.connect()
        manager = TaskManager(database)
        yield manager
        database.close()


class TestTaskManager:
    def test_create_task(self, task_manager):
        task_id = task_manager.create_task(crawler="tavily")
        assert task_id is not None
        assert len(task_id) == 36

    def test_get_task(self, task_manager):
        task_id = task_manager.create_task(crawler="tavily")
        task = task_manager.get_task(task_id)
        assert task is not None
        assert task["id"] == task_id
        assert task["status"] == "pending"
        assert task["crawler"] == "tavily"

    def test_get_nonexistent_task(self, task_manager):
        task = task_manager.get_task("nonexistent")
        assert task is None

    def test_update_status(self, task_manager):
        task_id = task_manager.create_task(crawler="tavily")
        task_manager.update_status(task_id, "running")
        task = task_manager.get_task(task_id)
        assert task["status"] == "running"

    def test_update_status_invalid(self, task_manager):
        task_id = task_manager.create_task(crawler="tavily")
        with pytest.raises(ValueError, match="Invalid status"):
            task_manager.update_status(task_id, "invalid")

    def test_update_progress(self, task_manager):
        task_id = task_manager.create_task(crawler="recursive")
        task_manager.update_progress(task_id, urls_crawled=10, chunks_created=50, progress=30)
        task = task_manager.get_task(task_id)
        assert task["urls_crawled"] == 10
        assert task["chunks_created"] == 50
        assert task["progress"] == 30

    def test_set_error(self, task_manager):
        task_id = task_manager.create_task(crawler="tavily")
        task_manager.set_error(task_id, "Connection timed out")
        task = task_manager.get_task(task_id)
        assert task["status"] == "failed"
        assert task["error"] == "Connection timed out"

    def test_list_tasks(self, task_manager):
        task_manager.create_task(crawler="tavily")
        task_manager.create_task(crawler="recursive")
        tasks = task_manager.list_tasks()
        assert len(tasks) == 2