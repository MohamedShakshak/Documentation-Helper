import os
import tempfile

import pytest

from doc_helper.config.settings import DatabaseSettings
from doc_helper.db.connection import Database


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        settings = DatabaseSettings(url=f"sqlite:///{db_path}")
        database = Database(settings)
        database.connect()
        yield database
        database.close()


class TestDatabase:
    def test_connect_creates_tables(self, db):
        cursor = db.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row["name"] for row in cursor.fetchall()}
        assert "conversations" in tables
        assert "messages" in tables
        assert "tasks" in tables
        assert "metadata" in tables

    def test_connect_idempotent(self, db):
        conn1 = db.connection
        conn2 = db.connection
        assert conn1 is conn2

    def test_close(self, db):
        db.close()
        assert db._connection is None

    def test_reconnect_after_close(self, db):
        db.close()
        conn = db.connect()
        assert conn is not None
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row["name"] for row in cursor.fetchall()}
        assert "conversations" in tables