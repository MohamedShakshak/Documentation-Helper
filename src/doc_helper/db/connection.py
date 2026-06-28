import sqlite3
from pathlib import Path

from doc_helper.config.settings import DatabaseSettings

_MIGRATIONS = [
    """CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );""",
    """CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        sources TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
    );""",
    """CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY,
        status TEXT NOT NULL DEFAULT 'pending',
        progress INTEGER DEFAULT 0,
        crawler TEXT NOT NULL,
        urls_crawled INTEGER DEFAULT 0,
        chunks_created INTEGER DEFAULT 0,
        error TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );""",
    """CREATE TABLE IF NOT EXISTS metadata (
        key TEXT PRIMARY KEY,
        value TEXT
    );""",
]


class Database:
    def __init__(self, settings: DatabaseSettings | None = None):
        if settings is None:
            settings = DatabaseSettings()
        self._url = settings.url
        self._connection: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        if self._connection is not None:
            return self._connection

        url = self._url
        if url.startswith("sqlite:///"):
            path = url.replace("sqlite:///", "")
            db_dir = Path(path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
        else:
            path = url

        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._run_migrations()
        return self._connection

    def _run_migrations(self) -> None:
        cursor = self._connection.cursor()
        for migration in _MIGRATIONS:
            cursor.execute(migration)
        self._connection.commit()

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            return self.connect()
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
