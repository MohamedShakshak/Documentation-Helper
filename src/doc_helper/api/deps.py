from functools import lru_cache

from doc_helper.config.settings import Settings, get_settings
from doc_helper.db.connection import Database
from doc_helper.db.conversations import ConversationManager
from doc_helper.db.tasks import TaskManager


_settings: Settings | None = None
_db: Database | None = None
_conversation_mgr: ConversationManager | None = None
_task_mgr: TaskManager | None = None
_agent = None


def get_config() -> Settings:
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings


def get_db() -> Database:
    global _db
    if _db is None:
        _db = Database(get_config().database)
        _db.connect()
    return _db


def get_conversation_manager() -> ConversationManager:
    global _conversation_mgr
    if _conversation_mgr is None:
        _conversation_mgr = ConversationManager(get_db())
    return _conversation_mgr


def get_task_manager() -> TaskManager:
    global _task_mgr
    if _task_mgr is None:
        _task_mgr = TaskManager(get_db())
    return _task_mgr


def get_agent():
    global _agent
    if _agent is None:
        from doc_helper.agents.rag_agent import create_rag_agent

        conv_mgr = get_conversation_manager()
        _agent = create_rag_agent(
            settings=get_config(),
            conversation_manager=conv_mgr,
        )
    return _agent


def reset_caches() -> None:
    global _settings, _db, _conversation_mgr, _task_mgr, _agent
    if _db is not None:
        _db.close()
    _settings = None
    _db = None
    _conversation_mgr = None
    _task_mgr = None
    _agent = None