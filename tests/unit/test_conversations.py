import os
import tempfile

import pytest

from doc_helper.config.settings import DatabaseSettings
from doc_helper.db.connection import Database
from doc_helper.db.conversations import ConversationManager


@pytest.fixture
def conversation_manager():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        settings = DatabaseSettings(url=f"sqlite:///{db_path}")
        database = Database(settings)
        database.connect()
        manager = ConversationManager(database)
        yield manager
        database.close()


class TestConversationManager:
    def test_create_conversation(self, conversation_manager):
        conv_id = conversation_manager.create_conversation(title="Test Chat")
        assert conv_id is not None
        assert len(conv_id) == 36  # UUID format

    def test_create_conversation_without_title(self, conversation_manager):
        conv_id = conversation_manager.create_conversation()
        assert conv_id is not None

    def test_add_and_get_messages(self, conversation_manager):
        conv_id = conversation_manager.create_conversation(title="Test")
        conversation_manager.add_message(conv_id, "user", "Hello")
        conversation_manager.add_message(conv_id, "assistant", "Hi there!", ["source1"])

        messages = conversation_manager.get_messages(conv_id)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi there!"
        assert messages[1]["sources"] == ["source1"]

    def test_get_langchain_messages(self, conversation_manager):
        conv_id = conversation_manager.create_conversation()
        conversation_manager.add_message(conv_id, "user", "What is LangChain?")
        conversation_manager.add_message(conv_id, "assistant", "LangChain is a framework")

        messages = conversation_manager.get_langchain_messages(conv_id)
        assert len(messages) == 2
        assert messages[0].content == "What is LangChain?"
        assert messages[1].content == "LangChain is a framework"

    def test_list_conversations(self, conversation_manager):
        conversation_manager.create_conversation(title="Chat 1")
        conversation_manager.create_conversation(title="Chat 2")

        convs = conversation_manager.list_conversations()
        assert len(convs) == 2

    def test_delete_conversation(self, conversation_manager):
        conv_id = conversation_manager.create_conversation(title="Test")
        conversation_manager.add_message(conv_id, "user", "Hello")

        conversation_manager.delete_conversation(conv_id)
        messages = conversation_manager.get_messages(conv_id)
        assert len(messages) == 0

    def test_messages_without_sources(self, conversation_manager):
        conv_id = conversation_manager.create_conversation()
        conversation_manager.add_message(conv_id, "user", "Hello")

        messages = conversation_manager.get_messages(conv_id)
        assert messages[0]["sources"] == []