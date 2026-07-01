from unittest.mock import MagicMock, patch

import pytest

from doc_helper.config.settings import LLMSettings
from doc_helper.llm.factory import create_chat_model


class TestCreateChatModel:
    def test_ollama_default(self):
        settings = LLMSettings()
        with patch("doc_helper.llm.factory.init_chat_model") as mock_init:
            mock_init.return_value = MagicMock()
            create_chat_model(settings)
            mock_init.assert_called_once_with(
                "qwen3.5:9b",
                model_provider="ollama",
                base_url="http://localhost:11434",
                temperature=0.0,
            )

    def test_ollama_custom_model(self):
        settings = LLMSettings(model="llama3:8b")
        with patch("doc_helper.llm.factory.init_chat_model") as mock_init:
            mock_init.return_value = MagicMock()
            create_chat_model(settings)
            mock_init.assert_called_once_with(
                "llama3:8b",
                model_provider="ollama",
                base_url="http://localhost:11434",
                temperature=0.0,
            )

    def test_openrouter_with_api_key(self):
        settings = LLMSettings(provider="openrouter", model="gpt-4o", openrouter_api_key="sk-test")
        with patch("langchain_openai.ChatOpenAI") as mock_chat:
            mock_chat.return_value = MagicMock()
            create_chat_model(settings)
            mock_chat.assert_called_once_with(
                model="gpt-4o",
                base_url="https://openrouter.ai/api/v1",
                api_key="sk-test",
                temperature=0.0,
            )

    def test_openrouter_without_api_key_raises(self):
        settings = LLMSettings(provider="openrouter", openrouter_api_key=None)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            create_chat_model(settings)

    def test_defaults_when_none(self):
        with patch("doc_helper.llm.factory.init_chat_model") as mock_init:
            mock_init.return_value = MagicMock()
            create_chat_model(None)
            mock_init.assert_called_once()
            assert mock_init.call_args[0][0] == "qwen3.5:9b"