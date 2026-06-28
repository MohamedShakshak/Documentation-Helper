from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from doc_helper.config.settings import LLMSettings


def create_chat_model(settings: LLMSettings | None = None) -> BaseChatModel:
    if settings is None:
        settings = LLMSettings()

    if settings.provider == "ollama":
        return init_chat_model(
            settings.model,
            model_provider="ollama",
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
        )

    if settings.provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ValueError(
                "LLM__OPENROUTER_API_KEY is required when LLM__PROVIDER=openrouter"
            )
        return init_chat_model(
            settings.model,
            model_provider="openrouter",
            openrouter_api_key=settings.openrouter_api_key,
            temperature=settings.temperature,
        )

    raise ValueError(
        f"Unknown LLM provider '{settings.provider}'. Available: ollama, openrouter"
    )
