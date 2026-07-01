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
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.model,
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
            temperature=settings.temperature,
        )

    if settings.provider == "gemini":
        if not settings.gemini_api_key:
            raise ValueError(
                "LLM__GEMINI_API_KEY is required when LLM__PROVIDER=gemini"
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.model,
            google_api_key=settings.gemini_api_key,
            temperature=settings.temperature,
        )

    raise ValueError(
        f"Unknown LLM provider '{settings.provider}'. "
        "Available: ollama, openrouter, gemini"
    )
