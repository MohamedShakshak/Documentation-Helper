import logging
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel

from doc_helper.config.settings import AgentSettings

logger = logging.getLogger(__name__)


def guardrails_middleware(
    settings: AgentSettings,
):
    """Pre-process hook: validate user input before it reaches the agent.
    Returns an error string if the input is rejected, or None to allow processing."""

    def _check_input(user_query: str) -> str | None:
        if not settings.guardrails_enabled:
            return None

        if len(user_query) > settings.guardrails_max_input_length:
            return (
                f"Input too long ({len(user_query)} chars). "
                f"Maximum allowed is {settings.guardrails_max_input_length}."
            )

        blocked_patterns = ["ignore previous", "disregard all", "you are now", "system prompt:"]
        lower = user_query.lower()
        for pattern in blocked_patterns:
            if pattern in lower:
                return "Input rejected by guardrails: detected potentially unsafe pattern."

        return None

    return _check_input


def summarization_middleware(
    conversation_manager,
    llm: BaseChatModel,
    settings: AgentSettings,
):
    """Pre-process hook: if conversation history exceeds the message threshold,
    summarize older messages into a compact context string. Mutates the
    conversation in-place by replacing old messages with a summary."""

    def _maybe_summarize(conversation_id: str | None) -> None:
        if not settings.summarization_enabled:
            return
        if conversation_id is None or conversation_manager is None:
            return

        messages = conversation_manager.get_messages(conversation_id)
        if len(messages) <= settings.summarization_threshold:
            return

        cutoff = settings.summarization_threshold // 2
        old_messages = messages[:cutoff]
        recent_messages = messages[cutoff:]

        summary_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in old_messages
        )

        try:
            response = llm.invoke(
                f"Summarize the following conversation concisely, preserving key context "
                f"and any specific details mentioned:\n\n{summary_text}"
            )
            summary = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.warning(f"Summarization failed: {e}. Proceeding with full history.")
            return

        conversation_manager.replace_messages(
            conversation_id,
            [{"role": "system", "content": f"Conversation summary:\n{summary}"}]
            + recent_messages,
        )

    return _maybe_summarize


def tool_retry_middleware(settings: AgentSettings):
    """Wraps a tool call function with retry logic. Returns a decorator."""

    max_retries = settings.max_tool_retries

    def _wrap(func: Callable) -> Callable:
        async def _retry_wrapper(*args, **kwargs) -> Any:
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"Tool call failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                    )
            raise last_error

        def _sync_wrapper(*args, **kwargs) -> Any:
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"Tool call failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                    )
            raise last_error

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return _retry_wrapper
        return _sync_wrapper

    return _wrap


def model_fallback_middleware(
    primary_llm: BaseChatModel,
    fallback_model_name: str | None = None,
    enabled: bool = True,
):
    """Wrap an LLM invocation with fallback to an alternative model.
    The fallback model is lazily created on first failure."""

    def _invoke(messages: list[dict]) -> Any:
        try:
            return primary_llm.invoke(messages)
        except Exception as e:
            if not enabled or not fallback_model_name:
                raise

            logger.warning(f"Primary LLM failed ({e}), falling back to {fallback_model_name}")

            from doc_helper.config.settings import LLMSettings
            from doc_helper.llm.factory import create_chat_model

            fallback_settings = LLMSettings(
                provider="ollama",
                model=fallback_model_name,
            )
            fallback_llm = create_chat_model(fallback_settings)
            return fallback_llm.invoke(messages)

    return _invoke
