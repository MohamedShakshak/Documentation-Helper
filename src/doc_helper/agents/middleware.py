import asyncio
import logging

from langchain_core.language_models import BaseChatModel

from doc_helper.config.settings import AgentSettings

logger = logging.getLogger(__name__)


def guardrails_middleware(settings: AgentSettings):
    def _check_input(user_query: str) -> str | None:
        if not settings.guardrails_enabled:
            return None

        if len(user_query) > settings.guardrails_max_input_length:
            return (
                f"Input too long ({len(user_query)} chars). "
                f"Maximum allowed is {settings.guardrails_max_input_length}."
            )

        # Basic injection detection — not exhaustive, sufficient for demo purposes
        blocked_patterns = [
            "ignore previous",
            "disregard all",
            "you are now",
            "system prompt:",
            "forget your instructions",
            "act as",
            "jailbreak",
            "dan mode",
        ]
        lower = user_query.lower().replace("  ", " ")
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
    async def _maybe_summarize(conversation_id: str | None) -> None:
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
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                llm.invoke,
                f"Summarize the following conversation concisely, preserving key context "
                f"and any specific details mentioned:\n\n{summary_text}",
            )
            summary = response.content if hasattr(response, "content") else str(response)
            logger.info(
                f"Summarized {len(old_messages)} messages into 1 for "
                f"conversation {conversation_id}"
            )
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
    max_retries = settings.max_tool_retries

    def _wrap(func):
        async def _retry_wrapper(*args, **kwargs):
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

        def _sync_wrapper(*args, **kwargs):
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

        if asyncio.iscoroutinefunction(func):
            return _retry_wrapper
        return _sync_wrapper

    return _wrap
