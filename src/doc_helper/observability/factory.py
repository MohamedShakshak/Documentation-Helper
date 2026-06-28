from doc_helper.config.settings import ObservabilitySettings
from doc_helper.observability.base import BaseTracer, NoOpTracer


def create_tracer(settings: ObservabilitySettings | None = None) -> BaseTracer:
    if settings is None:
        settings = ObservabilitySettings()

    if not settings.enabled:
        return NoOpTracer()

    if settings.provider == "langfuse":
        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            raise ValueError(
                "LangFuse tracer requires OBSERVABILITY__LANGFUSE_PUBLIC_KEY "
                "and OBSERVABILITY__LANGFUSE_SECRET_KEY"
            )
        from doc_helper.observability.langfuse_tracer import LangFuseTracer

        return LangFuseTracer(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )

    if settings.provider == "langsmith":
        if not settings.langsmith_api_key:
            raise ValueError("LangSmith tracer requires OBSERVABILITY__LANGSMITH_API_KEY")
        from doc_helper.observability.langsmith_tracer import LangSmithTracer

        return LangSmithTracer(api_key=settings.langsmith_api_key)

    raise ValueError(f"Unknown observability provider: {settings.provider}")
