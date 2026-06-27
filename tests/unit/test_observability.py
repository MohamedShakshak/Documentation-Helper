from unittest.mock import MagicMock, patch

import pytest

from doc_helper.config.settings import ObservabilitySettings
from doc_helper.observability.base import NoOpTracer
from doc_helper.observability.factory import create_tracer


class TestCreateTracer:
    def test_disabled_returns_noop(self):
        settings = ObservabilitySettings(enabled=False)
        tracer = create_tracer(settings)
        assert isinstance(tracer, NoOpTracer)

    def test_noop_tracer_methods_are_noops(self):
        tracer = NoOpTracer()
        assert tracer.trace_retrieval("query", []) is None
        assert tracer.trace_tool_call("tool", {}, None) is None
        assert tracer.trace_llm_call("prompt", "response") is None
        assert tracer.trace_agent_run("query", "answer", []) is None
        assert tracer.flush() is None

    def test_langfuse_without_keys_raises(self):
        settings = ObservabilitySettings(
            enabled=True, provider="langfuse",
            langfuse_public_key=None, langfuse_secret_key=None,
        )
        with pytest.raises(ValueError, match="LangFuse"):
            create_tracer(settings)

    def test_langsmith_without_key_raises(self):
        settings = ObservabilitySettings(
            enabled=True, provider="langsmith", langsmith_api_key=None,
        )
        with pytest.raises(ValueError, match="LangSmith"):
            create_tracer(settings)

    def test_langfuse_creates_tracer(self):
        with patch("langfuse.Langfuse") as mock_cls:
            mock_cls.return_value = MagicMock()

            settings = ObservabilitySettings(
                enabled=True,
                provider="langfuse",
                langfuse_public_key="pk-test",
                langfuse_secret_key="sk-test",
            )
            tracer = create_tracer(settings)
            assert hasattr(tracer, "trace_retrieval")
            assert hasattr(tracer, "flush")
            mock_cls.assert_called_once()

    def test_langsmith_creates_tracer(self):
        settings = ObservabilitySettings(
            enabled=True, provider="langsmith", langsmith_api_key="sk-test",
        )
        tracer = create_tracer(settings)
        assert hasattr(tracer, "trace_retrieval")
        assert hasattr(tracer, "flush")

    def test_unknown_provider_raises(self):
        settings = ObservabilitySettings(enabled=True)
        settings.provider = "unknown"
        with pytest.raises(ValueError, match="Unknown"):
            create_tracer(settings)


class TestNoOpTracer:
    def test_all_methods_return_none(self):
        tracer = NoOpTracer()
        assert tracer.trace_retrieval("q", []) is None
        assert tracer.trace_tool_call("t", {}, None) is None
        assert tracer.trace_llm_call("p", "r", {"k": "v"}) is None
        assert tracer.trace_agent_run("q", "a", ["s"]) is None
        assert tracer.flush() is None


class TestEvaluationDataset:
    def test_load_gold_dataset(self):
        from doc_helper.evaluation.dataset import load_gold_dataset

        dataset = load_gold_dataset()
        assert len(dataset) >= 25
        for item in dataset:
            assert "question" in item
            assert "answer" in item
            assert "difficulty" in item

    def test_get_questions_by_difficulty(self):
        from doc_helper.evaluation.dataset import get_questions_by_difficulty

        simple = get_questions_by_difficulty("simple")
        multi_hop = get_questions_by_difficulty("multi_hop")
        edge_case = get_questions_by_difficulty("edge_case")

        assert len(simple) > 0
        assert len(multi_hop) > 0
        assert len(edge_case) > 0

    def test_difficulty_grouping_covers_all(self):
        from doc_helper.evaluation.dataset import (
            get_questions_by_difficulty,
            load_gold_dataset,
        )

        dataset = load_gold_dataset()
        grouped_count = (
            len(get_questions_by_difficulty("simple"))
            + len(get_questions_by_difficulty("multi_hop"))
            + len(get_questions_by_difficulty("edge_case"))
        )
        assert grouped_count == len(dataset)