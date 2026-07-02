from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from doc_helper.evaluation.generator_metrics import METRIC_NAMES


class TestGeneratorEvaluator:
    def test_judge_uses_judge_settings_when_configured(self):
        from doc_helper.config.settings import (
            JudgeLLMSettings,
            Settings,
        )
        from doc_helper.evaluation.generator_evaluator import GeneratorEvaluator

        settings = Settings(
            judge_llm=JudgeLLMSettings(
                provider="openrouter", model="gpt-4o", openrouter_api_key="sk-test"
            ),
        )
        with patch("doc_helper.llm.factory.create_chat_model") as mock_create:
            mock_create.return_value = MagicMock()
            GeneratorEvaluator(settings)
            assert mock_create.call_count == 1
            judge_arg = mock_create.call_args[0][0]
            assert judge_arg.provider == "openrouter"
            assert judge_arg.model == "gpt-4o"

    def test_judge_falls_back_to_llm_settings(self):
        from doc_helper.config.settings import JudgeLLMSettings, Settings
        from doc_helper.evaluation.generator_evaluator import GeneratorEvaluator

        settings = Settings(judge_llm=JudgeLLMSettings())
        with patch("doc_helper.llm.factory.create_chat_model") as mock_create:
            mock_create.return_value = MagicMock()
            GeneratorEvaluator(settings)
            judge_arg = mock_create.call_args[0][0]
            assert judge_arg.provider == settings.llm.provider

    @pytest.mark.asyncio
    async def test_evaluate_with_mocked_agent_and_judge(self):
        from doc_helper.evaluation.generator_evaluator import GeneratorEvaluator

        mock_judge = MagicMock()
        mock_judge.ainvoke = AsyncMock(
            return_value=MagicMock(content='{"score": 0.8, "reason": "good"}')
        )

        mock_agent = MagicMock()
        mock_agent.run = MagicMock(
            return_value={
                "answer": "LangChain is a framework.",
                "context": [MagicMock(page_content="LangChain is a framework for LLMs.")],
                "sources": ["https://docs.langchain.com/oss/python/langchain"],
            }
        )

        evaluator = GeneratorEvaluator.__new__(GeneratorEvaluator)
        evaluator._settings = MagicMock()
        evaluator._judge = mock_judge

        with patch("doc_helper.agents.rag_agent.create_rag_agent", return_value=mock_agent):
            dataset = [
                {
                    "question": "What is LangChain?",
                    "reference_answer": "LangChain is a framework for LLMs.",
                    "difficulty": "simple",
                    "relevant_urls": ["https://docs.langchain.com/oss/python/langchain"],
                    "key_facts": ["framework", "LLM"],
                }
            ]
            report = await evaluator.evaluate(dataset)

        assert report["total_queries"] == 1
        assert "overall" in report
        for name in METRIC_NAMES:
            assert report["overall"][name] == 0.8
        assert len(report["per_query"]) == 1
        assert report["per_query"][0]["answer"] == "LangChain is a framework."
        assert len(report["worst_queries"]) == 1

    @pytest.mark.asyncio
    async def test_evaluate_agent_error_gets_zero_scores(self):
        from doc_helper.evaluation.generator_evaluator import GeneratorEvaluator

        mock_agent = MagicMock()
        mock_agent.run = MagicMock(side_effect=RuntimeError("API timeout"))

        evaluator = GeneratorEvaluator.__new__(GeneratorEvaluator)
        evaluator._settings = MagicMock()
        evaluator._judge = MagicMock()

        with patch("doc_helper.agents.rag_agent.create_rag_agent", return_value=mock_agent):
            dataset = [
                {
                    "question": "What is X?",
                    "reference_answer": "X is Y.",
                    "difficulty": "simple",
                    "relevant_urls": [],
                    "key_facts": ["Y"],
                }
            ]
            report = await evaluator.evaluate(dataset)

        assert report["total_queries"] == 1
        for name in METRIC_NAMES:
            assert report["overall"][name] == 0.0
        assert report["per_query"][0]["answer"].startswith("ERROR")

    @pytest.mark.asyncio
    async def test_evaluate_empty_dataset(self):
        from doc_helper.evaluation.generator_evaluator import GeneratorEvaluator

        evaluator = GeneratorEvaluator.__new__(GeneratorEvaluator)
        evaluator._settings = MagicMock()
        evaluator._judge = MagicMock()

        report = await evaluator.evaluate([])
        assert report["total_queries"] == 0
        assert report["overall"] == {}

    def test_aggregate_by_difficulty(self):
        from doc_helper.evaluation.generator_evaluator import GeneratorEvaluator

        per_query = [
            {
                "difficulty": "simple",
                "scores": {n: {"score": 1.0} for n in METRIC_NAMES},
                "total_score": 4.0,
            },
            {
                "difficulty": "simple",
                "scores": {n: {"score": 0.5} for n in METRIC_NAMES},
                "total_score": 2.0,
            },
            {
                "difficulty": "edge_case",
                "scores": {n: {"score": 0.0} for n in METRIC_NAMES},
                "total_score": 0.0,
            },
        ]
        result = GeneratorEvaluator._aggregate_by_difficulty(per_query)
        assert len(result) == 2
        assert result["simple"]["faithfulness"] == 0.75
        assert result["edge_case"]["faithfulness"] == 0.0

    def test_worst_queries_sorted_ascending(self):
        from doc_helper.evaluation.generator_evaluator import GeneratorEvaluator

        per_query = [
            {
                "question": f"Q{i}",
                "difficulty": "simple",
                "scores": {n: {"score": 1.0} for n in METRIC_NAMES},
                "total_score": float(i),
            }
            for i in range(5)
        ]
        worst = GeneratorEvaluator._worst_queries(per_query, n=3)
        assert len(worst) == 3
        assert worst[0]["total_score"] == 0.0
        assert worst[1]["total_score"] == 1.0
        assert worst[2]["total_score"] == 2.0
