from unittest.mock import AsyncMock, MagicMock

import pytest

from doc_helper.evaluation.generator_metrics import (
    answer_correctness,
    answer_relevancy,
    context_utilization,
    faithfulness,
    zero_scores,
)


def _mock_judge(response_text: str) -> MagicMock:
    judge = MagicMock()
    judge.ainvoke = AsyncMock(return_value=MagicMock(content=response_text))
    return judge


class TestFaithfulness:
    @pytest.mark.asyncio
    async def test_high_score(self):
        judge = _mock_judge('{"score": 0.95, "reason": "All claims supported by contexts"}')
        result = await faithfulness(judge, "What is X?", "X is a tool.", ["X is a tool for apps."])
        assert result["score"] == 0.95
        assert "supported" in result["reason"]

    @pytest.mark.asyncio
    async def test_low_score(self):
        judge = _mock_judge('{"score": 0.25, "reason": "1 of 4 claims supported"}')
        result = await faithfulness(judge, "Q", "A", ["C"])
        assert result["score"] == 0.25

    @pytest.mark.asyncio
    async def test_clamps_score_above_1(self):
        judge = _mock_judge('{"score": 1.5, "reason": "all"}')
        result = await faithfulness(judge, "Q", "A", ["C"])
        assert result["score"] == 1.0

    @pytest.mark.asyncio
    async def test_clamps_score_below_0(self):
        judge = _mock_judge('{"score": -0.5, "reason": "none"}')
        result = await faithfulness(judge, "Q", "A", ["C"])
        assert result["score"] == 0.0

    @pytest.mark.asyncio
    async def test_parse_failure_returns_zero(self):
        judge = _mock_judge("not valid json at all")
        result = await faithfulness(judge, "Q", "A", ["C"])
        assert result["score"] == 0.0
        assert "Failed to parse" in result["reason"]


class TestAnswerRelevancy:
    @pytest.mark.asyncio
    async def test_relevant_answer(self):
        judge = _mock_judge('{"score": 0.9, "reason": "Directly addresses the question"}')
        result = await answer_relevancy(judge, "What is X?", "X is a framework.")
        assert result["score"] == 0.9

    @pytest.mark.asyncio
    async def test_irrelevant_answer(self):
        judge = _mock_judge('{"score": 0.1, "reason": "Off-topic"}')
        result = await answer_relevancy(judge, "What is X?", "Bananas are yellow.")
        assert result["score"] == 0.1


class TestAnswerCorrectness:
    @pytest.mark.asyncio
    async def test_all_key_facts_present(self):
        judge = _mock_judge('{"score": 1.0, "reason": "All key facts found"}')
        result = await answer_correctness(
            judge, "Q", "A has all facts", "Reference", ["fact1", "fact2"]
        )
        assert result["score"] == 1.0

    @pytest.mark.asyncio
    async def test_missing_key_facts(self):
        judge = _mock_judge('{"score": 0.5, "reason": "Missing: fact2"}')
        result = await answer_correctness(
            judge, "Q", "A has fact1 only", "Reference", ["fact1", "fact2"]
        )
        assert result["score"] == 0.5
        assert "fact2" in result["reason"]


class TestContextUtilization:
    @pytest.mark.asyncio
    async def test_fully_derived_from_contexts(self):
        judge = _mock_judge('{"score": 0.95, "reason": "Uses contexts heavily"}')
        result = await context_utilization(judge, "Answer based on context", ["Context with info"])
        assert result["score"] == 0.95

    @pytest.mark.asyncio
    async def test_ignores_contexts(self):
        judge = _mock_judge('{"score": 0.1, "reason": "Answers from training data"}')
        result = await context_utilization(judge, "Answer from training", ["Context"])
        assert result["score"] == 0.1


class TestZeroScores:
    def test_all_metrics_zero(self):
        scores = zero_scores()
        assert len(scores) == 4
        for name, entry in scores.items():
            assert entry["score"] == 0.0
            assert "failed" in entry["reason"].lower()
