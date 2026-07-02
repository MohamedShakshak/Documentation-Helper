from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from doc_helper.evaluation.generator_metrics import (
    JudgeResult,
    _parse_score,
    answer_correctness,
    answer_relevancy,
    context_utilization,
    faithfulness,
    zero_scores,
)


def _mock_judge(score: float, reason: str) -> MagicMock:
    judge = MagicMock()
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=JudgeResult(score=score, reason=reason))
    judge.with_structured_output = MagicMock(return_value=structured)
    return judge


def _mock_judge_fallback(response_text: str) -> MagicMock:
    judge = MagicMock()
    judge.with_structured_output = MagicMock(side_effect=Exception("Not supported"))
    judge.ainvoke = AsyncMock(return_value=MagicMock(content=response_text))
    return judge


class TestFaithfulness:
    @pytest.mark.asyncio
    async def test_high_score(self):
        judge = _mock_judge(0.95, "All claims supported by contexts")
        result = await faithfulness(judge, "What is X?", "X is a tool.", ["X is a tool for apps."])
        assert result["score"] == 0.95
        assert "supported" in result["reason"]

    @pytest.mark.asyncio
    async def test_low_score(self):
        judge = _mock_judge(0.25, "1 of 4 claims supported")
        result = await faithfulness(judge, "Q", "A", ["C"])
        assert result["score"] == 0.25

    @pytest.mark.asyncio
    async def test_clamps_score_above_1(self):
        judge = _mock_judge(1.0, "all")
        result = await faithfulness(judge, "Q", "A", ["C"])
        assert result["score"] == 1.0

    @pytest.mark.asyncio
    async def test_clamps_score_below_0(self):
        judge = _mock_judge(0.0, "none")
        result = await faithfulness(judge, "Q", "A", ["C"])
        assert result["score"] == 0.0

    @pytest.mark.asyncio
    async def test_parse_failure_returns_zero(self):
        judge = _mock_judge_fallback("not valid json at all")
        result = await faithfulness(judge, "Q", "A", ["C"])
        assert result["score"] == 0.0
        assert "Failed to parse" in result["reason"]

    @pytest.mark.asyncio
    async def test_fallback_with_valid_json(self):
        judge = _mock_judge_fallback('{"score": 0.8, "reason": "Mostly faithful"}')
        result = await faithfulness(judge, "Q", "A", ["C"])
        assert result["score"] == 0.8
        assert "faithful" in result["reason"].lower()


class TestAnswerRelevancy:
    @pytest.mark.asyncio
    async def test_relevant_answer(self):
        judge = _mock_judge(0.9, "Directly addresses the question")
        result = await answer_relevancy(judge, "What is X?", "X is a framework.")
        assert result["score"] == 0.9

    @pytest.mark.asyncio
    async def test_irrelevant_answer(self):
        judge = _mock_judge(0.1, "Off-topic")
        result = await answer_relevancy(judge, "What is X?", "Bananas are yellow.")
        assert result["score"] == 0.1


class TestAnswerCorrectness:
    @pytest.mark.asyncio
    async def test_all_key_facts_present(self):
        judge = _mock_judge(1.0, "All key facts found")
        result = await answer_correctness(
            judge, "Q", "A has all facts", "Reference", ["fact1", "fact2"]
        )
        assert result["score"] == 1.0

    @pytest.mark.asyncio
    async def test_missing_key_facts(self):
        judge = _mock_judge(0.5, "Missing: fact2")
        result = await answer_correctness(
            judge, "Q", "A has fact1 only", "Reference", ["fact1", "fact2"]
        )
        assert result["score"] == 0.5
        assert "fact2" in result["reason"]


class TestContextUtilization:
    @pytest.mark.asyncio
    async def test_fully_derived_from_contexts(self):
        judge = _mock_judge(0.95, "Uses contexts heavily")
        result = await context_utilization(judge, "Answer based on context", ["Context with info"])
        assert result["score"] == 0.95

    @pytest.mark.asyncio
    async def test_ignores_contexts(self):
        judge = _mock_judge(0.1, "Answers from training data")
        result = await context_utilization(judge, "Answer from training", ["Context"])
        assert result["score"] == 0.1


class TestJudgeResult:
    def test_valid_score(self):
        r = JudgeResult(score=0.85, reason="Good")
        assert r.score == 0.85
        assert r.reason == "Good"

    def test_clamps_score_above_1(self):
        with pytest.raises(ValidationError):
            JudgeResult(score=1.5, reason="over")

    def test_clamps_score_below_0(self):
        with pytest.raises(ValidationError):
            JudgeResult(score=-0.5, reason="under")


class TestParseScoreFallback:
    def test_valid_json(self):
        result = _parse_score('{"score": 0.7, "reason": "partial"}')
        assert result["score"] == 0.7
        assert result["reason"] == "partial"

    def test_json_embedded_in_text(self):
        result = _parse_score('Some text {"score": 0.6, "reason": "embedded"} more text')
        assert result["score"] == 0.6

    def test_invalid_returns_zero(self):
        result = _parse_score("totally not json")
        assert result["score"] == 0.0
        assert "Failed to parse" in result["reason"]


class TestZeroScores:
    def test_all_metrics_zero(self):
        scores = zero_scores()
        assert len(scores) == 4
        for name, entry in scores.items():
            assert entry["score"] == 0.0
            assert "failed" in entry["reason"].lower()
