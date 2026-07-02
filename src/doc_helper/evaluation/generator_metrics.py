from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

METRIC_NAMES = ["faithfulness", "answer_relevancy", "answer_correctness", "context_utilization"]


class JudgeResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reason: str = ""


def _parse_score(raw: str) -> dict[str, float | str]:
    import json
    import re

    try:
        parsed = json.loads(raw)
        score = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
        reason = str(parsed.get("reason", ""))
        return {"score": score, "reason": reason}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    match = re.search(r'\{[^{}]*"score"\s*:\s*[\d.]+[^{}]*\}', raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            score = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
            reason = str(parsed.get("reason", ""))
            return {"score": score, "reason": reason}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    return {"score": 0.0, "reason": f"Failed to parse judge response: {raw[:200]}"}


def _result_to_dict(result: object) -> dict[str, float | str]:
    if isinstance(result, JudgeResult):
        return {"score": result.score, "reason": result.reason}
    if isinstance(result, dict):
        score = max(0.0, min(1.0, float(result.get("score", 0.0))))
        return {"score": score, "reason": str(result.get("reason", ""))}
    return _parse_score(str(result))


async def _invoke_judge(judge: BaseChatModel, prompt: str) -> dict[str, float | str]:
    try:
        structured = judge.with_structured_output(JudgeResult)
        result = await structured.ainvoke(prompt)
        return _result_to_dict(result)
    except Exception:
        pass

    response = await judge.ainvoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        content = "\n".join(text_parts) if text_parts else str(content)

    return _parse_score(str(content))


async def faithfulness(
    judge: BaseChatModel,
    question: str,
    answer: str,
    contexts: list[str],
) -> dict[str, float | str]:
    contexts_text = "\n---\n".join(contexts) if contexts else "(no contexts retrieved)"

    prompt = f"""You are an expert evaluator for RAG systems. Score the FAITHFULNESS of the answer.

Faithfulness measures whether the answer is grounded in the retrieved contexts (no hallucination).

Question: {question}

Retrieved Contexts:
{contexts_text}

Answer: {answer}

Instructions:
1. Identify each factual claim in the answer.
2. For each claim, check if it is supported by the retrieved contexts.
3. Score = supported_claims / total_claims (0.0 to 1.0).
4. If the answer contains no factual claims, score 1.0.

Respond ONLY with JSON: {{"score": <float>, "reason": "<brief explanation>"}}"""

    return await _invoke_judge(judge, prompt)


async def answer_relevancy(
    judge: BaseChatModel,
    question: str,
    answer: str,
) -> dict[str, float | str]:
    prompt = f"""You are an expert evaluator for RAG systems. Score the ANSWER RELEVANCY.

Answer Relevancy measures how directly the answer addresses the question asked.

Question: {question}

Answer: {answer}

Instructions:
1. Does the answer directly address what was asked?
2. Penalize tangential, evasive, or incomplete answers.
3. Score 1.0 if perfectly relevant, 0.0 if completely off-topic.

Respond ONLY with JSON: {{"score": <float>, "reason": "<brief explanation>"}}"""

    return await _invoke_judge(judge, prompt)


async def answer_correctness(
    judge: BaseChatModel,
    question: str,
    answer: str,
    reference_answer: str,
    key_facts: list[str],
) -> dict[str, float | str]:
    facts_text = "\n".join(f"- {f}" for f in key_facts)

    prompt = f"""You are an expert evaluator for RAG systems. Score the ANSWER CORRECTNESS.

Answer Correctness measures how the generated answer compares to the reference answer and key facts.

Question: {question}

Reference Answer: {reference_answer}

Key Facts (a correct answer must contain):
{facts_text}

Generated Answer: {answer}

Instructions:
1. Check if each key fact is present and correctly stated in the generated answer.
2. Score = key_facts_present / total_key_facts, adjusted for factual consistency with reference.
3. Penalize answers that contradict the reference answer.
4. Do not penalize differences in wording or style, only factual differences.

Respond ONLY with JSON: {{"score": <float>, "reason": "<brief explanation>"}}"""

    return await _invoke_judge(judge, prompt)


async def context_utilization(
    judge: BaseChatModel,
    answer: str,
    contexts: list[str],
) -> dict[str, float | str]:
    contexts_text = "\n---\n".join(contexts) if contexts else "(no contexts retrieved)"

    prompt = f"""You are an expert evaluator for RAG systems. Score the CONTEXT UTILIZATION.

Context Utilization measures whether the answer uses information from
the retrieved contexts, or if it ignores them and answers from training data.

Retrieved Contexts:
{contexts_text}

Answer: {answer}

Instructions:
1. Does the answer draw information from the retrieved contexts?
2. Does it ignore the contexts and answer from training data instead?
3. Score 1.0 if the answer is fully derived from contexts, 0.0 if contexts are completely ignored.
4. If no contexts were retrieved, score 0.0.

Respond ONLY with JSON: {{"score": <float>, "reason": "<brief explanation>"}}"""

    return await _invoke_judge(judge, prompt)


def zero_scores() -> dict[str, dict[str, float | str]]:
    zero = 0.0
    return {
        name: {"score": zero, "reason": "Query failed — agent did not produce an answer"}
        for name in METRIC_NAMES
    }
