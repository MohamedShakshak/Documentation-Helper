from langchain_core.language_models import BaseChatModel

METRIC_NAMES = ["faithfulness", "answer_relevancy", "answer_correctness", "context_utilization"]


def _parse_score(raw: str) -> dict[str, float | str]:
    import json

    try:
        parsed = json.loads(raw)
        score = float(parsed.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        reason = str(parsed.get("reason", ""))
        return {"score": score, "reason": reason}
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"score": 0.0, "reason": f"Failed to parse judge response: {raw[:200]}"}


async def _invoke_judge(judge: BaseChatModel, prompt: str) -> str:
    response = await judge.ainvoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


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

    raw = await _invoke_judge(judge, prompt)
    return _parse_score(raw)


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

    raw = await _invoke_judge(judge, prompt)
    return _parse_score(raw)


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

    raw = await _invoke_judge(judge, prompt)
    return _parse_score(raw)


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

    raw = await _invoke_judge(judge, prompt)
    return _parse_score(raw)


def zero_scores() -> dict[str, dict[str, float | str]]:
    zero = 0.0
    return {
        name: {"score": zero, "reason": "Query failed — agent did not produce an answer"}
        for name in METRIC_NAMES
    }
