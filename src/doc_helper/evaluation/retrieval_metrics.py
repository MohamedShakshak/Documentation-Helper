from urllib.parse import urlparse


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}{query}"


def _normalize_list(urls: list[str]) -> list[str]:
    return [normalize_url(u) for u in urls]


def hit_rate(retrieved_urls: list[str], relevant_urls: list[str], k: int) -> float:
    if not retrieved_urls or not relevant_urls:
        return 0.0
    top_k = retrieved_urls[:k]
    norm_top_k = _normalize_list(top_k)
    norm_relevant = set(_normalize_list(relevant_urls))
    for url in norm_top_k:
        if url in norm_relevant:
            return 1.0
    return 0.0


def reciprocal_rank(retrieved_urls: list[str], relevant_urls: list[str], k: int) -> float:
    if not retrieved_urls or not relevant_urls:
        return 0.0
    top_k = retrieved_urls[:k]
    norm_top_k = _normalize_list(top_k)
    norm_relevant = set(_normalize_list(relevant_urls))
    for rank, url in enumerate(norm_top_k, start=1):
        if url in norm_relevant:
            return 1.0 / rank
    return 0.0


def precision_at_k(retrieved_urls: list[str], relevant_urls: list[str], k: int) -> float:
    if not retrieved_urls:
        return 0.0
    top_k = retrieved_urls[:k]
    norm_top_k = _normalize_list(top_k)
    norm_relevant = set(_normalize_list(relevant_urls))
    denom = min(k, len(norm_top_k))
    if denom == 0:
        return 0.0
    relevant_count = sum(1 for url in norm_top_k if url in norm_relevant)
    return relevant_count / denom


def recall_at_k(retrieved_urls: list[str], relevant_urls: list[str], k: int) -> float:
    if not relevant_urls:
        return 0.0
    top_k = retrieved_urls[:k]
    norm_top_k = _normalize_list(top_k)
    norm_relevant = set(_normalize_list(relevant_urls))
    found = sum(1 for url in norm_relevant if url in norm_top_k)
    return found / len(norm_relevant)
