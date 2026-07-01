from doc_helper.evaluation.retrieval_metrics import (
    hit_rate,
    normalize_url,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestNormalizeUrl:
    def test_strips_trailing_slash(self):
        assert normalize_url("https://python.langchain.com/docs/") == (
            "https://python.langchain.com/docs"
        )

    def test_no_trailing_slash_unchanged(self):
        assert normalize_url("https://python.langchain.com/docs") == (
            "https://python.langchain.com/docs"
        )

    def test_strips_fragment(self):
        assert normalize_url("https://python.langchain.com/docs#agents") == (
            "https://python.langchain.com/docs"
        )

    def test_preserves_query_params(self):
        assert normalize_url("https://python.langchain.com/docs?from=x") == (
            "https://python.langchain.com/docs?from=x"
        )

    def test_lowercases_domain(self):
        assert normalize_url("HTTPS://Python.LangChain.COM/Docs") == (
            "https://python.langchain.com/Docs"
        )

    def test_strips_trailing_slash_with_query(self):
        assert normalize_url("https://python.langchain.com/docs/?from=x") == (
            "https://python.langchain.com/docs?from=x"
        )


class TestHitRate:
    def test_relevant_at_rank_1(self):
        assert hit_rate(["A", "B", "C"], ["A"], k=3) == 1.0

    def test_relevant_at_rank_3_within_k(self):
        assert hit_rate(["X", "Y", "A"], ["A"], k=3) == 1.0

    def test_relevant_beyond_k(self):
        assert hit_rate(["X", "B", "A"], ["A"], k=2) == 0.0

    def test_no_relevant_in_retrieved(self):
        assert hit_rate(["X", "Y", "Z"], ["A"], k=3) == 0.0

    def test_empty_retrieved(self):
        assert hit_rate([], ["A"], k=3) == 0.0

    def test_empty_relevant(self):
        assert hit_rate(["A", "B"], [], k=2) == 0.0

    def test_url_normalization_match(self):
        url_a = "https://python.langchain.com/docs/"
        url_b = "https://python.langchain.com/docs"
        assert hit_rate([url_a], [url_b], k=1) == 1.0


class TestReciprocalRank:
    def test_relevant_at_rank_1(self):
        assert reciprocal_rank(["A", "B", "C"], ["A"], k=3) == 1.0

    def test_relevant_at_rank_4(self):
        assert reciprocal_rank(["X", "Y", "Z", "A"], ["A"], k=4) == 0.25

    def test_relevant_at_rank_2(self):
        assert reciprocal_rank(["X", "A", "Z"], ["A"], k=3) == 0.5

    def test_no_hit(self):
        assert reciprocal_rank(["X", "Y", "Z"], ["A"], k=3) == 0.0

    def test_empty_retrieved(self):
        assert reciprocal_rank([], ["A"], k=3) == 0.0

    def test_empty_relevant(self):
        assert reciprocal_rank(["A", "B"], [], k=2) == 0.0

    def test_k_limits_search(self):
        assert reciprocal_rank(["X", "Y", "Z", "A"], ["A"], k=2) == 0.0


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["A", "B"], ["A", "B"], k=2) == 1.0

    def test_half_relevant(self):
        assert precision_at_k(["A", "X"], ["A", "B"], k=2) == 0.5

    def test_k_greater_than_retrieved(self):
        assert precision_at_k(["A"], ["A", "B"], k=4) == 1.0

    def test_empty_retrieved(self):
        assert precision_at_k([], ["A"], k=3) == 0.0

    def test_empty_relevant(self):
        assert precision_at_k(["A", "B"], [], k=2) == 0.0

    def test_none_relevant(self):
        assert precision_at_k(["X", "Y"], ["A", "B"], k=2) == 0.0

    def test_multiple_relevant_in_top_k(self):
        assert precision_at_k(["A", "B", "X"], ["A", "B"], k=3) == 2 / 3


class TestRecallAtK:
    def test_all_relevant_retrieved(self):
        assert recall_at_k(["A", "B"], ["A", "B"], k=2) == 1.0

    def test_partial_relevant_retrieved(self):
        assert recall_at_k(["A"], ["A", "B"], k=2) == 0.5

    def test_empty_relevant(self):
        assert recall_at_k(["A", "B"], [], k=2) == 0.0

    def test_empty_retrieved(self):
        assert recall_at_k([], ["A", "B"], k=3) == 0.0

    def test_all_relevant_in_top_k_with_extra(self):
        assert recall_at_k(["A", "X", "B"], ["A", "B"], k=3) == 1.0

    def test_relevant_beyond_k(self):
        assert recall_at_k(["A", "X"], ["A", "B"], k=2) == 0.5

    def test_duplicate_relevant_urls(self):
        assert recall_at_k(["A", "B"], ["A", "A", "B"], k=2) == 1.0
