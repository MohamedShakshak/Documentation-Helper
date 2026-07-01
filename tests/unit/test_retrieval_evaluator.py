from unittest.mock import MagicMock, patch

from doc_helper.config.settings import RetrievalSettings
from doc_helper.evaluation.retrieval_evaluator import RetrievalEvaluator


def _make_evaluator():
    with patch("doc_helper.evaluation.retrieval_evaluator.create_vector_store"):
        evaluator = RetrievalEvaluator.__new__(RetrievalEvaluator)
        evaluator._settings = MagicMock()
        evaluator._store = MagicMock()
        return evaluator


class TestBuildConfigGrid:
    def test_generates_12_configs_with_flashrank(self):
        evaluator = _make_evaluator()
        with patch("doc_helper.evaluation.retrieval_evaluator.flashrank", create=True):
            grid = evaluator._build_config_grid()
        assert len(grid) == 12

        search_types = {c.search_type for c in grid}
        assert search_types == {"similarity", "mmr"}

        k_values = sorted({c.search_k for c in grid})
        assert k_values == [4, 8, 16]

        reranker_values = {c.reranker_enabled for c in grid}
        assert reranker_values == {True, False}

    def test_prunes_reranker_when_flashrank_missing(self):
        evaluator = _make_evaluator()
        with patch.dict("sys.modules", {"flashrank": None}):
            import importlib

            importlib.reload(
                __import__(
                    "doc_helper.evaluation.retrieval_evaluator",
                    fromlist=["retrieval_evaluator"],
                )
            )
            grid = evaluator._build_config_grid()
        assert len(grid) == 6
        assert all(not c.reranker_enabled for c in grid)


class TestRunConfig:
    def test_extracts_source_url_from_retrieved_docs(self):
        evaluator = _make_evaluator()

        config = RetrievalSettings(search_type="similarity", search_k=4)
        dataset = [
            {
                "question": "What is LangChain?",
                "relevant_urls": ["https://python.langchain.com/docs/concepts/"],
                "difficulty": "simple",
            }
        ]

        mock_doc1 = MagicMock()
        mock_doc1.metadata = {"source_url": "https://python.langchain.com/docs/concepts/"}
        mock_doc2 = MagicMock()
        mock_doc2.metadata = {"source_url": "https://python.langchain.com/docs/other/"}

        with patch("doc_helper.evaluation.retrieval_evaluator.Retriever") as mock_retriever_cls:
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = [mock_doc1, mock_doc2]
            mock_retriever_cls.return_value = mock_retriever

            result = evaluator._run_config(config, dataset)

        metrics = result["metrics"]
        assert metrics["hit_rate"] == 1.0
        assert metrics["mrr"] == 1.0
        assert metrics["precision_at_k"] == 0.5
        assert metrics["recall_at_k"] == 1.0

    def test_no_hit_when_retrieved_urls_dont_match(self):
        evaluator = _make_evaluator()

        config = RetrievalSettings(search_type="similarity", search_k=4)
        dataset = [
            {
                "question": "What is LangChain?",
                "relevant_urls": ["https://python.langchain.com/docs/concepts/"],
                "difficulty": "simple",
            }
        ]

        mock_doc = MagicMock()
        mock_doc.metadata = {"source_url": "https://example.com/other/"}

        with patch("doc_helper.evaluation.retrieval_evaluator.Retriever") as mock_retriever_cls:
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = [mock_doc]
            mock_retriever_cls.return_value = mock_retriever

            result = evaluator._run_config(config, dataset)

        assert result["metrics"]["hit_rate"] == 0.0

    def test_normalization_applied_before_comparison(self):
        evaluator = _make_evaluator()

        config = RetrievalSettings(search_type="similarity", search_k=4)
        dataset = [
            {
                "question": "What is LangChain?",
                "relevant_urls": ["https://python.langchain.com/docs/concepts"],
                "difficulty": "simple",
            }
        ]

        mock_doc = MagicMock()
        mock_doc.metadata = {"source_url": "https://python.langchain.com/docs/concepts/"}

        with patch("doc_helper.evaluation.retrieval_evaluator.Retriever") as mock_retriever_cls:
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = [mock_doc]
            mock_retriever_cls.return_value = mock_retriever

            result = evaluator._run_config(config, dataset)

        assert result["metrics"]["hit_rate"] == 1.0


class TestAggregation:
    def test_aggregates_mean_across_queries(self):
        scores = [
            {"hit_rate": 1.0, "mrr": 0.5, "precision_at_k": 0.5, "recall_at_k": 1.0},
            {"hit_rate": 0.0, "mrr": 0.0, "precision_at_k": 0.0, "recall_at_k": 0.0},
        ]
        result = RetrievalEvaluator._aggregate(scores)
        assert result["hit_rate"] == 0.5
        assert result["mrr"] == 0.25
        assert result["precision_at_k"] == 0.25
        assert result["recall_at_k"] == 0.5

    def test_aggregates_per_difficulty(self):
        evaluator = _make_evaluator()

        config = RetrievalSettings(search_type="similarity", search_k=4)
        dataset = [
            {
                "question": "Q1",
                "relevant_urls": ["https://example.com/a"],
                "difficulty": "simple",
            },
            {
                "question": "Q2",
                "relevant_urls": ["https://example.com/b"],
                "difficulty": "simple",
            },
            {
                "question": "Q3",
                "relevant_urls": ["https://example.com/c"],
                "difficulty": "multi_hop",
            },
        ]

        doc_a = MagicMock()
        doc_a.metadata = {"source_url": "https://example.com/a"}
        doc_c = MagicMock()
        doc_c.metadata = {"source_url": "https://example.com/c"}

        with patch("doc_helper.evaluation.retrieval_evaluator.Retriever") as mock_retriever_cls:
            mock_retriever = MagicMock()

            def retrieve(query):
                if query == "Q1":
                    return [doc_a]
                elif query == "Q3":
                    return [doc_c]
                return []

            mock_retriever.retrieve.side_effect = retrieve
            mock_retriever_cls.return_value = mock_retriever

            result = evaluator._run_config(config, dataset)

        by_diff = result["by_difficulty"]
        assert by_diff["simple"]["hit_rate"] == 0.5
        assert by_diff["multi_hop"]["hit_rate"] == 1.0


class TestEvaluate:
    def test_empty_dataset_returns_empty_report(self):
        evaluator = _make_evaluator()
        report = evaluator.evaluate(dataset=[])
        assert report["total_queries"] == 0
        assert report["configs"] == []
        assert report["best_config"] is None

    def test_best_config_selection_by_hit_rate(self):
        evaluator = _make_evaluator()

        dataset = [
            {
                "question": "Q1",
                "relevant_urls": ["https://example.com/a"],
                "difficulty": "simple",
            }
        ]

        mock_doc = MagicMock()
        mock_doc.metadata = {"source_url": "https://example.com/a"}

        with patch("doc_helper.evaluation.retrieval_evaluator.Retriever") as mock_retriever_cls:
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = [mock_doc]
            mock_retriever_cls.return_value = mock_retriever

            with patch.object(evaluator, "_build_config_grid") as mock_grid:
                mock_grid.return_value = [
                    RetrievalSettings(search_type="similarity", search_k=4, reranker_enabled=False),
                    RetrievalSettings(search_type="mmr", search_k=4, reranker_enabled=False),
                ]

                report = evaluator.evaluate(dataset=dataset)

        assert report["best_config"] is not None
        assert report["best_config"]["metric"] == "hit_rate"
        assert report["best_config"]["value"] == 1.0
        assert report["best_config"]["search_type"] in ("similarity", "mmr")
