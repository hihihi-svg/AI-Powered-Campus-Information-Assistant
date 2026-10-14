import json
from pathlib import Path
from urllib.parse import urlparse

import mlflow

from src.evaluation import (
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_DATASET_PATH,
    compare_retrieval_conditions,
    create_degraded_dataset,
    evaluate_benchmark_queries,
    evaluate_degradation_thresholds,
    load_benchmark_queries,
    log_mlflow_benchmark,
    summarize_query_results,
)


def test_summarize_query_results_counts_hits():
    query_results = [
        {"query_id": "q1", "has_result": True, "max_score": 0.25},
        {"query_id": "q2", "has_result": False, "max_score": 0.0},
        {"query_id": "q3", "has_result": True, "max_score": 0.18},
    ]

    summary = summarize_query_results(query_results)

    assert summary["queries_evaluated"] == 3
    assert summary["queries_with_results"] == 2
    assert summary["queries_without_results"] == 1
    assert summary["relevance_rate"] == 0.6667
    assert summary["benchmark_has_relevance_labels"] is False


def test_evaluate_benchmark_queries_returns_metrics_for_real_dataset():
    queries = load_benchmark_queries(DEFAULT_BENCHMARK_PATH)[:5]
    summary = evaluate_benchmark_queries(queries=queries, dataset_path=DEFAULT_DATASET_PATH)

    assert summary["metrics"]["queries_evaluated"] == 5
    assert 0.0 <= summary["metrics"]["relevance_rate"] <= 1.0
    assert summary["metrics"]["benchmark_has_relevance_labels"] is False
    assert summary["config"]["evaluation_mode"] == "retrieval_hit_proxy"
    assert len(summary["queries"]) == 5


def test_log_mlflow_benchmark_uses_local_tracking_uri(tmp_path):
    local_tracking = (tmp_path / "mlruns").as_uri()
    benchmark_queries = load_benchmark_queries(DEFAULT_BENCHMARK_PATH)[:2]
    evaluation_summary = evaluate_benchmark_queries(
        queries=benchmark_queries,
        dataset_path=DEFAULT_DATASET_PATH,
        top_k=2,
        score_threshold=0.05,
    )

    run_info = log_mlflow_benchmark(
        evaluation_summary=evaluation_summary,
        experiment_name="test-rvce-retrieval-benchmark",
        run_name="test-run-local-tracking",
        tracking_uri=local_tracking,
    )

    assert run_info["tracking_uri"] == local_tracking
    assert run_info["run_id"]
    assert (tmp_path / "mlruns").exists()

    client = mlflow.tracking.MlflowClient(tracking_uri=local_tracking)
    experiment = client.get_experiment_by_name("test-rvce-retrieval-benchmark")
    assert experiment is not None
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) >= 1
    assert runs[0].data.tags.get("benchmark_has_relevance_labels") == "false"

    artifact_path = urlparse(runs[0].info.artifact_uri).path.lstrip("/")
    artifact_dir = Path(artifact_path)
    assert artifact_dir.exists()


def test_create_degraded_dataset_reduces_corpus_without_touching_source(tmp_path):
    degraded_path = tmp_path / "degraded.jsonl"
    output_path = create_degraded_dataset(
        source_path=DEFAULT_DATASET_PATH,
        output_path=degraded_path,
        keep_fraction=0.5,
    )

    assert output_path == degraded_path
    assert degraded_path.exists()
    with degraded_path.open("r", encoding="utf-8") as handle:
        rows = [line for line in handle if line.strip()]
    assert 1 <= len(rows) < 247
    assert DEFAULT_DATASET_PATH.exists()


def test_evaluate_degradation_thresholds_flags_warning_and_retraining():
    baseline = {"relevance_rate": 1.0, "average_best_score": 0.20}
    degraded = {"relevance_rate": 0.80, "average_best_score": 0.08}

    result = evaluate_degradation_thresholds(
        baseline_metrics=baseline,
        degraded_metrics=degraded,
        warning_drop=0.10,
        retraining_drop=0.25,
    )

    assert result["warning"] is True
    assert result["automatic_retraining"] is False
    assert result["decision"] == "warning"
    assert result["relevance_drop"] == 0.2
    assert result["average_best_score_drop"] == 0.12


def test_compare_retrieval_conditions_returns_simulated_degradation(tmp_path):
    degraded_path = tmp_path / "degraded.jsonl"
    create_degraded_dataset(
        source_path=DEFAULT_DATASET_PATH,
        output_path=degraded_path,
        keep_fraction=0.7,
    )

    comparison = compare_retrieval_conditions(
        queries=load_benchmark_queries(DEFAULT_BENCHMARK_PATH)[:5],
        baseline_dataset_path=DEFAULT_DATASET_PATH,
        degraded_dataset_path=degraded_path,
        top_k=3,
        score_threshold=0.05,
    )

    assert comparison["baseline"]["metrics"]["queries_evaluated"] == 5
    assert comparison["degraded"]["metrics"]["queries_evaluated"] == 5
    assert "relevance_rate_drop" in comparison["metric_differences"]
    assert comparison["simulated_intervention"]["type"] == "reduced_corpus_subset"
    assert "The benchmark contains no gold relevance labels" in comparison["limitations"][0]


def test_evaluate_benchmark_queries_supports_semantic_and_hybrid():
    from src.evaluation import evaluate_benchmark_queries

    queries = [
        {"query_id": "q1", "query_text": "What is the procedure for B.E. admission?", "intent": "admissions", "split_group": "test"},
        {"query_id": "q2", "query_text": "When is the fee payment deadline?", "intent": "fees_finance", "split_group": "test"},
    ]

    semantic_res = evaluate_benchmark_queries(
        queries=queries,
        dataset_path=DEFAULT_DATASET_PATH,
        retriever_type="semantic",
        top_k=2,
    )
    assert semantic_res["metrics"]["queries_evaluated"] == 2
    assert "retrieval_coverage" in semantic_res["metrics"]
    assert semantic_res["config"]["retriever_type"] == "semantic"

    hybrid_res = evaluate_benchmark_queries(
        queries=queries,
        dataset_path=DEFAULT_DATASET_PATH,
        retriever_type="hybrid",
        top_k=2,
    )
    assert hybrid_res["metrics"]["queries_evaluated"] == 2
    assert hybrid_res["config"]["retriever_type"] == "hybrid"


def test_evaluate_labelled_benchmark_calculates_ranking_metrics():
    from src.evaluation import DEFAULT_LABELLED_BENCHMARK_PATH, evaluate_labelled_benchmark

    summary = evaluate_labelled_benchmark(
        benchmark_path=DEFAULT_LABELLED_BENCHMARK_PATH,
        dataset_path=DEFAULT_DATASET_PATH,
        retriever_type="tfidf",
        top_k=3,
    )

    metrics = summary["metrics"]
    assert metrics["queries_evaluated"] == 8
    assert "recall_at_k" in metrics
    assert "precision_at_k" in metrics
    assert "mrr" in metrics
    assert "ndcg_at_k" in metrics
    assert 0.0 <= metrics["recall_at_k"] <= 1.0
    assert 0.0 <= metrics["mrr"] <= 1.0
    assert 0.0 <= metrics["ndcg_at_k"] <= 1.0
    assert metrics["benchmark_has_relevance_labels"] is True


def test_log_mlflow_model_artifact_and_reload_smoke_test(tmp_path):
    from src.evaluation import evaluate_benchmark_queries, log_mlflow_benchmark

    local_tracking = (tmp_path / "mlruns").as_uri()
    queries = [
        {"query_id": "q1", "query_text": "admission circulars and procedures", "intent": "admissions", "split_group": "test"}
    ]
    summary = evaluate_benchmark_queries(
        queries=queries,
        dataset_path=DEFAULT_DATASET_PATH,
        retriever_type="semantic",
        top_k=2,
    )

    run_info = log_mlflow_benchmark(
        evaluation_summary=summary,
        experiment_name="test-model-logging-exp",
        run_name="semantic-test-run",
        tracking_uri=local_tracking,
        log_model=True,
    )

    assert run_info["run_id"]
    assert run_info["model_uri"] is not None
    assert run_info["model_reload_verified"] is True

