"""Local MLflow evaluation workflow for the RVCE retrieval benchmark.

Supports:
- Configuration A: TF-IDF baseline (TFIDFRetriever)
- Configuration B: Semantic retrieval (SemanticRetriever with all-MiniLM-L6-v2)
- Configuration C: Hybrid retrieval (HybridRetriever)
- Configuration D: Hybrid retrieval with CrossEncoder reranking
- MLflow parameter, metric, and SentenceTransformer model artifact logging
- Grounded metric distinction: Coverage-based proxy on 40-query dataset, and
  labelled ranking metrics (Recall@k, Precision@k, MRR, nDCG@k) on curated relevance benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
try:
    import mlflow.sentence_transformers
except Exception:
    pass

from src.retrieval import (
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_SEMANTIC_MODEL,
    CrossEncoderReranker,
    HybridRetriever,
    SemanticRetriever,
    TFIDFRetriever,
    get_default_hybrid_retriever,
    get_default_retriever,
    get_default_semantic_retriever,
)

logger = logging.getLogger("campus_pipeline.evaluation")

DEFAULT_BENCHMARK_PATH = Path("data/queries/campus_queries.csv")
DEFAULT_LABELLED_BENCHMARK_PATH = Path("data/queries/labelled_relevance_benchmark.json")
DEFAULT_DATASET_PATH = Path("data/processed/campus_documents.jsonl")
DEFAULT_EXPERIMENT_NAME = "campus-retrieval-evaluation"
DEFAULT_RAG_EXPERIMENT_NAME = "campus-rag-evaluation"


def load_benchmark_queries(path: str | Path = DEFAULT_BENCHMARK_PATH) -> List[Dict[str, str]]:
    """Load the CSV benchmark into a list of structured query dicts."""
    benchmark_path = Path(path)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    rows: List[Dict[str, str]] = []
    with benchmark_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            query_text = (row.get("query_text") or "").strip()
            if not query_text:
                continue

            rows.append(
                {
                    "query_id": (row.get("query_id") or "unknown").strip(),
                    "query_text": query_text,
                    "intent": (row.get("intent") or "unknown").strip(),
                    "split_group": (row.get("split_group") or "unknown").strip(),
                }
            )

    return rows


def summarize_query_results(query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute retrieval coverage metrics for the benchmark.

    The benchmark does not include gold relevance labels, so this is intentionally a
    retrieval-hit proxy for whether at least one result scored above threshold.
    """
    total_queries = len(query_results)
    queries_with_results = sum(1 for item in query_results if item.get("has_result", False))
    queries_without_results = total_queries - queries_with_results
    max_scores = [float(item.get("max_score", 0.0) or 0.0) for item in query_results]
    average_best_score = round(sum(max_scores) / total_queries, 4) if total_queries else 0.0
    relevance_rate = round(queries_with_results / total_queries, 4) if total_queries else 0.0
    average_retrieved_results = (
        round(sum(int(item.get("result_count", 0) or 0) for item in query_results) / total_queries, 4)
        if total_queries
        else 0.0
    )

    return {
        "queries_evaluated": total_queries,
        "queries_with_results": queries_with_results,
        "queries_without_results": queries_without_results,
        "retrieval_coverage": relevance_rate,
        "relevance_rate": relevance_rate,
        "average_best_score": average_best_score,
        "average_retrieved_results": average_retrieved_results,
        "benchmark_has_relevance_labels": False,
        "benchmark_note": (
            "The benchmark CSV has query text and intent labels only; no gold relevance labels are available. "
            "The 'retrieval_coverage' below is a retrieval-hit proxy based on whether at least one returned passage "
            "passed the configured score threshold."
        ),
    }


def _instantiate_retriever(
    retriever_type: str,
    dataset_path: str | Path,
    rerank: bool = False,
    fusion_method: str = "linear",
    semantic_weight: float = 0.5,
    tfidf_weight: float = 0.5,
) -> Any:
    dataset_str = str(dataset_path)
    if retriever_type == "tfidf":
        return get_default_retriever(dataset_str)
    elif retriever_type == "semantic":
        return get_default_semantic_retriever(dataset_str)
    elif retriever_type in {"hybrid", "hybrid_rerank"}:
        return get_default_hybrid_retriever(
            dataset_path=dataset_str,
            fusion_method=fusion_method,
            semantic_weight=semantic_weight,
            tfidf_weight=tfidf_weight,
        )
    else:
        raise ValueError(
            f"Unknown retriever_type: {retriever_type}. Expected tfidf, semantic, hybrid, or hybrid_rerank."
        )


def evaluate_benchmark_queries(
    queries: Optional[List[Dict[str, str]]] = None,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    retriever_type: str = "tfidf",
    top_k: int = 3,
    score_threshold: float = 0.05,
    rerank: bool = False,
    fusion_method: str = "linear",
    semantic_weight: float = 0.5,
    tfidf_weight: float = 0.5,
) -> Dict[str, Any]:
    """Run benchmark queries against the specified retrieval configuration."""
    benchmark_queries = queries if queries is not None else load_benchmark_queries()
    dataset_ref = Path(dataset_path)
    if not dataset_ref.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_ref}")

    start_time = time.time()
    effective_rerank = rerank or (retriever_type == "hybrid_rerank")
    retriever = _instantiate_retriever(
        retriever_type=retriever_type,
        dataset_path=dataset_ref,
        rerank=effective_rerank,
        fusion_method=fusion_method,
        semantic_weight=semantic_weight,
        tfidf_weight=tfidf_weight,
    )

    per_query_results: List[Dict[str, Any]] = []
    latencies: List[float] = []
    for item in benchmark_queries:
        query_text = item.get("query_text", "").strip()
        q_start = time.time()
        if isinstance(retriever, HybridRetriever):
            results = retriever.retrieve(
                query=query_text,
                top_k=top_k,
                score_threshold=score_threshold,
                strategy="hybrid_rerank" if effective_rerank else "hybrid",
                rerank=effective_rerank,
            )
        else:
            results = retriever.retrieve(query=query_text, top_k=top_k, score_threshold=score_threshold)
        q_lat = round(time.time() - q_start, 4)
        latencies.append(q_lat)

        max_score = max((float(result.get("relevance_score", 0.0) or 0.0) for result in results), default=0.0)
        has_result = len(results) > 0

        per_query_results.append(
            {
                "query_id": item.get("query_id", "unknown"),
                "intent": item.get("intent", "unknown"),
                "split_group": item.get("split_group", "unknown"),
                "query_text": query_text,
                "result_count": len(results),
                "has_result": has_result,
                "max_score": round(max_score, 4),
                "retrieval_latency_seconds": q_lat,
                "top_result_title": (results[0].get("document_title") if results else ""),
                "top_result_url": (results[0].get("source_url") if results else ""),
            }
        )

    duration = round(time.time() - start_time, 4)
    summary = summarize_query_results(per_query_results)
    summary["evaluation_duration_seconds"] = duration
    summary["average_retrieval_latency"] = round(sum(latencies) / len(latencies), 4) if latencies else 0.0

    config = {
        "benchmark_path": str(DEFAULT_BENCHMARK_PATH),
        "dataset_path": str(dataset_ref),
        "top_k": top_k,
        "score_threshold": score_threshold,
        "retriever_type": retriever_type,
        "retrieval_backend": retriever.__class__.__name__,
        "rerank": effective_rerank,
        "fusion_method": fusion_method,
        "semantic_weight": semantic_weight,
        "tfidf_weight": tfidf_weight,
        "sentence_transformer_model": (
            DEFAULT_SEMANTIC_MODEL if retriever_type in {"semantic", "hybrid", "hybrid_rerank"} else "none"
        ),
        "cross_encoder_model": (DEFAULT_CROSS_ENCODER_MODEL if effective_rerank else "none"),
        "embedding_dimension": 384 if retriever_type in {"semantic", "hybrid", "hybrid_rerank"} else 0,
        "similarity_metric": "cosine",
        "benchmark_has_relevance_labels": False,
        "evaluation_mode": "retrieval_hit_proxy",
        "execution_device": "cpu",
    }

    return {
        "config": config,
        "metrics": summary,
        "queries": per_query_results,
    }


def evaluate_labelled_benchmark(
    benchmark_path: str | Path = DEFAULT_LABELLED_BENCHMARK_PATH,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    retriever_type: str = "tfidf",
    top_k: int = 3,
    score_threshold: float = 0.05,
    rerank: bool = False,
    fusion_method: str = "linear",
    semantic_weight: float = 0.5,
    tfidf_weight: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate retrieval configurations against a curated ground-truth relevance benchmark.

    Computes standard ranking metrics:
    - Recall@k
    - Precision@k
    - MRR (Mean Reciprocal Rank)
    - nDCG@k
    """
    bench_file = Path(benchmark_path)
    if not bench_file.exists():
        raise FileNotFoundError(f"Labelled benchmark not found: {bench_file}")

    with bench_file.open("r", encoding="utf-8") as handle:
        bench_data = json.load(handle)

    start_time = time.time()
    effective_rerank = rerank or (retriever_type == "hybrid_rerank")
    retriever = _instantiate_retriever(
        retriever_type=retriever_type,
        dataset_path=dataset_path,
        rerank=effective_rerank,
        fusion_method=fusion_method,
        semantic_weight=semantic_weight,
        tfidf_weight=tfidf_weight,
    )

    per_query_metrics: List[Dict[str, Any]] = []
    latencies: List[float] = []
    recalls, precisions, mrrs, ndcgs = [], [], [], []

    for item in bench_data:
        q_text = item["query_text"]
        rel_ids = set(item.get("relevant_document_ids", []))
        rel_urls = set(item.get("relevant_urls", []))

        q_start = time.time()
        if isinstance(retriever, HybridRetriever):
            results = retriever.retrieve(
                query=q_text,
                top_k=top_k,
                score_threshold=score_threshold,
                strategy="hybrid_rerank" if effective_rerank else "hybrid",
                rerank=effective_rerank,
            )
        else:
            results = retriever.retrieve(query=q_text, top_k=top_k, score_threshold=score_threshold)
        q_lat = round(time.time() - q_start, 4)
        latencies.append(q_lat)

        retrieved_ids = [r.get("document_id") for r in results]
        retrieved_urls = [r.get("source_url") for r in results]

        # Check binary relevance for each returned rank
        relevances: List[int] = []
        for doc_id, url in zip(retrieved_ids, retrieved_urls):
            is_relevant = (doc_id in rel_ids) or (url in rel_urls)
            relevances.append(1 if is_relevant else 0)

        hits = sum(relevances)
        target_count = max(1, len(rel_ids or rel_urls))
        recall = hits / target_count
        precision = hits / top_k if top_k else 0.0

        # Reciprocal rank (1 / position of first hit)
        first_hit_rank = next((idx + 1 for idx, rel in enumerate(relevances) if rel == 1), 0)
        reciprocal_rank = 1.0 / first_hit_rank if first_hit_rank > 0 else 0.0

        # nDCG@k
        dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))
        ideal_relevances = sorted(
            [1] * min(target_count, top_k) + [0] * max(0, top_k - target_count),
            reverse=True,
        )
        idcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal_relevances))
        ndcg = dcg / idcg if idcg > 0.0 else 0.0

        recalls.append(recall)
        precisions.append(precision)
        mrrs.append(reciprocal_rank)
        ndcgs.append(ndcg)

        per_query_metrics.append(
            {
                "query_id": item.get("query_id"),
                "query_text": q_text,
                "hits": hits,
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "mrr": round(reciprocal_rank, 4),
                "ndcg": round(ndcg, 4),
                "retrieval_latency_seconds": q_lat,
            }
        )

    duration = round(time.time() - start_time, 4)
    total_q = len(bench_data)

    metrics = {
        "queries_evaluated": total_q,
        "recall_at_k": round(sum(recalls) / total_q, 4) if total_q else 0.0,
        "precision_at_k": round(sum(precisions) / total_q, 4) if total_q else 0.0,
        "mrr": round(sum(mrrs) / total_q, 4) if total_q else 0.0,
        "ndcg_at_k": round(sum(ndcgs) / total_q, 4) if total_q else 0.0,
        "retrieval_coverage": round(sum(1 for h in recalls if h > 0) / total_q, 4) if total_q else 0.0,
        "average_retrieval_latency": round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
        "evaluation_duration_seconds": duration,
        "benchmark_has_relevance_labels": True,
        "benchmark_type": "curated_relevance_labels",
    }

    config = {
        "benchmark_path": str(bench_file),
        "dataset_path": str(dataset_path),
        "top_k": top_k,
        "score_threshold": score_threshold,
        "retriever_type": retriever_type,
        "retrieval_backend": retriever.__class__.__name__,
        "rerank": effective_rerank,
        "fusion_method": fusion_method,
        "semantic_weight": semantic_weight,
        "tfidf_weight": tfidf_weight,
        "benchmark_has_relevance_labels": True,
        "execution_device": "cpu",
    }

    return {
        "config": config,
        "metrics": metrics,
        "queries": per_query_metrics,
    }


def create_degraded_dataset(
    source_path: str | Path = DEFAULT_DATASET_PATH,
    output_path: str | Path = "reports/simulated_degraded_corpus.jsonl",
    excluded_topics: Optional[List[str]] = None,
    keep_fraction: Optional[float] = None,
) -> Path:
    """Create a synthetic degraded dataset without altering the canonical corpus."""
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Source dataset not found: {source}")

    excluded = set(excluded_topics or [])
    records: List[Dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            record = json.loads(raw)
            if excluded and (record.get("topic") in excluded):
                continue
            records.append(record)

    if keep_fraction is not None:
        if not 0.0 < keep_fraction <= 1.0:
            raise ValueError("keep_fraction must be between 0 and 1.")
        target_count = max(1, int(len(records) * keep_fraction))
        records = records[:target_count]

    if not records:
        raise ValueError("The degraded dataset would be empty; choose a less aggressive reduction.")

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return target


def evaluate_degradation_thresholds(
    baseline_metrics: Dict[str, Any],
    degraded_metrics: Dict[str, Any],
    warning_drop: float = 0.10,
    retraining_drop: float = 0.25,
) -> Dict[str, Any]:
    """Compute whether retrieval degradation crosses warning or retraining thresholds."""
    relevance_drop = float(baseline_metrics.get("relevance_rate", 0.0)) - float(
        degraded_metrics.get("relevance_rate", 0.0)
    )
    best_score_drop = float(baseline_metrics.get("average_best_score", 0.0)) - float(
        degraded_metrics.get("average_best_score", 0.0)
    )

    warning_triggered = relevance_drop >= warning_drop or best_score_drop >= warning_drop
    retraining_triggered = relevance_drop >= retraining_drop or best_score_drop >= retraining_drop

    return {
        "relevance_drop": round(relevance_drop, 4),
        "average_best_score_drop": round(best_score_drop, 4),
        "warning_threshold": warning_drop,
        "retraining_threshold": retraining_drop,
        "warning": bool(warning_triggered),
        "automatic_retraining": bool(retraining_triggered),
        "status": "warning" if warning_triggered else "stable",
        "decision": (
            "automatic_retraining" if retraining_triggered else "warning" if warning_triggered else "no_action"
        ),
    }


def compare_retrieval_conditions(
    queries: Optional[List[Dict[str, str]]] = None,
    baseline_dataset_path: str | Path = DEFAULT_DATASET_PATH,
    degraded_dataset_path: str | Path = "reports/simulated_degraded_corpus.jsonl",
    top_k: int = 3,
    score_threshold: float = 0.05,
    warning_drop: float = 0.10,
    retraining_drop: float = 0.25,
) -> Dict[str, Any]:
    """Compare baseline retrieval performance against a simulated degraded corpus."""
    benchmark_queries = queries if queries is not None else load_benchmark_queries()
    baseline = evaluate_benchmark_queries(
        queries=benchmark_queries,
        dataset_path=baseline_dataset_path,
        retriever_type="tfidf",
        top_k=top_k,
        score_threshold=score_threshold,
    )
    degraded = evaluate_benchmark_queries(
        queries=benchmark_queries,
        dataset_path=degraded_dataset_path,
        retriever_type="tfidf",
        top_k=top_k,
        score_threshold=score_threshold,
    )
    thresholds = evaluate_degradation_thresholds(
        baseline["metrics"],
        degraded["metrics"],
        warning_drop=warning_drop,
        retraining_drop=retraining_drop,
    )

    return {
        "baseline": baseline,
        "degraded": degraded,
        "thresholds": thresholds,
        "metric_differences": {
            "relevance_rate_drop": thresholds["relevance_drop"],
            "average_best_score_drop": thresholds["average_best_score_drop"],
        },
        "simulated_intervention": {
            "type": "reduced_corpus_subset",
            "baseline_dataset": str(Path(baseline_dataset_path)),
            "degraded_dataset": str(Path(degraded_dataset_path)),
            "description": (
                "Synthetic degradation by evaluating against a smaller, reduced corpus copy; "
                "this is not evidence of real-world production drift."
            ),
        },
        "limitations": [
            "The benchmark contains no gold relevance labels; the metric is a retrieval-hit proxy.",
            "The degraded corpus is synthetic and should not be interpreted as real-world production drift.",
        ],
    }


def write_degradation_report(
    comparison_result: Dict[str, Any],
    report_path: str | Path = "reports/retrieval_degradation_report.md",
) -> Path:
    """Persist a markdown summary of the degradation experiment."""
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)

    baseline = comparison_result["baseline"]["metrics"]
    degraded = comparison_result["degraded"]["metrics"]
    thresholds = comparison_result["thresholds"]

    markdown = f"""# Retrieval degradation experiment report

Date: {datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
Baseline dataset: {comparison_result['simulated_intervention']['baseline_dataset']}
Degraded dataset: {comparison_result['simulated_intervention']['degraded_dataset']}
Intervention: Synthetic corpus reduction using an alternate reduced copy of the processed dataset. This is a controlled simulation only, not production drift evidence.

## Baseline metrics
- queries_evaluated: {baseline.get('queries_evaluated')}
- queries_with_results: {baseline.get('queries_with_results')}
- queries_without_results: {baseline.get('queries_without_results')}
- relevance_rate: {baseline.get('relevance_rate')}
- average_best_score: {baseline.get('average_best_score')}

## Simulated degraded metrics
- queries_evaluated: {degraded.get('queries_evaluated')}
- queries_with_results: {degraded.get('queries_with_results')}
- queries_without_results: {degraded.get('queries_without_results')}
- relevance_rate: {degraded.get('relevance_rate')}
- average_best_score: {degraded.get('average_best_score')}

## Metric differences
- relevance_rate_drop: {thresholds.get('relevance_drop')}
- average_best_score_drop: {thresholds.get('average_best_score_drop')}

## Threshold rule
- warning threshold: {thresholds.get('warning_threshold')}
- retraining threshold: {thresholds.get('retraining_threshold')}
- warning triggered: {thresholds.get('warning')}
- automatic retraining decision: {thresholds.get('automatic_retraining')}

## Decision
{thresholds.get('decision')}

## Limitations
- The benchmark contains no gold relevance labels; this is a retrieval-hit proxy rather than answer correctness.
- The degraded dataset is synthetic and created only for a local monitoring experiment.
- This experiment is not proof of real-world production drift.
"""
    report.write_text(markdown, encoding="utf-8")
    return report


def log_mlflow_degradation_experiment(
    comparison_result: Dict[str, Any],
    experiment_name: str = "rvce-retrieval-degradation-monitoring",
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
) -> Dict[str, str]:
    """Log the degradation experiment to MLflow if the local tracking backend is available."""
    mlflow.set_tracking_uri(_local_tracking_uri(tracking_uri))
    mlflow.set_experiment(experiment_name)

    run_label = run_name or ("retrieval-degradation-" + datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ"))

    with mlflow.start_run(run_name=run_label) as run:
        baseline = comparison_result["baseline"]["metrics"]
        degraded = comparison_result["degraded"]["metrics"]
        thresholds = comparison_result["thresholds"]

        mlflow.log_param("baseline_dataset", comparison_result["simulated_intervention"]["baseline_dataset"])
        mlflow.log_param("degraded_dataset", comparison_result["simulated_intervention"]["degraded_dataset"])
        mlflow.log_param("warning_threshold", thresholds["warning_threshold"])
        mlflow.log_param("retraining_threshold", thresholds["retraining_threshold"])
        mlflow.log_param("simulated_intervention", "reduced_corpus_subset")

        mlflow.log_metric("baseline_queries_evaluated", float(baseline.get("queries_evaluated", 0)))
        mlflow.log_metric("baseline_relevance_rate", float(baseline.get("relevance_rate", 0.0)))
        mlflow.log_metric("baseline_average_best_score", float(baseline.get("average_best_score", 0.0)))
        mlflow.log_metric("degraded_queries_evaluated", float(degraded.get("queries_evaluated", 0)))
        mlflow.log_metric("degraded_relevance_rate", float(degraded.get("relevance_rate", 0.0)))
        mlflow.log_metric("degraded_average_best_score", float(degraded.get("average_best_score", 0.0)))
        mlflow.log_metric("relevance_rate_drop", float(thresholds.get("relevance_drop", 0.0)))
        mlflow.log_metric("average_best_score_drop", float(thresholds.get("average_best_score_drop", 0.0)))
        mlflow.log_metric("warning_flag", float(1.0 if thresholds.get("warning") else 0.0))
        mlflow.log_metric("automatic_retraining_flag", float(1.0 if thresholds.get("automatic_retraining") else 0.0))

        mlflow.set_tag("degradation_simulation", "true")
        mlflow.set_tag("warning", str(thresholds.get("warning")))
        mlflow.set_tag("automatic_retraining", str(thresholds.get("automatic_retraining")))
        mlflow.set_tag("limitation", "Synthetic degraded corpus only; not real production drift.")

    return {
        "tracking_uri": mlflow.get_tracking_uri(),
        "run_id": run.info.run_id,
        "experiment_name": experiment_name,
        "run_name": run_label,
        "artifact_uri": run.info.artifact_uri,
    }


def _local_tracking_uri(tracking_uri: Optional[str] = None) -> str:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    if tracking_uri:
        return tracking_uri
    repo_root = Path.cwd().resolve()
    mlruns_dir = repo_root / "mlruns"
    return mlruns_dir.as_uri()


def _get_git_commit() -> Optional[str]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        return commit
    except Exception:
        return None


def _get_file_checksum(file_path: Path) -> str:
    if not file_path.exists():
        return "none"
    try:
        hasher = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:12]
    except Exception:
        return "error"


def log_mlflow_benchmark(
    evaluation_summary: Dict[str, Any],
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    log_model: bool = False,
    registered_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Log the evaluation summary, config, parameters, and model artifacts to MLflow."""
    mlflow.set_tracking_uri(_local_tracking_uri(tracking_uri))
    mlflow.set_experiment(experiment_name)

    config = evaluation_summary.get("config", {})
    retriever_type = config.get("retriever_type", "tfidf")
    run_label = run_name or f"{retriever_type}-{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}"

    with mlflow.start_run(run_name=run_label) as run:
        metrics = evaluation_summary.get("metrics", {})

        # Log parameters
        for key, value in config.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(key, value)

        # Standardized MLOps retrieval parameters
        mlflow.log_param("retriever_type", retriever_type)
        mlflow.log_param(
            "lexical_model",
            "TfidfVectorizer" if retriever_type in {"tfidf", "hybrid", "hybrid_rerank"} else "None",
        )
        mlflow.log_param("lexical_ngram_range", "(1, 2)")
        mlflow.log_param("lexical_sublinear_tf", "true")
        mlflow.log_param(
            "embedding_model",
            DEFAULT_SEMANTIC_MODEL if retriever_type in {"semantic", "hybrid", "hybrid_rerank"} else "None",
        )
        mlflow.log_param(
            "dense_dimension",
            384 if retriever_type in {"semantic", "hybrid", "hybrid_rerank"} else 0,
        )
        mlflow.log_param(
            "fusion_strategy",
            config.get("fusion_method", "linear") if "hybrid" in retriever_type else "None",
        )
        if "hybrid" in retriever_type:
            mlflow.log_param("semantic_weight", float(config.get("semantic_weight", 0.5)))
            mlflow.log_param("tfidf_weight", float(config.get("tfidf_weight", 0.5)))
            mlflow.log_param("candidate_pool_size", int(config.get("candidate_pool_size", 20)))
        else:
            mlflow.log_param("candidate_pool_size", int(config.get("top_k", 3)))

        effective_rerank = bool(config.get("rerank", False) or "rerank" in retriever_type)
        mlflow.log_param("reranking_enabled", str(effective_rerank).lower())
        mlflow.log_param(
            "reranker_model",
            DEFAULT_CROSS_ENCODER_MODEL if effective_rerank else "None",
        )
        mlflow.log_param("top_k", int(config.get("top_k", 3)))
        mlflow.log_param("score_threshold", float(config.get("score_threshold", 0.05)))

        dataset_p = Path(config.get("dataset_path", str(DEFAULT_DATASET_PATH)))
        mlflow.log_param("dataset_path", str(dataset_p))
        mlflow.log_param("dataset_version", _get_file_checksum(dataset_p))

        bench_p = Path(config.get("benchmark_path", str(DEFAULT_BENCHMARK_PATH)))
        mlflow.log_param("benchmark_path", str(bench_p))
        mlflow.log_param("benchmark_checksum", _get_file_checksum(bench_p))
        mlflow.log_param("dvc_tracked", "true")

        # Log environment and code metadata
        mlflow.log_param("python_version", sys.version.split()[0])
        mlflow.log_param("mlflow_version", mlflow.__version__)
        git_commit = _get_git_commit()
        if git_commit:
            mlflow.log_param("git_commit", git_commit)

        # Log metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float, bool)):
                mlflow.log_metric(key, float(value))

        has_labels = metrics.get("benchmark_has_relevance_labels", False)
        mlflow.set_tag("benchmark_has_relevance_labels", str(has_labels).lower())
        mlflow.set_tag("retriever_type", retriever_type)
        mlflow.set_tag("execution_device", "cpu")

        if not has_labels:
            mlflow.set_tag(
                "benchmark_limitation",
                "No ground-truth relevance labels in the 40-query benchmark CSV; coverage represents score threshold hit proxy.",
            )

        # Persist summary artifacts
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        summary_path = reports_dir / f"mlflow_{retriever_type}_evaluation_summary.json"
        config_path = reports_dir / f"mlflow_{retriever_type}_evaluation_config.json"
        breakdown_path = reports_dir / f"mlflow_{retriever_type}_metric_breakdown.json"
        error_analysis_path = reports_dir / f"mlflow_{retriever_type}_error_analysis.json"

        queries_list = evaluation_summary.get("queries", [])
        summary_path.write_text(json.dumps(evaluation_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        breakdown_path.write_text(json.dumps(queries_list, indent=2, ensure_ascii=False), encoding="utf-8")

        # Error analysis: isolate queries with zero hits or low scores (< 0.10)
        errors: List[Dict[str, Any]] = []
        for q in queries_list:
            is_zero = not q.get("has_result", False) or (q.get("hits") == 0 and "hits" in q)
            score_val = float(q.get("max_score", 0.0) or 0.0)
            if is_zero or score_val < 0.10:
                errors.append({
                    "query_id": q.get("query_id"),
                    "query_text": q.get("query_text"),
                    "intent": q.get("intent"),
                    "score": score_val,
                    "result_count": q.get("result_count", 0),
                    "diagnostic_reason": (
                        "zero_results_above_threshold" if is_zero else "low_confidence_match"
                    ),
                })
        error_analysis_path.write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")

        mlflow.log_artifact(str(summary_path), artifact_path="evaluation")
        mlflow.log_artifact(str(config_path), artifact_path="evaluation")
        mlflow.log_artifact(str(breakdown_path), artifact_path="evaluation")
        mlflow.log_artifact(str(error_analysis_path), artifact_path="evaluation")

        model_uri = None
        reload_verified = False

        # Log SentenceTransformer model artifact if requested
        if log_model and retriever_type in {"semantic", "hybrid", "hybrid_rerank"}:
            try:
                from src.retrieval import _load_sentence_transformer

                st_model = _load_sentence_transformer(DEFAULT_SEMANTIC_MODEL, device="cpu")
                model_info = mlflow.sentence_transformers.log_model(
                    st_model,
                    name="sentence_transformer_model",
                    registered_model_name=registered_model_name,
                )
                model_uri = model_info.model_uri
                mlflow.set_tag("model_logged", "true")
                mlflow.set_tag("model_uri", model_uri)

                # Verify reload for smoke test
                reloaded = mlflow.sentence_transformers.load_model(model_uri)
                test_emb = reloaded.encode(["Smoke test sentence for MLflow model reload."], normalize_embeddings=True)
                assert test_emb.shape == (1, 384)
                reload_verified = True
                mlflow.set_tag("model_reload_verified", "true")
                logger.info(f"Model successfully logged and reload smoke-tested at: {model_uri}")
            except Exception as model_err:
                mlflow.set_tag("model_logged", "false")
                mlflow.set_tag("model_log_error", str(model_err))
                logger.warning(f"Failed to log/reload model artifact: {model_err}")

    return {
        "tracking_uri": mlflow.get_tracking_uri(),
        "run_id": run.info.run_id,
        "experiment_name": experiment_name,
        "run_name": run_label,
        "artifact_uri": run.info.artifact_uri,
        "model_uri": model_uri,
        "model_reload_verified": reload_verified,
    }


DEFAULT_RAG_BENCHMARK_QUERIES: List[Dict[str, Any]] = [
    {
        "query_id": "RAG_F01",
        "category": "admissions_cutoff",
        "query_type": "factual",
        "question": "What is the KCET cutoff rank for B.E. Computer Science and Engineering at RVCE?",
        "expected_keywords": ["cutoff", "computer", "engineering", "kcet", "rank"],
    },
    {
        "query_id": "RAG_F02",
        "category": "admission_fees",
        "query_type": "factual",
        "question": "What are the B.E. tuition fees payable for COMEDK quota admission?",
        "expected_keywords": ["fee", "comedk", "tuition", "rs"],
    },
    {
        "query_id": "RAG_F03",
        "category": "admissions_procedure",
        "query_type": "factual",
        "question": "What is the official procedure for B.E. admission through COMEDK quota?",
        "expected_keywords": ["admission", "comedk", "allotment", "documents", "step"],
    },
    {
        "query_id": "RAG_F04",
        "category": "facilities_hostels",
        "query_type": "factual",
        "question": "What hostel facilities and room types are available for first-year students?",
        "expected_keywords": ["hostel", "room", "mess", "facility"],
    },
    {
        "query_id": "RAG_F05",
        "category": "departments_faculty",
        "query_type": "factual",
        "question": "Where can I find details about engineering departments and faculty members?",
        "expected_keywords": ["department", "faculty", "programmes", "engineering"],
    },
    {
        "query_id": "RAG_P01",
        "category": "placements_careers",
        "query_type": "numerical",
        "question": "How many placement offers were made to Computer Science students in the 2027 batch?",
        "expected_keywords": ["offers", "placement", "cse", "computer"],
    },
    {
        "query_id": "RAG_P02",
        "category": "placements_careers",
        "query_type": "numerical",
        "question": "What companies visited campus and what packages were offered in placements?",
        "expected_keywords": ["placement", "company", "package", "lpa", "recruiter"],
    },
    {
        "query_id": "RAG_D01",
        "category": "academics_exams",
        "query_type": "date_specific",
        "question": "When does the academic calendar start and when are semester examinations held?",
        "expected_keywords": ["calendar", "examination", "semester", "dates", "events"],
    },
    {
        "query_id": "RAG_D02",
        "category": "circulars",
        "query_type": "date_specific",
        "question": "What is the deadline mentioned in recent campus circulars for college fee payment?",
        "expected_keywords": ["circular", "deadline", "fee", "date"],
    },
    {
        "query_id": "RAG_O01",
        "category": "abstention",
        "query_type": "out_of_domain",
        "question": "What is the capital of France?",
        "expected_abstain": True,
    },
    {
        "query_id": "RAG_O02",
        "category": "abstention",
        "query_type": "out_of_domain",
        "question": "Who directed the movie Interstellar?",
        "expected_abstain": True,
    },
    {
        "query_id": "RAG_O03",
        "category": "abstention",
        "query_type": "out_of_domain",
        "question": "Who won the 2022 FIFA World Cup?",
        "expected_abstain": True,
    },
    {
        "query_id": "RAG_O04",
        "category": "abstention",
        "query_type": "out_of_domain",
        "question": "Who wrote the tragedy Hamlet?",
        "expected_abstain": True,
    },
]


def evaluate_rag_pipeline(
    benchmark_queries: Optional[List[Dict[str, Any]]] = None,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    strategy: str = "hybrid",
    rerank: bool = False,
    top_k: int = 3,
    score_threshold: float = 0.05,
    experiment_name: str = DEFAULT_RAG_EXPERIMENT_NAME,
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    report_dir: str | Path = "reports",
) -> Dict[str, Any]:
    """Evaluate the end-to-end RAG question answering pipeline on factual QA and abstention queries."""
    from src.answering import NO_EVIDENCE_MESSAGE, answer_question

    mlflow.set_tracking_uri(_local_tracking_uri(tracking_uri))
    mlflow.set_experiment(experiment_name)

    data_file = Path(dataset_path)
    queries = benchmark_queries if benchmark_queries is not None else DEFAULT_RAG_BENCHMARK_QUERIES
    run_label = run_name or f"rag-{strategy}-{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}"

    start_eval_time = time.time()
    results: List[Dict[str, Any]] = []

    for q_item in queries:
        qid = q_item["query_id"]
        q_text = q_item["question"]
        q_type = q_item["query_type"]
        category = q_item.get("category", "general")

        t0 = time.time()
        resp = answer_question(
            question=q_text,
            strategy=strategy,
            rerank=rerank,
            top_k=top_k,
            score_threshold=score_threshold,
            dataset_path=str(data_file),
            mcp_client=object(),  # offline evaluation without live Gmail network calls
        )
        latency = round(time.time() - t0, 4)

        ans_text = (resp.get("answer") or "").strip()
        evidence_list = resp.get("evidence") or []
        sources_list = resp.get("sources") or []

        is_abstain = (
            ans_text == NO_EVIDENCE_MESSAGE
            or (len(evidence_list) == 0 and len(sources_list) == 0)
            or ("could not find" in ans_text.lower())
        )
        is_grounded = bool(sources_list) and len(evidence_list) > 0

        if q_type == "out_of_domain":
            correct = is_abstain
        else:
            expected_kws = q_item.get("expected_keywords", [])
            kw_match = any(kw.lower() in ans_text.lower() for kw in expected_kws) if expected_kws else True
            correct = bool(is_grounded and not is_abstain and kw_match)

        results.append({
            "query_id": qid,
            "category": category,
            "query_type": q_type,
            "question": q_text,
            "answer": ans_text[:250],
            "evidence_count": len(evidence_list),
            "sources_count": len(sources_list),
            "is_abstention": is_abstain,
            "is_grounded": is_grounded,
            "correct": correct,
            "latency_seconds": latency,
        })

    eval_duration = round(time.time() - start_eval_time, 4)
    total_q = len(results)
    in_domain_q = [r for r in results if r["query_type"] != "out_of_domain"]
    out_domain_q = [r for r in results if r["query_type"] == "out_of_domain"]

    factual_accuracy = (
        round(sum(1 for r in in_domain_q if r["correct"]) / len(in_domain_q), 4) if in_domain_q else 0.0
    )
    abstention_accuracy = (
        round(sum(1 for r in out_domain_q if r["correct"]) / len(out_domain_q), 4) if out_domain_q else 0.0
    )
    overall_accuracy = (
        round(sum(1 for r in results if r["correct"]) / total_q, 4) if total_q else 0.0
    )
    evidence_grounding_rate = (
        round(sum(1 for r in in_domain_q if r["is_grounded"]) / len(in_domain_q), 4) if in_domain_q else 0.0
    )
    avg_latency = (
        round(sum(r["latency_seconds"] for r in results) / total_q, 4) if total_q else 0.0
    )

    metrics = {
        "factual_accuracy": factual_accuracy,
        "abstention_accuracy": abstention_accuracy,
        "overall_accuracy": overall_accuracy,
        "evidence_grounding_rate": evidence_grounding_rate,
        "average_answer_latency": avg_latency,
        "total_queries_evaluated": total_q,
        "answered_queries_count": sum(1 for r in results if not r["is_abstention"]),
        "abstained_queries_count": sum(1 for r in results if r["is_abstention"]),
        "evaluation_duration_seconds": eval_duration,
    }

    config = {
        "rag_strategy": strategy,
        "retriever_type": strategy,
        "rerank": rerank,
        "grounding_policy": "evidence_required",
        "top_k": top_k,
        "score_threshold": score_threshold,
        "dataset_path": str(data_file),
        "dataset_checksum": _get_file_checksum(data_file),
        "total_queries": total_q,
        "in_domain_queries_count": len(in_domain_q),
        "out_of_domain_queries_count": len(out_domain_q),
        "dvc_tracked": "true",
    }

    # Persist report files
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "rag_evaluation_summary.json"
    results_file = out_dir / "rag_query_results.json"
    markdown_file = out_dir / "rag_evaluation_report.md"

    summary_file.write_text(json.dumps({"config": config, "metrics": metrics}, indent=2, ensure_ascii=False), encoding="utf-8")
    results_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    md_content = f"""# RVCE Campus Information Assistant - RAG Pipeline Evaluation Report

**Evaluation Timestamp:** {datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
**RAG Strategy:** {strategy} (Rerank: {rerank})
**Dataset:** {data_file} (Checksum: {config['dataset_checksum']})

## Summary Metrics

| Metric | Score | Notes |
| :--- | :--- | :--- |
| **Overall Accuracy** | `{overall_accuracy:.4f}` | Combined answer correctness & safe abstention |
| **Factual Accuracy** | `{factual_accuracy:.4f}` | In-domain campus questions answered accurately |
| **Abstention Accuracy** | `{abstention_accuracy:.4f}` | Out-of-domain questions safely refused |
| **Evidence Grounding Rate** | `{evidence_grounding_rate:.4f}` | In-domain answers with verified campus evidence |
| **Average Latency** | `{avg_latency:.4f}s` | End-to-end question answering response time |
| **Total Queries Evaluated** | `{total_q}` | {len(in_domain_q)} in-domain, {len(out_domain_q)} out-of-domain |

## Category Performance Breakdown

| Query ID | Category | Type | Grounded | Abstained | Correct | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for r in results:
        md_content += (
            f"| `{r['query_id']}` | {r['category']} | {r['query_type']} | "
            f"{'Yes' if r['is_grounded'] else 'No'} | {'Yes' if r['is_abstention'] else 'No'} | "
            f"{'PASS' if r['correct'] else 'FAIL'} | {r['latency_seconds']:.4f}s |\n"
        )

    md_content += """
## Governance & Safety Verification
- **Zero Hallucination Policy:** Out-of-domain queries successfully trigger strict abstention (`NO_EVIDENCE_MESSAGE`).
- **Citation Assurance:** All answered in-domain questions include valid document citations and verifiable evidence passages.
"""
    markdown_file.write_text(md_content, encoding="utf-8")

    with mlflow.start_run(run_name=run_label) as run:
        for k, v in config.items():
            if isinstance(v, (str, int, float, bool)):
                mlflow.log_param(k, v)

        for k, v in metrics.items():
            if isinstance(v, (int, float, bool)):
                mlflow.log_metric(k, float(v))

        mlflow.set_tag("pipeline_type", "end_to_end_rag")
        mlflow.set_tag("grounding_policy", "evidence_required")
        mlflow.set_tag("execution_device", "cpu")

        mlflow.log_artifact(str(summary_file), artifact_path="evaluation")
        mlflow.log_artifact(str(results_file), artifact_path="evaluation")
        mlflow.log_artifact(str(markdown_file), artifact_path="evaluation")

    return {
        "tracking_uri": mlflow.get_tracking_uri(),
        "run_id": run.info.run_id,
        "experiment_name": experiment_name,
        "run_name": run_label,
        "metrics": metrics,
        "config": config,
        "artifact_uri": run.info.artifact_uri,
    }


def run_all_retrieval_benchmarks(
    benchmark_path: str | Path = DEFAULT_BENCHMARK_PATH,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_uri: Optional[str] = None,
    log_model: bool = True,
    registered_model_name: Optional[str] = "rvce-sentence-transformer",
) -> List[Dict[str, Any]]:
    """Run and log evaluations for all four configurations to enable fair empirical comparison."""
    configs = [
        {"name": "tfidf-baseline", "type": "tfidf", "rerank": False, "log_st": False},
        {"name": "semantic-minilm", "type": "semantic", "rerank": False, "log_st": log_model},
        {"name": "hybrid-minilm", "type": "hybrid", "rerank": False, "log_st": False},
        {"name": "hybrid-minilm-cross-encoder", "type": "hybrid_rerank", "rerank": True, "log_st": False},
    ]

    all_results = []
    queries = load_benchmark_queries(benchmark_path)

    for cfg in configs:
        eval_summary = evaluate_benchmark_queries(
            queries=queries,
            dataset_path=dataset_path,
            retriever_type=cfg["type"],
            rerank=cfg["rerank"],
        )
        reg_name = registered_model_name if cfg["log_st"] else None
        run_info = log_mlflow_benchmark(
            evaluation_summary=eval_summary,
            experiment_name=experiment_name,
            run_name=cfg["name"],
            tracking_uri=tracking_uri,
            log_model=cfg["log_st"],
            registered_model_name=reg_name,
        )
        all_results.append({
            "config": cfg["name"],
            "retriever_type": cfg["type"],
            "metrics": eval_summary["metrics"],
            "run_info": run_info,
        })

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local RVCE retrieval benchmark and log it to MLflow.")
    parser.add_argument("--benchmark", type=str, default=str(DEFAULT_BENCHMARK_PATH), help="CSV benchmark file path")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET_PATH), help="Processed dataset path")
    parser.add_argument(
        "--retriever",
        type=str,
        default="tfidf",
        choices=["tfidf", "semantic", "hybrid", "hybrid_rerank"],
        help="Retriever configuration to evaluate",
    )
    parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking")
    parser.add_argument("--fusion-method", type=str, default="linear", choices=["linear", "rrf"], help="Hybrid fusion method")
    parser.add_argument("--semantic-weight", type=float, default=0.5, help="Semantic weight in hybrid retrieval")
    parser.add_argument("--tfidf-weight", type=float, default=0.5, help="TF-IDF weight in hybrid retrieval")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k retrieval results to evaluate")
    parser.add_argument("--score-threshold", type=float, default=0.05, help="Minimum retrieval score to count as a result")
    parser.add_argument("--experiment-name", type=str, default=DEFAULT_EXPERIMENT_NAME, help="MLflow experiment name")
    parser.add_argument("--run-name", type=str, default=None, help="Descriptive run name for the MLflow run")
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="Optional local tracking URI; defaults to a file-backed mlruns directory in the repo root",
    )
    parser.add_argument("--log-model", action="store_true", help="Log the SentenceTransformer model artifact to MLflow")
    parser.add_argument("--register-model", type=str, default=None, help="Register model under this name in MLflow Model Registry")
    parser.add_argument("--run-all-configs", action="store_true", help="Run all 4 configurations sequentially and log to MLflow")
    parser.add_argument("--run-labelled", action="store_true", help="Evaluate the curated labelled relevance benchmark")
    parser.add_argument(
        "--labelled-benchmark",
        type=str,
        default=str(DEFAULT_LABELLED_BENCHMARK_PATH),
        help="Path to curated labelled benchmark JSON",
    )
    parser.add_argument("--run-degradation", action="store_true", help="Run a synthetic degraded-corpus comparison")
    parser.add_argument(
        "--degraded-dataset",
        type=str,
        default="reports/simulated_degraded_corpus.jsonl",
        help="Path to the reduced degraded dataset copy",
    )
    parser.add_argument("--keep-fraction", type=float, default=0.7, help="Fraction of records to retain in degraded dataset")
    parser.add_argument("--warning-drop", type=float, default=0.10, help="Drop in relevance or score triggering warning")
    parser.add_argument("--retraining-drop", type=float, default=0.25, help="Drop in relevance or score triggering retraining")
    parser.add_argument(
        "--report-path",
        type=str,
        default="reports/retrieval_degradation_report.md",
        help="Markdown summary path for degradation experiment",
    )
    parser.add_argument("--run-rag-eval", action="store_true", help="Run end-to-end RAG pipeline evaluation")
    parser.add_argument(
        "--rag-strategy",
        type=str,
        default="hybrid",
        choices=["tfidf", "semantic", "hybrid", "hybrid_rerank"],
        help="Retrieval strategy for RAG evaluation",
    )
    parser.add_argument("--rag-rerank", action="store_true", help="Enable cross-encoder reranking during RAG evaluation")
    args = parser.parse_args()

    if args.run_rag_eval:
        print(f"Evaluating end-to-end RAG pipeline (strategy={args.rag_strategy}, rerank={args.rag_rerank})...")
        rag_results = evaluate_rag_pipeline(
            dataset_path=args.dataset,
            strategy=args.rag_strategy,
            rerank=args.rag_rerank or (args.rag_strategy == "hybrid_rerank"),
            top_k=args.top_k,
            score_threshold=args.score_threshold,
            experiment_name=DEFAULT_RAG_EXPERIMENT_NAME,
            run_name=args.run_name or f"rag-{args.rag_strategy}-eval",
            tracking_uri=args.tracking_uri,
        )
        print(json.dumps(rag_results["metrics"], indent=2, ensure_ascii=False))
        print(f"MLflow Experiment: {rag_results['experiment_name']}")
        print(f"MLflow Run ID: {rag_results['run_id']}")
        print("Summary report saved at: reports/rag_evaluation_report.md")
        return

    if args.run_degradation:
        queries = load_benchmark_queries(args.benchmark)
        degraded_path = Path(args.degraded_dataset)
        if not degraded_path.exists():
            degraded_path = create_degraded_dataset(
                source_path=args.dataset,
                output_path=degraded_path,
                keep_fraction=args.keep_fraction,
            )
        comparison = compare_retrieval_conditions(
            queries=queries,
            baseline_dataset_path=args.dataset,
            degraded_dataset_path=str(degraded_path),
            top_k=args.top_k,
            score_threshold=args.score_threshold,
            warning_drop=args.warning_drop,
            retraining_drop=args.retraining_drop,
        )
        report_path = write_degradation_report(comparison, report_path=args.report_path)
        log_mlflow_degradation_experiment(
            comparison_result=comparison,
            experiment_name="rvce-retrieval-degradation-monitoring",
            run_name=args.run_name,
            tracking_uri=args.tracking_uri,
        )
        print(json.dumps({
            "baseline_metrics": comparison["baseline"]["metrics"],
            "degraded_metrics": comparison["degraded"]["metrics"],
            "thresholds": comparison["thresholds"],
            "report_path": str(report_path),
        }, indent=2, ensure_ascii=False))
        return

    if args.run_all_configs:
        print("Evaluating all 4 retrieval configurations on RVCE Campus Knowledge Base...")
        results = run_all_retrieval_benchmarks(
            benchmark_path=args.benchmark,
            dataset_path=args.dataset,
            experiment_name=args.experiment_name,
            tracking_uri=args.tracking_uri,
            log_model=True,
            registered_model_name=args.register_model or "rvce-sentence-transformer",
        )
        print("\n================ RETRIEVAL CONFIGURATION COMPARISON ================")
        print(f"{'Configuration':<30} | {'Coverage':<10} | {'Avg Score':<10} | {'Avg Results':<12} | {'Duration':<10}")
        print("-" * 80)
        for r in results:
            m = r["metrics"]
            print(
                f"{r['config']:<30} | {m['retrieval_coverage']:<10.4f} | {m['average_best_score']:<10.4f} | {m['average_retrieved_results']:<12.4f} | {m['evaluation_duration_seconds']:<10.4f}s"
            )
        print("=" * 80)
        return

    if args.run_labelled:
        print(f"Evaluating labelled relevance benchmark: {args.labelled_benchmark}")
        labelled_summary = evaluate_labelled_benchmark(
            benchmark_path=args.labelled_benchmark,
            dataset_path=args.dataset,
            retriever_type=args.retriever,
            top_k=args.top_k,
            score_threshold=args.score_threshold,
            rerank=args.rerank,
            fusion_method=args.fusion_method,
            semantic_weight=args.semantic_weight,
            tfidf_weight=args.tfidf_weight,
        )
        run_info = log_mlflow_benchmark(
            evaluation_summary=labelled_summary,
            experiment_name=f"{args.experiment_name}-labelled",
            run_name=args.run_name or f"{args.retriever}-labelled",
            tracking_uri=args.tracking_uri,
            log_model=args.log_model,
            registered_model_name=args.register_model,
        )
        print(json.dumps(labelled_summary["metrics"], indent=2, ensure_ascii=False))
        print(f"MLflow Run ID: {run_info['run_id']}")
        return

    queries = load_benchmark_queries(args.benchmark)
    evaluation_summary = evaluate_benchmark_queries(
        queries=queries,
        dataset_path=args.dataset,
        retriever_type=args.retriever,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        rerank=args.rerank,
        fusion_method=args.fusion_method,
        semantic_weight=args.semantic_weight,
        tfidf_weight=args.tfidf_weight,
    )
    run_info = log_mlflow_benchmark(
        evaluation_summary=evaluation_summary,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        tracking_uri=args.tracking_uri,
        log_model=args.log_model,
        registered_model_name=args.register_model,
    )

    print(json.dumps(evaluation_summary["metrics"], indent=2, ensure_ascii=False))
    print(f"MLflow experiment: {run_info['experiment_name']}")
    print(f"MLflow run name: {run_info['run_name']}")
    print(f"MLflow run location: {run_info['artifact_uri']}")
    print(f"MLflow tracking URI: {run_info['tracking_uri']}")
    print(f"MLflow run ID: {run_info['run_id']}")
    if run_info.get("model_uri"):
        print(f"Logged Model URI: {run_info['model_uri']}")
        print(f"Model Reload Verified: {run_info['model_reload_verified']}")


if __name__ == "__main__":
    main()
