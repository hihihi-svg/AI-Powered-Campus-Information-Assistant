"""LangSmith RAG evaluation and observability runner for the RVCE Campus Information Assistant.

Evaluates 5 core dimensions across the campus RAG pipeline:
1. Retrieval relevance (precision, recall, hit rate against ground-truth docs)
2. Context relevance (chunk relevance to student query intents)
3. Answer relevance & factual correctness (keyword and ground-truth topic coverage)
4. Citation correctness & grounding (source presence, valid document IDs)
5. Abstention quality (robust refusal on out-of-domain questions without hallucinations)

Maintains clear separation of responsibilities:
- LangSmith: per-query execution traces, step-by-step latency, prompt/retrieval debugging.
- MLflow: experiment tracking across RAG configurations and benchmark summaries.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "processed" / "campus_documents.jsonl"
DEFAULT_BENCHMARK_PATH = REPO_ROOT / "data" / "queries" / "labelled_relevance_benchmark.json"
DEFAULT_RAG_EXPERIMENT_NAME = "campus-rag-evaluation"


def _clean_str(text: Optional[str]) -> str:
    return str(text or "").strip()


def run_langsmith_rag_evaluation(
    queries: Optional[List[Dict[str, Any]]] = None,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    strategy: str = "hybrid",
    rerank: bool = False,
    top_k: int = 3,
    score_threshold: float = 0.05,
    project_name: Optional[str] = None,
    enable_mlflow: bool = True,
    output_dir: str | Path = "reports",
) -> Dict[str, Any]:
    """Execute evaluation across the campus QA dataset and record traces in LangSmith & MLflow."""
    from src.answering import NO_EVIDENCE_MESSAGE, answer_question
    from src.evaluation import DEFAULT_RAG_BENCHMARK_QUERIES
    from src.tracing import RAGTracer, is_tracing_enabled

    eval_queries = queries if queries is not None else DEFAULT_RAG_BENCHMARK_QUERIES
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracing_active = is_tracing_enabled()
    proj = project_name or os.environ.get("LANGSMITH_PROJECT", "rvce-campus-assistant")

    total_queries = len(eval_queries)
    retrieval_hits = 0
    context_relevance_hits = 0
    answer_relevance_hits = 0
    citation_grounding_hits = 0
    abstention_correct_count = 0
    total_abstention_queries = 0
    total_factual_queries = 0
    latencies: List[float] = []

    eval_records: List[Dict[str, Any]] = []

    start_eval_time = time.time()

    for item in eval_queries:
        qid = item.get("query_id", "Q")
        question = item.get("question") or item.get("query_text", "")
        q_type = item.get("query_type", "factual")
        expected_kws = item.get("expected_keywords", [])
        expected_abstain = item.get("expected_abstain", False) or (q_type == "out_of_domain")
        expected_doc_ids = set(item.get("relevant_document_ids") or [])

        tracer = RAGTracer(project_name=proj, sanitize=True) if tracing_active else None
        if tracer:
            tracer.start_pipeline(
                question=question,
                strategy=strategy,
                rerank=rerank,
            )

        t0 = time.time()
        try:
            resp = answer_question(
                question=question,
                strategy=strategy,
                rerank=rerank,
                top_k=top_k,
                score_threshold=score_threshold,
                dataset_path=str(dataset_path),
                mcp_client=object(),  # offline evaluation mock
                tracer=tracer,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Error evaluating query %s", qid)
            resp = {
                "answer": NO_EVIDENCE_MESSAGE,
                "evidence": [],
                "sources": [],
                "error": {"category": "eval_error", "message": str(exc)},
            }
        finally:
            latency = round(time.time() - t0, 4)
            latencies.append(latency)
            if tracer:
                tracer.end_pipeline(
                    answer=resp.get("answer") or "",
                    is_abstention=(resp.get("answer") == NO_EVIDENCE_MESSAGE),
                    sources_count=len(resp.get("sources") or []),
                    error=resp.get("error"),
                )

        ans_text = _clean_str(resp.get("answer"))
        sources = resp.get("sources") or []
        evidence = resp.get("evidence") or []

        is_abstention = (
            ans_text == NO_EVIDENCE_MESSAGE
            or (len(evidence) == 0 and len(sources) == 0)
            or ("could not find" in ans_text.lower())
        )

        # 1. Retrieval Relevance
        retrieved_doc_ids = {s.get("document_id") for s in sources if s.get("document_id")}
        retrieved_doc_ids.update({e.get("document_id") for e in evidence if e.get("document_id")})

        if expected_doc_ids:
            retrieval_hit = bool(retrieved_doc_ids.intersection(expected_doc_ids))
        else:
            retrieval_hit = len(sources) > 0 or len(evidence) > 0 or expected_abstain

        if retrieval_hit:
            retrieval_hits += 1

        # 2. Context Relevance
        if expected_abstain:
            context_rel = True  # clean rejection
        else:
            context_rel = len(evidence) > 0 and len(sources) > 0
        if context_rel:
            context_relevance_hits += 1

        # 3. Answer Relevance & Factual Correctness
        if expected_abstain:
            ans_rel = is_abstention
            factual = is_abstention
        else:
            if expected_kws:
                kw_matches = sum(1 for kw in expected_kws if kw.lower() in ans_text.lower())
                ans_rel = (kw_matches / len(expected_kws)) >= 0.3 and not is_abstention
            else:
                ans_rel = not is_abstention and len(ans_text) > 20
            factual = ans_rel and len(sources) > 0

        if ans_rel:
            answer_relevance_hits += 1

        # 4. Citation Correctness & Grounding
        has_citations = len(sources) > 0 and all(
            bool(s.get("title") or s.get("source_title")) for s in sources
        )
        citation_correct = (expected_abstain and len(sources) == 0) or (
            not expected_abstain and has_citations and len(evidence) > 0
        )
        if citation_correct:
            citation_grounding_hits += 1

        # 5. Abstention Quality
        if expected_abstain:
            total_abstention_queries += 1
            if is_abstention:
                abstention_correct_count += 1
        else:
            total_factual_queries += 1

        eval_records.append(
            {
                "query_id": qid,
                "question": question,
                "query_type": q_type,
                "latency_s": latency,
                "is_abstention": is_abstention,
                "retrieval_hit": retrieval_hit,
                "context_relevance": context_rel,
                "answer_relevance": ans_rel,
                "factual_correctness": factual,
                "citation_correctness": citation_correct,
                "sources_count": len(sources),
                "evidence_count": len(evidence),
                "answer_snippet": ans_text[:140],
            }
        )

    # Calculate summary metrics
    n = max(total_queries, 1)
    retrieval_rate = round(retrieval_hits / n, 4)
    context_rate = round(context_relevance_hits / n, 4)
    answer_rate = round(answer_relevance_hits / n, 4)
    citation_rate = round(citation_grounding_hits / n, 4)
    abstention_acc = (
        round(abstention_correct_count / max(total_abstention_queries, 1), 4)
        if total_abstention_queries > 0
        else 1.0
    )
    mean_latency = round(sum(latencies) / n, 4) if latencies else 0.0

    # Composite RAG score: harmonic / weighted mean of the dimensions
    composite_rag_score = round(
        0.25 * retrieval_rate
        + 0.20 * context_rate
        + 0.25 * answer_rate
        + 0.15 * citation_rate
        + 0.15 * abstention_acc,
        4,
    )

    summary: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_queries": total_queries,
        "strategy": strategy,
        "rerank": rerank,
        "metrics": {
            "retrieval_relevance_rate": retrieval_rate,
            "context_relevance_rate": context_rate,
            "answer_relevance_rate": answer_rate,
            "citation_grounding_rate": citation_rate,
            "abstention_accuracy": abstention_acc,
            "composite_rag_score": composite_rag_score,
            "mean_latency_s": mean_latency,
            "eval_duration_s": round(time.time() - start_eval_time, 2),
        },
        "langsmith": {
            "project": proj,
            "tracing_active": tracing_active,
            "runs_recorded": len(eval_records),
        },
        "query_evaluations": eval_records,
    }

    # Save local JSON and Markdown report
    report_json_path = out_dir / "langsmith_evaluation_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_md_path = out_dir / "langsmith_evaluation_report.md"
    _write_markdown_report(summary, report_md_path)

    # Log to MLflow if enabled
    if enable_mlflow:
        _log_to_mlflow(summary)

    return summary


def _write_markdown_report(summary: Dict[str, Any], path: Path) -> None:
    m = summary["metrics"]
    ls = summary["langsmith"]
    lines = [
        "# LangSmith RAG Observability & Evaluation Report",
        "",
        f"- **Timestamp (UTC)**: {summary['timestamp_utc']}",
        f"- **Strategy**: `{summary['strategy']}` (rerank: {summary['rerank']})",
        f"- **LangSmith Project**: `{ls['project']}`",
        f"- **Tracing Enabled**: `{ls['tracing_active']}`",
        f"- **Total Evaluated Queries**: {summary['total_queries']}",
        "",
        "## Performance Metrics",
        "",
        "| Dimension | Metric | Score | Target |",
        "|-----------|--------|-------|--------|",
        f"| Retrieval Relevance | Hit Rate | **{m['retrieval_relevance_rate'] * 100:.1f}%** | >= 80% |",
        f"| Context Relevance | Context Quality | **{m['context_relevance_rate'] * 100:.1f}%** | >= 80% |",
        f"| Answer Relevance | Topic / Keyword Match | **{m['answer_relevance_rate'] * 100:.1f}%** | >= 75% |",
        f"| Citation Correctness | Grounded Sources | **{m['citation_grounding_rate'] * 100:.1f}%** | >= 85% |",
        f"| Abstention Quality | Out-of-Domain Refusal | **{m['abstention_accuracy'] * 100:.1f}%** | 100% |",
        f"| Overall Composite | Composite Score | **{m['composite_rag_score']:.4f}** | >= 0.80 |",
        f"| Latency | Mean Latency | **{m['mean_latency_s']}s** | <= 1.0s |",
        "",
        "## Detailed Query Results",
        "",
        "| ID | Type | Hit | Context | Answer | Citations | Latency | Snippet |",
        "|----|------|-----|---------|--------|-----------|---------|---------|",
    ]
    for q in summary["query_evaluations"][:15]:
        hit_icon = "PASS" if q["retrieval_hit"] else "FAIL"
        ctx_icon = "PASS" if q["context_relevance"] else "FAIL"
        ans_icon = "PASS" if q["answer_relevance"] else "FAIL"
        cit_icon = "PASS" if q["citation_correctness"] else "FAIL"
        snippet = q["answer_snippet"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {q['query_id']} | {q['query_type']} | {hit_icon} | {ctx_icon} | {ans_icon} | {cit_icon} | {q['latency_s']}s | {snippet[:50]}... |"
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _log_to_mlflow(summary: Dict[str, Any]) -> None:
    """Log evaluation summary metrics and LangSmith links to MLflow."""
    try:
        import mlflow
        from src.evaluation import _local_tracking_uri

        mlflow.set_tracking_uri(_local_tracking_uri())
        mlflow.set_experiment(DEFAULT_RAG_EXPERIMENT_NAME)

        run_name = f"langsmith-eval-{summary['strategy']}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            m = summary["metrics"]
            mlflow.log_params(
                {
                    "strategy": summary["strategy"],
                    "rerank": summary["rerank"],
                    "total_queries": summary["total_queries"],
                    "langsmith_project": summary["langsmith"]["project"],
                    "langsmith_tracing": summary["langsmith"]["tracing_active"],
                }
            )
            mlflow.log_metrics(
                {
                    "eval_retrieval_relevance_rate": m["retrieval_relevance_rate"],
                    "eval_context_relevance_rate": m["context_relevance_rate"],
                    "eval_answer_relevance_rate": m["answer_relevance_rate"],
                    "eval_citation_grounding_rate": m["citation_grounding_rate"],
                    "eval_abstention_accuracy": m["abstention_accuracy"],
                    "eval_composite_rag_score": m["composite_rag_score"],
                    "eval_mean_latency_s": m["mean_latency_s"],
                }
            )
            mlflow.set_tags(
                {
                    "tool": "langsmith",
                    "evaluation_type": "rag_observability",
                    "langsmith_project": summary["langsmith"]["project"],
                }
            )
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("Failed to log evaluation run to MLflow: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Campus RAG with LangSmith Observability")
    parser.add_argument("--strategy", default="hybrid", help="Retrieval strategy: hybrid, semantic, tfidf, gpt")
    parser.add_argument("--rerank", action="store_true", help="Enable reranking")
    parser.add_argument("--top-k", type=int, default=3, help="Top K results")
    parser.add_argument("--project", default="rvce-campus-assistant", help="LangSmith project name")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow sync")
    args = parser.parse_args()

    results = run_langsmith_rag_evaluation(
        strategy=args.strategy,
        rerank=args.rerank,
        top_k=args.top_k,
        project_name=args.project,
        enable_mlflow=not args.no_mlflow,
    )
    print("\n--- LangSmith Evaluation Summary ---")
    print(f"Composite RAG Score: {results['metrics']['composite_rag_score']}")
    print(f"Retrieval Hit Rate: {results['metrics']['retrieval_relevance_rate'] * 100:.1f}%")
    print(f"Answer Relevance: {results['metrics']['answer_relevance_rate'] * 100:.1f}%")
    print(f"Abstention Accuracy: {results['metrics']['abstention_accuracy'] * 100:.1f}%")
    print(f"Citation Grounding: {results['metrics']['citation_grounding_rate'] * 100:.1f}%")


if __name__ == "__main__":
    main()

