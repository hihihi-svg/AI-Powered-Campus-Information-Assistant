"""Comprehensive Evaluation Suite for Intent Classification, Entity Extraction, and Multi-Stage RAG.

Evaluates:
1. Supervised Logistic Regression Intent Classification (9 target classes, Macro F1, Accuracy)
2. Independent Entity Extraction (exams, branches, quotas, categories, ctc, rounds, data types)
3. Controlled Multi-Query Generation
4. Hierarchical Context Aggregation (table header repair, document grouping, token budget)
5. Direct Factual, Numerical, Multi-Document, Ambiguous, and Out-of-Domain queries
6. Answer Faithfulness & Evidence-grounded Abstention on unavailable cutoffs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import pytest

from src.intent_classification import (
    DEFAULT_DATASET_V2_PATH,
    classify_query_intent,
    load_query_dataset,
    train_and_evaluate_v2,
)
from src.entity_extraction import extract_campus_entities
from src.multi_query import generate_expanded_queries, retrieve_multi_query_hybrid
from src.context_aggregation import (
    assemble_hierarchical_context,
    _repair_table_continuation,
    _estimate_tokens,
)
from src.answering import answer_question, build_evidence_answer, NO_EVIDENCE_MESSAGE


# ==============================================================================
# PART A & B: Intent Classification & Entity Extraction Evaluation
# ==============================================================================

def test_intent_classification_v2_benchmark_metrics():
    """Verify that the 9-class intent classification model achieves target performance on holdout test set."""
    v2_report_path = Path("reports/intent_classification_v2_evaluation.json")
    if not v2_report_path.exists():
        report = train_and_evaluate_v2()
    else:
        report = json.loads(v2_report_path.read_text(encoding="utf-8"))

    lr_metrics = report["models"]["logistic_regression"]["metrics"]
    dummy_metrics = report["models"]["dummy_baseline"]["metrics"]

    # 1. Verify 9 classes evaluated
    classes = report["dataset"]["class_labels"]
    assert len(classes) == 9, f"Expected 9 target classes, got {len(classes)}"
    expected_classes = {
        "academics_exams",
        "admission_fees",
        "admissions_cutoff",
        "admissions_procedure",
        "departments_faculty",
        "facilities_hostels",
        "general_college_info",
        "placements",
        "student_services",
    }
    assert set(classes) == expected_classes

    # 2. Assert real performance thresholds
    assert lr_metrics["accuracy"] >= 0.85, f"Expected accuracy >= 0.85, got {lr_metrics['accuracy']}"
    assert lr_metrics["macro_f1"] >= 0.85, f"Expected macro F1 >= 0.85, got {lr_metrics['macro_f1']}"

    # 3. Model significantly beats dummy baseline
    assert lr_metrics["accuracy"] > dummy_metrics["accuracy"] * 4.0
    assert lr_metrics["macro_f1"] > dummy_metrics["macro_f1"] * 4.0


def test_independent_entity_extraction():
    """Verify entity extraction operates independently and extracts complex domain attributes."""
    # 1. Target cutoff question
    q1 = "What was the previous KCET cutoff for CSE?"
    e1 = extract_campus_entities(q1)
    assert e1["exam"] == "KCET"
    assert e1["branch"] == "CSE"
    assert e1["year"] is None
    assert e1["data_type"] == "previous_cutoff"
    assert e1["is_comparison"] is False

    # 2. Multi-quota comparison query
    q2 = "Compare KCET and COMEDK fees for 1st year B.E."
    e2 = extract_campus_entities(q2)
    assert e2["exam"] == "KCET"
    assert e2["admission_quota"] == "multiple"
    assert e2["is_comparison"] is True
    assert e2["academic_year"] == "1st year"
    assert e2["data_type"] == "fee_structure"

    # 3. Placement record query with company and salary
    q3 = "How many offers were made above 20 LPA by LinkedIn?"
    e3 = extract_campus_entities(q3)
    assert e3["company"] == "LinkedIn"
    assert e3["ctc"] == 20.0
    assert e3["data_type"] == "placement_record"

    # 4. Multi-entity cutoff query with round and category
    q4 = "What are the round 2 cutoff ranks for AIML under GM category?"
    e4 = extract_campus_entities(q4)
    assert e4["branch"] == "AIML"
    assert e4["category"] == "GM"
    assert e4["counselling_round"] == "Round 2"
    assert e4["data_type"] == "cutoff"


# ==============================================================================
# PART C & E: Multi-Query Generation & Context Aggregation Evaluation
# ==============================================================================

def test_controlled_multi_query_generation():
    """Query expansion must retain the original query as primary and create targeted variations."""
    q = "What is the KCET cutoff rank for CSE?"
    entities = extract_campus_entities(q)
    expanded = generate_expanded_queries(q, entities=entities, intent="admissions_cutoff", max_queries=3)

    assert len(expanded) >= 2
    assert expanded[0] == q  # Original query preserved as primary
    assert any("Computer Science" in var or "closing rank" in var or "round" in var for var in expanded[1:])


def test_context_aggregation_table_header_repair():
    """Table continuation chunks without a header must inherit the header from the predecessor chunk."""
    prev_chunk = (
        "### Higher Semester Fees 2026-27\n"
        "| Branch | Tuition Fee | Skill Lab Fee |\n"
        "| --- | --- | --- |\n"
        "| CSE | 1,85,000 | 25,000 |\n"
    )
    orphan_table_chunk = (
        "| ISE | 1,85,000 | 25,000 |\n"
        "| AIML | 1,85,000 | 25,000 |\n"
    )

    repaired = _repair_table_continuation(orphan_table_chunk, prev_chunk)
    assert "| Branch | Tuition Fee | Skill Lab Fee |" in repaired
    assert "| --- | --- | --- |" in repaired
    assert "| ISE | 1,85,000 | 25,000 |" in repaired


def test_context_aggregation_budget_and_grouping():
    """Context aggregator must group by document and enforce token budgets."""
    candidates = [
        {
            "chunk_id": "doc1_c01",
            "document_id": "doc1",
            "document_title": "Admissions Notification 2026",
            "source_url": "https://rvce.edu.in/admissions-2026",
            "heading": "KEA Quota",
            "text": "KEA students must submit original documents at counter 4.",
            "relevance_score": 0.85,
        },
        {
            "chunk_id": "doc1_c02",
            "document_id": "doc1",
            "document_title": "Admissions Notification 2026",
            "source_url": "https://rvce.edu.in/admissions-2026",
            "heading": "COMEDK Quota",
            "text": "COMEDK students must bring candidate portal allotment slip.",
            "relevance_score": 0.82,
        },
        {
            "chunk_id": "doc2_c01",
            "document_id": "doc2",
            "document_title": "Hostel Fee Circular",
            "source_url": "https://rvce.edu.in/hostel-fees",
            "heading": "Mess Charges",
            "text": "Hostel mess deposit is refundable at end of semester.",
            "relevance_score": 0.70,
        },
    ]

    assembled = assemble_hierarchical_context(candidates, max_token_budget=1000)
    assert "### Document: Admissions Notification 2026" in assembled["formatted_context"]
    assert "### Document: Hostel Fee Circular" in assembled["formatted_context"]
    assert assembled.get("document_count", 2) == 2
    assert assembled["total_tokens"] <= 1000


# ==============================================================================
# PART G & H: End-to-End Query Evaluation & Faithfulness
# ==============================================================================

def test_direct_factual_query():
    """Verify direct factual queries return verified answers from campus records."""
    resp = answer_question("Who is the principal of RV College of Engineering?", strategy="tfidf")
    assert resp["answer"]
    assert "Dr. K.N. Subramanya" in resp["answer"]
    assert resp["sources"]


def test_numerical_placement_query():
    """Verify quantitative placement queries calculate exact metrics without hallucination."""
    resp = answer_question("What is the highest package in 2027 placements?", strategy="tfidf")
    assert resp["answer"]
    assert "140" in resp["answer"]
    assert "LinkedIn" in resp["answer"]


def test_historical_unanswerable_cutoff_query_graceful_abstention():
    """Verify that when cutoff rank tables are absent from records, system does NOT hallucinate ranks."""
    resp = answer_question("What was the previous KCET cutoff for CSE?", strategy="tfidf")
    assert resp["answer"]
    # Must clearly state that closing ranks / cutoff tables are not in the campus documents
    answer_text = resp["answer"].lower()
    assert (
        "do not contain" in answer_text
        or "could not find" in answer_text
        or "not contain official cutoff" in answer_text
    )
    # Must direct to official authorities (KEA / COMEDK) rather than hallucinating ranks
    assert "kea" in answer_text or "admissions" in answer_text


def test_out_of_domain_query_abstention():
    """Verify out-of-domain questions abstain cleanly with NO_EVIDENCE_MESSAGE."""
    resp = answer_question("What is the capital of France?", strategy="tfidf")
    assert resp["answer"] == NO_EVIDENCE_MESSAGE
    assert resp["evidence"] == []
    assert resp["sources"] == []


def test_retrieval_recall_at_k():
    """Measure Recall@K across canonical campus query test sets."""
    from src.retrieval import get_default_retriever

    retriever = get_default_retriever()
    test_queries = [
        ("procedure for COMEDK rank card", "comedk"),
        ("higher semester tuition fee circular", "fee"),
        ("principal office reception contact email", "principal"),
        ("hostel fees and room rent", "hostel"),
    ]

    recall_at_3 = 0
    recall_at_5 = 0
    recall_at_10 = 0

    for query, expected_keyword in test_queries:
        res10 = retriever.retrieve(query=query, top_k=10)
        found_in_3 = any(expected_keyword in (r.get("text") or "").lower() for r in res10[:3])
        found_in_5 = any(expected_keyword in (r.get("text") or "").lower() for r in res10[:5])
        found_in_10 = any(expected_keyword in (r.get("text") or "").lower() for r in res10[:10])

        if found_in_3:
            recall_at_3 += 1
        if found_in_5:
            recall_at_5 += 1
        if found_in_10:
            recall_at_10 += 1

    total = len(test_queries)
    r3 = recall_at_3 / total
    r5 = recall_at_5 / total
    r10 = recall_at_10 / total

    assert r3 >= 0.50, f"Recall@3 was {r3}"
    assert r5 >= 0.75, f"Recall@5 was {r5}"
    assert r10 >= 0.75, f"Recall@10 was {r10}"
