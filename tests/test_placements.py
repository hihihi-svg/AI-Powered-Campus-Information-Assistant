"""Unit, integration, and regression tests for RVCE 2027 placement records integration.

Tests cover:
1. Scraper normalization (branches, companies, CTC, status, location).
2. Processed dataset integrity (403 records, 376 final, 27 non-final, disclaimer presence).
3. Quantitative calculations (highest CTC, average CTC, branch comparisons, top recruiters).
4. Placement query answering engine (structured responses with disclaimer).
5. Non-placement query regression (KCET fees, hostel fees, exam notices).
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.scrapers.rvce_placements_scraper import (
    DISCLAIMER_TEXT,
    deduplicate_records,
    normalize_branch,
    normalize_category,
    normalize_company,
    normalize_ctc,
    normalize_location,
    normalize_record,
    normalize_status,
)
from src.placement_analytics import (
    answer_quantitative_placement_query,
    compute_placement_metrics,
    get_top_recruiters,
    is_placement_query,
    load_placement_records,
    parse_query_branch,
    parse_query_branches_for_comparison,
    search_placement_records,
)
from src.answering import answer_question

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATASET_PATH = REPO_ROOT / "data" / "processed" / "rvce_placements_2027.jsonl"


# =============================================================================
# 1. Scraper Normalization Tests
# =============================================================================

def test_normalize_branch():
    assert normalize_branch("CSE") == "CS"
    assert normalize_branch("cs") == "CS"
    assert normalize_branch("Computer Science") == "CS"
    assert normalize_branch("ISE") == "ISE"
    assert normalize_branch("IS") == "ISE"
    assert normalize_branch("Information Science") == "ISE"
    assert normalize_branch("AIML") == "AIML"
    assert normalize_branch("AI & ML") == "AIML"
    assert normalize_branch("CD") == "CD"
    assert normalize_branch("CSD") == "CD"
    assert normalize_branch("CY") == "CY"
    assert normalize_branch("Cyber Security") == "CY"
    assert normalize_branch("") == "UNKNOWN"
    assert normalize_branch(None) == "UNKNOWN"


def test_normalize_company():
    assert normalize_company("arm") == "ARM"
    assert normalize_company("hpe") == "HPE"
    assert normalize_company("mercedes-benz") == "Mercedes-Benz"
    assert normalize_company("cisco") == "Cisco"
    assert normalize_company("  Auxia  ") == "Auxia"
    assert normalize_company(None) == "Unknown Company"


def test_normalize_ctc():
    val, raw = normalize_ctc(40.0)
    assert val == 40.0
    assert raw == "40.00 LPA"

    val, raw = normalize_ctc("16.9")
    assert val == 16.9
    assert raw == "16.90 LPA"

    val, raw = normalize_ctc(None)
    assert val is None
    assert raw is None

    val, raw = normalize_ctc("invalid")
    assert val is None
    assert raw == "invalid"


def test_normalize_status():
    assert normalize_status("final") == "final"
    assert normalize_status("Final") == "final"
    assert normalize_status("confirmed") == "final"
    assert normalize_status("non_final") == "non-final"
    assert normalize_status("non-final") == "non-final"
    assert normalize_status(None) == "non-final"


def test_normalize_location():
    assert normalize_location("bengaluru") == "Bengaluru"
    assert normalize_location("Bangalore") == "Bengaluru"
    assert normalize_location("pune, gurugram") == "Pune, Gurugram"
    assert normalize_location(None) == "Unknown Location"


def test_normalize_record():
    raw = {
        "id": "test-uuid-001",
        "company": "auxia",
        "branch": "cd",
        "status": "final",
        "ctc_lpa": 40.0,
        "base_lpa": 21.0,
        "stipend_kpm": 103.0,
        "offer_type": "Intern + FTE (PPO)",
        "dream_category": "Open Dream",
        "location": "bengaluru",
        "batch": 2027,
    }
    rec = normalize_record(raw, scraped_at="2026-11-10T00:00:00Z")
    assert rec["id"] == "test-uuid-001"
    assert rec["company_name"] == "Auxia"
    assert rec["branch"] == "CD"
    assert rec["offer_status"] == "final"
    assert rec["ctc"] == 40.0
    assert rec["location_or_city"] == "Bengaluru"
    assert rec["disclaimer"] == DISCLAIMER_TEXT
    assert rec["offer_year"] == 2027


def test_deduplicate_records():
    records = [
        {"id": "rec-1", "company": "A"},
        {"id": "rec-2", "company": "B"},
        {"id": "rec-1", "company": "A (duplicate)"},
    ]
    deduped = deduplicate_records(records)
    assert len(deduped) == 2
    assert [r["id"] for r in deduped] == ["rec-1", "rec-2"]


# =============================================================================
# 2. Processed Dataset Integrity Tests
# =============================================================================

def test_processed_dataset_integrity():
    assert PROCESSED_DATASET_PATH.exists(), f"Processed dataset missing at {PROCESSED_DATASET_PATH}"
    records = load_placement_records(PROCESSED_DATASET_PATH, force_reload=True)

    assert len(records) == 403, f"Expected 403 records, got {len(records)}"

    final_count = sum(1 for r in records if r["offer_status"] == "final")
    non_final_count = sum(1 for r in records if r["offer_status"] == "non-final")

    assert final_count == 376, f"Expected 376 final offers, got {final_count}"
    assert non_final_count == 27, f"Expected 27 non-final offers, got {non_final_count}"

    canonical_branches = {"AIML", "CD", "CS", "CY", "ISE"}
    for r in records:
        assert r["id"], "Record must have an id"
        assert r["company_name"], "Record must have a company_name"
        assert r["branch"] in canonical_branches, f"Unexpected branch: {r['branch']}"
        assert r["offer_status"] in {"final", "non-final"}
        assert r["offer_year"] == 2027
        assert r["disclaimer"] == DISCLAIMER_TEXT
        assert r["source_url"] == "https://rvce-placements-2027.pages.dev/offers"


# =============================================================================
# 3. Quantitative Aggregation & Analytics Tests
# =============================================================================

def test_compute_placement_metrics_overall():
    metrics = compute_placement_metrics()
    assert metrics["total_offers"] == 403
    assert metrics["final_offers"] == 376
    assert metrics["non_final_offers"] == 27
    assert metrics["highest_ctc"] == 140.0
    assert metrics["average_ctc"] == 22.32
    assert metrics["companies_count"] in {101, 102}
    assert set(metrics["branch_breakdown"].keys()) == {"AIML", "CD", "CS", "CY", "ISE"}


def test_compute_placement_metrics_by_branch():
    cs_metrics = compute_placement_metrics(branch="CS")
    assert cs_metrics["total_offers"] == 180
    assert cs_metrics["final_offers"] == 172
    assert cs_metrics["non_final_offers"] == 8
    assert cs_metrics["highest_ctc"] == 140.0
    assert cs_metrics["average_ctc"] == 22.84

    aiml_metrics = compute_placement_metrics(branch="AIML")
    assert aiml_metrics["total_offers"] == 47
    assert aiml_metrics["highest_ctc"] == 101.0
    assert aiml_metrics["average_ctc"] == 22.73


def test_top_recruiters():
    top = get_top_recruiters(limit=5)
    assert len(top) == 5
    assert all("company" in item and "offers" in item and "highest_ctc" in item for item in top)
    # Offers should be in descending order
    assert top[0]["offers"] >= top[1]["offers"]


# =============================================================================
# 4. Query Parsing & Intent Handling Tests
# =============================================================================

def test_query_parsing():
    assert parse_query_branch("What is the average package in CSE?") == "CS"
    assert parse_query_branch("Show me offers for Information Science") == "ISE"
    assert parse_query_branch("Tell me about AIML placements") == "AIML"
    assert parse_query_branch("What is the highest package in 2027?") is None

    branches = parse_query_branches_for_comparison("Compare CS and AIML placement stats")
    assert "CS" in branches
    assert "AIML" in branches


def test_is_placement_query():
    assert is_placement_query("What is the highest package in 2027?") is True
    assert is_placement_query("Average CTC for CS branch") is True
    assert is_placement_query("Who are the top recruiters?") is True
    assert is_placement_query("What are the fees for KCET quota students?") is False
    assert is_placement_query("What is the hostel mess fee?") is False


# =============================================================================
# 5. Question Answering Integration Tests
# =============================================================================

def test_highest_package_answer():
    resp = answer_question("What is the highest placement package in 2027?")
    assert resp["strategy"] == "structured_placement_analytics"
    assert "140.0 LPA" in resp["answer"]
    assert "linkedin" in resp["answer"].lower()
    assert DISCLAIMER_TEXT in resp["answer"]
    assert resp["sources"][0]["url"] == "https://rvce-placements-2027.pages.dev/offers"


def test_average_package_answer():
    resp = answer_question("What is the average CTC for CS branch in 2027?")
    assert resp["strategy"] == "structured_placement_analytics"
    assert "22.84 LPA" in resp["answer"]
    assert "180" in resp["answer"]
    assert DISCLAIMER_TEXT in resp["answer"]


def test_final_vs_non_final_answer():
    resp = answer_question("How many final vs non-final offers are there in 2027?")
    assert resp["strategy"] == "structured_placement_analytics"
    assert "376" in resp["answer"]
    assert "27" in resp["answer"]
    assert DISCLAIMER_TEXT in resp["answer"]


def test_branch_comparison_answer():
    resp = answer_question("Compare CS and AIML placements in 2027")
    assert resp["strategy"] == "structured_placement_analytics"
    assert "CS" in resp["answer"]
    assert "AIML" in resp["answer"]
    assert "22.84" in resp["answer"]
    assert "22.73" in resp["answer"]
    assert DISCLAIMER_TEXT in resp["answer"]


def test_salary_filter_answer():
    resp = answer_question("Which companies offered more than 50 LPA in 2027?")
    assert resp["strategy"] == "structured_placement_analytics"
    assert "LPA" in resp["answer"]
    assert DISCLAIMER_TEXT in resp["answer"]


def test_specific_company_answer():
    resp = answer_question("What offer did Auxia make in 2027 placements?")
    assert resp["strategy"] == "structured_placement_analytics"
    assert "Auxia" in resp["answer"]
    assert "40.0 LPA" in resp["answer"]
    assert "CD" in resp["answer"]
    assert DISCLAIMER_TEXT in resp["answer"]


def test_placement_search_records():
    chunks = search_placement_records("Auxia Bengaluru CD", top_k=3)
    assert len(chunks) >= 1
    assert "Auxia" in chunks[0]["text"]
    assert chunks[0]["topic"] == "placements_careers"
    assert DISCLAIMER_TEXT in chunks[0]["disclaimer"]


# =============================================================================
# 6. Non-Placement Regression Tests (Admissions, Fees, Exams)
# =============================================================================

def test_regression_admissions_fee_query():
    """Ensure existing admissions and fee answering pipelines are not broken or intercepted."""
    resp = answer_question("What are the fees for KCET quota students?", strategy="tfidf")
    assert resp["strategy"] in {"tfidf", "hybrid"}
    assert "Tuition" in resp["answer"] or "tuition" in resp["answer"].lower()
    assert len(resp["evidence"]) > 0
    # Must NOT be intercepted by placement analytics
    assert resp.get("is_placement_analytic") is not True


def test_regression_hostel_query():
    """Ensure hostel fee query is routed to campus knowledge base, not placements."""
    resp = answer_question("What is the hostel room rent at RVCE?", strategy="tfidf")
    assert resp["strategy"] in {"tfidf", "hybrid"}
    assert resp.get("is_placement_analytic") is not True
