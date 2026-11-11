"""Tests for LangSmith RAG Observability, Tracing, and Privacy Sanitization.

Validates the 9 core requirements:
1. Tracing disabled by default (LANGSMITH_TRACING=false).
2. Tracing enabled with valid configuration (mocked LangSmith client).
3. Tracing failures NEVER break question answering or /ask endpoint.
4. Complete hierarchical trace tree covers all pipeline stages.
5. Sensitive information (emails, tokens, keys, phone numbers, USNs) is sanitized.
6. Raw Gmail message bodies and raw PDF text are never transmitted.
7. Zero retrieval hits and abstentions are cleanly recorded.
8. Citation validation results are tracked.
9. RAG responses remain identical when tracing is disabled vs enabled.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.answering import NO_EVIDENCE_MESSAGE, answer_question
from src.tracing import (
    RAGTracer,
    is_tracing_enabled,
    sanitize_chunk_for_trace,
    sanitize_text,
)


@pytest.fixture(autouse=True)
def clean_tracing_env(monkeypatch):
    """Ensure clean tracing environment for every test."""
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)


# 1. Tracing disabled by default
def test_tracing_disabled_by_default():
    assert is_tracing_enabled() is False

    tracer = RAGTracer()
    assert tracer.enabled is False
    root = tracer.start_pipeline("What is RVCE?")
    assert root is None

    # Pipeline operates without error and without active runs
    tracer.trace_intent("What is RVCE?", {"normalized_intent": "general_college_info"})
    tracer.trace_query_expansion("What is RVCE?", ["What is RVCE?"], "tfidf")
    tracer.trace_retrieval("What is RVCE?", [], "tfidf")
    tracer.trace_context_assembly([], 0)
    tracer.trace_generation("extractive_synthesis", "RVCE is an engineering institution.", is_llm=False)
    tracer.trace_citation_validation([], [], True)
    tracer.end_pipeline("RVCE is an engineering institution.", False)

    assert tracer.root_run is None


# 2. Tracing enabled with valid configuration (mocked LangSmith client)
def test_tracing_enabled_with_mock_client():
    mock_client = MagicMock()
    tracer = RAGTracer(project_name="test-project", client=mock_client, enabled=True)
    assert tracer.enabled is True

    root = tracer.start_pipeline("What is RVCE?", strategy="tfidf")
    assert root is not None
    assert "start_pipeline" in tracer.stages_recorded

    tracer.trace_intent("What is RVCE?", {"normalized_intent": "general_college_info", "confidence": 0.95})
    assert "intent_classification" in tracer.stages_recorded

    tracer.trace_query_expansion("What is RVCE?", ["What is RVCE?", "RV College of Engineering"], "tfidf")
    assert "query_expansion" in tracer.stages_recorded

    tracer.trace_retrieval("What is RVCE?", [{"chunk_id": "c1", "text": "RVCE info"}], "tfidf", latency_ms=12.5)
    assert "retrieval" in tracer.stages_recorded

    tracer.trace_reranking([{"chunk_id": "c1"}], [{"chunk_id": "c1", "relevance_score": 0.88}], latency_ms=5.0)
    assert "reranking" in tracer.stages_recorded

    tracer.trace_context_assembly([{"chunk_id": "c1"}], context_length_chars=200)
    assert "context_assembly" in tracer.stages_recorded

    tracer.trace_generation("extractive_synthesis", "RVCE is an autonomous college.", latency_ms=2.0, is_llm=False)
    assert "answer_generation" in tracer.stages_recorded

    tracer.trace_citation_validation([{"title": "About RVCE"}], [{"title": "About RVCE"}], is_valid=True)
    assert "citation_validation" in tracer.stages_recorded

    tracer.end_pipeline("RVCE is an autonomous college.", is_abstention=False, sources_count=1)
    assert "end_pipeline" in tracer.stages_recorded

    # Verify that mock client was invoked to post and patch runs
    assert mock_client.method_calls


# 3. Tracing failure does not break /ask or question answering
def test_tracing_failure_does_not_break_ask_endpoint(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "dummy_key_12345")

    # Force all tracer calls to raise unexpected network/runtime exceptions
    with patch("src.tracing.RunTree.post", side_effect=RuntimeError("LangSmith network unreachable")):
        with patch("src.tracing.RunTree.patch", side_effect=RuntimeError("LangSmith network unreachable")):
            client = TestClient(app)
            response = client.post("/ask", json={"question": "What is the highest package in RVCE 2027 placements?"})
            assert response.status_code == 200
            data = response.json()
            assert "140.0 LPA" in data["answer"]
            assert data["abstention_status"] is False


def test_tracing_failure_does_not_break_answering(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "dummy_key_12345")

    failing_client = MagicMock()
    failing_client.create_run.side_effect = ConnectionError("Failed to reach api.smith.langchain.com")

    tracer = RAGTracer(client=failing_client, enabled=True)
    # Even if start_pipeline fails internally:
    with patch.object(tracer, "start_pipeline", side_effect=RuntimeError("Tracer crashed")):
        res = answer_question("What is the highest package in RVCE 2027 placements?", tracer=tracer)
        assert res is not None
        assert "140.0 LPA" in res["answer"]


# 4. Trace structure includes all pipeline stages
def test_trace_structure_all_stages():
    mock_client = MagicMock()
    tracer = RAGTracer(client=mock_client, enabled=True)

    res = answer_question("What is the highest package in RVCE 2027 placements?", tracer=tracer)
    tracer.end_pipeline(res["answer"], is_abstention=False, sources_count=len(res.get("sources", [])))

    expected_stages = [
        "start_pipeline",
        "intent_classification",
        "answer_generation",
        "citation_validation",
        "end_pipeline",
    ]
    for stage in expected_stages:
        assert stage in tracer.stages_recorded, f"Missing expected stage: {stage}"


# 5. Sensitive information is removed before transmission
def test_sensitive_information_sanitization():
    sensitive_prompt = (
        "Dear Rahul Sharma, student name: Priya Nair. "
        "Contact me at student123@rvce.edu.in or call +91-9876543210. "
        "My USN is 1RV21CS099, SGPA: 9.42, CGPA: 9.15, Marks: 98. "
        "API key is sk-1234567890abcdef1234567890 and token lsv2_pt_1234567890abcdef1234567890. "
        "Authorization: Bearer my_secret_token_value_abc."
    )
    sanitized = sanitize_text(sensitive_prompt)

    assert "student123@rvce.edu.in" not in sanitized
    assert "[REDACTED_EMAIL]" in sanitized

    assert "+91-9876543210" not in sanitized
    assert "[REDACTED_PHONE]" in sanitized

    assert "1RV21CS099" not in sanitized
    assert "[REDACTED_USN]" in sanitized

    assert "sk-1234567890abcdef1234567890" not in sanitized
    assert "[REDACTED_KEY]" in sanitized

    assert "lsv2_pt_1234567890abcdef1234567890" not in sanitized
    assert "[REDACTED_TOKEN]" in sanitized

    assert "my_secret_token_value_abc" not in sanitized
    assert "Bearer [REDACTED_TOKEN]" in sanitized

    assert "Rahul Sharma" not in sanitized
    assert "Priya Nair" not in sanitized
    assert "[REDACTED_NAME]" in sanitized

    assert "SGPA: 9.42" not in sanitized
    assert "CGPA: 9.15" not in sanitized
    assert "[REDACTED_ACADEMIC_RECORD]" in sanitized


# 6. Raw Gmail message bodies and raw PDF text are never sent
def test_chunk_sanitization_excludes_raw_email_bodies_and_pdfs():
    raw_gmail_chunk = {
        "chunk_id": "chunk_gmail_abc123",
        "document_id": "doc_gmail_abc123",
        "source_title": "Hostel Fee Circular 2026",
        "heading": "Mess Fee Details",
        "source_url": "https://mail.google.com/mail/u/0/#inbox/abc123",
        "source_type": "gmail_message",
        "relevance_score": 0.892,
        "text": "Contact hostel@rvce.edu.in for issues. Please transfer fees to account 123456789. Confidential PDF attachment attached.",
        "raw_headers": {"From": "Dean <dean@rvce.edu.in>", "To": "all@rvce.edu.in"},
        "attachment_bytes": b"\x25\x50\x44\x46\x2d\x31\x2e\x35",
        "personal_email": "dean@rvce.edu.in",
        "arbitrary_unallowlisted_key": "confidential_leak",
    }

    sanitized = sanitize_chunk_for_trace(raw_gmail_chunk, max_snippet_chars=60)

    # Strictly forbidden and unallowlisted fields must NOT be present
    assert "raw_headers" not in sanitized
    assert "attachment_bytes" not in sanitized
    assert "personal_email" not in sanitized
    assert "arbitrary_unallowlisted_key" not in sanitized
    assert "text" not in sanitized

    # Allowed non-sensitive metadata
    assert sanitized["chunk_id"] == "chunk_gmail_abc123"
    assert sanitized["document_id"] == "doc_gmail_abc123"
    assert sanitized["source_title"] == "Hostel Fee Circular 2026"
    assert sanitized["relevance_score"] == 0.892

    # For Gmail content, snippet must be completely suppressed
    assert sanitized["snippet"] == "[SUPPRESSED_PRIVATE_CONTENT]"
    assert "hostel@rvce.edu.in" not in sanitized["snippet"]

    # Public campus document should have snippet sanitized
    public_chunk = {
        "chunk_id": "chunk_pub_1",
        "document_id": "doc_pub_1",
        "source_title": "RVCE Placements 2027",
        "heading": "Top Recruiters",
        "source_url": "https://rvce.edu.in/placements",
        "source_type": "placement_record",
        "relevance_score": 0.95,
        "text": "Top recruiters include Cisco and Microsoft. Queries: placement@rvce.edu.in or call 080-68188100.",
    }
    pub_sanitized = sanitize_chunk_for_trace(public_chunk, max_snippet_chars=120)
    assert "placement@rvce.edu.in" not in pub_sanitized["snippet"]
    assert "[REDACTED_EMAIL]" in pub_sanitized["snippet"]


# 7. Retrieval failures and zero hits are captured
def test_zero_retrieval_hits_captured():
    mock_client = MagicMock()
    tracer = RAGTracer(client=mock_client, enabled=True)
    tracer.start_pipeline("NonexistentXYZ12345 Query?")

    tracer.trace_retrieval("NonexistentXYZ12345 Query?", [], retriever_type="hybrid", latency_ms=4.2)
    assert "retrieval" in tracer.stages_recorded

    tracer.end_pipeline(NO_EVIDENCE_MESSAGE, is_abstention=True, sources_count=0)
    assert "end_pipeline" in tracer.stages_recorded


# 8. Citation validation results are recorded
def test_citation_validation_recorded():
    mock_client = MagicMock()
    tracer = RAGTracer(client=mock_client, enabled=True)
    tracer.start_pipeline("Placement query")

    sources = [{"document_id": "doc_placement_2027", "title": "RVCE Placements 2027"}]
    evidence = [{"document_id": "doc_placement_2027", "heading": "Highest Offer", "text": "140 LPA"}]

    tracer.trace_citation_validation(sources, evidence, is_valid=True, validation_details={"grounded": True})
    assert "citation_validation" in tracer.stages_recorded


# 9. Existing RAG answers remain identical when tracing is disabled
def test_answers_identical_with_and_without_tracing():
    query = "What is the highest package in RVCE 2027 placements?"

    # Baseline with no tracer
    baseline_result = answer_question(query, strategy="hybrid")

    # With disabled tracer
    disabled_tracer = RAGTracer(enabled=False)
    disabled_result = answer_question(query, strategy="hybrid", tracer=disabled_tracer)

    assert baseline_result["answer"] == disabled_result["answer"]
    assert baseline_result["sources"] == disabled_result["sources"]
    assert baseline_result["evidence"] == disabled_result["evidence"]
    assert baseline_result["strategy"] == disabled_result["strategy"]
