"""Comprehensive end-to-end tests for scenarios A through L of the RVCE Campus Assistant RAG system.

Scenarios tested:
  A. Latest internship email -> latest relevant email retrieved and its date shown.
  B. Notices from a specific month (e.g., October 2026) -> only relevant notices from that period considered.
  C. General campus question -> proper synthesized explanation.
  D. Follow-up question -> previous conversational context preserved.
  E. New Gmail email arrives -> indexed -> retrieved without restarting the application.
  F. Email with attachment -> attachment content and parent email context both available.
  G. Duplicate email sync -> no duplicate chunks added.
  H. Provider quota exhaustion -> clean UI/API error, no raw API error exposed.
  I. MCP operational workflow & MCP failure -> correctly logged and safely handled.
  J. Newer notice supersedes an older one -> answer prioritizes the newer relevant notice.
  K. Missing date metadata -> system does not incorrectly rank the document as latest.
  L. Date sorting -> chronological ordering is correct.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import os
import sys
import types
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pytest

from src.answering import answer_question, build_evidence_answer
from src.api import app
from src.generation import CLEAN_GENERATION_ERROR_MESSAGE, clear_generation_cache, generate_answer
from src.gmail_ingestion import GmailSyncWorker, save_ingestion_state
from src.gmail_parser import parse_email_message
from src.mcp_server import CampusGmailMCPClient, CampusGmailMCPServer
from src.retrieval import TFIDFRetriever, _DEFAULT_RETRIEVER_CACHE, get_default_retriever
from src.temporal import (
    filter_and_rank_by_date,
    normalize_iso_date,
    parse_temporal_query,
    sort_results_chronologically,
)


@pytest.fixture
def multi_date_corpus_records() -> List[Dict[str, Any]]:
    """Fixture corpus with dated emails, attachments, superseding circulars, and an undated doc."""
    return [
        {
            "document_id": "doc_email_intern_old",
            "source_url": "gmail://messages/msg_intern_sep",
            "title": "Summer Internship Opportunity at Bosch for Pre-Final Year Students",
            "source_title": "Summer Internship Opportunity at Bosch for Pre-Final Year Students",
            "content": (
                "The Placement and Training cell invites applications for the Bosch Summer Internship. "
                "Eligible branches: CSE, ISE, ECE. Stipend: Rs. 25,000 per month. "
                "Application deadline: 20 September 2026."
            ),
            "source_type": "EMAIL",
            "sender": "placement@rvce.edu.in",
            "sender_name": "Placement Cell RVCE",
            "gmail_message_id": "msg_intern_sep",
            "received_at": "2026-09-10T09:00:00+00:00",
            "document_date": "2026-09-10T09:00:00+00:00",
            "last_modified": "2026-09-10T09:00:00+00:00",
            "event_dates": ["2026-09-20T00:00:00+00:00"],
            "topic": "placements_career",
            "extraction_status": "SUCCESS",
        },
        {
            "document_id": "doc_email_intern_latest",
            "source_url": "gmail://messages/msg_intern_oct",
            "title": "Winter Internship Recruitment Drive — Google & Qualcomm (2026 Batch)",
            "source_title": "Winter Internship Recruitment Drive — Google & Qualcomm (2026 Batch)",
            "content": (
                "Dear Students, Registration is now open for the Winter Internship Recruitment Drive "
                "with Google and Qualcomm for 7th semester B.E. students (CSE, ISE, AIML, ECE). "
                "Stipend: Rs. 60,000 per month. Registration closes on 5 November 2026 via the placement portal."
            ),
            "source_type": "EMAIL",
            "sender": "placement@rvce.edu.in",
            "sender_name": "Placement Cell RVCE",
            "gmail_message_id": "msg_intern_oct",
            "received_at": "2026-10-29T10:30:00+00:00",
            "document_date": "2026-10-29T10:30:00+00:00",
            "last_modified": "2026-10-29T10:30:00+00:00",
            "event_dates": ["2026-11-05T00:00:00+00:00"],
            "topic": "placements_career",
            "extraction_status": "SUCCESS",
        },
        {
            "document_id": "doc_email_intern_undated",
            "source_url": "https://rvce.edu.in/placement/general-internship-guidelines",
            "title": "General Internship Guidelines for B.E. Programmes",
            "source_title": "General Internship Guidelines for B.E. Programmes",
            "content": (
                "All undergraduate engineering students must complete a mandatory industry internship "
                "and submit an internship report verified by their faculty mentor."
            ),
            "source_type": "HTML",
            "received_at": None,
            "document_date": None,
            "published_at": None,
            "last_modified": None,
            "topic": "placements_career",
            "extraction_status": "SUCCESS",
        },
        {
            "document_id": "doc_email_unrelated_newest",
            "source_url": "gmail://messages/msg_sports_nov",
            "title": "Annual Inter-Departmental Sports Meet Registration",
            "source_title": "Annual Inter-Departmental Sports Meet Registration",
            "content": (
                "All students are invited to register for the Annual Inter-Departmental Cricket and "
                "Badminton tournament on the main sports ground by 12 November 2026."
            ),
            "source_type": "EMAIL",
            "sender": "sports@rvce.edu.in",
            "sender_name": "Sports Department RVCE",
            "gmail_message_id": "msg_sports_nov",
            "received_at": "2026-11-05T14:00:00+00:00",
            "document_date": "2026-11-05T14:00:00+00:00",
            "last_modified": "2026-11-05T14:00:00+00:00",
            "topic": "general_campus_info",
            "extraction_status": "SUCCESS",
        },
        {
            "document_id": "doc_fee_old",
            "source_url": "https://rvce.edu.in/circulars/ug-fee-payment-oct1.pdf",
            "title": "Circular: Higher Semester UG Tuition Fee Payment Schedule",
            "source_title": "Circular: Higher Semester UG Tuition Fee Payment Schedule",
            "content": (
                "All 3rd, 5th, and 7th semester B.E. students are hereby informed that the last date "
                "for payment of higher semester tuition fees without fine is 15.10.2026 via SAP portal."
            ),
            "source_type": "PDF",
            "published_at": "2026-10-01T08:00:00+00:00",
            "document_date": "2026-10-01T08:00:00+00:00",
            "last_modified": "2026-10-01T08:00:00+00:00",
            "event_dates": ["2026-10-15T00:00:00+00:00"],
            "topic": "fees_finance",
            "extraction_status": "SUCCESS",
        },
        {
            "document_id": "doc_fee_revised_new",
            "source_url": "https://rvce.edu.in/circulars/ug-fee-payment-revised-oct18.pdf",
            "title": "Revised Circular: Extension of Higher Semester UG Tuition Fee Payment Deadline",
            "source_title": "Revised Circular: Extension of Higher Semester UG Tuition Fee Payment Deadline",
            "content": (
                "In continuation to the earlier fee circular, the last date for payment of higher "
                "semester 3rd, 5th, and 7th semester B.E. tuition fees is extended up to 30.10.2026 "
                "without penalty via the SAP student portal."
            ),
            "source_type": "PDF",
            "published_at": "2026-10-18T11:00:00+00:00",
            "document_date": "2026-10-18T11:00:00+00:00",
            "last_modified": "2026-10-18T11:00:00+00:00",
            "event_dates": ["2026-10-30T00:00:00+00:00"],
            "topic": "fees_finance",
            "extraction_status": "SUCCESS",
        },
    ]


# ---------------------------------------------------------------------------
# Scenario A: Latest internship email
# ---------------------------------------------------------------------------
def test_scenario_a_latest_internship_email_retrieved_with_date(multi_date_corpus_records):
    """Scenario A: Querying the latest internship email retrieves the Oct 29 email and shows its date."""
    retriever = TFIDFRetriever()
    retriever.index_records(multi_date_corpus_records)

    ref_now = datetime(2026, 11, 8, 12, 0, 0, tzinfo=timezone.utc)
    result = answer_question(
        "Show the latest email regarding internships.",
        retriever=retriever,
        reference_now=ref_now,
    )

    assert result["sources"], "Expected at least one source for latest internship email"
    top_source = result["sources"][0]
    assert top_source["document_id"] == "doc_email_intern_latest"
    assert top_source["document_date"] == "2026-10-29T10:30:00+00:00"
    assert "2026-10-29" in result["answer"] or "5 November 2026" in result["answer"]
    assert "Google" in result["answer"] or "Qualcomm" in result["answer"]
    # Must not return the newer but unrelated sports email
    assert "Cricket" not in result["answer"]


# ---------------------------------------------------------------------------
# Scenario B: Notices from a specific month (October 2026)
# ---------------------------------------------------------------------------
def test_scenario_b_notices_from_specific_month_october_2026(multi_date_corpus_records):
    """Scenario B: Querying notices from October 2026 only considers October 2026 documents."""
    retriever = TFIDFRetriever()
    retriever.index_records(multi_date_corpus_records)

    ref_now = datetime(2026, 11, 8, 12, 0, 0, tzinfo=timezone.utc)
    result = answer_question(
        "What was announced in October 2026?",
        retriever=retriever,
        reference_now=ref_now,
    )

    assert result["sources"], "Expected October 2026 sources"
    for src in result["sources"]:
        doc_date = src.get("document_date") or ""
        assert doc_date.startswith("2026-10-"), f"Expected October 2026 source, got {doc_date}"

    # Querying a month with no documents must accurately report no notices found for that period
    empty_month_res = answer_question(
        "What notices were released in March 2025?",
        retriever=retriever,
        reference_now=ref_now,
    )
    assert "March 2025" in empty_month_res["answer"]
    assert empty_month_res["sources"] == []


# ---------------------------------------------------------------------------
# Scenario C: General campus question -> proper synthesized explanation
# ---------------------------------------------------------------------------
def test_scenario_c_general_campus_question_synthesized_explanation():
    """Scenario C: General campus question produces a coherent, synthesized explanation with key details."""
    records = [
        {
            "document_id": "doc_email_pvp",
            "source_url": "gmail://messages/msg_pvp_1",
            "title": "Registration for SEE Answer Paper Viewing (PVP)",
            "content": (
                "Registration for SEE Answer Paper Viewing (PVP) for UG III to VII semester "
                "(2022 scheme) Fast Track Examinations is open from 29.10.2026 to 01.11.2026 on "
                "the SAP portal. The registration fee is Rs. 500/- per course. Viewing will be "
                "held from 4th to 6th November 2026 at the Controller of Examinations office."
            ),
            "source_type": "EMAIL_ATTACHMENT",
            "sender": "coe@rvce.edu.in",
            "sender_name": "Controller of Examinations RVCE",
            "received_at": "2026-10-28T11:13:04+00:00",
            "document_date": "2026-10-28T11:13:04+00:00",
            "extraction_status": "SUCCESS",
        }
    ]
    retriever = TFIDFRetriever()
    retriever.index_records(records)

    result = answer_question(
        "What is the fee and schedule for SEE answer paper viewing (PVP)?",
        retriever=retriever,
    )
    assert "500" in result["answer"]
    assert "29.10.2026" in result["answer"] or "November 2026" in result["answer"]
    assert result["sources"][0]["source_type"] == "EMAIL_ATTACHMENT"
    assert result["sources"][0]["document_date"] == "2026-10-28T11:13:04+00:00"


# ---------------------------------------------------------------------------
# Scenario D: Follow-up question -> conversational context preserved
# ---------------------------------------------------------------------------
def test_scenario_d_followup_question_preserves_conversational_context():
    """Scenario D: A follow-up question ('What is the registration deadline and venue for it?')
    uses conversation_history to resolve 'it' to the Quantum Computing Workshop."""
    records = [
        {
            "document_id": "doc_email_qc_workshop",
            "source_url": "gmail://messages/msg_qc_01",
            "title": "Circular: Quantum Computing Workshop & Schedule",
            "content": (
                "The Department of CSE is organizing a 3-day Quantum Computing Workshop for 5th "
                "and 7th semester students. The registration deadline is 15 November 2026 and "
                "sessions will be held at the Main Auditorium."
            ),
            "source_type": "EMAIL",
            "sender": "dean.academics@rvce.edu.in",
            "sender_name": "Dean Academics RVCE",
            "received_at": "2026-11-02T09:00:00+00:00",
            "document_date": "2026-11-02T09:00:00+00:00",
            "extraction_status": "SUCCESS",
        },
        {
            "document_id": "doc_html_admissions",
            "source_url": "https://rvce.edu.in/admissions",
            "title": "First Year B.E. Admissions",
            "content": "Candidates joining through KEA must report to the Civil Engineering Department.",
            "source_type": "HTML",
            "document_date": "2026-08-10T00:00:00+00:00",
            "extraction_status": "SUCCESS",
        },
    ]
    retriever = TFIDFRetriever()
    retriever.index_records(records)

    history = [
        {"role": "user", "content": "Tell me about the Quantum Computing Workshop."},
        {
            "role": "assistant",
            "content": "The Department of CSE is organizing a 3-day Quantum Computing Workshop for 5th and 7th semester students.",
        },
    ]

    followup_res = answer_question(
        "What is the registration deadline and venue for it?",
        retriever=retriever,
        conversation_history=history,
    )

    assert "15 November 2026" in followup_res["answer"] or "Main Auditorium" in followup_res["answer"]
    assert followup_res["sources"][0]["document_id"] == "doc_email_qc_workshop"


# ---------------------------------------------------------------------------
# Scenario E & G: New Gmail email arrives -> indexed -> retrieved without restart,
#                and duplicate sync adds zero duplicate chunks
# ---------------------------------------------------------------------------
def test_scenarios_e_and_g_new_email_live_sync_and_duplicate_prevention(tmp_path):
    """Scenario E & G: New email arrives via Gmail sync, is indexed atomically without restart,
    is immediately retrievable, and re-syncing does not create duplicate chunks."""
    state_path = str(tmp_path / "state.json")
    raw_path = str(tmp_path / "raw.jsonl")
    proc_path = str(tmp_path / "proc.jsonl")
    manifest_path = str(tmp_path / "manifest.json")
    report_path = str(tmp_path / "report.json")

    seed_doc = {
        "document_id": "doc_html_seed_1",
        "source_url": "https://rvce.edu.in/admissions",
        "title": "RVCE Admission Overview",
        "content": "Undergraduate admissions are conducted via KEA CET and COMEDK counselling.",
        "source_type": "HTML",
        "document_date": "2026-08-01T00:00:00+00:00",
        "extraction_status": "SUCCESS",
    }
    with open(proc_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(seed_doc) + "\n")

    save_ingestion_state({"processed_message_ids": []}, state_path)

    # Initial retriever before new email arrives
    r_before = get_default_retriever(proc_path)
    assert len(r_before.chunks) == 1

    mock_service = MagicMock()
    mock_service.users().messages().list.return_value.execute.return_value = {
        "messages": [{"id": "msg_live_ai_hackathon"}],
        "nextPageToken": None,
    }
    mock_service.users().messages().get.return_value.execute.return_value = {
        "id": "msg_live_ai_hackathon",
        "internalDate": "1762680000000",
        "snippet": "National AI Hackathon registration open for RVCE students",
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": "Principal RVCE <principal@rvce.edu.in>"},
                {"name": "Subject", "value": "Circular: RVCE National AI Hackathon 2026 Registration"},
                {"name": "Date", "value": "Mon, 09 Nov 2026 10:00:00 +0530"},
            ],
            "body": {
                "data": base64.urlsafe_b64encode(
                    b"All B.E. and MCA students are invited to register for the RVCE National AI Hackathon 2026 "
                    b"before 20 November 2026. Prize pool: Rs. 1,00,000 at IEM Auditorium."
                ).decode("ascii")
            },
        },
    }

    worker = GmailSyncWorker(
        interval_seconds=60,
        state_path=state_path,
        raw_path=raw_path,
        processed_path=proc_path,
        manifest_path=manifest_path,
        report_path=report_path,
        service=mock_service,
    )

    # First sync: new email arrives
    sync1 = worker.sync_once()
    assert sync1["status"] == "ok"
    assert sync1["emails_stored"] == 1
    assert sync1["index_updated"] is True

    # Query without restarting application
    ans = answer_question(
        "When is the RVCE National AI Hackathon registration deadline and what is the prize pool?",
        dataset_path=proc_path,
    )
    assert "20 November 2026" in ans["answer"] or "1,00,000" in ans["answer"]
    assert ans["sources"][0]["source_type"] == "EMAIL"

    chunk_count_after_sync1 = len(get_default_retriever(proc_path).chunks)
    assert chunk_count_after_sync1 == 2

    # Scenario G: Run sync again with the exact same email in Gmail
    sync2 = worker.sync_once()
    assert sync2["status"] == "ok"
    assert sync2["emails_stored"] == 0
    chunk_count_after_sync2 = len(get_default_retriever(proc_path).chunks)
    assert chunk_count_after_sync2 == chunk_count_after_sync1

    _DEFAULT_RETRIEVER_CACHE.pop(proc_path, None)


# ---------------------------------------------------------------------------
# Scenario F: Email with attachment -> attachment content & parent email context both available
# ---------------------------------------------------------------------------
def test_scenario_f_email_with_attachment_preserves_parent_context():
    """Scenario F: Parsing and indexing an email with a PDF attachment links parent email metadata
    (subject, sender, received_at, parent body) onto the attachment chunk."""
    msg_dict = {
        "id": "msg_with_pdf_99",
        "internalDate": "1761649984000",
        "payload": {
            "mimeType": "multipart/mixed",
            "headers": [
                {"name": "Subject", "value": "Internship Stipend & Onboarding Circular"},
                {"name": "From", "value": "Placement Dean <placement@rvce.edu.in>"},
                {"name": "Date", "value": "Wed, 28 Oct 2026 16:43:04 +0530"},
            ],
            "body": {},
            "parts": [
                {
                    "mimeType": "text/plain",
                    "body": {
                        "data": base64.urlsafe_b64encode(
                            b"Dear Students, Please find attached the onboarding schedule for selected interns at Texas Instruments."
                        ).decode("ascii")
                    },
                },
                {
                    "filename": "TI_Onboarding_Schedule.pdf",
                    "mimeType": "application/pdf",
                    "body": {"attachmentId": "att_ti_01", "size": 2048},
                },
            ],
        },
    }

    with patch("src.gmail_parser._pdf_extractor.extract") as mock_pdf:
        mock_pdf.return_value = {
            "title": "TI Onboarding Schedule",
            "content": "Selected interns must report on 12 January 2027 with original NOC and ID card. Stipend: Rs. 55,000/month.",
            "extraction_status": "SUCCESS",
        }
        parent_rec, child_recs = parse_email_message(
            msg_dict,
            attachment_bytes_map={"att_ti_01": b"%PDF-1.4 mock"},
        )

    assert parent_rec is not None
    assert len(child_recs) == 1

    retriever = TFIDFRetriever()
    retriever.index_records([parent_rec] + child_recs)

    ans = answer_question(
        "When must Texas Instruments interns report and what is the stipend?",
        retriever=retriever,
    )
    assert "12 January 2027" in ans["answer"] or "55,000" in ans["answer"]
    # Verify parent email context (sender & received_at) is preserved on attachment evidence/source
    att_sources = [s for s in ans["sources"] if s.get("source_type") == "EMAIL_ATTACHMENT"]
    assert att_sources, "Expected EMAIL_ATTACHMENT in sources"
    assert "placement@rvce.edu.in" in (att_sources[0].get("sender") or "")
    assert att_sources[0].get("received_at") == "2026-10-28T11:13:04+00:00"


# ---------------------------------------------------------------------------
# Scenario H: Provider quota exhaustion -> clean UI error, no raw API error exposed
# ---------------------------------------------------------------------------
def test_scenario_h_provider_quota_exhaustion_returns_clean_error():
    """Scenario H: When OpenAI / Hugging Face fails with quota exhaustion, the API and generate_answer
    must return CLEAN_GENERATION_ERROR_MESSAGE and never expose raw API errors or keys."""
    clear_generation_cache()

    class MockQuotaError(Exception):
        status_code = 429
        code = "credit_balance_exhausted"
        type = "insufficient_quota"

    MockQuotaError.__name__ = "RateLimitError"

    fake_openai = types.ModuleType("openai")
    client_instance = MagicMock()
    client_instance.responses.create.side_effect = MockQuotaError(
        "Error code: 429 - You have no credits remaining on sk-proj-SECRET123456789."
    )
    fake_openai.OpenAI = MagicMock(return_value=client_instance)

    chunks = [
        {
            "chunk_id": "c1",
            "document_id": "doc1",
            "document_title": "RVCE Fee Circular",
            "source_url": "https://rvce.edu.in/fees",
            "heading": "Tuition Fees",
            "text": "The tuition fee for CSE is Rs 1,85,000.",
            "relevance_score": 0.9,
        }
    ]

    with patch.dict(sys.modules, {"openai": fake_openai}):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-proj-SECRET123456789"}):
            with patch("src.api.answer_question") as mock_ans:
                gen_res = generate_answer("What is the CSE fee?", chunks)
                mock_ans.return_value = gen_res

                api_client = TestClient(app)
                resp = api_client.post("/ask", json={"question": "What is the CSE fee?", "strategy": "gpt"})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["answer"] == CLEAN_GENERATION_ERROR_MESSAGE
    assert payload["error"]["message"] == CLEAN_GENERATION_ERROR_MESSAGE
    assert payload["error"]["category"] == "insufficient_quota"

    # Must NEVER expose raw backend/provider strings or secrets
    serialized = json.dumps(payload)
    for forbidden in (
        "RateLimitError",
        "OpenAI",
        "Hugging Face",
        "sk-proj-",
        "You have no credits remaining",
        "https://platform.openai.com",
    ):
        assert forbidden not in serialized, f"Forbidden raw detail '{forbidden}' leaked in API response!"


# ---------------------------------------------------------------------------
# Scenario I: Operational MCP Server/Client & Safe MCP Failure Handling
# ---------------------------------------------------------------------------
def test_scenario_i_mcp_operational_workflow_and_failure_handling(multi_date_corpus_records, tmp_path):
    """Scenario I: Verify MCP server initialization, tool discovery, retrieval tool execution,
    and safe error logging/handling when an MCP tool or Gmail call fails."""
    proc_path = str(tmp_path / "mcp_corpus.jsonl")
    with open(proc_path, "w", encoding="utf-8") as f:
        for rec in multi_date_corpus_records:
            f.write(json.dumps(rec) + "\n")

    mock_gmail = MagicMock()
    mock_gmail.users().messages().list.return_value.execute.return_value = {
        "messages": [{"id": "mcp_msg_1"}],
        "nextPageToken": None,
    }

    server = CampusGmailMCPServer(
        service=mock_gmail,
        processed_path=proc_path,
    )
    client = CampusGmailMCPClient(server=server)

    conn = client.connect()
    assert conn["connected"] is True
    assert "gmail_list_messages" in client.discovered_tools
    assert "gmail_get_message" in client.discovered_tools
    assert "gmail_get_attachment" in client.discovered_tools
    assert "gmail_sync_notices" in client.discovered_tools
    assert "campus_retrieve" in client.discovered_tools

    # Call campus_retrieve via MCP JSON-RPC 2.0
    ret_res = client.call_tool("campus_retrieve", {"query": "Bosch Summer Internship", "top_k": 2})
    assert ret_res["isError"] is False
    assert ret_res["data"]["count"] >= 1

    # Simulate an MCP tool failure (e.g., Gmail API raises RuntimeError)
    mock_gmail.users().messages().get.side_effect = RuntimeError("Simulated Gmail MCP transport failure")
    err_res = client.call_tool("gmail_get_message", {"message_id": "mcp_msg_1"})
    assert err_res["isError"] is True
    assert client.last_error is not None

    # Verify /mcp/status FastAPI endpoint works cleanly
    api_client = TestClient(app)
    status_resp = api_client.get("/mcp/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["mcp_connected"] is True

    _DEFAULT_RETRIEVER_CACHE.pop(proc_path, None)


# ---------------------------------------------------------------------------
# Scenario J: Newer notice supersedes an older one
# ---------------------------------------------------------------------------
def test_scenario_j_newer_notice_supersedes_older_circular(multi_date_corpus_records):
    """Scenario J: When a revised circular (Oct 18, deadline Oct 30) supersedes an earlier circular
    (Oct 1, deadline Oct 15), the answer prioritizes the Oct 18 notice and notes the update."""
    retriever = TFIDFRetriever()
    retriever.index_records(multi_date_corpus_records)

    result = answer_question(
        "What is the last date for payment of higher semester UG tuition fees?",
        retriever=retriever,
    )

    assert result["sources"][0]["document_id"] == "doc_fee_revised_new"
    assert "30.10.2026" in result["answer"]
    assert "supersedes" in result["answer"].lower() or "extended" in result["answer"].lower()


# ---------------------------------------------------------------------------
# Scenario K: Missing date metadata -> never ranked as latest
# ---------------------------------------------------------------------------
def test_scenario_k_missing_date_metadata_never_ranked_as_latest(multi_date_corpus_records):
    """Scenario K: Undated documents (document_date=None) must never outrank dated documents for 'latest'."""
    candidates = [
        {
            "chunk_id": "c_undated",
            "document_id": "doc_email_intern_undated",
            "document_title": "General Internship Guidelines",
            "text": "Mandatory internship guidelines for all students.",
            "document_date": None,
            "received_at": None,
            "last_modified": None,
            "relevance_score": 0.99,  # artificially high relevance score
        },
        {
            "chunk_id": "c_dated_oct",
            "document_id": "doc_email_intern_latest",
            "document_title": "Winter Internship Drive",
            "text": "Winter internship registration closes 5 November 2026.",
            "document_date": "2026-10-29T10:30:00+00:00",
            "received_at": "2026-10-29T10:30:00+00:00",
            "relevance_score": 0.45,
        },
        {
            "chunk_id": "c_dated_sep",
            "document_id": "doc_email_intern_old",
            "document_title": "Bosch Summer Internship",
            "text": "Bosch summer internship deadline 20 September 2026.",
            "document_date": "2026-09-10T09:00:00+00:00",
            "received_at": "2026-09-10T09:00:00+00:00",
            "relevance_score": 0.50,
        },
    ]

    temporal_info = parse_temporal_query("What is the latest internship update?")
    ranked = filter_and_rank_by_date(candidates, temporal_info)

    assert ranked[0]["document_id"] == "doc_email_intern_latest"
    assert ranked[1]["document_id"] == "doc_email_intern_old"
    assert ranked[-1]["document_id"] == "doc_email_intern_undated"


# ---------------------------------------------------------------------------
# Scenario L: Date sorting -> chronological ordering is accurate
# ---------------------------------------------------------------------------
def test_scenario_l_chronological_date_sorting():
    """Scenario L: Chronological sorting orders documents accurately (newest-first and oldest-first)
    while always placing undated documents last."""
    items = [
        {"document_id": "doc_mid", "document_date": "2026-10-01T08:00:00+00:00", "relevance_score": 0.5},
        {"document_id": "doc_none", "document_date": None, "relevance_score": 0.95},
        {"document_id": "doc_newest", "document_date": "2026-10-29T10:30:00+00:00", "relevance_score": 0.6},
        {"document_id": "doc_oldest", "document_date": "2026-09-10T09:00:00+00:00", "relevance_score": 0.7},
    ]

    desc_sorted = sort_results_chronologically(items, descending=True)
    assert [r["document_id"] for r in desc_sorted] == [
        "doc_newest",
        "doc_mid",
        "doc_oldest",
        "doc_none",
    ]

    asc_sorted = sort_results_chronologically(items, descending=False)
    assert [r["document_id"] for r in asc_sorted] == [
        "doc_oldest",
        "doc_mid",
        "doc_newest",
        "doc_none",
    ]


# ---------------------------------------------------------------------------
# Scenario M: Live Corpus 5-Question Relevance Benchmarks
# ---------------------------------------------------------------------------
def test_live_corpus_five_relevance_benchmarks():
    """Verify strict retrieval relevance and answer grounding on the 5 target questions."""
    # 1. What documents are required for PGCET admissions?
    q1 = answer_question("What documents are required for PGCET admissions?", strategy="tfidf")
    assert "KEA Confirmation Slip" in q1["answer"]
    assert "PGCET" in q1["answer"]
    assert "Study Abroad" not in q1["answer"]
    assert "Fateh" not in q1["answer"]
    assert "RTGS" not in q1["answer"]
    q1_titles = [e["title"] for e in q1["evidence"]]
    assert "M.Tech Mca Procedure And Balance Fees Details For Admission 2026 2027 2" in q1_titles
    assert all("Study Abroad" not in t and "3Rd Sem Fee" not in t for t in q1_titles)

    # 2. What is the M.Tech admission procedure?
    q2 = answer_question("What is the M.Tech admission procedure?", strategy="tfidf")
    assert "Step 1" in q2["answer"]
    assert "Step 4" in q2["answer"]
    assert "acknowledgement" in q2["answer"].lower()
    assert "Study Abroad" not in q2["answer"]

    # 3. What is the latest internship notice?
    q3 = answer_question("What is the latest internship notice?", strategy="tfidf")
    assert "PG Circular and Notifications (Academics)" in q3["answer"]
    assert "2026-10-06" in q3["answer"]
    assert "Revised dates for internship" in q3["answer"]
    q3_titles = [e["title"] for e in q3["evidence"]]
    assert all("Placement Success" not in t for t in q3_titles)

    # 4. What are the hostel fees?
    q4 = answer_question("What are the hostel fees?", strategy="tfidf")
    assert "080-68188256" in q4["answer"]
    assert "080-68188165" in q4["answer"]
    assert "10,500" in q4["answer"]
    q4_titles = [e["title"] for e in q4["evidence"]]
    assert all("See Pvp" not in t and "Circular 35" not in t for t in q4_titles)

    # 5. What are the placement eligibility requirements?
    q5 = answer_question("What are the placement eligibility requirements?", strategy="tfidf")
    assert "Associate Dean of Placement & Training" in q5["answer"]
    assert "not specified in the currently indexed campus documents" in q5["answer"]
    q5_headings = [e["heading"] for e in q5["evidence"]]
    assert "Admissions" not in q5_headings


# ---------------------------------------------------------------------------
# Scenario N: All 7 Submission Scenarios (Query-time Gmail MCP + LR Intent + Corpus)
# ---------------------------------------------------------------------------
def test_seven_submission_scenarios_end_to_end():
    """Verify all 7 submission scenarios:
    1. Question answered from the corpus.
    2. Question answered from query-time Gmail MCP (unindexed email).
    3. Question answered using BOTH corpus and query-time Gmail MCP.
    4. Question with no relevant evidence (honest abstention).
    5. Query classified by Logistic Regression.
    6. Query with an unknown intent (general search fallback).
    7. Date-specific query.
    """
    from src.intent_classification import classify_query_intent

    # 1. Answered from the corpus
    corpus_ans = answer_question("What documents are required for PGCET admissions?", strategy="tfidf")
    assert "KEA Confirmation Slip" in corpus_ans["answer"]
    assert any(s["source_type"] == "PDF" for s in corpus_ans["sources"])
    assert corpus_ans["intent"]["intent"] == "admissions"

    # Build a mock Gmail service with an unindexed email for query-time Gmail MCP search
    mock_gmail_service = MagicMock()
    mock_gmail_service.users().messages().list.return_value.execute.return_value = {
        "messages": [{"id": "msg_unindexed_nvidia_01"}],
        "nextPageToken": None,
    }
    mock_gmail_service.users().messages().get.return_value.execute.return_value = {
        "id": "msg_unindexed_nvidia_01",
        "internalDate": "1762685000000",
        "snippet": "NVIDIA GPU Research Fellowship & M.Tech Orientation Notice",
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": "Dean PG Studies <dean.pg@rvce.edu.in>"},
                {
                    "name": "Subject",
                    "value": "NVIDIA GPU Research Fellowship & M.Tech Orientation Schedule (Nov 2026)",
                },
                {"name": "Date", "value": "Mon, 09 Nov 2026 11:30:00 +0530"},
            ],
            "body": {
                "data": base64.urlsafe_b64encode(
                    b"Applications are open for the NVIDIA GPU Research Fellowship with a monthly stipend of Rs. 45,000 at the Supercomputing Lab. "
                    b"Additionally, the mandatory M.Tech admission orientation session will be held on 18 November 2026 at the IEM Auditorium."
                ).decode("ascii")
            },
        },
    }
    live_mcp_server = CampusGmailMCPServer(service=mock_gmail_service)
    live_mcp_client = CampusGmailMCPClient(server=live_mcp_server)

    # 2. Question answered exclusively from query-time Gmail MCP (not in local corpus)
    gmail_only_ans = answer_question(
        "What is the monthly stipend and venue for the NVIDIA GPU Research Fellowship?",
        strategy="tfidf",
        mcp_client=live_mcp_client,
    )
    assert "45,000" in gmail_only_ans["answer"]
    assert "Supercomputing Lab" in gmail_only_ans["answer"]
    assert any(s.get("source_type") == "EMAIL" for s in gmail_only_ans["sources"])

    # 3. Question answered using BOTH indexed corpus and query-time Gmail MCP
    both_sources_ans = answer_question(
        "What is the M.Tech admission orientation schedule and fee details?",
        strategy="tfidf",
        mcp_client=live_mcp_client,
    )
    source_types = {s.get("source_type") for s in both_sources_ans["sources"]}
    assert "EMAIL" in source_types
    assert ("PDF" in source_types or "HTML" in source_types)
    assert "18 November 2026" in both_sources_ans["answer"] or "IEM Auditorium" in both_sources_ans["answer"]

    # 4. Question with no relevant evidence -> honest abstention
    no_ev_ans = answer_question("What is the spacecraft launch schedule at Cape Canaveral?", strategy="tfidf")
    assert no_ev_ans["answer"] == "I could not find sufficient evidence in the available campus documents to answer this question."
    assert no_ev_ans["sources"] == []

    # 5. Query classified by Logistic Regression
    lr_info = classify_query_intent("What is the deadline for paying higher semester tuition fees?")
    assert lr_info["intent"] == "fees_finance"
    assert lr_info["intent_label"] == "Fees and Finance"
    assert lr_info["is_confident"] is True
    assert lr_info["routing"] == "intent_guided"

    # 6. Query with an unknown intent -> general search fallback
    unk_info = classify_query_intent("zyxwvutsrq plugh xyzzy")
    assert unk_info["intent"] == "unknown"
    assert unk_info["is_confident"] is False
    assert unk_info["routing"] == "general_fallback"

    # 7. Date-specific query
    date_ans = answer_question("What is the latest internship notice?", strategy="tfidf")
    assert "2026-10-06" in date_ans["answer"]
    assert "PG Circular and Notifications (Academics)" in date_ans["answer"]



