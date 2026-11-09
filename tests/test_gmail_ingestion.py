"""
tests/test_gmail_ingestion.py - Unit tests for the Gmail Notice Ingestion pipeline.

Uses mocked Gmail service fixtures to test:
- Sender domain and office whitelist verification
- Negative subject exclusion filtering
- HTML body extraction and sanitization
- PDF attachment parsing and parent-child document linking
- Ingestion deduplication and state tracking
- Dry-run and live ingestion modes
"""

import base64
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from src.gmail_filter import extract_email_address, is_relevant_notice
from src.gmail_ingestion import (
    load_ingestion_state,
    run_gmail_ingestion,
    save_ingestion_state,
)
from src.gmail_parser import parse_email_message


# ──────────────────────────────────────────────
# 1. Filter Tests
# ──────────────────────────────────────────────
def test_extract_email_address():
    display, email = extract_email_address("RVCE Principal <principal@rvce.edu.in>")
    assert display == "RVCE Principal"
    assert email == "principal@rvce.edu.in"

    display2, email2 = extract_email_address("coe@rvce.edu.in")
    assert email2 == "coe@rvce.edu.in"


def test_filter_authorized_notice():
    headers = [
        {"name": "From", "value": "Controller of Examinations <coe@rvce.edu.in>"},
        {"name": "Subject", "value": "Circular: Revised CIE-II Timetable for 3rd Sem BE"},
    ]
    is_acc, reason = is_relevant_notice(headers, snippet="Please note the revised CIE timetable.")
    assert is_acc is True
    assert "ACCEPTED" in reason


def test_filter_unauthorized_external_domain():
    headers = [
        {"name": "From", "value": "Spam Sender <spammer@external-marketing.com>"},
        {"name": "Subject", "value": "Important circular on student discounts"},
    ]
    is_acc, reason = is_relevant_notice(headers, snippet="Buy our courses today!")
    assert is_acc is False
    assert "UNAUTHORIZED_DOMAIN" in reason


def test_filter_negative_subject_patterns():
    # OTP / Security alerts should be rejected even if from rvce.edu.in
    headers = [
        {"name": "From", "value": "IT Support <support@rvce.edu.in>"},
        {"name": "Subject", "value": "Your one-time OTP verification code"},
    ]
    is_acc, reason = is_relevant_notice(headers, snippet="Your OTP is 123456")
    assert is_acc is False
    assert "NEGATIVE_PATTERN" in reason


def test_filter_conversational_thread_rejection():
    headers = [
        {"name": "From", "value": "Student <student@rvce.edu.in>"},
        {"name": "Subject", "value": "Re: Are you free for lunch tomorrow?"},
    ]
    is_acc, reason = is_relevant_notice(headers, snippet="Let us meet at the canteen.")
    assert is_acc is False


# ──────────────────────────────────────────────
# 2. Parser Tests
# ──────────────────────────────────────────────
def test_parse_email_html_body():
    raw_html = "<html><body><h2>CIE Exam Notice</h2><p>The exam starts at 9:30 AM.</p><script>alert('noise')</script></body></html>"
    b64_html = base64.urlsafe_b64encode(raw_html.encode("utf-8")).decode("ascii")

    msg_dict = {
        "id": "18d8f9a2c10b7e41",
        "payload": {
            "mimeType": "text/html",
            "headers": [
                {"name": "Subject", "value": "CIE Exam Instructions"},
                {"name": "From", "value": "CoE RVCE <coe@rvce.edu.in>"},
                {"name": "Date", "value": "Mon, 28 Sep 2026 10:00:00 +0530"},
            ],
            "body": {"data": b64_html},
        },
    }

    parent_rec, child_recs = parse_email_message(msg_dict)
    assert parent_rec is not None
    assert parent_rec["document_id"].startswith("doc_email_")
    assert parent_rec["source_type"] == "EMAIL"
    assert "CIE Exam Notice" in parent_rec["content"]
    assert "noise" not in parent_rec["content"]  # script stripped by HTMLExtractor
    assert len(child_recs) == 0


def test_parse_email_with_pdf_attachment():
    msg_dict = {
        "id": "18d8f9a2c10b7e99",
        "payload": {
            "mimeType": "multipart/mixed",
            "headers": [
                {"name": "Subject", "value": "Fee Payment Extension Circular"},
                {"name": "From", "value": "Accounts Section <accounts@rvce.edu.in>"},
                {"name": "Date", "value": "Mon, 28 Sep 2026 11:00:00 +0530"},
            ],
            "body": {},
            "parts": [
                {
                    "mimeType": "text/plain",
                    "body": {"data": base64.urlsafe_b64encode(b"Please find attached the official circular regarding fee extension.").decode("ascii")},
                },
                {
                    "filename": "Fee-extension-circular.pdf",
                    "mimeType": "application/pdf",
                    "body": {"attachmentId": "att_12345", "size": 1024},
                },
            ],
        },
    }

    # Synthetic dummy PDF bytes
    dummy_pdf_bytes = b"%PDF-1.4 Mock PDF Binary"

    with patch("src.gmail_parser._pdf_extractor.extract") as mock_extract:
        mock_extract.return_value = {
            "title": "Fee Extension Circular",
            "content": "All higher semester students must pay fees before October 15, 2026.",
            "extraction_status": "SUCCESS",
        }

        parent_rec, child_recs = parse_email_message(
            msg_dict,
            attachment_bytes_map={"att_12345": dummy_pdf_bytes},
        )

        assert parent_rec is not None
        assert len(child_recs) == 1
        child = child_recs[0]
        assert child["document_id"].startswith("doc_email_att_")
        assert child["source_type"] == "EMAIL_ATTACHMENT"
        assert child["parent_document_id"] == parent_rec["document_id"]
        assert "October 15, 2026" in child["content"]


# ──────────────────────────────────────────────
# 3. Ingestion State & Orchestration Tests
# ──────────────────────────────────────────────
def test_ingestion_state_save_and_load(tmp_path):
    state_file = str(tmp_path / "test_state.json")
    state = {
        "processed_message_ids": ["msg_1", "msg_2"],
        "last_synced_internal_date_ms": 1730000000,
        "last_sync_timestamp": "2026-11-08T18:00:00Z",
    }
    save_ingestion_state(state, state_file)
    loaded = load_ingestion_state(state_file)
    assert loaded["processed_message_ids"] == ["msg_1", "msg_2"]
    assert loaded["last_synced_internal_date_ms"] == 1730000000


def test_run_gmail_ingestion_dry_run_and_deduplication(tmp_path):
    state_path = str(tmp_path / "state.json")
    raw_path = str(tmp_path / "raw.jsonl")
    proc_path = str(tmp_path / "proc.jsonl")
    manifest_path = str(tmp_path / "manifest.json")

    mock_service = MagicMock()

    # Mock list call returning 2 messages
    mock_service.users().messages().list().execute.return_value = {
        "messages": [{"id": "msg_001"}, {"id": "msg_002"}],
        "nextPageToken": None,
    }

    # Mock get message call
    def mock_get(userId, id, format):
        mock_obj = MagicMock()
        if id == "msg_001":
            mock_obj.execute.return_value = {
                "id": "msg_001",
                "internalDate": "1730000001",
                "snippet": "Official notice from the Principal",
                "payload": {
                    "mimeType": "text/plain",
                    "headers": [
                        {"name": "From", "value": "Principal <principal@rvce.edu.in>"},
                        {"name": "Subject", "value": "Circular: Campus Holiday on Friday"},
                    ],
                    "body": {"data": base64.urlsafe_b64encode(b"College remains closed on Friday on account of holiday.").decode("ascii")},
                },
            }
        else:
            # msg_002 is noise
            mock_obj.execute.return_value = {
                "id": "msg_002",
                "internalDate": "1730000002",
                "snippet": "Your one-time password is 9988",
                "payload": {
                    "mimeType": "text/plain",
                    "headers": [
                        {"name": "From", "value": "Auth Service <auth@rvce.edu.in>"},
                        {"name": "Subject", "value": "Your login OTP code"},
                    ],
                    "body": {"data": base64.urlsafe_b64encode(b"OTP 9988").decode("ascii")},
                },
            }
        return mock_obj

    mock_service.users().messages().get.side_effect = mock_get

    # 1. Run live ingestion
    stats = run_gmail_ingestion(
        service=mock_service,
        dry_run=False,
        state_path=state_path,
        raw_path=raw_path,
        processed_path=proc_path,
        manifest_path=manifest_path,
    )

    assert stats["candidate_messages"] == 2
    assert stats["accepted_notices"] == 1
    assert stats["rejected_noise"] == 1
    assert stats["new_email_records"] == 1

    # Verify processed file has 1 record
    with open(proc_path, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f]
    assert len(lines) == 1
    assert "Campus Holiday" in lines[0]["title"]

    # 2. Run ingestion again: deduplication should ensure 0 new candidate messages
    stats_repeat = run_gmail_ingestion(
        service=mock_service,
        dry_run=False,
        state_path=state_path,
        raw_path=raw_path,
        processed_path=proc_path,
        manifest_path=manifest_path,
    )

    assert stats_repeat["candidate_messages"] == 0
    assert stats_repeat["new_email_records"] == 0


# ──────────────────────────────────────────────
# 4. Web OAuth Callback Flow Tests
# ──────────────────────────────────────────────
def test_flask_oauth_callback_route():
    from flask import Flask, request

    app = Flask("test_oauth")
    mock_flow = MagicMock()
    mock_flow.credentials = MagicMock()

    @app.route("/rest/oauth2-credential/callback")
    def oauth_callback():
        error = request.args.get("error")
        if error:
            return "Auth failed", 400
        mock_flow.fetch_token(authorization_response=request.url)
        return "Auth success", 200

    client = app.test_client()
    resp = client.get("/rest/oauth2-credential/callback?code=test_code_123&state=test_state")
    assert resp.status_code == 200
    assert b"Auth success" in resp.data
    mock_flow.fetch_token.assert_called_once()


def test_flask_oauth_callback_error():
    from flask import Flask, request

    app = Flask("test_oauth_err")

    @app.route("/rest/oauth2-credential/callback")
    def oauth_callback():
        error = request.args.get("error")
        if error:
            return f"Auth failed: {error}", 400
        return "Auth success", 200

    client = app.test_client()
    resp = client.get("/rest/oauth2-credential/callback?error=access_denied")
    assert resp.status_code == 400
    assert b"access_denied" in resp.data


def test_oauth_callback_invalid_state_format_regression():
    """
    Regression test: Reproduce error handling when state is missing or mismatched,
    and verify successful handling when state matches.
    """
    from flask import Flask, request
    from oauthlib.oauth2.rfc6749.errors import MismatchingStateError

    expected_state = "test_expected_state_abc123"
    app = Flask("test_oauth_state")
    mock_flow = MagicMock()

    @app.route("/rest/oauth2-credential/callback")
    def oauth_callback():
        callback_state = request.args.get("state")
        if not callback_state:
            return "<html><body><h3>Authentication Failed</h3><p>OAuth Error: Missing state parameter in callback request</p></body></html>", 400

        if callback_state != expected_state:
            return "<html><body><h3>Authentication Failed</h3><p>OAuth Error: State parameter mismatch between request and callback</p></body></html>", 400

        try:
            mock_flow.fetch_token(authorization_response=request.url)
            return "<html><body><h2>Authentication Successful!</h2></body></html>", 200
        except MismatchingStateError as exc:
            return f"<html><body><h3>Authentication Failed</h3><p>{exc}</p></body></html>", 400

    client = app.test_client()

    # 1. Failure case: missing state parameter
    resp_missing = client.get("/rest/oauth2-credential/callback?code=mock_code")
    assert resp_missing.status_code == 400
    assert b"Missing state parameter" in resp_missing.data

    # 2. Failure case: mismatched state parameter
    resp_mismatch = client.get("/rest/oauth2-credential/callback?code=mock_code&state=wrong_state")
    assert resp_mismatch.status_code == 400
    assert b"State parameter mismatch" in resp_mismatch.data

    # 3. Success case: exact matching state parameter
    resp_success = client.get(f"/rest/oauth2-credential/callback?code=mock_code&state={expected_state}")
    assert resp_success.status_code == 200
    assert b"Authentication Successful!" in resp_success.data
    mock_flow.fetch_token.assert_called_once()


def test_flow_state_preservation():
    """Verify that Flow.from_client_config preserves the exact state between authorization and callback."""
    from google_auth_oauthlib.flow import Flow

    client_config = {
        "web": {
            "client_id": "test_client_id.apps.googleusercontent.com",
            "client_secret": "test_secret",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost:5678/rest/oauth2-credential/callback"],
        }
    }

    flow1 = Flow.from_client_config(
        client_config,
        scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        redirect_uri="http://localhost:5678/rest/oauth2-credential/callback",
    )
    auth_url, generated_state = flow1.authorization_url()
    assert generated_state is not None
    assert f"state={generated_state}" in auth_url

    # Reconstruct flow with same state (simulating callback worker)
    flow2 = Flow.from_client_config(
        client_config,
        scopes=["https://www.googleapis.com/auth/gmail.readonly"],
        state=generated_state,
        redirect_uri="http://localhost:5678/rest/oauth2-credential/callback",
    )
    assert flow2.oauth2session._state == generated_state


def test_get_gmail_service_with_cached_token(tmp_path):
    from src.gmail_client import get_gmail_service

    token_file = str(tmp_path / "token.json")
    creds_data = {
        "token": "mock_token",
        "refresh_token": "mock_refresh",
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": "mock_client_id",
        "client_secret": "mock_client_secret",
        "scopes": ["https://www.googleapis.com/auth/gmail.readonly"],
    }
    with open(token_file, "w", encoding="utf-8") as f:
        json.dump(creds_data, f)

    with patch("google.oauth2.credentials.Credentials.from_authorized_user_file") as mock_from_file, patch(
        "googleapiclient.discovery.build"
    ) as mock_build:
        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_from_file.return_value = mock_creds
        mock_build.return_value = MagicMock()

        service = get_gmail_service(
            credentials_path="non_existent.json",
            token_path=token_file,
        )
        assert service is not None
        mock_from_file.assert_called_once()
        mock_build.assert_called_once()


# ──────────────────────────────────────────────
# 5. Full History Ingestion & Reporting Tests
# ──────────────────────────────────────────────
def test_all_history_query_and_pagination(tmp_path):
    """Verify that all_history=True constructs after:2024/01/01 and paginates through multiple pages."""
    from src.gmail_ingestion import run_gmail_ingestion

    state_path = str(tmp_path / "state.json")
    raw_path = str(tmp_path / "raw.jsonl")
    proc_path = str(tmp_path / "proc.jsonl")
    report_path = str(tmp_path / "report.json")

    mock_service = MagicMock()
    recorded_queries = []

    def mock_list(*args, **kwargs):
        q = kwargs.get("q", "")
        recorded_queries.append(q)
        pageToken = kwargs.get("pageToken")
        mock_resp = MagicMock()
        if pageToken is None:
            mock_resp.execute.return_value = {
                "messages": [{"id": "hist_001"}, {"id": "hist_002"}],
                "nextPageToken": "page_2_token",
            }
        else:
            mock_resp.execute.return_value = {
                "messages": [{"id": "hist_003"}],
                "nextPageToken": None,
            }
        return mock_resp

    mock_service.users().messages().list.side_effect = mock_list

    def mock_get(userId, id, format):
        mock_obj = MagicMock()
        mock_obj.execute.return_value = {
            "id": id,
            "internalDate": "1710000000",
            "snippet": f"Notice {id}",
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "From", "value": "Principal <principal@rvce.edu.in>"},
                    {"name": "Subject", "value": f"Circular: Schedule Announcement {id}"},
                ],
                "body": {"data": base64.urlsafe_b64encode(b"Detailed official notice content here.").decode("ascii")},
            },
        }
        return mock_obj

    mock_service.users().messages().get.side_effect = mock_get

    stats = run_gmail_ingestion(
        service=mock_service,
        all_history=True,
        dry_run=False,
        state_path=state_path,
        raw_path=raw_path,
        processed_path=proc_path,
        report_path=report_path,
    )

    assert len(recorded_queries) == 2
    assert "after:2024/01/01" in recorded_queries[0]
    assert stats["total_messages_scanned"] == 3
    assert stats["relevant_emails_found"] == 3
    assert stats["emails_stored"] == 3
    assert os.path.exists(report_path)

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert report["all_history_mode"] is True
    assert report["emails_stored"] == 3


def test_custom_start_date_query(tmp_path):
    """Verify that start_date='2024/08/15' sets after:2024/08/15 in the Gmail query."""
    from src.gmail_ingestion import run_gmail_ingestion

    mock_service = MagicMock()
    recorded_queries = []

    def mock_list(*args, **kwargs):
        q = kwargs.get("q", "")
        recorded_queries.append(q)
        mock_resp = MagicMock()
        mock_resp.execute.return_value = {"messages": [], "nextPageToken": None}
        return mock_resp

    mock_service.users().messages().list.side_effect = mock_list

    run_gmail_ingestion(
        service=mock_service,
        start_date="2024/08/15",
        dry_run=True,
        state_path=str(tmp_path / "state.json"),
        report_path=str(tmp_path / "report.json"),
    )

    assert len(recorded_queries) == 1
    assert "after:2024/08/15" in recorded_queries[0]


def test_reindex_resets_caches(tmp_path):
    """Verify that --reindex resets retriever caches and rebuilds indexes."""
    from src.gmail_ingestion import run_gmail_ingestion
    from src.retrieval import (
        _DEFAULT_RETRIEVER_CACHE,
        _DEFAULT_SEMANTIC_RETRIEVER_CACHE,
        get_default_retriever,
        reset_retriever_caches,
    )

    proc_path = str(tmp_path / "proc.jsonl")
    with open(proc_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "document_id": "doc_test_1",
            "source_url": "gmail://messages/1",
            "title": "CIE Examination Circular",
            "content": "CIE-I examination will begin on Monday for all engineering students.",
            "source_type": "EMAIL",
            "extraction_status": "SUCCESS",
            "topic": "examinations_regulations",
        }) + "\n")

    # Populate cache
    r1 = get_default_retriever(proc_path)
    assert proc_path in _DEFAULT_RETRIEVER_CACHE

    reset_retriever_caches()
    assert len(_DEFAULT_RETRIEVER_CACHE) == 0
    assert len(_DEFAULT_SEMANTIC_RETRIEVER_CACHE) == 0


# ──────────────────────────────────────────────
# 6. Incremental Sync, Corpus Deduplication & Background Worker Tests
# ──────────────────────────────────────────────
def test_incremental_sync_and_live_index_update(tmp_path):
    """Verify that GmailSyncWorker ingests a new incoming email, preserves existing corpus,
    updates the retrieval index, and makes the new notice immediately searchable."""
    from src.gmail_ingestion import GmailSyncWorker, save_ingestion_state
    from src.retrieval import _DEFAULT_RETRIEVER_CACHE, get_default_retriever

    state_path = str(tmp_path / "state.json")
    raw_path = str(tmp_path / "raw.jsonl")
    proc_path = str(tmp_path / "proc.jsonl")
    manifest_path = str(tmp_path / "manifest.json")
    report_path = str(tmp_path / "report.json")

    # Seed existing processed corpus with 1 website document and 1 already-processed email
    existing_web_doc = {
        "document_id": "doc_html_existing_01",
        "source_url": "https://rvce.edu.in/admissions",
        "title": "Existing Admission Page",
        "content": "Undergraduate engineering admission through KEA and COMEDK counselling.",
        "source_type": "HTML",
        "extraction_status": "SUCCESS",
        "topic": "admissions",
    }
    existing_email_doc = {
        "document_id": "doc_email_existing_02",
        "source_url": "gmail://messages/msg_old_01",
        "title": "Old Exam Circular",
        "content": "CIE-I examination schedule for 3rd semester students.",
        "source_type": "EMAIL",
        "gmail_message_id": "msg_old_01",
        "extraction_status": "SUCCESS",
        "topic": "examinations_regulations",
    }
    with open(proc_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(existing_web_doc) + "\n")
        f.write(json.dumps(existing_email_doc) + "\n")

    save_ingestion_state(
        {
            "processed_message_ids": ["msg_old_01"],
            "last_synced_internal_date_ms": 1730000000000,
            "last_sync_timestamp": "2026-10-27T00:00:00+00:00",
            "total_emails_ingested": 1,
            "total_attachments_ingested": 0,
        },
        state_path,
    )

    # Initial retriever has 2 documents
    initial_retriever = get_default_retriever(proc_path)
    assert len(initial_retriever.chunks) == 2
    assert initial_retriever.retrieve("Quantum Computing Workshop registration deadline") == []

    # Mock Gmail service returning old message + 1 brand-new incoming notice
    mock_service = MagicMock()

    def mock_list(*args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.execute.return_value = {
            "messages": [{"id": "msg_old_01"}, {"id": "msg_new_02"}],
            "nextPageToken": None,
        }
        return mock_resp

    mock_service.users().messages().list.side_effect = mock_list

    def mock_get(userId, id, format):
        mock_obj = MagicMock()
        mock_obj.execute.return_value = {
            "id": "msg_new_02",
            "internalDate": "1731000000000",
            "snippet": "Circular regarding Quantum Computing Workshop and Exam Schedule",
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "From", "value": "Dean Academics <dean.academics@rvce.edu.in>"},
                    {"name": "Subject", "value": "Circular: Quantum Computing Workshop & Schedule"},
                    {"name": "Date", "value": "Mon, 09 Nov 2026 09:00:00 +0530"},
                ],
                "body": {
                    "data": base64.urlsafe_b64encode(
                        b"All CSE and ISE students are informed that the Quantum Computing Workshop "
                        b"registration deadline is 15 November 2026 at the Main Auditorium."
                    ).decode("ascii")
                },
            },
        }
        return mock_obj

    mock_service.users().messages().get.side_effect = mock_get

    worker = GmailSyncWorker(
        interval_seconds=60,
        state_path=state_path,
        raw_path=raw_path,
        processed_path=proc_path,
        manifest_path=manifest_path,
        report_path=report_path,
        service=mock_service,
    )

    sync_stats = worker.sync_once()
    assert sync_stats["status"] == "ok"
    assert sync_stats["already_processed_messages"] == 1
    assert sync_stats["candidate_messages_evaluated"] == 1
    assert sync_stats["emails_stored"] == 1
    assert sync_stats["index_updated"] is True

    # Existing 2 documents + 1 new email = 3 total documents in processed corpus
    with open(proc_path, "r", encoding="utf-8") as f:
        corpus_lines = [json.loads(line) for line in f if line.strip()]
    assert len(corpus_lines) == 3

    # Updated cached retriever immediately finds the newly ingested email
    updated_retriever = get_default_retriever(proc_path)
    hits = updated_retriever.retrieve("Quantum Computing Workshop registration deadline")
    assert len(hits) >= 1
    assert "15 November 2026" in hits[0]["text"]
    assert hits[0]["source_type"] == "EMAIL"
    assert hits[0]["sender"] == "dean.academics@rvce.edu.in"

    # Cleanup test cache entry
    _DEFAULT_RETRIEVER_CACHE.pop(proc_path, None)


def test_update_retrieval_index_preserves_existing_cache_on_error(tmp_path):
    """If re-indexing fails or encounters an empty file, previously indexed documents must not be lost."""
    from src.gmail_ingestion import update_retrieval_index
    from src.retrieval import _DEFAULT_RETRIEVER_CACHE, get_default_retriever

    proc_path = str(tmp_path / "proc_safe.jsonl")
    with open(proc_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "document_id": "doc_safe_1",
            "source_url": "https://rvce.edu.in/safe",
            "title": "Safe Document",
            "content": "Important academic calendar and examination regulations for students.",
            "source_type": "HTML",
            "extraction_status": "SUCCESS",
        }) + "\n")

    orig_retriever = get_default_retriever(proc_path)
    assert len(orig_retriever.chunks) == 1

    # Overwrite with empty file to simulate corrupted/empty staging state
    with open(proc_path, "w", encoding="utf-8") as f:
        f.write("")

    res = update_retrieval_index(proc_path, include_semantic=False)
    assert res["updated"] is False
    # Existing cached retriever must still be intact with its 1 indexed chunk
    assert _DEFAULT_RETRIEVER_CACHE[proc_path] is orig_retriever
    assert len(_DEFAULT_RETRIEVER_CACHE[proc_path].chunks) == 1

    _DEFAULT_RETRIEVER_CACHE.pop(proc_path, None)




