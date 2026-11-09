"""Tests for src/generation.py — GPT-based answer generation.

All OpenAI API calls are fully mocked so that no network traffic or real API key
is required.  The existing 68-test suite baseline remains unchanged.
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

SAMPLE_CHUNKS: List[Dict[str, Any]] = [
    {
        "chunk_id": "doc1_chunk0",
        "document_id": "doc1",
        "document_title": "RVCE Admission Process",
        "source_url": "https://rvce.edu.in/admission",
        "heading": "COMEDK Procedure",
        "text": "Step 1: Download the allotment order. Step 2: Pay the balance fees online.",
        "topic": "admissions",
        "relevance_score": 0.82,
    },
    {
        "chunk_id": "doc2_chunk0",
        "document_id": "doc2",
        "document_title": "RVCE Fee Structure",
        "source_url": "https://rvce.edu.in/fees",
        "heading": "CSE Fee Details",
        "text": "The tuition fee for CSE is Rs 1,85,000 per year for COMEDK seats.",
        "topic": "fees",
        "relevance_score": 0.75,
    },
]


def _make_openai_response(content: str) -> MagicMock:
    """Build a minimal mock that mimics openai.ChatCompletion response structure."""
    msg = MagicMock()
    msg.content = content

    choice = MagicMock()
    choice.message = msg

    response = MagicMock()
    response.choices = [choice]
    return response


def _make_openai_module(response_content: str = "Here is the answer.") -> types.ModuleType:
    """Return a fake ``openai`` module with a mocked client."""
    fake_openai = types.ModuleType("openai")

    client_instance = MagicMock()
    client_instance.chat.completions.create.return_value = _make_openai_response(response_content)

    fake_openai.OpenAI = MagicMock(return_value=client_instance)
    return fake_openai


# ---------------------------------------------------------------------------
# Tests for generate_answer()
# ---------------------------------------------------------------------------


class TestGenerateAnswerBasic:
    """Core behaviour of generate_answer()."""

    def test_returns_gpt_strategy_key(self):
        """Result dict must include strategy='gpt'."""
        from src.generation import generate_answer

        fake_openai = _make_openai_module("Test answer")
        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
                result = generate_answer("What is the fee?", SAMPLE_CHUNKS)
        assert result["strategy"] == "gpt"

    def test_answer_text_from_api(self):
        """Answer text should be the content returned by the OpenAI API."""
        from src.generation import generate_answer

        expected = "The CSE fee is Rs 1,85,000 per year."
        fake_openai = _make_openai_module(expected)
        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
                result = generate_answer("What is the CSE fee?", SAMPLE_CHUNKS)
        assert result["answer"] == expected

    def test_evidence_built_from_chunks(self):
        """Evidence list should be populated from the provided chunks."""
        from src.generation import generate_answer

        fake_openai = _make_openai_module("Answer")
        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
                result = generate_answer("Any question", SAMPLE_CHUNKS)
        assert len(result["evidence"]) == len(SAMPLE_CHUNKS)
        assert result["evidence"][0]["title"] == "RVCE Admission Process"

    def test_sources_built_from_chunks(self):
        """Sources list should contain title/url/heading for each chunk."""
        from src.generation import generate_answer

        fake_openai = _make_openai_module("Answer")
        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
                result = generate_answer("Any question", SAMPLE_CHUNKS)
        assert len(result["sources"]) == len(SAMPLE_CHUNKS)
        for src in result["sources"]:
            assert "title" in src
            assert "url" in src
            assert "heading" in src


class TestGenerateAnswerFallbacks:
    """Fallback / abstention behaviour."""

    def test_empty_question_returns_no_evidence(self):
        """An empty question must return the no-evidence message without calling the API."""
        from src.generation import generate_answer, NO_EVIDENCE_MESSAGE

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
            result = generate_answer("", SAMPLE_CHUNKS)
        assert result["answer"] == NO_EVIDENCE_MESSAGE
        assert result["strategy"] == "gpt"

    def test_whitespace_only_question_returns_no_evidence(self):
        """A whitespace-only question must also trigger the no-evidence fallback."""
        from src.generation import generate_answer, NO_EVIDENCE_MESSAGE

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
            result = generate_answer("   ", SAMPLE_CHUNKS)
        assert result["answer"] == NO_EVIDENCE_MESSAGE

    def test_no_chunks_returns_no_evidence(self):
        """When no chunks are provided the API must NOT be called."""
        from src.generation import generate_answer, NO_EVIDENCE_MESSAGE

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
            result = generate_answer("What is the fee?", [])
        assert result["answer"] == NO_EVIDENCE_MESSAGE
        assert result["evidence"] == []
        assert result["sources"] == []

    def test_missing_api_key_returns_error_message(self):
        """When OPENAI_API_KEY is absent the function must return a clean, safe user error."""
        from src.generation import CLEAN_GENERATION_ERROR_MESSAGE, clear_generation_cache, generate_answer

        clear_generation_cache()
        env_without_key = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env_without_key, clear=True):
            with patch("src.generation._load_api_key", return_value=None):
                result = generate_answer("What is the fee?", SAMPLE_CHUNKS)
        assert result["answer"] == CLEAN_GENERATION_ERROR_MESSAGE
        assert result["error"]["category"] == "configuration_error"
        assert result["strategy"] == "gpt"


class TestGenerateAnswerErrorHandling:
    """API error handling."""

    def test_api_exception_returns_friendly_error(self):
        """When the OpenAI call raises, the function should return a clean user-facing message without raw stack traces."""
        from src.generation import CLEAN_GENERATION_ERROR_MESSAGE, clear_generation_cache, generate_answer

        clear_generation_cache()
        fake_openai = types.ModuleType("openai")
        client_instance = MagicMock()
        client_instance.chat.completions.create.side_effect = RuntimeError("network error")
        fake_openai.OpenAI = MagicMock(return_value=client_instance)

        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
                result = generate_answer("What is the fee?", SAMPLE_CHUNKS)
        assert result["answer"] == CLEAN_GENERATION_ERROR_MESSAGE
        assert "RuntimeError" not in result["answer"]
        assert "network error" not in result["answer"]
        assert result["strategy"] == "gpt"
        # Evidence and sources should still be populated
        assert len(result["evidence"]) == len(SAMPLE_CHUNKS)

    def test_api_timeout_returns_friendly_error(self):
        """Timeout errors should be caught and surfaced as a clean user-facing message."""
        from src.generation import CLEAN_GENERATION_ERROR_MESSAGE, clear_generation_cache, generate_answer

        clear_generation_cache()
        fake_openai = types.ModuleType("openai")
        client_instance = MagicMock()
        client_instance.chat.completions.create.side_effect = TimeoutError("timed out")
        fake_openai.OpenAI = MagicMock(return_value=client_instance)

        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
                result = generate_answer("Any?", SAMPLE_CHUNKS)
        assert result["answer"] == CLEAN_GENERATION_ERROR_MESSAGE
        assert result["error"]["category"] == "timeout"
        assert "TimeoutError" not in result["answer"]


class TestGenerateAnswerContextBuilding:
    """Context construction details."""

    def test_many_chunks_are_capped_at_max(self):
        """More than MAX_CHUNKS chunks should be silently capped."""
        from src.generation import generate_answer, MAX_CHUNKS

        many_chunks = SAMPLE_CHUNKS * 10  # 20 chunks
        fake_openai = _make_openai_module("Capped answer")
        captured_calls: list = []

        def _capture_create(**kwargs):
            captured_calls.append(kwargs)
            return _make_openai_response("Capped answer")

        client_instance = MagicMock()
        client_instance.chat.completions.create.side_effect = _capture_create
        fake_openai.OpenAI = MagicMock(return_value=client_instance)

        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
                result = generate_answer("Fee?", many_chunks)

        assert result["strategy"] == "gpt"
        # Evidence list itself should not exceed MAX_CHUNKS
        assert len(result["evidence"]) <= MAX_CHUNKS

    def test_chunk_with_missing_fields_does_not_crash(self):
        """Chunks with missing optional fields must be handled gracefully."""
        from src.generation import generate_answer

        minimal_chunk = [{"text": "Some campus info about fees."}]
        fake_openai = _make_openai_module("Minimal answer")
        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
                result = generate_answer("What about fees?", minimal_chunk)
        assert result["strategy"] == "gpt"
        assert result["evidence"][0]["title"] == "Unknown source"


# ---------------------------------------------------------------------------
# Integration tests with answer_question(strategy="gpt")
# ---------------------------------------------------------------------------


class TestAnswerQuestionGptStrategy:
    """Integration: answer_question dispatches to generate_answer when strategy='gpt'."""

    def test_gpt_strategy_reaches_generation(self):
        """answer_question(strategy='gpt') must call generate_answer."""
        from src.answering import answer_question

        mock_generate = MagicMock(
            return_value={
                "answer": "GPT answer here.",
                "evidence": [],
                "sources": [],
                "strategy": "gpt",
            }
        )
        with patch("src.answering.get_default_retriever") as mock_retriever_factory:
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = SAMPLE_CHUNKS
            mock_retriever_factory.return_value = mock_retriever
            with patch("src.generation.generate_answer", mock_generate):
                result = answer_question("What is the CSE fee?", strategy="gpt")
        assert result["strategy"] == "gpt"
        assert result["answer"] == "GPT answer here."

    def test_gpt_strategy_does_not_affect_tfidf(self):
        """The existing tfidf strategy must be unaffected by the gpt branch."""
        from src.answering import answer_question

        with patch("src.answering.get_default_retriever") as mock_retriever_factory:
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = []
            mock_retriever_factory.return_value = mock_retriever
            result = answer_question("What is the CSE fee?", strategy="tfidf")
        # Must not reach generate_answer — answer comes from build_evidence_answer
        assert result.get("strategy") is None or result.get("strategy") != "gpt"

    def test_gpt_strategy_with_explicit_retriever(self):
        """answer_question(strategy='gpt') should use a provided retriever."""
        from src.answering import answer_question

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = SAMPLE_CHUNKS

        mock_generate = MagicMock(
            return_value={
                "answer": "Answer from provided retriever.",
                "evidence": [],
                "sources": [],
                "strategy": "gpt",
            }
        )
        with patch("src.generation.generate_answer", mock_generate):
            result = answer_question(
                "What is the fee?",
                retriever=mock_retriever,
                strategy="gpt",
            )
        mock_retriever.retrieve.assert_called_once()
        assert result["strategy"] == "gpt"


# ---------------------------------------------------------------------------
# Source-Aware, Multi-Document Synthesis & Email Summarization Tests
# ---------------------------------------------------------------------------


class TestSourceAwareGeneration:
    """Tests for source classification, email notice summarization, website and mixed contexts."""

    def test_identify_chunk_source_does_not_invent_when_missing(self):
        """Missing metadata must yield category='unknown' and source_type=None."""
        from src.generation import identify_chunk_source

        info = identify_chunk_source({"text": "Bare chunk with no source metadata."})
        assert info["category"] == "unknown"
        assert info["source_type"] is None
        assert info["label"] is None

    def test_gpt_summarization_of_email_notices_and_signature_stripping(self):
        """Email notice chunks should include sender/date metadata and strip academic signature noise."""
        from src.generation import generate_answer

        email_chunks = [
            {
                "chunk_id": "doc_email_1_c00",
                "document_id": "doc_email_21e22bfc45ff2f3b",
                "document_title": "Registration for SEE answer PVP.",
                "source_url": "gmail://messages/1a182368de084d03",
                "heading": "Registration for SEE answer PVP.",
                "source_type": "EMAIL",
                "sender": "hod.cse@rvce.edu.in",
                "sender_name": "HOD Dept. of CS, RVCE",
                "last_modified": "2026-10-28T11:13:04+00:00",
                "topic": "departments",
                "text": (
                    "Dear All,\nKindly find the attached circular regarding registration for SEE answer PVP "
                    "of UG -III to VII (2022 scheme) Sept/ Oct 2026, Fast Track Examinations.\n"
                    "Google Scholar <https://scholar.google.com/citations> | ORCID <https://orcid.org/0000>"
                ),
                "relevance_score": 0.88,
            },
            {
                "chunk_id": "doc_email_att_1_c00",
                "document_id": "doc_email_att_05608db455f7d59c",
                "document_title": "See Pvp Ft",
                "source_url": "gmail://messages/1a182368de084d03/attachments/SEE PVP FT.pdf",
                "heading": "See Pvp Ft",
                "source_type": "EMAIL_ATTACHMENT",
                "sender": "hod.cse@rvce.edu.in",
                "sender_name": "HOD Dept. of CS, RVCE",
                "filename": "SEE PVP FT.pdf",
                "last_modified": "2026-10-28T11:13:04+00:00",
                "topic": "examinations_regulations",
                "text": (
                    "Registration for SEE Answer Paper Viewing (PVP) is open from 29.10.2026 to 01.11.2026 "
                    "on the SAP portal. Fee: Rs. 500/- per course. PVP conducted from 4th to 6th November 2026."
                ),
                "relevance_score": 0.91,
            },
        ]

        captured_prompts: List[str] = []
        expected_summary = (
            "Registration for the Fast Track SEE Answer Paper Viewing (PVP) for UG III–VII semester "
            "(2022 scheme) is open from 29 October to 1 November 2026 via the SAP portal.\n"
            "• Fee: Rs. 500 per course\n"
            "• Viewing Dates: 4–6 November 2026 at the CoE office\n"
            "Source: Registration for SEE answer PVP. (HOD Dept. of CS, RVCE) & SEE PVP FT.pdf"
        )

        fake_openai = types.ModuleType("openai")
        client_instance = MagicMock()

        def _capture_chat(**kwargs):
            captured_prompts.append(kwargs["messages"][1]["content"])
            return _make_openai_response(expected_summary)

        client_instance.chat.completions.create.side_effect = _capture_chat
        fake_openai.OpenAI = MagicMock(return_value=client_instance)

        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
                result = generate_answer("When is SEE answer paper viewing and what is the fee?", email_chunks)

        assert result["answer"] == expected_summary
        assert result["strategy"] == "gpt"
        assert len(result["sources"]) == 2
        assert result["sources"][0]["source_type"] == "EMAIL"
        assert "HOD Dept. of CS, RVCE" in (result["sources"][0].get("sender") or "")
        assert result["sources"][1]["source_type"] == "EMAIL_ATTACHMENT"
        # Verify prompt contains sender metadata and strips Google Scholar / ORCID noise
        assert len(captured_prompts) == 1
        prompt_sent = captured_prompts[0]
        assert "HOD Dept. of CS, RVCE" in prompt_sent
        assert "SEE PVP FT.pdf" in prompt_sent
        assert "scholar.google.com" not in prompt_sent
        assert "orcid.org" not in prompt_sent

    def test_website_derived_answer_generation(self):
        """Website-derived chunks must be classified as HTML/PDF and preserve URLs."""
        from src.generation import generate_answer

        web_chunks = [
            {
                "chunk_id": "web_1",
                "document_id": "doc_pdf_9bc00d1612a37895",
                "document_title": "Fee Payment Extension Circular 28.09.26",
                "source_url": "https://rvce.edu.in/wp-content/uploads/2026/09/Fee-payment-extension-circular-28.09.26.pdf",
                "heading": "Higher Semester UG Students",
                "source_type": "PDF",
                "text": "The last date for payment of higher semester UG fees has been extended up to 10.10.2026.",
                "relevance_score": 0.89,
            }
        ]

        expected = "The deadline for higher-semester UG fee payment has been extended to 10 October 2026."
        fake_openai = _make_openai_module(expected)
        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
                result = generate_answer("What is the extended deadline for UG fee payment?", web_chunks)

        assert result["answer"] == expected
        assert result["sources"][0]["source_type"] == "PDF"
        assert result["sources"][0]["url"] == "https://rvce.edu.in/wp-content/uploads/2026/09/Fee-payment-extension-circular-28.09.26.pdf"

    def test_mixed_email_and_website_context_synthesis(self):
        """When both website and email chunks are retrieved, prompt must note mixed sources and cite both."""
        from src.generation import generate_answer

        mixed_chunks = [
            {
                "chunk_id": "web_adm",
                "document_id": "doc_html_admission_01",
                "document_title": "RVCE Admission Circulars",
                "source_url": "https://rvce.edu.in/admission-circulars/",
                "heading": "COMEDK Quota",
                "source_type": "HTML",
                "text": "Candidates joining through COMEDK must submit original rank cards and pay fees online.",
                "relevance_score": 0.84,
            },
            {
                "chunk_id": "email_adm",
                "document_id": "doc_email_adm99",
                "document_title": "Updated Reporting Schedule for First Year B.E.",
                "source_url": "gmail://messages/1a1999999",
                "heading": "Reporting Dates",
                "source_type": "EMAIL",
                "sender": "admission@rvce.edu.in",
                "sender_name": "Admissions Office RVCE",
                "text": "First-year B.E. classes commence on Monday, 24 August 2026 at 9:00 AM.",
                "relevance_score": 0.81,
            },
        ]

        captured_prompts: List[str] = []
        synthesized = (
            "COMEDK candidates must submit original rank cards and pay balance fees online "
            "(RVCE Admission Circulars), and first-year B.E. classes commence on 24 August 2026 at 9:00 AM "
            "(Updated Reporting Schedule email from Admissions Office RVCE)."
        )

        fake_openai = types.ModuleType("openai")
        client_instance = MagicMock()

        def _capture_chat(**kwargs):
            captured_prompts.append(kwargs["messages"][1]["content"])
            return _make_openai_response(synthesized)

        client_instance.chat.completions.create.side_effect = _capture_chat
        fake_openai.OpenAI = MagicMock(return_value=client_instance)

        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
                result = generate_answer("How do COMEDK admissions and class commencement work?", mixed_chunks)

        assert result["answer"] == synthesized
        source_types = {s.get("source_type") for s in result["sources"]}
        assert source_types == {"HTML", "EMAIL"}
        assert "BOTH official RVCE website documents and campus email notices" in captured_prompts[0]

    def test_responses_api_support(self):
        """generate_answer should support OpenAI's official client.responses.create API."""
        from src.generation import generate_answer

        fake_openai = types.ModuleType("openai")
        client_instance = MagicMock()
        mock_resp = MagicMock()
        mock_resp.output_text = "Synthesized via OpenAI Responses API."
        client_instance.responses.create.return_value = mock_resp
        fake_openai.OpenAI = MagicMock(return_value=client_instance)

        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
                result = generate_answer("What is the CSE fee?", SAMPLE_CHUNKS)

        client_instance.responses.create.assert_called_once()
        assert result["answer"] == "Synthesized via OpenAI Responses API."

    def test_unsupported_query_in_gpt_strategy_abstains_without_api_call(self):
        """When answer_question(strategy='gpt') finds no matching chunks, it must abstain cleanly."""
        from src.answering import answer_question, NO_EVIDENCE_MESSAGE
        from src.retrieval import TFIDFRetriever

        retriever = TFIDFRetriever()
        retriever.index_records([
            {
                "document_id": "doc_html_1",
                "source_url": "https://rvce.edu.in/admissions",
                "title": "Admissions",
                "content": "Candidates must submit COMEDK allotment letters and pay tuition fees.",
                "source_type": "HTML",
                "extraction_status": "SUCCESS",
            }
        ])

        result = answer_question("What is the capital of France?", retriever=retriever, strategy="gpt")
        assert result["answer"] == NO_EVIDENCE_MESSAGE
        assert result["evidence"] == []
        assert result["sources"] == []


# ---------------------------------------------------------------------------
# Rate-Limit, Exponential Backoff, Retry-After & Quota Exhaustion Tests
# ---------------------------------------------------------------------------


class FakeRateLimitError(Exception):
    """Mock exception mimicking openai.RateLimitError attributes."""

    def __init__(
        self,
        message: str,
        status_code: int = 429,
        code: str | None = None,
        err_type: str | None = None,
        body: dict | None = None,
        headers: dict | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.type = err_type
        self.body = body
        if headers is not None:
            resp = MagicMock()
            resp.headers = headers
            self.response = resp
        else:
            self.response = None


FakeRateLimitError.__name__ = "RateLimitError"


class TestRateLimitAndBackoffHandling:
    """Tests for rate-limit classification, backoff with jitter, Retry-After, and quota errors."""

    def test_insufficient_quota_is_not_retried_and_no_duplicate_fallback(self):
        """credit_balance_exhausted / insufficient_quota must fail immediately without retries or fallback."""
        from src.generation import CLEAN_GENERATION_ERROR_MESSAGE, clear_generation_cache, generate_answer

        clear_generation_cache()
        quota_exc = FakeRateLimitError(
            "Error code: 429 - You have no credits remaining. Add credits to continue using the API.",
            status_code=429,
            code="credit_balance_exhausted",
            err_type="insufficient_quota",
            body={
                "message": "You have no credits remaining. Add credits to continue using the API.",
                "type": "insufficient_quota",
                "param": None,
                "code": "credit_balance_exhausted",
            },
        )

        fake_openai = types.ModuleType("openai")
        client_instance = MagicMock()
        client_instance.responses.create.side_effect = quota_exc
        fake_openai.OpenAI = MagicMock(return_value=client_instance)

        sleep_calls: List[float] = []

        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
                result = generate_answer(
                    "What is the CSE fee?",
                    SAMPLE_CHUNKS,
                    max_retries=3,
                    sleep_fn=sleep_calls.append,
                )

        # Must NOT retry insufficient_quota and must NOT fall back to chat.completions
        assert client_instance.responses.create.call_count == 1
        assert client_instance.chat.completions.create.call_count == 0
        assert sleep_calls == []
        assert "error" in result
        assert result["error"]["category"] == "insufficient_quota"
        assert result["error"]["error_code"] == "credit_balance_exhausted"
        assert result["error"]["retryable"] is False
        assert result["answer"] == CLEAN_GENERATION_ERROR_MESSAGE
        assert "OpenAI" not in result["answer"]
        assert "429" not in result["answer"]
        # Must NOT silently return raw chunk text as the answer
        assert "Step 1: Download the allotment order" not in result["answer"]

    def test_temporary_rpm_rate_limit_retries_with_exponential_backoff_and_succeeds(self):
        """Transient RPM HTTP 429 errors must be retried with exponential backoff + jitter."""
        from src.generation import clear_generation_cache, generate_answer

        clear_generation_cache()
        rpm_exc = FakeRateLimitError(
            "Rate limit reached for gpt-4o-mini on requests per min (RPM): Limit 3, Used 3, Requested 1.",
            status_code=429,
            code="rate_limit_exceeded",
            err_type="requests",
            headers={"x-ratelimit-remaining-requests": "0"},
        )

        fake_openai = types.ModuleType("openai")
        client_instance = MagicMock()
        client_instance.chat.completions.create.side_effect = [
            rpm_exc,
            rpm_exc,
            _make_openai_response("Synthesized answer after transient RPM backoff."),
        ]
        fake_openai.OpenAI = MagicMock(return_value=client_instance)

        sleep_calls: List[float] = []

        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
                result = generate_answer(
                    "What is the CSE fee?",
                    SAMPLE_CHUNKS,
                    max_retries=3,
                    base_delay=1.0,
                    sleep_fn=sleep_calls.append,
                )

        assert client_instance.chat.completions.create.call_count == 3
        assert len(sleep_calls) == 2
        assert sleep_calls[0] >= 1.0
        assert sleep_calls[1] >= 2.0
        assert result["answer"] == "Synthesized answer after transient RPM backoff."
        assert "error" not in result

    def test_temporary_tpm_rate_limit_classification_and_exhausted_retries(self):
        """Persistent TPM HTTP 429 errors must stop after max_retries and return a safe error."""
        from src.generation import CLEAN_GENERATION_ERROR_MESSAGE, clear_generation_cache, generate_answer

        clear_generation_cache()
        tpm_exc = FakeRateLimitError(
            "Rate limit reached for gpt-4o-mini on tokens per min (TPM): Limit 40000.",
            status_code=429,
            code="rate_limit_exceeded",
            err_type="tokens",
            headers={"x-ratelimit-remaining-tokens": "0"},
        )

        fake_openai = types.ModuleType("openai")
        client_instance = MagicMock()
        client_instance.chat.completions.create.side_effect = tpm_exc
        fake_openai.OpenAI = MagicMock(return_value=client_instance)

        sleep_calls: List[float] = []

        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
                result = generate_answer(
                    "What is the CSE fee?",
                    SAMPLE_CHUNKS,
                    max_retries=2,
                    base_delay=0.5,
                    sleep_fn=sleep_calls.append,
                )

        assert client_instance.chat.completions.create.call_count == 3
        assert len(sleep_calls) == 2
        assert result["error"]["category"] == "tpm"
        assert result["error"]["retryable"] is True
        assert result["answer"] == CLEAN_GENERATION_ERROR_MESSAGE

    def test_retry_after_header_is_respected(self):
        """When Retry-After header is present on HTTP 429, backoff must wait at least that duration."""
        from src.generation import clear_generation_cache, generate_answer

        clear_generation_cache()
        retry_after_exc = FakeRateLimitError(
            "Rate limit reached. Please retry after 4.5s.",
            status_code=429,
            headers={"Retry-After": "4.5"},
        )

        fake_openai = types.ModuleType("openai")
        client_instance = MagicMock()
        client_instance.chat.completions.create.side_effect = [
            retry_after_exc,
            _make_openai_response("Succeeded after Retry-After header wait."),
        ]
        fake_openai.OpenAI = MagicMock(return_value=client_instance)

        sleep_calls: List[float] = []

        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
                result = generate_answer(
                    "What is the CSE fee?",
                    SAMPLE_CHUNKS,
                    max_retries=2,
                    base_delay=1.0,
                    sleep_fn=sleep_calls.append,
                )

        assert len(sleep_calls) == 1
        assert sleep_calls[0] >= 4.5
        assert result["answer"] == "Succeeded after Retry-After header wait."

    def test_duplicate_gpt_requests_are_served_from_cache(self):
        """Identical questions with identical context must use the cache and avoid duplicate API calls."""
        from src.generation import clear_generation_cache, generate_answer

        clear_generation_cache()
        fake_openai = _make_openai_module("Cached synthesized answer.")
        client_instance = fake_openai.OpenAI.return_value

        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
                res1 = generate_answer("What is the CSE fee?", SAMPLE_CHUNKS)
                res2 = generate_answer("  what is the CSE fee?  ", SAMPLE_CHUNKS)

        assert res1["answer"] == "Cached synthesized answer."
        assert res2["answer"] == "Cached synthesized answer."
        assert client_instance.chat.completions.create.call_count == 1

    def test_api_key_is_never_exposed_in_error_messages(self):
        """API keys accidentally included in exception strings must be redacted and never exposed."""
        from src.generation import (
            CLEAN_GENERATION_ERROR_MESSAGE,
            classify_rate_limit_error,
            clear_generation_cache,
            generate_answer,
        )

        clear_generation_cache()
        leaked_key = "sk-proj-SECRETKEY9876543210abcdef"
        raw_exc = RuntimeError(f"Invalid auth for key {leaked_key}")
        classified = classify_rate_limit_error(raw_exc)
        assert leaked_key not in classified["message"]
        assert "sk-***REDACTED***" in classified["message"]

        fake_openai = types.ModuleType("openai")
        client_instance = MagicMock()
        client_instance.chat.completions.create.side_effect = raw_exc
        fake_openai.OpenAI = MagicMock(return_value=client_instance)

        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": leaked_key}):
                result = generate_answer("What is the fee?", SAMPLE_CHUNKS)

        assert result["answer"] == CLEAN_GENERATION_ERROR_MESSAGE
        assert leaked_key not in result["answer"]
        assert leaked_key not in str(result.get("error"))

    def test_huggingface_fallback_succeeds_when_openai_quota_exhausted(self):
        """When OpenAI fails with insufficient_quota and HF fallback succeeds, return the HF answer."""
        from src.generation import clear_generation_cache, generate_answer

        clear_generation_cache()
        quota_exc = FakeRateLimitError(
            "Error code: 429 - You have no credits remaining.",
            status_code=429,
            code="credit_balance_exhausted",
            err_type="insufficient_quota",
        )

        fake_openai = types.ModuleType("openai")
        client_instance = MagicMock()
        client_instance.chat.completions.create.side_effect = quota_exc
        fake_openai.OpenAI = MagicMock(return_value=client_instance)

        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
                with patch(
                    "src.generation._try_huggingface_fallback",
                    return_value=("Synthesized via Hugging Face fallback.", None),
                ):
                    result = generate_answer("What is the CSE fee?", SAMPLE_CHUNKS)

        assert result["answer"] == "Synthesized via Hugging Face fallback."
        assert result["strategy"] == "gpt"
        assert "error" not in result




