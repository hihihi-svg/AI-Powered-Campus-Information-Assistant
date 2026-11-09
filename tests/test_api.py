from fastapi.testclient import TestClient

from src.answering import NO_EVIDENCE_MESSAGE, build_evidence_answer
from src.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_page_serves_frontend():
    response = client.get("/")
    assert response.status_code == 200
    assert "RVCE Campus Assistant" in response.text
    assert "campus documents" in response.text.lower()


def test_static_assets_are_served():
    response = client.get("/static/app.js")
    assert response.status_code == 200
    assert "askQuestion" in response.text


def test_ask_supported_question():
    response = client.post("/ask", json={"question": "What is the procedure for COMEDK rank card?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert ("COMEDK" in payload["answer"] or "COMED-K" in payload["answer"])
    assert payload["abstention_status"] is False
    assert payload["sources"]


def test_ask_unsupported_question_returns_abstention():
    response = client.post("/ask", json={"question": "What is the capital of France?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "I could not find sufficient evidence in the available campus documents to answer this question."
    assert payload["abstention_status"] is True
    assert payload["sources"] == []
    assert payload["evidence"] == []


def test_answer_question_requires_supporting_evidence():
    results = [
        {
            "document_title": "Fee Circular",
            "source_url": "https://example.com/fee",
            "heading": "Fee Circular",
            "text": "Students must pay fees by the deadline. The principal office reminds students to clear dues.",
            "relevance_score": 0.85,
        },
        {
            "document_title": "Principal Office",
            "source_url": "https://example.com/principal",
            "heading": "Principal's Office",
            "text": "Dr. K.N. Subramanya is the principal of RV College of Engineering.",
            "relevance_score": 0.42,
        },
    ]

    payload = build_evidence_answer(results, question="who is the current principal of the college?")
    assert payload["answer"]
    assert "Dr. K.N. Subramanya" in payload["answer"]
    assert payload["evidence"]
    assert payload["sources"]


def test_answer_question_abstains_when_top_match_is_unrelated():
    results = [
        {
            "document_title": "Academic Calendar",
            "source_url": "https://example.com/calendar",
            "heading": "Academic Calendars",
            "text": "Stay ahead and plan your semester effectively with the official Academic Calendar.",
            "relevance_score": 0.92,
        }
    ]

    payload = build_evidence_answer(results, question="What is the official contact email and reception phone number for RVCE?")
    assert payload["answer"] == NO_EVIDENCE_MESSAGE
    assert payload["evidence"] == []
    assert payload["sources"] == []


def test_build_evidence_answer_deduplicates_conflicting_passages():
    results = [
        {
            "document_title": "Principal Office",
            "source_url": "https://example.com/principal",
            "heading": "Principal's Office",
            "text": "Dr. K.N. Subramanya is the principal of RV College of Engineering.",
            "relevance_score": 0.90,
        },
        {
            "document_title": "Principal Office",
            "source_url": "https://example.com/principal",
            "heading": "Principal's Office",
            "text": "Dr. K.N. Subramanya is the principal of RV College of Engineering.",
            "relevance_score": 0.89,
        },
        {
            "document_title": "Principal Office",
            "source_url": "https://example.com/principal",
            "heading": "Principal's Office",
            "text": "Dr. K.N. Subramanya is the principal of RV College of Engineering.",
            "relevance_score": 0.80,
        },
    ]

    payload = build_evidence_answer(results, question="who is the current principal of the college?")
    assert len(payload["evidence"]) == 1
    assert len(payload["sources"]) == 1


def test_build_evidence_answer_handles_missing_source_metadata():
    results = [
        {
            "document_title": "",
            "source_url": "",
            "heading": "Principal's Office",
            "text": "Dr. K.N. Subramanya is the principal of RV College of Engineering.",
            "relevance_score": 0.70,
        }
    ]

    payload = build_evidence_answer(results, question="who is the current principal of the college?")
    assert payload["answer"]
    assert payload["sources"][0]["title"] == "Unknown source"
    assert payload["sources"][0]["url"] == "Unknown URL"


def test_ask_invalid_question_returns_400():
    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 400
    assert "Question must be a non-empty string." in response.json()["detail"]


def test_ask_with_gpt_strategy():
    from unittest.mock import patch

    mock_gpt_result = {
        "answer": "Mocked GPT Answer from campus context.",
        "evidence": [
            {
                "title": "Admissions Guide",
                "url": "https://rvce.edu.in/admissions",
                "heading": "Admission Process",
                "text": "Admissions are conducted through COMEDK and KEA.",
                "relevance_score": 0.9,
            }
        ],
        "sources": [{"title": "Admissions Guide", "url": "https://rvce.edu.in/admissions", "heading": "Admission Process"}],
        "strategy": "gpt",
    }

    with patch("src.api.answer_question", return_value=mock_gpt_result):
        response = client.post("/ask", json={"question": "How do admissions work?", "strategy": "gpt"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Mocked GPT Answer from campus context."
    assert payload["strategy"] == "gpt"
    assert payload["abstention_status"] is False
    assert len(payload["sources"]) == 1
    assert len(payload["evidence"]) == 1


def test_sync_status_endpoint():
    response = client.get("/sync/status")
    assert response.status_code == 200
    payload = response.json()
    assert "worker_running" in payload
    assert "interval_seconds" in payload
    assert "last_sync_status" in payload


def test_sync_trigger_endpoint_non_blocking():
    from unittest.mock import patch

    mock_summary = {
        "scanned": 4,
        "processed_new": 1,
        "skipped_existing": 3,
        "index_updated": True,
        "corpus_total_documents": 106,
    }
    with patch("src.api.gmail_sync_worker.sync_once", return_value=mock_summary) as mock_sync:
        response = client.post("/sync/trigger")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "scheduled"
    mock_sync.assert_called_once()


def test_ask_gpt_surfaces_rate_limit_error_to_client():
    from unittest.mock import patch
    from src.generation import CLEAN_GENERATION_ERROR_MESSAGE

    mock_error_result = {
        "answer": CLEAN_GENERATION_ERROR_MESSAGE,
        "evidence": [
            {
                "title": "RVCE Fee Structure",
                "url": "https://rvce.edu.in/fees",
                "heading": "CSE Fee Details",
                "text": "The tuition fee for CSE is Rs 1,85,000 per year.",
                "relevance_score": 0.75,
            }
        ],
        "sources": [{"title": "RVCE Fee Structure", "url": "https://rvce.edu.in/fees", "heading": "CSE Fee Details"}],
        "strategy": "gpt",
        "error": {
            "category": "insufficient_quota",
            "code": "credit_balance_exhausted",
            "error_code": "credit_balance_exhausted",
            "message": CLEAN_GENERATION_ERROR_MESSAGE,
            "retryable": False,
        },
    }

    with patch("src.api.answer_question", return_value=mock_error_result):
        response = client.post("/ask", json={"question": "What is the CSE fee?", "strategy": "gpt"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["strategy"] == "gpt"
    assert payload["error"] is not None
    assert payload["error"]["category"] == "insufficient_quota"
    assert payload["error"]["error_code"] == "credit_balance_exhausted"
    assert payload["error"]["retryable"] is False
    assert payload["answer"] == CLEAN_GENERATION_ERROR_MESSAGE
    assert "OpenAI" not in payload["answer"]
    assert "429" not in payload["answer"]



