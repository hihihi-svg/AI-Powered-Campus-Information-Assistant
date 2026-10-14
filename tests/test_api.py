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
