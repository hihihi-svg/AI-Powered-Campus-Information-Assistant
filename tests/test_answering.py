import pytest

from src.answering import answer_question
from src.retrieval import DocumentChunk, TFIDFRetriever


@pytest.fixture
def sample_records():
    return [
        {
            "document_id": "doc_html_admission_01",
            "source_url": "https://rvce.edu.in/admission-circulars/",
            "title": "Admission Circulars",
            "content": (
                "## Admission Circulars\n"
                "Details regarding entrance exams for 2026-27.\n\n"
                "### COMEDK Quota\n"
                "Candidates joining through COMEDK quota must submit original rank cards.\n"
                "Balance fees must be paid via online net banking challan.\n\n"
                "### Management Quota\n"
                "First year B.E. management quota applications are open for approved seats.\n"
            ),
            "source_type": "HTML",
            "last_modified": "Mon, 12 Oct 2026 12:00:00 GMT",
            "topic": "admissions",
            "extraction_status": "SUCCESS",
        },
        {
            "document_id": "doc_html_exam_02",
            "source_url": "https://rvce.edu.in/academic-circular/",
            "title": "Academic Circular",
            "content": (
                "# Academic Circular\n"
                "## Autonomous Examination Scheme\n"
                "All B.E. students must maintain minimum 85% attendance for semester end exams.\n"
                "Fast-track examinations are conducted in July for backlogs.\n"
            ),
            "source_type": "HTML",
            "last_modified": None,
            "topic": "examinations_regulations",
            "extraction_status": "SUCCESS",
        },
    ]


def test_answer_question_supported_question(sample_records):
    retriever = TFIDFRetriever()
    retriever.index_records(sample_records)

    result = answer_question("What is the procedure for COMEDK rank card?", retriever=retriever)

    assert result["answer"]
    assert "COMEDK" in result["answer"]
    assert result["evidence"]
    assert result["answer"] != "I could not find sufficient evidence in the available campus documents to answer this question."


def test_answer_question_includes_source_attribution(sample_records):
    retriever = TFIDFRetriever()
    retriever.index_records(sample_records)

    result = answer_question("What is the procedure for COMEDK rank card?", retriever=retriever)

    assert any(source["title"] == "Admission Circulars" for source in result["sources"])
    assert any(source["url"] == "https://rvce.edu.in/admission-circulars/" for source in result["sources"])


def test_answer_question_returns_missing_evidence_message_for_unsupported_query(sample_records):
    retriever = TFIDFRetriever()
    retriever.index_records(sample_records)

    result = answer_question("What is the capital of France?", retriever=retriever)

    assert result["answer"] == "I could not find sufficient evidence in the available campus documents to answer this question."
    assert result["evidence"] == []
    assert result["sources"] == []


def test_answer_question_empty_input(sample_records):
    retriever = TFIDFRetriever()
    retriever.index_records(sample_records)

    result = answer_question("   ", retriever=retriever)

    assert result["answer"] == "I could not find sufficient evidence in the available campus documents to answer this question."
    assert result["evidence"] == []


def test_answer_question_handles_missing_source_metadata():
    retriever = TFIDFRetriever()
    chunks = [
        DocumentChunk(
            chunk_id="chunk_01",
            document_id=None,
            source_url=None,
            document_title="",
            heading=None,
            text="Hostel fee payment reminders are due this week.",
            searchable_text="Hostel fee payment reminders are due this week.",
            topic=None,
        )
    ]
    retriever.chunks = chunks
    retriever.tfidf_matrix = retriever.vectorizer.fit_transform([c.searchable_text for c in chunks])
    retriever.is_indexed = True

    result = answer_question("hostel fee payment reminders", retriever=retriever)

    assert result["answer"]
    assert any(source["title"] == "Unknown source" for source in result["sources"])
    assert any(source["url"] == "Unknown URL" for source in result["sources"])
