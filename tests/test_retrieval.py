"""
test_retrieval.py - Unit and integration tests for TF-IDF retrieval baseline.

Tests cover:
- Heading-aware document chunking
- Source URL, title, heading, and date metadata preservation
- Boilerplate chunk filtering
- Skipping empty / unextracted records
- Query ranking in descending score order
- Safe handling of empty, whitespace, and punctuation queries
- Safe handling of irrelevant / out-of-domain queries
- Safe handling of unindexed / empty dataset states
"""

import pytest

from src.retrieval import DocumentChunk, TFIDFRetriever, chunk_document


@pytest.fixture
def sample_records():
    """Sample records representing HTML pages and an empty PDF."""
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
                "First year B.E. management quota applications are open for approved seats.\n\n"
                "## Follow Us\n"
                "Designed by Department of Alumni Affairs & Communications, RSST"
            ),
            "source_type": "HTML",
            "collected_at": "2026-10-12T14:25:00Z",
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
                "Fast-track examinations are conducted in July for backlogs.\n\n"
                "## Grade Cards and Revaluation\n"
                "Students can apply for photocopy and revaluation within 7 days of results."
            ),
            "source_type": "HTML",
            "collected_at": "2026-10-12T14:26:00Z",
            "last_modified": None,
            "topic": "examinations_regulations",
            "extraction_status": "SUCCESS",
        },
        {
            "document_id": "doc_pdf_empty_03",
            "source_url": "https://rvce.edu.in/wp-content/uploads/scanned_fees.pdf",
            "title": "Scanned Fee Notice",
            "content": "",
            "source_type": "PDF",
            "collected_at": "2026-10-12T14:27:00Z",
            "last_modified": None,
            "topic": "fees_finance",
            "extraction_status": "EMPTY",
        },
    ]


# -----------------------------------------------------------------------------
# 1. Chunking Tests
# -----------------------------------------------------------------------------

def test_chunk_document_basic(sample_records):
    """Ensure document is divided into structured chunks by headings."""
    record = sample_records[0]
    chunks = chunk_document(record)

    assert len(chunks) >= 2
    for c in chunks:
        assert isinstance(c, DocumentChunk)
        assert c.document_id == "doc_html_admission_01"
        assert c.char_length > 0
        assert c.word_count > 0
        assert c.chunk_id.startswith("doc_html_admission_01_c")


def test_chunk_document_preserves_metadata(sample_records):
    """Verify source URL, title, heading, topic, and date metadata are preserved."""
    record = sample_records[0]
    chunks = chunk_document(record)

    for c in chunks:
        assert c.source_url == "https://rvce.edu.in/admission-circulars/"
        assert c.document_title == "Admission Circulars"
        assert c.topic == "admissions"
        assert c.last_modified == "Mon, 12 Oct 2026 12:00:00 GMT"
        assert c.heading in ("Admission Circulars", "COMEDK Quota", "Management Quota")


def test_chunk_document_filters_boilerplate(sample_records):
    """Ensure footer credits and boilerplate sections are excluded."""
    record = sample_records[0]
    chunks = chunk_document(record)

    for c in chunks:
        assert c.heading.lower() != "follow us"
        assert "department of alumni affairs" not in c.text.lower()


def test_chunk_document_empty_or_short():
    """Ensure empty or sub-threshold records yield zero chunks."""
    empty_record = {
        "document_id": "doc_empty",
        "source_url": "https://rvce.edu.in/empty",
        "title": "Empty Page",
        "content": "",
        "extraction_status": "EMPTY",
    }
    assert chunk_document(empty_record) == []

    stub_record = {
        "document_id": "doc_stub",
        "source_url": "https://rvce.edu.in/stub",
        "title": "Stub",
        "content": "Hi.",
        "extraction_status": "SUCCESS",
    }
    assert chunk_document(stub_record) == []


# -----------------------------------------------------------------------------
# 2. Indexing & Empty Record Filtering Tests
# -----------------------------------------------------------------------------

def test_retriever_skips_empty_extractions(sample_records):
    """Verify empty PDFs are never indexed as searchable content."""
    retriever = TFIDFRetriever()
    count = retriever.index_records(sample_records)

    assert count > 0
    # doc_pdf_empty_03 should not be in the chunks
    indexed_doc_ids = {c.document_id for c in retriever.chunks}
    assert "doc_pdf_empty_03" not in indexed_doc_ids
    assert "doc_html_admission_01" in indexed_doc_ids
    assert "doc_html_exam_02" in indexed_doc_ids


# -----------------------------------------------------------------------------
# 3. Ranking & Query Matching Tests
# -----------------------------------------------------------------------------

def test_retriever_ranking_relevant_query(sample_records):
    """Verify queries return top chunks sorted descending by score."""
    retriever = TFIDFRetriever()
    retriever.index_records(sample_records)

    results = retriever.retrieve("What is the procedure for COMEDK rank card?", top_k=3, score_threshold=0.01)

    assert len(results) >= 1
    # Check score descending order
    scores = [r["relevance_score"] for r in results]
    assert scores == sorted(scores, reverse=True)

    # First result should be the COMEDK chunk
    top_result = results[0]
    assert "comedk" in top_result["text"].lower() or "comedk" in top_result["heading"].lower()
    assert top_result["source_url"] == "https://rvce.edu.in/admission-circulars/"
    assert top_result["document_title"] == "Admission Circulars"
    assert top_result["relevance_score"] > 0.0


def test_retriever_attendance_query(sample_records):
    """Verify autonomous examination query ranks exam document first."""
    retriever = TFIDFRetriever()
    retriever.index_records(sample_records)

    results = retriever.retrieve("minimum attendance percentage for autonomous semester exams", top_k=2)

    assert len(results) >= 1
    top_result = results[0]
    assert "attendance" in top_result["text"].lower()
    assert top_result["source_url"] == "https://rvce.edu.in/academic-circular/"
    assert top_result["topic"] == "examinations_regulations"


# -----------------------------------------------------------------------------
# 4. Safe Query Handling Tests (Empty / Irrelevant / Unindexed)
# -----------------------------------------------------------------------------

def test_retriever_empty_queries(sample_records):
    """Verify empty, whitespace, and punctuation queries safely return empty list."""
    retriever = TFIDFRetriever()
    retriever.index_records(sample_records)

    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []
    assert retriever.retrieve("???!!!...") == []
    assert retriever.retrieve(None) == []  # type: ignore


def test_retriever_unknown_or_irrelevant_query(sample_records):
    """Verify out-of-domain query returns zero results or below threshold."""
    retriever = TFIDFRetriever()
    retriever.index_records(sample_records)

    # Out-of-domain query with no vocabulary overlap
    results = retriever.retrieve("astrophysics quantum tunneling in neutron star cores", top_k=5, score_threshold=0.1)
    assert results == []


def test_retriever_unindexed_state():
    """Verify retriever safely returns empty list if called before indexing."""
    retriever = TFIDFRetriever()
    assert retriever.retrieve("admissions") == []

    # Empty records list
    count = retriever.index_records([])
    assert count == 0
    assert retriever.retrieve("admissions") == []
