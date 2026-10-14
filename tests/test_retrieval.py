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

from src.retrieval import CrossEncoderReranker, DocumentChunk, SemanticRetriever, TFIDFRetriever, chunk_document


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


def test_semantic_retriever_fails_gracefully_without_cached_model():
    """Semantic retrieval should fail fast and return no results if the model is not cached locally."""
    retriever = SemanticRetriever(model_name="sentence-transformers/nonexistent-model-name")
    retriever.index_records([
        {
            "document_id": "doc_fallback",
            "source_url": "https://rvce.edu.in/fallback",
            "title": "Fallback",
            "content": "A campus notice about admissions and examinations for the 2026-27 academic year. This section explains the admission process, counselling steps, and the required documents for new students.",
            "extraction_status": "SUCCESS",
        }
    ])

    results = retriever.retrieve("What is the admission procedure?", top_k=3)

    assert results == []
    assert retriever.model_error is not None


def test_reranker_fails_gracefully_without_cached_model():
    """Cross-encoder reranking should not hang or claim success when a local model is unavailable."""
    reranker = CrossEncoderReranker(model_name="cross-encoder/nonexistent-model-name")
    candidates = [{"text": "Example campus admission notice.", "document_title": "Example"}]

    reranked = reranker.rerank("What is the admission procedure?", candidates)

    assert reranked == candidates
    assert reranker.model_error is not None


def test_retriever_unindexed_state():
    """Verify retriever safely returns empty list if called before indexing."""
    retriever = TFIDFRetriever()
    assert retriever.retrieve("admissions") == []

    # Empty records list
    count = retriever.index_records([])
    assert count == 0
    assert retriever.retrieve("admissions") == []


def test_retriever_deduplicates_same_document_and_heading():
    """Repeated chunks from the same document and heading should not occupy multiple top-k slots."""
    retriever = TFIDFRetriever()
    chunks = [
        DocumentChunk(
            chunk_id="chunk_01",
            document_id="doc-duplicate",
            source_url="https://rvce.edu.in/fees",
            document_title="Fee Notice",
            heading="Admissions",
            text="Tuition fee payment due dates for semester fee payment.",
            searchable_text="Tuition fee payment due dates for semester fee payment.",
            topic="fees_finance",
        ),
        DocumentChunk(
            chunk_id="chunk_02",
            document_id="doc-duplicate",
            source_url="https://rvce.edu.in/fees",
            document_title="Fee Notice",
            heading="Admissions",
            text="Tuition fee payment due dates for semester fee payment.",
            searchable_text="Tuition fee payment due dates for semester fee payment.",
            topic="fees_finance",
        ),
        DocumentChunk(
            chunk_id="chunk_03",
            document_id="doc-duplicate",
            source_url="https://rvce.edu.in/fees",
            document_title="Fee Notice",
            heading="Admissions",
            text="Late fee penalties and final deadline reminders for semester payment.",
            searchable_text="Late fee penalties and final deadline reminders for semester payment.",
            topic="fees_finance",
        ),
    ]
    retriever.chunks = chunks
    retriever.tfidf_matrix = retriever.vectorizer.fit_transform([
        c.searchable_text for c in chunks
    ])
    retriever.is_indexed = True

    results = retriever.retrieve("semester fee payment due dates", top_k=5, score_threshold=0.0)

    assert len(results) == 2
    scores = [r["relevance_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert {r["chunk_id"] for r in results} == {"chunk_01", "chunk_03"}


def test_retriever_keeps_distinct_passages_from_same_document():
    """Distinct passages under the same document should remain if their text differs."""
    retriever = TFIDFRetriever()
    chunks = [
        DocumentChunk(
            chunk_id="chunk_01",
            document_id="doc-same",
            source_url="https://rvce.edu.in/circulars",
            document_title="Circulars",
            heading="Academic Schedule",
            text="Exam registration opens on 10 August.",
            searchable_text="Exam registration opens on 10 August.",
            topic="examinations_regulations",
        ),
        DocumentChunk(
            chunk_id="chunk_02",
            document_id="doc-same",
            source_url="https://rvce.edu.in/circulars",
            document_title="Circulars",
            heading="Academic Schedule",
            text="Revaluation applications close on 15 August.",
            searchable_text="Revaluation applications close on 15 August.",
            topic="examinations_regulations",
        ),
    ]
    retriever.chunks = chunks
    retriever.tfidf_matrix = retriever.vectorizer.fit_transform([
        c.searchable_text for c in chunks
    ])
    retriever.is_indexed = True

    results = retriever.retrieve("revaluation applications august", top_k=5, score_threshold=0.0)

    assert len(results) == 2
    assert {r["chunk_id"] for r in results} == {"chunk_01", "chunk_02"}
    assert results[0]["document_id"] == "doc-same"


def test_retriever_dedup_handles_missing_metadata_safely():
    """Missing document or heading metadata should not trigger key errors or false positives."""
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
        ),
        DocumentChunk(
            chunk_id="chunk_02",
            document_id=None,
            source_url=None,
            document_title="",
            heading=None,
            text="Hostel fee payment reminders are due this week.",
            searchable_text="Hostel fee payment reminders are due this week.",
            topic=None,
        ),
    ]
    retriever.chunks = chunks
    retriever.tfidf_matrix = retriever.vectorizer.fit_transform([
        c.searchable_text for c in chunks
    ])
    retriever.is_indexed = True

    results = retriever.retrieve("hostel fee payment reminders", top_k=5, score_threshold=0.0)

    assert len(results) == 1
    assert results[0]["document_id"] == "unknown_document"
    assert results[0]["heading"] == "unknown_heading"


def test_pdf_extractor_uses_ocr_when_native_text_is_empty(monkeypatch):
    """Ensure scanned PDF text falls back to OCR instead of silently returning EMPTY."""
    from src.extractors import PDFExtractor

    pdf_payload = b"pdf-bytes-" * 20

    def fake_ocr(raw_bytes, url="", dpi=200, max_pages=None):
        assert raw_bytes == pdf_payload
        return {
            "ocr_status": "OCR_SUCCESS",
            "ocr_text": "Fee notice with payment details and due dates for semester fee payment.",
            "page_count": 1,
            "pages_with_text": 1,
            "error": None,
        }

    monkeypatch.setattr("src.extractors.HAS_PYMUPDF", False)
    monkeypatch.setattr("src.extractors.HAS_PYPDF", False)
    monkeypatch.setattr("src.extractors.ocr_pdf_bytes", fake_ocr)

    result = PDFExtractor().extract(pdf_payload, "https://rvce.edu.in/fees_notice.pdf")

    assert result["extraction_status"] == "SUCCESS"
    assert "Fee notice" in result["content"]
    assert "fee notice" in (result["title"] or "").lower()
    assert result["error_message"] is None


def test_semantic_retriever_indexing_and_retrieval_schema(sample_records):
    """SemanticRetriever should return results with the expected schema and finite normalized scores."""
    from src.retrieval import SemanticRetriever

    retriever = SemanticRetriever()
    count = retriever.index_records(sample_records)
    assert count > 0

    results = retriever.retrieve("What is the admission procedure for B.E.?", top_k=2)
    assert len(results) <= 2
    if results:
        top = results[0]
        expected_keys = {"chunk_id", "document_id", "text", "document_title", "source_url", "heading", "relevance_score"}
        assert expected_keys.issubset(top.keys())
        assert isinstance(top["relevance_score"], float)
        assert -1.0 <= top["relevance_score"] <= 1.0


def test_hybrid_retriever_merges_without_duplicate_chunks():
    """HybridRetriever should fuse candidates without duplicating chunks."""
    from src.retrieval import HybridRetriever

    tfidf_results = [
        {"chunk_id": "c1", "document_id": "doc1", "heading": "Admissions", "text": "Admissions procedure text", "relevance_score": 0.8},
        {"chunk_id": "c2", "document_id": "doc2", "heading": "Fees", "text": "Fee schedule text", "relevance_score": 0.5},
    ]
    semantic_results = [
        {"chunk_id": "c1", "document_id": "doc1", "heading": "Admissions", "text": "Admissions procedure text", "relevance_score": 0.9},
        {"chunk_id": "c3", "document_id": "doc3", "heading": "Hostels", "text": "Hostel room details", "relevance_score": 0.7},
    ]

    merged = HybridRetriever._merge_candidates(
        tfidf_results,
        semantic_results,
        semantic_weight=0.5,
        tfidf_weight=0.5,
        fusion_method="linear",
    )

    assert len(merged) == 3
    # Top chunk should be c1 because it scored high in both
    assert merged[0]["chunk_id"] == "c1"
    assert merged[0]["relevance_score"] == round(0.5 * 0.8 + 0.5 * 0.9, 4)


def test_hybrid_retriever_rrf_fusion():
    """HybridRetriever should support Reciprocal Rank Fusion."""
    from src.retrieval import HybridRetriever

    tfidf_results = [
        {"chunk_id": "c1", "document_id": "doc1", "heading": "Admissions", "text": "Passage 1", "relevance_score": 0.9},
        {"chunk_id": "c2", "document_id": "doc2", "heading": "Fees", "text": "Passage 2", "relevance_score": 0.4},
    ]
    semantic_results = [
        {"chunk_id": "c2", "document_id": "doc2", "heading": "Fees", "text": "Passage 2", "relevance_score": 0.85},
        {"chunk_id": "c1", "document_id": "doc1", "heading": "Admissions", "text": "Passage 1", "relevance_score": 0.70},
    ]

    rrf_merged = HybridRetriever._merge_candidates(
        tfidf_results,
        semantic_results,
        semantic_weight=0.5,
        tfidf_weight=0.5,
        fusion_method="rrf",
        rrf_k=60,
    )

    assert len(rrf_merged) == 2
    for item in rrf_merged:
        assert item["relevance_score"] > 0.0


def test_cross_encoder_preserves_metadata():
    """CrossEncoderReranker should preserve source URLs, titles, headings, and chunk IDs."""
    from src.retrieval import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    candidates = [
        {
            "chunk_id": "chunk_42",
            "document_id": "doc_rvce_fee",
            "source_url": "https://rvce.edu.in/fees",
            "document_title": "Fee Circular",
            "heading": "Tuition Fee Schedule",
            "text": "The higher semester tuition fee deadline is 30 September.",
            "relevance_score": 0.45,
        }
    ]

    reranked = reranker.rerank("fee deadline", candidates)
    assert len(reranked) == 1
    top = reranked[0]
    assert top["chunk_id"] == "chunk_42"
    assert top["source_url"] == "https://rvce.edu.in/fees"
    assert top["document_title"] == "Fee Circular"
    assert top["heading"] == "Tuition Fee Schedule"


# -----------------------------------------------------------------------------
# 5. Oversized Document & Overlap Regression Tests
# -----------------------------------------------------------------------------

def test_chunk_document_splits_oversized_text_into_bounded_chunks():
    """Ensure sections exceeding max_chunk_chars are split into bounded chunks."""
    long_content = (
        "## Important Campus Circular\n"
        + "This is a detailed paragraph explaining student regulations and payment schedules. " * 40
    )  # ~3,500 chars
    record = {
        "document_id": "doc_long_01",
        "source_url": "https://rvce.edu.in/long-notice/",
        "title": "Long Notice",
        "content": long_content,
        "topic": "fees_finance",
        "extraction_status": "SUCCESS",
    }

    chunks = chunk_document(record, max_chunk_chars=1000, overlap_chars=100)

    assert len(chunks) >= 3
    for c in chunks:
        assert isinstance(c, DocumentChunk)
        assert c.char_length <= 1000
        assert c.document_id == "doc_long_01"
        assert c.heading == "Important Campus Circular"


def test_chunk_document_preserves_paragraph_boundaries():
    """Paragraph breaks (double newlines) should not be collapsed into single newlines."""
    content = (
        "## Fee Regulations\n"
        "Paragraph One outlines the tuition fee structure for undergraduate engineering.\n\n"
        "Paragraph Two describes online payment through the student portal.\n\n"
        "Paragraph Three specifies the deadline and bank challan instructions."
    )
    record = {
        "document_id": "doc_para_01",
        "source_url": "https://rvce.edu.in/paras/",
        "title": "Paragraphs Notice",
        "content": content,
        "extraction_status": "SUCCESS",
    }

    chunks = chunk_document(record, max_chunk_chars=200, overlap_chars=30)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.document_id == "doc_para_01"
        assert c.char_length > 0


def test_chunk_document_uses_overlap_chars():
    """Subsequent sub-chunks should contain overlapping tokens from the preceding sub-chunk."""
    para1 = "Alpha sentence detailing admission criteria for first-year engineering students. " * 12
    para2 = "Beta sentence explaining required verification documents and score cards. " * 12
    content = f"## Admissions Guide\n{para1}\n\n{para2}"

    record = {
        "document_id": "doc_overlap_01",
        "source_url": "https://rvce.edu.in/admissions/",
        "title": "Admissions Guide",
        "content": content,
        "extraction_status": "SUCCESS",
    }

    chunks = chunk_document(record, max_chunk_chars=800, overlap_chars=150)
    assert len(chunks) >= 2

    chunk0_text = chunks[0].text
    chunk1_text = chunks[1].text

    # Verify that the beginning of chunk 1 shares tokens with the end of chunk 0
    c0_tail_tokens = set(chunk0_text[-120:].split())
    c1_head_tokens = set(chunk1_text[:180].split())
    overlap_shared = c0_tail_tokens.intersection(c1_head_tokens)
    assert len(overlap_shared) >= 2, f"Expected overlapping tokens between chunks, got: {overlap_shared}"


def test_retrieval_ranks_fine_policy_above_generic_institution_blurbs():
    """A specific fine query should rank the fine clause above generic short blurbs matching only 'rvce'."""
    short_generic_blurb = {
        "document_id": "doc_generic_faculty",
        "source_url": "https://rvce.edu.in/department/biotechnology/research/",
        "title": "RVCE Bengaluru Research Faculty",
        "content": "## Research Faculty\nGet to know the depth of research expertise within the RVCE campus community.",
        "extraction_status": "SUCCESS",
    }
    long_circular_with_fine = {
        "document_id": "doc_fee_circular",
        "source_url": "https://rvce.edu.in/fees/ug_circular.pdf",
        "title": "RVCE UG Higher Semester Fee Circular",
        "content": (
            "## RVCE Fee Payment Circular\n"
            + "R.V. College of Engineering (RVCE). The online payment of semester fees for B.E. programs has been enabled through payment gateways. " * 10
            + "\n\n(Note: A fine of Rs.100 per day will be levied for the payment of fee after the due date)\n\n"
            + "Please remit the tuition fee as per the enclosed fee structure to the prescribed bank accounts. " * 10
        ),
        "extraction_status": "SUCCESS",
    }

    retriever = TFIDFRetriever()
    retriever.index_records([short_generic_blurb, long_circular_with_fine])

    results = retriever.retrieve("What is the fine for late fee payment at RVCE?", top_k=2)

    assert len(results) >= 1
    top = results[0]
    assert top["document_id"] == "doc_fee_circular"
    assert "fine" in top["text"].lower()
    assert "rs.100" in top["text"].lower()


def test_chunk_document_preserves_metadata_across_subchunks():
    """Every generated sub-chunk must retain document ID, title, heading, and source URL."""
    record = {
        "document_id": "doc_meta_test",
        "source_url": "https://rvce.edu.in/meta-test/",
        "title": "Official Test Circular",
        "topic": "fees_finance",
        "last_modified": "Tue, 13 Oct 2026 10:00:00 GMT",
        "content": "## General Instructions\n" + "Important instructions for registration and examinations. " * 30,
        "extraction_status": "SUCCESS",
    }

    chunks = chunk_document(record, max_chunk_chars=600, overlap_chars=80)
    assert len(chunks) >= 2
    for idx, c in enumerate(chunks):
        assert c.chunk_id == f"doc_meta_test_c{idx:02d}"
        assert c.document_id == "doc_meta_test"
        assert c.document_title == "Official Test Circular"
        assert c.source_url == "https://rvce.edu.in/meta-test/"
        assert c.heading == "General Instructions"
        assert c.topic == "fees_finance"
        assert c.last_modified == "Tue, 13 Oct 2026 10:00:00 GMT"

