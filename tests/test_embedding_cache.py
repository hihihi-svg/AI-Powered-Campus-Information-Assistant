"""Unit tests for SemanticRetriever embedding cache validation and fingerprinting.

Verifies:
1. Same corpus: reuses cached embeddings without calling model.encode.
2. Changed corpus with identical chunk count: invalidates cache and regenerates embeddings.
3. Changed corpus with different chunk count: invalidates cache and regenerates embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.retrieval import DocumentChunk, SemanticRetriever


def _create_mock_model(dim: int = 384):
    """Return a mock SentenceTransformer model whose encode() returns deterministically shaped numpy arrays."""
    mock = MagicMock()

    def _fake_encode(texts, **kwargs):
        arr = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            arr[i, 0] = float(len(t))
        return arr

    mock.encode.side_effect = _fake_encode
    return mock


def _make_chunk(chunk_id: str, document_id: str, text: str, title: str = "T1", heading: str = "H1") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source_url="https://rvce.edu.in",
        document_title=title,
        heading=heading,
        text=text,
        searchable_text=text,
    )


def test_embedding_cache_reuse_for_identical_corpus(tmp_path: Path):
    """When corpus content is identical, encode_corpus reuses cached embeddings without re-encoding."""
    cache_file = tmp_path / "test_embeddings.npy"

    chunks = [
        _make_chunk(chunk_id="c1", document_id="d1", text="RVCE tuition fees for KCET"),
        _make_chunk(chunk_id="c2", document_id="d1", text="RVCE hostel fees and allotment"),
    ]

    r1 = SemanticRetriever()
    r1.chunks = list(chunks)
    r1.is_indexed = True
    r1.model = _create_mock_model()

    emb1 = r1.encode_corpus(cache_path=str(cache_file))
    assert emb1 is not None
    assert emb1.shape == (2, 384)
    assert cache_file.exists()
    assert cache_file.with_suffix(".meta.json").exists()
    assert r1.model.encode.call_count == 1

    # Second retriever instance on the exact same chunks
    r2 = SemanticRetriever()
    r2.chunks = list(chunks)
    r2.is_indexed = True
    r2.model = _create_mock_model()

    emb2 = r2.encode_corpus(cache_path=str(cache_file))
    assert emb2 is not None
    assert emb2.shape == (2, 384)
    np.testing.assert_array_equal(emb1, emb2)
    # Model encode should NOT have been called because cache was reused
    assert r2.model.encode.call_count == 0


def test_embedding_cache_invalidates_when_content_changes_with_same_chunk_count(tmp_path: Path):
    """When chunk count is identical but chunk text changes, cache must invalidate and regenerate."""
    cache_file = tmp_path / "test_embeddings.npy"

    original_chunks = [
        _make_chunk(chunk_id="c1", document_id="d1", text="Original text for chunk one"),
        _make_chunk(chunk_id="c2", document_id="d1", text="Original text for chunk two"),
    ]

    r1 = SemanticRetriever()
    r1.chunks = list(original_chunks)
    r1.is_indexed = True
    r1.model = _create_mock_model()

    emb1 = r1.encode_corpus(cache_path=str(cache_file))
    assert emb1 is not None
    assert r1.model.encode.call_count == 1

    # Modify text of chunk 2, keeping chunk count at exactly 2
    modified_chunks = [
        _make_chunk(chunk_id="c1", document_id="d1", text="Original text for chunk one"),
        _make_chunk(chunk_id="c2", document_id="d1", text="Completely REVISED syllabus and deadline rules"),
    ]

    r2 = SemanticRetriever()
    r2.chunks = list(modified_chunks)
    r2.is_indexed = True
    r2.model = _create_mock_model()

    emb2 = r2.encode_corpus(cache_path=str(cache_file))
    assert emb2 is not None
    assert emb2.shape == (2, 384)
    # Model encode MUST have been called because content changed despite equal length
    assert r2.model.encode.call_count == 1

    # Verify updated metadata fingerprint
    meta = json.loads(cache_file.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["fingerprint"] == r2.compute_corpus_fingerprint()


def test_embedding_cache_invalidates_when_chunk_count_changes(tmp_path: Path):
    """When chunk count differs, cache must invalidate and regenerate."""
    cache_file = tmp_path / "test_embeddings.npy"

    initial_chunks = [
        _make_chunk(chunk_id="c1", document_id="d1", text="First chunk text"),
        _make_chunk(chunk_id="c2", document_id="d1", text="Second chunk text"),
    ]

    r1 = SemanticRetriever()
    r1.chunks = list(initial_chunks)
    r1.is_indexed = True
    r1.model = _create_mock_model()

    emb1 = r1.encode_corpus(cache_path=str(cache_file))
    assert emb1 is not None
    assert emb1.shape == (2, 384)

    # Expanded chunks (3 chunks instead of 2)
    expanded_chunks = list(initial_chunks) + [
        _make_chunk(chunk_id="c3", document_id="d2", text="Third new circular chunk"),
    ]

    r2 = SemanticRetriever()
    r2.chunks = list(expanded_chunks)
    r2.is_indexed = True
    r2.model = _create_mock_model()

    emb2 = r2.encode_corpus(cache_path=str(cache_file))
    assert emb2 is not None
    assert emb2.shape == (3, 384)
    assert r2.model.encode.call_count == 1

    meta = json.loads(cache_file.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["chunk_count"] == 3
    assert meta["fingerprint"] == r2.compute_corpus_fingerprint()
