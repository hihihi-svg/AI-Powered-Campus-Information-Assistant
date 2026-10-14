from __future__ import annotations

"""
retrieval.py - TF-IDF Retrieval Baseline for Campus Knowledge Base.

Features:
- Markdown and section-aware document chunking
- Preservation of title, source URL, section heading, and date metadata
- Strict filtering of empty PDF/HTML extractions (no silent indexing of empty records)
- TF-IDF vectorization and cosine similarity scoring using scikit-learn
- Safe query validation (handling empty queries, non-alphanumeric queries, empty indices)
- Low-relevance filtering without fabricating answers
"""

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("campus_pipeline.retrieval")

DEFAULT_SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
_SEMANTIC_MODEL_CACHE: Dict[str, Any] = {}
_CROSS_ENCODER_CACHE: Dict[str, Any] = {}


def _is_model_cached_locally(model_name: str) -> bool:
    """Check the standard Hugging Face cache directory for a usable local copy."""
    cache_root = os.path.expanduser("~/.cache/huggingface/hub")
    if not os.path.isdir(cache_root):
        return False

    normalized_name = model_name.replace("/", "--")
    marker_files = {"config.json", "sentence_bert_config.json", "tokenizer_config.json", "model.safetensors", "pytorch_model.bin"}
    for entry in os.listdir(cache_root):
        if not entry.startswith(f"models--{normalized_name}"):
            continue
        candidate_dir = os.path.join(cache_root, entry)
        if not os.path.isdir(candidate_dir):
            continue
        for root, _, files in os.walk(candidate_dir):
            if set(files) & marker_files:
                return True
    return False


def _ensure_model_is_cached(model_name: str) -> None:
    """Validate that the model is already cached locally before using it."""
    if not _is_model_cached_locally(model_name):
        raise RuntimeError(f"Local model cache missing for {model_name}")


def _load_sentence_transformer(
    model_name: str = DEFAULT_SEMANTIC_MODEL,
    device: str = "cpu",
    local_files_only: bool = True,
) -> Any:
    """Load a single SentenceTransformer instance and cache it for reuse.

    The default local-only mode prevents accidental hangs when the model has not
    been downloaded yet. If the cached model is missing, the caller should fail
    gracefully and log a clear warning instead of silently pretending retrieval
    succeeded.
    """
    from sentence_transformers import SentenceTransformer

    _ensure_model_is_cached(model_name)
    cache_key = f"{model_name}:{device}:{local_files_only}"
    if cache_key not in _SEMANTIC_MODEL_CACHE:
        model = SentenceTransformer(model_name, device=device, local_files_only=local_files_only)
        _SEMANTIC_MODEL_CACHE[cache_key] = model
    return _SEMANTIC_MODEL_CACHE[cache_key]


def _load_cross_encoder(
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
    device: str = "cpu",
    local_files_only: bool = True,
) -> Any:
    """Load a single CrossEncoder instance and cache it for reuse."""
    from sentence_transformers import CrossEncoder

    _ensure_model_is_cached(model_name)
    cache_key = f"{model_name}:{device}:{local_files_only}"
    if cache_key not in _CROSS_ENCODER_CACHE:
        model = CrossEncoder(model_name, device=device, local_files_only=local_files_only)
        _CROSS_ENCODER_CACHE[cache_key] = model
    return _CROSS_ENCODER_CACHE[cache_key]


def _index_record_chunks(records: List[Dict[str, Any]]) -> List[DocumentChunk]:
    """Shared record-to-chunk conversion used by TF-IDF and semantic indexes."""
    all_chunks: List[DocumentChunk] = []
    for r in records:
        if r.get("extraction_status") == "EMPTY":
            continue
        content = r.get("content", "")
        if not content or not content.strip():
            continue
        all_chunks.extend(chunk_document(r))
    return all_chunks


@dataclass
class DocumentChunk:
    """Represents a discrete semantic chunk of a campus document."""

    chunk_id: str
    document_id: str
    source_url: str
    document_title: str
    heading: str
    text: str
    searchable_text: str
    topic: Optional[str] = None
    last_modified: Optional[str] = None
    char_length: int = 0
    word_count: int = 0


def chunk_document(
    record: Dict[str, Any],
    max_chunk_chars: int = 1200,
    min_chunk_chars: int = 35,
    overlap_chars: int = 150,
) -> List[DocumentChunk]:
    """
    Split a document into structured, heading-aware chunks.

    Preserves:
    - Document title
    - Source URL
    - Section hierarchy / headings
    - Last-modified metadata
    - Topic label
    """
    doc_id = record.get("document_id", "unknown_doc")
    source_url = record.get("source_url", "")
    title = (record.get("title") or "RVCE Document").strip()
    topic = record.get("topic")
    last_mod = record.get("last_modified")
    content = record.get("content", "").strip()

    if not content or len(content) < min_chunk_chars:
        return []

    # Split content by markdown headings (#, ##, ###, ####)
    heading_pattern = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)
    splits = heading_pattern.split(content)

    sections: List[Tuple[str, str]] = []  # (heading, text_block)
    current_heading = title

    idx = 0
    while idx < len(splits):
        part = splits[idx].strip()
        if not part:
            idx += 1
            continue

        if heading_pattern.match(part):
            current_heading = re.sub(r"^#+\s*", "", part).strip()
            # The next element (if exists) is the content under this heading
            if idx + 1 < len(splits):
                body = splits[idx + 1].strip()
                sections.append((current_heading, body))
                idx += 2
            else:
                sections.append((current_heading, ""))
                idx += 1
        else:
            # Preamble before the first heading
            sections.append((current_heading, part))
            idx += 1

    # Headings that identify site-wide sidebar widgets or footer blocks present on
    # virtually every RVCE page.  None of these sections contain substantive
    # campus information; they are navigation menus, contact sidebars, or branding.
    BOILERPLATE_HEADINGS = {
        # Original set
        "follow us", "quick links", "useful links", "connect with us",
        "social media", "designed by", "privacy policy", "terms of use", "disclaimer",
        # Site-wide sidebar widgets discovered during smoke test (Oct 2026)
        "academic student services",  # sidebar nav menu (Fee & Notifications / Scheme & Syllabus / Placements)
        "address:",                   # footer address widget
        "address",                    # alternate form
        # NOTE: Principal office cards contain legitimate answer-bearing text and
        # must remain searchable for questions about the current principal.
        "contact us",
        "about rvce",
        "about college",
        "key executives",
        "statutory committees",
        "anti ragging",
        "faq",
    }

    chunks: List[DocumentChunk] = []
    chunk_counter = 0

    for heading, text_block in sections:
        # Avoid boilerplate sections
        h_norm = heading.strip().lower()
        if h_norm in BOILERPLATE_HEADINGS:
            continue

        # Avoid empty stubs or pure punctuation
        cleaned_block = re.sub(r"[\r\n]+", "\n", text_block).strip()
        if not cleaned_block:
            continue

        # Filter out footer credits or navigation-only stubs
        if re.search(r"^designed by\b|department of alumni affairs & communications", cleaned_block, re.IGNORECASE):
            continue

        # Filter out navigation-list stubs: blocks where every non-empty line is
        # very short (≤ 30 chars) — these are sidebar link labels, not content.
        non_empty_lines = [ln for ln in cleaned_block.splitlines() if ln.strip()]
        if non_empty_lines and all(len(ln.strip()) <= 30 for ln in non_empty_lines):
            continue

        # If block is reasonably sized, keep as one cohesive chunk
        if len(cleaned_block) <= max_chunk_chars:
            if len(cleaned_block) >= min_chunk_chars:
                chunk_id = f"{doc_id}_c{chunk_counter:02d}"
                searchable = f"{title} | {heading}\n{cleaned_block}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        source_url=source_url,
                        document_title=title,
                        heading=heading,
                        text=cleaned_block,
                        searchable_text=searchable,
                        topic=topic,
                        last_modified=last_mod,
                        char_length=len(cleaned_block),
                        word_count=len(cleaned_block.split()),
                    )
                )
                chunk_counter += 1
        else:
            # Paragraph-based sub-chunking with overlap for long sections
            paragraphs = [p.strip() for p in cleaned_block.split("\n\n") if p.strip()]
            current_sub = ""
            for p in paragraphs:
                if len(current_sub) + len(p) + 2 <= max_chunk_chars:
                    current_sub = f"{current_sub}\n\n{p}".strip() if current_sub else p
                else:
                    if len(current_sub) >= min_chunk_chars:
                        chunk_id = f"{doc_id}_c{chunk_counter:02d}"
                        searchable = f"{title} | {heading}\n{current_sub}"
                        chunks.append(
                            DocumentChunk(
                                chunk_id=chunk_id,
                                document_id=doc_id,
                                source_url=source_url,
                                document_title=title,
                                heading=heading,
                                text=current_sub,
                                searchable_text=searchable,
                                topic=topic,
                                last_modified=last_mod,
                                char_length=len(current_sub),
                                word_count=len(current_sub.split()),
                            )
                        )
                        chunk_counter += 1
                    current_sub = p

            if len(current_sub) >= min_chunk_chars:
                chunk_id = f"{doc_id}_c{chunk_counter:02d}"
                searchable = f"{title} | {heading}\n{current_sub}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        source_url=source_url,
                        document_title=title,
                        heading=heading,
                        text=current_sub,
                        searchable_text=searchable,
                        topic=topic,
                        last_modified=last_mod,
                        char_length=len(current_sub),
                        word_count=len(current_sub.split()),
                    )
                )
                chunk_counter += 1

    return chunks


class TFIDFRetriever:
    """TF-IDF retrieval engine over processed campus document chunks."""

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 20000,
        sublinear_tf: bool = True,
    ):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z0-9_\-\.]{2,}\b",
        )
        self.chunks: List[DocumentChunk] = []
        self.tfidf_matrix = None
        self.is_indexed: bool = False

    def index_records(self, records: List[Dict[str, Any]]) -> int:
        """
        Chunk and index records with non-empty content.
        Skips records flagged as EMPTY or with empty content.
        """
        all_chunks = _index_record_chunks(records)

        if not all_chunks:
            logger.warning("No valid non-empty chunks found to index.")
            self.chunks = []
            self.tfidf_matrix = None
            self.is_indexed = False
            return 0

        corpus = [c.searchable_text for c in all_chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.chunks = all_chunks
        self.is_indexed = True

        logger.info(f"Indexed {len(self.chunks)} chunks from {len(records)} records.")
        return len(self.chunks)

    def load_and_index(self, dataset_path: str = "data/processed/campus_documents.jsonl") -> int:
        """Load JSONL dataset and build retrieval index."""
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        records = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        return self.index_records(records)

    @staticmethod
    def _normalize_dedupe_text(value: Any) -> str:
        """Normalize candidate text for stable deduplication."""
        raw = str(value or "").strip().lower()
        raw = re.sub(r"[\W_]+", " ", raw)
        return re.sub(r"\s+", " ", raw).strip()

    @classmethod
    def _dedup_key_for_result(cls, result: Dict[str, Any]) -> Tuple[str, str, str]:
        """Return a stable deduplication key for a ranked result.

        It includes document identity, heading, and a normalized text signature so
        repeated copies of the exact same passage are suppressed without removing
        distinct passages from the same document.
        """
        document_id = (
            result.get("document_id")
            or result.get("source_url")
            or result.get("document_title")
            or "unknown_document"
        )
        heading = result.get("heading") or "unknown_heading"
        text = result.get("text") or result.get("searchable_text") or ""
        return (
            str(document_id).strip() or "unknown_document",
            str(heading).strip() or "unknown_heading",
            cls._normalize_dedupe_text(text),
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant chunks for a natural-language question.

        Returns list of dicts with:
        - chunk_id
        - document_id
        - text
        - document_title
        - source_url
        - heading
        - topic
        - last_modified
        - relevance_score
        """
        # Safe handling of empty queries or unindexed states
        if not query or not isinstance(query, str) or not query.strip():
            return []

        if not self.is_indexed or not self.chunks or self.tfidf_matrix is None:
            return []

        # Check if query contains any alphanumeric character
        if not re.search(r"[a-zA-Z0-9]", query):
            return []

        try:
            query_vec = self.vectorizer.transform([query])
        except Exception as e:
            logger.warning(f"Error vectorizing query '{query}': {e}")
            return []

        # Compute cosine similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Check for zero similarity everywhere
        if np.all(similarities <= 0.0):
            return []

        # Consider all ranked candidates so deduplication can suppress duplicates
        # without reducing the number of genuinely distinct passages returned.
        top_indices = np.argsort(-similarities)

        ranked_results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < score_threshold:
                continue

            chunk = self.chunks[idx]
            result = {
                "chunk_id": chunk.chunk_id,
                "document_id": str(chunk.document_id or chunk.source_url or chunk.document_title or "unknown_document").strip() or "unknown_document",
                "text": chunk.text,
                "document_title": chunk.document_title or "",
                "source_url": chunk.source_url or "",
                "heading": chunk.heading or "unknown_heading",
                "topic": chunk.topic,
                "last_modified": chunk.last_modified,
                "relevance_score": round(score, 4),
            }
            ranked_results.append(result)

        deduplicated_results = []
        seen_keys = set()
        for result in ranked_results:
            key = self._dedup_key_for_result(result)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduplicated_results.append(result)
            if len(deduplicated_results) >= top_k:
                break

        return deduplicated_results


class SemanticRetriever:
    """Sentence-transformer retrieval over the same chunked corpus."""

    def __init__(
        self,
        model_name: str = DEFAULT_SEMANTIC_MODEL,
        device: str = "cpu",
        max_candidates: int = 200,
    ):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.model_error: Optional[str] = None
        self.chunks: List[DocumentChunk] = []
        self.document_embeddings: Optional[np.ndarray] = None
        self.is_indexed = False
        self.max_candidates = max_candidates

    def _load_model(self) -> None:
        if self.model is not None:
            return
        if not _is_model_cached_locally(self.model_name):
            self.model_error = f"Model unavailable: {self.model_name} (missing local cache)"
            logger.warning(self.model_error)
            return
        try:
            self.model = _load_sentence_transformer(self.model_name, device=self.device, local_files_only=True)
            self.model_error = None
        except Exception as exc:  # pragma: no cover - exercised by targeted tests
            self.model_error = f"Model unavailable: {self.model_name} ({exc})"
            logger.warning(self.model_error)

    def index_records(self, records: List[Dict[str, Any]]) -> int:
        self.chunks = _index_record_chunks(records)
        self.document_embeddings = None
        self.is_indexed = bool(self.chunks)
        return len(self.chunks)

    def load_and_index(self, dataset_path: str = "data/processed/campus_documents.jsonl") -> int:
        """Load JSONL dataset and index chunks for semantic retrieval."""
        records = _load_dataset_records(dataset_path)
        return self.index_records(records)

    def encode_corpus(self, batch_size: int = 32) -> Optional[np.ndarray]:
        """Encode all indexed chunks once and cache embeddings in memory."""
        if self.document_embeddings is not None:
            return self.document_embeddings
        if not self.chunks:
            return None

        self._load_model()
        if self.model_error or self.model is None:
            return None

        try:
            texts = [chunk.searchable_text or chunk.text for chunk in self.chunks]
            self.document_embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                device=self.device,
                show_progress_bar=False,
            )
            return self.document_embeddings
        except Exception as exc:  # pragma: no cover - exercised by targeted tests
            self.model_error = f"Embedding failed for {self.model_name}: {exc}"
            logger.warning(self.model_error)
            return None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        if not query or not isinstance(query, str) or not query.strip():
            return []
        if not self.is_indexed or not self.chunks:
            return []

        doc_embeddings = self.encode_corpus()
        if doc_embeddings is None:
            logger.warning("Semantic retrieval is unavailable: %s", self.model_error or "unknown model error")
            return []

        try:
            query_embedding = self.model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                device=self.device,
            )
        except Exception as exc:  # pragma: no cover - exercised by targeted tests
            self.model_error = f"Query embedding failed for {self.model_name}: {exc}"
            logger.warning(self.model_error)
            return []

        similarities = cosine_similarity(query_embedding, doc_embeddings).flatten()
        if np.all(similarities <= 0.0):
            return []

        ranked_indices = np.argsort(-similarities)
        ranked_results: List[Dict[str, Any]] = []
        for idx in ranked_indices[: self.max_candidates]:
            score = float(similarities[idx])
            if score < score_threshold:
                continue
            chunk = self.chunks[int(idx)]
            ranked_results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": str(chunk.document_id or chunk.source_url or chunk.document_title or "unknown_document").strip() or "unknown_document",
                    "text": chunk.text,
                    "document_title": chunk.document_title or "",
                    "source_url": chunk.source_url or "",
                    "heading": chunk.heading or "unknown_heading",
                    "topic": chunk.topic,
                    "last_modified": chunk.last_modified,
                    "relevance_score": round(score, 4),
                }
            )

        deduplicated_results: List[Dict[str, Any]] = []
        seen_keys = set()
        for result in ranked_results:
            key = TFIDFRetriever._dedup_key_for_result(result)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduplicated_results.append(result)
            if len(deduplicated_results) >= top_k:
                break
        return deduplicated_results


class CrossEncoderReranker:
    """Cross-encoder rerank of a candidate set using a pointwise relevance score."""

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.model_error: Optional[str] = None

    def _load_model(self) -> None:
        if self.model is not None:
            return
        if not _is_model_cached_locally(self.model_name):
            self.model_error = f"Reranker unavailable: {self.model_name} (missing local cache)"
            logger.warning(self.model_error)
            return
        try:
            self.model = _load_cross_encoder(self.model_name, device=self.device, local_files_only=True)
            self.model_error = None
        except Exception as exc:  # pragma: no cover - exercised by targeted tests
            self.model_error = f"Reranker unavailable: {self.model_name} ({exc})"
            logger.warning(self.model_error)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], max_candidates: int = 15) -> List[Dict[str, Any]]:
        """Rerank candidate passages against query while strictly preserving metadata."""
        if not query or not candidates:
            return list(candidates)

        self._load_model()
        if self.model_error or self.model is None:
            logger.warning("Cross-encoder reranking is unavailable: %s", self.model_error or "unknown model error")
            return list(candidates)

        rerank_pool = candidates[:max_candidates]
        tail = candidates[max_candidates:]

        try:
            pairs = [(query, c.get("text") or "") for c in rerank_pool]
            scores = self.model.predict(pairs)
        except Exception as exc:  # pragma: no cover - exercised by targeted tests
            self.model_error = f"Reranking failed for {self.model_name}: {exc}"
            logger.warning(self.model_error)
            return list(candidates)

        reranked: List[Dict[str, Any]] = []
        for candidate, score in zip(rerank_pool, scores):
            item = dict(candidate)
            item["relevance_score"] = round(float(score), 4)
            item["reranked"] = True
            reranked.append(item)
        reranked.sort(key=lambda item: float(item.get("relevance_score", 0.0)), reverse=True)
        return reranked + list(tail)


class HybridRetriever:
    """Combine lexical (TF-IDF) and semantic (Sentence Transformer) candidates."""

    def __init__(
        self,
        tfidf_retriever: Optional[TFIDFRetriever] = None,
        semantic_retriever: Optional[SemanticRetriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        semantic_weight: float = 0.5,
        tfidf_weight: float = 0.5,
        fusion_method: str = "linear",
        rrf_k: int = 60,
    ):
        self.tfidf_retriever = tfidf_retriever or TFIDFRetriever()
        self.semantic_retriever = semantic_retriever or SemanticRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.semantic_weight = semantic_weight
        self.tfidf_weight = tfidf_weight
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.is_indexed: bool = False

    def load_and_index(self, dataset_path: str = "data/processed/campus_documents.jsonl") -> int:
        records = _load_dataset_records(dataset_path)
        return self.index_records(records)

    def index_records(self, records: List[Dict[str, Any]]) -> int:
        c1 = self.tfidf_retriever.index_records(records)
        c2 = self.semantic_retriever.index_records(records)
        self.is_indexed = bool(c1 or c2)
        return max(c1, c2)

    @classmethod
    def _merge_candidates(
        cls,
        tfidf_results: List[Dict[str, Any]],
        semantic_results: List[Dict[str, Any]],
        semantic_weight: float = 0.5,
        tfidf_weight: float = 0.5,
        fusion_method: str = "linear",
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """Merge lexical and semantic candidates without chunk duplication.

        Supported fusion methods:
        - 'linear': Normalized weighted combination of cosine similarity scores.
        - 'rrf': Reciprocal Rank Fusion based on rank positions in both candidate lists.
        """
        merged_candidates: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        if fusion_method == "rrf":
            for rank, res in enumerate(tfidf_results, start=1):
                key = TFIDFRetriever._dedup_key_for_result(res)
                if key not in merged_candidates:
                    merged_candidates[key] = dict(res)
                    merged_candidates[key]["_rrf_score"] = 0.0
                merged_candidates[key]["_rrf_score"] += tfidf_weight / (rrf_k + rank)

            for rank, res in enumerate(semantic_results, start=1):
                key = TFIDFRetriever._dedup_key_for_result(res)
                if key not in merged_candidates:
                    merged_candidates[key] = dict(res)
                    merged_candidates[key]["_rrf_score"] = 0.0
                merged_candidates[key]["_rrf_score"] += semantic_weight / (rrf_k + rank)

            combined = []
            for item in merged_candidates.values():
                item["relevance_score"] = round(float(item.pop("_rrf_score", 0.0)), 6)
                combined.append(item)
            return sorted(combined, key=lambda it: float(it.get("relevance_score", 0.0)), reverse=True)

        # Default: linear score combination
        for res in tfidf_results:
            key = TFIDFRetriever._dedup_key_for_result(res)
            merged_candidates[key] = dict(res)
            merged_candidates[key]["_lexical_score"] = float(res.get("relevance_score", 0.0) or 0.0)
            merged_candidates[key]["_semantic_score"] = 0.0

        for res in semantic_results:
            key = TFIDFRetriever._dedup_key_for_result(res)
            if key in merged_candidates:
                merged_candidates[key]["_semantic_score"] = float(res.get("relevance_score", 0.0) or 0.0)
            else:
                item = dict(res)
                item["_semantic_score"] = float(res.get("relevance_score", 0.0) or 0.0)
                item["_lexical_score"] = 0.0
                merged_candidates[key] = item

        combined = []
        for candidate in merged_candidates.values():
            lexical = float(candidate.pop("_lexical_score", 0.0))
            semantic = float(candidate.pop("_semantic_score", 0.0))
            combined_score = (tfidf_weight * lexical) + (semantic_weight * semantic)
            candidate["relevance_score"] = round(combined_score, 4)
            combined.append(candidate)

        return sorted(combined, key=lambda it: float(it.get("relevance_score", 0.0)), reverse=True)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.05,
        strategy: str = "hybrid",
        rerank: bool = False,
    ) -> List[Dict[str, Any]]:
        if strategy == "tfidf":
            return self.tfidf_retriever.retrieve(query=query, top_k=top_k, score_threshold=score_threshold)
        if strategy == "semantic":
            return self.semantic_retriever.retrieve(query=query, top_k=top_k, score_threshold=score_threshold)

        pool_size = max(15, top_k * 4)
        tfidf_results = self.tfidf_retriever.retrieve(query=query, top_k=pool_size, score_threshold=0.0)
        semantic_results = self.semantic_retriever.retrieve(query=query, top_k=pool_size, score_threshold=0.0)

        combined = self._merge_candidates(
            tfidf_results,
            semantic_results,
            semantic_weight=self.semantic_weight,
            tfidf_weight=self.tfidf_weight,
            fusion_method=self.fusion_method,
            rrf_k=self.rrf_k,
        )

        filtered = [c for c in combined if float(c.get("relevance_score", 0.0) or 0.0) >= score_threshold]
        if rerank or strategy == "hybrid_rerank":
            filtered = self.reranker.rerank(query, filtered[:20])

        deduplicated: List[Dict[str, Any]] = []
        seen_keys = set()
        for result in filtered:
            key = TFIDFRetriever._dedup_key_for_result(result)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduplicated.append(result)
            if len(deduplicated) >= top_k:
                break
        return deduplicated


_DEFAULT_RETRIEVER_CACHE: Dict[str, TFIDFRetriever] = {}
_DEFAULT_SEMANTIC_RETRIEVER_CACHE: Dict[str, SemanticRetriever] = {}
_DEFAULT_HYBRID_RETRIEVER_CACHE: Dict[str, HybridRetriever] = {}


def get_default_retriever(
    dataset_path: str = "data/processed/campus_documents.jsonl",
) -> TFIDFRetriever:
    """Factory function returning an initialized, indexed TF-IDF retriever."""
    if dataset_path not in _DEFAULT_RETRIEVER_CACHE:
        retriever = TFIDFRetriever()
        retriever.load_and_index(dataset_path)
        _DEFAULT_RETRIEVER_CACHE[dataset_path] = retriever
    return _DEFAULT_RETRIEVER_CACHE[dataset_path]


def get_default_semantic_retriever(
    dataset_path: str = "data/processed/campus_documents.jsonl",
    model_name: str = DEFAULT_SEMANTIC_MODEL,
    device: str = "cpu",
) -> SemanticRetriever:
    """Factory function returning an initialized semantic retriever."""
    cache_key = f"{dataset_path}:{model_name}:{device}"
    if cache_key not in _DEFAULT_SEMANTIC_RETRIEVER_CACHE:
        retriever = SemanticRetriever(model_name=model_name, device=device)
        retriever.load_and_index(dataset_path)
        retriever.encode_corpus()
        _DEFAULT_SEMANTIC_RETRIEVER_CACHE[cache_key] = retriever
    return _DEFAULT_SEMANTIC_RETRIEVER_CACHE[cache_key]


def get_default_hybrid_retriever(
    dataset_path: str = "data/processed/campus_documents.jsonl",
    model_name: str = DEFAULT_SEMANTIC_MODEL,
    reranker_name: str = DEFAULT_CROSS_ENCODER_MODEL,
    device: str = "cpu",
    fusion_method: str = "linear",
    semantic_weight: float = 0.5,
    tfidf_weight: float = 0.5,
) -> HybridRetriever:
    """Factory function returning an initialized hybrid retriever."""
    cache_key = f"{dataset_path}:{model_name}:{reranker_name}:{device}:{fusion_method}:{semantic_weight}:{tfidf_weight}"
    if cache_key not in _DEFAULT_HYBRID_RETRIEVER_CACHE:
        tfidf_retriever = get_default_retriever(dataset_path)
        semantic_retriever = get_default_semantic_retriever(dataset_path=dataset_path, model_name=model_name, device=device)
        reranker = CrossEncoderReranker(model_name=reranker_name, device=device)
        hybrid = HybridRetriever(
            tfidf_retriever=tfidf_retriever,
            semantic_retriever=semantic_retriever,
            reranker=reranker,
            fusion_method=fusion_method,
            semantic_weight=semantic_weight,
            tfidf_weight=tfidf_weight,
        )
        hybrid.is_indexed = True
        _DEFAULT_HYBRID_RETRIEVER_CACHE[cache_key] = hybrid
    return _DEFAULT_HYBRID_RETRIEVER_CACHE[cache_key]



def _load_dataset_records(dataset_path: str = "data/processed/campus_documents.jsonl") -> List[Dict[str, Any]]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    records: List[Dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def query_campus_knowledge(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.05,
    dataset_path: str = "data/processed/campus_documents.jsonl",
    strategy: str = "tfidf",
    rerank: bool = False,
) -> List[Dict[str, Any]]:
    """
    Public callable interface:
    Accepts natural-language question and returns top-k ranked chunks.
    Strategy can be 'tfidf', 'semantic', 'hybrid', or 'hybrid_rerank'.
    """
    if strategy in {"semantic", "hybrid", "hybrid_rerank"}:
        hybrid = get_default_hybrid_retriever(dataset_path=dataset_path)
        return hybrid.retrieve(query=query, top_k=top_k, score_threshold=score_threshold, strategy=strategy, rerank=rerank)
    retriever = get_default_retriever(dataset_path)
    return retriever.retrieve(query=query, top_k=top_k, score_threshold=score_threshold)
