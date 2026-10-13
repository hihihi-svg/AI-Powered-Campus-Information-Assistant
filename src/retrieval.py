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
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("campus_pipeline.retrieval")


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
    min_chunk_chars: int = 60,
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
        "principal\u2019s office",    # right-column principal card (unicode right-quote)
        "principal's office",         # straight-quote variant
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
        all_chunks: List[DocumentChunk] = []
        for r in records:
            # Enforce rule: load only records with non-empty extracted content
            if r.get("extraction_status") == "EMPTY":
                continue
            content = r.get("content", "")
            if not content or not content.strip():
                continue

            doc_chunks = chunk_document(r)
            all_chunks.extend(doc_chunks)

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

        # Get top-k indices sorted descending
        top_indices = np.argsort(-similarities)[: top_k * 2]  # fetch extra for threshold filtering

        results = []
        seen_keys = set()  # This is our memory to track duplicates

        for idx in top_indices:
            score = float(similarities[idx])
            if score < score_threshold:
                continue

            chunk = self.chunks[idx]
            
            # Create a unique key using the document's URL and heading
            unique_key = (chunk.source_url, chunk.heading)
            
            # If we already added this section, skip it!
            if unique_key in seen_keys:
                continue
                
            # Otherwise, remember it and add it to our results
            seen_keys.add(unique_key)

            results.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "document_title": chunk.document_title,
                "source_url": chunk.source_url,
                "heading": chunk.heading,
                "topic": chunk.topic,
                "last_modified": chunk.last_modified,
                "relevance_score": round(score, 4),
            })
            if len(results) >= top_k:
                break

        return results


def get_default_retriever(
    dataset_path: str = "data/processed/campus_documents.jsonl",
) -> TFIDFRetriever:
    """Factory function returning an initialized, indexed TF-IDF retriever."""
    retriever = TFIDFRetriever()
    retriever.load_and_index(dataset_path)
    return retriever


def query_campus_knowledge(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.05,
    dataset_path: str = "data/processed/campus_documents.jsonl",
) -> List[Dict[str, Any]]:
    """
    Public callable interface:
    Accepts natural-language question and returns top-k ranked chunks.
    """
    retriever = get_default_retriever(dataset_path)
    return retriever.retrieve(query=query, top_k=top_k, score_threshold=score_threshold)
