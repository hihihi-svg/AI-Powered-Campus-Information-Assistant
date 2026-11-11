"""Context Aggregation and Assembly Module for Multi-Stage RAG.

Assembles cohesive, structured, and hierarchical context from multi-query candidates:
1. Deduplicates overlapping chunk passages.
2. Groups candidate chunks by source document.
3. Locates adjacent continuation chunks in the document index to restore missing context.
4. Preserves headings, sections, and markdown document hierarchy.
5. Preserves markdown tables, ensuring header rows and divider rows accompany table rows.
6. Enriches chunks with metadata (academic year, branch, category, admission quota).
7. Enforces a configurable token budget (~4 characters per token).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("campus_pipeline.context_aggregation")

DEFAULT_MAX_TOKEN_BUDGET = 2500  # ~10,000 characters
CHARS_PER_TOKEN = 4


@dataclass
class AggregatedDocumentContext:
    """Represents assembled context for a single source document."""

    document_id: str
    document_title: str
    source_url: str
    source_type: Optional[str]
    document_date: Optional[str]
    sections: List[Dict[str, Any]]
    tables: List[str]
    metadata: Dict[str, Any]
    total_tokens: int


def _estimate_tokens(text: str) -> int:
    """Fast approximation of token count based on character length and whitespace."""
    if not text:
        return 0
    words = len(text.split())
    chars = len(text)
    return max(words, chars // CHARS_PER_TOKEN)


def _extract_table_header(text: str) -> Optional[Tuple[str, str]]:
    """Extract markdown table header row and delimiter row if present in text."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    for i in range(len(lines) - 1):
        if "|" in lines[i] and "|" in lines[i + 1]:
            header_candidate = lines[i]
            delim_candidate = lines[i + 1]
            if re.match(r"^\|?[\s\-:|]+\|?$", delim_candidate):
                return header_candidate, delim_candidate
    return None


def _repair_table_continuation(chunk_text: str, prev_chunk_text: Optional[str]) -> str:
    """If chunk_text contains table rows without a header, borrow the table header from prev_chunk."""
    if not prev_chunk_text or "|" not in chunk_text:
        return chunk_text

    current_lines = [l.strip() for l in chunk_text.split("\n") if l.strip()]
    first_table_idx = -1
    for idx, l in enumerate(current_lines):
        if l.startswith("|") and l.endswith("|"):
            first_table_idx = idx
            break

    if first_table_idx == -1:
        return chunk_text

    # Check if this table in chunk_text already has a delimiter row right after first row
    if first_table_idx + 1 < len(current_lines):
        if re.match(r"^\|?[\s\-:|]+\|?$", current_lines[first_table_idx + 1]):
            return chunk_text  # already has header

    # If first row looks like data rows without a delimiter, search prev_chunk for header
    header_pair = _extract_table_header(prev_chunk_text)
    if header_pair:
        h_row, d_row = header_pair
        lines = chunk_text.split("\n")
        repaired_lines = []
        injected = False
        for line in lines:
            if not injected and line.strip().startswith("|") and line.strip().endswith("|"):
                repaired_lines.append(h_row)
                repaired_lines.append(d_row)
                repaired_lines.append(line)
                injected = True
            else:
                repaired_lines.append(line)
        return "\n".join(repaired_lines)

    return chunk_text


def _find_adjacent_chunks(
    retrieved_chunk_id: str,
    all_indexed_chunks: List[Any],
    max_adjacents: int = 1,
) -> List[Any]:
    """Find immediate predecessor and successor chunks for the given chunk ID."""
    if not retrieved_chunk_id or not all_indexed_chunks:
        return []

    target_idx = -1
    for idx, c in enumerate(all_indexed_chunks):
        c_id = getattr(c, "chunk_id", None) or (c.get("chunk_id") if isinstance(c, dict) else None)
        if c_id == retrieved_chunk_id:
            target_idx = idx
            break

    if target_idx == -1:
        return []

    adjacents = []
    target_doc_id = getattr(all_indexed_chunks[target_idx], "document_id", None) or (
        all_indexed_chunks[target_idx].get("document_id") if isinstance(all_indexed_chunks[target_idx], dict) else None
    )

    # Predecessor
    for step in range(1, max_adjacents + 1):
        prev_idx = target_idx - step
        if prev_idx >= 0:
            c = all_indexed_chunks[prev_idx]
            c_doc_id = getattr(c, "document_id", None) or (c.get("document_id") if isinstance(c, dict) else None)
            if c_doc_id == target_doc_id:
                adjacents.append(c)

    # Successor
    for step in range(1, max_adjacents + 1):
        next_idx = target_idx + step
        if next_idx < len(all_indexed_chunks):
            c = all_indexed_chunks[next_idx]
            c_doc_id = getattr(c, "document_id", None) or (c.get("document_id") if isinstance(c, dict) else None)
            if c_doc_id == target_doc_id:
                adjacents.append(c)

    return adjacents


def assemble_hierarchical_context(
    candidates: List[Dict[str, Any]],
    retriever: Any = None,
    query_entities: Optional[Dict[str, Any]] = None,
    max_token_budget: int = DEFAULT_MAX_TOKEN_BUDGET,
    include_adjacent: bool = True,
) -> Dict[str, Any]:
    """Assemble a unified, structured multi-document context respecting token limits."""
    if not candidates:
        return {
            "formatted_context": "",
            "document_contexts": [],
            "total_tokens": 0,
            "included_chunk_ids": [],
            "sources": [],
        }

    all_indexed_chunks: List[Any] = []
    if retriever is not None:
        all_indexed_chunks = getattr(retriever, "chunks", None) or getattr(
            getattr(retriever, "tfidf_retriever", None), "chunks", []
        ) or []

    # 1. Deduplicate candidates and collect adjacent chunks if needed
    seen_chunk_ids: Set[str] = set()
    initial_chunks: List[Dict[str, Any]] = []

    for cand in candidates:
        cid = cand.get("chunk_id")
        if cid and cid in seen_chunk_ids:
            continue
        if cid:
            seen_chunk_ids.add(cid)
        initial_chunks.append(cand)

    # 2. Expand with adjacent chunks
    pool: List[Dict[str, Any]] = list(initial_chunks)
    if include_adjacent and all_indexed_chunks:
        from src.retrieval import _chunk_to_result_dict

        for cand in initial_chunks[:3]:  # expand top 3 candidates only to keep precision high
            cid = cand.get("chunk_id")
            if not cid:
                continue
            adjacents = _find_adjacent_chunks(cid, all_indexed_chunks, max_adjacents=1)
            for adj in adjacents:
                adj_id = getattr(adj, "chunk_id", None) or (adj.get("chunk_id") if isinstance(adj, dict) else None)
                if adj_id and adj_id not in seen_chunk_ids:
                    seen_chunk_ids.add(adj_id)
                    adj_dict = _chunk_to_result_dict(adj, float(cand.get("relevance_score") or 0.0) * 0.9) if not isinstance(adj, dict) else dict(adj)
                    adj_dict["is_adjacent_expansion"] = True
                    pool.append(adj_dict)

    # 3. Group by source document
    docs_by_id: Dict[str, Dict[str, Any]] = {}
    for item in pool:
        doc_id = str(item.get("document_id") or item.get("source_url") or "unknown_doc").strip()
        if doc_id not in docs_by_id:
            docs_by_id[doc_id] = {
                "document_id": doc_id,
                "document_title": item.get("document_title") or item.get("source_title") or "RVCE Document",
                "source_url": item.get("source_url") or "",
                "source_type": item.get("source_type"),
                "document_date": item.get("document_date") or item.get("last_modified"),
                "chunks": [],
                "max_score": float(item.get("relevance_score") or 0.0),
            }
        docs_by_id[doc_id]["chunks"].append(item)
        if float(item.get("relevance_score") or 0.0) > docs_by_id[doc_id]["max_score"]:
            docs_by_id[doc_id]["max_score"] = float(item.get("relevance_score") or 0.0)

    # Sort documents by maximum candidate relevance
    sorted_docs = sorted(docs_by_id.values(), key=lambda d: d["max_score"], reverse=True)

    # 4. Assemble context blocks within token budget
    formatted_blocks: List[str] = []
    included_chunk_ids: List[str] = []
    sources: List[Dict[str, Any]] = []
    current_tokens = 0

    for doc in sorted_docs:
        doc_chunks = doc["chunks"]
        # Sort chunks within document by chunk_id for sequential reading continuity
        doc_chunks.sort(key=lambda c: str(c.get("chunk_id") or ""))

        doc_header = f"### Document: {doc['document_title']}"
        if doc.get("document_date"):
            doc_header += f" (Date: {doc['document_date']})"
        if doc.get("source_url"):
            doc_header += f"\nSource URL: {doc['source_url']}"

        doc_section_texts: List[str] = []
        prev_chunk_text = None

        for c in doc_chunks:
            c_text = (c.get("text") or "").strip()
            if not c_text:
                continue

            # Repair tables if continuation row is detected
            c_text = _repair_table_continuation(c_text, prev_chunk_text)
            prev_chunk_text = c_text

            heading = (c.get("heading") or "").strip()
            sec_header = f"#### Section: {heading}" if heading and heading != "unknown_heading" else ""

            # Check entity relevance boost if query entities are present
            entity_tag = ""
            if query_entities:
                br = query_entities.get("branch")
                quota = query_entities.get("admission_quota")
                if br and br.lower() in c_text.lower():
                    entity_tag = f" [Matched Branch: {br}]"
                if quota and str(quota).lower() in c_text.lower():
                    entity_tag += f" [Matched Quota: {quota}]"

            chunk_block = f"{sec_header}{entity_tag}\n{c_text}".strip()
            chunk_tokens = _estimate_tokens(chunk_block)

            if current_tokens + chunk_tokens > max_token_budget and formatted_blocks:
                logger.info("Context token budget reached (%d / %d tokens). Truncating further chunk assembly.",
                            current_tokens, max_token_budget)
                break

            doc_section_texts.append(chunk_block)
            current_tokens += chunk_tokens
            if c.get("chunk_id"):
                included_chunk_ids.append(c["chunk_id"])

        if doc_section_texts:
            full_doc_block = f"{doc_header}\n\n" + "\n\n".join(doc_section_texts)
            formatted_blocks.append(full_doc_block)
            sources.append({
                "title": doc["document_title"],
                "source_title": doc["document_title"],
                "url": doc["source_url"],
                "document_id": doc["document_id"],
                "document_date": doc["document_date"],
            })

        if current_tokens >= max_token_budget:
            break

    final_context = "\n\n---\n\n".join(formatted_blocks)
    return {
        "formatted_context": final_context,
        "document_count": len(sources),
        "total_tokens": current_tokens,
        "included_chunk_ids": included_chunk_ids,
        "sources": sources,
    }

