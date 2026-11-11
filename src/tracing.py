"""LangSmith RAG Observability and Tracing module for the RVCE Campus Information Assistant.

This module provides hierarchical tracing, privacy-preserving sanitization,
and observability across all stages of the RAG pipeline:
1. Intent classification
2. Entity extraction & temporal query parsing
3. Multi-query expansion
4. Lexical, dense, hybrid, and placement retrieval
5. Candidate merging and cross-encoder reranking
6. Context assembly and companion chunk enrichment
7. Answer generation (OpenAI GPT or extractive synthesis)
8. Citation validation and source verification

LangSmith tracing is optional and disabled by default (LANGSMITH_TRACING=false).
Tracing failures NEVER raise exceptions or degrade user request handling.
"""

from __future__ import annotations

from contextvars import ContextVar
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import langsmith
    from langsmith import Client
    from langsmith.run_trees import RunTree

    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    Client = None
    RunTree = None

# Context variable for thread-local/async active tracer
_ACTIVE_TRACER: ContextVar[Optional["RAGTracer"]] = ContextVar("active_rag_tracer", default=None)

# Regular expressions for data sanitization
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", re.IGNORECASE)
BEARER_PATTERN = re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]+", re.IGNORECASE)
API_KEY_PATTERN = re.compile(r"(?:sk-[a-zA-Z0-9_-]{20,}|AIza[0-9A-Za-z-_]{35}|[0-9a-f]{32,64})", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"\b(?:lsv2_[a-zA-Z0-9_\-]{20,}|ghp_[a-zA-Z0-9]{36})\b", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(?:\+91[-\s]?)?[6-9]\d{9}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b")
STUDENT_USN_PATTERN = re.compile(r"\b1RV\d{2}[A-Z]{2}\d{3}\b|\b\d{2}[A-Z]{2,3}\d{3,4}\b", re.IGNORECASE)
STUDENT_NAME_PATTERN = re.compile(
    r"\b(?:Student\s*Name|Candidate\s*Name|Name\s*of\s*Student|Student|Candidate)\s*[:=-]\s*([A-Za-z\s]{3,35})\b",
    re.IGNORECASE,
)
GREETING_NAME_PATTERN = re.compile(
    r"\b(?:Dear|Mr\.|Ms\.|Mrs\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
    re.IGNORECASE,
)
ACADEMIC_RECORD_PATTERN = re.compile(
    r"\b(?:SGPA|CGPA|Marks|Score|Roll\s*No|Rank|Grade)\s*[:=-]?\s*[\d.A-Za-z+]+\b",
    re.IGNORECASE,
)

# Strict allowlist of permitted metadata fields
ALLOWED_CHUNK_FIELDS = {
    "chunk_id",
    "document_id",
    "source_title",
    "heading",
    "url",
    "source_type",
    "relevance_score",
    "document_date",
    "is_latest_version",
    "superseded_by",
    "snippet",
}


def is_tracing_enabled() -> bool:
    """Return True only if LANGSMITH_TRACING is explicitly enabled and dependencies/credentials are ready."""
    flag = os.environ.get("LANGSMITH_TRACING", "").strip().lower()
    if flag not in {"1", "true", "yes"}:
        return False

    if not LANGSMITH_AVAILABLE:
        logger.debug("LANGSMITH_TRACING is set but the 'langsmith' package is not installed.")
        return False

    api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    if not api_key:
        logger.debug("LANGSMITH_TRACING is enabled but LANGSMITH_API_KEY is not set.")
        return False

    return True


def sanitize_text(text: Optional[str]) -> str:
    """Scrub email addresses, API tokens, phone numbers, student USNs, names, and academic records."""
    if not text:
        return ""
    cleaned = str(text)
    cleaned = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", cleaned)
    cleaned = BEARER_PATTERN.sub("Bearer [REDACTED_TOKEN]", cleaned)
    cleaned = API_KEY_PATTERN.sub("[REDACTED_KEY]", cleaned)
    cleaned = TOKEN_PATTERN.sub("[REDACTED_TOKEN]", cleaned)
    cleaned = PHONE_PATTERN.sub("[REDACTED_PHONE]", cleaned)
    cleaned = STUDENT_USN_PATTERN.sub("[REDACTED_USN]", cleaned)
    cleaned = STUDENT_NAME_PATTERN.sub("Student: [REDACTED_NAME]", cleaned)
    cleaned = GREETING_NAME_PATTERN.sub("[REDACTED_NAME]", cleaned)
    cleaned = ACADEMIC_RECORD_PATTERN.sub("[REDACTED_ACADEMIC_RECORD]", cleaned)
    return cleaned


def sanitize_chunk_for_trace(chunk: Dict[str, Any], max_snippet_chars: int = 160) -> Dict[str, Any]:
    """Sanitize retrieved chunk metadata for external tracing using strict field allowlists.

    Strictest Privacy Rule:
    Raw Gmail message bodies, attachments, and raw PDF streams are NEVER transmitted.
    Only non-sensitive document identifiers, headings, scores, and sanitized short snippets
    are exported to LangSmith.
    """
    if not isinstance(chunk, dict):
        return {}

    source_type = str(chunk.get("source_type") or "").lower()
    is_private_source = (
        "gmail" in source_type
        or "attachment" in source_type
        or "notice" in source_type
        or "email" in source_type
        or chunk.get("gmail_message_id") is not None
        or chunk.get("sender_email") is not None
    )

    if is_private_source:
        # Strictly suppress raw Gmail notice text and PDF attachment contents
        safe_snippet = "[SUPPRESSED_PRIVATE_CONTENT]"
    else:
        raw_text = str(chunk.get("text") or chunk.get("snippet") or "")
        sanitized_full = sanitize_text(raw_text)
        safe_snippet = sanitized_full[:max_snippet_chars].replace("\n", " ").strip()

    payload = {
        "chunk_id": str(chunk.get("chunk_id") or ""),
        "document_id": str(chunk.get("document_id") or ""),
        "source_title": sanitize_text(str(chunk.get("source_title") or chunk.get("title") or "")),
        "heading": sanitize_text(str(chunk.get("heading") or "")),
        "url": str(chunk.get("url") or chunk.get("source_url") or ""),
        "source_type": str(chunk.get("source_type") or "campus_document"),
        "relevance_score": round(float(chunk.get("relevance_score") or 0.0), 4),
        "document_date": str(chunk.get("document_date") or chunk.get("last_modified") or ""),
        "is_latest_version": bool(chunk.get("is_latest_version", False)),
        "superseded_by": str(chunk.get("superseded_by") or ""),
        "snippet": safe_snippet,
    }

    # Allowlist filter
    return {k: v for k, v in payload.items() if k in ALLOWED_CHUNK_FIELDS}


def get_current_tracer() -> Optional["RAGTracer"]:
    """Retrieve the currently active RAGTracer from context."""
    return _ACTIVE_TRACER.get()


def set_current_tracer(tracer: Optional["RAGTracer"]) -> None:
    """Set the currently active RAGTracer in context."""
    _ACTIVE_TRACER.set(tracer)


class RAGTracer:
    """Manages hierarchical LangSmith trace trees across the campus RAG pipeline.

    Guarantees:
    - Zero runtime overhead and completely inactive when LANGSMITH_TRACING=false.
    - All tracing actions are wrapped in try-except; no tracing exception ever
      bubbles up or affects question answering.
    - Raw private Gmail bodies and attachment binaries are never logged.
    """

    def __init__(
        self,
        project_name: Optional[str] = None,
        client: Optional[Any] = None,
        enabled: Optional[bool] = None,
        sanitize: Optional[bool] = None,
    ) -> None:
        self.project_name = (
            project_name
            or os.environ.get("LANGSMITH_PROJECT")
            or os.environ.get("LANGCHAIN_PROJECT")
            or "rvce-campus-assistant"
        )
        self.sanitize = True if sanitize is None else bool(sanitize)

        if enabled is not None:
            self.enabled = bool(enabled)
        else:
            self.enabled = is_tracing_enabled() or (client is not None)

        self.client = client
        if self.enabled and self.client is None and LANGSMITH_AVAILABLE:
            try:
                self.client = Client()
            except Exception as exc:  # pylint: disable=broad-except
                logger.debug("Failed to initialize LangSmith Client: %s", exc)
                self.enabled = False

        self.root_run: Optional[Any] = None
        self.child_runs: Dict[str, Any] = {}
        self.stages_recorded: List[str] = []
        self.start_time: float = time.time()
        self._token: Optional[Any] = None

    def __enter__(self) -> "RAGTracer":
        self._token = _ACTIVE_TRACER.set(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._token:
            _ACTIVE_TRACER.reset(self._token)

    def start_pipeline(
        self,
        question: str,
        strategy: str = "hybrid",
        rerank: bool = False,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[Any]:
        """Start the root RAG pipeline run trace."""
        self.stages_recorded.append("start_pipeline")
        if not self.enabled or not LANGSMITH_AVAILABLE or RunTree is None:
            return None

        try:
            clean_q = sanitize_text(question) if self.sanitize else question
            inputs = {
                "question": clean_q,
                "strategy": strategy,
                "rerank": rerank,
                "has_conversation_history": bool(conversation_history),
                "conversation_turns": len(conversation_history) if conversation_history else 0,
            }
            self.root_run = RunTree(
                name="campus_rag_pipeline",
                run_type="chain",
                inputs=inputs,
                project_name=self.project_name,
                client=self.client,
            )
            self.root_run.post()
            return self.root_run
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Tracing error in start_pipeline: %s", exc)
            return None

    def trace_intent(
        self,
        raw_question: str,
        intent_info: Dict[str, Any],
        temporal_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Trace intent classification and temporal parsing step."""
        self.stages_recorded.append("intent_classification")
        if not self.enabled or not self.root_run:
            return None

        try:
            clean_q = sanitize_text(raw_question) if self.sanitize else raw_question
            child = self.root_run.create_child(
                name="intent_classification",
                run_type="parser",
                inputs={"query": clean_q},
            )
            child.post()

            outputs = {
                "normalized_intent": intent_info.get("normalized_intent") or intent_info.get("intent"),
                "confidence": intent_info.get("confidence"),
                "model": intent_info.get("model") or "logistic_regression",
                "entities": intent_info.get("entities") or {},
                "has_temporal_intent": bool(temporal_info and temporal_info.get("has_temporal_intent")),
            }
            if temporal_info and temporal_info.get("intent"):
                outputs["temporal_intent"] = temporal_info.get("intent")
            if temporal_info and temporal_info.get("period_label"):
                outputs["period_label"] = temporal_info.get("period_label")

            child.end(outputs=outputs)
            child.patch()
            self.child_runs["intent_classification"] = child
            return child
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Tracing error in trace_intent: %s", exc)
            return None

    def trace_query_expansion(
        self,
        original_query: str,
        expanded_queries: List[str],
        strategy: str,
    ) -> Optional[Any]:
        """Trace multi-query expansion generation step."""
        self.stages_recorded.append("query_expansion")
        if not self.enabled or not self.root_run:
            return None

        try:
            clean_q = sanitize_text(original_query) if self.sanitize else original_query
            clean_expanded = (
                [sanitize_text(q) for q in expanded_queries] if self.sanitize else expanded_queries
            )
            child = self.root_run.create_child(
                name="query_expansion",
                run_type="tool",
                inputs={"original_query": clean_q, "strategy": strategy},
            )
            child.post()
            child.end(
                outputs={
                    "expanded_queries": clean_expanded,
                    "queries_count": len(expanded_queries),
                }
            )
            child.patch()
            self.child_runs["query_expansion"] = child
            return child
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Tracing error in trace_query_expansion: %s", exc)
            return None

    def trace_retrieval(
        self,
        retrieval_query: str,
        retrieved_chunks: List[Dict[str, Any]],
        retriever_type: str = "hybrid",
        latency_ms: Optional[float] = None,
    ) -> Optional[Any]:
        """Trace document retrieval stage, recording hits, zero-hits, and metadata."""
        self.stages_recorded.append("retrieval")
        if not self.enabled or not self.root_run:
            return None

        try:
            clean_q = sanitize_text(retrieval_query) if self.sanitize else retrieval_query
            child = self.root_run.create_child(
                name="retrieval",
                run_type="retriever",
                inputs={"query": clean_q, "retriever_type": retriever_type},
            )
            child.post()

            sanitized_chunks = (
                [sanitize_chunk_for_trace(c) for c in retrieved_chunks[:10]]
                if self.sanitize
                else retrieved_chunks[:10]
            )

            outputs = {
                "chunks_count": len(retrieved_chunks),
                "zero_hits": len(retrieved_chunks) == 0,
                "retriever_type": retriever_type,
                "chunks_sample": sanitized_chunks,
            }
            if latency_ms is not None:
                outputs["latency_ms"] = round(latency_ms, 2)

            child.end(outputs=outputs)
            child.patch()
            self.child_runs["retrieval"] = child
            return child
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Tracing error in trace_retrieval: %s", exc)
            return None

    def trace_reranking(
        self,
        candidates_in: List[Dict[str, Any]],
        candidates_out: List[Dict[str, Any]],
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        latency_ms: Optional[float] = None,
    ) -> Optional[Any]:
        """Trace cross-encoder candidate re-scoring and re-ordering."""
        self.stages_recorded.append("reranking")
        if not self.enabled or not self.root_run:
            return None

        try:
            child = self.root_run.create_child(
                name="reranking",
                run_type="tool",
                inputs={
                    "candidates_in_count": len(candidates_in),
                    "reranker_model": reranker_model,
                },
            )
            child.post()

            top_scores = [
                round(float(c.get("relevance_score") or 0.0), 4) for c in candidates_out[:5]
            ]
            outputs = {
                "candidates_out_count": len(candidates_out),
                "top_scores": top_scores,
            }
            if latency_ms is not None:
                outputs["latency_ms"] = round(latency_ms, 2)

            child.end(outputs=outputs)
            child.patch()
            self.child_runs["reranking"] = child
            return child
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Tracing error in trace_reranking: %s", exc)
            return None

    def trace_context_assembly(
        self,
        usable_chunks: List[Dict[str, Any]],
        context_length_chars: int = 0,
        token_count: Optional[int] = None,
        truncated: bool = False,
    ) -> Optional[Any]:
        """Trace context aggregation, companion chunk enrichment, and budgeting."""
        self.stages_recorded.append("context_assembly")
        if not self.enabled or not self.root_run:
            return None

        try:
            chunk_ids = [str(c.get("chunk_id") or "") for c in usable_chunks]
            child = self.root_run.create_child(
                name="context_assembly",
                run_type="prompt",
                inputs={"chunk_ids": chunk_ids},
            )
            child.post()

            outputs = {
                "usable_chunks_count": len(usable_chunks),
                "context_length_chars": context_length_chars,
                "truncated": truncated,
            }
            if token_count is not None:
                outputs["token_count"] = token_count

            child.end(outputs=outputs)
            child.patch()
            self.child_runs["context_assembly"] = child
            return child
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Tracing error in trace_context_assembly: %s", exc)
            return None

    def trace_generation(
        self,
        model_name: str,
        answer: str,
        latency_ms: Optional[float] = None,
        error: Optional[Dict[str, Any]] = None,
        token_usage: Optional[Dict[str, Any]] = None,
        is_llm: bool = True,
    ) -> Optional[Any]:
        """Trace answer generation (LLM call or extractive synthesis)."""
        self.stages_recorded.append("answer_generation")
        if not self.enabled or not self.root_run:
            return None

        try:
            clean_answer = sanitize_text(answer[:300]) if self.sanitize else answer[:300]
            child = self.root_run.create_child(
                name="answer_generation",
                run_type="llm" if is_llm else "tool",
                inputs={"model_name": model_name, "is_llm": is_llm},
            )
            child.post()

            outputs = {
                "answer_snippet": clean_answer,
                "answer_length": len(answer),
                "generation_success": error is None,
            }
            if latency_ms is not None:
                outputs["latency_ms"] = round(latency_ms, 2)
            if token_usage:
                outputs["token_usage"] = token_usage
            if error:
                outputs["error_category"] = error.get("category")
                outputs["error_code"] = error.get("code") or error.get("error_code")

            child.end(outputs=outputs, error=str(error.get("category")) if error else None)
            child.patch()
            self.child_runs["answer_generation"] = child
            return child
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Tracing error in trace_generation: %s", exc)
            return None

    def trace_citation_validation(
        self,
        sources: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
        is_valid: bool = True,
        validation_details: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """Trace citation validation and source grounding verification."""
        self.stages_recorded.append("citation_validation")
        if not self.enabled or not self.root_run:
            return None

        try:
            child = self.root_run.create_child(
                name="citation_validation",
                run_type="tool",
                inputs={
                    "sources_count": len(sources),
                    "evidence_count": len(evidence),
                },
            )
            child.post()

            source_docs = [
                str(s.get("document_id") or s.get("title") or "") for s in sources[:5]
            ]
            outputs = {
                "is_valid": is_valid,
                "sources_count": len(sources),
                "evidence_count": len(evidence),
                "source_documents_sample": source_docs,
            }
            if validation_details:
                outputs["validation_details"] = validation_details

            child.end(outputs=outputs)
            child.patch()
            self.child_runs["citation_validation"] = child
            return child
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Tracing error in trace_citation_validation: %s", exc)
            return None

    def end_pipeline(
        self,
        answer: str,
        is_abstention: bool,
        sources_count: int = 0,
        error: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Close the root RAG pipeline run trace."""
        self.stages_recorded.append("end_pipeline")
        if not self.enabled or not self.root_run:
            return

        try:
            clean_ans = sanitize_text(answer[:300]) if self.sanitize else answer[:300]
            outputs = {
                "answer_snippet": clean_ans,
                "abstention_status": is_abstention,
                "sources_count": sources_count,
            }
            if error:
                outputs["error"] = {
                    "category": error.get("category"),
                    "code": error.get("code") or error.get("error_code"),
                }
            self.root_run.end(
                outputs=outputs,
                error=str(error.get("category")) if error else None,
            )
            self.root_run.patch()
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Tracing error in end_pipeline: %s", exc)
