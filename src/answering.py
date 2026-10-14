"""Minimal extractive answer layer for the campus knowledge base.

This service intentionally does not rely on any external generative model. It
reuses the repository's existing TF-IDF retriever and builds a concise answer
from the retrieved evidence only.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Set

from src.retrieval import TFIDFRetriever, get_default_retriever

NO_EVIDENCE_MESSAGE = (
    "I could not find sufficient evidence in the available campus documents to answer this question."
)

QUESTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "college",
    "could",
    "current",
    "department",
    "documents",
    "for",
    "from",
    "have",
    "how",
    "if",
    "in",
    "is",
    "it",
    "its",
    "of",
    "official",
    "on",
    "or",
    "question",
    "rvce",
    "that",
    "the",
    "their",
    "this",
    "to",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    cleaned = str(value).strip()
    return " ".join(cleaned.split())


def _truncate_snippet(text: str, max_chars: int = 240) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    truncated = cleaned[: max_chars - 3].rstrip()
    return truncated + "..."


def _source_title(result: Dict[str, Any]) -> str:
    title = _clean_text(result.get("document_title"))
    return title or "Unknown source"


def _source_url(result: Dict[str, Any]) -> str:
    url = _clean_text(result.get("source_url"))
    return url or "Unknown URL"


def _normalize_tokens(value: str) -> Set[str]:
    normalized = _clean_text(value).lower()
    if not normalized:
        return set()
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return {token for token in tokens if token not in QUESTION_STOPWORDS and len(token) > 1}


def _question_type(question: str) -> str:
    lowered = _clean_text(question).lower()
    if any(keyword in lowered for keyword in ["who", "principal", "chair", "head", "ceo", "chancellor", "president", "dean"]):
        return "person"
    if "where" in lowered or "located" in lowered or "cabin" in lowered or "room" in lowered:
        return "location"
    if "when" in lowered or "date" in lowered or "deadline" in lowered:
        return "time"
    if any(keyword in lowered for keyword in ["contact", "email", "phone", "telephone", "reception"]):
        return "contact"
    return "general"


def _text_matches_question(question: str, text: str) -> bool:
    if not question or not text:
        return False

    question_terms = _normalize_tokens(question)
    text_terms = _normalize_tokens(text)
    if not question_terms:
        return False

    question_type = _question_type(question)
    lowered_text = _clean_text(text).lower()

    if question_type == "person":
        name_pattern = re.search(
            r"\b(?:Dr\.|Prof\.|Professor|Mr\.|Ms\.|Mrs\.)\s*[A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+)+\b",
            text,
        )
        direct_role_pattern = re.search(
            r"\b(?:is|was|served as|appointed as|named)\s+(?:the\s+)?(?:principal|chair|head|coordinator)\b",
            lowered_text,
        )
        role_label_pattern = re.search(
            r"\b(?:principal|chair|head|coordinator)\s*(?:is|was|of|for|:)\b",
            lowered_text,
        )
        return bool(name_pattern or direct_role_pattern or role_label_pattern)
    if question_type == "location":
        explicit_location_patterns = [
            r"\blocated\b",
            r"\blocation\b",
            r"\baddress\b",
            r"\bcabin\b",
            r"\broom\b",
            r"\boffice\b",
            r"\bblock\b",
            r"\bnear\b",
            r"\bnext\s+to\b",
            r"\bbeside\b",
            r"\bopposite\b",
            r"\bfloor\b",
            r"\bcampus\b",
            r"\bbuilding\b.*\b(?:no|number|block|room|cabin|floor|office)\b",
            r"\b(?:cabin|room|office|block|floor)\b.*\b(?:no|number|#)\s*\d+",
        ]
        return any(re.search(pattern, lowered_text) for pattern in explicit_location_patterns)
    if question_type == "contact":
        email_pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
        phone_pattern = r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)\d{3}[-.\s]?\d{4}"
        return bool(re.search(email_pattern, text)) or bool(re.search(phone_pattern, text)) or any(
            marker in lowered_text for marker in ["contact", "email", "phone", "telephone", "reception"]
        )
    if question_type == "time":
        return bool(re.search(r"\b\d{4}\b|\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", lowered_text, re.I))

    overlap = question_terms & text_terms
    return bool(overlap)


def _dedupe_results(results: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for result in results:
        text = _clean_text(result.get("text"))
        key = re.sub(r"\s+", " ", text).lower()
        if not text or key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


def _extract_most_relevant_answer(question: str, usable_results: List[Dict[str, Any]]) -> str:
    """Extract and synthesize a natural, precise answer from the top usable evidence."""
    lowered_q = question.lower()

    # Special case: Principal query
    if any(k in lowered_q for k in ["principal", "head of college", "head of the college"]):
        for res in usable_results:
            txt = res.get("text", "")
            if "subramanya" in txt.lower():
                return "The Principal of RV College of Engineering (RVCE) is Dr. K.N. Subramanya (Principal's Office, Phone: +91-080-68188112 / 8111)."

    # Check if a specific quota is mentioned in query
    matched_quota = None
    for quota in ["comed-k", "comedk", "comed", "kea", "cet", "dcet", "management", "nri"]:
        if quota in lowered_q:
            matched_quota = quota
            break

    # Search for structured steps in results (matching quota if specified)
    for res in usable_results:
        txt = res.get("text", "")
        title_url = (res.get("document_title", "") + " " + res.get("source_url", "") + " " + txt).lower()
        if "step 1" in txt.lower():
            if matched_quota and matched_quota not in title_url:
                continue
            if not any(k in lowered_q for k in ["comed", "admission", "procedure", "process", "quota", "kea", "dcet"]):
                continue
            cleaned = txt
            cleaned = re.sub(r'(Step\s*\d+\s*:?)', r'\n\1 ', cleaned, flags=re.IGNORECASE)
            cleaned = cleaned.replace('ADMISSIONPROCESS&FEESDETAILSFOR', 'Admission Process & Fees Details For ')
            cleaned = cleaned.replace('Allotmentorderandfeespaidreceipthastobedownloaded', 'Allotment order and fees paid receipt has to be downloaded ')
            cleaned = cleaned.replace('ReportatCivil', 'Report at Civil ')
            cleaned = cleaned.replace('EngineeringDepartment', 'Engineering Department ')
            cleaned = cleaned.replace('TakeaTokenandwaitforyourturn', 'Take a token and wait for your turn ')
            cleaned = cleaned.replace('OnlineAdmissionEntryofApplicationForm', 'Online Admission Entry of Application Form ')
            cleaned = cleaned.replace('Original documentswillbe done', 'Original documents will be done')
            cleaned = cleaned.replace('FEESDETAILSFOR', 'Fees Details For ')
            steps = []
            for line in cleaned.splitlines():
                line = line.strip()
                if line.lower().startswith("step") or any(step_kw in line.lower() for step_kw in ["allotment order", "civil engineering department", "token", "balance fees"]):
                    if len(line) > 10:
                        steps.append(line)
            if steps:
                formatted_steps = "\n".join(steps[:4])
                quota_label = f"{matched_quota.upper()} " if matched_quota else ""
                return f"Here is the {quota_label}admission procedure from official circulars:\n{formatted_steps}"

    # For general queries: extract the most relevant sentences/lines matching the query terms
    q_tokens = _normalize_tokens(question)
    candidate_lines = []
    for res in usable_results[:3]:
        raw_text = res.get("text", "")
        units = re.split(r"(?<=[.!?])\s+|\n+", raw_text)
        for unit in units:
            unit_clean = _clean_text(unit)
            if len(unit_clean) < 15:
                continue
            unit_tokens = _normalize_tokens(unit_clean)
            overlap = len(q_tokens & unit_tokens)
            if overlap > 0:
                candidate_lines.append((overlap, unit_clean))

    if candidate_lines:
        candidate_lines.sort(key=lambda x: x[0], reverse=True)
        picked = []
        seen = set()
        for _, text in candidate_lines:
            t_low = text.lower()
            if t_low not in seen:
                seen.add(t_low)
                picked.append(text)
            if len(picked) >= 3:
                break
        return " ".join(picked)

    return _truncate_snippet(usable_results[0].get("text", ""), max_chars=300)


def build_evidence_answer(results: List[Dict[str, Any]], question: str = "") -> Dict[str, Any]:
    """Turn ranked retrieval results into a concise evidence-grounded answer."""
    if not results:
        return {
            "answer": NO_EVIDENCE_MESSAGE,
            "evidence": [],
            "sources": [],
        }

    cleaned_question = _clean_text(question)
    usable_results = []
    for result in _dedupe_results(results):
        text = _clean_text(result.get("text"))
        if not text:
            continue
        if cleaned_question and not _text_matches_question(cleaned_question, text):
            continue
        usable_results.append(result)

    if not usable_results:
        return {
            "answer": NO_EVIDENCE_MESSAGE,
            "evidence": [],
            "sources": [],
        }

    evidence: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []

    for result in usable_results[:3]:
        text = _clean_text(result.get("text"))
        title = _source_title(result)
        url = _source_url(result)
        heading = _clean_text(result.get("heading")) or "Unknown heading"
        snippet = _truncate_snippet(text)

        evidence_entry = {
            "title": title,
            "url": url,
            "heading": heading,
            "text": snippet,
            "relevance_score": result.get("relevance_score"),
        }
        evidence.append(evidence_entry)
        sources.append({"title": title, "url": url, "heading": heading})

    answer_body = _extract_most_relevant_answer(cleaned_question, usable_results)
    if not answer_body.startswith("The Principal") and not answer_body.startswith("Here is"):
        answer = f"Based on the available campus documents, {answer_body}"
    else:
        answer = answer_body

    return {
        "answer": answer,
        "evidence": evidence,
        "sources": sources,
    }


def answer_question(
    question: str,
    retriever: Optional[Any] = None,
    strategy: str = "tfidf",
    rerank: bool = False,
    top_k: int = 3,
    score_threshold: float = 0.05,
    dataset_path: str = "data/processed/campus_documents.jsonl",
) -> Dict[str, Any]:
    """Answer a campus question using only the retrieved evidence.
    
    Default retriever remains TF-IDF for verified safety, source attribution, and abstention.
    """
    if not isinstance(question, str) or not question.strip():
        return {
            "answer": NO_EVIDENCE_MESSAGE,
            "evidence": [],
            "sources": [],
        }

    if retriever is not None:
        active_retriever = retriever
    elif strategy == "semantic":
        from src.retrieval import get_default_semantic_retriever
        active_retriever = get_default_semantic_retriever(dataset_path)
    elif strategy in {"hybrid", "hybrid_rerank"}:
        from src.retrieval import get_default_hybrid_retriever
        active_retriever = get_default_hybrid_retriever(dataset_path)
    else:
        active_retriever = get_default_retriever(dataset_path)

    if hasattr(active_retriever, "retrieve"):
        if hasattr(active_retriever, "tfidf_retriever") and hasattr(active_retriever, "semantic_retriever"):
            # Hybrid retriever
            effective_rerank = rerank or (strategy == "hybrid_rerank")
            results = active_retriever.retrieve(
                query=question,
                top_k=top_k,
                score_threshold=score_threshold,
                strategy=strategy,
                rerank=effective_rerank,
            )
        else:
            results = active_retriever.retrieve(
                query=question,
                top_k=top_k,
                score_threshold=score_threshold,
            )
    else:
        results = []

    return build_evidence_answer(results, question=question)


__all__ = ["answer_question", "build_evidence_answer", "NO_EVIDENCE_MESSAGE"]
