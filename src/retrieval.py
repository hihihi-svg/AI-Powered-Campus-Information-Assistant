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

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("campus_pipeline.retrieval")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = os.environ.get("CAMPUS_DATASET_PATH") or str(REPO_ROOT / "data" / "processed" / "campus_documents.jsonl")
DEFAULT_EMBEDDINGS_CACHE_PATH = os.environ.get("CAMPUS_EMBEDDINGS_PATH") or str(REPO_ROOT / "data" / "processed" / "corpus_semantic_embeddings.npy")


def _resolve_path(path: str | Path) -> str:
    """Resolve a path reliably against current working directory, then REPO_ROOT fallback."""
    if not path:
        return ""
    p = Path(path)
    if p.is_absolute() and p.exists():
        return str(p)
    if p.exists():
        return str(p.resolve())
    candidate = REPO_ROOT / p
    if candidate.exists():
        return str(candidate.resolve())
    return str(candidate if not p.is_absolute() else p)


DEFAULT_SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
_SEMANTIC_MODEL_CACHE: Dict[str, Any] = {}
_CROSS_ENCODER_CACHE: Dict[str, Any] = {}


def _resolve_local_model_path(model_name: str) -> str:
    """Resolve the latest local snapshot directory for a model if available."""
    cache_root = os.path.expanduser("~/.cache/huggingface/hub")
    normalized_name = model_name.replace("/", "--")
    candidate_hub_dir = os.path.join(cache_root, f"models--{normalized_name}")
    snapshots_dir = os.path.join(candidate_hub_dir, "snapshots")
    if os.path.isdir(snapshots_dir):
        snapshots = [
            os.path.join(snapshots_dir, s)
            for s in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, s))
        ]
        if snapshots:
            snapshots.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return snapshots[0]
    return model_name


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
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer

    _ensure_model_is_cached(model_name)
    resolved_path = _resolve_local_model_path(model_name)
    cache_key = f"{model_name}:{device}:{local_files_only}"
    if cache_key not in _SEMANTIC_MODEL_CACHE:
        model = SentenceTransformer(resolved_path, device=device, local_files_only=local_files_only)
        _SEMANTIC_MODEL_CACHE[cache_key] = model
    return _SEMANTIC_MODEL_CACHE[cache_key]


def _load_cross_encoder(
    model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
    device: str = "cpu",
    local_files_only: bool = True,
) -> Any:
    """Load a single CrossEncoder instance and cache it for reuse."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import CrossEncoder

    _ensure_model_is_cached(model_name)
    resolved_path = _resolve_local_model_path(model_name)
    cache_key = f"{model_name}:{device}:{local_files_only}"
    if cache_key not in _CROSS_ENCODER_CACHE:
        model = CrossEncoder(resolved_path, device=device, local_files_only=local_files_only)
        _CROSS_ENCODER_CACHE[cache_key] = model
    return _CROSS_ENCODER_CACHE[cache_key]


def _index_record_chunks(records: List[Dict[str, Any]]) -> List[DocumentChunk]:
    """Shared record-to-chunk conversion used by TF-IDF and semantic indexes."""
    # Map parent email metadata by document_id and gmail_message_id so child PDF
    # attachments inherit sender and subject metadata if not directly stored on the record.
    email_meta_by_doc_id: Dict[str, Dict[str, Any]] = {}
    email_meta_by_msg_id: Dict[str, Dict[str, Any]] = {}
    for r in records:
        if r.get("source_type") == "EMAIL" or str(r.get("document_id", "")).startswith("doc_email_"):
            if r.get("source_type") != "EMAIL_ATTACHMENT" and not str(r.get("document_id", "")).startswith("doc_email_att_"):
                meta = {
                    "sender": r.get("sender"),
                    "sender_name": r.get("sender_name"),
                    "email_subject": r.get("title"),
                    "received_at": r.get("received_at") or r.get("last_modified"),
                    "document_date": r.get("document_date") or r.get("last_modified"),
                    "parent_email_body": (r.get("content") or "")[:500],
                }
                if r.get("document_id"):
                    email_meta_by_doc_id[str(r["document_id"])] = meta
                if r.get("gmail_message_id"):
                    email_meta_by_msg_id[str(r["gmail_message_id"])] = meta

    all_chunks: List[DocumentChunk] = []
    for r in records:
        if r.get("extraction_status") == "EMPTY":
            continue
        content = r.get("content", "")
        if not content or not content.strip():
            continue

        record_to_chunk = r
        if r.get("source_type") == "EMAIL_ATTACHMENT" or str(r.get("document_id", "")).startswith("doc_email_att_"):
            parent_meta = (
                email_meta_by_doc_id.get(str(r.get("parent_document_id", "")))
                or email_meta_by_msg_id.get(str(r.get("gmail_message_id", "")))
            )
            if parent_meta:
                record_to_chunk = dict(r)
                for k, v in parent_meta.items():
                    if v and not record_to_chunk.get(k):
                        record_to_chunk[k] = v

        all_chunks.extend(chunk_document(record_to_chunk))
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
    source_type: Optional[str] = None
    sender: Optional[str] = None
    sender_name: Optional[str] = None
    filename: Optional[str] = None
    gmail_message_id: Optional[str] = None
    parent_document_id: Optional[str] = None
    document_date: Optional[str] = None
    received_at: Optional[str] = None
    published_at: Optional[str] = None
    updated_at: Optional[str] = None
    event_dates: Optional[List[str]] = None
    email_subject: Optional[str] = None
    parent_email_body: Optional[str] = None


def _chunk_to_result_dict(chunk: DocumentChunk, relevance_score: float) -> Dict[str, Any]:
    """Convert a DocumentChunk and score into the standardized retrieval result dictionary."""
    doc_id = str(chunk.document_id or chunk.source_url or chunk.document_title or "unknown_document").strip() or "unknown_document"
    doc_title = chunk.document_title or ""
    result: Dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "document_id": doc_id,
        "text": chunk.text,
        "document_title": doc_title,
        "source_title": doc_title or "Unknown source",
        "source_url": chunk.source_url or "",
        "heading": chunk.heading or "unknown_heading",
        "topic": chunk.topic,
        "last_modified": chunk.last_modified,
        "document_date": getattr(chunk, "document_date", None) or chunk.last_modified,
        "received_at": getattr(chunk, "received_at", None),
        "published_at": getattr(chunk, "published_at", None),
        "updated_at": getattr(chunk, "updated_at", None),
        "event_dates": list(getattr(chunk, "event_dates", None) or []),
        "source_type": getattr(chunk, "source_type", None),
        "sender": getattr(chunk, "sender", None),
        "sender_name": getattr(chunk, "sender_name", None),
        "filename": getattr(chunk, "filename", None),
        "gmail_message_id": getattr(chunk, "gmail_message_id", None),
        "parent_document_id": getattr(chunk, "parent_document_id", None),
        "email_subject": getattr(chunk, "email_subject", None),
        "relevance_score": round(float(relevance_score), 4),
    }
    parent_body = getattr(chunk, "parent_email_body", None)
    if parent_body:
        result["parent_email_body"] = parent_body
    return result


def _sub_chunk_section(
    cleaned_block: str,
    max_chunk_chars: int = 1200,
    min_chunk_chars: int = 35,
    overlap_chars: int = 150,
) -> List[str]:
    """Split an oversized section text block into bounded sub-chunks with overlap."""
    if len(cleaned_block) <= max_chunk_chars:
        return [cleaned_block] if len(cleaned_block) >= min_chunk_chars else []

    # If the text has double-newline paragraph boundaries, break by paragraphs first
    paragraphs = [p.strip() for p in cleaned_block.split("\n\n") if p.strip()]

    # If any paragraph is longer than max_chunk_chars, split it into smaller units
    units: List[str] = []
    for p in paragraphs:
        if len(p) <= max_chunk_chars:
            units.append(p)
        else:
            lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
            for ln in lines:
                if len(ln) <= max_chunk_chars:
                    units.append(ln)
                else:
                    words = ln.split(" ")
                    curr = ""
                    for w in words:
                        if len(curr) + len(w) + 1 <= max_chunk_chars:
                            curr = f"{curr} {w}".strip() if curr else w
                        else:
                            if curr:
                                units.append(curr)
                            curr = w
                    if curr:
                        units.append(curr)

    chunks: List[str] = []
    current_chunk = ""

    for u in units:
        sep = "\n\n" if current_chunk else ""
        if len(current_chunk) + len(sep) + len(u) <= max_chunk_chars:
            current_chunk = f"{current_chunk}{sep}{u}" if current_chunk else u
        else:
            if len(current_chunk) >= min_chunk_chars:
                chunks.append(current_chunk)

            if overlap_chars > 0 and len(current_chunk) > overlap_chars:
                overlap_text = current_chunk[-overlap_chars:].strip()
                first_ws = overlap_text.find(" ")
                if first_ws != -1 and first_ws < len(overlap_text) // 2:
                    overlap_text = overlap_text[first_ws + 1:].strip()
            else:
                overlap_text = ""

            sep = "\n\n" if overlap_text else ""
            if len(overlap_text) + len(sep) + len(u) <= max_chunk_chars:
                current_chunk = f"{overlap_text}{sep}{u}" if overlap_text else u
            else:
                current_chunk = u

    if len(current_chunk) >= min_chunk_chars:
        if len(current_chunk) <= max_chunk_chars:
            chunks.append(current_chunk)
        else:
            step = max(1, max_chunk_chars - overlap_chars)
            for s in range(0, len(current_chunk), step):
                sub = current_chunk[s : s + max_chunk_chars].strip()
                if len(sub) >= min_chunk_chars:
                    chunks.append(sub)

    return chunks


def _normalize_ocr_glued_text(text: str) -> str:
    """Repair OCR-glued tokens and digit-letter substitutions in scanned campus PDFs."""
    if not text:
        return ""
    cleaned = text
    ocr_fixes = {
        "H0STEL": "HOSTEL",
        "PGcET": "PGCET",
        "1oth": "10th",
        "RVcE": "RVCE",
        "RvcE": "RVCE",
        "Finaily": "Finally",
        "Answcr Paper Vicwing": "Answer Paper Viewing",
        "Answcr": "Answer",
        "Vicwing": "Viewing",
        "crcdentials": "credentials",
        "arrivc": "arrive",
        "Incontinuationtotheearlier": "In continuation to the earlier",
        "MANDATORYDOCUMENTSTOBESUBMITTEDATTHETIMEOFADMISSION": "MANDATORY DOCUMENTS TO BE SUBMITTED AT THE TIME OF ADMISSION",
        "ADMISSIONPROCESS&FEESDETAILSFOR1StYEARM.TECH&MCA": "ADMISSION PROCESS & FEES DETAILS FOR 1st YEAR M.TECH & MCA",
        "ADMISSIONPROCESS&FEESDETAILSFOR": "ADMISSION PROCESS & FEES DETAILS FOR ",
        "FEESDETAILSFOR": "FEES DETAILS FOR ",
        "FORBOYSHOSTELDETAILSCONTACT": "FOR BOYS HOSTEL DETAILS CONTACT",
        "FORGIRLSHOSTELDETAILSCONTACT": "FOR GIRLS HOSTEL DETAILS CONTACT",
        "COLLEGETRANSPORTATIONCHARGES": "COLLEGE TRANSPORTATION CHARGES",
        "HOWTOREACHRVCOLLEGEOFENGINEERING": "HOW TO REACH RV COLLEGE OF ENGINEERING",
        "ForPGCETAllotted": "For PGCET Allotted",
        "ForManagementQuota": "For Management Quota",
        "AllottedStudents": "Allotted Students",
        "KEAConfirmationSlip": "KEA Confirmation Slip",
        "VerificationSlip": "Verification Slip",
        "VerificationSlipDownloaded": "Verification Slip Downloaded",
        "VerificationSlipdownloaded": "Verification Slip downloaded",
        "fromKEA": "from KEA",
        "theranksheetofPGCETorValid": "the rank sheet of PGCET or Valid",
        "OriginalMarksCardsof10thand12th": "Original Marks Cards of 10th and 12th",
        "Allotmentorderandfeespaidreceipthastobedownloaded": "Allotment order and fees paid receipt has to be downloaded ",
        "necessarydocuments": "necessary documents",
        "Originaldocumentswillbedone": "Original documents will be done",
        "Original documentswillbe done": "Original documents will be done",
        "ReportatCivil": "Report at Civil ",
        "EngineeringDepartment": "Engineering Department ",
        "TakeaTokenandwaitforyourturn": "Take a token and wait for your turn ",
        "OnlineAdmissionEntryofApplicationForm": "Online Admission Entry of Application Form ",
        "KrishnaGardenHostel": "Krishna Garden Hostel",
        "ForGirlsStudentsstayingin": "For Girls Students staying in ",
        "GirlsStudentsstayingin": "Girls Students staying in ",
        "StudentloginofKE Aisrequired,alongwith": "Student login of KEA is required, along with",
        "togetOTP": "to get OTP",
        "canbepaid": "can be paid",
        "Ifpaying thefeesat": "If paying the fees at",
        "intheformofDD,then": "in the form of DD, then",
        "theback sideofDD.Thesame tobehandedover": "the back side of DD. The same to be handed over",
        "Cashierat": "Cashier at",
        "Studentwill bedirectedto": "Student will be directed to",
        "Labin": "Lab in",
        "todownload": "to download",
        "Downloadingthe": "Downloading the",
        "isrequiredput": "is required to put",
        "onceagainreport": "once again report",
        "Tokenwill beissued andwaitfortheturnfor": "Token will be issued and wait for the turn for",
        "Entryof": "Entry of",
        "Aftersuccessfulentryof applicationform,Printoutof": "After successful entry of application form, Printout of",
        "applicationwillbegiven,then,thecandidate&parent haveto": "application will be given, then, the candidate & parent have to",
        "asgivenbelowtothe": "as given below to the",
        "obtaintheacknowledgementforsubmitteddocumentsfromtheadmissionauthority": "obtain the acknowledgement for submitted documents from the admission authority",
    }
    for bad, good in ocr_fixes.items():
        cleaned = cleaned.replace(bad, good)

    # Split glued CamelCase / Acronym-Word boundaries (e.g., "SlipandVerification" -> "Slip and Verification")
    cleaned = re.sub(r"([a-z]{2,})([A-Z][a-z]{2,})", r"\1 \2", cleaned)
    cleaned = re.sub(r"([A-Z]{2,})([A-Z][a-z]{3,})", r"\1 \2", cleaned)
    cleaned = re.sub(r"\b(Step)(\d+)\b", r"\1 \2", cleaned)
    return cleaned


def _strip_distribution_footer_for_search(text: str) -> str:
    """Remove boilerplate 'Copy to: ... Chief Wardens, RVCE Boys & Girls Hostel ...' footers from search index text."""
    if not text:
        return ""
    return re.sub(
        r"Copy\s+to\s*:.*?(?:Boys\s*&\s*Girls\s*Hostel|for\s+information\s+and\s+needful).*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()


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
    from src.temporal import enrich_record_temporal_metadata, extract_dates_from_text, parse_datetime_utc

    doc_id = record.get("document_id", "unknown_doc")
    source_url = record.get("source_url", "")
    title = (record.get("title") or "RVCE Document").strip()
    topic = record.get("topic")
    content = _normalize_ocr_glued_text(record.get("content", "").strip())
    source_type = record.get("source_type")
    sender = record.get("sender")
    sender_name = record.get("sender_name")
    filename = record.get("filename")
    gmail_message_id = record.get("gmail_message_id")
    parent_document_id = record.get("parent_document_id")
    email_subject = record.get("email_subject")
    parent_email_body = record.get("parent_email_body")

    temporal_meta = enrich_record_temporal_metadata(record)
    doc_date = temporal_meta.get("document_date")
    received_at = temporal_meta.get("received_at")
    published_at = temporal_meta.get("published_at")
    updated_at = temporal_meta.get("updated_at")
    last_mod = record.get("last_modified") if "last_modified" in record else temporal_meta.get("last_modified")
    doc_dt = parse_datetime_utc(doc_date)
    default_year = doc_dt.year if doc_dt else None

    # Build optional extra metadata prefix for searchable_text when email/attachment fields exist
    extra_meta_parts = [p for p in (sender_name, sender, filename, email_subject) if p and str(p).strip() and str(p).strip() != title]
    extra_meta_prefix = f" | {' | '.join(str(p).strip() for p in extra_meta_parts)}" if extra_meta_parts else ""

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

        # Avoid duplicating generic sitewide footer blocks (e.g. Principal's Office)
        # across every single subpage or department page. Only keep it under the primary homepage.
        if "principal" in h_norm and source_url != "https://rvce.edu.in/":
            continue

        # Normalize line endings without destroying paragraph double-newlines
        norm_block = text_block.replace("\r\n", "\n").replace("\r", "\n")
        cleaned_block = re.sub(r"\n{3,}", "\n\n", norm_block).strip()
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
                searchable_body = _strip_distribution_footer_for_search(cleaned_block) or cleaned_block
                searchable = f"{title} | {heading}{extra_meta_prefix}\n{searchable_body}"
                chunk_events = extract_dates_from_text(f"{title} {heading}\n{cleaned_block}", default_year=default_year)
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
                        source_type=source_type,
                        sender=sender,
                        sender_name=sender_name,
                        filename=filename,
                        gmail_message_id=gmail_message_id,
                        parent_document_id=parent_document_id,
                        document_date=doc_date,
                        received_at=received_at,
                        published_at=published_at,
                        updated_at=updated_at,
                        event_dates=chunk_events or temporal_meta.get("event_dates"),
                        email_subject=email_subject,
                        parent_email_body=parent_email_body,
                    )
                )
                chunk_counter += 1
        else:
            # Paragraph and window-based sub-chunking with overlap for long sections
            sub_chunks = _sub_chunk_section(
                cleaned_block=cleaned_block,
                max_chunk_chars=max_chunk_chars,
                min_chunk_chars=min_chunk_chars,
                overlap_chars=overlap_chars,
            )
            for sub_text in sub_chunks:
                chunk_id = f"{doc_id}_c{chunk_counter:02d}"
                searchable_body = _strip_distribution_footer_for_search(sub_text) or sub_text
                searchable = f"{title} | {heading}{extra_meta_prefix}\n{searchable_body}"
                chunk_events = extract_dates_from_text(f"{title} {heading}\n{sub_text}", default_year=default_year)
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=doc_id,
                        source_url=source_url,
                        document_title=title,
                        heading=heading,
                        text=sub_text,
                        searchable_text=searchable,
                        topic=topic,
                        last_modified=last_mod,
                        char_length=len(sub_text),
                        word_count=len(sub_text.split()),
                        source_type=source_type,
                        sender=sender,
                        sender_name=sender_name,
                        filename=filename,
                        gmail_message_id=gmail_message_id,
                        parent_document_id=parent_document_id,
                        document_date=doc_date,
                        received_at=received_at,
                        published_at=published_at,
                        updated_at=updated_at,
                        event_dates=chunk_events or temporal_meta.get("event_dates"),
                        email_subject=email_subject,
                        parent_email_body=parent_email_body,
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

    def load_and_index(self, dataset_path: str = DEFAULT_DATASET_PATH) -> int:
        """Load JSONL dataset and build retrieval index."""
        resolved_path = _resolve_path(dataset_path)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Dataset path not found: {resolved_path}")

        records = _load_dataset_records(resolved_path)
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
            result = _chunk_to_result_dict(chunk, score)
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

    def load_and_index(self, dataset_path: str = DEFAULT_DATASET_PATH) -> int:
        """Load JSONL dataset and index chunks for semantic retrieval."""
        resolved_path = _resolve_path(dataset_path)
        records = _load_dataset_records(resolved_path)
        return self.index_records(records)

    def compute_corpus_fingerprint(self) -> str:
        """Compute a deterministic SHA-256 fingerprint over all indexed corpus chunks and texts."""
        hasher = hashlib.sha256()
        for chunk in self.chunks:
            cid = getattr(chunk, "chunk_id", "") or ""
            txt = getattr(chunk, "searchable_text", None) or getattr(chunk, "text", "") or ""
            hasher.update(cid.encode("utf-8", errors="ignore"))
            hasher.update(b"\x00")
            hasher.update(txt.encode("utf-8", errors="ignore"))
            hasher.update(b"\x00")
        return hasher.hexdigest()

    def encode_corpus(self, batch_size: int = 32, cache_path: Optional[str] = None) -> Optional[np.ndarray]:
        """Encode all indexed chunks once and cache embeddings in memory and on disk.

        Cached embeddings are reused only when the stored fingerprint and chunk count
        match the exact content and ordering of the current corpus.
        """
        if self.document_embeddings is not None:
            return self.document_embeddings
        if not self.chunks:
            return None

        effective_cache_path = _resolve_path(cache_path) if cache_path else _resolve_path(DEFAULT_EMBEDDINGS_CACHE_PATH)
        current_fingerprint = self.compute_corpus_fingerprint()
        meta_path = Path(effective_cache_path).with_suffix(".meta.json")

        if os.path.isfile(effective_cache_path):
            try:
                cache_valid = True
                if meta_path.is_file():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        if (
                            meta.get("fingerprint") != current_fingerprint
                            or meta.get("chunk_count") != len(self.chunks)
                            or (meta.get("model_name") and meta.get("model_name") != self.model_name)
                        ):
                            logger.info(
                                "Embedding cache fingerprint or chunk mismatch for %s; regenerating embeddings.",
                                effective_cache_path,
                            )
                            cache_valid = False
                    except Exception as meta_err:
                        logger.warning("Failed to read embedding cache metadata %s: %s", meta_path, meta_err)

                if cache_valid:
                    cached = np.load(effective_cache_path)
                    if isinstance(cached, np.ndarray) and cached.shape[0] == len(self.chunks) and cached.ndim == 2:
                        self.document_embeddings = cached
                        logger.info("Loaded %d semantic embeddings from cache: %s", len(self.chunks), effective_cache_path)
                        # Ensure metadata is persisted if missing
                        if not meta_path.is_file():
                            try:
                                meta_data = {
                                    "chunk_count": len(self.chunks),
                                    "fingerprint": current_fingerprint,
                                    "model_name": self.model_name,
                                }
                                meta_path.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")
                            except Exception:
                                pass
                        return self.document_embeddings
            except Exception as load_err:
                logger.warning("Failed to load cached embeddings from %s: %s", effective_cache_path, load_err)

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
            try:
                os.makedirs(os.path.dirname(effective_cache_path), exist_ok=True)
                np.save(effective_cache_path, self.document_embeddings)
                meta_data = {
                    "chunk_count": len(self.chunks),
                    "fingerprint": current_fingerprint,
                    "model_name": self.model_name,
                }
                meta_path.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")
                logger.info("Saved %d semantic embeddings and metadata to cache: %s", len(self.chunks), effective_cache_path)
            except Exception as save_err:
                logger.warning("Failed to save embeddings to %s: %s", effective_cache_path, save_err)
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

        self._load_model()
        if self.model_error or self.model is None:
            logger.warning("Query embedding model unavailable: %s", self.model_error or "unknown model error")
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
            ranked_results.append(_chunk_to_result_dict(chunk, score))

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

    def load_and_index(self, dataset_path: str = DEFAULT_DATASET_PATH) -> int:
        resolved_path = _resolve_path(dataset_path)
        records = _load_dataset_records(resolved_path)
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


def reset_retriever_caches() -> None:
    """Clear all in-memory retriever caches so fresh dataset changes are re-indexed."""
    _DEFAULT_RETRIEVER_CACHE.clear()
    _DEFAULT_SEMANTIC_RETRIEVER_CACHE.clear()
    _DEFAULT_HYBRID_RETRIEVER_CACHE.clear()
    logger.info("Retriever in-memory caches successfully reset.")


def get_default_retriever(
    dataset_path: str = DEFAULT_DATASET_PATH,
) -> TFIDFRetriever:
    """Factory function returning an initialized, indexed TF-IDF retriever."""
    resolved_path = _resolve_path(dataset_path)
    if resolved_path not in _DEFAULT_RETRIEVER_CACHE:
        retriever = TFIDFRetriever()
        retriever.load_and_index(resolved_path)
        _DEFAULT_RETRIEVER_CACHE[resolved_path] = retriever
    return _DEFAULT_RETRIEVER_CACHE[resolved_path]


def get_default_semantic_retriever(
    dataset_path: str = DEFAULT_DATASET_PATH,
    model_name: str = DEFAULT_SEMANTIC_MODEL,
    device: str = "cpu",
) -> SemanticRetriever:
    """Factory function returning an initialized semantic retriever."""
    resolved_path = _resolve_path(dataset_path)
    cache_key = f"{resolved_path}:{model_name}:{device}"
    if cache_key not in _DEFAULT_SEMANTIC_RETRIEVER_CACHE:
        retriever = SemanticRetriever(model_name=model_name, device=device)
        retriever.load_and_index(resolved_path)
        retriever.encode_corpus()
        _DEFAULT_SEMANTIC_RETRIEVER_CACHE[cache_key] = retriever
    return _DEFAULT_SEMANTIC_RETRIEVER_CACHE[cache_key]


def get_default_hybrid_retriever(
    dataset_path: str = DEFAULT_DATASET_PATH,
    model_name: str = DEFAULT_SEMANTIC_MODEL,
    reranker_name: str = DEFAULT_CROSS_ENCODER_MODEL,
    device: str = "cpu",
    fusion_method: str = "linear",
    semantic_weight: float = 0.5,
    tfidf_weight: float = 0.5,
) -> HybridRetriever:
    """Factory function returning an initialized hybrid retriever."""
    resolved_path = _resolve_path(dataset_path)
    cache_key = f"{resolved_path}:{model_name}:{reranker_name}:{device}:{fusion_method}:{semantic_weight}:{tfidf_weight}"
    if cache_key not in _DEFAULT_HYBRID_RETRIEVER_CACHE:
        tfidf_retriever = get_default_retriever(resolved_path)
        semantic_retriever = get_default_semantic_retriever(dataset_path=resolved_path, model_name=model_name, device=device)
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



def _load_dataset_records(dataset_path: str = DEFAULT_DATASET_PATH) -> List[Dict[str, Any]]:
    resolved_path = _resolve_path(dataset_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Dataset path not found: {resolved_path}")
    records: List[Dict[str, Any]] = []
    with open(resolved_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line %d in %s", line_num, resolved_path)
    return records


def enrich_with_companion_chunks(
    results: List[Dict[str, Any]],
    retriever: Any = None,
    max_companions: int = 2,
) -> List[Dict[str, Any]]:
    """Pair retrieved email notices with their PDF attachment chunks (and vice versa).

    Many campus email bodies simply say 'Kindly find the attached circular regarding X',
    while the substantive schedule/fee details live in the child PDF attachment record
    sharing the same ``gmail_message_id`` or ``parent_document_id``.
    """
    if not results or retriever is None:
        return list(results)

    chunks: List[DocumentChunk] = getattr(retriever, "chunks", None) or getattr(
        getattr(retriever, "tfidf_retriever", None), "chunks", []
    )
    if not chunks or not isinstance(chunks, list):
        return list(results)

    existing_chunk_ids = {r.get("chunk_id") for r in results if r.get("chunk_id")}
    target_msg_ids = {r.get("gmail_message_id") for r in results if r.get("gmail_message_id")}
    target_doc_ids = {r.get("document_id") for r in results if r.get("document_id")}
    target_parent_ids = {r.get("parent_document_id") for r in results if r.get("parent_document_id")}

    if not target_msg_ids and not target_parent_ids:
        # Also check gmail://messages/<id> URLs if gmail_message_id was not explicitly set
        for r in results:
            url = str(r.get("source_url") or "")
            m = re.match(r"^gmail://messages/([^/]+)", url)
            if m:
                target_msg_ids.add(m.group(1))

    if not target_msg_ids and not target_parent_ids:
        return list(results)

    enriched = list(results)
    added = 0
    base_score = float(results[0].get("relevance_score", 0.1) or 0.1)

    for chunk in chunks:
        if added >= max_companions:
            break
        if chunk.chunk_id in existing_chunk_ids:
            continue

        c_msg_id = getattr(chunk, "gmail_message_id", None)
        if not c_msg_id and chunk.source_url:
            m = re.match(r"^gmail://messages/([^/]+)", str(chunk.source_url))
            if m:
                c_msg_id = m.group(1)

        c_parent_id = getattr(chunk, "parent_document_id", None)
        c_doc_id = chunk.document_id

        is_companion = (
            (c_msg_id and c_msg_id in target_msg_ids)
            or (c_parent_id and c_parent_id in target_doc_ids)
            or (c_doc_id and c_doc_id in target_parent_ids)
        )
        if is_companion:
            existing_chunk_ids.add(chunk.chunk_id)
            comp_dict = _chunk_to_result_dict(chunk, max(0.05, base_score * 0.85))
            if not comp_dict.get("gmail_message_id") and c_msg_id:
                comp_dict["gmail_message_id"] = c_msg_id
            enriched.append(comp_dict)
            added += 1

    return enriched


def query_campus_knowledge(
    query: str,
    top_k: int = 5,
    score_threshold: float = 0.05,
    dataset_path: str = DEFAULT_DATASET_PATH,
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
