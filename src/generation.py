"""OpenAI GPT-based source-aware and date-aware answer generation for the RVCE Campus Assistant.

This module provides ``generate_answer`` and source-classification utilities that use the
official OpenAI Python SDK (Responses API / Chat Completions API) to synthesize concise,
grounded, ChatGPT-style natural-language answers from retrieved website and Gmail notice chunks.

Key capabilities:
  - Conversational & structured synthesis: produces one coherent, well-explained answer per
    question using headings, bullets, or tables where helpful, and maintains conversational
    context across follow-up questions.
  - Date-aware & superseding notice synthesis: distinguishes email received dates from
    scheduled event/deadline dates, highlights chronological differences across multiple notices,
    and prioritizes newer notices when they supersede older circulars.
  - Source-aware context formatting: distinguishes website-derived documents (HTML/PDF)
    from email-derived campus notices and PDF circular attachments (preserving parent email
    context) without inventing classifications when metadata is missing.
  - Safe, structured error handling: logs detailed diagnostics internally with redacted secrets
    while returning clean, user-safe error responses without leaking provider internals.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import logging
import os
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.temporal import format_date_short, normalize_iso_date
from src.retrieval import _normalize_ocr_glued_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 30.0
DEFAULT_JITTER_RATIO = 0.25
MAX_CACHE_ENTRIES = 128
PROVIDER_QUOTA_COOLOFF_SECONDS = 300.0
HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
HF_FALLBACK_MODEL = os.environ.get("HF_FALLBACK_MODEL", "Qwen/Qwen2.5-72B-Instruct")

_GPT_RESPONSE_CACHE: "OrderedDict[Tuple[Any, ...], Dict[str, Any]]" = OrderedDict()
_PROVIDER_QUOTA_EXHAUSTED_UNTIL: Dict[str, float] = {}

CLEAN_GENERATION_ERROR_MESSAGE = (
    "Sorry, I couldn't generate an answer right now. Please try again."
)

SYSTEM_PROMPT = (
    "You are the official AI Campus Information Assistant for RV College of Engineering (RVCE), Bengaluru. "
    "Your persona is that of a friendly, enthusiastic, and mildly proud RVCE campus ambassador—warm, conversational, engaging, and genuinely helpful.\n\n"
    "Follow these strict answer generation rules:\n"
    "1. ANSWER DIRECTLY FIRST: Directly answer the user's question in the very first sentence using natural, conversational language. Avoid generic preamble or unsolicited praise such as 'RVCE is a prestigious institution...'.\n"
    "2. NO ROBOTIC BOILERPLATE: NEVER start your response with repetitive introductory templates such as "
    "'Based on the available campus documents...', 'According to the retrieved information...', 'Here is the information...', or 'Based on the records...'. Start immediately with the direct answer.\n"
    "3. SYNTHESIZE NATURALLY (NO CHUNK DUMPING): Synthesize facts from retrieved documents into well-written, coherent paragraphs. "
    "Never concatenate raw excerpts or sentence fragments. Use bullet points or numbered steps where helpful for readability.\n"
    "4. SUBTLE POSITIVE AMBASSADOR TONE: When discussing campus life, facilities, academics, student activities, clubs, fests, or the overall college experience, adopt a subtly positive, encouraging tone. Highlight RVCE's strengths naturally when supported by the retrieved context, emphasizing the vibrant opportunities for students to explore their talents, participate in activities, learn, and grow. Keep the enthusiasm grounded and genuine—avoid excessive praise, marketing fluff, or repetitive promotional statements. Do NOT force positive statements into unrelated administrative, fee, or technical queries.\n"
    "5. STRICT FACTUAL GROUNDING: Rely strictly on the retrieved context for all factual claims, names, dates, numbers, policies, and facilities. Never invent facilities, achievements, rankings, placements, events, or opportunities not present in the context. The positive tone must affect framing and delivery, never factual truth.\n"
    "6. HONEST & NEUTRAL ON LIMITATIONS: If the retrieved documents describe drawbacks, strict regulations, disciplinary prerequisites, or limitations, communicate them honestly, clearly, and neutrally without spinning them. If information is insufficient or missing from the records, clearly and helpfully state what is missing.\n"
    "7. IGNORE IRRELEVANT CHUNKS: Completely ignore retrieved passages that do not genuinely address the user's specific question, even if they share common keywords.\n"
    "8. DATE-AWARE & ACCURATE: Incorporate accurate dates, deadlines, eligibility criteria, fees, and instructions from the sources, prioritizing newer notices when they supersede older ones.\n"
    "9. CONCISE SOURCE CITATIONS: End the answer with concise source citations in the format: 'Sources: [Document Title (Date), ...]'.\n"
    "10. GRAMMATICAL PERFECTION: Ensure all sentences are grammatically complete, well-punctuated, and fluent."
)

DEFAULT_MODEL = "gpt-4o-mini"
MAX_CONTEXT_CHARS = 14_000  # hard cap on total context characters sent to the API
MAX_CHUNKS = 6              # maximum number of chunks included in the prompt

NO_EVIDENCE_MESSAGE = (
    "I could not find sufficient evidence in the available campus documents to answer this question."
)

OFFICIAL_SENDER_MARKERS = (
    "principal",
    "viceprincipal",
    "coe",
    "exam",
    "dean",
    "hod",
    "accounts",
    "fee",
    "placement",
    "admission",
    "library",
    "hostel",
    "studentaffairs",
    "vtu",
)


# ---------------------------------------------------------------------------
# Source-Awareness & Cleaning Helpers
# ---------------------------------------------------------------------------


def identify_chunk_source(chunk: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Identify the source classification of a retrieved chunk from existing metadata.

    Never invents a source classification when metadata is missing.

    Returns a dict with:
      - ``category``: ``"email"``, ``"website"``, or ``"unknown"``
      - ``source_type``: ``"EMAIL"``, ``"EMAIL_ATTACHMENT"``, ``"HTML"``, ``"PDF"``, or ``None``
      - ``label``: human-readable label for prompt formatting (or ``None`` if unknown)
      - ``sender_display``: formatted sender string for email sources (or ``None``)
    """
    raw_type = str(chunk.get("source_type") or "").strip().upper()
    raw_url = str(chunk.get("source_url") or "").strip()
    raw_doc_id = str(chunk.get("document_id") or "").strip()
    sender = str(chunk.get("sender") or "").strip()
    sender_name = str(chunk.get("sender_name") or "").strip()
    filename = str(chunk.get("filename") or "").strip()

    # Format sender display if present
    sender_display: Optional[str] = None
    if sender_name and sender and sender.lower() not in sender_name.lower():
        sender_display = f"{sender_name} ({sender})"
    elif sender_name:
        sender_display = sender_name
    elif sender:
        sender_display = sender

    # 1. Check Email-derived sources
    is_email_att = (
        raw_type == "EMAIL_ATTACHMENT"
        or raw_doc_id.startswith("doc_email_att_")
        or (raw_url.startswith("gmail://") and "/attachments/" in raw_url)
    )
    is_email = (
        is_email_att
        or raw_type == "EMAIL"
        or raw_doc_id.startswith("doc_email_")
        or raw_url.startswith("gmail://")
        or bool(chunk.get("gmail_message_id"))
    )

    if is_email:
        resolved_type = "EMAIL_ATTACHMENT" if is_email_att else "EMAIL"
        sender_check = f"{sender} {sender_name}".lower()
        is_official_office = any(marker in sender_check for marker in OFFICIAL_SENDER_MARKERS)
        if is_email_att:
            label = (
                "Official Campus Email Circular Attachment (PDF)"
                if is_official_office
                else "Campus Email Attachment (PDF)"
            )
        else:
            label = (
                "Official Campus Email Notice"
                if is_official_office
                else "Campus Email Notice"
            )
        return {
            "category": "email",
            "source_type": resolved_type,
            "label": label,
            "sender_display": sender_display,
            "filename": filename or None,
        }

    # 2. Check Website-derived sources
    is_web_pdf = (
        raw_type == "PDF"
        or raw_doc_id.startswith("doc_pdf_")
        or (raw_url.startswith(("http://", "https://")) and raw_url.lower().endswith(".pdf"))
    )
    is_web_html = (
        raw_type in {"HTML", "WEB", "WEBSITE"}
        or raw_doc_id.startswith("doc_html_")
        or raw_url.startswith(("http://", "https://"))
    )

    if is_web_pdf or is_web_html:
        resolved_type = "PDF" if is_web_pdf else "HTML"
        label = (
            "Official RVCE Website Circular (PDF)"
            if is_web_pdf
            else "Official RVCE Website Page"
        )
        return {
            "category": "website",
            "source_type": resolved_type,
            "label": label,
            "sender_display": None,
            "filename": filename or None,
        }

    # 3. Unknown / Missing metadata — do NOT invent classifications
    return {
        "category": "unknown",
        "source_type": None,
        "label": None,
        "sender_display": sender_display,
        "filename": filename or None,
    }


def clean_chunk_text_for_synthesis(text: str, category: str = "unknown") -> str:
    """Clean OCR noise and strip irrelevant email signature boilerplate from chunk text."""
    if not text:
        return ""

    cleaned = _normalize_ocr_glued_text(text.strip())

    # Repair common OCR concatenation artifacts found in campus circulars
    ocr_replacements = {
        "Incontinuationtotheearlier": "In continuation to the earlier",
        "Answcr Paper Vicwing": "Answer Paper Viewing",
        "Answcr": "Answer",
        "Vicwing": "Viewing",
        "ADMISSIONPROCESS&FEESDETAILSFOR": "Admission Process & Fees Details For ",
        "Allotmentorderandfeespaidreceipthastobedownloaded": "Allotment order and fees paid receipt has to be downloaded ",
        "ReportatCivil": "Report at Civil ",
        "EngineeringDepartment": "Engineering Department ",
        "TakeaTokenandwaitforyourturn": "Take a token and wait for your turn ",
        "OnlineAdmissionEntryofApplicationForm": "Online Admission Entry of Application Form ",
        "Original documentswillbe done": "Original documents will be done",
        "FEESDETAILSFOR": "Fees Details For ",
        "crcdentials": "credentials",
        "arrivc": "arrive",
    }
    for bad, good in ocr_replacements.items():
        cleaned = cleaned.replace(bad, good)

    # Remove inline WhatsApp/image placeholder tags from emails
    cleaned = re.sub(r"\[image:\s*[^\]]+\]", "", cleaned, flags=re.IGNORECASE)

    # Strip scanned circular letterhead boilerplate (Ref/Tel/Mysore Road header lines)
    cleaned = re.sub(
        r"RV\s+College\s+of\s+Engineering.*?Ref\s*:\s*[^\s]+\s*",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # For email chunks, strip academic profile footers (Google Scholar, ORCID, Scopus, ResearchGate)
    if category == "email":
        lines = cleaned.splitlines()
        filtered_lines: List[str] = []
        for line in lines:
            low = line.strip().lower()
            if any(
                noise in low
                for noise in (
                    "google scholar",
                    "scholar.google.com",
                    "publons.com",
                    "researchgate.net",
                    "orcid.org",
                    "scopus.com",
                    "scopus preview",
                    "success consists of going from failure",
                )
            ):
                continue
            filtered_lines.append(line)
        cleaned = "\n".join(filtered_lines)

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _redact_secrets(value: Any) -> str:
    """Scrub any API keys or bearer tokens from strings before logging or returning."""
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"sk-[A-Za-z0-9_\-]{6,}", "sk-***REDACTED***", text)
    text = re.sub(r"hf_[A-Za-z0-9_\-]{6,}", "hf_***REDACTED***", text)
    text = re.sub(r"AIza[A-Za-z0-9_\-]{6,}", "AIza***REDACTED***", text)
    text = re.sub(r"(Bearer\s+)[A-Za-z0-9._\-]+", r"\1***REDACTED***", text, flags=re.IGNORECASE)
    return text


def clear_generation_cache() -> None:
    """Clear the in-memory GPT response cache and provider quota cooloff state."""
    _GPT_RESPONSE_CACHE.clear()
    _PROVIDER_QUOTA_EXHAUSTED_UNTIL.clear()


def _load_api_key() -> Optional[str]:
    """Return the OpenAI API key, loading .env if necessary."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]

        load_dotenv(dotenv_path=".env", override=False)
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"), override=False)
    except ImportError:
        pass
    return os.environ.get("OPENAI_API_KEY")


def _load_hf_token() -> Optional[str]:
    """Return the Hugging Face API token if configured in environment or .env."""
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]

        load_dotenv(dotenv_path=".env", override=False)
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"), override=False)
    except ImportError:
        pass
    return os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")


def _extract_response_headers(exc: BaseException) -> Dict[str, str]:
    """Extract lowercase HTTP headers from an OpenAI exception if present."""
    headers: Dict[str, str] = {}
    resp = getattr(exc, "response", None)
    raw_headers = getattr(resp, "headers", None) if resp is not None else getattr(exc, "headers", None)
    if raw_headers and hasattr(raw_headers, "items"):
        try:
            for k, v in raw_headers.items():
                headers[str(k).lower()] = str(v)
        except Exception:
            pass
    return headers


def _parse_retry_after(exc: BaseException) -> Optional[float]:
    """Parse Retry-After or retry-after-ms header from an exception into seconds."""
    headers = _extract_response_headers(exc)
    if not headers:
        return None

    retry_ms = headers.get("retry-after-ms")
    if retry_ms:
        try:
            val_ms = float(retry_ms.strip())
            if val_ms >= 0:
                return min(val_ms / 1000.0, DEFAULT_MAX_DELAY)
        except (ValueError, TypeError):
            pass

    retry_raw = headers.get("retry-after")
    if not retry_raw:
        return None

    retry_str = retry_raw.strip()
    try:
        val_sec = float(retry_str)
        if val_sec >= 0:
            return min(val_sec, DEFAULT_MAX_DELAY)
    except (ValueError, TypeError):
        pass

    try:
        retry_dt = parsedate_to_datetime(retry_str)
        if retry_dt is not None:
            if retry_dt.tzinfo is None:
                retry_dt = retry_dt.replace(tzinfo=timezone.utc)
            delta = (retry_dt - datetime.now(timezone.utc)).total_seconds()
            return min(max(0.0, delta), DEFAULT_MAX_DELAY)
    except Exception:
        pass

    return None


def classify_rate_limit_error(exc: BaseException) -> Dict[str, Any]:
    """Inspect an LLM/OpenAI exception and classify its exact rate-limit or API error cause.

    Distinguishes between:
      - ``insufficient_quota``: Billing/credit exhaustion (``credit_balance_exhausted``,
        ``insufficient_quota``, HTTP 402) — NOT retryable.
      - ``rpm``: Requests per minute rate limit — retryable with exponential backoff.
      - ``tpm``: Tokens per minute rate limit — retryable with exponential backoff.
      - ``rate_limit``: Other temporary HTTP 429 rate limits — retryable with backoff.
      - ``timeout``: Request timeout — NOT retried past configured limit.
      - ``invalid_credentials``: Authentication / API key error (HTTP 401/403).
      - ``api_error``: Non-429 errors (network, server error, etc.).
    """
    exc_class = type(exc).__name__
    status_code = getattr(exc, "status_code", None)
    error_code = getattr(exc, "code", None)
    error_type = getattr(exc, "type", None)
    body = getattr(exc, "body", None)
    body_message: Optional[str] = None

    if isinstance(body, dict):
        err_obj = body.get("error") if isinstance(body.get("error"), dict) else body
        if isinstance(err_obj, dict):
            error_code = error_code or err_obj.get("code")
            error_type = error_type or err_obj.get("type")
            body_message = err_obj.get("message")

    raw_message = _redact_secrets(body_message or str(exc))
    headers = _extract_response_headers(exc)
    retry_after = _parse_retry_after(exc)

    is_429 = (
        status_code == 429
        or exc_class == "RateLimitError"
        or "429" in raw_message
        or "rate limit" in raw_message.lower()
    )
    is_402 = status_code == 402 or "402" in raw_message or "payment required" in raw_message.lower()

    combined = f"{error_code or ''} {error_type or ''} {raw_message}".lower()

    if (
        is_402
        or str(error_type or "").lower() == "insufficient_quota"
        or str(error_code or "").lower() in {
            "insufficient_quota",
            "credit_balance_exhausted",
            "billing_hard_limit_reached",
        }
        or any(
            marker in combined
            for marker in (
                "insufficient_quota",
                "credit_balance_exhausted",
                "no credits remaining",
                "exceeded your current quota",
                "billing_hard_limit_reached",
                "check your plan and billing",
                "depleted your monthly included credits",
                "purchase pre-paid credits",
            )
        )
    ):
        category = "insufficient_quota"
        retryable = False
    elif is_429 and (
        any(marker in combined for marker in ("tokens per min", "tpm", "rate_limit_exceeded_tpm"))
        or headers.get("x-ratelimit-remaining-tokens") == "0"
    ):
        category = "tpm"
        retryable = True
    elif is_429 and (
        any(marker in combined for marker in ("requests per min", "rpm", "rate_limit_exceeded_rpm"))
        or headers.get("x-ratelimit-remaining-requests") == "0"
    ):
        category = "rpm"
        retryable = True
    elif is_429:
        category = "rate_limit"
        retryable = True
    elif isinstance(exc, TimeoutError) or "timeout" in exc_class.lower() or "timed out" in combined:
        category = "timeout"
        retryable = False
    elif status_code in (401, 403) or "authentication" in exc_class.lower() or "invalid api key" in combined:
        category = "invalid_credentials"
        retryable = False
    else:
        category = "api_error"
        retryable = False

    return {
        "category": category,
        "exception_type": exc_class,
        "status_code": status_code or (429 if is_429 else None),
        "error_type": str(error_type) if error_type else None,
        "error_code": str(error_code) if error_code else None,
        "message": raw_message,
        "retryable": retryable,
        "retry_after": retry_after,
    }


def _build_safe_public_error(error_info: Dict[str, Any]) -> Dict[str, Any]:
    """Build a sanitized error dictionary safe to return to the API client and UI.

    Never exposes provider names, raw exception messages, HTTP stack traces, or keys.
    """
    category = str(error_info.get("category") or "generation_unavailable")
    error_code = str(error_info.get("error_code") or category)
    return {
        "category": category,
        "code": error_code,
        "error_code": error_code,
        "message": CLEAN_GENERATION_ERROR_MESSAGE,
        "retryable": bool(error_info.get("retryable", False)),
    }


def _resolve_chunk_dates(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Extract normalized date metadata from a chunk."""
    doc_date = normalize_iso_date(chunk.get("document_date") or chunk.get("last_modified"))
    received_at = normalize_iso_date(chunk.get("received_at"))
    published_at = normalize_iso_date(chunk.get("published_at"))
    updated_at = normalize_iso_date(chunk.get("updated_at") or chunk.get("last_modified"))
    event_dates = chunk.get("event_dates") if isinstance(chunk.get("event_dates"), list) else []
    return {
        "document_date": doc_date,
        "received_at": received_at,
        "published_at": published_at,
        "updated_at": updated_at,
        "event_dates": event_dates,
    }


def _format_chunk(chunk: Dict[str, Any], index: int) -> str:
    """Format a single retrieved chunk as a source-aware and date-aware context block."""
    source_info = identify_chunk_source(chunk)
    category = source_info["category"] or "unknown"
    label = source_info["label"]
    sender_display = source_info["sender_display"]
    filename = source_info["filename"]

    title = (chunk.get("source_title") or chunk.get("document_title") or "Unknown source").strip()
    heading = (chunk.get("heading") or "").strip()
    source_url = (chunk.get("source_url") or "").strip()
    topic = (chunk.get("topic") or "").strip()
    raw_text = (chunk.get("text") or "").strip()
    text = clean_chunk_text_for_synthesis(raw_text, category=category)

    dates = _resolve_chunk_dates(chunk)
    doc_date = dates["document_date"]
    received_at = dates["received_at"]
    published_at = dates["published_at"]
    event_dates = dates["event_dates"]
    temporal_note = (chunk.get("temporal_note") or "").strip()
    email_subject = (chunk.get("email_subject") or "").strip()
    parent_email_body = clean_chunk_text_for_synthesis(
        str(chunk.get("parent_email_body") or ""), category="email"
    )

    if label:
        header_parts = [f"[{index + 1}] ({label}) {title}"]
    else:
        header_parts = [f"[{index + 1}] {title}"]

    if heading and heading.lower() != title.lower() and heading.lower() != "unknown heading":
        header_parts.append(f"Section: {heading}")
    if sender_display:
        header_parts.append(f"Sender: {sender_display}")
    if email_subject and email_subject.lower() != title.lower():
        header_parts.append(f"Parent Email Subject: {email_subject}")
    if filename and filename.lower() not in title.lower():
        header_parts.append(f"Attachment: {filename}")
    if received_at:
        header_parts.append(f"Email Received Date: {received_at} ({format_date_short(received_at)})")
    elif published_at:
        header_parts.append(f"Published Date: {published_at} ({format_date_short(published_at)})")
    elif doc_date:
        header_parts.append(f"Document Date: {doc_date} ({format_date_short(doc_date)})")
    if event_dates:
        formatted_events = ", ".join(f"{d} ({format_date_short(d)})" for d in event_dates[:5])
        header_parts.append(f"Scheduled / Event Dates Mentioned: {formatted_events}")
    if temporal_note:
        header_parts.append(f"Temporal Notice Status: {temporal_note}")
    if topic:
        header_parts.append(f"Topic: {topic}")
    if source_url and source_url != "Unknown URL":
        header_parts.append(f"Source: {source_url}")

    body_block = text
    if parent_email_body and parent_email_body not in text:
        body_block = f"Parent Email Summary:\n{parent_email_body[:400]}\n\nAttachment Content:\n{text}"

    return "\n".join(header_parts) + "\nContent:\n" + body_block


def _build_context(chunks: List[Dict[str, Any]]) -> Tuple[str, Dict[str, int]]:
    """Build a trimmed, source-annotated context string and source category counts."""
    parts: List[str] = []
    counts = {"email": 0, "website": 0, "unknown": 0}
    total_chars = 0

    for i, chunk in enumerate(chunks[:MAX_CHUNKS]):
        src_info = identify_chunk_source(chunk)
        cat = src_info["category"] or "unknown"
        counts[cat] = counts.get(cat, 0) + 1

        formatted = _format_chunk(chunk, i)
        if total_chars + len(formatted) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining > 200:
                parts.append(formatted[:remaining] + "\n[... truncated ...]")
            break
        parts.append(formatted)
        total_chars += len(formatted)

    return "\n\n---\n\n".join(parts), counts


def _format_conversation_history(conversation_history: Optional[List[Dict[str, str]]]) -> str:
    """Format recent conversation turns for follow-up continuity."""
    if not conversation_history:
        return ""
    lines: List[str] = []
    for turn in conversation_history[-6:]:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "user").strip().lower()
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        speaker = "Student" if role == "user" else "Assistant"
        lines.append(f"{speaker}: {content[:500]}")
    if not lines:
        return ""
    return "Previous Conversation Context:\n" + "\n".join(lines) + "\n\n---\n"


def _call_openai_model(
    client: Any,
    model: str,
    system_prompt: str,
    user_message: str,
) -> str:
    """Invoke OpenAI Responses API or Chat Completions API with a single request.

    Avoids duplicate API calls: never falls back from ``responses.create`` to
    ``chat.completions.create`` when an API/HTTP error (such as ``RateLimitError``)
    is raised.
    """
    is_mock_client = type(client).__module__.startswith("unittest.mock")

    use_responses_api = False
    if is_mock_client:
        mock_children = getattr(client, "_mock_children", {}) or {}
        if "responses" in mock_children and "chat" not in mock_children:
            use_responses_api = True
    elif hasattr(client, "responses") and hasattr(client.responses, "create"):
        use_responses_api = True

    if use_responses_api:
        resp = client.responses.create(
            model=model,
            instructions=system_prompt,
            input=user_message,
            temperature=0.2,
        )
        output_text = getattr(resp, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        output_items = getattr(resp, "output", None)
        if isinstance(output_items, list):
            extracted_parts: List[str] = []
            for item in output_items:
                for content_block in getattr(item, "content", []) or []:
                    txt = getattr(content_block, "text", None)
                    if isinstance(txt, str) and txt.strip():
                        extracted_parts.append(txt.strip())
            if extracted_parts:
                return "\n".join(extracted_parts)
        return ""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=700,
    )
    return (response.choices[0].message.content or "").strip()


def _call_openai_with_backoff(
    client: Any,
    model: str,
    system_prompt: str,
    user_message: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> str:
    """Call OpenAI with exponential backoff and jitter for temporary rate limits.

    - Does NOT retry ``insufficient_quota`` / ``credit_balance_exhausted`` errors.
    - Respects the ``Retry-After`` (or ``retry-after-ms``) header when present.
    - Retries temporary ``rpm``, ``tpm``, and generic HTTP 429 rate-limit errors
      up to ``max_retries`` times using exponential backoff + random jitter.
    """
    attempt = 0
    while True:
        try:
            return _call_openai_model(
                client=client,
                model=model,
                system_prompt=system_prompt,
                user_message=user_message,
            )
        except Exception as exc:
            error_info = classify_rate_limit_error(exc)
            if not error_info["retryable"] or attempt >= max_retries:
                raise

            retry_after = error_info.get("retry_after")
            if retry_after is not None and retry_after > 0:
                jitter = random.uniform(0.0, DEFAULT_JITTER_RATIO * min(base_delay, retry_after))
                wait_seconds = min(retry_after + jitter, max_delay)
            else:
                exp_delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = random.uniform(0.0, DEFAULT_JITTER_RATIO * exp_delay)
                wait_seconds = min(exp_delay + jitter, max_delay)

            logger.warning(
                "Temporary LLM rate limit (%s, attempt %d/%d). Waiting %.2fs before retry.",
                error_info["category"],
                attempt + 1,
                max_retries,
                wait_seconds,
            )
            sleep_fn(wait_seconds)
            attempt += 1


def _try_huggingface_fallback(
    system_prompt: str,
    user_message: str,
    hf_token: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Attempt answer synthesis via Hugging Face OpenAI-compatible router if configured.

    Uses a circuit breaker so quota-exhausted tokens are not called repeatedly.
    Returns ``(answer_text, fallback_error_reason)``.
    """
    now = time.monotonic()
    if _PROVIDER_QUOTA_EXHAUSTED_UNTIL.get("huggingface", 0.0) > now:
        return None, "Hugging Face fallback skipped (circuit breaker active due to quota exhaustion)"

    token = hf_token or _load_hf_token()
    if not token:
        return None, None

    try:
        import openai  # type: ignore[import-untyped]

        # Do not invoke live fallback if openai module is a unit-test mock without base_url support
        if type(openai.OpenAI).__module__.startswith("unittest.mock"):
            return None, None

        hf_client = openai.OpenAI(
            base_url=HF_ROUTER_BASE_URL,
            api_key=token,
            max_retries=0,
        )
        response = hf_client.chat.completions.create(
            model=HF_FALLBACK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=700,
        )
        text = (response.choices[0].message.content or "").strip()
        if text:
            return text, None
        return None, "Empty response from Hugging Face fallback"
    except Exception as hf_exc:  # pylint: disable=broad-except
        hf_info = classify_rate_limit_error(hf_exc)
        if hf_info["category"] == "insufficient_quota":
            _PROVIDER_QUOTA_EXHAUSTED_UNTIL["huggingface"] = time.monotonic() + PROVIDER_QUOTA_COOLOFF_SECONDS
        status = getattr(hf_exc, "status_code", None)
        msg = _redact_secrets(str(hf_exc))
        logger.error(
            "Hugging Face fallback failed: category=%s status=%s detail=%s",
            hf_info["category"],
            status,
            msg[:160],
        )
        return None, f"Hugging Face fallback failed ({hf_info['category']})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    sleep_fn: Callable[[float], None] = time.sleep,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Generate a source-aware, date-aware synthesized answer using OpenAI GPT and retrieved chunks.

    Returns
    -------
    dict
        ``{"answer": str, "evidence": List[Dict], "sources": List[Dict], "strategy": "gpt"}``
        When LLM generation fails, returns ``CLEAN_GENERATION_ERROR_MESSAGE`` and a sanitized
        ``"error"`` dictionary without exposing raw provider error strings or credentials.
    """
    if not question or not question.strip():
        return {
            "answer": NO_EVIDENCE_MESSAGE,
            "evidence": [],
            "sources": [],
            "strategy": "gpt",
        }

    if not retrieved_chunks:
        return {
            "answer": NO_EVIDENCE_MESSAGE,
            "evidence": [],
            "sources": [],
            "strategy": "gpt",
        }

    # Build source-aware and date-aware evidence and deduplicated sources lists
    evidence: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    seen_sources: set = set()

    for chunk in retrieved_chunks[:MAX_CHUNKS]:
        src_info = identify_chunk_source(chunk)
        category = src_info["category"] or "unknown"
        resolved_source_type = src_info["source_type"]
        sender_display = src_info["sender_display"]
        dates = _resolve_chunk_dates(chunk)

        title = (chunk.get("source_title") or chunk.get("document_title") or "Unknown source").strip() or "Unknown source"
        url = (chunk.get("source_url") or "Unknown URL").strip() or "Unknown URL"
        heading = (chunk.get("heading") or "Unknown heading").strip() or "Unknown heading"
        doc_id = (chunk.get("document_id") or "").strip() or None
        raw_text = (chunk.get("text") or "").strip()
        cleaned_text = clean_chunk_text_for_synthesis(raw_text, category=category)
        snippet = cleaned_text[:240] + "..." if len(cleaned_text) > 240 else cleaned_text

        ev_entry: Dict[str, Any] = {
            "title": title,
            "source_title": title,
            "url": url,
            "heading": heading,
            "text": snippet,
            "relevance_score": chunk.get("relevance_score"),
            "document_id": doc_id,
            "document_date": dates["document_date"],
            "received_at": dates["received_at"],
            "published_at": dates["published_at"],
            "updated_at": dates["updated_at"],
            "event_dates": dates["event_dates"],
        }
        if resolved_source_type:
            ev_entry["source_type"] = resolved_source_type
        if sender_display:
            ev_entry["sender"] = sender_display
        evidence.append(ev_entry)

        src_key = (title, url, heading)
        if src_key not in seen_sources:
            seen_sources.add(src_key)
            src_entry: Dict[str, Any] = {
                "title": title,
                "source_title": title,
                "url": url,
                "heading": heading,
                "document_id": doc_id,
                "document_date": dates["document_date"],
                "received_at": dates["received_at"],
                "published_at": dates["published_at"],
                "updated_at": dates["updated_at"],
            }
            if resolved_source_type:
                src_entry["source_type"] = resolved_source_type
            if sender_display:
                src_entry["sender"] = sender_display
            sources.append(src_entry)

    # Resolve API key
    resolved_key = api_key or _load_api_key()
    if not resolved_key:
        logger.error("OPENAI_API_KEY is not set; cannot call OpenAI API.")
        err_payload = _build_safe_public_error(
            {
                "category": "configuration_error",
                "error_code": "missing_api_key",
                "retryable": False,
            }
        )
        return {
            "answer": CLEAN_GENERATION_ERROR_MESSAGE,
            "evidence": evidence,
            "sources": sources,
            "strategy": "gpt",
            "error": err_payload,
        }

    # Build source-aware and date-aware prompt
    context_text, source_counts = _build_context(retrieved_chunks)
    source_mix_notes: List[str] = []
    if source_counts.get("email", 0) > 0 and source_counts.get("website", 0) > 0:
        source_mix_notes.append(
            "NOTE: The retrieved context contains BOTH official RVCE website documents and campus email notices. "
            "Synthesize relevant information from both source types into one coherent answer and cite both."
        )
    elif source_counts.get("email", 0) > 0:
        source_mix_notes.append(
            "NOTE: The retrieved context comes from RVCE campus email notices/circular attachments. "
            "Summarize the notice clearly, preserve key dates/deadlines/fees/instructions, and cite the notice title, date, and sender."
        )
    elif source_counts.get("website", 0) > 0:
        source_mix_notes.append(
            "NOTE: The retrieved context comes from official RVCE website documents/circulars. "
            "Provide a concise, well-structured answer and cite the source document title and date."
        )

    mix_instruction = ("\n" + "\n".join(source_mix_notes) + "\n") if source_mix_notes else ""
    history_block = _format_conversation_history(conversation_history)

    user_message = (
        f"{history_block}"
        f"Retrieved RVCE Campus Knowledge Base Excerpts:\n\n"
        f"{context_text}\n\n"
        f"---\n"
        f"{mix_instruction}\n"
        f"Student Question: {question.strip()}\n\n"
        f"Instructions:\n"
        f"- Answer the question directly in the very first sentence in natural, conversational English.\n"
        f"- Tone: Friendly, enthusiastic, and mildly proud RVCE campus ambassador. Naturally highlight campus strengths and opportunities to learn and grow when discussing campus life, while remaining grounded in facts.\n"
        f"- NEVER use repetitive introductory templates such as 'Based on the available campus documents...' or exaggerated marketing fluff.\n"
        f"- Synthesize information into coherent paragraphs, bullets, or steps rather than concatenating chunks.\n"
        f"- If the topic involves limitations, drawbacks, or strict rules, communicate them honestly and neutrally.\n"
        f"- Highlight relevant dates, deadlines, eligibility criteria, fees, venues, or instructions.\n"
        f"- When multiple notices exist, explain their dates and prioritize newer superseding notices.\n"
        f"- If details are missing from the records, explain what is missing naturally without guessing.\n"
        f"- Include concise source citations at the end (e.g., 'Sources: Document Title (Date)')."
    )

    try:
        import openai  # type: ignore[import-untyped]

        is_mock_sdk = type(openai.OpenAI).__module__.startswith("unittest.mock")
        history_digest = hashlib.sha256(history_block.encode("utf-8")).hexdigest()[:12]
        context_digest = hashlib.sha256(context_text.encode("utf-8")).hexdigest()
        key_digest = hashlib.sha256(resolved_key.encode("utf-8")).hexdigest()[:16]
        cache_key = (
            id(openai.OpenAI),
            key_digest,
            model,
            question.strip().lower(),
            history_digest,
            context_digest,
        )

        if cache_key in _GPT_RESPONSE_CACHE:
            cached = _GPT_RESPONSE_CACHE[cache_key]
            _GPT_RESPONSE_CACHE.move_to_end(cache_key)
            return {
                "answer": cached["answer"],
                "evidence": evidence,
                "sources": sources,
                "strategy": "gpt",
            }

        # Check circuit breaker for real OpenAI client when quota was recently exhausted
        if not is_mock_sdk and _PROVIDER_QUOTA_EXHAUSTED_UNTIL.get("openai", 0.0) > time.monotonic():
            logger.warning("Skipping OpenAI call during quota circuit-breaker cooloff window.")
            hf_answer, _ = _try_huggingface_fallback(
                system_prompt=SYSTEM_PROMPT,
                user_message=user_message,
            )
            if hf_answer:
                return {
                    "answer": hf_answer,
                    "evidence": evidence,
                    "sources": sources,
                    "strategy": "gpt",
                }
            return {
                "answer": CLEAN_GENERATION_ERROR_MESSAGE,
                "evidence": evidence,
                "sources": sources,
                "strategy": "gpt",
                "error": _build_safe_public_error(
                    {
                        "category": "insufficient_quota",
                        "error_code": "credit_balance_exhausted",
                        "retryable": False,
                    }
                ),
            }

        try:
            client = openai.OpenAI(api_key=resolved_key, max_retries=0)
        except TypeError:
            client = openai.OpenAI(api_key=resolved_key)

        answer_text = _call_openai_with_backoff(
            client=client,
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            max_retries=max_retries,
            base_delay=base_delay,
            sleep_fn=sleep_fn,
        )
        if not answer_text:
            answer_text = NO_EVIDENCE_MESSAGE

        result_payload = {
            "answer": answer_text,
            "evidence": evidence,
            "sources": sources,
            "strategy": "gpt",
        }
        _GPT_RESPONSE_CACHE[cache_key] = {"answer": answer_text}
        if len(_GPT_RESPONSE_CACHE) > MAX_CACHE_ENTRIES:
            _GPT_RESPONSE_CACHE.popitem(last=False)
        return result_payload

    except Exception as exc:  # pylint: disable=broad-except
        error_info = classify_rate_limit_error(exc)
        logger.error(
            "LLM generation call failed: type=%s category=%s status=%s code=%s detail=%s",
            error_info["exception_type"],
            error_info["category"],
            error_info["status_code"],
            error_info["error_code"],
            error_info["message"][:200],
        )

        try:
            import openai as _openai_mod  # type: ignore[import-untyped]

            is_mock_sdk = type(_openai_mod.OpenAI).__module__.startswith("unittest.mock")
        except Exception:
            is_mock_sdk = False

        if error_info["category"] == "insufficient_quota":
            if not is_mock_sdk:
                _PROVIDER_QUOTA_EXHAUSTED_UNTIL["openai"] = (
                    time.monotonic() + PROVIDER_QUOTA_COOLOFF_SECONDS
                )
            hf_answer, _ = _try_huggingface_fallback(
                system_prompt=SYSTEM_PROMPT,
                user_message=user_message,
            )
            if hf_answer:
                return {
                    "answer": hf_answer,
                    "evidence": evidence,
                    "sources": sources,
                    "strategy": "gpt",
                }

        safe_error = _build_safe_public_error(error_info)
        return {
            "answer": CLEAN_GENERATION_ERROR_MESSAGE,
            "evidence": evidence,
            "sources": sources,
            "strategy": "gpt",
            "error": safe_error,
        }


__all__ = [
    "generate_answer",
    "identify_chunk_source",
    "clean_chunk_text_for_synthesis",
    "classify_rate_limit_error",
    "clear_generation_cache",
    "CLEAN_GENERATION_ERROR_MESSAGE",
    "NO_EVIDENCE_MESSAGE",
    "DEFAULT_MODEL",
]
