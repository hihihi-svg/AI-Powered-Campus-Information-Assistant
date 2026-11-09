"""
gmail_parser.py - Structured parser for Gmail messages and PDF attachments.

Reuses existing Phase I components:
- HTMLExtractor (src/extractors.py) for HTML email body parsing and boilerplate removal.
- PDFExtractor (src/extractors.py) and RapidOCR (src/ocr.py) for PDF circular extraction.
- clean_text and derive_topic (src/cleaning.py) for normalization and categorization.
"""

import base64
import email.utils
import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.cleaning import clean_text, derive_topic
from src.extractors import HTMLExtractor, PDFExtractor
from src.gmail_filter import extract_email_address, get_header_value
from src.temporal import extract_dates_from_text, normalize_iso_date, parse_datetime_utc

logger = logging.getLogger("campus_pipeline.gmail_parser")

_html_extractor = HTMLExtractor()
_pdf_extractor = PDFExtractor()


def _format_date(date_str: str) -> Optional[str]:
    """Parse RFC 2822 or ISO email date string into ISO 8601 UTC timestamp."""
    if not date_str:
        return None
    norm = normalize_iso_date(date_str)
    if norm:
        return norm
    try:
        parsed_tuple = email.utils.parsedate_to_datetime(date_str)
        if parsed_tuple.tzinfo is None:
            parsed_tuple = parsed_tuple.replace(tzinfo=timezone.utc)
        return parsed_tuple.astimezone(timezone.utc).isoformat()
    except Exception:
        return None


def _extract_body_parts(payload: Dict[str, Any]) -> Tuple[str, str]:
    """
    Recursively extract (plain_text, html_text) from a Gmail message payload.
    """
    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data", "")

    plain_content = ""
    html_content = ""

    if body_data:
        try:
            decoded = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace")
            if "html" in mime_type.lower():
                html_content = decoded
            else:
                plain_content = decoded
        except Exception as e:
            logger.warning("Error decoding body data: %s", e)

    parts = payload.get("parts", [])
    for part in parts:
        p_plain, p_html = _extract_body_parts(part)
        if p_plain and not plain_content:
            plain_content = p_plain
        if p_html and not html_content:
            html_content = p_html

    return plain_content, html_content


def _find_attachments(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Recursively collect attachment metadata from a Gmail payload.
    Returns list of dicts: {'filename', 'mimeType', 'attachmentId', 'size'}
    """
    attachments = []
    filename = payload.get("filename", "")
    body = payload.get("body", {})
    att_id = body.get("attachmentId")

    if filename and att_id:
        attachments.append({
            "filename": filename,
            "mimeType": payload.get("mimeType", "application/octet-stream"),
            "attachmentId": att_id,
            "size": body.get("size", 0),
        })

    for part in payload.get("parts", []):
        attachments.extend(_find_attachments(part))

    return attachments


def parse_email_message(
    msg_dict: Dict[str, Any],
    attachment_bytes_map: Optional[Dict[str, bytes]] = None,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parse a full Gmail message dict into:
    1. Parent email document record
    2. List of child attachment document records

    Args:
        msg_dict: Full message payload returned by Gmail API.
        attachment_bytes_map: Mapping of attachmentId -> raw bytes downloaded.

    Returns:
        (parent_record, list_of_child_records)
    """
    attachment_bytes_map = attachment_bytes_map or {}
    message_id = msg_dict.get("id", "")
    payload = msg_dict.get("payload", {})
    headers = payload.get("headers", [])

    subject = get_header_value(headers, "Subject") or "Untitled Notice"
    from_header = get_header_value(headers, "From")
    date_header = get_header_value(headers, "Date")
    display_name, sender_email = extract_email_address(from_header)
    iso_date = _format_date(date_header) or normalize_iso_date(msg_dict.get("internalDate"))
    now_iso = datetime.now(timezone.utc).isoformat()
    email_dt = parse_datetime_utc(iso_date)
    default_year = email_dt.year if email_dt else None

    # 1. Parse Email Body
    plain_text, html_text = _extract_body_parts(payload)

    body_text = ""
    if html_text:
        extracted = _html_extractor.extract(
            raw_html=html_text,
            url=f"gmail://messages/{message_id}",
        )
        body_text = extracted.get("content", "")

    if not body_text and plain_text:
        body_text = clean_text(plain_text)

    # Deterministic document ID for email
    email_hash = hashlib.sha256(message_id.encode("utf-8")).hexdigest()[:16]
    email_doc_id = f"doc_email_{email_hash}"

    parent_record: Optional[Dict[str, Any]] = None

    # Derive topic
    topic = derive_topic(
        url=f"gmail://messages/{message_id}",
        title=subject,
        content=body_text,
    ) or "academics"

    email_content = body_text.strip()
    if not email_content or len(email_content) < 30:
        # Prepend subject to short body so short/PFA circular announcements remain searchable
        email_content = f"{subject.strip()}\n\n{email_content}".strip()

    if email_content and len(email_content) >= 15:
        email_event_dates = extract_dates_from_text(
            f"{subject.strip()}\n{email_content}",
            default_year=default_year,
        )
        parent_record = {
            "document_id": email_doc_id,
            "source_url": f"gmail://messages/{message_id}",
            "title": subject.strip(),
            "source_title": subject.strip(),
            "content": email_content,
            "source_type": "EMAIL",
            "collected_at": now_iso,
            "last_modified": iso_date,
            "document_date": iso_date,
            "received_at": iso_date,
            "published_at": iso_date,
            "updated_at": iso_date or now_iso,
            "event_dates": email_event_dates,
            "topic": topic,
            "extraction_status": "SUCCESS",
            "sender": sender_email,
            "sender_name": display_name,
            "gmail_message_id": message_id,
        }

    # 2. Parse PDF Attachments
    child_records: List[Dict[str, Any]] = []
    attachments = _find_attachments(payload)

    for att in attachments:
        filename = att["filename"]
        att_id = att["attachmentId"]

        # Only process PDFs
        if not filename.lower().endswith(".pdf"):
            continue

        raw_bytes = attachment_bytes_map.get(att_id)
        if not raw_bytes:
            continue

        att_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]
        att_doc_id = f"doc_email_att_{att_hash}"

        # Extract text via PDFExtractor (with auto RapidOCR fallback)
        att_url = f"gmail://messages/{message_id}/attachments/{filename}"
        extracted_pdf = _pdf_extractor.extract(
            raw_bytes=raw_bytes,
            url=att_url,
        )

        pdf_content = extracted_pdf.get("content", "")
        pdf_title = extracted_pdf.get("title") or filename

        # Derive topic from attachment content
        att_topic = derive_topic(
            url=att_url,
            title=pdf_title,
            content=pdf_content,
        ) or topic

        status = "SUCCESS" if pdf_content and len(pdf_content.strip()) >= 50 else "EMPTY"
        att_event_dates = extract_dates_from_text(
            f"{subject.strip()}\n{pdf_title}\n{pdf_content[:3000]}",
            default_year=default_year,
        )

        child_record = {
            "document_id": att_doc_id,
            "source_url": att_url,
            "title": pdf_title.strip(),
            "source_title": pdf_title.strip(),
            "content": pdf_content.strip(),
            "source_type": "EMAIL_ATTACHMENT",
            "collected_at": now_iso,
            "last_modified": iso_date,
            "document_date": iso_date,
            "received_at": iso_date,
            "published_at": iso_date,
            "updated_at": iso_date or now_iso,
            "event_dates": att_event_dates,
            "topic": att_topic,
            "extraction_status": status,
            "parent_document_id": email_doc_id,
            "filename": filename,
            "sender": sender_email,
            "sender_name": display_name,
            "email_subject": subject.strip(),
            "gmail_message_id": message_id,
            "raw_bytes": raw_bytes,  # for raw persistence
        }
        child_records.append(child_record)

    return parent_record, child_records

