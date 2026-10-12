"""
cleaning.py - Text normalization, deterministic ID generation, and verifiable topic categorization.

Adheres strictly to the guideline:
- Do not invent content, dates, labels, or source metadata. Leave unknown values null.
- Topic labels are derived from verifiable URL paths, source metadata, and headings.
"""

import hashlib
import re
import unicodedata
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

# Verifiable URL segment and heading patterns for transparent topic classification
TOPIC_RULES = [
    (
        "departments",
        [r"[/_-]department[/_-]", r"/department/"],
        [r"department of", r"dept\. of", r"computer science", r"mechanical", r"electronics", r"biotechnology", r"civil engineering"],
    ),
    (
        "admissions",
        [r"[/_-]admissions?[/_-]", r"/admission", r"/admissions", r"/management-quota"],
        [r"admission", r"admissions", r"eligibility criteria", r"seat matrix", r"management quota"],
    ),
    (
        "examinations_regulations",
        [r"[/_-]examinations?[/_-]", r"/examination", r"/exam-circular", r"/regulations"],
        [r"examination", r"examinations", r"exam circular", r"academic regulations", r"controller of examination", r"evaluation"],
    ),
    (
        "fees_finance",
        [r"[/_-]fees?[/_-]", r"/fee", r"/fees", r"/fee_payment", r"[/_-]scholarships?[/_-]"],
        [r"fees?", r"fee structure", r"fee circular", r"tuition fee", r"scholarship", r"fee payment"],
    ),
    (
        "placements_careers",
        [r"[/_-]placements?[/_-]", r"/placement", r"/placement_and_training", r"/career", r"/recruitment"],
        [r"placement", r"placements", r"placement and training", r"campus placement", r"internships", r"recruiting companies", r"career"],
    ),
    (
        "facilities_infrastructure",
        [r"[/_-]facilities[/_-]", r"/facilities", r"/library", r"/infrastructure", r"/sports", r"/gym"],
        [r"campus facilities", r"central library", r"sports facility", r"health centre", r"laboratories", r"infrastructure"],
    ),
    (
        "hostels",
        [r"[/_-]hostels?[/_-]", r"/hostel", r"/hostels", r"/accommodation"],
        [r"hostel", r"hostels", r"hostel accommodation", r"hostel rules", r"mess", r"student residence"],
    ),
    (
        "academics",
        [r"[/_-]academics?[/_-]", r"/academics", r"/programmes", r"/academic-circular", r"/curriculum"],
        [r"academic", r"academics", r"academic programmes", r"academic calendar", r"undergraduate programmes", r"postgraduate programmes", r"syllabus"],
    ),
    (
        "research_consulting",
        [r"[/_-]research[/_-]", r"/research", r"/centre-of-excellence", r"/consultancy", r"/publications"],
        [r"research", r"research & development", r"centre of excellence", r"consultancy", r"patents", r"sponsored research"],
    ),
    (
        "student_services_clubs",
        [r"[/_-]nss[/_-]", r"[/_-]ncc[/_-]", r"/student_services", r"/clubs", r"/cultural"],
        [r"student services", r"student welfare", r"clubs and societies", r"nss", r"ncc", r"extra-curricular"],
    ),
    (
        "administration_governance",
        [r"/about_us", r"/iqac", r"/policies-documents", r"/governance", r"/contact", r"/reach-out"],
        [r"governance", r"board of governance", r"internal quality assurance", r"policies and documents", r"contact us", r"administration"],
    ),
]


def generate_document_id(source_url: str, content_type: str = "HTML") -> str:
    """Generate a deterministic, verifiable document ID based on source URL and content type."""
    normalized = (source_url or "").strip().lower()
    hash_digest = hashlib.sha256(f"{normalized}::{content_type}".encode("utf-8")).hexdigest()[:16]
    prefix = "doc_pdf" if content_type.upper() == "PDF" else "doc_html"
    return f"{prefix}_{hash_digest}"


def clean_text(raw_text: str) -> str:
    """
    Normalize text:
    - Unicode normalization (NFKC)
    - Normalize quotes, dashes, non-breaking spaces
    - Remove duplicate blank lines and duplicate consecutive sentences
    - Strip carriage returns and terminal control characters
    """
    if not raw_text:
        return ""

    # Normalize unicode characters
    text = unicodedata.normalize("NFKC", raw_text)

    # Standardize whitespace characters
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ").replace("\u200b", "")

    # Replace multiple spaces with a single space (preserve newlines)
    text = re.sub(r"[ \t]+", " ", text)

    # Clean redundant blank lines (more than two consecutive newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Deduplicate consecutive identical lines (responsive layout artifacts)
    lines = text.split("\n")
    deduped_lines = []
    prev_line = None
    for line in lines:
        stripped = line.strip()
        if stripped != prev_line or not stripped:
            deduped_lines.append(line)
            if stripped:
                prev_line = stripped

    return "\n".join(deduped_lines).strip()


def derive_topic(url: str, title: Optional[str] = None, content: Optional[str] = None) -> Optional[str]:
    """
    Derives topic label transparently from verifiable URL paths, title, and prominent headings.
    Returns None if no verifiable rule matches, adhering strictly to the no-invention policy.
    """
    url_lower = (url or "").lower()
    title_lower = (title or "").lower()

    # Extract first 500 characters of content to check top-level headings without false-positive full text
    snippet_lower = (content[:500] if content else "").lower()

    for topic_name, url_patterns, heading_patterns in TOPIC_RULES:
        # Check URL path match
        for u_pat in url_patterns:
            if re.search(u_pat, url_lower):
                return topic_name

        # Check title match
        for h_pat in heading_patterns:
            if re.search(r"\b" + h_pat + r"\b", title_lower):
                return topic_name

        # Check prominent heading in snippet
        for h_pat in heading_patterns:
            if re.search(r"\b" + h_pat + r"\b", snippet_lower):
                return topic_name

    # Return None if no verifiable topic can be assigned
    return None


def clean_and_process_record(extracted_record: Dict[str, Any], min_length: int = 50) -> Dict[str, Any]:
    """
    Transforms an extracted record into the final processed dataset schema:
    - document_id
    - source_url
    - title
    - content
    - source_type
    - collected_at
    - last_modified
    - topic
    - extraction_status
    """
    source_url = extracted_record.get("source_url")
    source_type = extracted_record.get("source_type", "HTML").upper()
    title = extracted_record.get("title")
    raw_content = extracted_record.get("content", "")
    collected_at = extracted_record.get("collected_at")
    last_modified = extracted_record.get("last_modified")

    cleaned_content = clean_text(raw_content)
    document_id = generate_document_id(source_url, source_type)
    topic = derive_topic(source_url, title, cleaned_content)

    # Determine status
    if not cleaned_content or len(cleaned_content) < min_length:
        status = "EMPTY" if not cleaned_content else "PARTIAL"
    else:
        status = extracted_record.get("extraction_status", "SUCCESS")
        if status not in ("SUCCESS", "PARTIAL", "EMPTY", "FAILED"):
            status = "SUCCESS"

    return {
        "document_id": document_id,
        "source_url": source_url,
        "title": title if title and title.strip() else None,
        "content": cleaned_content,
        "source_type": source_type,
        "collected_at": collected_at,
        "last_modified": last_modified if last_modified and last_modified.strip() else None,
        "topic": topic,
        "extraction_status": status,
    }
