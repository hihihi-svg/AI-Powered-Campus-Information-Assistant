"""
gmail_filter.py - Three-tier noise filtering for RVCE campus notices.

Filters incoming emails to ensure only official institutional circulars,
exam schedules, fee updates, and placement announcements enter the corpus.
Excludes personal threads, marketing, spam, and non-campus notifications.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("campus_pipeline.gmail_filter")

# ──────────────────────────────────────────────
# Whitelists
# ──────────────────────────────────────────────
AUTHORIZED_DOMAINS = {
    "rvce.edu.in",
    "rvei.edu.in",
    "vtu.ac.in",
}

AUTHORIZED_OFFICE_PREFIXES = (
    "principal",
    "viceprincipal",
    "coe",
    "dean",
    "hod",
    "accounts",
    "fee",
    "placement",
    "library",
    "hostel",
    "exam",
    "studentaffairs",
    "admission",
)

# ──────────────────────────────────────────────
# Institutional Notice Relevancy Keywords
# ──────────────────────────────────────────────
RELEVANCY_KEYWORDS = [
    r"\bcircular\b",
    r"\bnotification\b",
    r"\bschedule\b",
    r"\btimetable\b",
    r"\bcalendar\b",
    r"\bcie\b",
    r"\bsee\b",
    r"\bexam(?:ination)?s?\b",
    r"\bfees?\b",
    r"\bconcession\b",
    r"\brevaluation\b",
    r"\bgrade\s*cards?\b",
    r"\bmarks\b",
    r"\battendance\b",
    r"\bplacements?\b",
    r"\binternships?\b",
    r"\brecruitment\b",
    r"\bhall\s*tickets?\b",
    r"\badmissions?\b",
    r"\bhostels?\b",
    r"\bcanteen\b",
    r"\bholidays?\b",
    r"\bacademic\b",
    r"\bregulations?\b",
]

RELEVANCY_REGEX = re.compile("|".join(RELEVANCY_KEYWORDS), re.IGNORECASE)

# ──────────────────────────────────────────────
# Negative Patterns (Noise, Personal, Marketing)
# ──────────────────────────────────────────────
EXCLUSION_SUBJECT_PATTERNS = [
    r"^re:\s*(?![a-z0-9/_-]+\s*(circular|notification|exam))",
    r"^fwd:\s*(?![a-z0-9/_-]+\s*(circular|notification|exam))",
    r"\botp\b",
    r"\bverification\s*code\b",
    r"\bpassword\s*(?:reset|changed)\b",
    r"\bsecurity\s*alert\b",
    r"\bwebinar\b",
    r"\bnewsletter\b",
    r"\blinkedin\b",
    r"\bgithub\b",
    r"\bgoogle\s*play\b",
    r"\bswiggy\b",
    r"\bzomato\b",
    r"\bwelcome\s*to\b",
    r"\blogin\s*attempt\b",
]

EXCLUSION_SUBJECT_REGEX = re.compile("|".join(EXCLUSION_SUBJECT_PATTERNS), re.IGNORECASE)


def extract_email_address(from_header: str) -> Tuple[str, str]:
    """
    Parse a 'From' header into (display_name, email_address).
    Example: 'Principal RVCE <principal@rvce.edu.in>' -> ('Principal RVCE', 'principal@rvce.edu.in')
    """
    if not from_header:
        return "", ""
    match = re.search(r"^(.*?)(?:<([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)>)?$", from_header.strip())
    if match:
        display_name = match.group(1).strip().strip('"\'')
        email_addr = match.group(2) if match.group(2) else match.group(1).strip()
        return display_name, email_addr.lower()
    return "", from_header.strip().lower()


def get_header_value(headers: List[Dict[str, str]], name: str) -> str:
    """Retrieve the value of a specific MIME header case-insensitively."""
    name_lower = name.lower()
    for h in headers:
        if h.get("name", "").lower() == name_lower:
            return h.get("value", "")
    return ""


def is_relevant_notice(
    headers: List[Dict[str, str]],
    snippet: str = "",
    attachment_filenames: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """
    Evaluates whether an email represents an official RVCE campus notice.

    Args:
        headers: List of MIME header dicts ({'name': ..., 'value': ...}).
        snippet: Short text snippet extracted by Gmail.
        attachment_filenames: List of attachment filenames.

    Returns:
        (is_accepted: bool, reason: str)
    """
    from_header = get_header_value(headers, "From")
    subject = get_header_value(headers, "Subject")

    if not from_header:
        return False, "REJECTED_MISSING_FROM_HEADER"

    display_name, email_addr = extract_email_address(from_header)

    # 1. Domain Check
    domain = email_addr.split("@")[-1].lower() if "@" in email_addr else ""
    is_whitelisted_domain = any(domain == d or domain.endswith("." + d) for d in AUTHORIZED_DOMAINS)
    is_whitelisted_office = any(email_addr.startswith(prefix) for prefix in AUTHORIZED_OFFICE_PREFIXES)

    if not is_whitelisted_domain and not is_whitelisted_office:
        return False, f"REJECTED_UNAUTHORIZED_DOMAIN_{domain}"

    # 2. Negative Exclusion Patterns
    if not subject or not subject.strip():
        # Empty subject without attachments is noise
        if not attachment_filenames:
            return False, "REJECTED_EMPTY_SUBJECT"

    if subject and EXCLUSION_SUBJECT_REGEX.search(subject):
        return False, f"REJECTED_NEGATIVE_PATTERN_SUBJECT_{subject[:30]}"

    # 3. Notice Relevancy Keywords
    # Check subject, attachment names, and snippet
    text_to_check = f"{subject} {snippet} {' '.join(attachment_filenames or [])}"

    if RELEVANCY_REGEX.search(text_to_check):
        return True, "ACCEPTED_RELEVANT_CAMPUS_NOTICE"

    # If sender is an authoritative office (Principal, CoE, Accounts), accept by default
    if any(email_addr.startswith(prefix) for prefix in ("principal", "coe", "accounts", "dean.academics")):
        return True, "ACCEPTED_OFFICIAL_OFFICE_SENDER"

    return False, "REJECTED_NO_CAMPUS_NOTICE_KEYWORDS"

