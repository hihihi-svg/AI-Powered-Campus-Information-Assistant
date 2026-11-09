"""Date normalization, event-date extraction, and temporal query understanding.

Provides canonical ISO 8601 date handling across website documents, PDFs,
emails, and email attachments, and supports date-aware retrieval, filtering,
chronological sorting, and superseding-notice resolution.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import email.utils
import re
from typing import Any, Dict, List, Optional, Set, Tuple

MONTH_NAME_TO_NUM: Dict[str, int] = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

TEMPORAL_FILLER_TOKENS: Set[str] = {
    "latest",
    "recent",
    "most",
    "newest",
    "new",
    "last",
    "upcoming",
    "next",
    "future",
    "yesterday",
    "today",
    " chronological",
    "chronologically",
    "announced",
    "released",
    "sent",
    "received",
    "published",
    "update",
    "updates",
    "month",
    "week",
    "year",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
}


def parse_datetime_utc(value: Any) -> Optional[datetime]:
    """Parse any supported date/timestamp representation into a UTC-aware datetime.

    Never fabricates a date when ``value`` is empty or unparseable.
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    if isinstance(value, (int, float)):
        try:
            # Handle millisecond epoch timestamps (e.g. Gmail internalDate)
            ts = float(value)
            if ts > 1e11:
                ts = ts / 1000.0
            if ts <= 0:
                return None
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None

    raw = str(value).strip()
    if not raw:
        return None

    # Pure digits: millisecond or second epoch
    if raw.isdigit():
        try:
            ts = float(raw)
            if ts > 1e11:
                ts = ts / 1000.0
            if ts > 946684800:  # >= 2000-01-01
                return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            pass

    # 1. Try ISO 8601
    iso_candidate = raw
    if iso_candidate.endswith("Z"):
        iso_candidate = iso_candidate[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(iso_candidate)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass

    # 2. Try RFC 2822 / RFC 1123 (e.g. "Wed, 30 Sep 2026 04:50:20 GMT")
    try:
        dt = email.utils.parsedate_to_datetime(raw)
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
    except Exception:
        pass

    # 3. Try DD.MM.YYYY, DD/MM/YYYY, DD-MM-YYYY, or DD.MM.YY
    m_dmy = re.match(r"^(\d{1,2})[./\-](\d{1,2})[./\-](\d{2,4})$", raw)
    if m_dmy:
        day, month, year = int(m_dmy.group(1)), int(m_dmy.group(2)), int(m_dmy.group(3))
        if year < 100:
            year += 2000
        try:
            return datetime(year, month, day, tzinfo=timezone.utc)
        except ValueError:
            pass

    # 4. Try YYYY/MM/DD or YYYY.MM.DD
    m_ymd = re.match(r"^(\d{4})[./\-](\d{1,2})[./\-](\d{1,2})$", raw)
    if m_ymd:
        year, month, day = int(m_ymd.group(1)), int(m_ymd.group(2)), int(m_ymd.group(3))
        try:
            return datetime(year, month, day, tzinfo=timezone.utc)
        except ValueError:
            pass

    # 5. Try "15 November 2026" or "Nov 15, 2026"
    m_named1 = re.match(
        r"^(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)[,\s]+(\d{4})$",
        raw,
        re.IGNORECASE,
    )
    if m_named1:
        day = int(m_named1.group(1))
        mon = MONTH_NAME_TO_NUM.get(m_named1.group(2).lower())
        year = int(m_named1.group(3))
        if mon:
            try:
                return datetime(year, mon, day, tzinfo=timezone.utc)
            except ValueError:
                pass

    m_named2 = re.match(
        r"^([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})$",
        raw,
        re.IGNORECASE,
    )
    if m_named2:
        mon = MONTH_NAME_TO_NUM.get(m_named2.group(1).lower())
        day = int(m_named2.group(2))
        year = int(m_named2.group(3))
        if mon:
            try:
                return datetime(year, mon, day, tzinfo=timezone.utc)
            except ValueError:
                pass

    return None


def normalize_iso_date(value: Any) -> Optional[str]:
    """Normalize a date string/timestamp into canonical UTC ISO 8601 format.

    Returns ``None`` if ``value`` is missing or cannot be parsed.
    """
    dt = parse_datetime_utc(value)
    if dt is None:
        return None
    return dt.isoformat()


def format_date_short(value: Any) -> Optional[str]:
    """Return ``YYYY-MM-DD`` for a parseable date, or ``None``."""
    dt = parse_datetime_utc(value)
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d")


def extract_dates_from_text(text: str, default_year: Optional[int] = None) -> List[str]:
    """Extract explicit calendar dates mentioned in a title, email body, or attachment.

    Returns a sorted list of unique ``YYYY-MM-DD`` strings.
    """
    if not text:
        return []

    found: Set[str] = set()

    # 1. ISO dates: YYYY-MM-DD
    for m in re.finditer(r"\b(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b", text):
        try:
            dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            found.add(dt.strftime("%Y-%m-%d"))
        except ValueError:
            pass

    # 2. Numeric DD.MM.YYYY, DD/MM/YYYY, DD-MM-YYYY, or DD.MM.YY (common in RVCE circulars like 28.09.26)
    for m in re.finditer(r"\b(\d{1,2})[./\-](\d{1,2})[./\-](20\d{2}|\d{2})\b", text):
        day, mon, yr_raw = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year = yr_raw + 2000 if yr_raw < 100 else yr_raw
        if 2020 <= year <= 2035 and 1 <= mon <= 12 and 1 <= day <= 31:
            try:
                dt = datetime(year, mon, day)
                found.add(dt.strftime("%Y-%m-%d"))
            except ValueError:
                pass

    # 3. Range pattern: "4th to 6th November 2026" or "4 - 6 Nov 2026"
    for m in re.finditer(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:to|\-|–|and)\s*(\d{1,2})(?:st|nd|rd|th)?\s+"
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
        r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"(?:[,\s]+(20\d{2}))?\b",
        text,
        re.IGNORECASE,
    ):
        d1, d2 = int(m.group(1)), int(m.group(2))
        mon = MONTH_NAME_TO_NUM.get(m.group(3).lower())
        yr = int(m.group(4)) if m.group(4) else default_year
        if mon and yr and 2020 <= yr <= 2035:
            for d in (d1, d2):
                try:
                    found.add(datetime(yr, mon, d).strftime("%Y-%m-%d"))
                except ValueError:
                    pass

    # 4. "29 October 2026", "24th August 2026", "29 Oct" (when default_year is known)
    for m in re.finditer(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
        r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"(?:[,\s]+(20\d{2}))?\b",
        text,
        re.IGNORECASE,
    ):
        day = int(m.group(1))
        mon = MONTH_NAME_TO_NUM.get(m.group(2).lower())
        yr = int(m.group(3)) if m.group(3) else default_year
        if mon and yr and 2020 <= yr <= 2035:
            try:
                found.add(datetime(yr, mon, day).strftime("%Y-%m-%d"))
            except ValueError:
                pass

    # 5. "October 29, 2026" or "Nov 15, 2026"
    for m in re.finditer(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
        r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(20\d{2}))?\b",
        text,
        re.IGNORECASE,
    ):
        mon = MONTH_NAME_TO_NUM.get(m.group(1).lower())
        day = int(m.group(2))
        yr = int(m.group(3)) if m.group(3) else default_year
        if mon and yr and 2020 <= yr <= 2035:
            try:
                found.add(datetime(yr, mon, day).strftime("%Y-%m-%d"))
            except ValueError:
                pass

    return sorted(found)


def enrich_record_temporal_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    """Compute canonical ISO 8601 temporal metadata for a document or chunk dict.

    Distinguishes:
      - ``received_at``: When an email/attachment was received.
      - ``published_at``: When a website circular/document was published.
      - ``updated_at``: Normalized ``last_modified`` or ``collected_at`` timestamp.
      - ``document_date``: Primary canonical ISO 8601 date of the document (``None`` if missing).
      - ``event_dates``: Explicit scheduled/deadline dates extracted from title/content.
    """
    source_type = str(record.get("source_type") or "").strip().upper()
    doc_id = str(record.get("document_id") or "").strip()
    source_url = str(record.get("source_url") or "").strip()
    title = str(record.get("document_title") or record.get("title") or "").strip()
    text = str(record.get("text") or record.get("content") or "").strip()

    is_email = (
        source_type in {"EMAIL", "EMAIL_ATTACHMENT"}
        or doc_id.startswith("doc_email_")
        or source_url.startswith("gmail://")
        or bool(record.get("gmail_message_id"))
    )

    raw_received = record.get("received_at")
    raw_published = record.get("published_at")
    raw_doc_date = record.get("document_date")
    raw_last_mod = record.get("last_modified")
    raw_updated = record.get("updated_at")
    raw_collected = record.get("collected_at")

    norm_last_mod = normalize_iso_date(raw_last_mod)
    norm_received = normalize_iso_date(raw_received)
    norm_published = normalize_iso_date(raw_published)
    norm_doc_date = normalize_iso_date(raw_doc_date)
    norm_updated = normalize_iso_date(raw_updated) or norm_last_mod or normalize_iso_date(raw_collected)

    # Check if title or URL embeds a circular date (e.g. "28.09.26" or "29.09.26")
    title_dates = extract_dates_from_text(f"{title} {source_url}")

    if is_email:
        received_at = norm_received or norm_doc_date or norm_last_mod
        published_at = norm_published or received_at
        document_date = norm_doc_date or received_at
    else:
        received_at = norm_received
        published_at = norm_published or norm_doc_date or norm_last_mod
        if not published_at and title_dates:
            published_at = normalize_iso_date(title_dates[-1])
        document_date = norm_doc_date or published_at

    # Determine default year for event date extraction from document_date if available
    default_year: Optional[int] = None
    doc_dt = parse_datetime_utc(document_date)
    if doc_dt is not None:
        default_year = doc_dt.year

    existing_events = record.get("event_dates")
    if isinstance(existing_events, list) and existing_events:
        event_dates = sorted({str(d) for d in existing_events if d})
    else:
        event_dates = extract_dates_from_text(f"{title}\n{text[:3000]}", default_year=default_year)

    return {
        "document_date": document_date,
        "received_at": received_at,
        "published_at": published_at,
        "updated_at": norm_updated,
        "last_modified": document_date or norm_last_mod,
        "event_dates": event_dates,
        "source_title": title or "Unknown source",
    }


def parse_temporal_query(
    question: str,
    reference_now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Analyze a user query for temporal intent, date ranges, and sorting requirements.

    Returns a dictionary with:
      - ``is_temporal``: bool
      - ``intent``: ``"latest"`` | ``"upcoming"`` | ``"date_range"`` | ``"chronological"`` | ``None``
      - ``start_date``: Optional[datetime] (UTC inclusive)
      - ``end_date``: Optional[datetime] (UTC inclusive)
      - ``sort_order``: ``"desc"`` | ``"asc"`` | ``None``
      - ``period_label``: Optional[str]
      - ``subject_query``: query string with purely temporal qualifiers stripped
    """
    now = parse_datetime_utc(reference_now) or datetime.now(timezone.utc)
    q_clean = " ".join((question or "").strip().split())
    q_lower = q_clean.lower()

    intent: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    sort_order: Optional[str] = None
    period_label: Optional[str] = None

    # 1. Check explicit relative days ("yesterday", "today", "last N days", "last week")
    if re.search(r"\byesterday\b", q_lower):
        y_day = (now - timedelta(days=1)).date()
        start_date = datetime(y_day.year, y_day.month, y_day.day, 0, 0, 0, tzinfo=timezone.utc)
        end_date = datetime(y_day.year, y_day.month, y_day.day, 23, 59, 59, tzinfo=timezone.utc)
        intent = "date_range"
        sort_order = "desc"
        period_label = f"yesterday ({y_day.isoformat()})"
    elif re.search(r"\btoday\b", q_lower):
        t_day = now.date()
        start_date = datetime(t_day.year, t_day.month, t_day.day, 0, 0, 0, tzinfo=timezone.utc)
        end_date = datetime(t_day.year, t_day.month, t_day.day, 23, 59, 59, tzinfo=timezone.utc)
        intent = "date_range"
        sort_order = "desc"
        period_label = f"today ({t_day.isoformat()})"
    elif re.search(r"\blast\s+(\d+)\s+days?\b", q_lower):
        m_days = re.search(r"\blast\s+(\d+)\s+days?\b", q_lower)
        n_days = int(m_days.group(1)) if m_days else 7
        start_date = now - timedelta(days=n_days)
        end_date = now
        intent = "date_range"
        sort_order = "desc"
        period_label = f"the last {n_days} days"
    elif re.search(r"\b(?:last|past)\s+week\b", q_lower):
        start_date = now - timedelta(days=7)
        end_date = now
        intent = "date_range"
        sort_order = "desc"
        period_label = "the past 7 days"

    # 2. Check explicit "Month YYYY" or "in Month" (e.g., "October 2026", "in November 2026", "in October")
    if start_date is None:
        m_month_yr = re.search(
            r"\b(?:in|during|from|for|of)?\s*"
            r"(january|february|march|april|may|june|july|august|september|october|november|december|"
            r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
            r"(?:\s+(20\d{2}))?\b",
            q_lower,
        )
        # Ensure we don't accidentally match a specific day like "15 November 2026" as whole-month if day precedes/follows
        has_day_with_month = bool(
            re.search(
                r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",
                q_lower,
            )
            or re.search(
                r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?\b",
                q_lower,
            )
        )
        if m_month_yr and not has_day_with_month:
            m_name = m_month_yr.group(1).lower()
            # Only treat bare month name without year as a month filter if preceded by in/from/during/of or accompanied by year
            has_prep_or_year = bool(m_month_yr.group(2)) or bool(
                re.search(
                    r"\b(?:in|during|from|for|of)\s+" + re.escape(m_name) + r"\b",
                    q_lower,
                )
            )
            if has_prep_or_year:
                mon_num = MONTH_NAME_TO_NUM[m_name]
                yr_num = int(m_month_yr.group(2)) if m_month_yr.group(2) else now.year
                start_date = datetime(yr_num, mon_num, 1, 0, 0, 0, tzinfo=timezone.utc)
                if mon_num == 12:
                    end_date = datetime(yr_num + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc) - timedelta(seconds=1)
                else:
                    end_date = datetime(yr_num, mon_num + 1, 1, 0, 0, 0, tzinfo=timezone.utc) - timedelta(seconds=1)
                intent = "date_range"
                sort_order = "desc"
                month_title = datetime(yr_num, mon_num, 1).strftime("%B %Y")
                period_label = month_title

    # 3. Check explicit specific calendar date in query (e.g., "on 6 November 2026" or "28.09.2026")
    if start_date is None:
        explicit_q_dates = extract_dates_from_text(q_clean, default_year=now.year)
        if len(explicit_q_dates) == 1 and any(
            w in q_lower for w in ("on ", "dated ", "from ", "released ", "sent ", "announced ")
        ):
            dt_only = parse_datetime_utc(explicit_q_dates[0])
            if dt_only:
                start_date = datetime(dt_only.year, dt_only.month, dt_only.day, 0, 0, 0, tzinfo=timezone.utc)
                end_date = datetime(dt_only.year, dt_only.month, dt_only.day, 23, 59, 59, tzinfo=timezone.utc)
                intent = "date_range"
                sort_order = "desc"
                period_label = explicit_q_dates[0]
        elif len(explicit_q_dates) >= 2 and ("between" in q_lower or "from" in q_lower):
            d_start = parse_datetime_utc(explicit_q_dates[0])
            d_end = parse_datetime_utc(explicit_q_dates[-1])
            if d_start and d_end:
                start_date = datetime(d_start.year, d_start.month, d_start.day, 0, 0, 0, tzinfo=timezone.utc)
                end_date = datetime(d_end.year, d_end.month, d_end.day, 23, 59, 59, tzinfo=timezone.utc)
                intent = "date_range"
                sort_order = "desc"
                period_label = f"{explicit_q_dates[0]} to {explicit_q_dates[-1]}"

    # 4. Check "latest", "most recent", "newest", "last"
    if intent is None:
        if re.search(
            r"\b(?:latest|most\s+recent|newest|recent|last\s+(?:email|notice|circular|update|announcement))\b",
            q_lower,
        ):
            intent = "latest"
            sort_order = "desc"
        elif re.search(r"\b(?:upcoming|next|forthcoming|future)\b", q_lower):
            intent = "upcoming"
            sort_order = "asc"
        elif re.search(r"\b(?:chronological|chronologically|timeline|oldest\s+first|sort\s+by\s+date)\b", q_lower):
            intent = "chronological"
            sort_order = "asc" if "oldest" in q_lower else "desc"

    # Build topical subject query (removing purely temporal tokens so topical matching focuses on the subject)
    raw_words = re.findall(r"[a-zA-Z0-9_\-\.]+", q_clean)
    subject_words = [
        w
        for w in raw_words
        if w.lower() not in TEMPORAL_FILLER_TOKENS and not re.match(r"^20\d{2}$", w)
    ]
    subject_query = " ".join(subject_words).strip() or q_clean

    return {
        "is_temporal": intent is not None,
        "has_temporal_intent": intent is not None,
        "intent": intent,
        "start_date": start_date,
        "end_date": end_date,
        "sort_order": sort_order,
        "period_label": period_label,
        "subject_query": subject_query,
    }


def _candidate_effective_datetime(candidate: Dict[str, Any]) -> Optional[datetime]:
    """Return the primary UTC datetime of a candidate chunk, or None if missing.

    Never invents a date when metadata is absent.
    """
    for field in ("document_date", "received_at", "published_at", "last_modified", "updated_at"):
        val = candidate.get(field)
        if val:
            dt = parse_datetime_utc(val)
            if dt is not None:
                return dt
    return None


def _candidate_matches_date_window(
    candidate: Dict[str, Any],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
) -> bool:
    """Check whether a candidate's document/received date or scheduled event dates fall in [start_date, end_date]."""
    if start_date is None and end_date is None:
        return True

    doc_dt = _candidate_effective_datetime(candidate)
    if doc_dt is not None:
        in_start = start_date is None or doc_dt >= start_date
        in_end = end_date is None or doc_dt <= end_date
        if in_start and in_end:
            return True

    # Also check extracted event dates only when the document itself has valid date context or explicit event dates
    event_dates = candidate.get("event_dates") or []
    for ev_str in event_dates:
        ev_dt = parse_datetime_utc(ev_str)
        if ev_dt is not None:
            in_start = start_date is None or ev_dt >= start_date
            in_end = end_date is None or ev_dt <= end_date
            if in_start and in_end:
                return True

    return False


def _topic_signature(candidate: Dict[str, Any]) -> str:
    """Compute a normalized subject signature to detect when a newer notice supersedes an older one."""
    title = str(candidate.get("document_title") or candidate.get("title") or "").lower()
    # Strip revision/reminder/extension prefixes and dates to match related notices on the same subject
    cleaned = re.sub(
        r"\b(?:revised|revision|extension|extended|updated|update|reminder\s*\d*|corrigendum|schedule\s+of|registration\s+for|circular\s*:?)\b",
        " ",
        title,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\b\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}\b", " ", cleaned)
    cleaned = re.sub(r"\b20\d{2}(?:\-\d{2,4})?\b", " ", cleaned)
    tokens = [
        t
        for t in re.findall(r"[a-z0-9]+", cleaned)
        if len(t) > 2 and t not in {"the", "and", "for", "with", "from", "pdf", "doc", "rvce", "circular", "notice"}
    ]
    if not tokens:
        return ""
    return " ".join(sorted(set(tokens[:5])))


def annotate_superseding_notices(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Annotate candidates when a newer notice supersedes or updates an older notice on the same subject."""
    if len(candidates) <= 1:
        return list(candidates)

    # Group dated candidates by topic signature
    groups: Dict[str, List[Tuple[datetime, int]]] = {}
    for idx, cand in enumerate(candidates):
        dt = _candidate_effective_datetime(cand)
        sig = _topic_signature(cand)
        if dt is not None and sig:
            groups.setdefault(sig, []).append((dt, idx))

    annotated = [dict(c) for c in candidates]
    for sig, items in groups.items():
        if len(items) < 2:
            continue
        items_sorted = sorted(items, key=lambda x: x[0], reverse=True)
        newest_dt, newest_idx = items_sorted[0]
        oldest_dt, _ = items_sorted[-1]
        if newest_dt <= oldest_dt:
            continue

        newest_title = annotated[newest_idx].get("document_title") or annotated[newest_idx].get("title") or "Newer Notice"
        newest_date_str = newest_dt.strftime("%Y-%m-%d")
        annotated[newest_idx]["temporal_note"] = (
            f"LATEST / SUPERSEDING NOTICE (dated {newest_date_str}) — supersedes earlier notice(s) on this subject."
        )
        annotated[newest_idx]["is_latest_version"] = True

        for dt_val, older_idx in items_sorted[1:]:
            if dt_val < newest_dt:
                older_date_str = dt_val.strftime("%Y-%m-%d")
                annotated[older_idx]["temporal_note"] = (
                    f"OLDER NOTICE (dated {older_date_str}) — superseded/updated by '{newest_title}' ({newest_date_str})."
                )
                annotated[older_idx]["superseded_by"] = newest_title
                annotated[older_idx]["is_latest_version"] = False

    return annotated


def filter_and_rank_by_date(
    candidates: List[Dict[str, Any]],
    temporal_info: Dict[str, Any],
    reference_now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Apply date range filtering, chronological/recency ranking, and superseding-notice resolution.

    Guarantees:
      - Missing dates are NEVER treated as the latest (always ranked below valid dates).
      - Relevance still matters: only relevant candidates passed in are ranked, and
        when a newer notice supersedes an older notice, the newer one is prioritized.
    """
    if not candidates:
        return []

    enriched_candidates: List[Dict[str, Any]] = []
    for c in candidates:
        item = dict(c)
        meta = enrich_record_temporal_metadata(item)
        for k, v in meta.items():
            if v is not None and (k not in item or not item.get(k)):
                item[k] = v
            elif k in {"document_date", "received_at", "published_at", "updated_at", "event_dates"} and v is not None:
                item[k] = v
        enriched_candidates.append(item)

    intent = temporal_info.get("intent")
    start_date = temporal_info.get("start_date")
    end_date = temporal_info.get("end_date")
    sort_order = temporal_info.get("sort_order")

    # 1. Explicit date range filtering
    if intent == "date_range" and (start_date is not None or end_date is not None):
        filtered = [
            c
            for c in enriched_candidates
            if _candidate_matches_date_window(c, start_date=start_date, end_date=end_date)
        ]
    else:
        filtered = list(enriched_candidates)

    if not filtered:
        return []

    # 2. Annotate superseding notices across the candidate set
    filtered = annotate_superseding_notices(filtered)

    # 3. Sort / rank according to temporal intent while respecting relevance and never ranking None as latest
    now_dt = parse_datetime_utc(reference_now) or datetime.now(timezone.utc)
    now_str = now_dt.strftime("%Y-%m-%d")

    if intent == "latest" or (intent == "date_range" and sort_order == "desc") or (intent == "chronological" and sort_order == "desc"):
        # Filter out low-relevance outliers relative to top match so an unrelated newer doc never wins
        max_rel = max(float(c.get("relevance_score", 0.0) or 0.0) for c in filtered)
        rel_floor = max(0.04, max_rel * 0.30)

        def _latest_sort_key(c: Dict[str, Any]) -> Tuple[int, int, float, float]:
            dt = _candidate_effective_datetime(c)
            rel = float(c.get("relevance_score", 0.0) or 0.0)
            is_topical = 1 if rel >= rel_floor else 0
            has_date = 1 if dt is not None else 0
            ts = dt.timestamp() if dt is not None else float("-inf")
            return (is_topical, has_date, ts, rel)

        filtered.sort(key=_latest_sort_key, reverse=True)

    elif intent == "upcoming":
        def _upcoming_sort_key(c: Dict[str, Any]) -> Tuple[int, float, int, float]:
            ev_dates = [d for d in (c.get("event_dates") or []) if d >= now_str]
            dt = _candidate_effective_datetime(c)
            rel = float(c.get("relevance_score", 0.0) or 0.0)
            has_future_event = 1 if ev_dates else 0
            # For upcoming events, earliest future event comes first; otherwise newest notice date
            if ev_dates:
                ev_dt = parse_datetime_utc(ev_dates[0])
                future_sort = -(ev_dt.timestamp() if ev_dt else 0.0)
            else:
                future_sort = float("-inf")
            has_doc_date = 1 if dt is not None else 0
            doc_ts = dt.timestamp() if dt is not None else float("-inf")
            return (has_future_event, future_sort, has_doc_date, doc_ts + rel)

        filtered.sort(key=_upcoming_sort_key, reverse=True)

    elif intent == "chronological" and sort_order == "asc":
        # Oldest first, but missing dates always go last (never first!)
        def _asc_sort_key(c: Dict[str, Any]) -> Tuple[int, float]:
            dt = _candidate_effective_datetime(c)
            if dt is None:
                return (1, float("inf"))
            return (0, dt.timestamp())

        filtered.sort(key=_asc_sort_key)

    else:
        # Non-temporal query: still prioritize a superseding newer notice over its older version when scores are close
        def _default_sort_key(c: Dict[str, Any]) -> Tuple[float, int, float]:
            rel = float(c.get("relevance_score", 0.0) or 0.0)
            is_latest_ver = 1 if c.get("is_latest_version") is True else (0 if c.get("is_latest_version") is None else -1)
            dt = _candidate_effective_datetime(c)
            ts = dt.timestamp() if dt is not None else 0.0
            # Give a modest boost to superseding notices so revised circulars rank above outdated ones
            effective_rel = rel * (1.25 if is_latest_ver == 1 else (0.80 if is_latest_ver == -1 else 1.0))
            return (round(effective_rel, 3), is_latest_ver, ts)

        filtered.sort(key=_default_sort_key, reverse=True)

    return filtered


def sort_results_chronologically(
    candidates: List[Dict[str, Any]],
    descending: bool = True,
) -> List[Dict[str, Any]]:
    """Sort a list of candidate dicts chronologically by their canonical date.

    Documents with missing/unparseable dates are always placed at the end,
    never treated as the newest or oldest dated document.
    """
    dated: List[Tuple[float, Dict[str, Any]]] = []
    undated: List[Dict[str, Any]] = []

    for c in candidates:
        item = dict(c)
        meta = enrich_record_temporal_metadata(item)
        for k, v in meta.items():
            if v is not None and not item.get(k):
                item[k] = v
        dt = _candidate_effective_datetime(item)
        if dt is not None:
            dated.append((dt.timestamp(), item))
        else:
            undated.append(item)

    dated.sort(key=lambda x: x[0], reverse=descending)
    return [item for _, item in dated] + undated


__all__ = [
    "parse_datetime_utc",
    "normalize_iso_date",
    "format_date_short",
    "extract_dates_from_text",
    "enrich_record_temporal_metadata",
    "parse_temporal_query",
    "annotate_superseding_notices",
    "filter_and_rank_by_date",
    "sort_results_chronologically",
]

