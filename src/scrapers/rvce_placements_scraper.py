"""Scraper and parser for RVCE 2027 placement records from rvce-placements-2027.pages.dev.

Extracts placement records via the backend Supabase REST endpoint or cached data,
normalizes company names, branch codes, CTC values, offer statuses, and locations,
and outputs structured datasets for RAG retrieval and quantitative analytics.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("campus_pipeline.scrapers.rvce_placements")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_SOURCE_URL = "https://rvce-placements-2027.pages.dev/offers"
SUPABASE_URL = "https://abnmqqbaadiozohjgvbw.supabase.co"
SUPABASE_ANON_KEY = "sb_publishable_PZoFpHSioR3yeVz-fHTQdA_7Rdzn0Zc"

DEFAULT_RAW_OUTPUT = REPO_ROOT / "data" / "raw" / "rvce_placements_2027.json"
DEFAULT_PROCESSED_OUTPUT = REPO_ROOT / "data" / "processed" / "rvce_placements_2027.jsonl"

DISCLAIMER_TEXT = "Unofficial placement statistics. Not verified or endorsed by RVCE or the companies listed."

# Branch normalization dictionary mapping aliases and acronyms to canonical display names
BRANCH_CANONICAL_MAP = {
    "CS": "CS",
    "CSE": "CS",
    "COMPUTER SCIENCE": "CS",
    "COMPUTER SCIENCE AND ENGINEERING": "CS",
    "ISE": "ISE",
    "IS": "ISE",
    "INFORMATION SCIENCE": "ISE",
    "INFORMATION SCIENCE AND ENGINEERING": "ISE",
    "AIML": "AIML",
    "AI": "AIML",
    "AI/ML": "AIML",
    "AI & ML": "AIML",
    "ARTIFICIAL INTELLIGENCE": "AIML",
    "CD": "CD",
    "CSD": "CD",
    "COMPUTER SCIENCE AND DESIGN": "CD",
    "CY": "CY",
    "CYS": "CY",
    "CYBER SECURITY": "CY",
    "CYBERSECURITY": "CY",
}

BRANCH_FULL_NAMES = {
    "CS": "Computer Science & Engineering",
    "ISE": "Information Science & Engineering",
    "AIML": "Artificial Intelligence & Machine Learning",
    "CD": "Computer Science & Design",
    "CY": "Cyber Security",
}

# Known company casing preserves acronyms and brand names
COMPANY_CASING_OVERRIDES = {
    "arm": "ARM",
    "bmc": "BMC",
    "hpe": "HPE",
    "sap": "SAP",
    "sap labs": "SAP Labs",
    "amd": "AMD",
    "cdot": "CDOT",
    "c-dot": "CDOT",
    "cisco": "Cisco",
    "auxia": "Auxia",
    "aveva": "Aveva",
    "airbus": "Airbus",
    "ather": "Ather",
    "arcesium": "Arcesium",
    "mercedes-benz": "Mercedes-Benz",
    "mercedes benz": "Mercedes-Benz",
    "natwest": "NatWest",
    "natwest group": "NatWest Group",
    "ubs": "UBS",
    "ti": "Texas Instruments",
    "texas instruments": "Texas Instruments",
    "nxp": "NXP Semiconductors",
    "nxp semiconductors": "NXP Semiconductors",
    "sharechat": "ShareChat",
    "linkedin": "LinkedIn",
    "palo alto": "Palo Alto Networks",
    "palo alto networks": "Palo Alto Networks",
    "ibm": "IBM",
    "inmobi": "InMobi",
    "mufg": "MUFG",
    "zs associates": "ZS Associates",
    "deutsche bank": "Deutsche Bank",
    "wells fargo": "Wells Fargo",
}


def normalize_branch(raw_branch: Optional[str]) -> str:
    """Normalize branch string to canonical branch code."""
    if not raw_branch:
        return "UNKNOWN"
    cleaned = str(raw_branch).strip().upper()
    return BRANCH_CANONICAL_MAP.get(cleaned, cleaned)


def normalize_company(raw_company: Optional[str]) -> str:
    """Normalize company name with trimmed whitespace and brand casing."""
    if not raw_company:
        return "Unknown Company"
    cleaned = re.sub(r"\s+", " ", str(raw_company)).strip()
    lowered = cleaned.lower()
    if lowered in COMPANY_CASING_OVERRIDES:
        return COMPANY_CASING_OVERRIDES[lowered]
    # Default to title-case if all lowercase
    if cleaned.islower():
        return cleaned.title()
    return cleaned


def normalize_location(raw_location: Optional[str]) -> str:
    """Normalize location string with proper capitalization."""
    if not raw_location:
        return "Unknown Location"
    cleaned = re.sub(r"\s+", " ", str(raw_location)).strip()
    # City synonym normalization
    parts = [p.strip().title() for p in cleaned.split(",")]
    normalized_parts = []
    for p in parts:
        if p.lower() in {"bengaluru", "bangalore"}:
            normalized_parts.append("Bengaluru")
        elif p.lower() == "pune":
            normalized_parts.append("Pune")
        elif p.lower() in {"gurugram", "gurgaon"}:
            normalized_parts.append("Gurugram")
        elif p.lower() in {"hyderabad", "hyd"}:
            normalized_parts.append("Hyderabad")
        elif p.lower() in {"mumbai", "bombay"}:
            normalized_parts.append("Mumbai")
        elif p.lower() in {"noida"}:
            normalized_parts.append("Noida")
        elif p.lower() in {"chennai", "madras"}:
            normalized_parts.append("Chennai")
        else:
            normalized_parts.append(p)
    return ", ".join(normalized_parts)


def normalize_ctc(raw_ctc: Any) -> Tuple[Optional[float], Optional[str]]:
    """Parse CTC to float (LPA) and raw string representation."""
    if raw_ctc is None or raw_ctc == "":
        return None, None
    raw_str = str(raw_ctc).strip()
    try:
        val = float(raw_str)
        return round(val, 2), f"{val:.2f} LPA"
    except (ValueError, TypeError):
        # Look for numbers in string (e.g., '40.00 LPA' or '40')
        match = re.search(r"(\d+(?:\.\d+)?)", raw_str)
        if match:
            try:
                val = float(match.group(1))
                return round(val, 2), raw_str
            except ValueError:
                pass
    return None, raw_str


def normalize_status(raw_status: Optional[str]) -> str:
    """Normalize offer status to 'final' or 'non-final'."""
    if not raw_status:
        return "non-final"
    s = str(raw_status).strip().lower()
    if s in {"final", "accepted", "confirmed"}:
        return "final"
    return "non-final"


def normalize_category(raw_category: Optional[str]) -> str:
    """Normalize dream category (Open Dream, Super Dream, Dream)."""
    if not raw_category:
        return "Regular"
    c = str(raw_category).strip()
    c_lower = c.lower()
    if "open" in c_lower:
        return "Open Dream"
    if "super" in c_lower:
        return "Super Dream"
    if "dream" in c_lower:
        return "Dream"
    return c.title()


def normalize_record(raw: Dict[str, Any], scraped_at: Optional[str] = None) -> Dict[str, Any]:
    """Convert raw Supabase record to standardized placement document dictionary."""
    if scraped_at is None:
        scraped_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    rec_id = str(raw.get("id") or "").strip()
    company = normalize_company(raw.get("company"))
    branch = normalize_branch(raw.get("branch"))
    status = normalize_status(raw.get("status"))
    category = normalize_category(raw.get("dream_category"))
    offer_type = (raw.get("offer_type") or "").strip() or "Standard"
    location = normalize_location(raw.get("location"))

    ctc_val, ctc_str = normalize_ctc(raw.get("ctc_lpa"))
    base_val, _ = normalize_ctc(raw.get("base_lpa"))
    stipend_val, _ = normalize_ctc(raw.get("stipend_kpm"))

    batch = int(raw.get("batch") or 2027)
    gender = (raw.get("gender") or "").strip() or None

    additional_details = {
        "person_group": raw.get("person_group"),
        "base_lpa": base_val,
        "stipend_kpm": stipend_val,
        "version": raw.get("version"),
    }

    return {
        "id": rec_id,
        "company_name": company,
        "branch": branch,
        "branch_name": BRANCH_FULL_NAMES.get(branch, branch),
        "offer_status": status,
        "offer_category": category,
        "offer_type": offer_type,
        "ctc": ctc_val,
        "ctc_lpa": ctc_val,
        "ctc_raw": ctc_str,
        "base_lpa": base_val,
        "stipend_kpm": stipend_val,
        "location_or_city": location,
        "gender": gender,
        "offer_year": batch,
        "batch": batch,
        "additional_details": additional_details,
        "source_url": DEFAULT_SOURCE_URL,
        "scraped_at": scraped_at,
        "disclaimer": DISCLAIMER_TEXT,
    }


def fetch_from_supabase(timeout_seconds: int = 25) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fetch all placement offers from Supabase REST API."""
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "User-Agent": "RVCE-Campus-Assistant/1.0",
    }

    # 1. Fetch active version and site state
    state_url = f"{SUPABASE_URL}/rest/v1/site_state?id=eq.1&select=*"
    req = urllib.request.Request(state_url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        state_list = json.loads(resp.read().decode("utf-8"))

    if not state_list:
        raise RuntimeError("No site_state records returned by Supabase.")

    state = state_list[0]
    active_version = state.get("active_version")
    if not active_version:
        raise RuntimeError(f"No active_version in site_state: {state}")

    logger.info("Fetched site_state: active_version=%s, final=%s, non_final=%s",
                active_version, state.get("final_count"), state.get("non_final_count"))

    # 2. Page through placement_offers
    all_raw: List[Dict[str, Any]] = []
    page_size = 250
    offset = 0

    while True:
        offers_url = (
            f"{SUPABASE_URL}/rest/v1/placement_offers"
            f"?version=eq.{active_version}"
            f"&order=branch.asc,company.asc,id.asc"
        )
        page_req = urllib.request.Request(
            offers_url,
            headers={**headers, "Range": f"{offset}-{offset + page_size - 1}"},
        )
        with urllib.request.urlopen(page_req, timeout=timeout_seconds) as resp:
            page = json.loads(resp.read().decode("utf-8"))
            all_raw.extend(page)
            if len(page) < page_size:
                break
            offset += page_size

    return all_raw, state


def load_from_raw_cache(cache_path: str | Path = DEFAULT_RAW_OUTPUT) -> List[Dict[str, Any]]:
    """Load previously cached raw placement JSON records."""
    p = Path(cache_path)
    if not p.exists():
        raise FileNotFoundError(f"Raw cache file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "offers" in data:
        return data["offers"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected JSON format in cache: {p}")


def deduplicate_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate records by unique id while preserving ordering."""
    seen_ids = set()
    deduped = []
    for r in records:
        rid = r.get("id")
        if rid:
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
        deduped.append(r)
    return deduped


def save_raw_dataset(raw_records: List[Dict[str, Any]], state: Dict[str, Any], output_path: str | Path = DEFAULT_RAW_OUTPUT) -> Path:
    """Save raw extracted data and state metadata to JSON."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "source_url": DEFAULT_SOURCE_URL,
            "scraped_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_count": len(raw_records),
            "state": state,
            "disclaimer": DISCLAIMER_TEXT,
        },
        "offers": raw_records,
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d raw placement records to %s", len(raw_records), p)
    return p


def save_processed_dataset(records: List[Dict[str, Any]], output_path: str | Path = DEFAULT_PROCESSED_OUTPUT) -> Path:
    """Save normalized structured records as JSONL."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Saved %d processed placement records to %s", len(records), p)
    return p


def run_scrape_pipeline(
    raw_output: str | Path = DEFAULT_RAW_OUTPUT,
    processed_output: str | Path = DEFAULT_PROCESSED_OUTPUT,
    use_cache: bool = False,
) -> Dict[str, Any]:
    """Execute end-to-end scraper pipeline with caching and normalization."""
    scraped_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    state: Dict[str, Any] = {}

    if use_cache:
        logger.info("Loading placement records from local raw cache: %s", raw_output)
        raw_records = load_from_raw_cache(raw_output)
    else:
        try:
            logger.info("Fetching live placement records from Supabase...")
            raw_records, state = fetch_from_supabase()
            save_raw_dataset(raw_records, state, raw_output)
        except Exception as exc:
            logger.warning("Live fetch failed (%s). Attempting local raw cache fallback...", exc)
            if Path(raw_output).exists():
                raw_records = load_from_raw_cache(raw_output)
            else:
                raise

    # Deduplicate raw records
    raw_deduped = deduplicate_records(raw_records)

    # Normalize each record
    processed_records = [normalize_record(r, scraped_at=scraped_at) for r in raw_deduped]

    # Save to processed JSONL
    save_processed_dataset(processed_records, processed_output)

    # Compute high-level statistics for reporting
    total = len(processed_records)
    final_count = sum(1 for r in processed_records if r["offer_status"] == "final")
    non_final_count = sum(1 for r in processed_records if r["offer_status"] == "non-final")
    unique_companies = sorted(list({r["company_name"] for r in processed_records}))
    branches = sorted(list({r["branch"] for r in processed_records}))

    ctc_vals = [r["ctc"] for r in processed_records if r["ctc"] is not None]
    highest_ctc = max(ctc_vals) if ctc_vals else 0.0
    avg_ctc = round(sum(ctc_vals) / len(ctc_vals), 2) if ctc_vals else 0.0

    stats = {
        "total_records": total,
        "final_count": final_count,
        "non_final_count": non_final_count,
        "unique_companies_count": len(unique_companies),
        "companies": unique_companies,
        "branches": branches,
        "highest_ctc_lpa": highest_ctc,
        "avg_ctc_lpa": avg_ctc,
        "scraped_at": scraped_at,
        "disclaimer": DISCLAIMER_TEXT,
    }

    logger.info("Placement scrape complete: %d records (%d final, %d non-final) across %d companies",
                total, final_count, non_final_count, len(unique_companies))
    return stats


def main() -> None:
    """CLI entrypoint for running the scraper."""
    parser = argparse.ArgumentParser(description="Scrape and normalize RVCE 2027 placement records.")
    parser.add_argument("--raw-path", type=str, default=str(DEFAULT_RAW_OUTPUT), help="Path to raw JSON output")
    parser.add_argument("--processed-path", type=str, default=str(DEFAULT_PROCESSED_OUTPUT), help="Path to processed JSONL output")
    parser.add_argument("--use-cache", action="store_true", help="Use cached raw JSON instead of fetching from API")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    stats = run_scrape_pipeline(
        raw_output=args.raw_path,
        processed_output=args.processed_path,
        use_cache=args.use_cache,
    )
    print("\n================== SCRAPING SUMMARY ==================")
    print(f"Total Offers Extracted: {stats['total_records']}")
    print(f"Final Offers:           {stats['final_count']}")
    print(f"Non-Final Offers:       {stats['non_final_count']}")
    print(f"Unique Companies:       {stats['unique_companies_count']}")
    print(f"Branches:               {', '.join(stats['branches'])}")
    print(f"Highest CTC:            {stats['highest_ctc_lpa']} LPA")
    print(f"Average CTC:            {stats['avg_ctc_lpa']} LPA")
    print(f"Disclaimer:             {stats['disclaimer']}")
    print("======================================================")


if __name__ == "__main__":
    main()
