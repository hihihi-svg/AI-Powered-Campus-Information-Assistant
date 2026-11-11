"""Placement Analytics and Structured Query Processing for RVCE 2027.

Provides:
1. Exact quantitative calculations (highest package, average CTC, median CTC,
   offer counts by branch, final vs non-final counts, top recruiters, salary filters).
2. Deterministic branch-to-branch comparisons.
3. Natural language query answering for placement queries with verified metrics
   and official disclaimer:
   "Unofficial placement statistics. Not verified or endorsed by RVCE or the companies listed."
4. Conversion of placement records into standardized DocumentChunk objects for RAG retrieval.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("campus_pipeline.placement_analytics")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PLACEMENTS_PATH = REPO_ROOT / "data" / "processed" / "rvce_placements_2027.jsonl"
DEFAULT_SOURCE_URL = "https://rvce-placements-2027.pages.dev/offers"
DISCLAIMER_TEXT = "Unofficial placement statistics. Not verified or endorsed by RVCE or the companies listed."

BRANCH_SYNONYMS = {
    "cs": "CS",
    "cse": "CS",
    "computer science": "CS",
    "computer science and engineering": "CS",
    "is": "ISE",
    "ise": "ISE",
    "information science": "ISE",
    "information science and engineering": "ISE",
    "aiml": "AIML",
    "ai": "AIML",
    "ai/ml": "AIML",
    "ai & ml": "AIML",
    "artificial intelligence": "AIML",
    "cd": "CD",
    "csd": "CD",
    "computer science and design": "CD",
    "design": "CD",
    "cy": "CY",
    "cys": "CY",
    "cyber security": "CY",
    "cybersecurity": "CY",
}

_CACHED_RECORDS: Optional[List[Dict[str, Any]]] = None


def load_placement_records(filepath: str | Path = DEFAULT_PLACEMENTS_PATH, force_reload: bool = False) -> List[Dict[str, Any]]:
    """Load processed 2027 placement records from JSONL."""
    global _CACHED_RECORDS
    if _CACHED_RECORDS is not None and not force_reload:
        return _CACHED_RECORDS

    p = Path(filepath)
    if not p.is_absolute() and not p.exists() and (REPO_ROOT / p).exists():
        p = REPO_ROOT / p

    if not p.exists():
        logger.warning("Placement dataset not found at %s", p)
        return []

    records: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    _CACHED_RECORDS = records
    logger.info("Loaded %d placement records from %s", len(records), p)
    return records


def parse_query_branch(query: str) -> Optional[str]:
    """Identify a branch code mentioned in a natural language query."""
    q = query.lower()
    # Check multi-word patterns first
    if "computer science and design" in q or "csd" in q:
        return "CD"
    if "computer science" in q or "cse" in q:
        return "CS"
    if "information science" in q or re.search(r"\bise\b", q) or re.search(r"\bis\s+(?:branch|dept|department)\b", q):
        return "ISE"
    if "artificial intelligence" in q or "aiml" in q or "ai/ml" in q or "ai & ml" in q:
        return "AIML"
    if "cyber security" in q or "cybersecurity" in q or re.search(r"\bcy\b", q) or re.search(r"\bcys\b", q):
        return "CY"
    if re.search(r"\bcd\b", q):
        return "CD"
    if re.search(r"\bcs\b", q) or re.search(r"\bcse\b", q):
        return "CS"
    return None


def parse_query_branches_for_comparison(query: str) -> List[str]:
    """Identify two or more branches to compare from query."""
    q = query.lower()
    found: List[str] = []
    checks = [
        ("CS", [r"\bcse\b", r"\bcs\b", r"computer science(?! and design)"]),
        ("ISE", [r"\bise\b", r"information science", r"\bis\s+branch\b"]),
        ("AIML", [r"\baiml\b", r"ai/ml", r"ai & ml", r"artificial intelligence"]),
        ("CD", [r"\bcd\b", r"\bcsd\b", r"computer science and design"]),
        ("CY", [r"\bcy\b", r"\bcys\b", r"cyber security", r"cybersecurity"]),
    ]
    for code, patterns in checks:
        for pat in patterns:
            if re.search(pat, q):
                if code not in found:
                    found.append(code)
                break
    return found


def compute_placement_metrics(
    records: Optional[List[Dict[str, Any]]] = None,
    branch: Optional[str] = None,
    company: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute deterministic metrics across all placement records or filtered subset."""
    if records is None:
        records = load_placement_records()

    filtered = records
    if branch:
        canonical_b = BRANCH_SYNONYMS.get(branch.lower(), branch.upper())
        filtered = [r for r in filtered if r.get("branch") == canonical_b]

    if company:
        comp_lower = company.strip().lower()
        filtered = [r for r in filtered if comp_lower in (r.get("company_name") or "").lower()]

    if status:
        norm_s = "final" if "final" in status.lower() and "non" not in status.lower() else "non-final"
        filtered = [r for r in filtered if r.get("offer_status") == norm_s]

    total_offers = len(filtered)
    if total_offers == 0:
        return {
            "total_offers": 0,
            "final_offers": 0,
            "non_final_offers": 0,
            "highest_ctc": 0.0,
            "average_ctc": 0.0,
            "median_ctc": 0.0,
            "lowest_ctc": 0.0,
            "companies_count": 0,
            "companies": [],
            "branch_breakdown": {},
        }

    final_count = sum(1 for r in filtered if r.get("offer_status") == "final")
    non_final_count = total_offers - final_count

    ctc_list = [float(r["ctc"]) for r in filtered if r.get("ctc") is not None]
    if ctc_list:
        highest_ctc = round(max(ctc_list), 2)
        lowest_ctc = round(min(ctc_list), 2)
        avg_ctc = round(sum(ctc_list) / len(ctc_list), 2)
        sorted_ctc = sorted(ctc_list)
        n = len(sorted_ctc)
        median_ctc = round(
            (sorted_ctc[n // 2] if n % 2 != 0 else (sorted_ctc[n // 2 - 1] + sorted_ctc[n // 2]) / 2.0),
            2,
        )
    else:
        highest_ctc = lowest_ctc = avg_ctc = median_ctc = 0.0

    companies = sorted(list({r.get("company_name") for r in filtered if r.get("company_name")}))

    # Highest CTC offer details
    highest_offers = [r for r in filtered if r.get("ctc") == highest_ctc]

    # Branch breakdown
    branch_map: Dict[str, Dict[str, Any]] = {}
    for r in filtered:
        b = r.get("branch") or "Unknown"
        if b not in branch_map:
            branch_map[b] = {
                "branch_name": r.get("branch_name", b),
                "offers": 0,
                "final": 0,
                "non_final": 0,
                "ctcs": [],
            }
        branch_map[b]["offers"] += 1
        if r.get("offer_status") == "final":
            branch_map[b]["final"] += 1
        else:
            branch_map[b]["non_final"] += 1
        if r.get("ctc") is not None:
            branch_map[b]["ctcs"].append(float(r["ctc"]))

    branch_breakdown = {}
    for b, data in branch_map.items():
        b_ctcs = data["ctcs"]
        b_avg = round(sum(b_ctcs) / len(b_ctcs), 2) if b_ctcs else 0.0
        b_max = round(max(b_ctcs), 2) if b_ctcs else 0.0
        branch_breakdown[b] = {
            "branch_name": data["branch_name"],
            "total_offers": data["offers"],
            "final_offers": data["final"],
            "non_final_offers": data["non_final"],
            "average_ctc": b_avg,
            "highest_ctc": b_max,
        }

    return {
        "total_offers": total_offers,
        "final_offers": final_count,
        "non_final_offers": non_final_count,
        "highest_ctc": highest_ctc,
        "highest_offers": highest_offers,
        "average_ctc": avg_ctc,
        "median_ctc": median_ctc,
        "lowest_ctc": lowest_ctc,
        "companies_count": len(companies),
        "companies": companies,
        "branch_breakdown": branch_breakdown,
        "disclaimer": DISCLAIMER_TEXT,
    }


def get_top_recruiters(limit: int = 10, records: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Return top recruiting companies ranked by number of offers."""
    if records is None:
        records = load_placement_records()
    comp_offers: Dict[str, Dict[str, Any]] = {}
    for r in records:
        c = r.get("company_name") or "Unknown"
        if c not in comp_offers:
            comp_offers[c] = {"company": c, "offers": 0, "final": 0, "highest_ctc": 0.0, "branches": set()}
        comp_offers[c]["offers"] += 1
        if r.get("offer_status") == "final":
            comp_offers[c]["final"] += 1
        if r.get("ctc") and float(r["ctc"]) > comp_offers[c]["highest_ctc"]:
            comp_offers[c]["highest_ctc"] = float(r["ctc"])
        if r.get("branch"):
            comp_offers[c]["branches"].add(r.get("branch"))

    ranked = sorted(comp_offers.values(), key=lambda x: (x["offers"], x["highest_ctc"]), reverse=True)
    out = []
    for item in ranked[:limit]:
        item_copy = dict(item)
        item_copy["branches"] = sorted(list(item["branches"]))
        out.append(item_copy)
    return out


def is_placement_query(query: str) -> bool:
    """Check if query is asking about placement records, packages, offers, or recruiters."""
    q = query.lower()
    placement_terms = [
        "placement",
        "placements",
        "package",
        "ctc",
        "salary",
        "lpa",
        "highest package",
        "average package",
        "avg package",
        "highest ctc",
        "average ctc",
        "lowest ctc",
        "recruiter",
        "recruiters",
        "offer",
        "offers",
        "offered",
        "offering",
        "hiring",
        "hired",
        "internship",
        "companies visited",
        "ppo",
        "2027 placement",
        "2027 batch",
    ]
    return any(term in q for term in placement_terms)


def answer_quantitative_placement_query(
    query: str,
    records: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Answer placement questions deterministically with computed metrics and official disclaimer.

    Returns None if the query is not a placement quantitative query.
    """
    if not query or not query.strip():
        return None

    q = query.lower()
    if not is_placement_query(q):
        return None

    if records is None:
        records = load_placement_records()

    if not records:
        return None

    # Detect branch comparison: "compare CS and ISE", "CS vs AIML"
    compare_branches = parse_query_branches_for_comparison(q)
    if len(compare_branches) >= 2 and any(w in q for w in ["compare", "vs", "versus", "difference"]):
        b1, b2 = compare_branches[0], compare_branches[1]
        m1 = compute_placement_metrics(records, branch=b1)
        m2 = compute_placement_metrics(records, branch=b2)
        answer = (
            f"### RVCE 2027 Placement Comparison: {b1} vs {b2}\n\n"
            f"- **{b1} ({m1['branch_breakdown'].get(b1, {}).get('branch_name', b1)})**:\n"
            f"  - Total Offers: **{m1['total_offers']}** ({m1['final_offers']} final, {m1['non_final_offers']} non-final)\n"
            f"  - Average CTC: **{m1['average_ctc']} LPA**\n"
            f"  - Highest CTC: **{m1['highest_ctc']} LPA**\n\n"
            f"- **{b2} ({m2['branch_breakdown'].get(b2, {}).get('branch_name', b2)})**:\n"
            f"  - Total Offers: **{m2['total_offers']}** ({m2['final_offers']} final, {m2['non_final_offers']} non-final)\n"
            f"  - Average CTC: **{m2['average_ctc']} LPA**\n"
            f"  - Highest CTC: **{m2['highest_ctc']} LPA**\n\n"
            f"> *Note: {DISCLAIMER_TEXT}*"
        )
        return _build_response_payload(answer, query)

    # Detect single branch query
    branch = parse_query_branch(q)

    # 1. Highest package query
    if (
        re.search(r"\b(highest|max|maximum|top)\b.*?\b(package|ctc|salary|offer)\b", q)
        or any(k in q for k in ["highest package", "highest ctc", "maximum package", "max package", "top package", "highest salary"])
    ):
        metrics = compute_placement_metrics(records, branch=branch)
        top_offers = metrics.get("highest_offers", [])
        if not top_offers:
            return None
        ctc = metrics["highest_ctc"]
        comps = sorted(list({r.get("company_name") for r in top_offers if r.get("company_name")}))
        brs = sorted(list({r.get("branch") for r in top_offers if r.get("branch")}))
        comp_str = ", ".join(comps)
        br_str = ", ".join(brs)

        target_context = f"for **{branch}** branch" if branch else "for the **2027 batch**"
        first = top_offers[0]
        answer = (
            f"The **highest CTC package** {target_context} is **{ctc} LPA**, "
            f"offered by **{comp_str}** ({br_str}).\n\n"
            f"**Offer Details:**\n"
            f"- Company: **{comp_str}**\n"
            f"- Package: **{ctc} LPA**\n"
            f"- Eligible Branch(es): **{br_str}**\n"
            f"- Category: **{first.get('offer_category')}**\n"
            f"- Offer Type: **{first.get('offer_type')}**\n"
            f"- Location: **{first.get('location_or_city')}**\n\n"
            f"> *Note: {DISCLAIMER_TEXT}*"
        )
        return _build_response_payload(answer, query)

    # 2. Average package query
    if (
        re.search(r"\b(average|avg|mean)\b.*?\b(package|ctc|salary|offer)\b", q)
        or any(k in q for k in ["average package", "avg package", "average ctc", "avg ctc", "mean package", "average salary"])
    ):
        metrics = compute_placement_metrics(records, branch=branch)
        avg = metrics["average_ctc"]
        med = metrics["median_ctc"]
        total = metrics["total_offers"]
        target_context = f"for **{branch}** branch" if branch else "across all branches for the **2027 batch**"
        answer = (
            f"The **average CTC package** {target_context} is **{avg} LPA** "
            f"(median: **{med} LPA** across **{total}** recorded offers).\n\n"
        )
        if not branch and metrics["branch_breakdown"]:
            answer += "**Branch-Wise Average CTC:**\n"
            for b_code, b_data in sorted(metrics["branch_breakdown"].items()):
                answer += f"- **{b_code}**: {b_data['average_ctc']} LPA ({b_data['total_offers']} offers)\n"
            answer += "\n"
        answer += f"> *Note: {DISCLAIMER_TEXT}*"
        return _build_response_payload(answer, query)

    # 3. Final vs Non-Final offers query
    if (
        ("final" in q and "non" in q)
        or any(k in q for k in ["final vs non-final", "final offers", "non-final offers", "how many final", "final count"])
    ):
        metrics = compute_placement_metrics(records, branch=branch)
        target_context = f"for **{branch}**" if branch else "for the **2027 batch**"
        answer = (
            f"### RVCE 2027 Placement Offer Status {target_context}\n\n"
            f"- **Final Offers:** **{metrics['final_offers']}**\n"
            f"- **Non-Final Offers:** **{metrics['non_final_offers']}**\n"
            f"- **Total Recorded Offers:** **{metrics['total_offers']}**\n\n"
            f"> *Note: {DISCLAIMER_TEXT}*"
        )
        return _build_response_payload(answer, query)

    # 4. Total offers / count query
    if (
        ("how many" in q and "offer" in q)
        or any(k in q for k in ["total offers", "number of offers", "offer count", "placement statistics", "placement record", "placement overview"])
    ):
        metrics = compute_placement_metrics(records, branch=branch)
        target_context = f"for **{branch}**" if branch else "for the **2027 batch**"
        answer = (
            f"### RVCE 2027 Placement Overview {target_context}\n\n"
            f"- **Total Offers:** **{metrics['total_offers']}**\n"
            f"- **Final Offers:** **{metrics['final_offers']}**\n"
            f"- **Non-Final Offers:** **{metrics['non_final_offers']}**\n"
            f"- **Unique Companies:** **{metrics['companies_count']}**\n"
            f"- **Highest CTC:** **{metrics['highest_ctc']} LPA**\n"
            f"- **Average CTC:** **{metrics['average_ctc']} LPA**\n\n"
        )
        if not branch and metrics["branch_breakdown"]:
            answer += "**Branch Breakdown:**\n"
            for b_code, b_data in sorted(metrics["branch_breakdown"].items()):
                answer += (
                    f"- **{b_code}** ({b_data['branch_name']}): **{b_data['total_offers']}** offers "
                    f"(Avg: {b_data['average_ctc']} LPA, Max: {b_data['highest_ctc']} LPA)\n"
                )
            answer += "\n"
        answer += f"> *Note: {DISCLAIMER_TEXT}*"
        return _build_response_payload(answer, query)

    # 5. Salary filter query (e.g. "companies offering more than 30 LPA")
    match_above = re.search(r"(?:above|more than|greater than|>|at least)\s*(\d+(?:\.\d+)?)\s*(?:lpa|lakh|lakhs)?", q)
    if match_above and any(k in q for k in ["company", "companies", "package", "ctc", "offer", "recruit"]):
        threshold = float(match_above.group(1))
        matching_offers = [r for r in records if r.get("ctc") and float(r["ctc"]) >= threshold]
        comp_summary: Dict[str, Dict[str, Any]] = {}
        for r in matching_offers:
            c = r["company_name"]
            if c not in comp_summary:
                comp_summary[c] = {"company": c, "max_ctc": float(r["ctc"]), "offers": 0, "branches": set()}
            comp_summary[c]["offers"] += 1
            if float(r["ctc"]) > comp_summary[c]["max_ctc"]:
                comp_summary[c]["max_ctc"] = float(r["ctc"])
            comp_summary[c]["branches"].add(r["branch"])

        sorted_comps = sorted(comp_summary.values(), key=lambda x: (x["max_ctc"], x["offers"]), reverse=True)
        answer = (
            f"### Companies Offering ≥ {threshold} LPA (RVCE 2027 Placements)\n\n"
            f"Found **{len(sorted_comps)}** companies with offers at or above **{threshold} LPA** "
            f"({len(matching_offers)} total offers):\n\n"
        )
        for i, item in enumerate(sorted_comps[:15], 1):
            br_str = ", ".join(sorted(list(item["branches"])))
            answer += f"{i}. **{item['company']}**: Up to **{item['max_ctc']} LPA** ({item['offers']} offers, Branches: {br_str})\n"
        if len(sorted_comps) > 15:
            answer += f"\n*...and {len(sorted_comps) - 15} more companies.*\n"
        answer += f"\n> *Note: {DISCLAIMER_TEXT}*"
        return _build_response_payload(answer, query)

    # 6. Lowest package query
    if any(k in q for k in ["lowest package", "lowest ctc", "minimum package", "min package", "lowest salary"]):
        metrics = compute_placement_metrics(records, branch=branch)
        target_context = f"for **{branch}** branch" if branch else "for the **2027 batch**"
        answer = (
            f"The **minimum recorded CTC package** {target_context} is **{metrics['lowest_ctc']} LPA** "
            f"(with an average of **{metrics['average_ctc']} LPA** across **{metrics['total_offers']}** offers).\n\n"
            f"> *Note: {DISCLAIMER_TEXT}*"
        )
        return _build_response_payload(answer, query)

    # 7. Top companies / recruiters query
    if any(k in q for k in ["top recruiters", "top companies", "major recruiters", "companies visited", "who visited", "list of companies"]):
        top_list = get_top_recruiters(limit=10, records=records)
        answer = (
            "### Top Recruiters for RVCE 2027 Batch (by Offers)\n\n"
        )
        for i, item in enumerate(top_list, 1):
            branches_str = ", ".join(item["branches"])
            answer += (
                f"{i}. **{item['company']}**: **{item['offers']}** offers "
                f"(Max CTC: **{item['highest_ctc']} LPA**, Branches: {branches_str})\n"
            )
        answer += f"\n> *Note: {DISCLAIMER_TEXT}*"
        return _build_response_payload(answer, query)

    # 8. Specific company lookup in query (e.g. "Auxia", "Cisco", "HPE", "Mercedes-Benz")
    unique_companies = {r["company_name"] for r in records if r.get("company_name")}
    matched_company = None
    for c in unique_companies:
        if re.search(rf"\b{re.escape(c.lower())}\b", q):
            matched_company = c
            break

    if matched_company:
        comp_records = [r for r in records if r.get("company_name") == matched_company]
        total_co = len(comp_records)
        final_co = sum(1 for r in comp_records if r.get("offer_status") == "final")
        co_ctcs = [float(r["ctc"]) for r in comp_records if r.get("ctc") is not None]
        co_max = max(co_ctcs) if co_ctcs else "N/A"
        branches_str = ", ".join(sorted(list({r.get("branch") for r in comp_records if r.get("branch")})))
        types_str = ", ".join(sorted(list({r.get("offer_type") for r in comp_records if r.get("offer_type")})))
        locs_str = ", ".join(sorted(list({r.get("location_or_city") for r in comp_records if r.get("location_or_city")})))

        answer = (
            f"### 2027 Placement Record: {matched_company}\n\n"
            f"- **Total Offers:** **{total_co}** ({final_co} final, {total_co - final_co} non-final)\n"
            f"- **Package (CTC):** **{co_max} LPA**\n"
            f"- **Eligible Branches:** **{branches_str}**\n"
            f"- **Offer Type:** **{types_str}**\n"
            f"- **Location:** **{locs_str}**\n\n"
            f"> *Note: {DISCLAIMER_TEXT}*"
        )
        return _build_response_payload(answer, query)

    return None


def _build_response_payload(answer: str, query: str) -> Dict[str, Any]:
    """Format structured answer dictionary adhering to RAG answer schema."""
    evidence_chunk = {
        "chunk_id": "chunk_placement_2027_analytics",
        "document_id": "doc_rvce_placements_2027",
        "text": answer,
        "document_title": "RVCE 2027 Placement Statistics (Unofficial)",
        "source_title": "RVCE 2027 Placement Records",
        "source_url": DEFAULT_SOURCE_URL,
        "heading": "Placement Statistics 2027",
        "topic": "placements_careers",
        "relevance_score": 0.98,
        "disclaimer": DISCLAIMER_TEXT,
    }
    source_item = {
        "title": "RVCE 2027 Placement Statistics (Unofficial)",
        "url": DEFAULT_SOURCE_URL,
    }
    intent_dict = {
        "intent": "placements_careers",
        "normalized_intent": "placements_careers",
        "raw_intent": "placements_careers",
        "intent_label": "Placements & Careers",
        "confidence": 0.99,
        "is_confident": True,
        "routing": "placement_analytics_engine",
        "model": "deterministic_placement_calculator",
        "entities": {"main_entity": "placement"},
        "retrieval_expansion": "RVCE 2027 placement statistics CTC offers recruiters",
    }
    return {
        "answer": answer,
        "evidence": [evidence_chunk],
        "sources": [source_item],
        "intent": intent_dict,
        "strategy": "structured_placement_analytics",
        "is_placement_analytic": True,
        "disclaimer": DISCLAIMER_TEXT,
    }

def search_placement_records(
    query: str,
    top_k: int = 5,
    records: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Lexical and entity search over placement records returning standard evidence dicts."""
    if not query or not query.strip():
        return []
    if records is None:
        records = load_placement_records()
    if not records:
        return []

    q_lower = query.lower()
    q_tokens = set(re.findall(r"\w+", q_lower))
    # Remove generic stopwords
    stopwords = {"what", "is", "the", "for", "in", "at", "rvce", "batch", "show", "tell", "give", "me", "details", "about"}
    meaningful = q_tokens - stopwords

    scored_candidates = []
    for r in records:
        comp = (r.get("company_name") or "").lower()
        br = (r.get("branch") or "").lower()
        cat = (r.get("offer_category") or "").lower()
        off_type = (r.get("offer_type") or "").lower()
        loc = (r.get("location_or_city") or "").lower()
        rec_text = f"{comp} {br} {cat} {off_type} {loc}".lower()

        score = 0.0
        # Exact company match gives large boost
        if comp and comp in q_lower:
            score += 0.5
        # Branch match
        if br and (f" {br} " in f" {q_lower} " or br in q_lower):
            score += 0.2
        # Token overlap
        overlap = sum(1 for t in meaningful if t in rec_text)
        score += 0.1 * overlap

        if score > 0.15:
            text = (
                f"RVCE 2027 Placement Offer: Company: {r.get('company_name')} | "
                f"Branch: {r.get('branch')} ({r.get('branch_name')}) | "
                f"Status: {r.get('offer_status')} | "
                f"Package: {r.get('ctc_raw') or (str(r.get('ctc_lpa')) + ' LPA')} | "
                f"Base Pay: {r.get('base_lpa')} LPA | "
                f"Stipend: {r.get('stipend_kpm')} K/month | "
                f"Category: {r.get('offer_category')} | "
                f"Type: {r.get('offer_type')} | "
                f"Location: {r.get('location_or_city')}. "
                f"{r.get('disclaimer')}"
            )
            chunk_dict = {
                "chunk_id": f"chunk_placement_{r.get('id')}",
                "document_id": f"doc_placement_{r.get('id')}",
                "text": text,
                "document_title": f"RVCE 2027 Placement Offer - {r.get('company_name')}",
                "source_title": "RVCE 2027 Placement Records (Unofficial)",
                "source_url": r.get("source_url") or DEFAULT_SOURCE_URL,
                "heading": f"{r.get('company_name')} ({r.get('branch')})",
                "topic": "placements_careers",
                "relevance_score": round(min(0.95, score), 4),
                "disclaimer": r.get("disclaimer") or DISCLAIMER_TEXT,
            }
            scored_candidates.append((score, chunk_dict))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in scored_candidates[:top_k]]
