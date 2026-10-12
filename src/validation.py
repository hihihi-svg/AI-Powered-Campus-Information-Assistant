"""
validation.py - Comprehensive Quality Assurance and validation for campus datasets.

Performs:
- Schema validation for processed records
- Exact duplicate detection (URL and content hash)
- Near-duplicate content detection (Jaccard similarity on word 3-grams)
- Extraction status flagging (SUCCESS, EMPTY, FAILED)
- Aggregation of crawl errors and unsupported content types
- Generation of the data collection QA report (reports/data_collection_report.json)
"""

import hashlib
import json
import logging
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

logger = logging.getLogger("campus_pipeline.validation")

REQUIRED_SCHEMA_FIELDS = {
    "document_id": str,
    "source_url": str,
    "title": (str, type(None)),
    "content": str,
    "source_type": str,
    "collected_at": str,
    "last_modified": (str, type(None)),
    "topic": (str, type(None)),
    "extraction_status": str,
}


def validate_url(url: str) -> bool:
    """Validate that a URL has valid scheme and network location."""
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url)
        return bool(parsed.scheme in ("http", "https") and parsed.netloc)
    except Exception:
        return False


def validate_record_schema(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate processed record against expected schema and types."""
    errors = []
    for field, expected_type in REQUIRED_SCHEMA_FIELDS.items():
        if field not in record:
            errors.append(f"Missing required field: '{field}'")
            continue
        val = record[field]
        if not isinstance(val, expected_type):
            errors.append(f"Field '{field}' expected {expected_type}, got {type(val).__name__}")

    if not validate_url(record.get("source_url", "")):
        errors.append(f"Invalid URL: '{record.get('source_url')}'")

    status = record.get("extraction_status")
    if status not in ("SUCCESS", "PARTIAL", "EMPTY", "FAILED"):
        errors.append(f"Invalid extraction_status: '{status}'")

    return len(errors) == 0, errors


def compute_shingles(text: str, n: int = 3) -> Set[str]:
    """Generate n-gram word shingles for near-duplicate Jaccard similarity."""
    tokens = [w.lower() for w in text.split() if w.isalnum() or len(w) > 1]
    if len(tokens) < n:
        return set(tokens)
    return set(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity coefficient between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


class QualityAssuranceEngine:
    """QA suite for dataset integrity, duplicate detection, and reporting."""

    def __init__(self, near_duplicate_threshold: float = 0.85):
        self.near_duplicate_threshold = near_duplicate_threshold

    def analyze_dataset(
        self,
        records: List[Dict[str, Any]],
        crawl_errors: Optional[List[Dict[str, Any]]] = None,
        config_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes QA analysis over the processed records collection.
        Returns a detailed report dictionary.
        """
        crawl_errors = crawl_errors or []
        total_records = len(records)

        schema_valid_count = 0
        validation_issues: List[Dict[str, Any]] = []

        seen_urls: Set[str] = set()
        url_duplicates: List[str] = []

        content_hashes: Dict[str, str] = {}  # hash -> first doc_id
        exact_content_duplicates: List[Dict[str, str]] = []

        record_shingles: List[Tuple[str, str, Set[str]]] = []  # (doc_id, url, shingles)
        near_duplicates: List[Dict[str, Any]] = []

        type_counts: Dict[str, int] = {}
        status_counts: Dict[str, int] = {}
        topic_counts: Dict[str, int] = {}
        content_lengths: List[int] = []
        word_counts: List[int] = []

        for rec in records:
            # 1. Schema check
            is_valid, errors = validate_record_schema(rec)
            if is_valid:
                schema_valid_count += 1
            else:
                validation_issues.append({"document_id": rec.get("document_id"), "errors": errors})

            # 2. URL duplicates
            url = rec.get("source_url")
            if url in seen_urls:
                url_duplicates.append(url)
            else:
                seen_urls.add(url)

            # 3. Exact content duplicates
            content = rec.get("content", "")
            content_len = len(content)
            content_lengths.append(content_len)
            word_count = len(content.split())
            word_counts.append(word_count)

            if content_len > 0:
                chash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                if chash in content_hashes:
                    exact_content_duplicates.append({
                        "duplicate_id": rec.get("document_id"),
                        "original_id": content_hashes[chash],
                        "source_url": url,
                    })
                else:
                    content_hashes[chash] = rec.get("document_id")

                # Shingles for near-duplicate analysis (only analyze substantive docs)
                if content_len >= 100:
                    shingles = compute_shingles(content, n=3)
                    record_shingles.append((rec.get("document_id"), url, shingles))

            # 4. Distributions
            stype = rec.get("source_type", "UNKNOWN")
            type_counts[stype] = type_counts.get(stype, 0) + 1

            status = rec.get("extraction_status", "UNKNOWN")
            status_counts[status] = status_counts.get(status, 0) + 1

            topic = rec.get("topic") or "unclassified"
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        # 5. Near-duplicate pairwise comparison (optimized)
        for i in range(len(record_shingles)):
            id_a, url_a, shingles_a = record_shingles[i]
            for j in range(i + 1, min(i + 30, len(record_shingles))):  # check neighborhood window
                id_b, url_b, shingles_b = record_shingles[j]
                sim = jaccard_similarity(shingles_a, shingles_b)
                if sim >= self.near_duplicate_threshold:
                    near_duplicates.append({
                        "doc_id_1": id_a,
                        "doc_id_2": id_b,
                        "url_1": url_a,
                        "url_2": url_b,
                        "jaccard_similarity": round(sim, 4),
                    })

        # Length statistics
        len_stats = {
            "min_chars": min(content_lengths) if content_lengths else 0,
            "max_chars": max(content_lengths) if content_lengths else 0,
            "mean_chars": round(statistics.mean(content_lengths), 2) if content_lengths else 0,
            "median_chars": round(statistics.median(content_lengths), 2) if content_lengths else 0,
            "min_words": min(word_counts) if word_counts else 0,
            "max_words": max(word_counts) if word_counts else 0,
            "mean_words": round(statistics.mean(word_counts), 2) if word_counts else 0,
        }

        config_snapshot = config_snapshot or {}
        report = {
            "metadata": {
                "report_timestamp": datetime.now(timezone.utc).isoformat(),
                "pipeline_version": "1.0.0",
                "target_domain": config_snapshot.get("crawler", {}).get("allowed_domains", ["rvce.edu.in"]),
            },
            "summary": {
                "total_documents_collected": total_records,
                "schema_valid_records": schema_valid_count,
                "validation_issues_count": len(validation_issues),
                "successful_extractions": status_counts.get("SUCCESS", 0),
                "empty_extractions": status_counts.get("EMPTY", 0),
                "partial_extractions": status_counts.get("PARTIAL", 0),
                "failed_extractions": status_counts.get("FAILED", 0),
                "crawl_errors_count": len(crawl_errors),
            },
            "source_coverage": {
                "by_type": type_counts,
                "by_topic": topic_counts,
            },
            "duplicate_analysis": {
                "exact_url_duplicates_count": len(url_duplicates),
                "exact_content_duplicates_count": len(exact_content_duplicates),
                "exact_content_duplicates": exact_content_duplicates[:20],
                "near_duplicates_count": len(near_duplicates),
                "near_duplicates_threshold": self.near_duplicate_threshold,
                "near_duplicates_samples": near_duplicates[:20],
            },
            "content_length_statistics": len_stats,
            "crawl_errors": crawl_errors[:50],
            "validation_issues": validation_issues[:20],
        }

        return report

    def save_report(self, report: Dict[str, Any], filepath: str) -> None:
        """Write JSON report to disk with clean indentation."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Quality Assurance Report successfully saved to {filepath}")
