"""
build_dataset.py - End-to-end pipeline runner for Campus Knowledge Acquisition.

Coordinates:
1. Website crawling (robots.txt, polite delay, priority frontier)
2. Raw data persistence & manifest provenance tracking (data/raw/campus_documents.jsonl)
3. Structured extraction & boilerplate removal (HTML/PDF)
4. Cleaning, normalization, deterministic ID generation, and verifiable topic assignment
5. Cleaned dataset persistence (data/processed/campus_documents.jsonl)
6. Quality Assurance, duplicate detection, and reporting (reports/data_collection_report.json)
"""

import argparse
import base64
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import yaml

from src.cleaning import clean_and_process_record
from src.crawler import CampusCrawler
from src.extractors import extract_record
from src.validation import QualityAssuranceEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("campus_pipeline.build_dataset")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load pipeline YAML configuration."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_raw_record(record: Dict[str, Any], filepath: str) -> None:
    """
    Append raw crawled record to JSONL.
    PDF binary content is safely base64-encoded to preserve exact byte fidelity.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    serializable = dict(record)
    if serializable.get("raw_bytes"):
        serializable["raw_bytes_base64"] = base64.b64encode(serializable["raw_bytes"]).decode("ascii")
        del serializable["raw_bytes"]

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(serializable, ensure_ascii=False) + "\n")


def load_raw_records(filepath: str) -> List[Dict[str, Any]]:
    """Load raw records from JSONL restoring binary bytes where applicable."""
    records = []
    if not os.path.exists(filepath):
        return records
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "raw_bytes_base64" in rec:
                rec["raw_bytes"] = base64.b64decode(rec["raw_bytes_base64"])
            records.append(rec)
    return records


def update_provenance_manifest(
    manifest_path: str,
    raw_records: List[Dict[str, Any]],
    processed_records: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> None:
    """
    Maintains a dataset manifest linking every processed record to its original source URL,
    timestamps, hashes, and collection parameters for complete reproducibility.
    """
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    manifest_entries = {}
    for p_rec in processed_records:
        manifest_entries[p_rec["document_id"]] = {
            "document_id": p_rec["document_id"],
            "source_url": p_rec["source_url"],
            "title": p_rec["title"],
            "source_type": p_rec["source_type"],
            "collected_at": p_rec["collected_at"],
            "last_modified": p_rec["last_modified"],
            "topic": p_rec["topic"],
            "extraction_status": p_rec["extraction_status"],
            "content_length_chars": len(p_rec.get("content", "")),
        }

    manifest = {
        "manifest_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_documents": len(manifest_entries),
        "crawler_config": config.get("crawler", {}),
        "records": manifest_entries,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"Provenance manifest updated at {manifest_path} ({len(manifest_entries)} entries)")


def run_pipeline(config_path: str = "config.yaml", max_pages_override: Optional[int] = None) -> Dict[str, Any]:
    """Execute end-to-end campus knowledge acquisition pipeline."""
    config = load_config(config_path)

    if max_pages_override is not None:
        config["crawler"]["max_pages"] = max_pages_override

    storage_cfg = config.get("storage", {})
    raw_path = storage_cfg.get("raw_data_path", "data/raw/campus_documents.jsonl")
    processed_path = storage_cfg.get("processed_data_path", "data/processed/campus_documents.jsonl")
    manifest_path = storage_cfg.get("manifest_path", "data/raw/collection_manifest.json")
    report_path = storage_cfg.get("report_path", "reports/data_collection_report.json")

    clean_cfg = config.get("cleaning", {})
    min_length = clean_cfg.get("min_content_length", 50)
    near_dup_thresh = clean_cfg.get("near_duplicate_similarity_threshold", 0.85)

    # 1. Crawl website
    logger.info("=================================================================")
    logger.info("Phase 1: Crawling Campus Website (rvce.edu.in)")
    logger.info(f"Seed URLs: {config['crawler']['start_urls']}")
    logger.info(f"Max Pages: {config['crawler']['max_pages']}, Max PDFs: {config['crawler']['max_pdfs']}")
    logger.info("=================================================================")

    crawler = CampusCrawler(config)
    collected_raw_records: List[Dict[str, Any]] = []

    # Reset raw file for this run if starting fresh snapshot
    if os.path.exists(raw_path):
        os.remove(raw_path)

    for item in crawler.crawl():
        collected_raw_records.append(item)
        save_raw_record(item, raw_path)

    logger.info(f"Raw collection complete: {len(collected_raw_records)} documents saved to {raw_path}")

    # 2. Extract and clean records
    logger.info("=================================================================")
    logger.info("Phase 2: Extracting Text, Removing Boilerplate & Cleaning")
    logger.info("=================================================================")

    processed_records: List[Dict[str, Any]] = []

    for raw_rec in collected_raw_records:
        extracted = extract_record(raw_rec)
        cleaned = clean_and_process_record(extracted, min_length=min_length)
        processed_records.append(cleaned)

    # Save processed records
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    with open(processed_path, "w", encoding="utf-8") as f:
        for p_rec in processed_records:
            f.write(json.dumps(p_rec, ensure_ascii=False) + "\n")
    logger.info(f"Processed dataset saved to {processed_path} ({len(processed_records)} records)")

    # 3. Update manifest
    update_provenance_manifest(manifest_path, collected_raw_records, processed_records, config)

    # 4. Run Quality Assurance Engine
    logger.info("=================================================================")
    logger.info("Phase 3: Quality Assurance, Duplicate Detection & Reporting")
    logger.info("=================================================================")

    qa_engine = QualityAssuranceEngine(near_duplicate_threshold=near_dup_thresh)
    qa_report = qa_engine.analyze_dataset(
        records=processed_records,
        crawl_errors=crawler.errors,
        config_snapshot=config,
    )

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    qa_engine.save_report(qa_report, report_path)

    # 5. Print summary statistics
    summary = qa_report["summary"]
    coverage = qa_report["source_coverage"]
    duplicates = qa_report["duplicate_analysis"]
    lengths = qa_report["content_length_statistics"]

    print("\n" + "=" * 65)
    print("           CAMPUS KNOWLEDGE ACQUISITION SUMMARY")
    print("=" * 65)
    print(f"Total Documents Collected:    {summary['total_documents_collected']}")
    print(f"HTML Documents:               {coverage['by_type'].get('HTML', 0)}")
    print(f"PDF Documents:                {coverage['by_type'].get('PDF', 0)}")
    print(f"Successful Extractions:       {summary['successful_extractions']}")
    print(f"Empty Extractions:            {summary['empty_extractions']}")
    print(f"Failed Extractions:           {summary['failed_extractions']}")
    print(f"Crawl Errors Encountered:     {summary['crawl_errors_count']}")
    print(f"Exact Content Duplicates:     {duplicates['exact_content_duplicates_count']}")
    print(f"Near Duplicates (>= {near_dup_thresh}): {duplicates['near_duplicates_count']}")
    print(f"Content Length (mean/median): {lengths['mean_chars']} / {lengths['median_chars']} chars")
    print(f"Word Count (mean):            {lengths['mean_words']} words")
    print("\nTopic Breakdown:")
    for topic, count in sorted(coverage["by_topic"].items(), key=lambda x: -x[1]):
        print(f"  - {topic:28s}: {count:3d} docs")
    print("=" * 65)
    print(f"Raw Dataset:       {raw_path}")
    print(f"Processed Dataset: {processed_path}")
    print(f"Provenance Manifest: {manifest_path}")
    print(f"QA Report:         {report_path}")
    print("=" * 65 + "\n")

    return qa_report


def main():
    parser = argparse.ArgumentParser(description="Campus Knowledge Acquisition Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--max-pages", type=int, default=None, help="Override maximum pages to crawl")
    args = parser.parse_args()

    run_pipeline(config_path=args.config, max_pages_override=args.max_pages)


if __name__ == "__main__":
    main()
