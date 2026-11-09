"""
gmail_ingestion.py - End-to-end and incremental ingestion pipeline for authorized RVCE Gmail notices.

Features:
- Complete historical ingestion mode (--all-history) supporting emails from college joining date (2024)
- Incremental synchronization via state watermark (data/raw/gmail_ingestion_state.json)
- Corpus-aware deduplication using Gmail message IDs, document IDs, and attachment SHA-256 hashes
- Safe noise and personal thread filtering (src/gmail_filter.py)
- Reuses HTMLExtractor, PDFExtractor, and RapidOCR (src/gmail_parser.py)
- Appends new documents to data/raw/campus_documents.jsonl and data/processed/campus_documents.jsonl
- Provenance manifest updates (data/raw/collection_manifest.json)
- Atomic, fault-tolerant re-indexing of retrieval models (--reindex) without losing existing index state
- Background scheduled sync worker (GmailSyncWorker / --watch) that never blocks FastAPI requests
- Comprehensive reporting (reports/gmail_ingestion_report.json)
- Zero exposure of credentials or private tokens in logs
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from src.build_dataset import load_config
from src.gmail_client import (
    fetch_attachment_bytes,
    fetch_message_full,
    fetch_message_ids,
    get_gmail_service,
)
from src.gmail_filter import is_relevant_notice
from src.gmail_parser import parse_email_message

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("campus_pipeline.gmail_ingestion")

DEFAULT_STATE_PATH = "data/raw/gmail_ingestion_state.json"
DEFAULT_RAW_PATH = "data/raw/campus_documents.jsonl"
DEFAULT_PROCESSED_PATH = "data/processed/campus_documents.jsonl"
DEFAULT_MANIFEST_PATH = "data/raw/collection_manifest.json"
DEFAULT_REPORT_PATH = "reports/gmail_ingestion_report.json"
DEFAULT_COLLEGE_JOIN_DATE = "2024/01/01"


def load_ingestion_state(state_path: str = DEFAULT_STATE_PATH) -> Dict[str, Any]:
    """Load incremental synchronization state from disk."""
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data.setdefault("processed_message_ids", [])
                    data.setdefault("last_synced_internal_date_ms", 0)
                    data.setdefault("last_sync_timestamp", None)
                    data.setdefault("total_emails_ingested", 0)
                    data.setdefault("total_attachments_ingested", 0)
                    return data
        except Exception as e:
            logger.warning("Could not read state file %s: %s. Starting fresh.", state_path, e)
    return {
        "processed_message_ids": [],
        "last_synced_internal_date_ms": 0,
        "last_sync_timestamp": None,
        "total_emails_ingested": 0,
        "total_attachments_ingested": 0,
    }


def save_ingestion_state(state: Dict[str, Any], state_path: str = DEFAULT_STATE_PATH) -> None:
    """Save updated synchronization state to disk atomically."""
    os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
    tmp_path = f"{state_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, state_path)
    logger.info("Saved incremental ingestion state to %s", state_path)


def get_existing_corpus_ids(processed_path: str = DEFAULT_PROCESSED_PATH) -> Tuple[Set[str], Set[str]]:
    """Return (existing_document_ids, existing_gmail_message_ids) from the processed JSONL corpus."""
    doc_ids: Set[str] = set()
    msg_ids: Set[str] = set()
    if not os.path.exists(processed_path):
        return doc_ids, msg_ids

    try:
        with open(processed_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    did = rec.get("document_id")
                    if did:
                        doc_ids.add(str(did))
                    mid = rec.get("gmail_message_id")
                    if mid:
                        msg_ids.add(str(mid))
                except Exception:
                    continue
    except Exception as exc:
        logger.warning("Could not scan existing corpus IDs from %s: %s", processed_path, exc)
    return doc_ids, msg_ids


def save_raw_record(record: Dict[str, Any], filepath: str = DEFAULT_RAW_PATH) -> None:
    """Append raw record to raw JSONL dataset, encoding bytes as base64."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    serializable = dict(record)
    if serializable.get("raw_bytes"):
        serializable["raw_bytes_base64"] = base64.b64encode(serializable["raw_bytes"]).decode("ascii")
        del serializable["raw_bytes"]

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(serializable, ensure_ascii=False) + "\n")


def append_processed_record(
    record: Dict[str, Any],
    filepath: str = DEFAULT_PROCESSED_PATH,
    existing_doc_ids: Optional[Set[str]] = None,
) -> bool:
    """Append cleaned, schema-compliant record to processed JSONL dataset if not already present.

    Returns True if the record was appended, False if skipped as duplicate.
    """
    doc_id = str(record.get("document_id") or "")
    if existing_doc_ids is not None and doc_id and doc_id in existing_doc_ids:
        logger.debug("Skipping duplicate document_id %s in %s", doc_id, filepath)
        return False

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    clean_rec = dict(record)
    clean_rec.pop("raw_bytes", None)

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(clean_rec, ensure_ascii=False) + "\n")

    if existing_doc_ids is not None and doc_id:
        existing_doc_ids.add(doc_id)
    return True


def update_manifest_entry(record: Dict[str, Any], manifest_path: str = DEFAULT_MANIFEST_PATH) -> None:
    """Add or update an entry in the provenance manifest."""
    manifest: Dict[str, Any] = {"manifest_version": "1.0.0", "records": {}}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            pass

    records = manifest.setdefault("records", {})
    records[record["document_id"]] = {
        "document_id": record["document_id"],
        "source_url": record["source_url"],
        "title": record.get("title"),
        "source_type": record["source_type"],
        "collected_at": record.get("collected_at"),
        "last_modified": record.get("last_modified"),
        "topic": record.get("topic"),
        "extraction_status": record.get("extraction_status"),
        "content_length_chars": len(record.get("content", "")),
    }
    manifest["total_documents"] = len(records)
    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()

    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def get_corpus_statistics(processed_path: str = DEFAULT_PROCESSED_PATH) -> Dict[str, int]:
    """Calculate document counts across source categories in the processed dataset."""
    counts = {
        "total_documents": 0,
        "email_documents": 0,
        "email_attachment_documents": 0,
        "web_documents": 0,
    }
    if not os.path.exists(processed_path):
        return counts

    with open(processed_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            counts["total_documents"] += 1
            try:
                rec = json.loads(line)
                st = rec.get("source_type")
                did = rec.get("document_id", "")
                if st == "EMAIL_ATTACHMENT" or did.startswith("doc_email_att_"):
                    counts["email_attachment_documents"] += 1
                elif st == "EMAIL" or did.startswith("doc_email_"):
                    counts["email_documents"] += 1
                else:
                    counts["web_documents"] += 1
            except Exception:
                pass
    return counts


def update_retrieval_index(
    processed_path: str = DEFAULT_PROCESSED_PATH,
    include_semantic: bool = True,
) -> Dict[str, Any]:
    """Atomically rebuild and update in-memory retrieval indexes from the processed corpus.

    Builds the new TF-IDF (and optionally Semantic/Hybrid) index BEFORE replacing
    the cached retriever instances so that previously indexed documents are never lost
    if an error occurs during re-indexing.
    """
    from src.retrieval import (
        _DEFAULT_HYBRID_RETRIEVER_CACHE,
        _DEFAULT_RETRIEVER_CACHE,
        _DEFAULT_SEMANTIC_RETRIEVER_CACHE,
        TFIDFRetriever,
        get_default_hybrid_retriever,
        get_default_semantic_retriever,
    )

    if not os.path.exists(processed_path):
        logger.warning("Cannot update retrieval index: %s does not exist.", processed_path)
        return {"updated": False, "reason": "dataset_not_found", "chunks_indexed": 0}

    try:
        # 1. Build new TF-IDF index in a staging instance first
        new_tfidf = TFIDFRetriever()
        chunk_count = new_tfidf.load_and_index(processed_path)
        if chunk_count <= 0:
            logger.warning("Staging index produced 0 chunks; preserving existing index cache.")
            return {"updated": False, "reason": "empty_chunks", "chunks_indexed": 0}

        # 2. Atomically swap the TF-IDF cache entry
        _DEFAULT_RETRIEVER_CACHE[processed_path] = new_tfidf
        logger.info("TF-IDF Index updated atomically: %d total chunks indexed.", chunk_count)

        # 3. Clear dependent semantic/hybrid caches for this dataset path and optionally warm them
        keys_to_remove_sem = [k for k in _DEFAULT_SEMANTIC_RETRIEVER_CACHE if k.startswith(f"{processed_path}:")]
        for k in keys_to_remove_sem:
            _DEFAULT_SEMANTIC_RETRIEVER_CACHE.pop(k, None)

        keys_to_remove_hyb = [k for k in _DEFAULT_HYBRID_RETRIEVER_CACHE if k.startswith(f"{processed_path}:")]
        for k in keys_to_remove_hyb:
            _DEFAULT_HYBRID_RETRIEVER_CACHE.pop(k, None)

        if include_semantic:
            try:
                semantic = get_default_semantic_retriever(processed_path)
                logger.info("Semantic Index updated: %d total chunks indexed.", len(semantic.chunks))
                get_default_hybrid_retriever(processed_path)
                logger.info("Hybrid Retrieval pipeline updated successfully.")
            except Exception as sem_exc:
                logger.warning("Semantic/Hybrid warm-up skipped (%s); TF-IDF index is active.", sem_exc)

        return {"updated": True, "chunks_indexed": chunk_count}
    except Exception as exc:
        logger.warning("Failed to update retrieval index (%s); existing index preserved.", exc)
        return {"updated": False, "reason": str(exc), "chunks_indexed": 0}


def generate_ingestion_report(stats: Dict[str, Any], report_path: str = DEFAULT_REPORT_PATH) -> None:
    """Save structured ingestion report and print formatted summary."""
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info("Saved ingestion report to %s", report_path)

    print("\n" + "=" * 70)
    print(" RVCE GMAIL CAMPUS NOTICE INGESTION REPORT")
    print("=" * 70)
    print(f" Mode: {'DRY RUN (Preview Only)' if stats.get('dry_run') else 'LIVE INGESTION'}")
    print(f" Execution Date: {stats.get('execution_timestamp')}")
    print(f" Search Query: {stats.get('search_query')}")
    print("-" * 70)
    print(" INGESTION METRICS:")
    print(f"  • Total Gmail messages scanned:       {stats.get('total_messages_scanned', 0)}")
    print(f"  • Already processed in earlier runs:  {stats.get('already_processed_messages', 0)}")
    print(f"  • New candidate messages evaluated:   {stats.get('candidate_messages_evaluated', 0)}")
    print(f"  • Relevant RVCE notices found:        {stats.get('relevant_emails_found', 0)}")
    print(f"  • Rejected noise / personal emails:   {stats.get('rejected_noise_emails', 0)}")
    print(f"  • Parent email documents stored:      {stats.get('emails_stored', 0)}")
    print(f"  • PDF attachments extracted:          {stats.get('pdf_attachments_extracted', 0)}")
    print(f"  • Failed / empty extractions:         {stats.get('failed_attachment_extractions', 0)}")
    print("-" * 70)
    corpus = stats.get("corpus_statistics", {})
    print(" PROCESSED CORPUS STATUS:")
    print(f"  • Total documents in processed corpus:{corpus.get('total_documents', 0)}")
    print(f"  • Website-collected documents:        {corpus.get('web_documents', 0)}")
    print(f"  • Email notice documents:             {corpus.get('email_documents', 0)}")
    print(f"  • Email attachment documents:         {corpus.get('email_attachment_documents', 0)}")
    print("=" * 70 + "\n")


def run_gmail_ingestion(
    service: Optional[Any] = None,
    credentials_path: str = "secrets/credentials.json",
    token_path: str = "secrets/token.json",
    query: Optional[str] = None,
    all_history: bool = False,
    start_date: Optional[str] = None,
    days: int = 30,
    max_emails: Optional[int] = 100,
    dry_run: bool = False,
    reindex: bool = False,
    incremental: bool = False,
    allow_interactive_oauth: bool = True,
    state_path: str = DEFAULT_STATE_PATH,
    raw_path: str = DEFAULT_RAW_PATH,
    processed_path: str = DEFAULT_PROCESSED_PATH,
    manifest_path: str = DEFAULT_MANIFEST_PATH,
    report_path: str = DEFAULT_REPORT_PATH,
) -> Dict[str, Any]:
    """
    Executes incremental or full-history Gmail notice ingestion.

    Supports:
    - Incremental synchronization using persistent state watermark and message ID deduplication
    - Complete college email history fetch (--all-history / --start-date 2024/01/01)
    - Full pagination across all matching messages
    - Deduplication using Gmail message IDs, corpus document IDs, and attachment SHA-256 hashes
    - Preservation of existing 105-document corpus without re-ingesting historical emails
    - Atomic re-indexing of the retrieval pipeline (--reindex)
    - Safe error handling per message and per run
    """
    state = load_ingestion_state(state_path)
    processed_ids: Set[str] = set(state.get("processed_message_ids", []))
    existing_doc_ids, existing_corpus_msg_ids = get_existing_corpus_ids(processed_path)
    # Never re-ingest messages already persisted in the processed corpus
    processed_ids.update(existing_corpus_msg_ids)

    max_internal_date: int = int(state.get("last_synced_internal_date_ms", 0) or 0)

    # 1. Initialize Gmail Service
    if service is None:
        if dry_run and not os.path.exists(credentials_path):
            logger.info("Dry-run invoked without credentials file; skipping live API calls.")
            return {"status": "dry_run_no_credentials"}
        try:
            service = get_gmail_service(
                credentials_path=credentials_path,
                token_path=token_path,
                allow_interactive_oauth=allow_interactive_oauth,
            )
        except Exception as auth_exc:
            logger.error("Failed to initialize Gmail service: %s", type(auth_exc).__name__)
            return {
                "status": "error",
                "error": f"Gmail authentication failed: {type(auth_exc).__name__}",
                "emails_stored": 0,
                "pdf_attachments_extracted": 0,
                "corpus_statistics": get_corpus_statistics(processed_path),
            }

    # 2. Build Query
    base_query = query or "from:(*@rvce.edu.in OR *@rvei.edu.in OR *@vtu.ac.in) -label:SPAM -label:TRASH"
    if not query:
        if all_history or start_date:
            effective_start = start_date or DEFAULT_COLLEGE_JOIN_DATE
            base_query = f"{base_query} after:{effective_start}"
        elif incremental and max_internal_date > 0:
            # Use watermark date with 1-day safety overlap; message ID set guarantees exact deduplication
            watermark_dt = datetime.fromtimestamp(max(0, (max_internal_date / 1000.0) - 86400), tz=timezone.utc)
            cutoff_date = watermark_dt.strftime("%Y/%m/%d")
            base_query = f"{base_query} after:{cutoff_date}"
        elif days > 0:
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y/%m/%d")
            base_query = f"{base_query} after:{cutoff_date}"

    # If all-history is requested and max_emails was not explicitly set or is default 100, allow unlimited
    effective_max_emails = None if (all_history and max_emails == 100) else max_emails

    logger.info(
        "Starting Gmail Ingestion. Query: '%s' (All History: %s, Incremental: %s, Max: %s, Dry Run: %s)",
        base_query,
        all_history,
        incremental,
        effective_max_emails or "unlimited",
        dry_run,
    )

    # 3. Paginate through matching message IDs
    new_message_ids: List[str] = []
    page_token = None
    page_num = 0
    total_query_matches = 0

    try:
        while True:
            page_num += 1
            page_limit = (
                100
                if (effective_max_emails is None or effective_max_emails <= 0)
                else min(100, effective_max_emails - len(new_message_ids))
            )
            if page_limit <= 0:
                break

            batch_ids, next_page = fetch_message_ids(
                service=service,
                query=base_query,
                max_results=page_limit,
                page_token=page_token,
                raise_on_error=True,
            )
            total_query_matches += len(batch_ids)

            for mid in batch_ids:
                if mid not in processed_ids and mid not in new_message_ids:
                    new_message_ids.append(mid)
                if effective_max_emails and len(new_message_ids) >= effective_max_emails:
                    break

            logger.info(
                "Page %d: retrieved %d message IDs (%d new candidates to process, %d total query matches so far)",
                page_num,
                len(batch_ids),
                len(new_message_ids),
                total_query_matches,
            )

            if not next_page or (effective_max_emails and len(new_message_ids) >= effective_max_emails):
                break
            page_token = next_page
    except Exception as list_exc:
        logger.error("Error listing Gmail messages during sync: %s", type(list_exc).__name__)
        return {
            "status": "error",
            "error": f"Failed listing messages: {type(list_exc).__name__}",
            "emails_stored": 0,
            "pdf_attachments_extracted": 0,
            "corpus_statistics": get_corpus_statistics(processed_path),
        }

    already_processed_count = total_query_matches - len(new_message_ids)
    logger.info(
        "Pagination complete: %d total messages scanned across %d pages (%d new candidates to evaluate, %d already processed).",
        total_query_matches,
        page_num,
        len(new_message_ids),
        already_processed_count,
    )

    stats: Dict[str, Any] = {
        "status": "ok",
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "search_query": base_query,
        "all_history_mode": all_history,
        "incremental_mode": incremental,
        "total_messages_scanned": total_query_matches,
        "already_processed_messages": already_processed_count,
        "candidate_messages_evaluated": len(new_message_ids),
        "candidate_messages": len(new_message_ids),
        "relevant_emails_found": 0,
        "accepted_notices": 0,
        "rejected_noise_emails": 0,
        "rejected_noise": 0,
        "emails_stored": 0,
        "new_email_records": 0,
        "pdf_attachments_found": 0,
        "pdf_attachments_extracted": 0,
        "new_attachment_records": 0,
        "failed_attachment_extractions": 0,
        "failed_message_ingestions": 0,
        "index_updated": False,
        "newly_indexed_documents": [],
    }

    # 4. Process each new candidate message safely
    for idx, mid in enumerate(new_message_ids, 1):
        try:
            msg = fetch_message_full(service, mid)
            if not msg:
                stats["failed_message_ingestions"] += 1
                continue

            internal_date = int(msg.get("internalDate", 0) or 0)
            if internal_date > max_internal_date:
                max_internal_date = internal_date

            payload = msg.get("payload", {})
            headers = payload.get("headers", [])
            snippet = msg.get("snippet", "")

            # Find attachment filenames for filtering
            from src.gmail_parser import _find_attachments

            attachments_meta = _find_attachments(payload)
            att_filenames = [a["filename"] for a in attachments_meta]

            is_accepted, reason = is_relevant_notice(headers, snippet, att_filenames)
            if not is_accepted:
                logger.debug("[%d/%d] Message %s rejected: %s", idx, len(new_message_ids), mid, reason)
                stats["rejected_noise_emails"] += 1
                stats["rejected_noise"] += 1
                processed_ids.add(mid)
                continue

            stats["relevant_emails_found"] += 1
            stats["accepted_notices"] += 1
            logger.info("[%d/%d] Message %s ACCEPTED (%s)", idx, len(new_message_ids), mid, reason)

            # Download attachment bytes for PDFs
            att_bytes_map: Dict[str, bytes] = {}
            for att in attachments_meta:
                if att["filename"].lower().endswith(".pdf"):
                    stats["pdf_attachments_found"] += 1
                    try:
                        raw_b = fetch_attachment_bytes(service, mid, att["attachmentId"])
                    except Exception as att_exc:
                        logger.warning("Error downloading attachment %s for message %s: %s", att["filename"], mid, type(att_exc).__name__)
                        raw_b = None
                    if raw_b:
                        att_bytes_map[att["attachmentId"]] = raw_b
                    else:
                        logger.warning("Failed to download attachment %s for message %s", att["filename"], mid)
                        stats["failed_attachment_extractions"] += 1

            # Parse message and attachments
            parent_rec, child_recs = parse_email_message(msg, att_bytes_map)

            if dry_run:
                if parent_rec:
                    logger.info("  [DRY RUN] Email Document: %s (%s)", parent_rec["title"], parent_rec["topic"])
                    stats["emails_stored"] += 1
                    stats["new_email_records"] += 1
                for ch in child_recs:
                    if ch.get("extraction_status") == "SUCCESS":
                        stats["pdf_attachments_extracted"] += 1
                        stats["new_attachment_records"] += 1
                        logger.info("  [DRY RUN] Attachment Document: %s (%d chars, %s)", ch["title"], len(ch["content"]), ch["topic"])
                    else:
                        stats["failed_attachment_extractions"] += 1
                        logger.warning("  [DRY RUN] Empty attachment: %s", ch["title"])
            else:
                if parent_rec:
                    appended = append_processed_record(parent_rec, processed_path, existing_doc_ids=existing_doc_ids)
                    if appended:
                        save_raw_record(parent_rec, raw_path)
                        update_manifest_entry(parent_rec, manifest_path)
                        stats["emails_stored"] += 1
                        stats["new_email_records"] += 1
                        stats["newly_indexed_documents"].append({
                            "document_id": parent_rec["document_id"],
                            "title": parent_rec.get("title"),
                            "source_type": "EMAIL",
                            "document_date": parent_rec.get("document_date") or parent_rec.get("last_modified"),
                        })

                for ch in child_recs:
                    appended_att = append_processed_record(ch, processed_path, existing_doc_ids=existing_doc_ids)
                    if appended_att:
                        save_raw_record(ch, raw_path)
                        update_manifest_entry(ch, manifest_path)
                        if ch.get("extraction_status") == "SUCCESS":
                            stats["pdf_attachments_extracted"] += 1
                            stats["new_attachment_records"] += 1
                            stats["newly_indexed_documents"].append({
                                "document_id": ch["document_id"],
                                "title": ch.get("title"),
                                "source_type": "EMAIL_ATTACHMENT",
                                "document_date": ch.get("document_date") or ch.get("last_modified"),
                            })
                        else:
                            stats["failed_attachment_extractions"] += 1

            processed_ids.add(mid)
        except Exception as msg_exc:
            logger.warning("Error processing Gmail message %s: %s", mid, type(msg_exc).__name__)
            stats["failed_message_ingestions"] += 1

    # 5. Update state
    if not dry_run:
        state["processed_message_ids"] = list(processed_ids)
        state["last_synced_internal_date_ms"] = max_internal_date
        state["last_sync_timestamp"] = datetime.now(timezone.utc).isoformat()
        state["total_emails_ingested"] = state.get("total_emails_ingested", 0) + stats["emails_stored"]
        state["total_attachments_ingested"] = state.get("total_attachments_ingested", 0) + stats["pdf_attachments_extracted"]
        save_ingestion_state(state, state_path)

        # 6. Re-index retrieval models if requested or if new documents were added when reindex=True
        if reindex:
            index_info = update_retrieval_index(processed_path, include_semantic=True)
            stats["index_updated"] = bool(index_info.get("updated"))

    # 7. Collect corpus accounting and write report
    stats["corpus_statistics"] = get_corpus_statistics(processed_path)
    generate_ingestion_report(stats, report_path)

    return stats


class GmailSyncWorker:
    """Background worker for non-blocking, incremental Gmail notice synchronization.

    Periodically checks for new RVCE Gmail notices, filters and extracts relevant emails
    and PDF attachments, appends new documents to the processed corpus, persists sync state,
    and atomically updates the in-memory retrieval index.
    """

    def __init__(
        self,
        interval_seconds: int = 300,
        credentials_path: str = "secrets/credentials.json",
        token_path: str = "secrets/token.json",
        state_path: str = DEFAULT_STATE_PATH,
        raw_path: str = DEFAULT_RAW_PATH,
        processed_path: str = DEFAULT_PROCESSED_PATH,
        manifest_path: str = DEFAULT_MANIFEST_PATH,
        report_path: str = DEFAULT_REPORT_PATH,
        service: Optional[Any] = None,
    ) -> None:
        self.interval_seconds = max(10, int(interval_seconds))
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.state_path = state_path
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.manifest_path = manifest_path
        self.report_path = report_path
        self.service = service

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.sync_in_progress: bool = False
        self.last_sync_timestamp: Optional[str] = None
        self.last_successful_sync_timestamp: Optional[str] = None
        self.last_sync_status: str = "idle"
        self.last_sync_error: Optional[str] = None
        self.last_sync_stats: Optional[Dict[str, Any]] = None
        self.total_sync_runs: int = 0
        self.failed_sync_count: int = 0
        self.newly_indexed_documents: List[Dict[str, Any]] = []
        self.last_index_update_timestamp: Optional[str] = None
        self.chunks_indexed: int = 0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def sync_once(self) -> Dict[str, Any]:
        """Run a single incremental synchronization pass safely."""
        if not self._lock.acquire(blocking=False):
            return {"status": "busy", "message": "A Gmail sync pass is already in progress."}

        try:
            self.sync_in_progress = True
            self.last_sync_status = "running"
            # When no mock service is injected, verify cached token exists so background
            # worker never blocks waiting on an interactive browser OAuth consent prompt.
            if self.service is None and not os.path.exists(self.token_path):
                msg = f"OAuth token not found at {self.token_path}; skipping background sync."
                logger.info(msg)
                self.last_sync_status = "skipped_no_token"
                self.last_sync_error = msg
                return {"status": "skipped_no_token", "message": msg}

            stats = run_gmail_ingestion(
                service=self.service,
                credentials_path=self.credentials_path,
                token_path=self.token_path,
                incremental=True,
                days=7,
                max_emails=50,
                dry_run=False,
                reindex=False,  # We conditionally re-index below only when new docs arrive
                allow_interactive_oauth=False,
                state_path=self.state_path,
                raw_path=self.raw_path,
                processed_path=self.processed_path,
                manifest_path=self.manifest_path,
                report_path=self.report_path,
            )

            self.total_sync_runs += 1
            now_iso = datetime.now(timezone.utc).isoformat()
            self.last_sync_timestamp = now_iso

            if stats.get("status") == "error":
                self.failed_sync_count += 1
                self.last_sync_status = "error"
                self.last_sync_error = stats.get("error")
                self.last_sync_stats = stats
                return stats

            new_docs = int(stats.get("emails_stored", 0)) + int(stats.get("pdf_attachments_extracted", 0))
            if new_docs > 0:
                idx_res = update_retrieval_index(self.processed_path, include_semantic=False)
                stats["index_updated"] = bool(idx_res.get("updated"))
                stats["chunks_indexed"] = idx_res.get("chunks_indexed", 0)
                if stats["index_updated"]:
                    self.last_index_update_timestamp = now_iso
                    self.chunks_indexed = int(stats["chunks_indexed"])

            self.newly_indexed_documents = list(stats.get("newly_indexed_documents") or [])
            self.last_successful_sync_timestamp = now_iso
            self.last_sync_status = stats.get("status", "ok")
            self.last_sync_error = None
            self.last_sync_stats = stats
            return stats
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: sync failed"
            logger.warning("Background Gmail sync failed safely: %s", err_msg)
            self.failed_sync_count += 1
            self.last_sync_status = "error"
            self.last_sync_error = err_msg
            return {"status": "error", "error": err_msg}
        finally:
            self.sync_in_progress = False
            self._lock.release()

    def _run_loop(self) -> None:
        logger.info("GmailSyncWorker started (interval=%ds).", self.interval_seconds)
        while not self._stop_event.is_set():
            self.sync_once()
            if self._stop_event.wait(timeout=self.interval_seconds):
                break
        logger.info("GmailSyncWorker stopped.")

    def start(self) -> bool:
        """Start the background daemon thread if not already running."""
        if self.is_running:
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="gmail-sync-worker", daemon=True)
        self._thread.start()
        return True

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the background worker to stop and wait briefly for termination."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None

    def get_status(self) -> Dict[str, Any]:
        """Return a snapshot of worker, MCP, and corpus synchronization status."""
        state = load_ingestion_state(self.state_path)
        corpus = get_corpus_statistics(self.processed_path)
        return {
            "worker_running": self.is_running,
            "sync_in_progress": self.sync_in_progress,
            "interval_seconds": self.interval_seconds,
            "last_sync_status": self.last_sync_status,
            "last_sync_timestamp": self.last_sync_timestamp or state.get("last_sync_timestamp"),
            "last_successful_sync_timestamp": self.last_successful_sync_timestamp or state.get("last_sync_timestamp"),
            "last_sync_error": self.last_sync_error,
            "total_sync_runs": self.total_sync_runs,
            "failed_sync_count": self.failed_sync_count,
            "newly_indexed_documents": list(self.newly_indexed_documents),
            "last_index_update_timestamp": self.last_index_update_timestamp,
            "chunks_indexed": self.chunks_indexed,
            "tracked_message_ids": len(state.get("processed_message_ids", [])),
            "corpus_statistics": corpus,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest RVCE notices from authorized Gmail account.")
    parser.add_argument("--credentials", default="secrets/credentials.json", help="Path to OAuth client credentials JSON")
    parser.add_argument("--token", default="secrets/token.json", help="Path to cached OAuth token JSON")
    parser.add_argument("--query", default=None, help="Custom Gmail search query filter")
    parser.add_argument("--all-history", action="store_true", help="Fetch full history from college joining date (2024 to present)")
    parser.add_argument("--incremental", action="store_true", help="Fetch only new incoming emails since last sync watermark")
    parser.add_argument("--start-date", default=None, help="Start date in YYYY/MM/DD format (default: 2024/01/01 when --all-history is set)")
    parser.add_argument("--days", type=int, default=30, help="Lookback days for historical fetch (ignored if --all-history is set)")
    parser.add_argument("--max-emails", type=int, default=None, help="Maximum emails to process (default: unlimited for --all-history, 100 for recent)")
    parser.add_argument("--report", default=DEFAULT_REPORT_PATH, help="Output path for structured JSON report")
    parser.add_argument("--dry-run", action="store_true", help="Preview candidate notices without saving to disk")
    parser.add_argument("--reindex", action="store_true", help="Re-index retrieval models upon completion")
    parser.add_argument("--watch", action="store_true", help="Run continuously as a scheduled background sync worker")
    parser.add_argument("--interval", type=int, default=300, help="Polling interval in seconds when --watch is enabled (default: 300)")

    args = parser.parse_args()

    if args.watch:
        worker = GmailSyncWorker(
            interval_seconds=args.interval,
            credentials_path=args.credentials,
            token_path=args.token,
            report_path=args.report,
        )
        logger.info("Starting scheduled Gmail synchronization worker (poll interval: %ds). Press Ctrl+C to stop.", args.interval)
        try:
            while True:
                worker.sync_once()
                time.sleep(max(10, args.interval))
        except KeyboardInterrupt:
            logger.info("Stopped scheduled Gmail synchronization worker.")
        return

    max_emails = args.max_emails
    if max_emails is None and not args.all_history:
        max_emails = 100

    run_gmail_ingestion(
        credentials_path=args.credentials,
        token_path=args.token,
        query=args.query,
        all_history=args.all_history,
        incremental=args.incremental,
        start_date=args.start_date,
        days=args.days,
        max_emails=max_emails,
        dry_run=args.dry_run,
        reindex=args.reindex,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
