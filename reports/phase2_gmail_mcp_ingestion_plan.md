# Dynamic Campus Notice Ingestion Plan: Gmail MCP Integration

**Project:** AI-Powered Campus Information Assistant (RVCE) — Phase II  
**Document ID:** `reports/phase2_gmail_mcp_ingestion_plan.md`  
**Status:** Planning & Compatibility Audit (Read-Only)  
**Target:** Authorized Gmail Ingestion for Real-Time Campus Notices, Exam Schedules, Fee Circulars, and Placement Alerts  

---

## 1. Executive Summary & Strategic Objective

The RVCE Campus AI Assistant must remain continuously synchronized with time-sensitive academic, administrative, fee, and placement updates. While Phase I relied on a static web crawler hitting `rvce.edu.in` periodically, critical university communications (such as fast-breaking VTU circulars, CIE/SEE timetable revisions, emergency fee payment extensions, and placement interview schedules) are disseminated primarily through official email distribution lists to faculty, staff, and students.

This document establishes the architecture, compatibility audit, security protocols, parsing rules, and metadata schemas required to integrate a **Gmail Model Context Protocol (MCP)** ingestion engine. This engine will ingest notices without requiring modifications to existing Phase I core retrievers, breaking DVC dataset versioning, or compromising MLflow evaluation pipelines.

```
+-----------------------------------------------------------------------------------+
|                           Authorized RVCE Gmail Account                           |
|       (Official Circulars, CIE/SEE Schedules, Fee Updates, Placement Alerts)      |
+-----------------------------------------------------------------------------------+
                                          |
                                          v  [OAuth 2.0 / Read-Only / Stdio]
+-----------------------------------------------------------------------------------+
|                                 Gmail MCP Server                                  |
|   Tools: list_messages, get_message, get_attachment | Resources: gmail://messages   |
+-----------------------------------------------------------------------------------+
                                          |
                                          v  [Structured JSON Payload]
+-----------------------------------------------------------------------------------+
|                         Ingestion & Filtering Pipeline                            |
|  - Domain Whitelisting (@rvce.edu.in, @vtu.ac.in)                                 |
|  - DKIM/SPF Authenticity Check & Noise Exclusion                                  |
|  - Multi-tier Keyword Relevancy Routing                                           |
+-----------------------------------------------------------------------------------+
                                          |
                     +--------------------+--------------------+
                     |                                         |
                     v                                         v
        +--------------------------+              +--------------------------+
        |   Email Body Extractor   |              |  PDF Attachment Handler  |
        |  - HTMLExtractor / Strip |              |  - Digital: PyMuPDF      |
        |  - Normalization (NFKC)  |              |  - Scanned: RapidOCR     |
        +--------------------------+              +--------------------------+
                     |                                         |
                     +--------------------+--------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        Revision & Deduplication Engine                            |
|  - RFC 822 Message-ID & Attachment SHA-256 Hash Matching                         |
|  - Superseded Circular State Machine (ACTIVE -> SUPERSEDED / EXTENDED / EXPIRED)  |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     Corpus Persistence & Version Control                          |
|  - data/raw/campus_documents.jsonl (Raw bytes / MIME base64)                      |
|  - data/processed/campus_documents.jsonl (Normalized chunks + Extended Metadata)  |
|  - data/raw/collection_manifest.json (Provenance Tracking)                        |
|  - DVC Commit & MLflow Run Tracking                                               |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                               Serving & Retrieval                                 |
|  - TFIDFRetriever & SemanticRetriever Index Auto-Refresh                          |
|  - HybridRetriever with Validity Filtering & Recency Weighting                    |
+-----------------------------------------------------------------------------------+
```

---

## 2. Ingestion Pipeline & Knowledge Base Compatibility Audit

### 2.1 Existing Pipeline Architecture
The Phase I pipeline consists of five clean, modular stages:
1. **Raw Ingestion (`src/crawler.py`, `src/build_dataset.py`):** Saves crawled records into `data/raw/campus_documents.jsonl` with binary attachments preserved as base64 strings (`raw_bytes_base64`).
2. **Extraction (`src/extractors.py`):** Dispatches HTML to `HTMLExtractor` and PDF binaries to `PDFExtractor` (with automatic fallback to `ocr_pdf_bytes()` from `src/ocr.py`).
3. **Cleaning & Metadata Tagging (`src/cleaning.py`):** Cleans text, normalizes Unicode (NFKC), collapses whitespace, assigns deterministic document IDs (`doc_html_*`, `doc_pdf_*`), and derives topics.
4. **Validation & Manifest Provenance (`src/validation.py`):** Enforces strict schema compliance, detects exact/near-duplicates via 3-gram shingles, updates `data/raw/collection_manifest.json`, and outputs `reports/data_collection_report.json`.
5. **Retrieval & Benchmark (`src/retrieval.py`, `src/evaluation.py`):** Indexes `data/processed/campus_documents.jsonl` into TF-IDF sparse matrices and `all-MiniLM-L6-v2` dense vectors.

### 2.2 Compatibility Assessment
| Pipeline Component | Compatibility Status | Required Extension / Adaptation |
|---|---|---|
| **Raw Storage (`data/raw/`)** | Fully Compatible | Store email message payloads as JSON records with `content_type: "EMAIL"` and attachments as separate records linked via `parent_id`. |
| **Processed Storage (`data/processed/`)** | Fully Compatible | Follow existing JSONL schema. Add extended Phase II metadata fields (`email_subject`, `sender`, `validity_status`, `supersedes_id`). |
| **Extractors (`src/extractors.py`)** | Fully Compatible | Existing `HTMLExtractor` and `PDFExtractor` handle email HTML and PDF attachments directly. |
| **OCR Module (`src/ocr.py`)** | Fully Compatible | Scanned circulars attached to emails will automatically route to `rapidocr-onnxruntime` + PyMuPDF. |
| **Cleaning (`src/cleaning.py`)** | Mostly Compatible | Extend `TOPIC_RULES` to classify email notice subjects (e.g., CIE circulars, revaluation deadlines). |
| **QA Validation (`src/validation.py`)** | Fully Compatible | `REQUIRED_SCHEMA_FIELDS` will continue to validate without schema violations. |
| **Retrievers (`src/retrieval.py`)** | Compatible with Enhancements | Retrievers read `content` and `title` directly. Can add metadata filtering on `validity_status != "SUPERSEDED"`. |

---

## 3. Gmail MCP Integration & Retrieval Mechanics

### 3.1 Model Context Protocol (MCP) Primitive Architecture
The Gmail MCP client will act as a standard MCP client interfacing with a specialized Gmail MCP server via standard input/output (`stdio`) or SSE (Server-Sent Events).

```
[ Antigravity / Pipeline Orchestrator ]
                  |
         JSON-RPC 2.0 (stdio)
                  v
       [ Gmail MCP Server ]
                  |
         Google API Client (HTTPS)
                  v
         [ Gmail REST API v1 ]
```

### 3.2 Required MCP Tools & Resources
The Gmail MCP server exposes the following standard tools and resources:

1. **`list_messages` (Tool):**
   - **Arguments:**
     - `query` (string): Gmail search query (e.g., `from:(*@rvce.edu.in) after:2026/10/01`).
     - `max_results` (int): Batch size (default: 50, maximum: 100).
     - `page_token` (optional string): Pagination token.
     - `include_spam_trash` (bool): `false`.
   - **Returns:** List of message summaries containing `id`, `threadId`, and label IDs.

2. **`get_message` (Tool):**
   - **Arguments:**
     - `message_id` (string): Unique Gmail message ID.
     - `format` (string): `"full"` (MIME structure, headers, parts, snippet).
   - **Returns:** Complete message payload including RFC 2822 headers (`From`, `Subject`, `Date`, `Message-ID`), body parts (`text/plain`, `text/html`), and attachment metadata (`attachmentId`, `filename`, `mimeType`, `size`).

3. **`get_attachment` (Tool):**
   - **Arguments:**
     - `message_id` (string): Message ID containing attachment.
     - `attachment_id` (string): Unique attachment ID.
   - **Returns:** Base64URL-encoded raw attachment byte payload.

4. **`gmail://messages/{message_id}` (Resource):**
   - Provides URI-based addressability for direct context retrieval during RAG synthesis.

---

## 4. MCP Server Options & Authentication Requirements

### 4.1 Server Options Comparison

| Option | Implementation | Pros | Cons | Recommendation |
|---|---|---|---|---|
| **Option A: Python FastMCP Server (`src/mcp/gmail_server.py`)** | In-house FastMCP server using `google-api-python-client` & `google-auth` | - Seamless integration with existing `.venv`<br>- Zero external Node.js dependencies<br>- Direct reuse of `src/ocr.py` and `src/cleaning.py`<br>- Exact control over scopes and error handling | - Custom code to maintain | **Recommended (P0)** |
| **Option B: Community Node.js Gmail MCP (`@modelcontextprotocol/server-gmail`)** | Standard Node.js package running via `npx` | - Maintained by Anthropic/MCP ecosystem<br>- Standardized tool interfaces | - Requires Node.js in deployment container<br>- External process orchestration<br>- Cannot share Python OCR runtime directly | Secondary alternative |
| **Option C: Direct IMAP/OAuth Worker** | Custom Python script with `imaplib` + OAuth 2.0 | - Bypasses Google API quota limits | - Does not adhere to MCP protocol standard<br>- More complex MIME parsing | Not recommended |

### 4.2 Authentication & Authorization Protocol

```
+-----------------------------------------------------------------------------------+
|                        Google Cloud Project (GCP Console)                         |
|  - Enable "Gmail API"                                                             |
|  - Configure OAuth Consent Screen ("Internal" for RVCE Google Workspace)          |
|  - Create OAuth 2.0 Client ID ("Desktop Application")                             |
+-----------------------------------------------------------------------------------+
                                          |
                                          v  (Downloads credentials.json)
+-----------------------------------------------------------------------------------+
|                          Local Secret Management (.env)                           |
|  - Store GMAIL_CLIENT_SECRETS_FILE="secrets/gmail_credentials.json"               |
|  - Store GMAIL_TOKEN_CACHE_FILE="secrets/gmail_token.json"                        |
|  - STRICT: Add secrets/ to .gitignore and .dvcignore                              |
+-----------------------------------------------------------------------------------+
                                          |
                                          v  (One-time browser consent)
+-----------------------------------------------------------------------------------+
|                             OAuth 2.0 Consent Flow                                |
|  - Request exact scope: https://www.googleapis.com/auth/gmail.readonly            |
|  - Offline access mode to obtain long-lived Refresh Token                         |
|  - Auto-refresh expired access tokens without prompt                              |
+-----------------------------------------------------------------------------------+
```

#### Security Boundaries:
- **Least Privilege Scope:** Strictly limited to `https://www.googleapis.com/auth/gmail.readonly`. The application must **never** request write, delete, send, or full account access.
- **Zero Credentials in Repository:** `secrets/` directory is isolated and excluded from version control (`.gitignore`, `.dvcignore`, and Docker build contexts).
- **Service Account Alternative:** If managed centrally by RVCE IT, a Google Workspace Service Account with Domain-Wide Delegation can be scoped to read from a single designated intake mailbox (e.g., `assistant-intake@rvce.edu.in`).

---

## 5. Noise Filtering, Whitelisting, and Spam Rejection Logic

Personal and general promotional emails must be filtered before touching raw storage. A three-tier filtering pipeline ensures only authenticated, official institutional notifications are ingested.

```
                              Incoming Email
                                    |
                                    v
+-----------------------------------------------------------------------+
|  Tier 1: Gmail Query Filter (Pre-Retrieval)                           |
|  - from:(*@rvce.edu.in OR *@rvei.edu.in OR *@vtu.ac.in)              |
|  - -is:chat -is:draft -label:SPAM -label:TRASH                        |
+-----------------------------------------------------------------------+
                                    | Passes
                                    v
+-----------------------------------------------------------------------+
|  Tier 2: Authenticity & Domain Header Verification                    |
|  - Cryptographic verification (SPF=PASS, DKIM=PASS)                   |
|  - Whitelisted Sender / Subdomain Matching:                           |
|    * principal@rvce.edu.in, coe@rvce.edu.in                           |
|    * accounts@rvce.edu.in, placement@rvce.edu.in                      |
|    * dean.academics@rvce.edu.in, dean.studentaffairs@rvce.edu.in      |
|    * hod.<branch>@rvce.edu.in, warden.*@rvce.edu.in                   |
+-----------------------------------------------------------------------+
                                    | Passes
                                    v
+-----------------------------------------------------------------------+
|  Tier 3: Content Relevance & Exclusion Regex Engine                  |
|  - Drop personal conversational threads (Subject: "Re:", "Lunch?")    |
|  - Drop OTPs, verification links, social alerts, mass marketing       |
|  - Require institutional notice signature or keyword match:           |
|    * "circular", "notification", "schedule", "deadline", "cie",       |
|      "see", "fee", "concession", "placement", "timetable", "holiday"  |
+-----------------------------------------------------------------------+
                                    | Passes
                                    v
                            Approved for Ingestion
```

### 5.1 Sender Domain & Account Whitelist

```python
AUTHORIZED_DOMAINS = {
    "rvce.edu.in",
    "rvei.edu.in",
    "vtu.ac.in",
}

AUTHORIZED_OFFICE_ACCOUNTS = {
    "principal@rvce.edu.in",
    "viceprincipal@rvce.edu.in",
    "coe@rvce.edu.in",
    "dean.academics@rvce.edu.in",
    "dean.studentaffairs@rvce.edu.in",
    "accounts@rvce.edu.in",
    "fee@rvce.edu.in",
    "placement@rvce.edu.in",
    "library@rvce.edu.in",
    "hostel.office@rvce.edu.in",
}
```

### 5.2 Negative Exclusion Filters
Emails matching any of the following patterns will be discarded immediately:
- **Subject patterns:** `^Re:\s`, `^Fwd:\s(?![A-Z0-9/_-]+\s*(Circular|Notification))`, `OTP`, `Verification Code`, `Password Reset`, `LinkedIn`, `GitHub`, `Newsletter`, `Webinar Invitation`.
- **MIME headers:** `Auto-Submitted: auto-generated`, `X-Auto-Response-Suppress: All`.
- **Sender headers:** `noreply@google.com`, `notifications@*`, `support@*`.

---

## 6. Parsing Strategy: Email Bodies and PDF Attachments

An official college email commonly contains an introductory summary in the body and a detailed official circular attached as a PDF.

```
+---------------------------------------------------------------------------------+
|                                 Email Notice                                    |
|                                                                                 |
|  From: coe@rvce.edu.in                                                          |
|  Subject: Revised SEE Timetable - 3rd Sem B.E. Dec 2026                         |
|                                                                                 |
|  [Email Body HTML]                                                              |
|  "Dear Students, please find attached the revised Semester End Exam schedule..." |
|                                                                                 |
|  [Attachment: REVISED-SEE-TIMETABLE-DEC2026.pdf] (Scanned Stamp Notice)         |
+---------------------------------------------------------------------------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
      [Parent Ingestion Record]               [Child Ingestion Record]
      doc_email_a71e8c4f92                    doc_email_att_c81f3b01a2
      - source_type: "EMAIL"                  - source_type: "EMAIL_ATTACHMENT"
      - parent_document_id: null              - parent_document_id: "doc_email_a71e8c4f92"
      - content: Extracted Body               - content: RapidOCR Page Output
      - title: "Revised SEE Timetable..."     - title: "REVISED-SEE-TIMETABLE-DEC2026.pdf"
```

### 6.1 Email Body Parsing
1. **Multipart Handling:** If `text/html` is present, it is favored over `text/plain` because tabular exam schedules and formatted bullet points are preserved.
2. **Boilerplate Stripping:** Use `HTMLExtractor` (`src/extractors.py`) to strip:
   - Tracking images / web beacons (`<img width="1" height="1">`).
   - Outlook / Gmail auto-generated wrappers and CSS blocks (`<style>`, `MsoNormal`).
   - Standard email confidentiality footers (*"This message and any attachments are intended solely..."*).
3. **Text Normalization:** Pass extracted string through `clean_text()` (`src/cleaning.py`) for NFKC normalization and line deduplication.

### 6.2 Attachment Ingestion & RapidOCR Execution
1. **MIME Whitelisting:** Process `application/pdf`, `.pdf`. Skip `.exe`, `.zip`, `.mp4`.
2. **Text Extraction:** Pass raw bytes to `PDFExtractor` (`src/extractors.py`).
3. **Automatic OCR Fallback:** If `len(native_text) < 50`:
   - Invoke `ocr_pdf_bytes()` (`src/ocr.py`).
   - PyMuPDF rasterizes pages at 200 DPI to NumPy arrays.
   - RapidOCR ONNX extracts Kannada/English text lines, table cells, and stamps.
   - Text is formatted with page break delimiters: `\n\n--- Page Break ---\n\n`.
4. **Safety Verification:** Check maximum file size limit (≤ 25 MB). Reject corrupt or password-protected PDFs with explicit status flag `EXTRACTION_STATUS: "FAILED"`.

---

## 7. Deduplication, Revision Tracking, and Superseded-Notice Handling

A core failure mode in campus RAG systems is returning obsolete fee deadlines or superseded exam timetables when a later circular has amended them.

```
Timeline:
1. 19.08.2026: Circular RVE/ACCTS/702/2026-27 (PG Higher Sem Fee: Rs 53,220) -> [ACTIVE]
2. 28.09.2026: VTU Circular reduced fee by Rs 710
3. 29.09.2026: Circular RVE/ACCTS/925/2026-27 (Revised PG Fee: Rs 52,510)    -> [ACTIVE]
                                                                              -> [MARKS EARLIER CIRCULAR SUPERSEDED]
```

### 7.1 Revision Detection Heuristics
When a new notice is parsed, the revision analyzer inspects:
1. **Subject line markers:** `REVISED`, `EXTENSION`, `RESCHEDULED`, `POSTPONED`, `ADDENDUM`, `CORRIGENDUM`, `CIRCULAR NO: ... IN CONTINUATION OF`.
2. **Regex extraction of reference circulars:**
   ```python
   CIRCULAR_REF_REGEX = r"(?:circular\s+no\.?|vide\s+no\.?|ref(?:\.|\s+no\.?):?)\s*([A-Za-z0-9/_-]+)"
   CONTINUATION_REGEX = r"in\s+(?:continuation|supersession)\s+to\s+(?:the\s+earlier\s+circular|circular\s+dated)\s*([0-9\.\/-]+)?"
   ```
3. **Automated State Transition:**
   - When Circular B explicitly cites Circular A as superseded:
     - Circular A: `validity_status = "SUPERSEDED"`, `superseded_by_document_id = "Circular_B_ID"`
     - Circular B: `validity_status = "ACTIVE"`, `supersedes_document_id = "Circular_A_ID"`
   - If an explicit circular reference is missing but the topic, academic year, and target student cohort match with high semantic similarity, Circular A is marked `validity_status = "POTENTIALLY_SUPERSEDED"`.

### 7.2 Retrieval Query Routing
- In `src/retrieval.py`, active retrieval queries will filter or downweight superseded notices:
  - Default: Retrieve only `validity_status == "ACTIVE"`.
  - Historical queries (e.g. *"What was the fee circular on Aug 19?"*): Allow `validity_status == "SUPERSEDED"`, but append a warning banner:
    > *[WARNING: This notice has been superseded by Circular RVE/ACCTS/925/2026-27 dated 29.09.2026]*

---

## 8. Extended Metadata Schema Definition

All ingested email and attachment documents will conform to the following schema, extending the Phase I standard without breaking backwards compatibility:

```jsonc
{
  // Core Phase I Fields (Preserved for compatibility)
  "document_id": "doc_email_att_4eba733b0af1d307",
  "source_url": "gmail://messages/18d8f9a2c10b7e41/attachments/ANGjdJ-8...",
  "title": "Revised Pg Higher Semester Fees Circular 29.09.26",
  "content": "R.V College of Engineering Bengaluru - 560 059...",
  "source_type": "EMAIL_ATTACHMENT",              // "EMAIL" | "EMAIL_ATTACHMENT"
  "collected_at": "2026-11-04T00:30:00Z",
  "last_modified": "2026-09-29T11:45:00Z",
  "topic": "fees_finance",
  "extraction_status": "SUCCESS",                 // "SUCCESS" | "EMPTY" | "FAILED"

  // Phase II Gmail Extensions
  "email_metadata": {
    "source_email_identifier": "<CA+xyz123@mail.gmail.com>",
    "gmail_message_id": "18d8f9a2c10b7e41",
    "gmail_thread_id": "18d8f9a00a12e340",
    "email_subject": "Revised Fee Circular for Higher Semester PG Students 2026-27",
    "sender_display_name": "RVCE Accounts Section",
    "sender_email": "accounts@rvce.edu.in",
    "recipient_list": ["pg.students@rvce.edu.in", "hods@rvce.edu.in"],
    "received_date": "2026-09-29T11:45:00Z",
    "has_attachments": true,
    "attachment_filename": "REVISED-PG-HIGHER-SEMESTER-FEES-CIRCULAR-29.09.26.pdf",
    "attachment_content_type": "application/pdf",
    "attachment_size_bytes": 220847,
    "parent_document_id": "doc_email_18d8f9a2c10b7e41"
  },

  // Notice Governance & Validity Fields
  "notice_governance": {
    "document_category": "fees_finance",           // fees_finance | examinations_regulations | placements_careers | ...
    "academic_year": "2026-27",
    "target_cohort": "3rd Semester PG (MCA & M.Tech)",
    "issuing_authority": "Principal / Accounts Section",
    "circular_reference_no": "RVE/ACCTS/925/2026-2027",
    "validity_status": "ACTIVE",                   // "ACTIVE" | "SUPERSEDED" | "EXPIRED" | "UNKNOWN"
    "supersedes_document_id": "doc_email_att_702accts19082026",
    "superseded_by_document_id": null,
    "effective_from": "2026-09-29",
    "effective_until": "2026-10-09",               // Extracted deadline
    "ocr_extracted": true,
    "ocr_engine": "rapidocr_onnxruntime"
  }
}
```

---

## 9. Integration with DVC, MLflow, Retrieval & Evaluation

### 9.1 Data Version Control (DVC)
- Raw emails and attachments appended to `data/raw/campus_documents.jsonl`.
- Processed entries appended to `data/processed/campus_documents.jsonl`.
- DVC tracks data changes deterministically:
  ```bash
  dvc commit data/processed/campus_documents.jsonl.dvc
  git add data/processed/campus_documents.jsonl.dvc
  git commit -m "dvc: Ingested 12 new RVCE notices from Gmail MCP"
  ```

### 9.2 MLflow Experiment Tracking
Every periodic email ingestion job runs under an MLflow run in the experiment `campus_data_ingestion`:
- **Parameters logged:** `ingestion_source="gmail_mcp"`, `batch_start_date`, `batch_end_date`, `sender_filter_count`.
- **Metrics logged:** `emails_scanned`, `emails_accepted`, `emails_rejected_noise`, `attachments_extracted`, `ocr_pages_rendered`, `superseded_circulars_updated`.
- **Artifacts logged:** Sync diff report (`reports/ingestion_diff_<timestamp>.json`), updated provenance manifest.

### 9.3 Retrieval Auto-Refresh
When new documents enter `data/processed/campus_documents.jsonl`:
- `TFIDFRetriever.load_and_index()` and `SemanticRetriever.load_and_index()` can re-index in **< 3 seconds** for current corpus size (< 100 documents).
- In-memory index reload endpoint in FastAPI:
  ```python
  @app.post("/admin/reindex")
  def trigger_reindex():
      hybrid_retriever.reload()
      return {"status": "reindexed", "total_docs": len(hybrid_retriever.corpus)}
  ```

### 9.4 Evaluation Benchmark Safety
To guarantee that newly ingested emails do not cause model degradation:
- The 40-query retrieval benchmark (`data/queries/campus_queries.csv`) and the 54 unit tests (`pytest tests/`) are executed in the CI pipeline.
- Degraded evaluation triggers an alert if Recall@3 or MRR drops below baseline thresholds.

---

## 10. Secure Architecture for Periodic Email Ingestion

```
+------------------------------------------------------------------------------------+
|                                Execution Triggers                                  |
|   1. Scheduled Cron / GitHub Actions Workflow / Task Scheduler (Every 3 hours)     |
|   2. FastAPI Admin Webhook (/admin/ingest-emails - Auth Required)                  |
|   3. Google Cloud Pub/Sub Webhook on new email event                               |
+------------------------------------------------------------------------------------+
                                          |
                                          v
+------------------------------------------------------------------------------------+
|                         src/ingestion/gmail_ingest.py                              |
|  - Loads encrypted OAuth token from secrets/token.json                             |
|  - Spawns Python FastMCP Client                                                    |
|  - Queries: after:{last_synced_time} label:CollegeNotices                          |
+------------------------------------------------------------------------------------+
                                          |
                                          v
+------------------------------------------------------------------------------------+
|                          Isolated Execution Sandbox                                |
|  - Read-Only OAuth Scope (https://www.googleapis.com/auth/gmail.readonly)          |
|  - HTML Body Sanitization (Strip script, iframe, external fonts)                   |
|  - Attachment Isolation (< 25 MB max, non-executable MIME only)                    |
|  - RapidOCR Execution in memory (No disk temp file exposure)                       |
+------------------------------------------------------------------------------------+
                                          |
                                          v
+------------------------------------------------------------------------------------+
|                        Corpus & Manifest Delta Sync                                |
|  - Appends records to raw and processed JSONL datasets                             |
|  - Updates data/raw/collection_manifest.json                                       |
|  - Triggers Hot-Reload on FastAPI Retrieval Engine                                 |
+------------------------------------------------------------------------------------+
```

### 10.1 Key Security Safeguards
1. **Network Isolation:** Ingestion runs as an independent worker; the user-facing FastAPI query endpoint never has direct access to Gmail API credentials.
2. **Ephemeral Memory Processing:** PDF attachment bytes are decoded, rasterized to NumPy arrays, OCR-processed, and saved as text directly into JSONL. Raw binaries are never saved to unmonitored temporary folders.
3. **No PII Retention:** Personal student emails, grades, medical certificates, and disciplinary records are strictly excluded by Tier 2 and Tier 3 filters. Only institutional circulars addressing cohorts are retained.

---

## 11. Step-by-Step Implementation Roadmap

When authorized to begin implementation, the work should proceed in the following order:

```
[Phase 2.1] Setup & Auth
  ├── Create secrets/ configuration directory (git-ignored)
  ├── Scaffold Google Cloud Console OAuth 2.0 Desktop credentials
  └── Implement token manager with auto-refresh (src/mcp/auth.py)

[Phase 2.2] MCP Server & Tools
  ├── Implement FastMCP server (src/mcp/gmail_server.py)
  ├── Expose list_messages, get_message, get_attachment tools
  └── Write unit tests for MCP tools using mock email payloads

[Phase 2.3] Filtering & Parsing Pipeline
  ├── Implement sender domain & DKIM validation (src/mcp/filters.py)
  ├── Integrate HTMLExtractor & RapidOCR attachment extraction
  └── Implement circular revision & superseded-state detection

[Phase 2.4] Storage, Manifest & Retrieval Sync
  ├── Implement incremental JSONL writer with extended metadata
  ├── Wire provenance manifest update (data/raw/collection_manifest.json)
  └── Add FastAPI /admin/reindex hook for hot-reloading retrievers

[Phase 2.5] End-to-End Testing & CI/CD
  ├── Add mock ingestion integration test to pytest suite
  └── Log test ingestion run to MLflow experiment
```

---

## 12. Prompt and Action Execution Log

- **Prompt 1 (2026-11-03 23:00 IST):** Read-only repository audit of Phase I components. Delivered `phase2_audit_report.md`.
- **Prompt 2 (2026-11-03 23:20 IST):** Created comprehensive Phase II Data Acquisition Plan in `reports/phase2_data_acquisition_plan.md`.
- **Prompt 3 (2026-11-03 23:30 IST):** Generated 15-PDF reference table with exact download URLs.
- **Prompt 4 (2026-11-03 23:38 IST):** User manually downloaded all 15 scanned PDFs into `data/raw/`.
- **Prompt 5 (2026-11-04 00:05 IST):** Verified PDFs, installed `rapidocr-onnxruntime`, confirmed 60/60 processed records extraction, synchronized manifest and QA report, passed all 54 tests.
- **Prompt 6 (2026-11-04 00:25 IST):** Conducted planning and compatibility audit for Dynamic Campus Notice Ingestion using Gmail MCP. Produced comprehensive design report `reports/phase2_gmail_mcp_ingestion_plan.md`.

