# Phase II: Gmail MCP Dynamic Notice Ingestion — Implementation Checklist

**Document ID:** `reports/phase2_gmail_mcp_implementation_checklist.md`  
**Purpose:** Actionable, phased engineering checklist to implement dynamic campus notice ingestion from authorized Gmail into the RVCE Campus AI Assistant codebase.  
**Governing Design Document:** [`reports/phase2_gmail_mcp_ingestion_plan.md`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/reports/phase2_gmail_mcp_ingestion_plan.md)  
**Status:** Pre-Implementation Planning (Read-Only)  

---

## 1. Core Architectural Constraints & Guardrails

1. **Strict Credential Isolation:** Credentials (`credentials.json`) and user tokens (`token.json`) must reside strictly in `secrets/` and be permanently ignored by both `.gitignore` and `.dvcignore`.
2. **Read-Only Privilege:** OAuth 2.0 authorization scope must be strictly limited to `https://www.googleapis.com/auth/gmail.readonly`.
3. **Reuse Existing Infrastructure:** No duplicate parsing or cleaning code.
   - Email HTML bodies $\rightarrow$ [`HTMLExtractor`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/src/extractors.py#L46)
   - PDF attachments $\rightarrow$ [`PDFExtractor`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/src/extractors.py#L217) and [`ocr_pdf_bytes()`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/src/ocr.py#L167)
   - Text normalization $\rightarrow$ [`clean_text()`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/src/cleaning.py#L83)
   - Storage $\rightarrow$ Append to [`data/raw/campus_documents.jsonl`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/data/raw/campus_documents.jsonl) and [`data/processed/campus_documents.jsonl`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/data/processed/campus_documents.jsonl)
4. **Safe Revision Handling (Zero Keyword-Only Supersession):**
   - **Never** mark a circular superseded based purely on topical or keyword similarity.
   - Superseding requires an **explicit administrative citation** (e.g., matching reference circular number `RVE/ACCTS/702/2026-27` or date cited in *"in continuation to circular dated..."*).
   - Historical notices must **never be deleted or discarded**; they remain indexed with `validity_status = "SUPERSEDED"` so historical queries remain fully answerable.
5. **Manual-First Ingestion:** Ingestion must be triggered via an explicit manual CLI command (`python -m src.gmail_ingestion`) before any cron, webhook, or scheduled background automation is introduced.

---

## 2. Phased Implementation Checklist

### Phase 1: Security, Secrets Isolation & Dependencies
- [ ] **Task 1.1:** Add `secrets/`, `*credentials*.json`, and `*token*.json` to [`.gitignore`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/.gitignore).
- [ ] **Task 1.2:** Add `secrets/` to [`.dvcignore`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/.dvcignore).
- [ ] **Task 1.3:** Create `secrets/.gitkeep` (or create directory locally) and verify Git status treats secret files as untracked and uncommitted.
- [ ] **Task 1.4:** Update [`requirements.txt`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/requirements.txt) to record:
  - `google-api-python-client>=2.100.0`
  - `google-auth-httplib2>=0.2.0`
  - `google-auth-oauthlib>=1.2.0`
  - `rapidocr-onnxruntime>=1.4.0`

---

### Phase 2: Gmail OAuth Connection & Client Wrapper
- [ ] **Task 2.1:** Create module `src/gmail_client.py`.
- [ ] **Task 2.2:** Implement `get_gmail_service(credentials_path, token_path)`:
  - Loads client configuration from `secrets/credentials.json`.
  - Reads cached token from `secrets/token.json` if available.
  - Automatically refreshes expired access tokens via refresh token.
  - Falls back to `InstalledAppFlow.from_client_secrets_file` for initial consent with scope `https://www.googleapis.com/auth/gmail.readonly`.
- [ ] **Task 2.3:** Implement `fetch_message_summaries(service, query, max_results)` using `users.messages.list`.
- [ ] **Task 2.4:** Implement `fetch_message_full(service, message_id)` using `users.messages.get(format='full')`.
- [ ] **Task 2.5:** Implement `fetch_attachment_bytes(service, message_id, attachment_id)` decoding base64URL data.

---

### Phase 3: Filtering & Whitelist Engine
- [ ] **Task 3.1:** Create `src/gmail_filter.py`.
- [ ] **Task 3.2:** Implement sender verification:
  - Sender email domain must match `@rvce.edu.in`, `@rvei.edu.in`, or `@vtu.ac.in`.
  - Validate against known administrative accounts (`principal@`, `coe@`, `accounts@`, `placement@`, `dean.*@`, `hod.*@`).
- [ ] **Task 3.3:** Implement negative exclusion filters:
  - Discard conversational threads (subject starting with `Re:` or `Fwd:` without official circular tags).
  - Discard automated notifications, OTPs, password resets, webinar invites, marketing, and social alerts.
- [ ] **Task 3.4:** Implement notice relevancy triggers:
  - Require subject or snippet match on institutional notice terms: `circular`, `notification`, `schedule`, `timetable`, `deadline`, `cie`, `see`, `fee`, `revaluation`, `placement`, `concession`, `grade card`.
- [ ] **Task 3.5:** Return boolean `is_relevant_notice(msg_payload)` with transparent rejection reasons for audit logs.

---

### Phase 4: Extraction & Normalization
- [ ] **Task 4.1:** Create `src/gmail_parser.py`.
- [ ] **Task 4.2:** Email Body Extraction:
  - Traverse MIME payload for `text/html` (or fallback `text/plain`).
  - Pass HTML through existing [`HTMLExtractor.extract()`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/src/extractors.py#L62) to strip styles, tracking tags, and boilerplate signatures.
  - Normalize text with [`clean_text()`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/src/cleaning.py#L83).
- [ ] **Task 4.3:** PDF Attachment Extraction:
  - Filter attachments for MIME `application/pdf` or filename ending in `.pdf`.
  - Pass raw bytes directly to existing [`PDFExtractor.extract()`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/src/extractors.py#L220).
  - Automatically triggers [`ocr_pdf_bytes()`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/src/ocr.py#L167) when native text is under 50 characters.
- [ ] **Task 4.4:** Assign verifiable topics:
  - Apply existing [`derive_topic()`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/src/cleaning.py#L121) using email subject, body headings, and circular text.

---

### Phase 5: Deduplication, Revision Tracking & Corpus Persistence
- [ ] **Task 5.1:** Implement exact deduplication:
  - Check Gmail `message_id` and RFC 822 `Message-ID` against existing document URLs in [`data/raw/collection_manifest.json`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/data/raw/collection_manifest.json).
  - Check SHA-256 hash of attachment bytes to skip identical re-sent attachments.
- [ ] **Task 5.2:** Implement safe revision linking:
  - Extract circular numbers via regex: `r"(?:circular\s+no\.?|vide\s+no\.?|ref(?:\.|\s+no\.?):?)\s*([A-Za-z0-9/_-]+)"`.
  - Extract continuation dates via regex: `r"in\s+(?:continuation|supersession)\s+to\s+.*?(?:circular\s+dated|dated)\s*([0-9\.\/-]+)"`.
  - If and **only if** an explicit reference to an earlier document ID or circular number matches:
    - Set new record `notice_governance.supersedes_document_id = <prior_id>`.
    - Update prior record in `data/processed/` with `notice_governance.validity_status = "SUPERSEDED"` and `superseded_by_document_id = <new_id>`.
  - If no explicit citation is present: leave earlier records as `"ACTIVE"` and mark new record as `"ACTIVE"`.
- [ ] **Task 5.3:** Corpus Persistence:
  - Append raw records to [`data/raw/campus_documents.jsonl`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/data/raw/campus_documents.jsonl).
  - Append cleaned records with extended metadata to [`data/processed/campus_documents.jsonl`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/data/processed/campus_documents.jsonl).
  - Update [`data/raw/collection_manifest.json`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/data/raw/collection_manifest.json) with new provenance entries.

---

### Phase 6: Manual Ingestion CLI
- [ ] **Task 6.1:** Create entry point `src/gmail_ingestion.py`.
- [ ] **Task 6.2:** Implement CLI arguments:
  - `--query`: Custom Gmail search filter (default: `from:(*@rvce.edu.in OR *@vtu.ac.in) -label:SPAM`).
  - `--days`: Limit lookback window (default: 7).
  - `--dry-run`: Parse and display candidate notices and attachments without writing to disk.
  - `--reindex`: Automatically trigger retrieval re-indexing upon completion.
- [ ] **Task 6.3:** Log ingestion summary statistics to console and append to execution report.

---

### Phase 7: Retrieval Integration & Historical Query Preservation
- [ ] **Task 7.1:** Update [`src/retrieval.py`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/src/retrieval.py):
  - Propagate `validity_status` and `superseded_by` in `SearchResult`.
  - In `HybridRetriever.retrieve()`:
    - Default ranking: prioritize `validity_status == "ACTIVE"`.
    - If a superseded document is returned for a query, annotate it with a warning metadata flag `is_superseded: True`.
  - Guarantee historical notices remain fully retrievable when queries request prior dates or circular references.
- [ ] **Task 7.2:** Add `reindex()` method to `TFIDFRetriever` and `SemanticRetriever` to reload updated JSONL datasets in memory without restarting Python processes.

---

### Phase 8: Unit Testing & Verification Suite
- [ ] **Task 8.1:** Create `tests/test_gmail_ingestion.py`.
- [ ] **Task 8.2:** Test sender filtering & negative patterns using mocked email headers (no network).
- [ ] **Task 8.3:** Test email body HTML parsing and boilerplate stripping using existing `HTMLExtractor`.
- [ ] **Task 8.4:** Test PDF attachment parsing and OCR dispatch using synthetic base64 payloads.
- [ ] **Task 8.5:** Test duplicate detection (duplicate message ID, duplicate attachment hash).
- [ ] **Task 8.6:** Test safe circular revision handling:
  - Verify explicit citation updates `validity_status` to `"SUPERSEDED"`.
  - Verify keyword-only similarity does **not** mutate prior records.
  - Verify historical notices remain retrievable.
- [ ] **Task 8.7:** Verify full test suite passes (`pytest tests/`) without regression against existing 54 tests.

---

## 3. The Single First Implementation Task

To begin execution safely and establish the baseline environment without modifying any existing production pipelines:

### **Task 1: Secret Protection and Dependency Configuration**
- **Action 1:** Add `secrets/`, `*credentials*.json`, and `*token*.json` to `.gitignore` and `.dvcignore`.
- **Action 2:** Create the local directory `secrets/` to house future OAuth tokens securely.
- **Action 3:** Append Google API client libraries (`google-api-python-client`, `google-auth-httplib2`, `google-auth-oauthlib`) and `rapidocr-onnxruntime` to `requirements.txt`.
- **Validation:** Run `git status` to verify secrets remain untracked, and verify pip dependencies install cleanly.

