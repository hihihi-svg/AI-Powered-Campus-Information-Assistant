# Retrieval Baseline — Smoke Test Report

**Date:** October 12, 2026  
**Dataset:** `data/processed/campus_documents.jsonl` (60 records — 45 non-empty HTML, 15 empty scanned PDFs)  
**Retriever:** `TFIDFRetriever` in `src/retrieval.py`

---

## 1. Pre-Test State

| Item | Value |
|---|---|
| Total records in dataset | 60 |
| Records with non-empty content (indexable) | 45 |
| Empty scanned PDF records (correctly excluded) | 15 |
| Topics represented in non-empty records | 7 |
| Topics with zero non-empty records | `facilities_hostels`, `student_services` |
| Test suite status before smoke test | **10 / 10 passed** |

---

## 2. Smoke Test Questions

| ID | Intent | Question |
|---|---|---|
| Q-ADM-1 | admissions | What is the COMEDK rank cutoff for Computer Science at RVCE? |
| Q-ADM-2 | admissions | How do I apply for lateral entry through DCET? |
| Q-FEE-1 | fees_finance | What is the tuition fee for B.E. first year at RVCE? |
| Q-FEE-2 | fees_finance | What is the late fee penalty for delayed tuition payment? |
| Q-EXA-1 | examinations_regulations | What is the minimum attendance required for semester exams? |
| Q-EXA-2 | examinations_regulations | How can I apply for revaluation of my exam paper? |
| Q-DEP-1 | departments_faculty | Who is the head of the Computer Science department? |
| Q-PLA-1 | placements_careers | What companies recruited from RVCE in the last placement season? |
| Q-FAC-1 | facilities_hostels | Is there a hostel facility available for girls at RVCE? |
| Q-GEN-1 | general_college_info | What is the address and contact number of RVCE? |

---

## 3. Bugs Discovered and Fixed

Three concrete bugs were identified by reading actual retrieved chunk text and fixed in `src/retrieval.py`. All 10 unit tests continued to pass.

### Bug 1 — Incomplete `BOILERPLATE_HEADINGS` set

**Symptom:** Q-FEE-1 top-1 score 0.44, Q-FEE-2 top-1 score 0.42 — for this 48-char sidebar navigation stub:
```
heading: "Academic Student Services"
text:    "Fee & Notifications\nScheme & Syllabus\nPlacements"
```
This block appears on dozens of RVCE department pages. Also surfacing as high-scoring results:
- `"Address:"` → footer widget (`RV College of Engineering, RV Vidyanikethan Post, Mysuru Road, Bengaluru – 560 059`)
- `"Principal's Office"` → right-sidebar card (`Dr. K.N. Subramanya Principal RV College of Engineering`)

**Fix:** Added 11 RVCE site-wide widget headings to `BOILERPLATE_HEADINGS`:
```python
"academic student services", "address:", "address",
"principal\u2019s office", "principal's office",
"contact us", "about rvce", "about college",
"key executives", "statutory committees", "anti ragging", "faq"
```

### Bug 2 — Navigation-list stubs passing the content guard

**Symptom:** Blocks where every line is a short link label (≤ 30 chars) were being indexed as content.

**Fix:** Added guard after the text-content filter:
```python
non_empty_lines = [ln for ln in cleaned_block.splitlines() if ln.strip()]
if non_empty_lines and all(len(ln.strip()) <= 30 for ln in non_empty_lines):
    continue
```

### Bug 3 — `len(heading) > 15` fallback admitted sub-threshold sidebar chunks

**Symptom:** The condition `len(cleaned_block) >= min_chunk_chars or len(heading) > 15` kept the 48-char "Fee & Notifications..." sidebar chunk alive because `len("Academic Student Services") = 25 > 15`.

**Fix:** Removed the `or len(heading) > 15` branch. All chunks shorter than `min_chunk_chars` (60) are now unconditionally discarded.

---

## 4. Per-Question Results (Post-Fix)

### Q-ADM-1 — COMEDK rank cutoff for CS
**Verdict: ❌ NOT RELEVANT**

Top-1: ECE Research page (score 0.2051) — latches on token `"computer science"` in boilerplate description. Corpus contains no rank cutoff data; all admissions PDFs with rank tables are scanned with empty text.

---

### Q-ADM-2 — Lateral entry through DCET
**Verdict: ✅ TOPICALLY RELEVANT**

Top-1: Admission Circulars (score 0.1249, heading: "Admission Details 2026-27"):
> "DCET: Procedure and Balance Fees Details for BE (Lateral Entry) 2026-27…"

Only one result returned. The score is low because the document is a list of circular titles; the actual procedure is in a scanned PDF not yet OCR'd.

---

### Q-FEE-1 — Tuition fee for B.E. first year
**Verdict: ❌ NOT RELEVANT (corpus gap)**

Pre-fix top-1 was sidebar boilerplate (score 0.44). Post-fix top-1 is "Academic Circular" (score 0.1592, heading: "B.E. first year") — mentions academic circulars and timetables, not fee amounts. All 9 `fees_finance` PDFs are scanned and empty; the fee circulars HTML page extracts as heading-only (all circular titles are `##` headings with no body text under them).

---

### Q-FEE-2 — Late fee penalty
**Verdict: ❌ NOT RELEVANT (corpus gap)**

Pre-fix top-1 was sidebar boilerplate (score 0.42). Post-fix top-1 is homepage admissions section (score 0.1189) — mentions "fee payments" in a navigation description. No late-fee penalty rule exists in the corpus.

---

### Q-EXA-1 — Minimum attendance for semester exams
**Verdict: ❌ NOT RELEVANT (corpus gap)**

Pre-fix top-1 was a placement salary table — "Minimum X Maximum X Average X" repeated 10× — scoring 0.16 because `minimum` matched the salary field names. Post-fix correctly suppresses those salary-table stubs; top-1 is now the homepage "Examinations" navigation section (score 0.10) which mentions exam circulars but not attendance percentages. The attendance rule exists only in scanned regulation PDFs.

---

### Q-EXA-2 — Revaluation procedure
**Verdict: ⚠️ WEAKLY TOPICAL**

Top-1: Homepage "Examinations" section (score 0.1115):
> "Check for official circulars related to examination timetables, registration processes, results announcements, **revaluation** dates and supplementary exams."

Top-2: Examination Circulars page (score 0.0715), heading "Archived Examination Circulars":
> "Schedule for SEE Answer Paper Viewing (PVP) of PG – M.Tech & MCA…"

Results are topically adjacent (mention revaluation) but do not provide the actual procedure.

---

### Q-DEP-1 — Head of Computer Science department
**Verdict: ❌ NOT RELEVANT**

Top-1 through top-3: ECE Research page (scores 0.25–0.25). The department mentions "computer science" in boilerplate research descriptions. The corpus has no faculty directory page. Results #2 and #3 are **identical chunks from the same document** — the same "Publications" passage appears twice in the index (see Section 5, remaining issue #1).

---

### Q-PLA-1 — Companies that recruited from RVCE
**Verdict: ⚠️ WEAKLY TOPICAL / CORPUS GAP**

Pre-fix top-1 was sidebar boilerplate. Post-fix top-1 is Chemical Engineering Placement page (score 0.1944):
> "RVCE's commitment to career success. Our students are placed in the world's most innovative companies…"

Top-2: CHE placement statistics (score 0.1153):
> "2024-25 Number of companies visited – 23 Number of offers made – 28…"

The corpus contains numeric placement statistics but no named company lists. Company names are in PDFs or image embeds.

---

### Q-FAC-1 — Hostel for girls
**Verdict: ❌ NOT RELEVANT (corpus gap)**

Single result: Placement & Training page (score 0.0671), heading "Infrastructure & Facilities" — describes the placement office, not student hostels. Zero hostel-related documents in corpus.

---

### Q-GEN-1 — Address and contact number of RVCE
**Verdict: ❌ NOT RELEVANT**

Pre-fix top-1 was the footer address widget (boilerplate sidebar, score 0.12). Post-fix correctly suppresses it, but the address is now unretrievable — the corpus has no dedicated contact page. Top results are placement statistics pages that happen to contain "number of".

---

## 5. Summary Table

| ID | Intent | Verdict | Pre-fix Issue Eliminated? | Root Cause |
|---|---|---|---|---|
| Q-ADM-1 | admissions | ❌ Not relevant | N/A | No rank cutoff data in corpus |
| Q-ADM-2 | admissions | ✅ Topically relevant | N/A | Scanned PDF has procedure details |
| Q-FEE-1 | fees_finance | ❌ Not relevant | ✅ Sidebar boilerplate (0.44→0.16) | Fee amounts only in scanned PDFs |
| Q-FEE-2 | fees_finance | ❌ Not relevant | ✅ Sidebar boilerplate (0.42→0.12) | Fee penalty only in scanned PDFs |
| Q-EXA-1 | exams | ❌ Not relevant | ✅ Salary-table noise (0.16→0.10) | Attendance rules in scanned PDF |
| Q-EXA-2 | exams | ⚠️ Weakly topical | N/A | Revaluation procedure not in corpus |
| Q-DEP-1 | departments | ❌ Not relevant | N/A | No faculty directory in corpus |
| Q-PLA-1 | placements | ⚠️ Weakly topical | ✅ Sidebar boilerplate removed | Company names not in indexed text |
| Q-FAC-1 | facilities | ❌ Not relevant | N/A | Zero hostel documents in corpus |
| Q-GEN-1 | general | ❌ Not relevant | ✅ Address sidebar removed | No contact page in corpus |

**Overall: 1 topically relevant, 2 weakly topical, 7 not relevant.**

---

## 6. Remaining Issues

### Remaining Code Issue (not yet fixed)

| # | Issue | Description |
|---|---|---|
| 1 | Duplicate chunk pairs in results | `retrieve()` returns two chunks with identical text from the same document heading (Q-DEP-1: "Publications" from ECE research page appeared at rank #2 and #3 with identical scores and text). Fix: deduplicate on `(document_id, heading, text[:50])` before truncating to `top_k`. |

### Corpus Gaps (not code bugs — require data work)

| # | Gap | Affected Questions | Remedy |
|---|---|---|---|
| 1 | Fee amounts and penalty rules | Q-FEE-1, Q-FEE-2 | OCR the 9 scanned fee PDFs |
| 2 | Attendance and exam regulation rules | Q-EXA-1 | OCR the 6 scanned regulation PDFs |
| 3 | Hostel and student accommodation | Q-FAC-1 | Crawl hostel pages |
| 4 | Faculty directory and HOD contacts | Q-DEP-1 | Crawl department faculty pages |
| 5 | Contact page (address, phone) | Q-GEN-1 | Crawl `rvce.edu.in/contact/` |
| 6 | Named company placement lists | Q-PLA-1 | OCR placement PDFs or crawl recruiter pages |
| 7 | COMEDK rank cutoff tables | Q-ADM-1 | External data (not on rvce.edu.in); document as out-of-scope |

---

## 7. Files Changed

| File | Change |
|---|---|
| `src/retrieval.py` | Fixed 3 boilerplate-filtering bugs |
| `scratch/smoke_test.py` | New: smoke test runner |
| `scratch/debug_chunks.py` | New: chunk inspection debug script |
| `scratch/inspect_dataset.py` | New: dataset inspection script |
| `reports/smoke_test_raw.json` | New: raw JSON results (post-fix) |
| `reports/retrieval_smoke_test.md` | New: this report |

**Unchanged:** `tests/test_retrieval.py`, `src/crawler.py`, `src/extractors.py`, `src/cleaning.py`, `src/validation.py`, `src/build_dataset.py`, `data/processed/campus_documents.jsonl`.

---

## 8. Single Next Recommended Task

> [!IMPORTANT]
> **Next task A (code — 15 minutes):** Add result deduplication in `TFIDFRetriever.retrieve()`.  
> Deduplicate returned results on `(document_id, heading)` before truncating to `top_k`.  
> Add a regression test in `tests/test_retrieval.py` for the case where chunking produces two identical sub-chunks.
>
> **Next task B (data — higher impact):** Implement OCR for the 15 scanned PDFs in `data/raw/`.  
> Create `src/ocr.py` with `pytesseract` (or `easyocr` as fallback), integrate into `src/extractors.py`,  
> re-run extraction on the 15 empty records, and update `data/processed/campus_documents.jsonl`.  
> This would immediately unlock `fees_finance` (9 PDFs) and exam regulation content.
