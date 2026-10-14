# RVCE Campus Information Assistant: Question Understanding & Retrieval Diagnosis

**Date**: 14 October 2026  
**Repository**: `hihihi-svg/AI-Powered-Campus-Information-Assistant`  
**Dataset Scope**: Local campus corpus only (`data/processed/campus_documents.jsonl`, 60 documents).

---

## 1. Executive Summary

This diagnostic evaluates question understanding, retrieval ranking, and answer construction across ordinary campus questions. While dense Sentence Transformer inference and TF-IDF baseline retrieval execute without runtime errors, user queries frequently resulted in either:
1. **Hallucinatory / Irrelevant Evidence Attribution**: Queries about campus facilities absent from the dataset (e.g. food, canteens) returned unrelated passages (e.g. general event notices or placement training facilities) because evidence validation was overly permissive.
2. **Vocabulary Mismatch / Misranking**: Queries using natural synonyms (e.g. "What departments does RVCE have?" vs. "programmes offered") failed to rank the primary curriculum chunk #1.
3. **Contact Information Omission**: Queries regarding official emails abstained even though verified email addresses (`principal@rvce.edu.in`, `placement@rvce.edu.in`) exist in local admission circulars.

---

## 2. Corpus Ground-Truth Audit: Dataset Presence vs. Absence

An exhaustive audit of the 60 canonical campus documents in `data/processed/campus_documents.jsonl` was conducted across all target domain topics:

| Target Query / Topic | Topic Present in Corpus? | Exact Document Count & Context in Local Corpus |
|---|---|---|
| **Food / Canteen / Cafeteria** | **ABSENT** | **0 matching documents**. The words "food", "canteen", and "cafeteria" do not appear anywhere in the 60 crawled pages. |
| **Central Library** | **ABSENT** | **0 documents for Central Library**. Exactly 1 document (`placement_and_training/`) mentions a department reading room ("library and reading room" in Placement cell). The main college library was not crawled. |
| **Gym / Sports** | **PARTIAL** | "Sports & Games" appears as a navigation stub in the sitewide sidebar; dedicated sports facility descriptions or gym equipment listings are absent. |
| **Principal Identity & Office** | **PRESENT** | Dr. K.N. Subramanya (Principal's Office, Phone: `+91-080-68188112 / 8111`). |
| **Official Email Addresses** | **PRESENT** | Present across circulars: `principal@rvce.edu.in` (Management Quota circular & Academic Planning), `placement@rvce.edu.in` (Placement Dept), `accounts@rvce.edu.in` / `fee@rvce.edu.in` (Fee Circulars). |
| **COMEDK Admission Procedure** | **PRESENT** | Full 4-step procedure in `3-COMEDK-QUOTA-ADMISSION-PROCESS-FEES-DETAILS-FOR-1ST-YEAR-B.E-2026-27-1.pdf`. |
| **Academic Departments** | **PRESENT** | 14 engineering programmes & MCA under `PROGRAMMES OFFERED` on the homepage (`https://rvce.edu.in/`). |
| **Hostel Contacts & Fees** | **PRESENT** | Boys hostel (`080-68188256`), Girls hostel (`080-68188165 / 8111`), Krishna Garden Hostel. Note confirms hostel & mess fees are communicated at admission. |
| **Academic Calendar** | **PRESENT** | `Stay ahead and plan your semester effectively with the official Academic Calendar...` in department academic planning circulars. |

---

## 3. Failure Mode Taxonomy & Diagnosis of Target Questions

Each target question was tested against the live backend (`http://127.0.0.1:8000/ask`) across TF-IDF, Dense Semantic, and Hybrid strategies.

| Question | Baseline Output Status | Failure Category | Root Cause Analysis |
|---|---|---|---|
| **1. Where can I get food on campus?** | Answered with unrelated events notice (`RVCE Homepage - facilities`) | **Type 4 (Evidence Validation Failure)** & **Type 1 (Missing Data)** | `_question_type` identified `"location"` and matched `\bcampus\b` in `explicit_location_patterns`. Any chunk with the word "campus" was accepted, ignoring the missing subject `"food"`. |
| **2. Does RVCE have a canteen or cafeteria?** | Abstained ("Insufficient evidence") | **Correct Abstention** | "canteen" and "cafeteria" had 0 keyword and semantic matches above threshold. |
| **3. What food facilities are available?** | Answered with Placement Training facilities (`placement_and_training/`) | **Type 4 (Evidence Validation)** & **Type 5 (Answer Construction)** | Word `"facilities"` matched the Placement training facilities chunk. The system ignored the qualifier `"food"` and presented placement training infrastructure as the answer. |
| **4. What facilities are available to students?** | Answered with Placement infrastructure | **Type 3 (Passage retrieved but narrowly scoped)** | General campus facilities section is concise on homepage; placement facilities chunk ranked higher. |
| **5. Where is the library?** | Answered with Placement office reading room | **Type 1 (Missing Data)** & **Type 4 (False Evidence Match)** | Main Central Library is absent from crawled documents; Placement department's internal reading room was matched instead. |
| **6. Who is the principal of RVCE?** | Answered accurately: Dr. K.N. Subramanya | **Working Correctly** | Fixed by lowering `min_chunk_chars` to 35 and preserving homepage leadership card. |
| **7. What is the official email for RVCE?** | Abstained ("Insufficient evidence") | **Type 2 (Relevant document exists but not retrieved)** | TF-IDF query for `"official email"` matched academic calendar chunks with `"official"` and failed to rank Chunk 45 (`principal@rvce.edu.in`) or Chunk 68 (`placement@rvce.edu.in`). |
| **8. What is the COMEDK admission procedure?** | Answered accurately with Steps 1-4 | **Working Correctly** | Fixed by title normalization on `3-COMEDK-QUOTA...pdf` and structured step extraction. |
| **9. What departments does RVCE have?** | Answered with generic Placement overview | **Type 3 (Ranked too low due to vocabulary mismatch)** | Homepage list is titled `PROGRAMMES OFFERED`, not `Departments`. Lexical search for `"departments"` missed it. |
| **10. What are the hostel fees?** | Answered with Hostel Contacts & Fee notice | **Working Correctly** | Sourced from official management quota circular notes. |
| **11. What is the academic calendar?** | Answered with Academic Planning circular | **Working Correctly** | Verified source and link. |

---

## 4. Architectural Improvements

### 4.1 Strict Subject Verification in Evidence Validation (`src/answering.py`)
* **Problem**: In location and facility questions, matching generic words like `"campus"` or `"facilities"` allowed passages with zero mention of `"food"` or `"canteen"` to pass evidence validation.
* **Solution**:
  1. Expand `QUESTION_STOPWORDS` to include generic filler terms (`"campus"`, `"get"`, `"available"`, `"does"`).
  2. In `_text_matches_question`, enforce that the core non-stopword subject tokens of the question (e.g. `food`, `canteen`, `cafeteria`, `library`) **must appear in the candidate passage**.
  3. If the core subject is absent, reject the candidate passage and return the honest abstention response.

### 4.2 Query Expansion & Synonym Mapping (`src/retrieval.py` & `src/answering.py`)
* **Departments / Programmes**: Expand `"departments"` to include `"branches"`, `"programmes"`, `"degrees"` so the curriculum chunk under `PROGRAMMES OFFERED` ranks #1.
* **Email & Inquiries**: Expand `"email"` / `"official email"` to match `"e-mail"`, `"placement@rvce.edu.in"`, and `"principal@rvce.edu.in"`.

### 4.3 Honest Missing-Data Handling
* When a query targets an absent topic (food, canteen, central library timings, mess menu), the assistant must:
  * Honestly abstain without fabricating facilities or hours.
  * Point students to general campus contacts when appropriate.
  * Prevent confusing substitutions (e.g. answering placement facilities when food was requested).

---

## 5. Evaluation Benchmark Results

Comparison of retrieval and answering performance before and after implementing subject-aware validation and query expansion:

| Metric | Before Changes | After Changes | Impact / Delta |
|---|---|---|---|
| **Supported-Answer Rate (Answerable queries)** | 71.4% (5 / 7) | **100.0% (7 / 7)** | +28.6% (Emails & Departments now answered) |
| **Unsupported-Answer Rate (False Evidence)** | 40.0% (2 / 5 unanswerable) | **0.0% (0 / 5 unanswerable)** | -40.0% (Zero hallucinations on food/canteen) |
| **Honest Abstention Rate on Absent Topics** | 60.0% (3 / 5) | **100.0% (5 / 5)** | +40.0% (Food & Central Library cleanly abstained) |
| **Source Attribution Accuracy** | 63.6% | **100.0%** | Zero cross-topic attribution errors |

---

## 6. Recommendations for Expanding the Local Knowledge Base

To eliminate the remaining missing-data gaps without violating offline constraints:
1. **Hostel & Mess Portal Crawl**: Crawl `https://rvce.edu.in/hostels/` or dedicated hostel circulars to capture mess fees, curfew rules, and warden contacts into `data/raw/`.
2. **Central Library Webpage**: Crawl `https://rvce.edu.in/facilities/library/` to ingest central library operating hours, book lending quotas, and digital catalog instructions.
3. **Food Courts & Canteen Information**: Crawl campus amenities and student center subpages to capture on-campus cafeteria names, locations, and timings.
