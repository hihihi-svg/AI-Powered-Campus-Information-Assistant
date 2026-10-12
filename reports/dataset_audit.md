# Campus AI Assistant — Dataset Audit & Pipeline Quality Report

**Audit Date:** October 12, 2026  
**Audited Datasets:**
- Cleaned Processed Dataset: [`data/processed/campus_documents.jsonl`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/data/processed/campus_documents.jsonl)
- QA Report: [`reports/data_collection_report.json`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/reports/data_collection_report.json)
- Provenance Manifest: [`data/raw/collection_manifest.json`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/data/raw/collection_manifest.json)
- Raw Dataset: [`data/raw/campus_documents.jsonl`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/data/raw/campus_documents.jsonl)

---

## 1. Executive Summary

An in-depth, rigorous audit was performed on the existing real-data acquisition pipeline output without initiating any new network crawls. The current collection consists of **60 real documents** collected directly from `https://rvce.edu.in/`.

### Key Audit Findings:
1. **Schema Compliance:** 100% of the 60 records strictly conform to the defined schema without missing keys or malformed datatypes.
2. **Text Extraction Disparity:**
   - **HTML (45 pages):** 100% successfully extracted, with high quality, structured Markdown formatting, preserved headings, and clean boilerplate removal.
   - **PDF (15 documents):** 100% returned empty text extractions (`extraction_status: EMPTY`) because every single collected PDF is a **scanned physical notice** containing raster page images with **0 embedded text streams**.
3. **Severe Topic Imbalance & Blind Spots:**
   - `departments` accounts for **51.7%** of all records.
   - All 9 records in `fees_finance` and 9 out of 10 records in `admissions` are scanned PDFs with 0 extracted text.
   - High-value student life topics (`hostels`, `facilities`, `student_services`) currently have **0 documents** in the dataset.
4. **Duplication Patterns:**
   - **3 Exact Duplicates:** Identified as WordPress URL routing aliases (e.g., duplicate paths pointing to identical content).
   - **17 Near-Duplicates ($J \ge 0.85$):** Caused by inter-departmental shared boilerplate (e.g., identical academic planning stubs replicated across engineering departments).
5. **Supervised ML Feasibility:** In its current state, this dataset **cannot support a robust supervised document-topic classification model** due to zero-text classes, extreme class imbalance, and significant data leakage risks.

---

## 2. Dataset Metrics & Statistical Distributions

### 2.1 Record Counts & Format Breakdown
- **Total Processed Records:** 60
- **HTML Web Pages:** 45 (75.0%)
- **PDF Documents:** 15 (25.0%)

### 2.2 Record Schema & Types
Every record adheres strictly to the target schema:

| Field | Type | Non-Null Count | Null Count | Validation Status |
|---|---|---|---|---|
| `document_id` | `str` | 60 | 0 | Valid (deterministic hash format) |
| `source_url` | `str` | 60 | 0 | Valid (canonical HTTP/HTTPS URLs) |
| `title` | `str \| null` | 60 | 0 | Valid |
| `content` | `str` | 60 (45 non-empty, 15 empty) | 0 | Valid |
| `source_type` | `str` | 60 (`HTML` or `PDF`) | 0 | Valid |
| `collected_at` | `str` | 60 | 0 | Valid (ISO 8601 UTC) |
| `last_modified` | `str \| null` | 60 | 0 | Valid (HTTP headers / meta tags) |
| `topic` | `str \| null` | 59 | 1 (`None` for homepage) | Valid |
| `extraction_status` | `str` | 60 (`SUCCESS` or `EMPTY`) | 0 | Valid |

### 2.3 Extraction Status Distribution
- `SUCCESS`: **45 records** (75.0%) — All HTML documents
- `EMPTY`: **15 records** (25.0%) — All PDF documents
- `PARTIAL`: **0 records** (0.0%)
- `FAILED`: **0 records** (0.0%)

### 2.4 Topic Distribution (Across All 60 Records)

| Topic Label | Total Records | % of Dataset | HTML (Success) | PDF (Empty Text) |
|---|---|---|---|---|
| `departments` | 31 | 51.7% | 31 | 0 |
| `admissions` | 10 | 16.7% | 1 | 9 |
| `fees_finance` | 9 | 15.0% | 0 | 9 |
| `examinations_regulations` | 6 | 10.0% | 6 | 0 |
| `academics` | 2 | 3.3% | 2 | 0 |
| `placements_careers` | 1 | 1.7% | 1 | 0 |
| `None` (Unclassified Hub) | 1 | 1.7% | 1 | 0 |
| `hostels` | 0 | 0.0% | 0 | 0 |
| `facilities_infrastructure` | 0 | 0.0% | 0 | 0 |
| `student_services_clubs` | 0 | 0.0% | 0 | 0 |
| **Total** | **60** | **100.0%** | **45** | **15** |

### 2.5 Content Length & Word Count Statistics

| Metric | All Records ($N=60$) | Non-Empty Records ($N=45$) |
|---|---|---|
| **Min Characters** | 0 | 1,247 |
| **Max Characters** | 21,874 | 21,874 |
| **Mean Characters** | 3,144.57 | 4,192.76 |
| **Median Characters** | 2,032.50 | 3,319.00 |
| **Min Words** | 0 | 186 |
| **Max Words** | 3,132 | 3,132 |
| **Mean Words** | 475.28 | 633.71 |
| **Median Words** | 329.50 | 461.00 |

---

## 3. Representative Records Per Topic

Below are representative records demonstrating actual title, URL, source type, extraction status, and content excerpts:

### 3.1 `departments` (31 records total)
- **Record 1:**
  - **Document ID:** `doc_html_737e30d70b04bc7a`
  - **Title:** `RVCE Bengaluru | AI & ML Dept Placement Success Stories`
  - **Source URL:** [https://rvce.edu.in/department/ai_ml/al_ml_placement/](https://rvce.edu.in/department/ai_ml/al_ml_placement/)
  - **Source Type / Status:** HTML / `SUCCESS`
  - **Content Excerpt:**  
    > `## PLACEMENT Statistics OF AI-ML  The Department of Artificial Intelligence and Machine Learning follows a project-based curriculum complemented by personality development programmes to prepare students for dynamic industry needs.  ### Key Highlights  - Comprehensive placement training from 5th semester onwards  - Hands-on hackathons and capstone projects with industry mentorship...`
- **Record 2:**
  - **Document ID:** `doc_html_7d07c0b0a88ef11a`
  - **Title:** `DEPARTMENT OF CIVIL ENGINEERING`
  - **Source URL:** [https://rvce.edu.in/department/civil_engineering/department-of-civil-engineering/](https://rvce.edu.in/department/civil_engineering/department-of-civil-engineering/)
  - **Source Type / Status:** HTML / `SUCCESS`
  - **Content Excerpt:**  
    > `## DEPARTMENT OF CIVIL ENGINEERING  Civil Engineering is the oldest engineering discipline that deals with the design, construction, and maintenance of the physical and naturally built environment, including works such as bridges, roads, canals, dams, and buildings...`

### 3.2 `examinations_regulations` (6 records total)
- **Record 1:**
  - **Document ID:** `doc_html_68feab075a35ae79`
  - **Title:** `Academic Circular`
  - **Source URL:** [https://rvce.edu.in/academic-circular/](https://rvce.edu.in/academic-circular/)
  - **Source Type / Status:** HTML / `SUCCESS`
  - **Content Excerpt:**  
    > `# Academic Circular  ## B.E. first year  Here, you’ll find all foundational academic circulars, updates on VTU/Autonomous schemes, first-semester examination schedules, timetables, induction programme details, and essential academic guidelines...`
- **Record 2:**
  - **Document ID:** `doc_html_32fd10141b0a842b`
  - **Title:** `Fee payment circulars` *(Note: Heuristically labeled under examinations due to `/academics_and_examinations/` path)*
  - **Source URL:** [https://rvce.edu.in/academics_and_examinations/fee_payment_circulars/](https://rvce.edu.in/academics_and_examinations/fee_payment_circulars/)
  - **Source Type / Status:** HTML / `SUCCESS`
  - **Content Excerpt:**  
    > `## Fee Circulars  ## 2026-27  ## 28.09.2026 / 925 : Revised Fee Circulars 2026-27 - Higher semester PG Students  ## 28.09.2026 / 921 : Fee Circulars 2026-27 - Higher semester UG Students  ## 09.09.2026 / 798 : Fee Circulars 2026-27 - 2nd year to 4th year...`

### 3.3 `admissions` (10 records total: 1 HTML, 9 PDF)
- **Record 1 (HTML):**
  - **Document ID:** `doc_html_299187313000b213`
  - **Title:** `Admission Circulars`
  - **Source URL:** [https://rvce.edu.in/admission-circulars/](https://rvce.edu.in/admission-circulars/)
  - **Source Type / Status:** HTML / `SUCCESS`
  - **Content Excerpt:**  
    > `## Admission Circulars  ## Admission Details 2026-27  - PGCET & Management: M.Tech & MCA : Documents and Balance Fees Details for Admission 2026-2027  - DCET: Procedure and Balance Fees Details for BE (Lateral Entry) 2026-27  - COMED-K: Balance Fees Details for B.E. 2026-27...`
- **Record 2 (PDF - Scanned):**
  - **Document ID:** `doc_pdf_c1b2c554907a996f`
  - **Title:** `1St Year Be Program Management Quota Admission 2026 27`
  - **Source URL:** [https://rvce.edu.in/wp-content/uploads/2026/03/1ST-YEAR-BE-PROGRAM-MANAGEMENT-QUOTA-ADMISSION-2026-27.pdf](https://rvce.edu.in/wp-content/uploads/2026/03/1ST-YEAR-BE-PROGRAM-MANAGEMENT-QUOTA-ADMISSION-2026-27.pdf)
  - **Source Type / Status:** PDF / `EMPTY`
  - **Content Excerpt:** `[NO TEXT CONTENT — SCANNED CIRCULAR IMAGE]`

### 3.4 `fees_finance` (9 records total: all 9 are PDFs)
- **Record 1 (PDF - Scanned):**
  - **Document ID:** `doc_pdf_4eba733b0af1d307`
  - **Title:** `Revised Pg Higher Semester Fees Circular 29.09.26`
  - **Source URL:** [https://rvce.edu.in/wp-content/uploads/2026/09/REVISED-PG-HIGHER-SEMESTER-FEES-CIRCULAR-29.09.26.pdf](https://rvce.edu.in/wp-content/uploads/2026/09/REVISED-PG-HIGHER-SEMESTER-FEES-CIRCULAR-29.09.26.pdf)
  - **Source Type / Status:** PDF / `EMPTY`
  - **Content Excerpt:** `[NO TEXT CONTENT — SCANNED CIRCULAR IMAGE]`
- **Record 2 (PDF - Scanned):**
  - **Document ID:** `doc_pdf_9bc00d1612a37895`
  - **Title:** `Fee Payment Extension Circular 28.09.26`
  - **Source URL:** [https://rvce.edu.in/wp-content/uploads/2026/09/Fee-payment-extension-circular-28.09.26.pdf](https://rvce.edu.in/wp-content/uploads/2026/09/Fee-payment-extension-circular-28.09.26.pdf)
  - **Source Type / Status:** PDF / `EMPTY`
  - **Content Excerpt:** `[NO TEXT CONTENT — SCANNED CIRCULAR IMAGE]`

### 3.5 `academics` (2 records total)
- **Record 1:**
  - **Document ID:** `doc_html_7811ef2588f98ec8`
  - **Title:** `Programmes & Faculty Details`
  - **Source URL:** [https://rvce.edu.in/programmes-faculty-details/](https://rvce.edu.in/programmes-faculty-details/)
  - **Source Type / Status:** HTML / `SUCCESS`
  - **Content Excerpt:**  
    > `## Programmes & Faculty Details  ### 1. List of Undergraduate Programmes  ### 2. List of Postgraduate Programmes  ### 3. List of UG & PG Faculty ACY 2025-26  ### 4. List of faculty handing first Year courses ACY 2025-26 ( ODD Semester)...`
- **Record 2:**
  - **Document ID:** `doc_html_4281ee85bbffcfa4`
  - **Title:** `Calendar of Events`
  - **Source URL:** [https://rvce.edu.in/calendar-of-events/](https://rvce.edu.in/calendar-of-events/)
  - **Source Type / Status:** HTML / `SUCCESS`
  - **Content Excerpt:**  
    > `## Academic Calender  Bachelor of Engineering( B.E.) 2026-2027. 04 August 2026/2080:ACADEMIC CALENDAR FOR I SEMESTER B.E. PROGRAMS 22 July 2026/2065:ACADEMIC CALENDAR FOR III SEMESTER B.E. PROGRAMS...`

### 3.6 `placements_careers` (1 record total)
- **Record 1:**
  - **Document ID:** `doc_html_a51a140d346ffcbb`
  - **Title:** `Placement Statistics`
  - **Source URL:** [https://rvce.edu.in/placement_and_training/placement_statistics/](https://rvce.edu.in/placement_and_training/placement_statistics/)
  - **Source Type / Status:** HTML / `SUCCESS`
  - **Content Excerpt:**  
    > `## Placement Statistics  ## *Placement Statistics - 2026  ## Number of companies visited  ## B.E.  ## M.Tech.  ## MCA  ## Number of offers made  ## Average salary package...`

### 3.7 `None` (Unclassified Hub - 1 record total)
- **Record 1:**
  - **Document ID:** `doc_html_b9ceb436db517957`
  - **Title:** `RV College of Engineering Bengaluru | Top Engineering Institution`
  - **Source URL:** [https://rvce.edu.in/](https://rvce.edu.in/)
  - **Source Type / Status:** HTML / `SUCCESS`
  - **Content Excerpt:**  
    > `### Notifications  # Classes for newly admitted B.E. students of the 2026 batch will commence on Monday, 24 August 2026...  ## Approvals, Affiliation, Accreditation & Ranking...`

---

## 4. Audit of Empty-Extraction Records & OCR Requirements

All 15 records with `extraction_status: EMPTY` are officially uploaded PDF documents. Detailed inspection of each PDF binary stream revealed that **every document is a multi-page scanned paper notice with embedded raster image bitmaps and zero digital font characters**.

### Exhaustive List of Empty Records & OCR Needs:

| # | Document ID | Target File / URL | Pages | Embedded Images | Native Text Chars | Needs OCR? | Content Summary |
|---|---|---|---|---|---|---|---|
| 1 | `doc_pdf_4eba733b0af1d307` | `REVISED-PG-HIGHER-SEMESTER-FEES-CIRCULAR-29.09.26.pdf` | 3 | 6 | 0 | **YES** | Revised fee schedule for PG higher semesters |
| 2 | `doc_pdf_9bc00d1612a37895` | `Fee-payment-extension-circular-28.09.26.pdf` | 1 | 2 | 0 | **YES** | Deadline extension notice for fee payments |
| 3 | `doc_pdf_3a8b4b19db6e01a8` | `PG-Higher-semester-Fee-circular_revised.pdf` | 1 | 2 | 0 | **YES** | Revision of PG semester fee amounts |
| 4 | `doc_pdf_63be9c7f9996d926` | `UG-PG-Higher-semester-Fee-circular2026-27.pdf` | 6 | 12 | 0 | **YES** | Consolidated UG & PG fee slabs for 2026-27 |
| 5 | `doc_pdf_e6e9b462da2054ff` | `UG-Higher-Semester-Fee-Circular_2025.pdf` | 13 | 25 | 0 | **YES** | Multi-branch UG fee circular breakdown |
| 6 | `doc_pdf_ba93f5383fec67c6` | `UG-higher-sem-fee-extension-circular.pdf` | 1 | 2 | 0 | **YES** | UG higher semester payment date extension |
| 7 | `doc_pdf_818035bc87b5bc60` | `NRI-quota-3rd-semester-Fee-Circular.pdf` | 10 | 20 | 0 | **YES** | NRI quota fee structure and bank accounts |
| 8 | `doc_pdf_73144a1fa8c0fe13` | `PG-3rd-sem-fee-circular-2025-26.pdf` | 13 | 26 | 0 | **YES** | PG 3rd semester fee guidelines and dates |
| 9 | `doc_pdf_55bb24c3e8e192f1` | `3-M.Tech_MCA-Procedure-and-Balance-Fees...pdf` | 2 | 3 | 0 | **YES** | Admission balance fee and document checklist |
| 10 | `doc_pdf_86f5c88e7b998cfb` | `6-DCET-ADMISSION-PROCESS-FEES-DETAILS...pdf` | 2 | 4 | 0 | **YES** | Lateral entry diploma admission instructions |
| 11 | `doc_pdf_33100236aee0e722` | `3-COMEDK-QUOTA-ADMISSION-PROCESS-FEES...pdf` | 2 | 4 | 0 | **YES** | COMEDK quota admission step-by-step process |
| 12 | `doc_pdf_125e9858348873ad` | `2-KEA-QUOTA-ADMISSION-PROCESS-FEES...pdf` | 2 | 4 | 0 | **YES** | Government quota (KEA) verification details |
| 13 | `doc_pdf_b4b32ae1593c72b9` | `4-MQ-QUOTA-ADMISSION-PROCESS-FEES...pdf` | 2 | 4 | 0 | **YES** | Management quota admission procedure |
| 14 | `doc_pdf_17db912165fb5880` | `5-NRI-QUOTA-ADMISSION-PROCESS-FEES...pdf` | 2 | 4 | 0 | **YES** | NRI quota eligibility and admission rules |
| 15 | `doc_pdf_c1b2c554907a996f` | `1ST-YEAR-BE-PROGRAM-MANAGEMENT-QUOTA...pdf` | 1 | 1 | 0 | **YES** | First year B.E. management quota circular |

### Diagnosis & Action Required:
Because standard text-layer extraction (`PyMuPDF` / `pypdf`) requires pre-existing text streams, it appropriately returned empty strings without failing. To convert these high-value official notices into accessible knowledge for the RAG engine, an **OCR stage** (using `pytesseract` or `easyocr`) must be introduced as an optional fallback for scanned image PDFs.

---

## 5. Duplicate and Near-Duplicate Analysis

The QA suite flagged **3 exact content duplicates** and **17 near-duplicate pairs** ($Jaccard \ge 0.85$). An inspection of the actual page contents revealed a critical distinction between **genuine web routing duplicates** and **legitimate shared templates across distinct departments**.

### 5.1 Exact Content Duplicates (3 pairs)

```text
Pair A: Genuine URL Routing Alias
  Doc 1: doc_html_5a3381b6b3bbdbe5 | URL: https://rvce.edu.in/academics_and_examinations/academic_student_services/
  Doc 2: doc_html_b7d1938d436c0421 | URL: https://rvce.edu.in/academic_student_services/
  Length: 1,385 chars (Identical title, headers, body)
  Verdict: GENUINE SYSTEM DUPLICATE. The same page is served under two WordPress permalink paths.

Pair B: Legitimate Multi-Department Stub Overlap
  Doc 1: doc_html_f21f055fd4d98af2 | URL: https://rvce.edu.in/department/ae/academic_planning/ (Aerospace)
  Doc 2: doc_html_ce38440aeb14457a | URL: https://rvce.edu.in/department/chemical_engineering/academic_planning/ (Chemical)
  Length: 1,248 chars
  Verdict: LEGITIMATE SHARED BOILERPLATE. Both department pages point to identical campus-wide academic calendar announcements, but represent distinct engineering departments.

Pair C: Genuine Department URL Alias
  Doc 1: doc_html_52ec91e7b93d9a82 | URL: https://rvce.edu.in/department/etc/placement/
  Doc 2: doc_html_b91ce3f237697b46 | URL: https://rvce.edu.in/department/etc/ete-placement/
  Length: 3,512 chars
  Verdict: GENUINE SYSTEM DUPLICATE. The Electronics & Telecommunication Engineering department created duplicate URLs ('placement' and 'ete-placement') serving the exact same placement report.
```

### 5.2 Near-Duplicate Clusters ($0.86 \le Jaccard \le 0.97$)
- **Cluster Characteristics:** 14 of the 17 near-duplicate pairs occur among `/department/<dept_name>/academic_planning/` URLs across:
  - `ai_ml` (Artificial Intelligence & Machine Learning)
  - `ae` (Aerospace Engineering)
  - `chemical_engineering` (Chemical Engineering)
  - `civil_engineering` (Civil Engineering)
  - `etc` (Electronics & Telecommunication Engineering)
  - `eee` (Electrical & Electronics Engineering)
  - `ece` (Electronics & Communication Engineering)
- **Root Cause:** RVCE departments use a standardized Elementor template for "Academic Planning" that provides a uniform blurb about academic calendars, VTU schemes, and syllabus structures. Only minor departmental headings differ, producing high lexical Jaccard similarity.
- **RAG / ML Impact:** If indexing into vector databases or training text models, having 7 nearly identical academic planning pages will saturate top-k vector retrieval with redundant departmental stubs.

---

## 6. Topic Grounding & Classification Task Feasibility

### 6.1 Are Topic Labels Verifiably Grounded?
Yes, but with **two notable heuristic classification edge cases**:
1. **URL Prefix Collisions:**
   - The path `/academics_and_examinations/` contains both examination notices and fee payment circulars (`/academics_and_examinations/fee_payment_circulars/`).
   - Because the regex pattern for `examinations_regulations` matched the substring `examinations`, the fee payment circular was categorized as `examinations_regulations` rather than `fees_finance`.
2. **Generic Heading Over-Matching:**
   - The college-wide placement page `https://rvce.edu.in/placement_and_training/` contained an internal heading `## Department of Placement & Training`.
   - Because `departments` had a heading pattern matching `department of`, this college-level career placement page was categorized under `departments`.

### 6.2 Can Current Data Support Supervised Topic Classification?
**Verdict: NO, not in its present state.**

| Evaluation Criterion | Current State | Required for Supervised ML | Verdict |
|---|---|---|---|
| **Sample Size** | 45 non-empty documents | $\ge 200 - 500$ documents minimum | ❌ Insufficient |
| **Class Balance** | 31 docs in 1 class; 1 doc in another | Reasonably balanced across categories | ❌ Severe Imbalance |
| **Zero-Sample Classes** | `fees_finance` has 0 non-empty docs | All target classes populated with text | ❌ Critical Blocker |
| **Missing Target Classes** | `hostels` (0), `facilities` (0) | Full taxonomy represented | ❌ Incomplete |
| **Label Independence** | Labels derived by heuristics from text | Human-curated or external gold ground truth | ❌ Leakage Risk |

---

## 7. Data Quality Issues & Leakage Risks

### 7.1 Data Quality Problems Identified
1. **Scanned PDF Blind Spot:**
   - 100% of collected fee notifications and 90% of admission quota guides exist only as scanned PDFs. Without OCR, the pipeline has zero usable text for fees and finances.
2. **WordPress URL Alias Redundancy:**
   - Distinct URLs returning bit-for-bit identical content dilute search relevance and inflate dataset counts.
3. **Template Boilerplate Bloat:**
   - Academic planning stubs across 7 engineering departments repeat identical general announcements, causing near-duplicate cluster noise.

### 7.2 Data & Machine Learning Leakage Risks
1. **Feature-to-Label Leakage:**
   - The topic labels were assigned based on regex matches on URL slugs and page headings.
   - If a machine learning model is trained on page content that includes these top-level headings (e.g. `## Department of Computer Science`, `## Admission Circulars`), the model will trivially memorize the exact heuristic keyword rule rather than learning semantic domain context.
2. **Train/Test Split Leakage (from Near-Duplicates):**
   - If identical department stubs (e.g., Aerospace Academic Planning and Chemical Engineering Academic Planning) are randomly split into train and test sets, the model will be evaluated on test examples that are 96.6% identical to training examples, artificially inflating accuracy metrics.

---

## 8. Recommendations & Next Steps for Phase 1

To transition this knowledge acquisition engine into a production-ready data foundation for the Campus AI Assistant, the following sequential steps are recommended:

### Step 1: Implement OCR Fallback for Scanned Circulars (High Priority)
- Add a lightweight OCR fallback in `src/extractors.py` (e.g. using `pytesseract` or `easyocr`) triggered whenever a PDF contains embedded image pages but zero readable text characters.
- This will unlock the 15 currently empty circulars, immediately populating the `fees_finance` and `admissions` categories with real tabular fee data.

### Step 2: Refine Heuristic Categorization Rules
- Update `TOPIC_RULES` in `src/cleaning.py` so that specific sub-paths (e.g., `/fee_payment_circulars/`) take precedence over parent folder names (e.g., `/academics_and_examinations/`).
- Distinguish academic department portals from administrative departments (e.g., Placement & Training, Physical Education) to prevent category pollution.

### Step 3: Expand Seed Scope for Missing Campus Classes
- Introduce targeted seed entry points in `config.yaml` to ensure discovery of under-represented domains:
  - `https://rvce.edu.in/facilities/` (Hostels, central library, sports, medical centre)
  - `https://rvce.edu.in/hostels/` or hostel sub-pages
  - `https://rvce.edu.in/national_service_scheme_nss/` and cultural clubs

### Step 4: De-duplicate URL Routing Aliases
- In `src/build_dataset.py`, add a post-processing filter that drops identical content duplicates originating from WordPress URL aliases, keeping only the canonical canonical permalink.

### Step 5: Prepare Query Intent Classification Dataset
- Rather than attempting full-document multiclass classification on 45 web pages, follow the project blueprint: curate a labeled corpus of student-style queries (e.g., 200–500 natural language queries mapped to the 8 campus intent classes) and utilize these crawled documents as the retrieved context in Phase 2 RAG.
