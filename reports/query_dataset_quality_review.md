# Campus Query Dataset — Quality, Ambiguity & Taxonomy Review

**Review Date:** October 12, 2026  
**Audited Artifacts:**
- Query Dataset: [`data/queries/campus_queries.csv`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/data/queries/campus_queries.csv)
- Metadata Definition: [`data/queries/dataset_metadata.json`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/data/queries/dataset_metadata.json)
- Initial Seed Review: [`reports/query_dataset_review.md`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/reports/query_dataset_review.md)

---

## 1. Executive Context & Scope Reminder

> [!IMPORTANT]
> **Methodological Grounding:**
> - The 40 questions in `campus_queries.csv` constitute an **initial seed set** designed strictly for schema validation, taxonomy sanity checking, and initial pipeline plumbing.
> - These are **manually authored benchmark questions**, not real student survey responses or search logs.
> - **Product Goal Alignment:** The primary objective of the Campus AI Assistant is to answer a wide variety of RVCE-related inquiries using a robust, authoritative Knowledge Base and reliable retrieval. The query classification model functions as a **routing and dispatch component**, which directs queries to the appropriate knowledge domain without limiting the final system's retrieval scope.

---

## 2. Intent Assignment Validity Audit (Question-by-Question)

Every question in the 40-query seed set was evaluated to determine whether its assigned label is logically sound and verifiable.

### 2.1 Unambiguous, Strong Assignments (33 of 40)
- **`admissions` (Q001, Q002, Q003, Q005):** Inquiries regarding COMEDK, DCET lateral entry, Management Quota, and KCET cutoff ranks belong squarely to admissions.
- **`fees_finance` (Q006, Q007, Q008, Q009):** Questions focusing on tuition fee deadlines, late payment fines, SSP portal scholarships, and installment requests clearly target financial administration.
- **`academics_exams` (Q011, Q012, Q013, Q014, Q015):** Clear alignment with academic calendars, fast-track exam attendance rules, revaluation, credit vertical progression, and autonomous scheme syllabi.
- **`departments_faculty` (Q016, Q017, Q018, Q019):** Clear queries targeting HODs, faculty directories, department cabins, and office contacts.
- **`placements_careers` (Q021, Q022, Q023, Q024, Q025):** Well-targeted inquiries regarding placement packages, 6-month internships, company CGPA eligibility, resume preparation, and backlogs.
- **`facilities_hostels` (Q026, Q027, Q028, Q029):** Explicit questions regarding boys hostel admission, library exam hours, curfew rules, and sports/gym facilities.
- **`student_services_clubs` (Q031, Q032, Q033, Q035):** Strong alignment with technical teams (Astra Robotics, Team Chimera), bonafide student certificates, NSS/NCC enrollment, and cultural fest organization.
- **`general_college_info` (Q036, Q037, Q038, Q039, Q040):** Institutional contact numbers, VTU autonomous status, campus address/metro commute, AICTE/NIRF reports, and principal identity.

### 2.2 Boundary Ambiguities & Dual-Intent Challenges (7 of 40)
The audit identified 7 questions exhibiting semantic boundary overlap where a student query spans multiple administrative or academic domains:

| Query ID | Question Text | Assigned Intent | Secondary / Competing Intent | Ambiguity Root Cause |
|---|---|---|---|---|
| `Q004` | *"Documents required for M.Tech PGCET document verification"* | `admissions` | `student_services_clubs` / Administrative Documents | Document checklists straddle admissions verification and student documentation offices. |
| `Q010` | *"where to submit the fee challan receipt after online payment"* | `fees_finance` | `general_college_info` / Physical Administration | Involves fee payment (`fees_finance`), but asks for the physical administrative counter location. |
| `Q020` | *"Does the MCA department offer electives in cloud computing?"* | `departments_faculty` | `academics_exams` | Asks about departmental course offerings (`departments`), but centers on curriculum electives (`academics`). |
| `Q025` | *"Can students with an active backlog sit for mass recruiter drives?"* | `placements_careers` | `academics_exams` | "Backlog" is an autonomous exam concept, but the decision intent is placement eligibility. |
| `Q030` | *"is mess food fee included in hostel charges or paid separately?"* | `facilities_hostels` | `fees_finance` | Contains the keyword `"fee"` and financial terms, but the core subject is hostel mess accommodation policy. |
| `Q034` | *"How do I register for NAD Digilocker to access my semester grade sheets?"* | `student_services_clubs` | `academics_exams` | Mentions `"grade sheets"` (examination records), but DigiLocker portal issuance is managed by student services. |
| `Q035` | *"who to contact for organizing an inter-college cultural fest event?"* | `student_services_clubs` | `general_college_info` | Asking for an institutional contact person regarding student cultural activities. |

---

## 3. Linguistic Diversity vs. Repetitiveness Analysis

An automated lexical and structural analysis was conducted on `campus_queries.csv`:

```text
[METRIC] Total word tokens:         405 words
[METRIC] Unique vocabulary:         237 unique words
[METRIC] Mean query length:         10.1 words (Min: 6 words, Max: 14 words)
[METRIC] Exact duplicate pairs:     0
[METRIC] Near-duplicate pairs:      0 (All pairwise Jaccard similarities < 0.45)
```

### 3.1 Syntactic Variety
The 40 queries avoid repetitive templates through structural variation:
1. **Wh-Questions (19 queries):** *"What is..."*, *"Where is..."*, *"Who is..."*, *"When will..."*, *"Which top companies..."*.
2. **Action / Workflow How-To (8 queries):** *"How do I apply..."*, *"How can I join..."*, *"how to get seat under..."*.
3. **Modal Verification (5 queries):** *"Can diploma students apply..."*, *"Is there any fine..."*, *"Does the MCA department..."*.
4. **Keyword / Telegraphese Queries (8 queries):** *"Documents required for..."*, *"procedure to apply for..."*, *"cutoff rank for cse branch..."*, *"college address and nearest metro station..."*.

### 3.2 Student Jargon & Stylistic Nuance
The dataset incorporates genuine collegiate terminology:
- Entrance quotas: *KCET round 2, COMEDK, PGCET, DCET lateral entry, Management Quota (MQ)*
- Placement terms: *Tier 1 software companies, 6-month internships, mass recruiter drives, average package*
- Regulatory terms: *Fast-track examinations, vertical progression credit cutoff, autonomous 2022 scheme syllabus*
- Administrative jargon: *Bonafide certificate, challan receipt, post-matric SSP scholarship, NAD DigiLocker, curfew / in-time rules*
- Casing: Mixed standard capitalization and lowercase student inputs (*"how to get seat under management quota in rvce?"*).

---

## 4. Intent Overlap & Cross-Intent Lexical Collisions

A key finding of this audit is that **certain high-frequency campus content words appear across multiple intents**, creating potential confusion for linear/bag-of-words text classifiers:

```text
Cross-Intent Content Words:
- 'apply'        -> Appears in 4 intents (admissions, fees_finance, academics_exams, facilities_hostels)
- 'students'     -> Appears in 3 intents (admissions, facilities_hostels, placements_careers)
- 'engineering'  -> Appears in 3 intents (admissions, departments_faculty, general_college_info)
- 'semester'     -> Appears in 3 intents (academics_exams, fees_finance, student_services_clubs)
- 'contact'      -> Appears in 3 intents (departments_faculty, general_college_info, student_services_clubs)
- 'campus'       -> Appears in 3 intents (facilities_hostels, general_college_info, placements_careers)
- 'procedure'    -> Appears in 2 intents (admissions, academics_exams)
- 'fee'          -> Appears in 2 intents (fees_finance, facilities_hostels)
- 'grade'        -> Appears in 2 intents (academics_exams, student_services_clubs)
- 'exam'         -> Appears in 2 intents (academics_exams, facilities_hostels)
```

### Critical Classifier Risk Example:
In `Q030` (*"is mess food fee included in hostel charges or paid separately?"*), the presence of the word **`"fee"`** represents a strong distractor for TF-IDF models, which will naturally associate `"fee"` with `fees_finance` rather than `facilities_hostels`.

---

## 5. Train/Test Split Leakage & Out-of-Vocabulary (OOV) Analysis

The current 40-query dataset allocates 32 queries to `train` (4 per class) and 8 queries to `test` (1 per class).

### 5.1 Leakage Evaluation
- **No Data Leakage:** There is **zero sample contamination or sentence-level duplication** between the train and test splits. The test queries represent distinct semantic scenarios within each class.

### 5.2 Out-of-Vocabulary (OOV) Sparsity Issue
Because each intent class has only 4 training examples, a bag-of-words or TF-IDF vectorizer trained strictly on `train` faces severe vocabulary starvation:

```text
[METRIC] Training vocabulary size:   201 words
[METRIC] Test vocabulary size:       63 words
[METRIC] Test words NOT in training: 41 words (65.1% Out-of-Vocabulary rate)
```

### Examples of Test OOV Blindspots:
- **`admissions` (`Q005`):** Words `"cutoff"`, `"rank"`, `"kcet"`, and `"round"` never appear anywhere in the training set.
- **`academics_exams` (`Q015`):** Words `"syllabus"`, `"scheme"`, `"autonomous"`, and `"download"` never appear in training.
- **`placements_careers` (`Q025`):** Words `"backlog"`, `"active"`, `"mass"`, `"recruiter"`, and `"drives"` are completely OOV.
- **`facilities_hostels` (`Q030`):** Words `"mess"`, `"food"`, `"charges"`, `"included"`, and `"separately"` are OOV, leaving only `"fee"` as a recognized token—directly steering the model toward `fees_finance`.

> [!CAUTION]
> This 65.1% OOV rate proves conclusively that **40 examples are insufficient for evaluating an ML classifier**. Any accuracy metric computed on these 8 test queries will be dominated by random lexical overlap rather than true semantic understanding.

---

## 6. Taxonomy Adequacy for the Final Campus Assistant

The 8 predefined intent classes cover the core administrative divisions of RVCE. However, when evaluating the final product goal (answering a broad range of student questions), the following observations apply:

### 6.1 Strengths of the 8-Class Taxonomy
1. **Direct Alignment with Official College Infrastructure:**
   The 8 intents map directly to the crawled web sections:
   - `admissions` $\rightarrow$ Admission circulars & quota guidelines
   - `fees_finance` $\rightarrow$ Fee circulars & bank accounts
   - `academics_exams` $\rightarrow$ Autonomous regulations & calendars
   - `departments_faculty` $\rightarrow$ Department directories & faculty profiles
   - `placements_careers` $\rightarrow$ Placement statistics & training cell
   - `facilities_hostels` $\rightarrow$ Hostel notices & campus infrastructure
   - `student_services_clubs` $\rightarrow$ Bonafide certificates, NSS/NCC & clubs
   - `general_college_info` $\rightarrow$ Campus address, contacts, and NIRF disclosures
2. **Modular Retrieval Partitioning:**
   The classifier can serve as an upstream router that dynamically restricts the retrieval search space (e.g., routing a fee query exclusively to fee circular chunks), significantly improving RAG precision.

### 6.2 Missing Use Cases & Gaps for Future Expansion
1. **Lack of an `out_of_scope` / Fallback Class:**
   Students will inevitably submit queries unrelated to college administration (*"Who won the Cricket World Cup?"*, *"Write a sorting algorithm in C++"*). Without an `out_of_scope` class or a confidence-threshold rejection mechanism, the classifier will be forced to map irrelevant queries into one of the 8 campus classes.
2. **Subjective / Peer Student Experience Queries:**
   Questions regarding mess food quality, senior advice, club workload, or peer culture (*"How is the mess food really?"*, *"Which technical club is best for CSE freshers?"*) do not have official website answers. In Phase 1, the assistant must either route these to `facilities_hostels` / `student_services_clubs` with a factual disclaimer or route to a dedicated community channel in later phases.

---

## 7. Recommended Actions & Next Steps

Based on this review, the following sequential corrections and refinements are recommended:

1. **Retain the 40 Seed Queries as a Benchmark Validation Set:**
   Keep `data/queries/campus_queries.csv` exactly as-is to verify baseline pipeline execution, DVC tracking, and MLflow logging.
2. **Implement Confidence Thresholding in the Intent Router:**
   Design the inference router with a softmax probability or margin threshold (e.g., $P < 0.40 \rightarrow \text{fallback to general retrieval}$), preventing OOV test queries from being misrouted with false confidence.
3. **Plan Query Dataset Expansion (Post-Plumbing):**
   Once the MLflow tracking and baseline training scripts are validated, expand the query dataset to **160–240 queries** (20–30 queries per intent class) to lower the OOV rate below 25% and enable reliable 5-fold cross-validation.
4. **Ensure Knowledge Base Primacy:**
   Maintain the architectural principle that **the Knowledge Base and retrieval engine remain the source of truth**. Even if a query is borderline between `fees_finance` and `facilities_hostels` (e.g. mess fees), hybrid dense/sparse retrieval across the full corpus will successfully retrieve the relevant source-backed evidence.
