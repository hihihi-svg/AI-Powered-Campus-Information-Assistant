# Campus Query Dataset — Initial Seed Set Review

**Dataset File:** [`data/queries/campus_queries.csv`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/data/queries/campus_queries.csv)  
**Metadata File:** [`data/queries/dataset_metadata.json`](file:///c:/Users/lalan/Desktop/AI-Powered-Campus-Information-Assistant/data/queries/dataset_metadata.json)  
**Review Date:** October 12, 2026  
**Status:** Prepared for Review & Approval  

---

## 1. Provenance & Dataset Methodology

> [!IMPORTANT]
> **Provenance Disclosure:**  
> The 40 questions in this dataset are **manually authored benchmark examples** crafted by the engineering team to seed the Phase 1 Campus Query Intent Classification task.  
> They are **NOT** survey responses, interview transcripts, or search logs collected from actual students. No student identities or survey metrics have been invented or claimed.

### Characteristics & Design Rules:
1. **Questions Only (No Answers):** This is purely a query dataset for supervised multiclass intent classification. It contains zero answers, and no college policies, fee figures, or deadlines are invented.
2. **Linguistic Diversity:** Includes formal inquiries, informal student phrasing (*"how to get seat"*, *"in-time rules"*, *"mess food fee"*), colloquial terms (*"mass recruiter drives"*, *"cutoff rank"*), and mixed case/punctuation.
3. **Class Parity:** Exactly 8 intent classes, exactly 5 questions per class, exactly 40 unique questions.
4. **Stratified Split:** 32 training questions (4 per class) and 8 test questions (1 per class) to establish a deterministic, reproducible evaluation baseline.

> [!WARNING]
> **Sample Size Notice:**  
> This 40-example seed set is designed solely for **schema verification, taxonomy alignment, and pipeline sanity testing**. It is **not sufficient for statistically reliable model evaluation or generalizable deployment**. Scaling to a larger curated dataset (e.g., 160–240 queries) will be conducted after this review.

---

## 2. Intent Taxonomy & Example Questions

### 2.1 `admissions`
*Covers entrance quotas (KCET, COMEDK, Management Quota, PGCET, DCET), eligibility criteria, rank cutoffs, and document verification.*

| Query ID | Question Text | Split | Phrasing / Style Notes |
|---|---|---|---|
| `Q001` | What is the procedure for B.E. admission through COMEDK quota? | `train` | Formal, quota-specific |
| `Q002` | Can diploma students apply for lateral entry into 2nd year engineering? | `train` | Eligibility inquiry (DCET) |
| `Q003` | how to get seat under management quota in rvce? | `train` | Lowercase, informal student query |
| `Q004` | Documents required for M.Tech PGCET document verification | `train` | Concise keyword-style checklist query |
| `Q005` | cutoff rank for cse branch in kcet round 2 | `test` | High-frequency student jargon query |

---

### 2.2 `fees_finance`
*Covers tuition fee schedules, installment permissions, late payment fines, challans/receipts, and scholarships.*

| Query ID | Question Text | Split | Phrasing / Style Notes |
|---|---|---|---|
| `Q006` | What is the deadline for paying higher semester tuition fees? | `train` | Formal deadline inquiry |
| `Q007` | Is there any fine if fee payment is delayed by a week? | `train` | Penalty policy question |
| `Q008` | how do I apply for post-matric state scholarship on SSP portal? | `train` | Government scholarship workflow (SSP) |
| `Q009` | Can I pay the college fee in two installments? | `train` | Financial accommodation request |
| `Q010` | where to submit the fee challan receipt after online payment | `test` | Administrative physical submission query |

---

### 2.3 `academics_exams`
*Covers academic calendars, timetables, scheme syllabi, attendance criteria, fast-track exams, credits, and revaluation.*

| Query ID | Question Text | Split | Phrasing / Style Notes |
|---|---|---|---|
| `Q011` | When will the academic calendar and first semester exam timetable be released? | `train` | Multi-event scheduling inquiry |
| `Q012` | What is the minimum attendance percentage required to be eligible for fast-track examinations? | `train` | Formal autonomous regulation question |
| `Q013` | procedure to apply for grade card revaluation and photocopy | `train` | Post-result administrative workflow |
| `Q014` | how many credits needed to register for 5th semester? | `train` | Vertical progression rule query |
| `Q015` | Where can I download the autonomous 2022 scheme syllabus for mechanical? | `test` | Scheme-specific syllabus download query |

---

### 2.4 `departments_faculty`
*Covers academic branch administration, HODs, faculty directories, office/cabin locations, and branch electives.*

| Query ID | Question Text | Split | Phrasing / Style Notes |
|---|---|---|---|
| `Q016` | Who is the current Head of Department for Computer Science and Engineering? | `train` | Specific leadership query |
| `Q017` | Where is the cabin of the Aerospace Engineering department coordinator? | `train` | Physical campus directory inquiry |
| `Q018` | faculty list and research areas for biotechnology dept | `train` | Academic profiling query |
| `Q019` | how to contact civil engineering department office? | `train` | Informal department communication query |
| `Q020` | Does the MCA department offer electives in cloud computing? | `test` | Course/elective offering query |

---

### 2.5 `placements_careers`
*Covers campus placements, salary packages, visiting recruiters, 6-month internships, training cell assistance, and backlog eligibility.*

| Query ID | Question Text | Split | Phrasing / Style Notes |
|---|---|---|---|
| `Q021` | What was the average package and highest salary offered in 2025-26 campus placements? | `train` | Placement statistics inquiry |
| `Q022` | Which top companies visit the campus for 6-month internships? | `train` | Industry engagement inquiry |
| `Q023` | eligibility criteria for tier 1 software companies visiting for placements | `train` | CGPA/criteria threshold query |
| `Q024` | how does the placement and training cell help with resume preparation? | `train` | Student service support query |
| `Q025` | Can students with an active backlog sit for mass recruiter drives? | `test` | Common placement policy anxiety query |

---

### 2.6 `facilities_hostels`
*Covers on-campus accommodation, hostel curfew rules, mess fees, central library hours, sports infrastructure, and gym.*

| Query ID | Question Text | Split | Phrasing / Style Notes |
|---|---|---|---|
| `Q026` | How do I apply for on-campus boys hostel accommodation? | `train` | Residential application process |
| `Q027` | What are the central library working hours during exam week? | `train` | Facility operational hours query |
| `Q028` | hostel in-time rules and night curfew for first years | `train` | Residential discipline policy inquiry |
| `Q029` | Does the campus have badminton and gym facilities for students? | `train` | Sports/infrastructure inquiry |
| `Q030` | is mess food fee included in hostel charges or paid separately? | `test` | Hostel billing and mess structure query |

---

### 2.7 `student_services_clubs`
*Covers bonafide certificates, identity verification, technical/cultural student clubs, NSS/NCC, and DigiLocker portals.*

| Query ID | Question Text | Split | Phrasing / Style Notes |
|---|---|---|---|
| `Q031` | How can I join student technical clubs like Astra Robotics or Team Chimera? | `train` | RVCE-specific technical teams query |
| `Q032` | Where do I submit the application to get a bonafide student certificate? | `train` | Administrative document issuance query |
| `Q033` | nss and ncc enrollment process for freshers | `train` | Co-curricular service enrollment query |
| `Q034` | How do I register for NAD Digilocker to access my semester grade sheets? | `train` | Government portal credential inquiry |
| `Q035` | who to contact for organizing an inter-college cultural fest event? | `test` | Student activity leadership query |

---

### 2.8 `general_college_info`
*Covers official reception contact numbers, campus address, transit (metro), VTU autonomous affiliation, NIRF reports, and leadership.*

| Query ID | Question Text | Split | Phrasing / Style Notes |
|---|---|---|---|
| `Q036` | What is the official contact email and reception phone number for RVCE? | `train` | Core institutional contact inquiry |
| `Q037` | Is RV College of Engineering affiliated with VTU or a deemed university? | `train` | Institutional accreditation inquiry |
| `Q038` | college address and nearest metro station to reach campus | `train` | Transit and commute query |
| `Q039` | Where can I find the mandatory AICTE disclosures and NIRF ranking report? | `train` | Statutory compliance inquiry |
| `Q040` | who is the current principal of the college? | `test` | Executive administrative inquiry |

---

## 3. Automated Validation Results

All checks were executed programmatically against `data/queries/campus_queries.csv`:

```text
[CHECK 1] Row Count:                40 rows (EXACT MATCH)
[CHECK 2] Unique Query IDs:         40 unique (EXACT MATCH, Q001..Q040)
[CHECK 3] Unique Query Texts:       40 unique (EXACT MATCH, zero duplicates)
[CHECK 4] Intent Label Count:       8 unique labels (EXACT MATCH)
[CHECK 5] Per-Intent Distribution:  5 examples per intent (PERFECTLY BALANCED)
[CHECK 6] Split Allocation:         32 train, 8 test (EXACT 4:1 PER INTENT)
[CHECK 7] Null/Empty Values:        0 nulls (CLEAN)
```

**Verification Status:** **PASS (100% compliant)**
