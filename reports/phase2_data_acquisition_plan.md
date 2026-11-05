# Phase II Data Acquisition Plan — RVCE Campus AI Assistant

**Created:** 2026-11-03  
**Status:** Planning Only — No Data Downloaded or Modified  
**Author:** Phase II Planning (AI-assisted, read-only audit)

---

## Part 0: Existing Corpus Inventory (Do Not Re-Collect)

Before specifying new data to collect, the following is a precise inventory of what Phase I already gathered. All new collection must avoid duplicating these sources.

### 0.1 Crawled HTML Pages (45 documents, all successful)

| Topic (Phase I label) | URL | Chars |
|---|---|---|
| *(homepage)* | `https://rvce.edu.in/` | 8,199 |
| `academics` | `https://rvce.edu.in/programmes-faculty-details/` | 1,263 |
| `academics` | `https://rvce.edu.in/calendar-of-events/` | 7,528 |
| `admissions` | `https://rvce.edu.in/admission-circulars/` | 1,865 |
| `admissions` | `https://rvce.edu.in/b-e-first-year-academic-circulars/` | 6,117 |
| `admissions` | `https://rvce.edu.in/b-e-higher-studies-academic-circular/` | 10,332 |
| `examinations_regulations` | `https://rvce.edu.in/academic-circular/` | 1,841 |
| `examinations_regulations` | `https://rvce.edu.in/academics_and_examinations/` | 2,906 |
| `examinations_regulations` | `https://rvce.edu.in/academics_and_examinations/academic_student_services/` | 1,385 |
| `examinations_regulations` | `https://rvce.edu.in/academics_and_examinations/fee_payment_circulars/` | 1,822 |
| `examinations_regulations` | `https://rvce.edu.in/examination-circulars/` | 21,874 |
| `examinations_regulations` | `https://rvce.edu.in/m-tech-and-mca-academic-circulars/` | 9,130 |
| `fees_finance` | `https://rvce.edu.in/academic_student_services/` | 1,385 |
| `placements_careers` | `https://rvce.edu.in/placement_and_training/placement_statistics/` | 1,792 |
| `departments` | `https://rvce.edu.in/placement_and_training/` | 9,347 |
| `departments` | `https://rvce.edu.in/department/ai_ml/al_ml_placement/` | 2,200 |
| `departments` | `https://rvce.edu.in/department/ai_ml/academic_planning/` | 1,247 |
| `departments` | `https://rvce.edu.in/department/ai_ml/research/` | 3,319 |
| `departments` | `https://rvce.edu.in/department/ae/academic_planning/` | 1,248 |
| `departments` | `https://rvce.edu.in/department/ae/ase_placement/` | 2,703 |
| `departments` | `https://rvce.edu.in/department/ae/research/` | 4,014 |
| `departments` | `https://rvce.edu.in/department/biotechnology/placement/` | 5,006 |
| `departments` | `https://rvce.edu.in/department/biotechnology/research/` | 3,952 |
| `departments` | `https://rvce.edu.in/department/chemical_engineering/academic_planning/` | 1,248 |
| `departments` | `https://rvce.edu.in/department/chemical_engineering/ce_placement/` | 2,501 |
| `departments` | `https://rvce.edu.in/department/chemical_engineering/research/` | 3,014 |
| `departments` | `https://rvce.edu.in/department/civil_engineering/academic_planning/` | 1,311 |
| `departments` | `https://rvce.edu.in/department/civil_engineering/department-of-civil-engineering/` | *(varies)* |
| `departments` | `https://rvce.edu.in/department/civil_engineering/placement/` | 7,095 |
| `departments` | `https://rvce.edu.in/department/civil_engineering/research/` | 3,148 |
| `departments` | `https://rvce.edu.in/department/cse/academic_planning/` | 1,285 |
| `departments` | `https://rvce.edu.in/department/cse/placement/` | 8,497 |
| `departments` | `https://rvce.edu.in/department/cse/research/` | 3,546 |
| `departments` | `https://rvce.edu.in/department/ece/academic_planning/` | 1,426 |
| `departments` | `https://rvce.edu.in/department/ece/placement/` | 6,024 |
| `departments` | `https://rvce.edu.in/department/eee/academic_planning/` | 1,311 |
| `departments` | `https://rvce.edu.in/department/eee/eee_placement/` | *(varies)* |
| `departments` | `https://rvce.edu.in/department/eee/research/` | 3,629 |
| `departments` | `https://rvce.edu.in/department/etc/placement/` | 3,512 |
| `departments` | `https://rvce.edu.in/department/etc/ete-placement/` | 3,512 *(duplicate)* |
| `departments` | `https://rvce.edu.in/department/etc/academic_planning/` | 1,311 |
| `departments` | `https://rvce.edu.in/department/iem/placement/` | 3,873 |
| `departments` | `https://rvce.edu.in/department/iem/research/` | 5,930 |
| `departments` | `https://rvce.edu.in/department/ise/ise_placement/` | 4,489 |

### 0.2 Scanned PDFs (15 documents, all EMPTY — needs OCR)

All 15 PDFs are scanned images with zero embedded text. OCR is required before any of these can be used. They cover:
- **Fee circulars:** 9 PDFs (UG/PG/NRI fee slabs, payment extensions, revised fees)
- **Admission procedure guides:** 6 PDFs (COMEDK, KEA/KCET, Management Quota, DCET lateral entry, NRI quota, M.Tech/MCA balance fees)

### 0.3 Critical Gaps in Phase I Corpus

| Category | Phase I Coverage | Gap Level |
|---|---|---|
| Hostel / accommodation | **Zero documents** | 🔴 Complete blind spot |
| Canteen / mess | **Zero documents** | 🔴 Complete blind spot |
| Library | **Zero documents** | 🔴 Complete blind spot |
| Medical / health centre | **Zero documents** | 🔴 Complete blind spot |
| Transport / bus routes | **Zero documents** | 🔴 Complete blind spot |
| Sports / gym / infrastructure | **Zero documents** | 🔴 Complete blind spot |
| Student clubs / NSS / NCC | **Zero documents** | 🔴 Complete blind spot |
| Administrative procedures (bonafide, duplicate grade card, FIR) | **Zero documents** | 🔴 Complete blind spot |
| Scholarships | **Zero documents** | 🔴 Complete blind spot |
| Grievance / complaints | **Zero documents** | 🔴 Complete blind spot |
| Placement notices / individual company announcements | **1 statistics page only** | 🟡 Heavily under-represented |
| Faculty & HOD details | Partial (placement pages only, no faculty directory) | 🟡 Heavily under-represented |
| Fees (actual amounts) | 9 PDFs but all EMPTY | 🟡 Needs OCR unlock |
| Admission procedures (step-by-step) | 6 PDFs but all EMPTY | 🟡 Needs OCR unlock |
| Syllabus / curriculum per department | **Zero documents** | 🔴 Complete blind spot |

---

## Part 1: Academic Information

### 1.1 Syllabus and Scheme Documents

| Field | Details |
|---|---|
| **Specific data** | B.E. scheme documents (2022 Autonomous scheme, previous VTU schemes), subject codes, credit structure, CIE/SEE split (typically 50:50), syllabus per subject per semester per branch |
| **Source type** | Official — RVCE autonomous scheme document links published on academic planning pages |
| **Suggested source URLs** | `https://rvce.edu.in/academics_and_examinations/` subpages; department-specific scheme PDFs linked from `https://rvce.edu.in/department/<dept>/academic_planning/` |
| **Verified/Suggested** | Suggested — these links exist on pages already crawled but the PDFs themselves were not collected |
| **Static/Dynamic** | Mostly static; updated once per scheme revision (every 4–6 years); minor annual updates for electives |
| **Collection method** | Targeted web crawler (extend max_pdfs limit); OCR if scanned |
| **Required metadata** | `scheme_year`, `branch`, `semester`, `document_type` (syllabus/scheme), `effective_from`, `source_url` |
| **Update frequency** | On scheme revision (~every 4 years); minor elective updates annually |
| **Priority** | **P0** — Core academic data; students constantly ask about credits, subjects, syllabus |
| **Permissions** | Publicly available on rvce.edu.in; no login required |

### 1.2 Academic Calendar

| Field | Details |
|---|---|
| **Specific data** | Semester start/end dates, CIE exam windows, SEE schedule, holiday list, makeup exam dates, project submission deadlines, internship windows |
| **Source type** | Official — published annually by RVCE academics section |
| **Suggested source URLs** | `https://rvce.edu.in/calendar-of-events/` (already crawled, 7,528 chars — needs re-crawl as page updates frequently); `https://rvce.edu.in/academic-circular/` (already crawled) |
| **Verified/Suggested** | Verified — pages confirmed to exist in Phase I |
| **Static/Dynamic** | Dynamic — updated at semester start and mid-semester for revisions |
| **Collection method** | Incremental web crawler targeting these URLs specifically; re-crawl every 2 weeks during active semester |
| **Required metadata** | `academic_year`, `semester`, `event_type`, `event_date`, `branch_applicability` |
| **Update frequency** | Every semester (twice a year); revision notices mid-semester |
| **Priority** | **P0** — Highest query volume from students |
| **Permissions** | Public; no authentication required |

### 1.3 Attendance Rules and Fast-Track Regulations

| Field | Details |
|---|---|
| **Specific data** | Minimum attendance % (typically 85%), calculation methodology, condonation procedures, fast-track exam eligibility, medical leave provisions |
| **Source type** | Official — RVCE autonomous regulations document |
| **Suggested source URLs** | `https://rvce.edu.in/examination-circulars/` (already crawled, 21,874 chars — richest page); `https://rvce.edu.in/academics_and_examinations/` |
| **Verified/Suggested** | Verified — examination-circulars page is confirmed and has high content |
| **Static/Dynamic** | Static — changes only with scheme revision |
| **Collection method** | Already partially collected; need structured extraction of attendance rules section |
| **Required metadata** | `rule_type`, `effective_scheme`, `value`, `exception_conditions` |
| **Update frequency** | Per scheme revision |
| **Priority** | **P0** |
| **Permissions** | Public |

### 1.4 CIE/SEE Assessment Pattern

| Field | Details |
|---|---|
| **Specific data** | Number of CIE tests, marks weightage (e.g., 3 CIEs of which best 2 count), SEE pattern, passing criteria, revaluation and photocopy procedure |
| **Source type** | Official |
| **Suggested source URLs** | `https://rvce.edu.in/examination-circulars/` (already partially collected); scheme regulation documents linked from academic pages |
| **Verified/Suggested** | Verified (page exists); specific regulation PDFs are suggested |
| **Static/Dynamic** | Static per scheme |
| **Collection method** | Extract from already-crawled examination circulars; supplement with targeted PDF collection |
| **Required metadata** | `assessment_component`, `marks_total`, `passing_marks`, `scheme_year`, `programme` |
| **Update frequency** | Per scheme revision |
| **Priority** | **P0** |
| **Permissions** | Public |

### 1.5 Credit and Progression Requirements

| Field | Details |
|---|---|
| **Specific data** | Credits required to register for each semester, lateral entry credit mapping, CGPA calculation method, grade scale (O/A+/A/B+/B/C/P/F) |
| **Source type** | Official — autonomous scheme regulation document |
| **Suggested source URLs** | `https://rvce.edu.in/academics_and_examinations/` and scheme regulation PDFs |
| **Verified/Suggested** | Suggested |
| **Static/Dynamic** | Static per scheme |
| **Collection method** | Targeted PDF crawler; OCR if scanned |
| **Required metadata** | `programme`, `scheme`, `semester`, `minimum_credits`, `grade_scale` |
| **Update frequency** | Per scheme |
| **Priority** | **P1** |
| **Permissions** | Public |

---

## Part 2: Administrative Procedures

> [!IMPORTANT]
> This entire category has **zero documents** in Phase I. It is the highest-impact gap for student utility.

### 2.1 Duplicate Grade Card / Marks Card

| Field | Details |
|---|---|
| **Specific data** | Step-by-step procedure: required documents (affidavit, fee receipt, FIR copy if lost), application form, office to visit (Registrar/Exams section), processing time, fee amount |
| **Source type** | Official — RVCE examination section notices and circulars |
| **Suggested source URLs** | `https://rvce.edu.in/examination-circulars/` (partially collected, may contain this); `https://rvce.edu.in/academics_and_examinations/academic_student_services/` |
| **Verified/Suggested** | Suggested — the page exists but the specific procedure content may not be present in the crawled text; requires manual verification |
| **Static/Dynamic** | Semi-static — procedure changes rarely but fee amounts update annually |
| **Collection method** | Manual extraction from official circular PDFs; structured KB entry |
| **Required metadata** | `procedure_name`, `required_docs`, `fee_amount`, `office`, `processing_days`, `contact` |
| **Update frequency** | Annual (fee updates); procedure rarely changes |
| **Priority** | **P0** — among the top 5 most-asked administrative questions |
| **Permissions** | Public information; no authentication required |

### 2.2 Transcripts and Official Document Attestation

| Field | Details |
|---|---|
| **Specific data** | Transcript issuance procedure (official/provisional), medium of instruction certificate, migration certificate, attestation process, VTU vs RVCE autonomous documents |
| **Source type** | Official |
| **Suggested source URLs** | `https://rvce.edu.in/academics_and_examinations/academic_student_services/` (partially collected); registrar office notice PDFs |
| **Verified/Suggested** | Suggested |
| **Static/Dynamic** | Semi-static |
| **Collection method** | Manual KB entry from official notices; contact registrar office directly for confirmation |
| **Required metadata** | `document_type`, `required_docs`, `processing_time`, `fee`, `office`, `contact_email` |
| **Update frequency** | Annual |
| **Priority** | **P0** |
| **Permissions** | Public |

### 2.3 Bonafide Certificate

| Field | Details |
|---|---|
| **Specific data** | Who issues it (department office or academic section), purpose codes (visa/bank/scholarship), required documents (fee payment proof, ID), turnaround time, online vs offline request |
| **Source type** | Official |
| **Suggested source URLs** | `https://rvce.edu.in/academics_and_examinations/academic_student_services/` |
| **Verified/Suggested** | Suggested |
| **Static/Dynamic** | Static |
| **Collection method** | Manual KB entry; student services section |
| **Required metadata** | `purpose`, `issuing_office`, `turnaround_days`, `online_portal`, `required_docs` |
| **Update frequency** | Rarely changes |
| **Priority** | **P0** |
| **Permissions** | Public |

### 2.4 FIR Requirements for Document Loss

| Field | Details |
|---|---|
| **Specific data** | When FIR is mandatory (lost grade card, lost ID card), nearest police station to RVCE campus, online FIR portal (Karnataka Police), process for submitting FIR copy to college |
| **Source type** | Official (college procedure) + External (Karnataka Police portal) |
| **Suggested source URLs** | College: `https://rvce.edu.in/examination-circulars/`; External reference: `https://www.ksp.gov.in/` (Karnataka State Police — do NOT crawl) |
| **Verified/Suggested** | Suggested |
| **Static/Dynamic** | Static |
| **Collection method** | Manual KB entry from official circulars; do not scrape police portal |
| **Required metadata** | `document_type`, `fir_required`, `nearest_police_station`, `online_fir_portal_reference`, `office_to_submit_copy` |
| **Update frequency** | Rarely changes |
| **Priority** | **P1** |
| **Permissions** | Public |

### 2.5 Grade Revaluation and Photocopy of Answer Scripts

| Field | Details |
|---|---|
| **Specific data** | Revaluation application form, deadline (usually within 15 days of result), fee amount, photocopy request process, office submission location, result announcement timeline |
| **Source type** | Official — exam circular |
| **Suggested source URLs** | `https://rvce.edu.in/examination-circulars/` (high content page, 21,874 chars; revaluation info likely present) |
| **Verified/Suggested** | Verified page exists; content presence to be confirmed |
| **Static/Dynamic** | Semi-static (deadline and fee change per year) |
| **Collection method** | Extract from existing crawled examination circulars page; supplement with targeted PDF collection |
| **Required metadata** | `process_step`, `deadline_days_after_result`, `fee_per_subject`, `submission_office`, `result_timeline_days` |
| **Update frequency** | Per semester |
| **Priority** | **P0** |
| **Permissions** | Public |

### 2.6 Fee Payment Procedures and Penalties

| Field | Details |
|---|---|
| **Specific data** | Online payment portal (if any), bank account details, challan submission process, late payment fine amounts, installment eligibility, fee waiver/exemption process |
| **Source type** | Official |
| **Suggested source URLs** | `https://rvce.edu.in/academics_and_examinations/fee_payment_circulars/` (crawled, 1,822 chars); fee PDFs already in corpus (need OCR) |
| **Verified/Suggested** | Verified — 9 fee PDFs exist in corpus but are EMPTY. OCR of existing PDFs is the primary action |
| **Static/Dynamic** | Dynamic — amounts change annually |
| **Collection method** | **Step 1: OCR existing 9 fee PDFs** (highest leverage action); Step 2: re-crawl fee circulars page each semester |
| **Required metadata** | `fee_type`, `amount`, `due_date`, `bank_account`, `fine_per_day`, `academic_year` |
| **Update frequency** | Per semester |
| **Priority** | **P0** |
| **Permissions** | Public |

---

## Part 3: Student Services

### 3.1 Scholarships

| Field | Details |
|---|---|
| **Specific data** | Post-Matric Scholarship (SSP portal), SC/ST/OBC government scholarships, merit scholarships, private/trust scholarships offered by RVCE, application procedure, eligibility (income limits, caste certificates, academic criteria), deadlines |
| **Source type** | Official (RVCE student services + Karnataka government SSP portal reference) |
| **Suggested source URLs** | `https://rvce.edu.in/scholarship/` (unconfirmed — to be verified during crawl); `https://rvce.edu.in/academics_and_examinations/academic_student_services/` |
| **Verified/Suggested** | Suggested — these URLs are guessed from the site structure; must be verified |
| **Static/Dynamic** | Semi-dynamic — amounts and deadlines change annually |
| **Collection method** | Web crawler with targeted seed; do NOT scrape SSP portal |
| **Required metadata** | `scholarship_name`, `eligibility`, `amount`, `deadline`, `portal_url`, `documents_required`, `income_limit` |
| **Update frequency** | Annual (October–December application window typically) |
| **Priority** | **P0** — extremely common student query |
| **Permissions** | Public; SSP portal is external — reference URL only, do not scrape |

### 3.2 Student Grievance and Complaint Procedure

| Field | Details |
|---|---|
| **Specific data** | Internal Complaints Committee (ICC), anti-ragging procedure and helpline, academic grievance committee, administrative grievance procedure, ombudsperson contact |
| **Source type** | Official — RVCE statutory committees pages |
| **Suggested source URLs** | `https://rvce.edu.in/anti-ragging/` (unconfirmed); `https://rvce.edu.in/internal_complaints_committee/` (unconfirmed); AICTE/UGC grievance portals (external — reference only) |
| **Verified/Suggested** | Suggested — URLs guessed from RVCE site structure |
| **Static/Dynamic** | Static |
| **Collection method** | Targeted web crawler |
| **Required metadata** | `grievance_type`, `committee_name`, `contact_email`, `contact_phone`, `external_portal` |
| **Update frequency** | Annually (committee membership may change) |
| **Priority** | **P1** |
| **Permissions** | Public |

### 3.3 NAD DigiLocker and Digital Records

| Field | Details |
|---|---|
| **Specific data** | NAD DigiLocker registration steps for RVCE, how to access grade sheets and provisional certificate, Academic Bank of Credits (ABC), linking procedure |
| **Source type** | Official — RVCE examination section + NAD reference |
| **Suggested source URLs** | `https://rvce.edu.in/academics_and_examinations/academic_student_services/` (partially collected, 1,385 chars) |
| **Verified/Suggested** | Suggested (page exists but content is sparse in crawled version) |
| **Static/Dynamic** | Semi-static |
| **Collection method** | Manual KB entry; do not scrape NAD portal |
| **Required metadata** | `service_name`, `registration_steps`, `portal_url`, `documents_accessible` |
| **Update frequency** | Annual |
| **Priority** | **P1** |
| **Permissions** | Public |

### 3.4 Student ID Card and Identity Documents

| Field | Details |
|---|---|
| **Specific data** | How to get ID card for first years, replacement procedure for lost ID, library card issuance |
| **Source type** | Official |
| **Suggested source URLs** | `https://rvce.edu.in/academics_and_examinations/academic_student_services/` |
| **Verified/Suggested** | Suggested |
| **Static/Dynamic** | Static |
| **Collection method** | Manual KB entry |
| **Required metadata** | `document_type`, `issuing_office`, `fee`, `required_docs`, `turnaround` |
| **Update frequency** | Rarely |
| **Priority** | **P2** |
| **Permissions** | Public |

---

## Part 4: Campus Facilities

> [!IMPORTANT]
> **Zero documents** exist for any facilities category in Phase I. This is the most significant topical gap.

### 4.1 Hostel Rules and Accommodation

| Field | Details |
|---|---|
| **Specific data** | Boys' and girls' hostel application process, availability and allotment criteria, room types and fees, mess timings, in-time/curfew rules, warden contact details, visitor policy, leave procedures, complaint escalation |
| **Source type** | Official — RVCE hostel office |
| **Suggested source URLs** | `https://rvce.edu.in/hostel/` or `https://rvce.edu.in/facilities/hostel/` (unconfirmed; likely exists but was NOT crawled in Phase I despite `hostel` being a priority keyword — possible JS-rendered page) |
| **Verified/Suggested** | Suggested — Phase I crawler had `hostel` as priority keyword but collected zero hostel pages; page may be dynamically rendered or URL path differs |
| **Static/Dynamic** | Semi-static (rules static; fees update annually) |
| **Collection method** | Targeted web crawler; if JS-rendered, use manual collection or document submission from hostel warden |
| **Required metadata** | `hostel_type` (boys/girls), `capacity`, `room_type`, `fee_per_semester`, `curfew_time`, `warden_contact`, `mess_timings` |
| **Update frequency** | Annual (fees); Rules rarely change |
| **Priority** | **P0** — hostel queries among top 5 student concerns |
| **Permissions** | Public information; hostel office may provide official document on request |

### 4.2 Mess and Canteen

| Field | Details |
|---|---|
| **Specific data** | Hostel mess menu, meal timings, mess fee (included vs separate), canteen locations on campus, canteen operating hours, food options, special dietary provisions |
| **Source type** | Official (mess timings and fees) + Community (menu and reviews) |
| **Suggested source URLs** | None confirmed for official canteen data; hostel office brochure or warden notice for mess details |
| **Verified/Suggested** | Suggested — no official URL known |
| **Static/Dynamic** | Dynamic (weekly menu changes; fees semi-static) |
| **Collection method** | **Official part:** Manual document from hostel/canteen office; **Community part:** student survey or structured feedback form (see Category 7) |
| **Required metadata** | `facility_name`, `location_on_campus`, `meal_type`, `timings`, `fee_per_meal`, `dietary_options` |
| **Update frequency** | Weekly (menu); semester (fees) |
| **Priority** | **P0** — highly queried by hostel residents |
| **Permissions** | Official data: public; community data: student consent required |

### 4.3 Central Library

| Field | Details |
|---|---|
| **Specific data** | Working hours (regular and exam season), borrowing rules (number of books, loan period), e-resources access (NPTEL, ScienceDirect, IEEE Xplore), reading room rules, holiday schedule |
| **Source type** | Official — RVCE library |
| **Suggested source URLs** | `https://rvce.edu.in/library/` (unconfirmed; likely exists, was NOT crawled) |
| **Verified/Suggested** | Suggested |
| **Static/Dynamic** | Semi-static (hours change for exam season) |
| **Collection method** | Targeted web crawler; supplement with manual verification |
| **Required metadata** | `facility`, `hours_regular`, `hours_exam`, `borrowing_limit`, `e_resources`, `contact` |
| **Update frequency** | Annually and during exam season |
| **Priority** | **P1** |
| **Permissions** | Public |

### 4.4 Medical/Health Centre

| Field | Details |
|---|---|
| **Specific data** | Location on campus, operating hours, available services (first aid, consultation, health insurance), doctor contact, emergency protocol, nearest hospital |
| **Source type** | Official |
| **Suggested source URLs** | `https://rvce.edu.in/health-centre/` or `https://rvce.edu.in/facilities/` (unconfirmed) |
| **Verified/Suggested** | Suggested |
| **Static/Dynamic** | Static |
| **Collection method** | Targeted web crawler or manual KB entry |
| **Required metadata** | `location`, `hours`, `services_offered`, `doctor_contact`, `emergency_contact`, `nearest_hospital` |
| **Update frequency** | Annual |
| **Priority** | **P1** |
| **Permissions** | Public |

### 4.5 Transport / Bus Routes

| Field | Details |
|---|---|
| **Specific data** | RVCE college bus routes, pickup/drop points, bus pass issuance procedure, timing, nearest metro station (already partially answered in answering.py as Rajarajeshwari Nagar metro) |
| **Source type** | Official — transport office or student affairs |
| **Suggested source URLs** | `https://rvce.edu.in/transport/` (unconfirmed) |
| **Verified/Suggested** | Suggested |
| **Static/Dynamic** | Semi-static (routes change at semester start) |
| **Collection method** | Manual document from transport office; web crawl if page exists |
| **Required metadata** | `route_number`, `pickup_points`, `drop_points`, `timing`, `bus_pass_fee`, `contact` |
| **Update frequency** | Per semester |
| **Priority** | **P1** |
| **Permissions** | Public |

### 4.6 Sports and Recreation

| Field | Details |
|---|---|
| **Specific data** | Available sports facilities (badminton courts, gym, cricket ground, basketball), booking procedures, Physical Education (PE) credits, inter-college sports participation |
| **Source type** | Official |
| **Suggested source URLs** | `https://rvce.edu.in/physical-education/` or `https://rvce.edu.in/sports/` (unconfirmed) |
| **Verified/Suggested** | Suggested |
| **Static/Dynamic** | Static |
| **Collection method** | Targeted web crawler |
| **Required metadata** | `facility_name`, `location`, `hours`, `booking_procedure`, `pe_credit_applicable` |
| **Update frequency** | Rarely |
| **Priority** | **P2** |
| **Permissions** | Public |

---

## Part 5: Placements and Internships

### 5.1 Placement Statistics (Detailed, Per-Branch)

| Field | Details |
|---|---|
| **Specific data** | Per-branch placement percentage, average and highest CTC, number of companies visited, year-wise trend (2023–26), sector breakdown (IT/core/startups) |
| **Source type** | Official |
| **Suggested source URLs** | `https://rvce.edu.in/placement_and_training/placement_statistics/` (already crawled, 1,792 chars — sparse); department-wise placement pages (partially crawled) |
| **Verified/Suggested** | Verified — pages confirmed, but content is sparse; re-crawl and supplement with individual department placement pages not yet crawled (ISE, EEE, ECE, MECH, BT missing) |
| **Static/Dynamic** | Dynamic — annual updates |
| **Collection method** | Targeted re-crawl of all department `*/placement/` pages; target remaining departments not yet crawled |
| **Required metadata** | `academic_year`, `branch`, `companies_count`, `avg_package_lpa`, `highest_package_lpa`, `placement_percentage`, `top_recruiters` |
| **Update frequency** | Annual (post-placement season, typically May–July) |
| **Priority** | **P0** |
| **Permissions** | Public |

### 5.2 Placement Notices and Company Announcements

| Field | Details |
|---|---|
| **Specific data** | Individual company recruitment notices (eligibility criteria, CGPA cutoff, branch eligibility, registration process, test/interview rounds, dates), internship announcements |
| **Source type** | Official — Placement and Training cell notices |
| **Suggested source URLs** | `https://rvce.edu.in/placement_and_training/` (crawled, 9,347 chars); individual placement notice PDFs linked from this page |
| **Verified/Suggested** | Suggested — the placement page exists and is content-rich; individual notice PDFs have not been collected |
| **Static/Dynamic** | Highly dynamic — new notices weekly during placement season |
| **Collection method** | Targeted web crawler with incremental updates; email ingestor (MCP) for placement cell email announcements (Phase II-E) |
| **Required metadata** | `company_name`, `job_role`, `branches_eligible`, `cgpa_cutoff`, `backlogs_allowed`, `registration_deadline`, `test_date`, `process_steps` |
| **Update frequency** | Weekly during placement season (Aug–Feb) |
| **Priority** | **P0** |
| **Permissions** | Authorized students only for actual registration; announcement data is public |

### 5.3 Placement Eligibility Rules

| Field | Details |
|---|---|
| **Specific data** | Global RVCE placement eligibility (minimum CGPA, backlog limits, attendance requirement), dream company vs mass recruiter policies, one-offer rule, de-registration rules |
| **Source type** | Official — Placement and Training cell |
| **Suggested source URLs** | `https://rvce.edu.in/placement_and_training/` (already crawled) |
| **Verified/Suggested** | Verified — page exists and is content-rich |
| **Static/Dynamic** | Semi-static |
| **Collection method** | Already partially collected; extract structured policy |
| **Required metadata** | `rule_name`, `threshold`, `exceptions`, `applicable_from` |
| **Update frequency** | Annual |
| **Priority** | **P0** |
| **Permissions** | Public |

### 5.4 Internship Programs

| Field | Details |
|---|---|
| **Specific data** | 6-month internship program details, registration procedure, credit recognition, approved company list, internship report submission guidelines |
| **Source type** | Official |
| **Suggested source URLs** | `https://rvce.edu.in/placement_and_training/` (rich page) |
| **Verified/Suggested** | Suggested (partial data may be in crawled page) |
| **Static/Dynamic** | Semi-static |
| **Collection method** | Extract from existing crawled page; targeted supplement |
| **Required metadata** | `programme`, `duration`, `credit_value`, `registration_procedure`, `report_submission` |
| **Update frequency** | Annual |
| **Priority** | **P1** |
| **Permissions** | Public |

---

## Part 6: Faculty and Departments

### 6.1 Department Home Pages (Missing Departments)

| Field | Details |
|---|---|
| **Specific data** | Department overview, programmes offered, vision/mission, facilities, key achievements |
| **Source type** | Official |
| **Departments NOT yet crawled** | Mechanical Engineering (ME), Information Science (ISE home page — only placement crawled), ECE full page, EEE full page, ETE, BT full page, MCA, MBA (if offered) |
| **Suggested source URLs** | `https://rvce.edu.in/department/me/`, `https://rvce.edu.in/department/ise/`, `https://rvce.edu.in/department/mca/` (all unconfirmed slugs) |
| **Verified/Suggested** | Suggested — slug format guessed from existing pattern `https://rvce.edu.in/department/<slug>/` |
| **Static/Dynamic** | Static |
| **Collection method** | Targeted web crawler |
| **Required metadata** | `department_name`, `slug`, `programmes`, `hod_name`, `contact` |
| **Update frequency** | Annually |
| **Priority** | **P1** |
| **Permissions** | Public |

### 6.2 Faculty Directory and HOD Contacts

| Field | Details |
|---|---|
| **Specific data** | Faculty names, designations, qualifications, research areas, email addresses, phone extensions, HOD name and contact per department |
| **Source type** | Official |
| **Suggested source URLs** | `https://rvce.edu.in/programmes-faculty-details/` (already crawled, 1,263 chars — links to PDFs not collected); department-specific faculty pages (e.g., `https://rvce.edu.in/department/cse/faculty/`) |
| **Verified/Suggested** | Suggested — faculty detail PDFs linked from programmes-faculty-details page were not downloaded |
| **Static/Dynamic** | Dynamic (appointments, promotions) |
| **Collection method** | Targeted PDF crawler for faculty list PDFs; OCR if scanned |
| **Required metadata** | `faculty_name`, `designation`, `department`, `qualification`, `research_areas`, `email`, `phone_ext`, `is_hod` |
| **Update frequency** | Annual (or on appointment change) |
| **Priority** | **P1** |
| **Permissions** | Publicly listed on official website; no privacy concern for published contact info |

### 6.3 Research and Lab Facilities

| Field | Details |
|---|---|
| **Specific data** | Research labs, funded projects, collaboration details, laboratory equipment |
| **Source type** | Official |
| **Suggested source URLs** | Department research pages — already partially crawled (7 research pages exist: AI/ML, AE, BT, CE, Civil, CSE, IEM, EEE); remaining departments not crawled |
| **Verified/Suggested** | Verified — existing research pages confirmed |
| **Static/Dynamic** | Dynamic (new projects) |
| **Collection method** | Complete the crawl for all remaining department research pages |
| **Required metadata** | `department`, `lab_name`, `funded_by`, `pi_name`, `grant_amount` |
| **Update frequency** | Annually |
| **Priority** | **P2** |
| **Permissions** | Public |

---

## Part 7: Student Community Feedback

> [!IMPORTANT]
> This is the only category requiring **community-generated data**. No such data exists anywhere in Phase I. All collection here requires explicit student consent.

### 7.1 Canteen and Mess Feedback

| Field | Details |
|---|---|
| **Specific data** | Student ratings and reviews of canteen food quality, mess menu satisfaction, hygiene, pricing fairness, variety, service speed; complaint categories |
| **Source type** | Community-generated — student feedback |
| **Suggested source URL** | No existing source; must create: a structured feedback form (Google Forms / Typeform) or a purpose-built `/feedback` API endpoint |
| **Verified/Suggested** | Does not exist; must be created |
| **Static/Dynamic** | Dynamic — ongoing submissions |
| **Collection method** | Student submission form; with explicit consent notice |
| **Required metadata** | `feedback_category`, `rating_1_to_5`, `comment_text`, `semester`, `is_hostel_resident`, `timestamp` (anonymized — no student ID) |
| **Update frequency** | Ongoing (collect monthly batches) |
| **Priority** | **P1** — needed for sentiment analysis model |
| **Permissions** | **Requires:** Informed consent from students; IRB/ethics approval if used for research publication; data anonymization before storage |

### 7.2 Hostel Feedback

| Field | Details |
|---|---|
| **Specific data** | Student ratings of hostel cleanliness, room condition, warden response, security, infrastructure, internet connectivity, complaints about hostel rules |
| **Source type** | Community-generated |
| **Suggested source URL** | Must be created (same feedback form as canteen, separate category) |
| **Verified/Suggested** | Does not exist |
| **Static/Dynamic** | Dynamic |
| **Collection method** | Student submission form with consent |
| **Required metadata** | `hostel_type` (boys/girls), `feedback_category`, `rating_1_to_5`, `comment_text`, `semester`, `timestamp` (anonymized) |
| **Update frequency** | Ongoing |
| **Priority** | **P1** |
| **Permissions** | Informed consent; anonymization mandatory |

### 7.3 General Campus Facility Feedback

| Field | Details |
|---|---|
| **Specific data** | Library feedback, sports facilities, medical centre, transport, WiFi connectivity, cleanliness of campus areas |
| **Source type** | Community-generated |
| **Suggested source URL** | Same feedback form, `facility` category |
| **Verified/Suggested** | Does not exist |
| **Static/Dynamic** | Dynamic |
| **Collection method** | Student submission form |
| **Required metadata** | `facility_name`, `rating_1_to_5`, `comment_text`, `issue_type`, `timestamp` (anonymized) |
| **Update frequency** | Ongoing |
| **Priority** | **P2** |
| **Permissions** | Informed consent; anonymization mandatory |

### 7.4 What Cannot Currently Be Obtained

The following community data sources are **currently inaccessible** or ethically/legally impractical:

| Source | Reason Inaccessible |
|---|---|
| Google Maps reviews for RVCE campus | Copyright; cannot scrape or store Google content |
| Quora discussions about RVCE | Terms of service prohibit scraping |
| Reddit (`r/RVCE`, `r/Bengaluru`) | Terms of service; public but scraping for model training raises ToS issues |
| Glassdoor alumni reviews | Login-gated; ToS prohibits scraping |
| WhatsApp group messages | Private; no consent; impossible to collect ethically |
| Anonymous complaint portals | If any exist internally, data is confidential |
| Student Union official statements | No public digital record found |

---

## Part 8: College Announcements and Circulars

### 8.1 Academic and Administrative Circulars

| Field | Details |
|---|---|
| **Specific data** | All circulars from RVCE: academic (exam schedules, calendar updates, holiday notices), administrative (fee deadlines, compliance), event announcements |
| **Source type** | Official |
| **Suggested source URLs** | `https://rvce.edu.in/examination-circulars/` (already crawled — richest page, 21,874 chars); `https://rvce.edu.in/academic-circular/` (already crawled); `https://rvce.edu.in/admission-circulars/` (already crawled) |
| **Verified/Suggested** | Verified |
| **Static/Dynamic** | Highly dynamic — new circulars issued weekly |
| **Collection method** | Incremental re-crawl of confirmed circular index pages; download linked PDFs with OCR fallback |
| **Required metadata** | `circular_number`, `circular_date`, `subject`, `applicable_to` (UG/PG/all), `issuing_office`, `effective_date` |
| **Update frequency** | Weekly |
| **Priority** | **P0** |
| **Permissions** | Public |

### 8.2 Events, Workshops, and Cultural Fest Notices

| Field | Details |
|---|---|
| **Specific data** | Technical events (hackathons, workshops, symposiums), cultural fests, guest lectures, seminars, registration links, dates, organizers |
| **Source type** | Official + Semi-official (student clubs) |
| **Suggested source URLs** | `https://rvce.edu.in/events/` (unconfirmed); department notice boards (each dept may post separately); student club pages |
| **Verified/Suggested** | Suggested |
| **Static/Dynamic** | Highly dynamic |
| **Collection method** | Targeted web crawler; social media monitoring is out of scope for Phase II |
| **Required metadata** | `event_name`, `event_type`, `organizing_dept_or_club`, `date`, `venue`, `registration_link`, `contact` |
| **Update frequency** | Weekly during academic semester |
| **Priority** | **P1** |
| **Permissions** | Public |

### 8.3 NIRF, NAAC, and Accreditation Reports

| Field | Details |
|---|---|
| **Specific data** | NIRF ranking submission data, NAAC accreditation grade, NBA accreditation per department, AICTE disclosure |
| **Source type** | Official |
| **Suggested source URLs** | `https://rvce.edu.in/nirf/` or `https://rvce.edu.in/naac/` (unconfirmed); linked from homepage which was crawled |
| **Verified/Suggested** | Suggested |
| **Static/Dynamic** | Mostly static (annual updates) |
| **Collection method** | Targeted web crawler |
| **Required metadata** | `report_type`, `year`, `rank` or `grade`, `accrediting_body`, `valid_until` |
| **Update frequency** | Annual |
| **Priority** | **P2** |
| **Permissions** | Public |

---

## Part 9: General College Information

### 9.1 Admissions — Detailed Procedure Guides

| Field | Details |
|---|---|
| **Specific data** | Step-by-step COMEDK/KCET/MQ/NRI/DCET/PGCET procedures: document checklist, balance fee payment, reporting schedule |
| **Source type** | Official |
| **Existing data** | 6 PDFs (COMEDK, KCET/KEA, MQ, NRI, DCET, M.Tech/MCA) already in corpus — all EMPTY due to scanned images |
| **Suggested action** | **OCR all 6 admission PDFs** — this is the highest-ROI action for this category |
| **Suggested source URLs** | All PDF URLs are already in `data/raw/collection_manifest.json` |
| **Verified/Suggested** | Verified |
| **Collection method** | OCR existing PDFs (highest leverage); re-crawl `https://rvce.edu.in/admission-circulars/` for updated notices |
| **Required metadata** | `quota`, `step_number`, `instruction`, `required_document`, `office`, `contact` |
| **Update frequency** | Annual (admission cycle) |
| **Priority** | **P0** |
| **Permissions** | Public |

### 9.2 Clubs and Student Organizations

| Field | Details |
|---|---|
| **Specific data** | List of technical clubs (Astra Robotics, Team Chimera mentioned in query dataset), cultural clubs, NSS, NCC, IEEE student chapter, joining procedure, faculty coordinator |
| **Source type** | Official + Semi-official |
| **Suggested source URLs** | `https://rvce.edu.in/student-clubs/` or `https://rvce.edu.in/national_service_scheme_nss/` (suggested in Phase I dataset audit); individual club pages |
| **Verified/Suggested** | Suggested |
| **Static/Dynamic** | Semi-static |
| **Collection method** | Targeted web crawler |
| **Required metadata** | `club_name`, `category`, `faculty_coordinator`, `joining_procedure`, `contact_email` |
| **Update frequency** | Annual |
| **Priority** | **P1** |
| **Permissions** | Public |

### 9.3 College Regulations and Code of Conduct

| Field | Details |
|---|---|
| **Specific data** | Student code of conduct, dress code, mobile phone policy, examination malpractice rules and consequences, disciplinary procedures |
| **Source type** | Official |
| **Suggested source URLs** | `https://rvce.edu.in/regulations/` (unconfirmed); may be embedded in student handbook PDF |
| **Verified/Suggested** | Suggested |
| **Static/Dynamic** | Static |
| **Collection method** | Targeted web crawler; manual KB entry if handbook PDF |
| **Required metadata** | `rule_category`, `rule_text`, `consequence`, `effective_from` |
| **Update frequency** | Rarely |
| **Priority** | **P2** |
| **Permissions** | Public |

### 9.4 Campus Map and Navigation

| Field | Details |
|---|---|
| **Specific data** | Block/building names and their departments, key landmarks (principal office block, library, canteen, hostel, main gate), nearest metro station, college address |
| **Source type** | Official |
| **Suggested source URLs** | `https://rvce.edu.in/campus-map/` (unconfirmed) |
| **Verified/Suggested** | Suggested |
| **Static/Dynamic** | Static |
| **Collection method** | Manual KB entry; campus map image (not crawled — image format); text description from official page |
| **Required metadata** | `location_name`, `building_block`, `departments_housed`, `notable_features` |
| **Update frequency** | Rarely |
| **Priority** | **P2** |
| **Permissions** | Public |

---

## Part 10: Priority Summary and Execution Roadmap

### 10.1 Priority Matrix

| Priority | Category | Data Items | Collection Method | Effort |
|---|---|---|---|---|
| **P0** | Existing OCR | 15 scanned PDFs (9 fee + 6 admissions) | OCR existing corpus | Low effort, high ROI |
| **P0** | Administrative procedures | Duplicate grade card, bonafide, revaluation, FIR | Manual KB entry + targeted crawl | Medium |
| **P0** | Academic info | Syllabus, calendar, attendance rules, CIE/SEE | Targeted web crawl + PDF | Medium |
| **P0** | Placements | Per-branch stats, notices, eligibility rules | Targeted re-crawl | Low (pages exist) |
| **P0** | Fee procedures | Exact amounts, deadlines, payment process | OCR (already in corpus) | Low |
| **P0** | Hostel information | Rules, fees, application process | Targeted crawl + manual | Medium |
| **P0** | Admissions (text) | Step-by-step per quota | OCR existing PDFs | Low |
| **P0** | Scholarships | SSP, government, merit scholarships | Targeted crawl + manual KB | Medium |
| **P0** | Circulars (incremental) | Weekly exam/fee/academic notices | Incremental re-crawl | Ongoing |
| **P1** | Canteen/mess | Timings, fees, location | Manual KB + feedback form | Medium |
| **P1** | Library | Hours, rules, e-resources | Targeted crawl | Low |
| **P1** | Faculty directory | HOD contacts, faculty list | PDF crawl + OCR | Medium |
| **P1** | Missing dept pages | ME, ISE, MCA full pages | Targeted crawl | Low |
| **P1** | Student feedback | Canteen, hostel sentiment data | Feedback form creation | Medium |
| **P1** | Grievance procedure | ICC, anti-ragging, committees | Targeted crawl | Low |
| **P1** | Clubs and NSS/NCC | Club list, joining procedures | Targeted crawl | Low |
| **P1** | Events and workshops | Seminar/workshop notices | Targeted crawl | Ongoing |
| **P2** | Research labs | Funding, projects, labs | Complete dept crawl | Low |
| **P2** | Sports/recreation | Facilities, PE credits | Targeted crawl | Low |
| **P2** | Transport/bus | Routes, bus pass | Manual KB | Low |
| **P2** | NIRF/NAAC | Accreditation reports | Targeted crawl | Low |
| **P2** | Regulations | Code of conduct, rules | Manual KB | Low |
| **P2** | Campus map | Block/building guide | Manual KB | Low |

### 10.2 Immediate Actions (Before Any New Crawl)

> [!IMPORTANT]
> **Step 1 (Highest leverage):** Run OCR on the 15 existing EMPTY PDFs in the corpus. This unlocks fee data and admission procedures **without any new crawl**.

```
OCR targets (already in data/raw/campus_documents.jsonl):
  Fee circulars (9 PDFs):
    - REVISED-PG-HIGHER-SEMESTER-FEES-CIRCULAR-29.09.26.pdf
    - Fee-payment-extension-circular-28.09.26.pdf
    - PG-Higher-semester-Fee-circular_revised.pdf
    - UG-PG-Higher-semester-Fee-circular2026-27.pdf
    - UG-Higher-Semester-Fee-Circular_2025.pdf
    - UG-higher-sem-fee-extension-circular.pdf
    - NRI-quota-3rd-semester-Fee-Circular.pdf
    - PG-3rd-sem-fee-circular-2025-26.pdf
    - (1 more fee PDF)
  
  Admission procedure PDFs (6 PDFs):
    - 3-M.Tech_MCA-Procedure-and-Balance-Fees-Details-for-Admission-2026-2027.pdf
    - 6-DCET-ADMISSION-PROCESS-FEES-DETAILS-FOR-2ND-YEAR-B.E-2026-27.pdf
    - 3-COMEDK-QUOTA-ADMISSION-PROCESS-FEES-DETAILS-FOR-1ST-YEAR-B.E-2026-27.pdf
    - 2-KEA-QUOTA-ADMISSION-PROCESS-FEES-DETAILS-FOR-1ST-YEAR-B.E-2026-27.pdf
    - 4-MQ-QUOTA-ADMISSION-PROCESS-FEES-DETAILS-FOR-1ST-YEAR-B.E-2026-27.pdf
    - 5-NRI-QUOTA-ADMISSION-PROCESS-FEES-DETAILS-FOR-1ST-YEAR-B.E-2026-27.pdf
    - 1ST-YEAR-BE-PROGRAM-MANAGEMENT-QUOTA-ADMISSION-2026-27.pdf
```

> [!NOTE]
> **Step 2:** Run targeted crawl of the following confirmed-missing high-priority URLs:
> - `https://rvce.edu.in/hostel/` (or equivalent)
> - `https://rvce.edu.in/library/`
> - `https://rvce.edu.in/scholarship/`
> - All department home pages not yet crawled (ME, ISE home, MCA, etc.)
> - All department faculty pages

### 10.3 Metadata Schema for New Documents

All new documents must conform to the existing schema plus these Phase II additions:

```jsonc
{
  "document_id": "doc_<type>_<hash>",
  "source_url": "https://...",
  "title": "...",
  "content": "...",
  "source_type": "HTML | PDF | MANUAL_KB | STUDENT_FEEDBACK",
  "collected_at": "<ISO 8601>",
  "last_modified": "<ISO 8601 or null>",
  "topic": "<topic_label>",
  "extraction_status": "SUCCESS | EMPTY | OCR_SUCCESS | MANUAL",
  // Phase II additions:
  "subcategory": "hostel_rules | canteen | library | admin_procedure | ...",
  "data_class": "official | community | structured_kb",
  "requires_consent": false,          // true for community feedback only
  "update_frequency": "static | annual | semester | weekly",
  "phase": 2
}
```

### 10.4 What Cannot Be Obtained (Summary)

| Data Item | Reason |
|---|---|
| Real-time KCET/COMEDK cutoffs | Released by KEA/COMEDK, not RVCE; external websites |
| VTU exam results | VTU portal; login-gated; external domain |
| Individual student CIE marks | Private; student data |
| Internal complaint records | Confidential |
| WhatsApp/social media discussions | ToS; privacy |
| Google Maps / Quora / Reddit content | Copyright / ToS |
| Faculty personal contact outside official directory | Privacy |
| Real-time bus tracking | No known public API |
| Hostel allotment status | Student-specific; login-gated |
| Mess menu (weekly real-time) | No official digital source found |

---

## Appendix A: De-duplication Rules for Phase II Collection

1. **Never re-collect** URLs already in `data/raw/collection_manifest.json` unless it is an incremental update of a dynamic page (circulars, calendar).
2. **Dynamic pages** (circulars, calendar, placement notices) must use a `last_collected_at` comparison before re-crawl.
3. **WordPress URL aliases** — collect canonical permalink only (e.g., prefer `/academics_and_examinations/academic_student_services/` over `/academic_student_services/`).
4. **Department academic planning pages** — highly near-duplicate (Jaccard ≥ 0.86). Collect all, but mark with `is_template_boilerplate: true` for chunk filtering.

## Appendix B: Source Verification Status

| Source | Verification Method | Status |
|---|---|---|
| `https://rvce.edu.in/examination-circulars/` | Confirmed crawled in Phase I | ✅ Verified |
| `https://rvce.edu.in/calendar-of-events/` | Confirmed crawled in Phase I | ✅ Verified |
| `https://rvce.edu.in/placement_and_training/` | Confirmed crawled in Phase I | ✅ Verified |
| `https://rvce.edu.in/hostel/` | Not crawled; inferred from site structure | ⚠️ To Be Verified |
| `https://rvce.edu.in/library/` | Not crawled; inferred | ⚠️ To Be Verified |
| `https://rvce.edu.in/scholarship/` | Not crawled; inferred | ⚠️ To Be Verified |
| `https://rvce.edu.in/student-clubs/` | Not crawled; inferred | ⚠️ To Be Verified |
| `https://rvce.edu.in/health-centre/` | Not crawled; inferred | ⚠️ To Be Verified |
| `https://rvce.edu.in/transport/` | Not crawled; inferred | ⚠️ To Be Verified |
| Department faculty sub-pages | Not crawled; inferred from dept pattern | ⚠️ To Be Verified |
| Student feedback form | Does not exist yet; must be created | ❌ Needs Creation |
| Karnataka Police FIR portal | External; reference only, do NOT crawl | 🚫 External Only |
| SSP Scholarship portal | External; reference only | 🚫 External Only |

---

## Appendix C: Prompt and Action Execution Log

- **Prompt 1 (2026-11-03 23:00 IST):** Read-only repository audit of Phase I components (LR classifier, retrievers, evaluation, FastAPI, MLflow, DVC, CI/CD). Delivered audit report in brain artifact `phase2_audit_report.md`.
- **Prompt 2 (2026-11-03 23:20 IST):** Created comprehensive Phase II Data Acquisition Plan covering all 9 campus information categories, metadata schemas, and deduplication rules in `reports/phase2_data_acquisition_plan.md`.
- **Prompt 3 (2026-11-03 23:30 IST):** Extracted reference table of all 15 scanned PDFs (9 fee circulars + 6 admission guides) with direct download URLs and official source pages.
- **Prompt 4 (2026-11-03 23:38 IST):** User manually downloaded all 15 scanned PDFs and placed them into `data/raw/`.
- **Prompt 5 (2026-11-04 00:05 IST):** 
  - Verified all 15 scanned PDFs present in `data/raw/` (0 native text chars confirmed).
  - Installed and verified `rapidocr-onnxruntime` engine.
  - Verified and benchmarked OCR pipeline (`src/ocr.py`).
  - Confirmed all 60 documents (45 HTML + 15 PDF) are now fully extracted with status `SUCCESS` in `data/processed/campus_documents.jsonl`.
  - Synchronized provenance manifest (`data/raw/collection_manifest.json`) and Quality Assurance report (`reports/data_collection_report.json`) to reflect 60/60 successful extractions (0 empty).
  - Executed full test suite (`pytest tests/`) — 54 passed.
  - Cleaned up temporary scratch validation scripts.
- **Prompt 6 (2026-11-04 00:25 IST):** Conducted planning and compatibility audit for Dynamic Campus Notice Ingestion using Gmail MCP. Produced comprehensive design report `reports/phase2_gmail_mcp_ingestion_plan.md`.
- **Prompt 7 (2026-11-04 00:30 IST):** Created minimal, actionable Phase II Gmail MCP implementation checklist in `reports/phase2_gmail_mcp_implementation_checklist.md` with explicit safe revision guardrails, secret protection, and identification of the single first implementation task.



