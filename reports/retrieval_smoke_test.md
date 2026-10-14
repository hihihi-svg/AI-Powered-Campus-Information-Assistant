# Retrieval smoke test report

Date: 2026-10-13
Dataset: data/processed/campus_documents.jsonl
Retriever: TFIDFRetriever in src/retrieval.py
Validation: pytest -q tests/test_retrieval.py and python scratch/smoke_test.py

## Current state

- OCR integration and rebuilt processed dataset are present and unchanged by this task.
- The deduplication fix was added to src/retrieval.py without altering the OCR workflow or raw data files.
- The current dataset indexes 247 chunks from 60 records.

## Smoke-test questions and outcomes

| ID | Question | Outcome |
|---|---|---|
| Q-ADM-1 | What is the COMEDK rank cutoff for Computer Science at RVCE? | Irrelevant |
| Q-ADM-2 | How do I apply for lateral entry through DCET? | Relevant |
| Q-FEE-1 | What is the tuition fee for B.E. first year at RVCE? | Relevant |
| Q-FEE-2 | What is the late fee penalty for delayed tuition payment? | Relevant |
| Q-EXA-1 | What is the minimum attendance required for semester exams? | Weakly topical |
| Q-EXA-2 | How can I apply for revaluation of my exam paper? | Weakly topical |
| Q-DEP-1 | Who is the head of the Computer Science department? | Irrelevant |
| Q-PLA-1 | What companies recruited from RVCE in the last placement season? | Weakly topical |
| Q-FAC-1 | Is there a hostel facility available for girls at RVCE? | Irrelevant |
| Q-GEN-1 | What is the address and contact number of RVCE? | Irrelevant |

## Relevant results

- Q-ADM-2: a valid admission circular directly mentions DCET lateral-entry procedure and balance-fee details.
- Q-FEE-1: the top-3 results include fee-circular PDF chunks that mention tuition/fees for higher semester and first-year academic processes; the generic academic-circular page still ranks first, but relevant fee text is present.
- Q-FEE-2: the top results include fee circular text that mentions fee-payment and delayed-payment context, which is relevant to the question.

## Weakly topical results

- Q-EXA-1: the retriever surfaces the home-page examination section and some fee circular text, but it does not contain the required attendance rule.
- Q-EXA-2: it finds examination circulars and revaluation references, but not the procedure itself.
- Q-PLA-1: it returns placement statistics and company/offer language, but not a list of employer names or recruiter details.

## Irrelevant results

- Q-ADM-1: the results are research and home-page content around computer science, not rank-cutoff tables.
- Q-DEP-1: the top results are research and home-page snippets about computer science, not department leadership or HOD contact data.
- Q-FAC-1: no results were returned.
- Q-GEN-1: results are placement statistics, not address or phone number data.

## Actual status after fix

- The deduplication patch removed duplicate same-document same-heading repeats while preserving distinct passages.
- The retrieval smoke test shows the rebuilt dataset now returns actual fee-circular and admission-circular content, which is evidence that the OCR rebuild is feeding useful content into retrieval.
- The remaining retrieval issues are mostly corpus gaps rather than deduplication defects: missing faculty-directory data, hostel data, contact page data, and exact company lists.

## Remaining retrieval problems

1. Faculty/department head data is not present in the indexed corpus.
2. Hostel and contact details are missing from the dataset.
3. Exact company names and detailed placement recruiter information are not indexed.
4. The attendance rule and other policy documents still depend on full OCR coverage for the relevant scanned PDFs.

## Recommended next task

Add targeted content acquisition for the missing factual pages rather than more retrieval logic: crawl and OCR the missing RVCE pages for contact, hostel, department leadership, and the specific regulation PDFs that answer attendance and revaluation policy questions.
