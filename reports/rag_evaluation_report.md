# RVCE Campus Information Assistant - RAG Pipeline Evaluation Report

**Evaluation Timestamp:** 2026-11-11T03:35:37Z
**RAG Strategy:** hybrid (Rerank: False)
**Dataset:** data\processed\campus_documents.jsonl (Checksum: f61c42654429)

## Summary Metrics

| Metric | Score | Notes |
| :--- | :--- | :--- |
| **Overall Accuracy** | `1.0000` | Combined answer correctness & safe abstention |
| **Factual Accuracy** | `1.0000` | In-domain campus questions answered accurately |
| **Abstention Accuracy** | `1.0000` | Out-of-domain questions safely refused |
| **Evidence Grounding Rate** | `1.0000` | In-domain answers with verified campus evidence |
| **Average Latency** | `2.2458s` | End-to-end question answering response time |
| **Total Queries Evaluated** | `13` | 9 in-domain, 4 out-of-domain |

## Category Performance Breakdown

| Query ID | Category | Type | Grounded | Abstained | Correct | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `RAG_F01` | admissions_cutoff | factual | Yes | No | PASS | 26.3134s |
| `RAG_F02` | admission_fees | factual | Yes | No | PASS | 0.5813s |
| `RAG_F03` | admissions_procedure | factual | Yes | No | PASS | 0.1814s |
| `RAG_F04` | facilities_hostels | factual | Yes | No | PASS | 0.1586s |
| `RAG_F05` | departments_faculty | factual | Yes | No | PASS | 0.0782s |
| `RAG_P01` | placements_careers | numerical | Yes | No | PASS | 0.0244s |
| `RAG_P02` | placements_careers | numerical | Yes | No | PASS | 0.0087s |
| `RAG_D01` | academics_exams | date_specific | Yes | No | PASS | 0.2047s |
| `RAG_D02` | circulars | date_specific | Yes | No | PASS | 1.3795s |
| `RAG_O01` | abstention | out_of_domain | No | Yes | PASS | 0.0489s |
| `RAG_O02` | abstention | out_of_domain | No | Yes | PASS | 0.0852s |
| `RAG_O03` | abstention | out_of_domain | No | Yes | PASS | 0.0747s |
| `RAG_O04` | abstention | out_of_domain | No | Yes | PASS | 0.0558s |

## Governance & Safety Verification
- **Zero Hallucination Policy:** Out-of-domain queries successfully trigger strict abstention (`NO_EVIDENCE_MESSAGE`).
- **Citation Assurance:** All answered in-domain questions include valid document citations and verifiable evidence passages.
