# Model card: RVCE Campus Information Assistant

## 1. Purpose and intended users

This project provides a local, evidence-based campus information assistant for RV College of Engineering (RVCE) content. It is intended for repository use, experimentation, and local retrieval benchmarking rather than production deployment.

Intended users include:
- project maintainers and students exploring the repository locally
- researchers validating retrieval behavior on a campus knowledge corpus
- developers testing a minimal end-to-end API and evaluation workflow

The system is designed to answer campus questions using retrieved passages from official RVCE content, not to generate unsupported facts from a general-purpose model.

## 2. System architecture and retrieval approach

The implementation is a retrieval-first stack built around the existing project code:

- `src/retrieval.py` implements a TF-IDF retriever using `TfidfVectorizer` and cosine similarity.
- `src/answering.py` performs evidence-only answer construction and abstains when no sufficient evidence is found.
- `src/api.py` exposes a minimal FastAPI service with `/health` and `/ask`.
- `src/evaluation.py` runs the benchmark and records local MLflow metrics.

The retrieval flow is:
1. load the processed JSONL corpus from `data/processed/campus_documents.jsonl`
2. chunk and index the content using text-aware chunking and heading filters
3. retrieve the top results for a query using TF-IDF similarity
4. build a concise answer using the retrieved evidence only
5. abstain if the evidence is insufficient

This is not a production LLM application, and it does not claim answer correctness beyond what the corpus supports.

## 3. Dataset sources, composition, and known coverage gaps

### Sources

The repository collects official RVCE material from the public website `rvce.edu.in` and stores both raw and processed versions under:
- `data/raw/campus_documents.jsonl`
- `data/raw/collection_manifest.json`
- `data/processed/campus_documents.jsonl`

The raw corpus contains source snapshots and manifest provenance. The processed dataset is the canonical corpus used by retrieval and answering.

### Composition

The data collection report in [reports/data_collection_report.json](reports/data_collection_report.json) indicates:
- 60 total documents collected
- 45 HTML documents, 15 PDF documents
- 0 validation issues reported
- 45 successful extractions, 15 empty extractions, 0 failed extractions
- major topic coverage includes admissions, fees/finance, examinations, academics, and departments

### Known coverage gaps

The repository notes several factual coverage gaps, including:
- faculty and leadership contact data not consistently present
- hostel and contact details not fully indexed
- full placement company lists and recruiter details are incomplete
- some policy and procedure answers depend on scanned or missing pages
- the benchmark query set contains labels for intent only, not gold relevance labels

The smoke test and degradation report also record that the current corpus is better for administrative and fee-related material than for exact contact, hostel, or faculty lookup questions.

## 4. Evaluation methodology and actual reported metrics

### Methodology

The evaluation workflow in `src/evaluation.py` does the following:
- loads the benchmark from `data/queries/campus_queries.csv`
- runs the current TF-IDF retriever against the corpus
- counts a query as a hit if at least one result passes the configured score threshold
- logs the run to a local MLflow file-backed store

This is explicitly a retrieval-hit proxy, not a gold-label evaluation. The benchmark CSV contains query text and intent labels, but no answer or relevance labels.

### Actual metrics reported

The local benchmark run executed in this workspace reported:
- queries_evaluated: 40
- queries_with_results: 40
- queries_without_results: 0
- relevance_rate: 1.0
- average_best_score: 0.1474

These numbers were logged by the local MLflow evaluation workflow and are also recorded in the benchmark output and summary artifacts.

## 5. Known limitations

This project has several important limitations that must be considered before using it as a factual decision support system:

- No gold relevance labels are available in the benchmark.
- The `relevance_rate` metric is a retrieval-hit proxy, not a measure of answer correctness.
- The corpus is limited to publicly available source pages and may miss newer or uncollected content.
- Some pages are empty or partially extracted, especially scanned documents.
- The retrieval model is a classic TF-IDF baseline; it does not use dense retrieval or modern reranking.
- The project does not claim production drift detection, real-time monitoring, or automatic retraining.

## 6. Abstention behavior, risks, and appropriate use

The answer layer intentionally abstains when evidence is insufficient.

The implemented abstention message is:
> I could not find sufficient evidence in the available campus documents to answer this question.

This is the correct behavior for unsupported or out-of-scope questions. The API tests verify that unsupported questions return an abstention response rather than a fabricated answer.

### Risks

- unsupported or missing content may be misinterpreted as negative evidence
- retrieval may surface relevant but incomplete passages
- claims about exact fees, rankings, or policies should be verified against the original RVCE documents
- no external validation or fact-checking layer is implemented

### Appropriate use

This system is best used as a local retrieval and evidence-grounding helper for campus information lookup in a controlled, academic setting. It should not be treated as authoritative for final admissions, fee, compliance, or policy decisions without document verification.

## 7. Monitoring experiment and synthetic nature

The degradation experiment compares a baseline corpus against a reduced synthetic copy of the processed data and records the delta in retrieval metrics. This is implemented in `src/evaluation.py` and reported at [reports/retrieval_degradation_report.md](reports/retrieval_degradation_report.md).

Actual reported degradation experiment results:
- baseline relevance_rate: 1.0
- degraded relevance_rate: 0.95
- average_best_score drop: 0.0038
- warning threshold: 0.10
- retraining threshold: 0.25
- decision: `no_action`

This experiment is explicitly synthetic and does not constitute proof of real production drift.

## 8. Versioning and reproducibility

The repository uses DVC to version the canonical dataset files without modifying their contents. DVC was initialized in this workspace and the current status is `Data and pipelines are up to date.`

Relevant dataset metadata:
- [data/raw/campus_documents.jsonl.dvc](data/raw/campus_documents.jsonl.dvc)
- [data/raw/collection_manifest.json.dvc](data/raw/collection_manifest.json.dvc)
- [data/processed/campus_documents.jsonl.dvc](data/processed/campus_documents.jsonl.dvc)

The project also stores local MLflow artifacts under `mlruns/`, which are treated as local generated outputs rather than source data.

### Reproduction instructions

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.evaluation --benchmark data/queries/campus_queries.csv --dataset data/processed/campus_documents.jsonl --top-k 3 --score-threshold 0.05
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

No remote DVC storage or remote backup is configured in this repository.

## 9. Summary

This is a minimal, local, retrieval-based campus information system grounded in a curated corpus of RVCE content. It demonstrates a practical evidence-based pipeline, local evaluation workflow, and synthetic monitoring experiment, but it remains a research and local-development artifact rather than a production-ready student-facing system.
