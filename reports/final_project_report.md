# Final project report: RVCE Campus Information Assistant

## 1. Problem statement and objectives

The project addresses the need for a local, transparent, campus information retrieval system grounded in official RVCE source material. The objective is to collect and structure public campus information, index it in a reproducible way, and provide a minimal retrieval-and-answer workflow with benchmark evaluation and local tracking.

The repository is intentionally scoped to a minimal but real pipeline:
- data collection from the RVCE website
- dataset validation and quality reporting
- TF-IDF retrieval over processed content
- evidence-based answering with abstention
- benchmark evaluation and local MLflow logging
- DVC dataset versioning
- minimal FastAPI API and local monitoring experiment

It does not claim production readiness beyond local experimentation and repository testing.

## 2. Architecture and data flow

The repository follows a straightforward local pipeline:

1. Raw content is collected from RVCE pages and PDFs and stored in `data/raw/`.
2. The raw records are processed and validated into `data/processed/campus_documents.jsonl`.
3. `src/retrieval.py` chunks the processed content and indexes it with TF-IDF.
4. `src/answering.py` uses ranked retrieval results to compose a concise evidence-grounded answer.
5. `src/api.py` exposes the retrieval-and-answer layer via HTTP.
6. `src/evaluation.py` loads benchmark queries and records metrics in local MLflow storage.
7. `reports/` stores generated evaluations, smoke tests, and degradation reports.

The result is a local document retrieval pipeline rather than a generative answer engine or a production-ready API stack.

## 3. Implementation and MLOps lifecycle

### Data pipeline

The project includes a dataset build and validation flow rooted in the repository’s existing scripts and reports. The actual report in [reports/data_collection_report.json](reports/data_collection_report.json) shows:
- 60 total records collected
- 45 HTML and 15 PDF source documents
- 0 validation issues
- 45 successful extractions and 15 empty extractions

### Retrieval and answer layer

The retrieval stack is implemented in `src/retrieval.py` and uses TF-IDF with cosine similarity. The answer layer in `src/answering.py` intentionally avoids fabricated reasoning: it only uses retrieved evidence and abstains when the evidence is insufficient.

### Evaluation and tracking

The evaluation workflow in `src/evaluation.py` loads the benchmark CSV, runs the retrieval engine, and logs metrics to a local MLflow file store under `mlruns/`.

### Dataset versioning

The repository now uses DVC to track canonical data files. The tracked files are:
- `data/raw/campus_documents.jsonl`
- `data/raw/collection_manifest.json`
- `data/processed/campus_documents.jsonl`

No remote DVC storage is configured in this workspace. `dvc status` reports `Data and pipelines are up to date.`

### API and deployment scaffolding

The repository includes a minimal FastAPI app in `src/api.py` and a Dockerfile for a containerized runtime. These are present in the repo, but the environment-specific runtime status must be described accurately:
- FastAPI app behavior is verified by tests.
- Docker build status is not verified in this environment because Docker was unavailable locally.
- GitHub Actions workflow files are present, but they were not executed here.

## 4. Actual test results and evaluation findings

### Test results

The repository test suite was run locally and passed:

```bash
pytest -q
```

Actual result:
- 29 passed
- 1 warning
- warning: a pending deprecation notice from `starlette` / `python_multipart`

### Benchmark evaluation findings

The local benchmark run produced these actual metrics:
- queries_evaluated: 40
- queries_with_results: 40
- queries_without_results: 0
- relevance_rate: 1.0
- average_best_score: 0.1474

These numbers reflect the available benchmark and ingestion state, not a claim of universal answer correctness.

### Limitations revealed by evaluation

The benchmark CSV contains no ground-truth relevance labels, so the repository uses retrieval-hit coverage rather than gold relevance or answer accuracy. This is explicitly documented in `src/evaluation.py` and the generated summary outputs.

## 5. DVC, MLflow, FastAPI, Docker, and CI status

### DVC

Status: implemented and verified locally.
- DVC version: `3.67.1`
- `dvc status`: `Data and pipelines are up to date.`
- No remote storage configured

### MLflow

Status: local file-backed evaluation is implemented and executed.
- experiments are stored under `mlruns/`
- evaluation summaries and config files are produced under `reports/`

### FastAPI

Status: implemented and verified by tests.
- `src/api.py` defines `/health` and `/ask`
- Test clients confirm health and abstention behavior

### Docker

Status: configured but not verified in this environment.
- `Dockerfile` exists and runs an app container based on Python 3.11 (from the repository file)
- no successful local Docker build was performed here because Docker was unavailable in the environment

### GitHub Actions CI

Status: configuration exists but was not executed here.
- `.github/workflows/ci.yml` defines a Python 3.11 `pytest -q` job
- the workflow file is present in the repo but this environment did not run GitHub-hosted CI

## 6. Drift/degradation experiment results

The project includes a minimal synthetic degradation monitoring experiment in `src/evaluation.py` and [reports/retrieval_degradation_report.md](reports/retrieval_degradation_report.md).

Actual results:
- baseline relevance_rate: 1.0
- degraded relevance_rate: 0.95
- average_best_score: 0.1474 -> 0.1436
- relevance_rate_drop: 0.05
- average_best_score_drop: 0.0038
- warning threshold: 0.10
- retraining threshold: 0.25
- decision: `no_action`

This experiment is intentionally synthetic and should not be interpreted as real-world drift or production monitoring evidence.

## 7. Limitations and future work

The project is a strong local baseline, but it has clear limitations:

- no gold relevance labels for the benchmark
- retrieval is heuristic and limited to the current corpus
- information gaps remain in key campus topics such as hostel, faculty, and exact company lists
- no remote storage or external monitoring stack is configured
- no model registration, no auto-retraining, and no production deployment stack were implemented in this repository state

Future work that would be technically valid next steps would include:
- targeted data acquisition for missing campus pages
- richer benchmark labels and manual relevance annotation
- more robust retrieval strategies and reranking
- production-safe deployment and monitoring if the project is approved for that scope

## 8. Exact setup, run, test, and demonstration commands

### Setup

```bash
cd AI-Powered-Campus-Information-Assistant
python -m pip install -r requirements.txt
```

### Run evaluation

```bash
python -m src.evaluation --benchmark data/queries/campus_queries.csv --dataset data/processed/campus_documents.jsonl --top-k 3 --score-threshold 0.05
```

### Run degradation experiment

```bash
python -m src.evaluation --run-degradation --benchmark data/queries/campus_queries.csv --dataset data/processed/campus_documents.jsonl --top-k 3 --score-threshold 0.05 --warning-drop 0.10 --retraining-drop 0.25 --report-path reports/retrieval_degradation_report.md
```

### Start the API

```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

### Health check

```bash
curl http://127.0.0.1:8000/health
```

### Ask a supported question

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question": "What is the procedure for COMEDK rank card?"}'
```

### Ask an unsupported question

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question": "What is the capital of France?"}'
```

### Run tests

```bash
pytest -q
```

## 9. Conclusion

This repository demonstrates a credible local campus knowledge retrieval system with benchmark evaluation, local monitoring, DVC dataset versioning, and a minimal API service. The work is valuable as a reproducible baseline and research artifact, but it is not a claim of production readiness or real-world operational monitoring.
