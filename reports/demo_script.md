# Demo script: RVCE Campus Information Assistant

This is a concise 3–5 minute demonstration sequence based on the repository’s verified implementation.

## 1. Start the API

```bash
cd AI-Powered-Campus-Information-Assistant
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Expected behavior:
- the app starts with the FastAPI app defined in `src/api.py`
- the `/health` endpoint should respond with a JSON status payload
- the `/ask` endpoint accepts a question and returns an answer plus evidence and sources

## 2. Health check

```bash
curl http://127.0.0.1:8000/health
```

Expected output:

```json
{"status":"ok","service":"rvce-campus-information-assistant"}
```

## 3. Supported question

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question": "What is the procedure for COMEDK rank card?"}'
```

Expected behavior:
- HTTP 200
- `abstention_status` is `false`
- answer text contains `COMEDK` or similar documented terminology from the campus corpus
- `sources` contains at least one source entry

This is grounded in the repository test in `tests/test_api.py`.

## 4. Unsupported question

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question": "What is the capital of France?"}'
```

Expected behavior:
- HTTP 200
- answer equals the repository’s abstention message
- `abstention_status` is `true`
- `sources` and `evidence` are empty lists

The exact abstention message is:

> I could not find sufficient evidence in the available campus documents to answer this question.

## 5. Run local evaluation

```bash
python -m src.evaluation --benchmark data/queries/campus_queries.csv --dataset data/processed/campus_documents.jsonl --top-k 3 --score-threshold 0.05
```

Expected behavior:
- runs the benchmark defined in `data/queries/campus_queries.csv`
- logs metrics to MLflow under the local `mlruns/` directory
- prints metrics including `queries_evaluated`, `queries_with_results`, and `relevance_rate`

## 6. Review the generated reports

Relevant files:
- [reports/retrieval_degradation_report.md](reports/retrieval_degradation_report.md)
- [reports/data_collection_report.json](reports/data_collection_report.json)
- [reports/mlflow_retrieval_evaluation_summary.json](reports/mlflow_retrieval_evaluation_summary.json)
- [reports/mlflow_retrieval_evaluation_config.json](reports/mlflow_retrieval_evaluation_config.json)

## 7. Expected findings and boundaries

- The assistant is retrieval-based and evidence-first.
- It does not claim real production drift or automatic retraining.
- It does not guarantee answer correctness beyond the indexed corpus and retrieval threshold.
- The benchmark has no ground-truth labels; it is a retrieval-hit proxy, not an accuracy metric.
- The synthetic degradation experiment is a local monitoring exercise only.

## 8. Suggested closing notes

The demo can end by saying:

> This repository demonstrates a minimal retrieval-based campus assistant grounded in a local RVCE corpus. It is a reproducible baseline with benchmark evaluation, DVC tracking, and local MLflow logging, but it remains a research and local-development artifact rather than a production-grade system.
