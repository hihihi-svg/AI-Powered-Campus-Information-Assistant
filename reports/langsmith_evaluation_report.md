# LangSmith RAG Observability & Evaluation Report

- **Timestamp (UTC)**: 2026-11-11T04:12:16.473587+00:00
- **Strategy**: `tfidf` (rerank: False)
- **LangSmith Project**: `rvce-campus-assistant`
- **Tracing Enabled**: `False`
- **Total Evaluated Queries**: 2

## Performance Metrics

| Dimension | Metric | Score | Target |
|-----------|--------|-------|--------|
| Retrieval Relevance | Hit Rate | **100.0%** | >= 80% |
| Context Relevance | Context Quality | **100.0%** | >= 80% |
| Answer Relevance | Topic / Keyword Match | **100.0%** | >= 75% |
| Citation Correctness | Grounded Sources | **100.0%** | >= 85% |
| Abstention Quality | Out-of-Domain Refusal | **100.0%** | 100% |
| Overall Composite | Composite Score | **1.0000** | >= 0.80 |
| Latency | Mean Latency | **0.3323s** | <= 1.0s |

## Detailed Query Results

| ID | Type | Hit | Context | Answer | Citations | Latency | Snippet |
|----|------|-----|---------|--------|-----------|---------|---------|
| RAG_F01 | factual | PASS | PASS | PASS | PASS | 0.5499s | The available RVCE campus records do not contain o... |
| RAG_F02 | factual | PASS | PASS | PASS | PASS | 0.1147s | Undergraduate (1st Year B.E.) COMED-K quota fees a... |
