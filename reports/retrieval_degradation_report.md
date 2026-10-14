# Retrieval degradation experiment report

Date: 2026-10-13T17:41:23Z
Baseline dataset: data\processed\campus_documents.jsonl
Degraded dataset: reports\simulated_degraded_corpus.jsonl
Intervention: Synthetic corpus reduction using an alternate reduced copy of the processed dataset. This is a controlled simulation only, not production drift evidence.

## Baseline metrics
- queries_evaluated: 40
- queries_with_results: 40
- queries_without_results: 0
- relevance_rate: 1.0
- average_best_score: 0.1474

## Simulated degraded metrics
- queries_evaluated: 40
- queries_with_results: 38
- queries_without_results: 2
- relevance_rate: 0.95
- average_best_score: 0.1436

## Metric differences
- relevance_rate_drop: 0.05
- average_best_score_drop: 0.0038

## Threshold rule
- warning threshold: 0.1
- retraining threshold: 0.25
- warning triggered: False
- automatic retraining decision: False

## Decision
no_action

## Limitations
- The benchmark contains no gold relevance labels; this is a retrieval-hit proxy rather than answer correctness.
- The degraded dataset is synthetic and created only for a local monitoring experiment.
- This experiment is not proof of real-world production drift.
