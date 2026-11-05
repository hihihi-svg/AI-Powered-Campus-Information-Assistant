"""Tests for standalone supervised intent classification pipeline."""

import json
from pathlib import Path
import pytest
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

from src.intent_classification import (
    DEFAULT_DATASET_PATH,
    build_dummy_baseline,
    build_logistic_regression_pipeline,
    evaluate_classifier,
    load_query_dataset,
    log_intent_experiment_to_mlflow,
)


def test_load_query_dataset_split_and_no_leakage():
    """Verify that train and test partitions are stratified, non-empty, and strictly disjoint."""
    train_rows, test_rows, classes = load_query_dataset(DEFAULT_DATASET_PATH)

    assert len(train_rows) == 32
    assert len(test_rows) == 8
    assert len(classes) == 8

    train_ids = {r["query_id"] for r in train_rows}
    test_ids = {r["query_id"] for r in test_rows}
    assert train_ids.isdisjoint(test_ids), "Query IDs must not overlap between train and test splits!"

    train_texts = {r["query_text"].strip().lower() for r in train_rows}
    test_texts = {r["query_text"].strip().lower() for r in test_rows}
    assert train_texts.isdisjoint(test_texts), "Exact query text must not leak between train and test splits!"

    # Verify stratification (4 train, 1 test per intent)
    for intent in classes:
        train_count = sum(1 for r in train_rows if r["intent"] == intent)
        test_count = sum(1 for r in test_rows if r["intent"] == intent)
        assert train_count == 4, f"Intent {intent} expected 4 train samples, got {train_count}"
        assert test_count == 1, f"Intent {intent} expected 1 test sample, got {test_count}"


def test_pipeline_fitting_and_predictions():
    """Ensure TF-IDF + LogisticRegression pipeline fits on train split and predicts valid class labels."""
    train_rows, test_rows, classes = load_query_dataset(DEFAULT_DATASET_PATH)

    X_train = [r["query_text"] for r in train_rows]
    y_train = [r["intent"] for r in train_rows]
    X_test = [r["query_text"] for r in test_rows]

    pipeline = build_logistic_regression_pipeline()
    assert isinstance(pipeline, Pipeline)

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    assert len(preds) == len(test_rows)
    for pred in preds:
        assert pred in classes, f"Predicted class {pred} not in known classes {classes}"


def test_dummy_baseline_fitting():
    """DummyClassifier should predict most frequent class across all test samples."""
    train_rows, test_rows, classes = load_query_dataset(DEFAULT_DATASET_PATH)

    X_train = [r["query_text"] for r in train_rows]
    y_train = [r["intent"] for r in train_rows]
    X_test = [r["query_text"] for r in test_rows]

    dummy = build_dummy_baseline(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    preds = dummy.predict(X_test)

    assert len(preds) == len(test_rows)
    # Strategy most_frequent outputs single repeated label
    assert len(set(preds)) == 1
    assert preds[0] in classes


def test_evaluate_classifier_metrics_schema():
    """Evaluation output must produce standard metrics, confusion matrix, and preliminary limitation warning."""
    train_rows, test_rows, classes = load_query_dataset(DEFAULT_DATASET_PATH)

    X_train = [r["query_text"] for r in train_rows]
    y_train = [r["intent"] for r in train_rows]

    pipeline = build_logistic_regression_pipeline()
    pipeline.fit(X_train, y_train)

    results = evaluate_classifier(pipeline, test_rows, classes)

    expected_keys = {
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
        "test_sample_count",
        "confusion_matrix",
        "class_labels",
        "per_class_metrics",
        "predictions",
        "evaluation_warning",
    }
    assert expected_keys.issubset(results.keys())

    assert 0.0 <= results["accuracy"] <= 1.0
    assert 0.0 <= results["macro_f1"] <= 1.0
    assert results["test_sample_count"] == 8

    # Confusion matrix is 8x8 with sum = 8
    conf_mat = results["confusion_matrix"]
    assert len(conf_mat) == 8
    assert all(len(row) == 8 for row in conf_mat)
    total_in_mat = sum(sum(row) for row in conf_mat)
    assert total_in_mat == 8

    # Per-class metrics exist for every class
    for c in classes:
        assert c in results["per_class_metrics"]
        assert "precision" in results["per_class_metrics"][c]
        assert "recall" in results["per_class_metrics"][c]
        assert "f1_score" in results["per_class_metrics"][c]
        assert results["per_class_metrics"][c]["support"] == 1


def test_model_artifact_logging_and_reload(tmp_path):
    """Fitted LogisticRegression model artifact must log to MLflow and reload cleanly."""
    train_rows, test_rows, classes = load_query_dataset(DEFAULT_DATASET_PATH)

    X_train = [r["query_text"] for r in train_rows]
    y_train = [r["intent"] for r in train_rows]

    pipeline = build_logistic_regression_pipeline()
    pipeline.fit(X_train, y_train)
    eval_results = evaluate_classifier(pipeline, test_rows, classes)

    temp_mlruns = tmp_path / "mlruns"
    temp_mlruns.mkdir()
    tracking_uri = temp_mlruns.as_uri()

    run_info = log_intent_experiment_to_mlflow(
        model=pipeline,
        model_name="logistic_regression_tfidf",
        eval_results=eval_results,
        dataset_path=DEFAULT_DATASET_PATH,
        train_count=len(train_rows),
        test_count=len(test_rows),
        experiment_name="test-intent-experiment",
        run_name="test-model-reload-run",
        tracking_uri=tracking_uri,
        log_model_artifact=True,
    )

    assert run_info["model_uri"] is not None
    assert run_info["model_reload_verified"] is True
