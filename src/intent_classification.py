"""Standalone supervised intent classification pipeline for the RVCE Campus Information Assistant.

Trains and evaluates:
1. Baseline: DummyClassifier (strategy="most_frequent")
2. Primary Model: scikit-learn Pipeline (TfidfVectorizer + LogisticRegression)

Uses the pre-assigned stratified 32-train / 8-test split from campus_queries.csv.
Logs metrics, parameters, and model artifacts to MLflow experiment 'rvce-campus-intent-classification'.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

logger = logging.getLogger("campus_pipeline.intent_classification")

DEFAULT_DATASET_PATH = Path("data/queries/campus_queries.csv")
DEFAULT_EXPERIMENT_NAME = "rvce-campus-intent-classification"
DEFAULT_REPORT_PATH = Path("reports/intent_classification_evaluation.json")

EVALUATION_LIMITATION_NOTE = (
    "Preliminary Evaluation Warning: Evaluation was conducted on a test set of exactly 8 samples "
    "(1 per intent class). These results serve as a functional baseline smoke-test and pipeline "
    "verification; they are not statistically significant for claiming production-level generalization."
)


def _local_tracking_uri(tracking_uri: Optional[str] = None) -> str:
    """Resolve local file-backed MLflow tracking URI."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    if tracking_uri:
        return tracking_uri
    repo_root = Path.cwd().resolve()
    mlruns_dir = repo_root / "mlruns"
    return mlruns_dir.as_uri()


def _get_git_commit() -> Optional[str]:
    """Retrieve current Git commit hash if in a git repository."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return commit
    except Exception:
        return None


def _get_file_checksum(file_path: Path) -> str:
    """Compute 12-char SHA-256 checksum for tracking dataset versions."""
    if not file_path.exists():
        return "none"
    try:
        hasher = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:12]
    except Exception:
        return "error"


def load_query_dataset(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[str]]:
    """Load query CSV and partition strictly by the predefined split_group.

    Returns:
        (train_rows, test_rows, sorted_unique_classes)
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    train_rows: List[Dict[str, str]] = []
    test_rows: List[Dict[str, str]] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split_group = (row.get("split_group") or "").strip().lower()
            record = {
                "query_id": (row.get("query_id") or "").strip(),
                "query_text": (row.get("query_text") or "").strip(),
                "intent": (row.get("intent") or "").strip(),
                "split_group": split_group,
            }
            if split_group == "train":
                train_rows.append(record)
            elif split_group == "test":
                test_rows.append(record)
            else:
                logger.warning("Skipping row %s with unassigned split_group: %s", record["query_id"], split_group)

    if not train_rows:
        raise ValueError(f"No training rows found in dataset: {path}")
    if not test_rows:
        raise ValueError(f"No test rows found in dataset: {path}")

    all_intents = sorted(list({r["intent"] for r in train_rows + test_rows if r["intent"]}))
    return train_rows, test_rows, all_intents


def build_logistic_regression_pipeline(
    c_param: float = 1.0,
    max_iter: int = 200,
    random_state: int = 42,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Pipeline:
    """Construct an end-to-end TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=ngram_range,
                    stop_words="english",
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b[a-zA-Z0-9_\-\.]{2,}\b",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c_param,
                    max_iter=max_iter,
                    random_state=random_state,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def build_dummy_baseline(strategy: str = "most_frequent") -> DummyClassifier:
    """Construct a heuristic DummyClassifier baseline."""
    return DummyClassifier(strategy=strategy)


def evaluate_classifier(
    model: Any,
    test_rows: List[Dict[str, str]],
    class_labels: List[str],
) -> Dict[str, Any]:
    """Compute standard classification evaluation metrics on the test partition."""
    X_test = [r["query_text"] for r in test_rows]
    y_test = [r["intent"] for r in test_rows]

    predictions = model.predict(X_test)

    acc = round(float(accuracy_score(y_test, predictions)), 4)
    macro_f1 = round(float(f1_score(y_test, predictions, average="macro", zero_division=0)), 4)
    weighted_f1 = round(float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 4)
    macro_precision = round(float(precision_score(y_test, predictions, average="macro", zero_division=0)), 4)
    macro_recall = round(float(recall_score(y_test, predictions, average="macro", zero_division=0)), 4)

    conf_mat = confusion_matrix(y_test, predictions, labels=class_labels).tolist()

    report_dict = classification_report(
        y_test, predictions, labels=class_labels, zero_division=0, output_dict=True
    )

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    for label in class_labels:
        entry = report_dict.get(label, {})
        per_class_metrics[label] = {
            "precision": round(float(entry.get("precision", 0.0)), 4),
            "recall": round(float(entry.get("recall", 0.0)), 4),
            "f1_score": round(float(entry.get("f1-score", 0.0)), 4),
            "support": int(entry.get("support", 0)),
        }

    per_query_predictions = [
        {
            "query_id": r["query_id"],
            "query_text": r["query_text"],
            "true_intent": r["intent"],
            "predicted_intent": str(pred),
            "correct": bool(r["intent"] == pred),
        }
        for r, pred in zip(test_rows, predictions)
    ]

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "test_sample_count": len(test_rows),
        "confusion_matrix": conf_mat,
        "class_labels": class_labels,
        "per_class_metrics": per_class_metrics,
        "predictions": per_query_predictions,
        "evaluation_warning": EVALUATION_LIMITATION_NOTE,
    }


def log_intent_experiment_to_mlflow(
    model: Any,
    model_name: str,
    eval_results: Dict[str, Any],
    dataset_path: Path,
    train_count: int,
    test_count: int,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    log_model_artifact: bool = False,
) -> Dict[str, Any]:
    """Log model parameters, dataset checksum, and evaluation metrics to MLflow."""
    mlflow.set_tracking_uri(_local_tracking_uri(tracking_uri))
    mlflow.set_experiment(experiment_name)

    run_label = run_name or f"{model_name}-{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}"

    with mlflow.start_run(run_name=run_label) as run:
        # 1. Parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("dataset_path", str(dataset_path))
        mlflow.log_param("dataset_checksum_sha256", _get_file_checksum(dataset_path))
        mlflow.log_param("train_samples", train_count)
        mlflow.log_param("test_samples", test_count)
        mlflow.log_param("class_count", len(eval_results["class_labels"]))
        mlflow.log_param("python_version", sys.version.split()[0])
        import sklearn
        mlflow.log_param("sklearn_version", sklearn.__version__)
        git_commit = _get_git_commit()
        if git_commit:
            mlflow.log_param("git_commit", git_commit)

        # Log hyperparameters
        if hasattr(model, "get_params"):
            params = model.get_params()
            for p_key in [
                "classifier__C",
                "classifier__max_iter",
                "classifier__random_state",
                "classifier__class_weight",
                "tfidf__ngram_range",
                "tfidf__sublinear_tf",
                "strategy",
            ]:
                if p_key in params:
                    mlflow.log_param(p_key.replace("__", "_"), str(params[p_key]))

        # 2. Scalar Metrics
        mlflow.log_metric("accuracy", eval_results["accuracy"])
        mlflow.log_metric("macro_f1", eval_results["macro_f1"])
        mlflow.log_metric("weighted_f1", eval_results["weighted_f1"])
        mlflow.log_metric("macro_precision", eval_results["macro_precision"])
        mlflow.log_metric("macro_recall", eval_results["macro_recall"])

        # Per-class metrics
        for c_label, c_m in eval_results["per_class_metrics"].items():
            mlflow.log_metric(f"f1_{c_label}", c_m["f1_score"])

        # 3. Tags
        mlflow.set_tag("evaluation_stage", "baseline_verification")
        mlflow.set_tag("dataset_warning", "8-sample test split preliminary evaluation")

        # 4. Model Artifact (for primary pipeline)
        model_uri = None
        reload_verified = False
        if log_model_artifact:
            model_info = mlflow.sklearn.log_model(model, name="model")
            model_uri = model_info.model_uri

            # Verify reload
            reloaded_pipeline = mlflow.sklearn.load_model(model_uri)
            reloaded_preds = list(reloaded_pipeline.predict([p["query_text"] for p in eval_results["predictions"]]))
            original_preds = [p["predicted_intent"] for p in eval_results["predictions"]]
            assert reloaded_preds == original_preds, "Reloaded model predictions mismatch original predictions!"
            reload_verified = True
            mlflow.set_tag("model_reload_verified", "true")
            mlflow.set_tag("model_uri", model_uri)

    return {
        "run_id": run.info.run_id,
        "run_name": run_label,
        "experiment_name": experiment_name,
        "artifact_uri": run.info.artifact_uri,
        "model_uri": model_uri,
        "model_reload_verified": reload_verified,
    }


def train_and_evaluate_all(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_uri: Optional[str] = None,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> Dict[str, Any]:
    """Execute training, evaluation, artifact logging, and reporting for both models."""
    data_file = Path(dataset_path)
    train_rows, test_rows, classes = load_query_dataset(data_file)

    X_train = [r["query_text"] for r in train_rows]
    y_train = [r["intent"] for r in train_rows]

    # --- 1. Baseline: DummyClassifier ---
    logger.info("Training DummyClassifier baseline on %d samples...", len(train_rows))
    dummy_model = build_dummy_baseline(strategy="most_frequent")
    dummy_model.fit(X_train, y_train)
    dummy_results = evaluate_classifier(dummy_model, test_rows, classes)

    dummy_run_info = log_intent_experiment_to_mlflow(
        model=dummy_model,
        model_name="dummy_most_frequent",
        eval_results=dummy_results,
        dataset_path=data_file,
        train_count=len(train_rows),
        test_count=len(test_rows),
        experiment_name=experiment_name,
        run_name="dummy-most-frequent-baseline",
        tracking_uri=tracking_uri,
        log_model_artifact=False,
    )

    # --- 2. Primary Model: Logistic Regression Pipeline ---
    logger.info("Training LogisticRegression pipeline on %d samples...", len(train_rows))
    lr_pipeline = build_logistic_regression_pipeline()
    lr_pipeline.fit(X_train, y_train)
    lr_results = evaluate_classifier(lr_pipeline, test_rows, classes)

    lr_run_info = log_intent_experiment_to_mlflow(
        model=lr_pipeline,
        model_name="logistic_regression_tfidf",
        eval_results=lr_results,
        dataset_path=data_file,
        train_count=len(train_rows),
        test_count=len(test_rows),
        experiment_name=experiment_name,
        run_name="tfidf-logistic-regression",
        tracking_uri=tracking_uri,
        log_model_artifact=True,
    )

    overall_report = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": {
            "path": str(data_file),
            "checksum_sha256": _get_file_checksum(data_file),
            "train_samples": len(train_rows),
            "test_samples": len(test_rows),
            "class_labels": classes,
        },
        "evaluation_limitation": EVALUATION_LIMITATION_NOTE,
        "models": {
            "dummy_baseline": {
                "metrics": {
                    "accuracy": dummy_results["accuracy"],
                    "macro_f1": dummy_results["macro_f1"],
                    "weighted_f1": dummy_results["weighted_f1"],
                    "macro_precision": dummy_results["macro_precision"],
                    "macro_recall": dummy_results["macro_recall"],
                },
                "per_class": dummy_results["per_class_metrics"],
                "confusion_matrix": dummy_results["confusion_matrix"],
                "mlflow_run": dummy_run_info,
            },
            "logistic_regression": {
                "metrics": {
                    "accuracy": lr_results["accuracy"],
                    "macro_f1": lr_results["macro_f1"],
                    "weighted_f1": lr_results["weighted_f1"],
                    "macro_precision": lr_results["macro_precision"],
                    "macro_recall": lr_results["macro_recall"],
                },
                "per_class": lr_results["per_class_metrics"],
                "confusion_matrix": lr_results["confusion_matrix"],
                "predictions": lr_results["predictions"],
                "mlflow_run": lr_run_info,
            },
        },
    }

    # Persist machine-readable report
    rep_path = Path(report_path)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(overall_report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved evaluation report to %s", rep_path)

    return overall_report


def main() -> None:
    """CLI entrypoint for standalone intent classification experiment."""
    parser = argparse.ArgumentParser(
        description="Run supervised intent classification training and evaluation on campus_queries.csv"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help="Path to campus_queries.csv",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="Custom MLflow tracking URI",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=str(DEFAULT_REPORT_PATH),
        help="Output JSON path for machine-readable evaluation report",
    )
    args = parser.parse_args()

    print("======================================================================")
    print(" RVCE Campus Assistant: Supervised Intent Classification Pipeline")
    print("======================================================================")
    results = train_and_evaluate_all(
        dataset_path=args.dataset,
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        report_path=args.report_path,
    )

    models = results["models"]
    dummy = models["dummy_baseline"]["metrics"]
    lr = models["logistic_regression"]["metrics"]

    print("\n------------------------- MODEL COMPARISON -------------------------")
    print(f"{'Metric':<20} | {'Dummy Baseline':<16} | {'Logistic Regression':<18}")
    print("-" * 60)
    print(f"{'Accuracy':<20} | {dummy['accuracy']:<16.4f} | {lr['accuracy']:<18.4f}")
    print(f"{'Macro F1':<20} | {dummy['macro_f1']:<16.4f} | {lr['macro_f1']:<18.4f}")
    print(f"{'Weighted F1':<20} | {dummy['weighted_f1']:<16.4f} | {lr['weighted_f1']:<18.4f}")
    print(f"{'Macro Precision':<20} | {dummy['macro_precision']:<16.4f} | {lr['macro_precision']:<18.4f}")
    print(f"{'Macro Recall':<20} | {dummy['macro_recall']:<16.4f} | {lr['macro_recall']:<18.4f}")
    print("=" * 60)
    print("\nMLflow Experiment:", args.experiment_name)
    print("Dummy Run ID:", models["dummy_baseline"]["mlflow_run"]["run_id"])
    print("Logistic Regression Run ID:", models["logistic_regression"]["mlflow_run"]["run_id"])
    print("Model Artifact URI:", models["logistic_regression"]["mlflow_run"]["model_uri"])
    print("Model Reload Verified:", models["logistic_regression"]["mlflow_run"]["model_reload_verified"])
    print("Report saved at:", args.report_path)
    print("\nNOTE:", results["evaluation_limitation"])


if __name__ == "__main__":
    main()
