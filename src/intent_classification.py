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

from src.entity_extraction import extract_campus_entities

logger = logging.getLogger("campus_pipeline.intent_classification")

DEFAULT_DATASET_PATH = Path("data/queries/campus_queries.csv")
DEFAULT_DATASET_V2_PATH = Path("data/queries/campus_queries_v2.csv")
DEFAULT_EXPERIMENT_NAME = "campus-intent-classification"
DEFAULT_REPORT_PATH = Path("reports/intent_classification_evaluation.json")
DEFAULT_REPORT_V2_PATH = Path("reports/intent_classification_v2_evaluation.json")

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
        is_pipeline = hasattr(model, "named_steps") or "logistic" in model_name.lower()
        algorithm_name = "LogisticRegression" if is_pipeline else model.__class__.__name__

        mlflow.log_param("model_type", model_name)
        mlflow.log_param("model_algorithm", algorithm_name)
        mlflow.log_param("dataset_path", str(dataset_path))
        mlflow.log_param("dataset_checksum_sha256", _get_file_checksum(dataset_path))
        mlflow.log_param("train_samples", train_count)
        mlflow.log_param("test_samples", test_count)
        total_samples = train_count + test_count
        mlflow.log_param("train_test_split_ratio", round(test_count / total_samples, 4) if total_samples else 0.0)
        mlflow.log_param("class_count", len(eval_results["class_labels"]))
        mlflow.log_param("dvc_tracked", "true")
        mlflow.log_param("python_version", sys.version.split()[0])
        import sklearn
        mlflow.log_param("sklearn_version", sklearn.__version__)
        git_commit = _get_git_commit()
        if git_commit:
            mlflow.log_param("git_commit", git_commit)

        # Log hyperparameters
        training_config: Dict[str, Any] = {
            "model_name": model_name,
            "model_algorithm": algorithm_name,
            "dataset_path": str(dataset_path),
            "dataset_checksum_sha256": _get_file_checksum(dataset_path),
            "train_samples": train_count,
            "test_samples": test_count,
            "class_labels": eval_results["class_labels"],
            "random_seed": 42,
        }

        if hasattr(model, "get_params"):
            params = model.get_params()
            training_config["hyperparameters"] = {}
            for p_key in [
                "classifier__C",
                "classifier__max_iter",
                "classifier__random_state",
                "classifier__class_weight",
                "classifier__solver",
                "tfidf__ngram_range",
                "tfidf__sublinear_tf",
                "tfidf__norm",
                "tfidf__max_features",
                "strategy",
            ]:
                if p_key in params:
                    clean_key = p_key.replace("__", "_")
                    val = params[p_key]
                    mlflow.log_param(clean_key, str(val))
                    training_config["hyperparameters"][clean_key] = val

            if "classifier__random_state" in params:
                mlflow.log_param("random_seed", int(params["classifier__random_state"] or 42))
            elif "random_state" in params:
                mlflow.log_param("random_seed", int(params["random_state"] or 42))
            else:
                mlflow.log_param("random_seed", 42)

        # 2. Scalar Metrics
        acc = float(eval_results["accuracy"])
        macro_prec = float(eval_results["macro_precision"])
        macro_rec = float(eval_results["macro_recall"])
        macro_f1_val = float(eval_results["macro_f1"])
        weighted_f1_val = float(eval_results["weighted_f1"])

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("macro_precision", macro_prec)
        mlflow.log_metric("macro_recall", macro_rec)
        mlflow.log_metric("macro_f1", macro_f1_val)
        mlflow.log_metric("weighted_f1", weighted_f1_val)

        correct_count = sum(1 for p in eval_results["predictions"] if p.get("correct"))
        incorrect_count = len(eval_results["predictions"]) - correct_count
        mlflow.log_metric("correct_predictions", correct_count)
        mlflow.log_metric("incorrect_predictions", incorrect_count)

        # Per-class metrics
        for c_label, c_m in eval_results["per_class_metrics"].items():
            mlflow.log_metric(f"precision_{c_label}", float(c_m["precision"]))
            mlflow.log_metric(f"recall_{c_label}", float(c_m["recall"]))
            mlflow.log_metric(f"f1_{c_label}", float(c_m["f1_score"]))

        # 3. Tags
        mlflow.set_tag("evaluation_stage", "v2_production_intent" if "v2" in str(dataset_path) else "baseline_verification")
        mlflow.set_tag("model_algorithm", algorithm_name)

        # 4. Save and Log Rich Artifacts
        artifacts_dir = Path("reports") / "mlflow_temp" / run.info.run_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # a. Classification report JSON & TXT
        class_rep_json_path = artifacts_dir / "classification_report.json"
        class_rep_json_path.write_text(
            json.dumps(eval_results["per_class_metrics"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(class_rep_json_path), artifact_path="evaluation")

        class_rep_txt_path = artifacts_dir / "classification_report.txt"
        lines = [
            f"{'Class Label':<28} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}",
            "-" * 75,
        ]
        for c_lbl, c_m in eval_results["per_class_metrics"].items():
            lines.append(
                f"{c_lbl:<28} | {c_m['precision']:<10.4f} | {c_m['recall']:<10.4f} | {c_m['f1_score']:<10.4f} | {c_m['support']:<8}"
            )
        lines.append("-" * 75)
        lines.append(
            f"{'Macro Average':<28} | {macro_prec:<10.4f} | {macro_rec:<10.4f} | {macro_f1_val:<10.4f} | {test_count:<8}"
        )
        lines.append(
            f"{'Weighted Average':<28} | {'-':<10} | {'-':<10} | {weighted_f1_val:<10.4f} | {test_count:<8}"
        )
        lines.append(f"{'Overall Accuracy':<28} | {acc:<10.4f} | {'-':<10} | {'-':<10} | {test_count:<8}")
        class_rep_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(class_rep_txt_path), artifact_path="evaluation")

        # b. Confusion Matrix JSON & TXT
        conf_mat_json_path = artifacts_dir / "confusion_matrix.json"
        conf_mat_json_path.write_text(
            json.dumps({
                "labels": eval_results["class_labels"],
                "matrix": eval_results["confusion_matrix"],
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(conf_mat_json_path), artifact_path="evaluation")

        conf_mat_txt_path = artifacts_dir / "confusion_matrix.txt"
        cm_labels = eval_results["class_labels"]
        cm_matrix = eval_results["confusion_matrix"]
        header_labels = [lbl[:6] for lbl in cm_labels]
        cm_title = "True \\ Pred"
        cm_lines = [f"{cm_title:<24} | " + " | ".join(f"{hl:>6}" for hl in header_labels)]
        cm_lines.append("-" * (27 + 9 * len(cm_labels)))
        for i, row_label in enumerate(cm_labels):
            row_str = " | ".join(f"{cm_matrix[i][j]:>6}" for j in range(len(cm_labels)))
            cm_lines.append(f"{row_label:<24} | {row_str}")
        conf_mat_txt_path.write_text("\n".join(cm_lines) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(conf_mat_txt_path), artifact_path="evaluation")

        # c. Training configuration
        config_path = artifacts_dir / "training_config.json"
        config_path.write_text(json.dumps(training_config, indent=2, ensure_ascii=False), encoding="utf-8")
        mlflow.log_artifact(str(config_path), artifact_path="evaluation")

        # 5. Model Artifact (for primary pipeline)
        model_uri = None
        reload_verified = False
        if log_model_artifact:
            model_info = mlflow.sklearn.log_model(model, name="model", pip_requirements=["scikit-learn", "numpy"])
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


INTENT_DISPLAY_LABELS: Dict[str, str] = {
    # 9 target classes
    "admissions_cutoff": "Admissions Cutoffs & Ranks",
    "admission_fees": "Admission Fees & Fee Structure",
    "admissions_procedure": "Admissions Procedure & Quotas",
    "placements": "Placements & Careers",
    "academics_exams": "Academics & Examinations",
    "departments_faculty": "Departments & Faculty",
    "facilities_hostels": "Facilities & Hostels",
    "student_services": "Student Services & Activities",
    "general_college_info": "General College Information",
    # Legacy aliases
    "admission_fee": "Admission Fees",
    "hostel_fee": "Hostel Fees",
    "fee_penalty": "Late Fee Penalty",
    "general_fee": "Fee Structure",
    "admissions": "Admissions",
    "fees_finance": "Fees and Finance",
    "placements_careers": "Placements and Careers",
    "student_services_clubs": "Student Services and Clubs",
    "unknown": "Unknown",
}

_CACHED_INTENT_PIPELINE: Optional[Pipeline] = None
_CACHED_INTENT_PIPELINES: Dict[str, Pipeline] = {}


def get_intent_classifier(
    dataset_path: Optional[str | Path] = None,
    force_reload: bool = False,
) -> Optional[Pipeline]:
    """Load the trained Logistic Regression intent classifier (preferring v2 if available)."""
    global _CACHED_INTENT_PIPELINE, _CACHED_INTENT_PIPELINES

    repo_root = Path(__file__).resolve().parent.parent

    if dataset_path is not None:
        target_path = Path(dataset_path)
        if not target_path.is_absolute() and not target_path.exists() and (repo_root / target_path).exists():
            target_path = repo_root / target_path
    else:
        v2_candidate = repo_root / DEFAULT_DATASET_V2_PATH
        if v2_candidate.exists():
            target_path = v2_candidate
        else:
            target_path = repo_root / DEFAULT_DATASET_PATH

    cache_key = str(target_path.resolve()) if target_path.exists() else str(target_path)

    if not force_reload:
        if cache_key in _CACHED_INTENT_PIPELINES:
            _CACHED_INTENT_PIPELINE = _CACHED_INTENT_PIPELINES[cache_key]
            return _CACHED_INTENT_PIPELINE
        if dataset_path is None and _CACHED_INTENT_PIPELINE is not None:
            return _CACHED_INTENT_PIPELINE

    # Try loading persisted MLflow sklearn model pickle from mlruns if dataset_path is default v1
    if target_path == (repo_root / DEFAULT_DATASET_PATH):
        import pickle

        mlruns_dir = repo_root / "mlruns"
        if mlruns_dir.exists():
            for pkl_path in sorted(mlruns_dir.rglob("model.pkl"), key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    with pkl_path.open("rb") as f:
                        loaded = pickle.load(f)
                    if isinstance(loaded, Pipeline) and hasattr(loaded, "predict_proba"):
                        _CACHED_INTENT_PIPELINES[cache_key] = loaded
                        _CACHED_INTENT_PIPELINE = loaded
                        return loaded
                except Exception:
                    continue

    # Fit LogisticRegression pipeline on target dataset train split
    try:
        train_rows, _, _ = load_query_dataset(target_path)
        pipeline = build_logistic_regression_pipeline()
        pipeline.fit([r["query_text"] for r in train_rows], [r["intent"] for r in train_rows])
        _CACHED_INTENT_PIPELINES[cache_key] = pipeline
        _CACHED_INTENT_PIPELINE = pipeline
        return pipeline
    except Exception as exc:
        logger.warning("Could not initialize intent classifier for %s: %s", target_path, type(exc).__name__)
        return None


def train_and_evaluate_v2(
    dataset_path: str | Path = DEFAULT_DATASET_V2_PATH,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_uri: Optional[str] = None,
    report_path: str | Path = DEFAULT_REPORT_V2_PATH,
) -> Dict[str, Any]:
    """Train and evaluate the upgraded 9-class intent classifier on campus_queries_v2.csv."""
    data_file = Path(dataset_path)
    if not data_file.exists():
        repo_root = Path(__file__).resolve().parent.parent
        data_file = repo_root / dataset_path

    train_rows, test_rows, classes = load_query_dataset(data_file)
    X_train = [r["query_text"] for r in train_rows]
    y_train = [r["intent"] for r in train_rows]

    # Baseline: DummyClassifier
    dummy_model = build_dummy_baseline(strategy="most_frequent")
    dummy_model.fit(X_train, y_train)
    dummy_results = evaluate_classifier(dummy_model, test_rows, classes)

    dummy_run_info = log_intent_experiment_to_mlflow(
        model=dummy_model,
        model_name="dummy_most_frequent_v2",
        eval_results=dummy_results,
        dataset_path=data_file,
        train_count=len(train_rows),
        test_count=len(test_rows),
        experiment_name=experiment_name,
        run_name="dummy-baseline-v2",
        tracking_uri=tracking_uri,
        log_model_artifact=False,
    )

    # Primary: Logistic Regression
    lr_pipeline = build_logistic_regression_pipeline()
    lr_pipeline.fit(X_train, y_train)
    lr_results = evaluate_classifier(lr_pipeline, test_rows, classes)

    lr_run_info = log_intent_experiment_to_mlflow(
        model=lr_pipeline,
        model_name="logistic_regression_v2",
        eval_results=lr_results,
        dataset_path=data_file,
        train_count=len(train_rows),
        test_count=len(test_rows),
        experiment_name=experiment_name,
        run_name="tfidf-logistic-regression-v2",
        tracking_uri=tracking_uri,
        log_model_artifact=True,
    )

    report = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": {
            "path": str(data_file),
            "checksum_sha256": _get_file_checksum(data_file),
            "train_samples": len(train_rows),
            "test_samples": len(test_rows),
            "class_labels": classes,
        },
        "target_classes_count": len(classes),
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

    rep_path = Path(report_path)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved v2 evaluation report to %s", rep_path)
    return report


def extract_query_entities_and_intent(
    question: str,
    raw_intent: str = "unknown",
    raw_confidence: float = 0.0,
) -> Dict[str, Any]:
    """Extract domain entities and normalize semantically equivalent queries to structured intents.

    Identifies:
      - normalized_intent ('admissions_cutoff', 'admission_fee', 'admissions_procedure', 'placements', etc.)
      - main_entity ('admission_cutoff', 'admission_fee', 'hostel', 'placement', 'examination', 'facility', 'faculty')
      - entities: structured dict with exam, branch, admission_quota, category, academic_year, year, etc.
      - is_comparison (bool, True if comparing multiple quotas or branches)
      - retrieval_expansion (str of domain terms to enrich retrieval)
    """
    import re

    raw_q = (question or "").strip()
    lowered = raw_q.lower()

    # Extract independent domain entities (PART B)
    campus_ent = extract_campus_entities(raw_q)

    # Typo correction and synonym normalization for entity detection
    typo_map = {
        r"\bkket\b": "kcet",
        r"\bkkea\b": "kea",
        r"\bcomed-k\b": "comedk",
        r"\bmgmt\b": "management",
        r"\bhostle\b": "hostel",
        r"\bhostal\b": "hostel",
        r"\bintership\b": "internship",
        r"\badmisson\b": "admission",
        r"\badmision\b": "admission",
    }
    normalized_q = lowered
    for pat, rep in typo_map.items():
        normalized_q = re.sub(pat, rep, normalized_q)

    # 1. Quota entity detection
    has_kea_cet = bool(
        re.search(r"\b(k-?cet|kcet|kea|kkea|govt(?:\s+quota)?|government(?:\s+quota)?)\b", normalized_q)
        or re.search(r"\bfees?\s+(?:through|for|under|via)\s+cet\b", normalized_q)
        or re.search(r"\b(?<!pg)(?<!d)cet\b", normalized_q)
    )
    has_comedk = bool(re.search(r"\b(comed-?k|comed)\b", normalized_q))
    has_mgmt = bool(re.search(r"\b(management(?:\s+quota)?|mgmt(?:\s+quota)?)\b", normalized_q))
    has_pgcet = bool(re.search(r"\b(pgcet|m\.?tech|mtech|mca)\b", normalized_q))
    has_dcet = bool(re.search(r"\b(dcet|lateral\s+entry)\b", normalized_q))

    # Quota comparison detection
    quota_flags = [("kea_cet", has_kea_cet), ("comedk", has_comedk), ("management", has_mgmt), ("pgcet", has_pgcet)]
    detected_quotas = [q for q, present in quota_flags if present]
    is_comparison = bool(
        campus_ent.get("is_comparison")
        or re.search(r"\b(compare|comparison|difference\s+between|vs|versus)\b", normalized_q)
        or len(detected_quotas) >= 2
    )

    admission_quota = campus_ent.get("admission_quota")
    if is_comparison or len(detected_quotas) >= 2:
        admission_quota = "multiple"
    elif has_kea_cet:
        admission_quota = "kea_cet"
    elif has_comedk:
        admission_quota = "comedk"
    elif has_mgmt:
        admission_quota = "management"
    elif has_pgcet:
        admission_quota = "pgcet"
    elif has_dcet:
        admission_quota = "dcet"

    # 2. Fee category detection
    has_fee_word = bool(
        re.search(
            r"\b(fees?|tuition|cost|costs|charges?|pay|payment|amount|dues?|fine|penalty)\b",
            normalized_q,
        )
    )
    is_hostel = bool(re.search(r"\b(hostel|mess|food|boarding)\b", normalized_q))
    is_penalty = bool(re.search(r"\b(late|fine|penalty|delay|due\s+date|last\s+date)\b", normalized_q))
    is_tuition = bool(re.search(r"\b(tuition|college\s+fee|skill\s+lab)\b", normalized_q))

    fee_category = None
    if has_fee_word or raw_intent in {"fees_finance", "admission_fees", "admission_fee"}:
        if is_hostel:
            fee_category = "hostel"
        elif is_penalty:
            fee_category = "late_penalty"
        elif is_tuition:
            fee_category = "tuition"
        elif admission_quota:
            fee_category = "admission_fee"
        else:
            fee_category = "general_fee"

    # 3. Academic year & level
    academic_year = campus_ent.get("academic_year")
    year_match = re.search(r"\b(202[4-7](?:-2[5-8])?)\b", normalized_q)
    if year_match:
        academic_year = year_match.group(1)
    elif re.search(r"\b(1st\s+year|first\s+year|freshers?)\b", normalized_q):
        academic_year = "1st year"
    elif re.search(r"\b(2nd\s+year|second\s+year|lateral\s+entry)\b", normalized_q):
        academic_year = "2nd year"
    elif re.search(r"\b(higher\s+semester)\b", normalized_q):
        academic_year = "higher semester"

    # 4. Board distinction
    board = None
    if re.search(r"\b(karnataka(?:\s+board)?)\b", normalized_q) and not re.search(r"\bnon-?karnataka\b", normalized_q):
        board = "karnataka"
    elif re.search(r"\b(non-?karnataka|cbse|icse|outside\s+state)\b", normalized_q):
        board = "non_karnataka"
    elif re.search(r"\b(outside\s+india|nri|foreign)\b", normalized_q):
        board = "outside_india"

    # 5. Normalized Intent resolution
    normalized_intent = raw_intent

    # Check Cutoff intent
    has_cutoff_word = bool(re.search(r"\b(cutoff|cut-off|closing\s+rank|cut\s+off|last\s+rank)\b", normalized_q))
    is_cutoff_query = bool(
        campus_ent.get("data_type") in {"cutoff", "previous_cutoff"}
        or has_cutoff_word
        or (raw_intent == "admissions_cutoff" and not has_fee_word)
    )

    if is_cutoff_query:
        normalized_intent = "admissions_cutoff"
        main_entity = "admission_cutoff"
    elif has_fee_word or raw_intent in {"admission_fee", "admission_fees", "fees_finance"}:
        if is_hostel:
            normalized_intent = "hostel_fee"
            main_entity = "hostel"
        elif is_penalty:
            normalized_intent = "fee_penalty"
            main_entity = "fee_penalty"
        elif admission_quota or any(k in normalized_q for k in ["admission", "seat", "allotment", "quota", "comedk", "kcet", "kea", "cet"]):
            normalized_intent = "admission_fee"
            main_entity = "admission_fee"
        else:
            normalized_intent = "admission_fee"
            main_entity = "admission_fee"
    elif campus_ent.get("data_type") == "procedure" or raw_intent == "admissions_procedure" or any(k in normalized_q for k in ["procedure", "process", "steps", "verification", "how to apply"]):
        normalized_intent = "admissions_procedure"
        main_entity = "admissions_procedure"
    elif is_hostel:
        normalized_intent = "facilities_hostels"
        main_entity = "hostel"
    elif raw_intent in {"placements", "placements_careers"} or any(k in normalized_q for k in ["placement", "package", "salary", "internship", "recruiter", "tier 1", "highest offer"]):
        normalized_intent = "placements"
        main_entity = "placement"
    elif raw_intent == "academics_exams" or any(k in normalized_q for k in ["exam", "see", "fast track", "revaluation", "grade card", "credits", "syllabus", "timetable"]):
        normalized_intent = "academics_exams"
        main_entity = "examination"
    elif raw_intent == "departments_faculty" or any(k in normalized_q for k in ["department", "hod", "cabin", "faculty", "dean", "principal", "contact", "reception"]):
        normalized_intent = "departments_faculty"
        main_entity = "faculty"
    elif raw_intent in {"student_services", "student_services_clubs"} or any(k in normalized_q for k in ["club", "fest", "activity", "nss", "ncc", "hackathon", "bonafide"]):
        normalized_intent = "student_services"
        main_entity = "student_services"
    elif raw_intent in INTENT_DISPLAY_LABELS and raw_intent != "unknown":
        normalized_intent = raw_intent
        main_entity = raw_intent
    else:
        normalized_intent = "general_college_info"
        main_entity = "campus_info"

    # 6. Retrieval Expansion string
    expansion_terms = []
    if is_cutoff_query:
        exam_txt = campus_ent.get("exam") or "KCET COMEDK"
        branch_txt = campus_ent.get("branch") or ""
        expansion_terms.append(f"RVCE {exam_txt} {branch_txt} cutoff closing rank round 1 round 2".strip())
    elif admission_quota == "kea_cet":
        expansion_terms.append("KEA CET KCET government quota admission process fees details 1st Year B.E. tuition college fees")
    elif admission_quota == "comedk":
        expansion_terms.append("COMEDK COMED-K quota admission process fees details 1st Year B.E. candidate portal")
    elif admission_quota == "management":
        expansion_terms.append("management quota admission process fees details M.Tech MCA B.E. college fees")
    elif admission_quota == "pgcet":
        expansion_terms.append("PGCET M.Tech MCA admission procedure mandatory documents KEA balance fees")
    elif admission_quota == "multiple" or is_comparison:
        expansion_terms.append("KEA CET KCET COMEDK quota admission fees structure comparison 1st Year B.E.")
    elif is_hostel:
        expansion_terms.append("hostel fees room rent mess transportation boys girls hostel contact")
    elif is_penalty:
        expansion_terms.append("late fee penalty payment deadline extension circular Dean Academics")

    retrieval_expansion = " ".join(expansion_terms)

    structured_entities = {
        # Core domain entities (PART B)
        "exam": campus_ent.get("exam"),
        "branch": campus_ent.get("branch"),
        "branch_name": campus_ent.get("branch_name"),
        "admission_quota": admission_quota,
        "category": campus_ent.get("category"),
        "academic_year": academic_year,
        "year": campus_ent.get("year"),
        "counselling_round": campus_ent.get("counselling_round"),
        "company": campus_ent.get("company"),
        "ctc": campus_ent.get("ctc"),
        "ctc_raw": campus_ent.get("ctc_raw"),
        "data_type": campus_ent.get("data_type"),
        "is_comparison": is_comparison,
        # Backward compatibility entities
        "main_entity": main_entity,
        "detected_quotas": detected_quotas,
        "fee_category": fee_category,
        "board": board,
    }

    return {
        "normalized_intent": normalized_intent,
        "main_entity": main_entity,
        "entities": structured_entities,
        "retrieval_expansion": retrieval_expansion,
    }


def classify_query_intent(
    question: str,
    confidence_threshold: float = 0.14,
    dataset_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Classify a user query using the trained TF-IDF + Logistic Regression pipeline with safe unknown fallback."""
    cleaned = (question or "").strip()
    if not cleaned:
        extracted = extract_query_entities_and_intent("")
        return {
            "intent": "unknown",
            "normalized_intent": "unknown",
            "canonical_intent": "unknown",
            "raw_intent": "unknown",
            "intent_label": "Unknown",
            "confidence": 0.0,
            "is_confident": False,
            "routing": "general_fallback",
            "model": "logistic_regression_tfidf",
            "entities": extracted["entities"],
            "retrieval_expansion": extracted["retrieval_expansion"],
        }

    pipeline = get_intent_classifier(dataset_path=dataset_path)
    pred_intent = "unknown"
    confidence = 0.0
    is_confident = False

    if pipeline is not None:
        try:
            tfidf_step = pipeline.named_steps.get("tfidf")
            vec_valid = True
            if tfidf_step is not None:
                vec = tfidf_step.transform([cleaned])
                if vec.nnz == 0:
                    vec_valid = False

            if vec_valid:
                probs = pipeline.predict_proba([cleaned])[0]
                classes = list(pipeline.classes_)
                best_idx = int(probs.argmax())
                pred_intent = str(classes[best_idx])
                confidence = round(float(probs[best_idx]), 4)
                if confidence >= confidence_threshold:
                    is_confident = True
        except Exception as exc:
            logger.warning("Intent classification failed safely: %s", type(exc).__name__)

    # Extract entities and normalize intent semantically
    extracted = extract_query_entities_and_intent(
        cleaned, raw_intent=pred_intent, raw_confidence=confidence
    )
    normalized_intent = extracted["normalized_intent"]
    entities = extracted["entities"]
    retrieval_expansion = extracted["retrieval_expansion"]

    # Mark as confident if entities (quota, hostel, penalty) provide strong semantic signals
    has_strong_entity = bool(
        entities.get("admission_quota")
        or entities.get("fee_category")
        or entities.get("is_comparison")
        or entities.get("exam")
        or entities.get("branch")
        or entities.get("data_type")
    )
    final_confident = is_confident or has_strong_entity

    if not final_confident and (pred_intent == "unknown" or confidence < confidence_threshold):
        primary_intent = "unknown"
        normalized_intent = "unknown"
    else:
        # For backward-compatibility with tests expecting 'admission_fee' for fee queries
        primary_intent = "admission_fee" if normalized_intent in {"admission_fee", "admission_fees"} else normalized_intent

    return {
        "intent": primary_intent,
        "normalized_intent": normalized_intent,
        "canonical_intent": normalized_intent,
        "raw_intent": pred_intent,
        "intent_label": INTENT_DISPLAY_LABELS.get(normalized_intent, INTENT_DISPLAY_LABELS.get(pred_intent, pred_intent)),
        "confidence": confidence,
        "is_confident": final_confident,
        "routing": "intent_guided" if final_confident else "general_fallback",
        "model": "logistic_regression_tfidf",
        "entities": entities,
        "retrieval_expansion": retrieval_expansion,
    }


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
