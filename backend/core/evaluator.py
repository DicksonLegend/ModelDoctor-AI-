"""
Model Evaluator — compute comprehensive classification metrics.
Works with ANY scikit-learn compatible model (including Pipelines).
Always deterministic — same input = same output.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
)


MAX_CLASSES_FOR_DETAILED_REPORT = 120


def evaluate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate ANY scikit-learn compatible model and return full metrics dict.
    Works with raw models, Pipelines, and any estimator with .predict().
    """
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    # Determine averaging strategy
    unique_classes = np.unique(y_test)
    n_classes = len(unique_classes)
    is_binary = n_classes == 2

    # Core metrics
    acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, y_train_pred)

    # Weighted (default for multi-class)
    prec_w = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_w = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Macro
    prec_m = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_m = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_m = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # Confusion matrix can explode in memory for high-cardinality targets.
    if n_classes <= MAX_CLASSES_FOR_DETAILED_REPORT:
        cm = confusion_matrix(y_test, y_pred).tolist()
    else:
        cm = []

    # Error rates
    misclassified = int(np.sum(y_pred != y_test))
    total_test = len(y_test)
    error_rate = round(1.0 - acc, 4)

    # Per-class report
    try:
        per_class = {}
        if n_classes <= MAX_CLASSES_FOR_DETAILED_REPORT:
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            for cls_key, cls_metrics in report.items():
                if isinstance(cls_metrics, dict) and cls_key not in ("accuracy", "macro avg", "weighted avg"):
                    per_class[str(cls_key)] = {
                        "precision": round(cls_metrics.get("precision", 0), 4),
                        "recall": round(cls_metrics.get("recall", 0), 4),
                        "f1_score": round(cls_metrics.get("f1-score", 0), 4),
                        "support": int(cls_metrics.get("support", 0)),
                    }
    except Exception:
        per_class = {}

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec_w, 4),
        "recall": round(rec_w, 4),
        "f1_score": round(f1_w, 4),
        "confusion_matrix": cm,
        "train_accuracy": round(train_acc, 4),
        # Extended metrics
        "macro_precision": round(prec_m, 4),
        "macro_recall": round(rec_m, 4),
        "macro_f1": round(f1_m, 4),
        "error_rate": round(error_rate, 4),
        "misclassified": misclassified,
        "total_test_samples": total_test,
        "n_classes": n_classes,
        "per_class": per_class,
    }


def evaluate_regression_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Evaluate scikit-learn regressors and return regression metrics."""
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    # Keep these keys for backward compatibility in shared UI flows.
    bounded_r2 = max(min(float(r2), 1.0), -1.0)

    return {
        "accuracy": round(max(bounded_r2, 0.0), 4),
        "precision": round(max(bounded_r2, 0.0), 4),
        "recall": round(max(bounded_r2, 0.0), 4),
        "f1_score": round(max(bounded_r2, 0.0), 4),
        "confusion_matrix": [],
        "train_accuracy": round(max(min(train_r2, 1.0), -1.0), 4),
        "macro_precision": None,
        "macro_recall": None,
        "macro_f1": None,
        "error_rate": None,
        "misclassified": None,
        "total_test_samples": len(y_test),
        "n_classes": None,
        "per_class": {},
        "mae": round(float(mae), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(rmse), 4),
        "r2_score": round(float(r2), 4),
        "explained_variance": round(float(evs), 4),
        "task_type": "regression",
    }


def parse_metrics_json(metrics_json: dict) -> dict:
    """
    Parse and validate a user-uploaded metrics.json.
    Required keys: accuracy, precision, recall, f1_score.
    Optional: confusion_matrix, train_accuracy and all extended fields.
    """
    # Support both classification and regression metrics payloads.
    if "r2_score" in metrics_json or "rmse" in metrics_json or "mae" in metrics_json:
        return {
            "accuracy": float(metrics_json.get("r2_score", 0.0)),
            "precision": float(metrics_json.get("r2_score", 0.0)),
            "recall": float(metrics_json.get("r2_score", 0.0)),
            "f1_score": float(metrics_json.get("r2_score", 0.0)),
            "confusion_matrix": [],
            "train_accuracy": float(metrics_json["train_r2"]) if "train_r2" in metrics_json else None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "error_rate": None,
            "misclassified": None,
            "total_test_samples": int(metrics_json.get("total_test_samples", 0)),
            "n_classes": None,
            "per_class": {},
            "mae": float(metrics_json.get("mae", 0.0)),
            "mse": float(metrics_json.get("mse", 0.0)),
            "rmse": float(metrics_json.get("rmse", 0.0)),
            "r2_score": float(metrics_json.get("r2_score", 0.0)),
            "explained_variance": float(metrics_json.get("explained_variance", 0.0)),
            "task_type": "regression",
        }

    required = ["accuracy", "precision", "recall", "f1_score"]
    for key in required:
        if key not in metrics_json:
            raise ValueError(f"Missing required metric: '{key}' in metrics.json")

    result = {
        "accuracy": float(metrics_json["accuracy"]),
        "precision": float(metrics_json["precision"]),
        "recall": float(metrics_json["recall"]),
        "f1_score": float(metrics_json["f1_score"]),
        "confusion_matrix": metrics_json.get("confusion_matrix", []),
        "train_accuracy": float(metrics_json["train_accuracy"]) if "train_accuracy" in metrics_json else None,
        # Fill extended with defaults
        "macro_precision": float(metrics_json.get("macro_precision", metrics_json["precision"])),
        "macro_recall": float(metrics_json.get("macro_recall", metrics_json["recall"])),
        "macro_f1": float(metrics_json.get("macro_f1", metrics_json["f1_score"])),
        "error_rate": round(1.0 - float(metrics_json["accuracy"]), 4),
        "misclassified": int(metrics_json.get("misclassified", 0)),
        "total_test_samples": int(metrics_json.get("total_test_samples", 0)),
        "n_classes": int(metrics_json.get("n_classes", 2)),
        "per_class": metrics_json.get("per_class", {}),
        "mae": None,
        "mse": None,
        "rmse": None,
        "r2_score": None,
        "explained_variance": None,
        "task_type": "classification",
    }
    return result
