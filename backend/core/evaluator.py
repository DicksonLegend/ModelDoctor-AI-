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
)


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

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Error rates
    misclassified = int(np.sum(y_pred != y_test))
    total_test = len(y_test)
    error_rate = round(1.0 - acc, 4)

    # Per-class report
    try:
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        per_class = {}
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


def parse_metrics_json(metrics_json: dict) -> dict:
    """
    Parse and validate a user-uploaded metrics.json.
    Required keys: accuracy, precision, recall, f1_score.
    Optional: confusion_matrix, train_accuracy and all extended fields.
    """
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
    }
    return result
