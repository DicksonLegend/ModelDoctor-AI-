"""
Model Evaluator — compute classification metrics from a model + data.
Supports both Mode A (model file) and Mode B (pre-computed metrics JSON).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate a scikit-learn compatible model and return full metrics dict.
    """
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    # Determine if binary or multi-class
    unique_classes = np.unique(y_test)
    average = "binary" if len(unique_classes) == 2 else "weighted"

    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average=average, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, average=average, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, average=average, zero_division=0), 4),
        "confusion_matrix": cm,
        "train_accuracy": round(accuracy_score(y_train, y_train_pred), 4),
    }


def parse_metrics_json(metrics_json: dict) -> dict:
    """
    Parse and validate a user-uploaded metrics.json.
    Expected keys: accuracy, precision, recall, f1_score.
    Optional: confusion_matrix, train_accuracy.
    """
    required = ["accuracy", "precision", "recall", "f1_score"]
    for key in required:
        if key not in metrics_json:
            raise ValueError(f"Missing required metric: '{key}' in metrics.json")

    return {
        "accuracy": float(metrics_json["accuracy"]),
        "precision": float(metrics_json["precision"]),
        "recall": float(metrics_json["recall"]),
        "f1_score": float(metrics_json["f1_score"]),
        "confusion_matrix": metrics_json.get("confusion_matrix", []),
        "train_accuracy": float(metrics_json["train_accuracy"]) if "train_accuracy" in metrics_json else None,
    }
