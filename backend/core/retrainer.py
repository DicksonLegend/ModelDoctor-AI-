"""
Retrainer — applies diagnosed suggestions and retrains the model.
Uses only Logistic Regression and Random Forest (scikit-learn).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from core.evaluator import evaluate_model


def retrain_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    diagnosis: list[dict],
) -> tuple:
    """
    Apply improvements based on diagnosis and retrain the model.

    Returns: (new_model, new_metrics, cv_scores, applied_actions)
    """
    applied_actions = []
    X_tr = X_train.copy()
    y_tr = y_train.copy()
    X_te = X_test.copy()

    # ── Detect model type ────────────────────────────────────────────
    model_type = _detect_model_type(model)
    params = _get_current_params(model)

    # ── Build problem set ────────────────────────────────────────────
    problems = {d["problem"] for d in diagnosis}

    # ── 1. Handle Class Imbalance → SMOTE ────────────────────────────
    if "Severe Class Imbalance" in problems or "Class Imbalance" in problems:
        try:
            smote = SMOTE(random_state=42)
            X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
            applied_actions.append("Applied SMOTE resampling")
        except Exception:
            applied_actions.append("SMOTE failed — using class_weight='balanced' instead")
            params["class_weight"] = "balanced"

    # ── 2. Handle Feature Scaling ────────────────────────────────────
    if "Feature Scaling Needed" in problems:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        applied_actions.append("Applied StandardScaler")

    # ── 3. Handle Overfitting ────────────────────────────────────────
    if "Overfitting" in problems or "Mild Overfitting" in problems:
        if model_type == "random_forest":
            params["max_depth"] = min(params.get("max_depth", 20) or 20, 8)
            params["min_samples_split"] = max(params.get("min_samples_split", 2), 10)
            params["min_samples_leaf"] = max(params.get("min_samples_leaf", 1), 5)
            applied_actions.append("Reduced tree depth & increased min_samples")
        elif model_type == "logistic_regression":
            params["C"] = max(params.get("C", 1.0) * 0.1, 0.01)
            applied_actions.append("Increased regularization (reduced C)")

    # ── 4. Handle Underfitting ───────────────────────────────────────
    if "Underfitting" in problems or "Low Accuracy" in problems:
        if model_type == "random_forest":
            params["n_estimators"] = min(params.get("n_estimators", 100) * 2, 500)
            params["max_depth"] = min((params.get("max_depth") or 10) + 5, 30)
            applied_actions.append("Increased n_estimators and max_depth")
        elif model_type == "logistic_regression":
            params["C"] = min(params.get("C", 1.0) * 10, 100)
            params["max_iter"] = 1000
            applied_actions.append("Increased C (less regularization) and max_iter")

    # ── 5. Build & Train New Model ───────────────────────────────────
    new_model = _build_model(model_type, params)
    new_model.fit(X_tr, y_tr)

    # ── 6. Evaluate ──────────────────────────────────────────────────
    # Use original X_train for evaluation (not resampled) for fair comparison
    new_metrics = evaluate_model(new_model, X_train, y_train, X_te, y_test)

    # ── 7. Cross Validation ──────────────────────────────────────────
    try:
        cv_scores = cross_val_score(new_model, X_tr, y_tr, cv=5, scoring="accuracy").tolist()
    except Exception:
        cv_scores = []

    if not applied_actions:
        applied_actions.append("No changes needed — retrained with same parameters")

    return new_model, new_metrics, cv_scores, applied_actions


def _detect_model_type(model) -> str:
    """Detect whether the model is RandomForest or LogisticRegression."""
    class_name = type(model).__name__.lower()

    if "randomforest" in class_name:
        return "random_forest"
    elif "logistic" in class_name:
        return "logistic_regression"
    else:
        # Default to Random Forest for unknown models
        return "random_forest"


def _get_current_params(model) -> dict:
    """Extract current model parameters."""
    try:
        return model.get_params()
    except Exception:
        return {}


def _build_model(model_type: str, params: dict):
    """Build a new model with the given parameters."""
    # Clean params to only include valid ones
    if model_type == "random_forest":
        valid = [
            "n_estimators", "max_depth", "min_samples_split",
            "min_samples_leaf", "class_weight", "random_state",
        ]
        clean = {k: v for k, v in params.items() if k in valid}
        clean.setdefault("random_state", 42)
        clean.setdefault("n_estimators", 100)
        return RandomForestClassifier(**clean)

    elif model_type == "logistic_regression":
        valid = ["C", "max_iter", "class_weight", "random_state", "solver"]
        clean = {k: v for k, v in params.items() if k in valid}
        clean.setdefault("random_state", 42)
        clean.setdefault("max_iter", 500)
        clean.setdefault("solver", "lbfgs")
        return LogisticRegression(**clean)

    else:
        return RandomForestClassifier(n_estimators=100, random_state=42)
