"""
Model Health Score Calculator — produces a 0-100 score with breakdown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_health_score(
    metrics: dict,
    df: pd.DataFrame,
    target_col: str,
    cv_scores: list[float] | None = None,
    task_type: str = "classification",
) -> dict:
    """
    Calculate an overall health score (0-100) for the model.

    Factors & Weights:
      - Accuracy:            30%
      - Generalization Gap:  25%
      - Data Quality:        20%
      - Class Balance:       15%
      - Stability (CV):      10%
    """
    breakdown = {}

    if task_type == "regression":
        return _calculate_regression_health(metrics, df, cv_scores)

    # ── 1. Accuracy Score (30%) ──────────────────────────────────────
    acc = metrics.get("accuracy", 0)
    acc_score = min(acc * 100, 100)
    breakdown["accuracy"] = round(acc_score)

    # ── 2. Generalization Gap (25%) ─────────────────────────────────
    train_acc = metrics.get("train_accuracy")
    if train_acc is not None:
        gap = abs(train_acc - acc)
        # 0% gap = 100 score, 30%+ gap = 0 score
        gap_score = max(0, 100 - (gap * 100 / 0.30) * 100 / 100)
        gap_score = min(gap_score, 100)
    else:
        gap_score = 70  # neutral score when no train data available
    breakdown["generalization"] = round(gap_score)

    # ── 3. Data Quality (20%) ───────────────────────────────────────
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    missing_ratio = missing_cells / max(total_cells, 1)
    # 0% missing = 100, 50%+ missing = 0
    quality_score = max(0, 100 - (missing_ratio * 200))
    breakdown["data_quality"] = round(quality_score)

    # ── 4. Class Balance (15%) ──────────────────────────────────────
    if target_col in df.columns:
        class_counts = df[target_col].value_counts()
        if len(class_counts) >= 2:
            majority = class_counts.iloc[0]
            minority = class_counts.iloc[-1]
            ratio = majority / max(minority, 1)
            # ratio 1:1 = 100, ratio 10:1+ = 0
            balance_score = max(0, 100 - (ratio - 1) * 100 / 9)
        else:
            balance_score = 100
    else:
        balance_score = 70
    breakdown["class_balance"] = round(balance_score)

    # ── 5. Stability — Cross Validation Variance (10%) ──────────────
    if cv_scores and len(cv_scores) > 1:
        cv_std = np.std(cv_scores)
        # std 0 = 100, std 0.1+ = 0
        stability_score = max(0, 100 - cv_std * 1000)
    else:
        stability_score = 70  # neutral when CV not available
    breakdown["stability"] = round(stability_score)

    # ── Weighted Final Score ────────────────────────────────────────
    weights = {
        "accuracy": 0.30,
        "generalization": 0.25,
        "data_quality": 0.20,
        "class_balance": 0.15,
        "stability": 0.10,
    }
    final_score = sum(breakdown[k] * weights[k] for k in weights)
    final_score = round(min(max(final_score, 0), 100))

    # Status label
    if final_score >= 85:
        status = "Excellent"
    elif final_score >= 70:
        status = "Good"
    elif final_score >= 50:
        status = "Needs Tuning"
    else:
        status = "Poor"

    return {
        "score": final_score,
        "status": status,
        "breakdown": breakdown,
    }


def _calculate_regression_health(metrics: dict, df: pd.DataFrame, cv_scores: list[float] | None = None) -> dict:
    """Health score for regression models."""
    breakdown = {}

    r2 = float(metrics.get("r2_score", 0.0) or 0.0)
    r2_score = max(0.0, min(100.0, ((r2 + 1.0) / 2.0) * 100.0))
    breakdown["fit_quality"] = round(r2_score)

    train_r2 = metrics.get("train_accuracy")
    if train_r2 is not None:
        gap = abs(float(train_r2) - r2)
        gen_score = max(0.0, 100.0 - min(1.0, gap / 0.4) * 100.0)
    else:
        gen_score = 70.0
    breakdown["generalization"] = round(gen_score)

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    missing_ratio = missing_cells / max(total_cells, 1)
    quality_score = max(0.0, 100.0 - (missing_ratio * 200.0))
    breakdown["data_quality"] = round(quality_score)

    rmse = float(metrics.get("rmse", 0.0) or 0.0)
    target_std = float(df.iloc[:, -1].std()) if len(df.columns) > 0 else 0.0
    if target_std > 0:
        rel_err = rmse / target_std
        error_score = max(0.0, 100.0 - min(2.0, rel_err) * 50.0)
    else:
        error_score = 70.0
    breakdown["error_quality"] = round(error_score)

    if cv_scores and len(cv_scores) > 1:
        cv_std = np.std(cv_scores)
        stability_score = max(0.0, 100.0 - min(0.2, cv_std) * 500.0)
    else:
        stability_score = 70.0
    breakdown["stability"] = round(stability_score)

    weights = {
        "fit_quality": 0.35,
        "generalization": 0.25,
        "data_quality": 0.20,
        "error_quality": 0.10,
        "stability": 0.10,
    }
    final_score = sum(breakdown[k] * weights[k] for k in weights)
    final_score = round(min(max(final_score, 0), 100))

    if final_score >= 85:
        status = "Excellent"
    elif final_score >= 70:
        status = "Good"
    elif final_score >= 50:
        status = "Needs Tuning"
    else:
        status = "Poor"

    return {
        "score": final_score,
        "status": status,
        "breakdown": breakdown,
    }
