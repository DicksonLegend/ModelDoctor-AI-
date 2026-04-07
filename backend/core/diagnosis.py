"""
Model Diagnosis Engine — detects overfitting, underfitting, class imbalance,
feature quality issues, and data leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def diagnose(metrics: dict, df: pd.DataFrame, target_col: str) -> list[dict]:
    """
    Analyze metrics + dataset and return a list of diagnosed problems.

    Each item: { "problem": str, "severity": str, "reason": str }
    """
    problems = []

    # ── 1. Overfitting / Underfitting ─────────────────────────────────
    test_acc = metrics.get("accuracy", 0)
    train_acc = metrics.get("train_accuracy")

    if train_acc is not None:
        gap = train_acc - test_acc

        if gap > 0.15:
            problems.append({
                "problem": "Overfitting",
                "severity": "high",
                "reason": (
                    f"High training accuracy ({train_acc*100:.1f}%) vs "
                    f"test accuracy ({test_acc*100:.1f}%). "
                    f"Gap of {gap*100:.1f}% indicates the model memorized training data."
                ),
            })
        elif gap > 0.08:
            problems.append({
                "problem": "Mild Overfitting",
                "severity": "medium",
                "reason": (
                    f"Training accuracy ({train_acc*100:.1f}%) is moderately higher than "
                    f"test accuracy ({test_acc*100:.1f}%). Gap: {gap*100:.1f}%."
                ),
            })

    if test_acc < 0.6:
        problems.append({
            "problem": "Underfitting",
            "severity": "high",
            "reason": (
                f"Test accuracy is only {test_acc*100:.1f}%, which suggests the model "
                f"is too simple or features are not informative enough."
            ),
        })
    elif test_acc < 0.7:
        problems.append({
            "problem": "Low Accuracy",
            "severity": "medium",
            "reason": (
                f"Test accuracy of {test_acc*100:.1f}% is below acceptable threshold. "
                f"The model may need better features or tuning."
            ),
        })

    # ── 2. Class Imbalance ────────────────────────────────────────────
    if target_col in df.columns:
        class_counts = df[target_col].value_counts()
        if len(class_counts) >= 2:
            majority = class_counts.iloc[0]
            minority = class_counts.iloc[-1]
            ratio = majority / max(minority, 1)

            if ratio > 5:
                problems.append({
                    "problem": "Severe Class Imbalance",
                    "severity": "high",
                    "reason": (
                        f"Majority class has {majority} samples vs minority {minority} "
                        f"(ratio {ratio:.1f}:1). This heavily biases the model."
                    ),
                })
            elif ratio > 3:
                problems.append({
                    "problem": "Class Imbalance",
                    "severity": "medium",
                    "reason": (
                        f"Class distribution is skewed with ratio {ratio:.1f}:1. "
                        f"Consider resampling or using class weights."
                    ),
                })

    # ── 3. Feature Quality — Missing Values ──────────────────────────
    missing_pct = (df.isnull().sum() / len(df) * 100)
    high_missing = missing_pct[missing_pct > 20]
    if len(high_missing) > 0:
        cols = ", ".join(high_missing.index.tolist()[:5])
        problems.append({
            "problem": "High Missing Values",
            "severity": "medium",
            "reason": (
                f"Columns with >20% missing data: {cols}. "
                f"This can reduce model performance significantly."
            ),
        })

    any_missing = missing_pct[missing_pct > 0]
    if len(any_missing) > 0 and len(high_missing) == 0:
        problems.append({
            "problem": "Missing Values Present",
            "severity": "low",
            "reason": (
                f"{len(any_missing)} column(s) have missing values. "
                f"Ensure proper imputation is applied."
            ),
        })

    # ── 4. Feature Quality — Low Variance ────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        variances = df[numeric_cols].var()
        low_var = variances[variances < 0.01]
        if len(low_var) > 0:
            cols = ", ".join(low_var.index.tolist()[:5])
            problems.append({
                "problem": "Low Variance Features",
                "severity": "low",
                "reason": (
                    f"Columns with near-zero variance: {cols}. "
                    f"These features add little predictive value."
                ),
            })

    # ── 5. Data Leakage Warning ──────────────────────────────────────
    if test_acc > 0.99 and train_acc is not None and train_acc > 0.99:
        problems.append({
            "problem": "Possible Data Leakage",
            "severity": "high",
            "reason": (
                "Both train and test accuracy exceed 99%. "
                "This is suspiciously high and may indicate data leakage."
            ),
        })

    # ── 6. Scaling Issues ────────────────────────────────────────────
    if len(numeric_cols) >= 2:
        ranges = df[numeric_cols].max() - df[numeric_cols].min()
        if ranges.max() / max(ranges.min(), 1e-10) > 100:
            problems.append({
                "problem": "Feature Scaling Needed",
                "severity": "medium",
                "reason": (
                    "Numeric features have very different scales. "
                    "Applying StandardScaler or MinMaxScaler is recommended."
                ),
            })

    # If no problems found
    if not problems:
        problems.append({
            "problem": "No Major Issues Detected",
            "severity": "low",
            "reason": "The model and data appear to be in good shape.",
        })

    return problems
