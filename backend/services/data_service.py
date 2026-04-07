"""
Data Service — load CSV, preprocess, split.
Handles all types of CSV data robustly.
Fully deterministic — same data = same output.
"""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_dataset(file_bytes: bytes) -> pd.DataFrame:
    """Load a CSV file from uploaded bytes. Handles various encodings."""
    try:
        return pd.read_csv(io.BytesIO(file_bytes))
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")


def auto_detect_target(df: pd.DataFrame) -> str:
    """
    Heuristic to detect the target column.
    Looks for common target column names, or picks the last column.
    """
    common_names = [
        "target", "label", "class", "y", "output", "result",
        "survived", "species", "diagnosis", "category", "type",
        "status", "grade", "rating", "sentiment", "is_fraud",
        "default", "churn", "outcome",
    ]
    for col in df.columns:
        if col.lower().strip() in common_names:
            return col

    # Fallback: last column
    return df.columns[-1]


def preprocess(
    df: pd.DataFrame,
    target_col: str,
) -> tuple:
    """
    Preprocess the dataset:
      1. Separate features and target
      2. Handle missing values (median for numeric, mode for categorical)
      3. Encode categorical features (LabelEncoder)
      4. Drop constant / ID-like columns
      5. Return (X, y, processed_df, feature_names)

    Fully deterministic — same input always produces same output.
    """
    df = df.copy()

    # Separate target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    y_series = df[target_col].copy()
    X_df = df.drop(columns=[target_col])

    # ── Drop ID-like columns (unique values == n_rows) ───────────────
    cols_to_drop = []
    for col in X_df.columns:
        n_unique = X_df[col].nunique()
        # Drop if all unique (likely an ID column) or all same value
        if n_unique == len(X_df) or n_unique <= 1:
            cols_to_drop.append(col)
    if cols_to_drop:
        X_df = X_df.drop(columns=cols_to_drop)

    # ── Handle missing values ────────────────────────────────────────
    for col in X_df.columns:
        if X_df[col].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
            median_val = X_df[col].median()
            X_df[col] = X_df[col].fillna(median_val)
        else:
            mode_val = X_df[col].mode()
            fill = mode_val.iloc[0] if len(mode_val) > 0 else "unknown"
            X_df[col] = X_df[col].fillna(fill)

    # Handle missing in target
    if y_series.isnull().any():
        mode_val = y_series.mode()
        y_series = y_series.fillna(mode_val.iloc[0] if len(mode_val) > 0 else 0)

    # ── Encode categorical features ──────────────────────────────────
    for col in X_df.columns:
        if X_df[col].dtype == object or X_df[col].dtype.name == "category":
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col].astype(str))

    # Encode target if categorical
    if y_series.dtype == object or y_series.dtype.name == "category":
        le = LabelEncoder()
        y_series = pd.Series(le.fit_transform(y_series.astype(str)), name=target_col)

    # ── Ensure all numeric ───────────────────────────────────────────
    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce").fillna(0)

    feature_names = X_df.columns.tolist()
    X = X_df.values.astype(np.float64)
    y = y_series.values

    # Make sure y is integer for classification
    try:
        y = y.astype(np.int64)
    except (ValueError, TypeError):
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))

    # Rebuild processed DataFrame for diagnosis
    processed_df = X_df.copy()
    processed_df[target_col] = y

    return X, y, processed_df, feature_names


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Split data into train/test sets.
    Deterministic — always uses random_state=42.
    Handles stratify failures gracefully.
    """
    try:
        # Try stratified split first
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        # Stratify fails when a class has too few samples
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
