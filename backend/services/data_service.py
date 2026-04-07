"""
Data Service — load CSV, preprocess, split.
Handles missing values, encoding, and scaling automatically.
"""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_dataset(file_bytes: bytes) -> pd.DataFrame:
    """Load a CSV file from uploaded bytes."""
    return pd.read_csv(io.BytesIO(file_bytes))


def auto_detect_target(df: pd.DataFrame) -> str | None:
    """
    Heuristic to detect the target column.
    Looks for common target column names, or picks the last column.
    """
    common_names = [
        "target", "label", "class", "y", "output", "result",
        "survived", "species", "diagnosis", "category",
    ]
    for col in df.columns:
        if col.lower().strip() in common_names:
            return col

    # Fallback: last column
    return df.columns[-1]


def preprocess(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """
    Preprocess the dataset:
      1. Separate features and target
      2. Handle missing values (median for numeric, mode for categorical)
      3. Encode categorical features (LabelEncoder)
      4. Return (X, y, processed_df, feature_names)
    """
    df = df.copy()

    # Separate target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    y_series = df[target_col].copy()
    X_df = df.drop(columns=[target_col])

    # ── Handle missing values ────────────────────────────────────────
    for col in X_df.columns:
        if X_df[col].dtype in [np.float64, np.int64, float, int]:
            X_df[col] = X_df[col].fillna(X_df[col].median())
        else:
            mode_val = X_df[col].mode()
            X_df[col] = X_df[col].fillna(mode_val[0] if len(mode_val) > 0 else "unknown")

    # Handle missing in target
    if y_series.isnull().any():
        y_series = y_series.fillna(y_series.mode()[0])

    # ── Encode categorical features ──────────────────────────────────
    le_map = {}
    for col in X_df.columns:
        if X_df[col].dtype == object or X_df[col].dtype.name == "category":
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col].astype(str))
            le_map[col] = le

    # Encode target if categorical
    if y_series.dtype == object or y_series.dtype.name == "category":
        le = LabelEncoder()
        y_series = pd.Series(le.fit_transform(y_series.astype(str)), name=target_col)

    feature_names = X_df.columns.tolist()
    X = X_df.values.astype(np.float64)
    y = y_series.values.astype(np.int64)

    # Rebuild processed DataFrame for diagnosis
    processed_df = X_df.copy()
    processed_df[target_col] = y

    return X, y, processed_df, feature_names


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
