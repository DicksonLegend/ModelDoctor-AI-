"""
Pydantic models for all API request / response validation.
"""

from __future__ import annotations
from typing import Optional, Any
from pydantic import BaseModel


# ── Shared / Reusable ──────────────────────────────────────────────────

class MetricsOut(BaseModel):
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    confusion_matrix: list = []
    train_accuracy: Optional[float] = None
    # Extended metrics
    macro_precision: Optional[float] = None
    macro_recall: Optional[float] = None
    macro_f1: Optional[float] = None
    error_rate: Optional[float] = None
    misclassified: Optional[int] = None
    total_test_samples: Optional[int] = None
    n_classes: Optional[int] = None
    per_class: Optional[dict] = None
    # Regression metrics
    mae: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    r2_score: Optional[float] = None
    explained_variance: Optional[float] = None
    task_type: Optional[str] = None


class DiagnosisItem(BaseModel):
    problem: str
    severity: str          # "high" | "medium" | "low"
    reason: str


class SuggestionItem(BaseModel):
    issue: str
    action: str
    explanation: str


class HealthScoreOut(BaseModel):
    score: int             # 0-100
    status: str            # "Excellent" | "Good" | "Needs Tuning" | "Poor"
    breakdown: dict        # factor → sub-score


# ── /analyze ───────────────────────────────────────────────────────────

class AnalyzeResponse(BaseModel):
    model_version: str
    task_type: str = "classification"
    metrics: MetricsOut
    diagnosis: list[DiagnosisItem]
    suggestions: list[SuggestionItem]
    health_score: HealthScoreOut
    class_distribution: Optional[dict] = None
    plain_language_summary: Optional[str] = None
    metrics_source: str = "model_evaluation_code"


# ── /retrain ───────────────────────────────────────────────────────────

class RetrainRequest(BaseModel):
    model_version: str
    apply_suggestions: list[str] = []   # list of issue keys to apply


class RetrainResponse(BaseModel):
    new_model_version: str
    task_type: str = "classification"
    old_metrics: MetricsOut
    new_metrics: MetricsOut
    improvements: dict
    new_health_score: HealthScoreOut
    plain_language_summary: Optional[str] = None
    metrics_source: str = "model_evaluation_code"


# ── /predict ───────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    model_version: Optional[str] = None   # defaults to best
    data: list[dict]


class PredictResponse(BaseModel):
    predictions: list
    model_version: str


# ── /download_model ────────────────────────────────────────────────────

class ModelInfoOut(BaseModel):
    version: str
    accuracy: float
    health_score: int
    created_at: str


# ── Model Comparison ───────────────────────────────────────────────────

class ComparisonRow(BaseModel):
    version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    health_score: int
    is_best: bool = False
