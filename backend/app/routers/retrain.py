"""
/retrain endpoint — retrains the model with diagnosed improvements.
"""

from __future__ import annotations

import re
import numpy as np
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional

from app.schemas import RetrainResponse
from core.evaluator import evaluate_model, evaluate_regression_model
from core.diagnosis import diagnose
from core.health_score import calculate_health_score
from core.retrainer import retrain_model, retrain_regression_model
from core.suggestions import get_rule_suggestions, enhance_with_ai, build_plain_language_summary
from core.monitor import log_retrain, log_error
from services.model_service import (
    load_model_by_version,
    save_model,
    get_next_version,
)
from services.data_service import load_dataset, preprocess, split_data, auto_detect_target
from services.data_service import ensure_classification_target
from services.data_service import infer_task_type_from_target
from services.mlflow_service import start_run, log_metrics, log_params, log_model, end_run

router = APIRouter()


def _suggest_target_columns(df, current_target: str) -> list[str]:
    """Suggest likely classification target columns with low/moderate cardinality."""
    max_unique = min(40, max(10, int(len(df) * 0.05)))
    object_like = []
    numeric_like = []

    for col in df.columns:
        if col == current_target:
            continue
        nunique = int(df[col].nunique(dropna=True))
        if nunique < 2 or nunique > max_unique:
            continue
        if str(df[col].dtype) in ("object", "category", "bool"):
            object_like.append(col)
        else:
            numeric_like.append(col)

    return (object_like + numeric_like)[:6]


def _extract_version_round(version: str) -> int:
    """Extract numeric part from version strings like v1, v2, ..."""
    match = re.search(r"(\d+)$", version.strip())
    return int(match.group(1)) if match else 0


@router.post("/retrain", response_model=RetrainResponse)
async def retrain(
    dataset_file: UploadFile = File(...),
    model_version: str = Form(...),
    target_column: Optional[str] = Form(None),
):
    """
    Retrain a previously analyzed model with automatic improvements.
    Uses the same dataset + applies suggestions from diagnosis.
    """
    try:
        # ── Load existing model ──────────────────────────────────────
        try:
            model = load_model_by_version(model_version)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

        # ── Load & preprocess dataset ────────────────────────────────
        dataset_bytes = await dataset_file.read()
        df = load_dataset(dataset_bytes)

        if not target_column:
            target_column = auto_detect_target(df)

        if target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{target_column}' not found.",
            )

        task_type = infer_task_type_from_target(df[target_column])
        X, y, processed_df, feature_names = preprocess(df, target_column, task_type=task_type)
        if task_type == "classification":
            try:
                ensure_classification_target(y)
            except ValueError as ve:
                suggestions = _suggest_target_columns(df, target_column)
                hint = (
                    f" Try setting target_column to one of: {', '.join(suggestions)}"
                    if suggestions else
                    " Try setting target_column explicitly to your class label column."
                )
                raise HTTPException(status_code=400, detail=f"{ve}.{hint}")
        X_train, X_test, y_train, y_test = split_data(
            X,
            y,
            stratify=(task_type == "classification"),
        )

        # ── Evaluate original model ──────────────────────────────────
        old_metrics = (
            evaluate_model(model, X_train, y_train, X_test, y_test)
            if task_type == "classification"
            else evaluate_regression_model(model, X_train, y_train, X_test, y_test)
        )
        old_health = calculate_health_score(old_metrics, processed_df, target_column, task_type=task_type)

        # ── Diagnose issues ──────────────────────────────────────────
        diagnosis_results = diagnose(old_metrics, processed_df, target_column, task_type=task_type)

        # ── Retrain with improvements ────────────────────────────────
        retrain_round = _extract_version_round(model_version)
        if task_type == "classification":
            new_model, new_metrics, cv_scores, applied_actions = retrain_model(
                model, X_train, y_train, X_test, y_test, diagnosis_results, retrain_round
            )
        else:
            new_model, new_metrics, cv_scores, applied_actions = retrain_regression_model(
                model, X_train, y_train, X_test, y_test, diagnosis_results, retrain_round
            )

        # Detect whether the new model behavior is actually different
        try:
            old_pred = model.predict(X_test)
            new_pred = new_model.predict(X_test)
            if len(old_pred) == len(new_pred) and len(new_pred) > 0:
                prediction_change_rate = float(np.mean(old_pred != new_pred))
            else:
                prediction_change_rate = 1.0
        except Exception:
            prediction_change_rate = 1.0

        # ── Compute new health score ─────────────────────────────────
        new_health = calculate_health_score(
            new_metrics, processed_df, target_column, cv_scores, task_type=task_type
        )

        # Generate new diagnosis/suggestions for user-facing explanation
        new_diagnosis = diagnose(new_metrics, processed_df, target_column, task_type=task_type)
        new_suggestions = get_rule_suggestions(new_diagnosis)
        new_suggestions = await enhance_with_ai(new_diagnosis, new_suggestions)

        class_dist = df[target_column].value_counts().to_dict()
        class_dist = {str(k): int(v) for k, v in class_dist.items()}
        plain_language_summary = await build_plain_language_summary(
            metrics=new_metrics,
            diagnosis=new_diagnosis,
            suggestions=new_suggestions,
            health=new_health,
            class_distribution=class_dist,
        )

        # ── Calculate improvements ───────────────────────────────────
        improvements = {
            "accuracy_delta": round(new_metrics["accuracy"] - old_metrics["accuracy"], 4),
            "precision_delta": round(new_metrics["precision"] - old_metrics["precision"], 4),
            "recall_delta": round(new_metrics["recall"] - old_metrics["recall"], 4),
            "f1_delta": round(new_metrics["f1_score"] - old_metrics["f1_score"], 4),
            "health_delta": new_health["score"] - old_health["score"],
            "prediction_change_rate": round(prediction_change_rate, 4),
            "r2_delta": round((new_metrics.get("r2_score") or 0.0) - (old_metrics.get("r2_score") or 0.0), 4),
            "rmse_delta": round((new_metrics.get("rmse") or 0.0) - (old_metrics.get("rmse") or 0.0), 4),
            "actions_applied": applied_actions,
        }

        # ── Save model only when user-visible model quality changes ──
        # This avoids creating duplicate versions with identical dashboard values.
        core_keys = ["accuracy", "precision", "recall", "f1_score"]
        if task_type == "regression":
            core_keys = ["r2_score", "rmse", "mae"]
        def _safe_num(d: dict, key: str) -> float:
            v = d.get(key, 0.0)
            return float(v) if v is not None else 0.0

        same_core_metrics = all(
            round(_safe_num(new_metrics, k), 4) == round(_safe_num(old_metrics, k), 4)
            for k in core_keys
        )
        metric_keys = ["accuracy_delta", "precision_delta", "recall_delta", "f1_delta"]
        if task_type == "regression":
            metric_keys = ["r2_delta", "rmse_delta"]

        metric_change = any(abs(improvements[k]) >= 0.001 for k in metric_keys)
        meaningful_change = metric_change or (not same_core_metrics)

        if meaningful_change:
            new_version = get_next_version()
            save_model(new_model, new_version, new_metrics, new_health["score"])
        else:
            new_version = model_version
            applied_actions.append(
                "No user-visible metric change detected; existing model version retained"
            )

        # ── MLflow logging ───────────────────────────────────────────
        try:
            with start_run(run_name=f"retrain_{new_version}"):
                log_metrics({
                    "accuracy": new_metrics["accuracy"],
                    "precision": new_metrics["precision"],
                    "recall": new_metrics["recall"],
                    "f1_score": new_metrics["f1_score"],
                    "health_score": new_health["score"],
                })
                log_params({
                    "model_version": new_version,
                    "base_version": model_version,
                    "actions_applied": ", ".join(applied_actions),
                    "target_column": target_column,
                    "retained_existing_version": str(not meaningful_change),
                })
                if meaningful_change:
                    log_model(new_model)
        except Exception:
            pass

        # ── Monitoring log ───────────────────────────────────────────
        log_retrain(
            model_version, new_version,
            old_metrics["accuracy"], new_metrics["accuracy"],
            applied_actions,
        )

        return RetrainResponse(
            new_model_version=new_version,
            task_type=task_type,
            old_metrics=old_metrics,
            new_metrics=new_metrics,
            improvements=improvements,
            new_health_score=new_health,
            plain_language_summary=plain_language_summary,
            metrics_source="model_evaluation_code",
        )

    except HTTPException:
        raise
    except Exception as e:
        log_error("/retrain", str(e))
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")
