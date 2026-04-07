"""
/retrain endpoint — retrains the model with diagnosed improvements.
"""

from __future__ import annotations

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional

from app.schemas import RetrainResponse
from core.evaluator import evaluate_model
from core.diagnosis import diagnose
from core.health_score import calculate_health_score
from core.retrainer import retrain_model
from core.monitor import log_retrain, log_error
from services.model_service import (
    load_model_by_version,
    save_model,
    get_next_version,
)
from services.data_service import load_dataset, preprocess, split_data, auto_detect_target
from services.mlflow_service import start_run, log_metrics, log_params, log_model, end_run

router = APIRouter()


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

        X, y, processed_df, feature_names = preprocess(df, target_column)
        X_train, X_test, y_train, y_test = split_data(X, y)

        # ── Evaluate original model ──────────────────────────────────
        old_metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        old_health = calculate_health_score(old_metrics, processed_df, target_column)

        # ── Diagnose issues ──────────────────────────────────────────
        diagnosis_results = diagnose(old_metrics, processed_df, target_column)

        # ── Retrain with improvements ────────────────────────────────
        new_model, new_metrics, cv_scores, applied_actions = retrain_model(
            model, X_train, y_train, X_test, y_test, diagnosis_results
        )

        # ── Compute new health score ─────────────────────────────────
        new_health = calculate_health_score(
            new_metrics, processed_df, target_column, cv_scores
        )

        # ── Save new model ───────────────────────────────────────────
        new_version = get_next_version()
        save_model(new_model, new_version, new_metrics, new_health["score"])

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
                })
                log_model(new_model)
        except Exception:
            pass

        # ── Calculate improvements ───────────────────────────────────
        improvements = {
            "accuracy_delta": round(new_metrics["accuracy"] - old_metrics["accuracy"], 4),
            "precision_delta": round(new_metrics["precision"] - old_metrics["precision"], 4),
            "recall_delta": round(new_metrics["recall"] - old_metrics["recall"], 4),
            "f1_delta": round(new_metrics["f1_score"] - old_metrics["f1_score"], 4),
            "health_delta": new_health["score"] - old_health["score"],
            "actions_applied": applied_actions,
        }

        # ── Monitoring log ───────────────────────────────────────────
        log_retrain(
            model_version, new_version,
            old_metrics["accuracy"], new_metrics["accuracy"],
            applied_actions,
        )

        return RetrainResponse(
            new_model_version=new_version,
            old_metrics=old_metrics,
            new_metrics=new_metrics,
            improvements=improvements,
            new_health_score=new_health,
        )

    except HTTPException:
        raise
    except Exception as e:
        log_error("/retrain", str(e))
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")
