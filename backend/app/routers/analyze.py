"""
/analyze endpoint — the core analysis pipeline.
Accepts model (.pkl) + dataset (.csv) OR metrics (.json) + dataset (.csv).
"""

from __future__ import annotations

import json
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional

from app.schemas import AnalyzeResponse
from core.evaluator import evaluate_model, parse_metrics_json
from core.diagnosis import diagnose
from core.suggestions import get_rule_suggestions, enhance_with_ai
from core.health_score import calculate_health_score
from core.monitor import log_analysis, log_error
from services.model_service import load_model_from_file, save_model, get_next_version
from services.data_service import load_dataset, preprocess, split_data, auto_detect_target
from services.mlflow_service import start_run, log_metrics, log_params, log_model, end_run
from services.dvc_service import version_dataset

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    dataset_file: UploadFile = File(...),
    model_file: Optional[UploadFile] = File(None),
    metrics_file: Optional[UploadFile] = File(None),
    target_column: Optional[str] = Form(None),
):
    """
    Analyze a model's performance.

    Mode A: Upload model_file (.pkl) + dataset_file (.csv)
    Mode B: Upload metrics_file (.json) + dataset_file (.csv)
    """
    try:
        # ── Validate inputs ──────────────────────────────────────────
        if model_file is None and metrics_file is None:
            raise HTTPException(
                status_code=400,
                detail="Please upload either a model file (.pkl) or a metrics file (.json)",
            )

        # ── Load dataset ─────────────────────────────────────────────
        dataset_bytes = await dataset_file.read()
        df = load_dataset(dataset_bytes)

        # Auto-detect target if not specified
        if not target_column:
            target_column = auto_detect_target(df)

        if target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{target_column}' not found. Available: {list(df.columns)}",
            )

        # ── Preprocess data ──────────────────────────────────────────
        X, y, processed_df, feature_names = preprocess(df, target_column)
        X_train, X_test, y_train, y_test = split_data(X, y)

        # ── Get class distribution ───────────────────────────────────
        class_dist = df[target_column].value_counts().to_dict()
        class_dist = {str(k): int(v) for k, v in class_dist.items()}

        # ── Mode A: Model file uploaded ──────────────────────────────
        model = None
        if model_file is not None:
            model_bytes = await model_file.read()
            model = load_model_from_file(model_bytes)
            metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

        # ── Mode B: Metrics JSON uploaded ────────────────────────────
        elif metrics_file is not None:
            metrics_bytes = await metrics_file.read()
            metrics_json = json.loads(metrics_bytes.decode("utf-8"))
            metrics = parse_metrics_json(metrics_json)

        # ── DVC: Version the dataset ─────────────────────────────────
        from app.config import DATA_RAW_DIR
        dataset_path = DATA_RAW_DIR / dataset_file.filename
        with open(dataset_path, "wb") as f:
            f.write(dataset_bytes)
        version_dataset(dataset_path)

        # ── Diagnose ─────────────────────────────────────────────────
        diagnosis_results = diagnose(metrics, processed_df, target_column)

        # ── Generate suggestions ─────────────────────────────────────
        suggestions = get_rule_suggestions(diagnosis_results)
        suggestions = await enhance_with_ai(diagnosis_results, suggestions)

        # ── Health Score ─────────────────────────────────────────────
        health = calculate_health_score(metrics, processed_df, target_column)

        # ── Save model & register version ────────────────────────────
        version = get_next_version()
        if model is not None:
            save_model(model, version, metrics, health["score"])

        # ── MLflow logging ───────────────────────────────────────────
        try:
            with start_run(run_name=f"analyze_{version}"):
                log_metrics({
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "health_score": health["score"],
                })
                log_params({
                    "model_version": version,
                    "target_column": target_column,
                    "dataset_rows": len(df),
                    "dataset_cols": len(df.columns),
                    "n_features": len(feature_names),
                })
                if model is not None:
                    log_model(model)
        except Exception:
            pass  # MLflow logging is optional

        # ── Monitoring log ───────────────────────────────────────────
        log_analysis(version, metrics, health["score"])

        # ── Response ─────────────────────────────────────────────────
        return AnalyzeResponse(
            model_version=version,
            metrics=metrics,
            diagnosis=diagnosis_results,
            suggestions=suggestions,
            health_score=health,
            class_distribution=class_dist,
        )

    except HTTPException:
        raise
    except Exception as e:
        log_error("/analyze", str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
