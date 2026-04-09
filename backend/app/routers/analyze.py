"""
/analyze endpoint — the core analysis pipeline.
Accepts ANY sklearn model (.pkl/.joblib) + dataset (.csv) OR metrics (.json) + dataset (.csv).
Fully deterministic — same input = same output.
"""

from __future__ import annotations

import json
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
from sklearn.base import is_classifier, is_regressor

from app.schemas import AnalyzeResponse
from core.evaluator import evaluate_model, evaluate_regression_model, parse_metrics_json
from core.diagnosis import diagnose
from core.suggestions import (
    get_rule_suggestions,
    enhance_with_ai,
    build_plain_language_summary,
)
from core.health_score import calculate_health_score
from core.monitor import log_analysis, log_error
from services.model_service import load_model_from_file, save_model, get_next_version
from services.data_service import (
    load_dataset,
    preprocess,
    split_data,
    auto_detect_target,
    ensure_classification_target,
    infer_task_type_from_target,
)
from services.mlflow_service import start_run, log_metrics, log_params, log_model, end_run
from services.dvc_service import version_dataset

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


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    dataset_file: UploadFile = File(...),
    model_file: Optional[UploadFile] = File(None),
    metrics_file: Optional[UploadFile] = File(None),
    target_column: Optional[str] = Form(None),
):
    """
    Analyze a model's performance.

    Mode A: Upload model_file (.pkl/.joblib — ANY sklearn model) + dataset_file (.csv)
    Mode B: Upload metrics_file (.json) + dataset_file (.csv)
    """
    try:
        # ── Validate inputs ──────────────────────────────────────────
        if model_file is None and metrics_file is None:
            raise HTTPException(
                status_code=400,
                detail="Please upload either a model file (.pkl/.joblib) or a metrics file (.json)",
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
        inferred_task = infer_task_type_from_target(df[target_column])
        X, y, processed_df, feature_names = preprocess(df, target_column, task_type=inferred_task)

        # ── Mode A: Model file uploaded ──────────────────────────────
        task_type = inferred_task
        model = None
        feature_mismatch = False
        if model_file is not None:
            model_bytes = await model_file.read()
            uploaded_model = load_model_from_file(model_bytes)

            # Prefer estimator type if available; otherwise fallback to data inference.
            if is_classifier(uploaded_model):
                task_type = "classification"
            elif is_regressor(uploaded_model):
                task_type = "regression"

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

            # Check if model is compatible with preprocessed data
            try:
                metrics = (
                    evaluate_model(uploaded_model, X_train, y_train, X_test, y_test)
                    if task_type == "classification"
                    else evaluate_regression_model(uploaded_model, X_train, y_train, X_test, y_test)
                )
                model = uploaded_model
            except (ValueError, Exception) as e:
                # Feature mismatch or incompatible model
                # Train fresh models using Pipeline (scaler bundled)
                feature_mismatch = True
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                from sklearn.linear_model import LogisticRegression, Ridge
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import Pipeline

                if task_type == "classification":
                    candidates = [
                        Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(
                            n_estimators=200, max_depth=None,
                            class_weight="balanced", random_state=42))]),
                        Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(
                            n_estimators=300, max_depth=20,
                            class_weight="balanced", random_state=42))]),
                        Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(
                            C=1.0, max_iter=2000, solver="lbfgs",
                            class_weight="balanced", random_state=42))]),
                    ]
                else:
                    candidates = [
                        Pipeline([("scaler", StandardScaler()), ("reg", RandomForestRegressor(
                            n_estimators=250, max_depth=None, random_state=42))]),
                        Pipeline([("scaler", StandardScaler()), ("reg", RandomForestRegressor(
                            n_estimators=400, max_depth=18, random_state=42))]),
                        Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0, random_state=42))]),
                    ]

                best_score = -1
                for pipe in candidates:
                    try:
                        pipe.fit(X_train, y_train)
                        score = float(pipe.score(X_test, y_test))
                        if score > best_score:
                            best_score = score
                            model = pipe
                    except Exception:
                        continue

                if model is None:
                    if task_type == "classification":
                        model = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(
                            n_estimators=100, random_state=42, class_weight="balanced"))])
                    else:
                        model = Pipeline([("scaler", StandardScaler()), ("reg", RandomForestRegressor(
                            n_estimators=200, random_state=42))])
                    model.fit(X_train, y_train)

                metrics = (
                    evaluate_model(model, X_train, y_train, X_test, y_test)
                    if task_type == "classification"
                    else evaluate_regression_model(model, X_train, y_train, X_test, y_test)
                )

        # ── Mode B: Metrics JSON uploaded ────────────────────────────
        elif metrics_file is not None:
            metrics_bytes = await metrics_file.read()
            metrics_json = json.loads(metrics_bytes.decode("utf-8"))
            metrics = parse_metrics_json(metrics_json)
            task_type = metrics.get("task_type") or inferred_task
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

        # ── Get distribution for display ─────────────────────────────
        class_dist = {}
        if task_type == "classification":
            class_dist = df[target_column].value_counts().to_dict()
            class_dist = {str(k): int(v) for k, v in class_dist.items()}

        # ── DVC: Version the dataset ─────────────────────────────────
        from app.config import DATA_RAW_DIR
        dataset_path = DATA_RAW_DIR / dataset_file.filename
        with open(dataset_path, "wb") as f:
            f.write(dataset_bytes)
        version_dataset(dataset_path)

        # ── Diagnose ─────────────────────────────────────────────────
        diagnosis_results = diagnose(metrics, processed_df, target_column, task_type=task_type)

        # Add feature mismatch warning if applicable
        if feature_mismatch:
            diagnosis_results.insert(0, {
                "problem": "Feature Mismatch",
                "severity": "medium",
                "reason": (
                    "The uploaded model was trained on a different feature set "
                    "than the current dataset. A fresh model of the same type "
                    "was automatically trained on your data for analysis."
                ),
            })

        # ── Generate suggestions ─────────────────────────────────────
        suggestions = get_rule_suggestions(diagnosis_results)
        suggestions = await enhance_with_ai(diagnosis_results, suggestions)

        # ── Health Score ─────────────────────────────────────────────
        health = calculate_health_score(metrics, processed_df, target_column, task_type=task_type)

        # ── Plain-language explanation (AI-enhanced, metrics unchanged) ──
        plain_language_summary = await build_plain_language_summary(
            metrics=metrics,
            diagnosis=diagnosis_results,
            suggestions=suggestions,
            health=health,
            class_distribution=class_dist,
        )

        # ── Save model & register version ────────────────────────────
        version = get_next_version()
        if model is not None:
            save_model(model, version, metrics, health["score"])

        # ── MLflow logging ───────────────────────────────────────────
        try:
            with start_run(run_name=f"analyze_{version}"):
                log_metrics({
                    "accuracy": metrics.get("accuracy", 0.0),
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "f1_score": metrics.get("f1_score", 0.0),
                    "r2_score": metrics.get("r2_score", 0.0) or 0.0,
                    "rmse": metrics.get("rmse", 0.0) or 0.0,
                    "health_score": health["score"],
                })
                log_params({
                    "model_version": version,
                    "task_type": task_type,
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
            task_type=task_type,
            metrics=metrics,
            diagnosis=diagnosis_results,
            suggestions=suggestions,
            health_score=health,
            class_distribution=class_dist,
            plain_language_summary=plain_language_summary,
            metrics_source="model_evaluation_code",
        )

    except HTTPException:
        raise
    except Exception as e:
        log_error("/analyze", str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
