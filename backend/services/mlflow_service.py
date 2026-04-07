"""
MLflow Service — experiment tracking, metric/param logging, model registration.
"""

from __future__ import annotations

import mlflow
from app.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME


def init_mlflow():
    """Initialize MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def start_run(run_name: str):
    """Start a new MLflow run."""
    return mlflow.start_run(run_name=run_name)


def log_metrics(metrics: dict):
    """Log metrics to the active MLflow run."""
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value)


def log_params(params: dict):
    """Log parameters to the active MLflow run."""
    for key, value in params.items():
        try:
            mlflow.log_param(key, str(value))
        except Exception:
            pass  # skip non-serializable params


def log_model(model, model_name: str = "model"):
    """Log a scikit-learn model to MLflow."""
    mlflow.sklearn.log_model(model, model_name)


def end_run():
    """End the current MLflow run."""
    mlflow.end_run()


def get_experiment_runs() -> list[dict]:
    """Retrieve all runs from the current experiment."""
    try:
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            return []

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
        )

        results = []
        for _, row in runs.iterrows():
            run_data = {
                "run_id": row.get("run_id", ""),
                "run_name": row.get("tags.mlflow.runName", ""),
                "status": row.get("status", ""),
                "start_time": str(row.get("start_time", "")),
            }
            # Add metrics
            for col in runs.columns:
                if col.startswith("metrics."):
                    metric_name = col.replace("metrics.", "")
                    run_data[metric_name] = row[col]
                elif col.startswith("params."):
                    param_name = col.replace("params.", "")
                    run_data[f"param_{param_name}"] = row[col]

            results.append(run_data)

        return results

    except Exception:
        return []
