"""
Monitoring Module — logs predictions, errors, and retraining history.
Simple JSON-based file logging.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import LOGS_DIR


def _log_file(name: str) -> Path:
    return LOGS_DIR / f"{name}.jsonl"


def _append_log(name: str, entry: dict):
    """Append a JSON line to the named log file."""
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    path = _log_file(name)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def log_prediction(model_version: str, input_count: int, predictions: list):
    """Log a batch of predictions."""
    _append_log("predictions", {
        "model_version": model_version,
        "input_count": input_count,
        "sample_predictions": predictions[:10],  # store max 10
    })


def log_analysis(model_version: str, metrics: dict, health_score: int):
    """Log an analysis event."""
    _append_log("analyses", {
        "model_version": model_version,
        "accuracy": metrics.get("accuracy"),
        "health_score": health_score,
    })


def log_retrain(old_version: str, new_version: str, old_acc: float, new_acc: float, actions: list):
    """Log a retraining event."""
    _append_log("retraining", {
        "old_version": old_version,
        "new_version": new_version,
        "old_accuracy": old_acc,
        "new_accuracy": new_acc,
        "actions_applied": actions,
    })


def log_error(endpoint: str, error: str):
    """Log an error event."""
    _append_log("errors", {
        "endpoint": endpoint,
        "error": str(error),
    })


def get_logs(name: str, limit: int = 50) -> list[dict]:
    """Read the last `limit` log entries from a log file."""
    path = _log_file(name)
    if not path.exists():
        return []

    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return entries[-limit:]
