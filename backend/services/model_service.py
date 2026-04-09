"""
Model Service — load, save, list, and manage .pkl / .joblib model files.
"""

from __future__ import annotations

import json
import joblib
from datetime import datetime, timezone
from pathlib import Path
from sklearn.base import is_classifier, is_regressor

from app.config import MODELS_DIR


# In-memory registry of model metadata
_registry: dict[str, dict] = {}
_REGISTRY_FILE = MODELS_DIR / "_registry.json"


def _load_registry():
    """Load registry from disk on startup."""
    global _registry
    if _REGISTRY_FILE.exists():
        with open(_REGISTRY_FILE, "r") as f:
            _registry = json.load(f)


def _save_registry():
    """Persist registry to disk."""
    with open(_REGISTRY_FILE, "w") as f:
        json.dump(_registry, f, indent=2)


def load_model_from_file(file_bytes: bytes):
    """Load a scikit-learn model from uploaded .pkl or .joblib bytes."""
    import io
    import pickle
    buf = io.BytesIO(file_bytes)
    try:
        return joblib.load(buf)
    except Exception:
        buf.seek(0)
        return pickle.load(buf)


def save_model(model, version: str, metrics: dict, health_score: int, task_type: str = "classification"):
    """Save model to disk and register metadata."""
    path = MODELS_DIR / f"model_{version}.pkl"
    joblib.dump(model, path)

    _registry[version] = {
        "version": version,
        "path": str(path),
        "task_type": task_type,
        "accuracy": metrics.get("accuracy", 0),
        "precision": metrics.get("precision", 0),
        "recall": metrics.get("recall", 0),
        "f1_score": metrics.get("f1_score", 0),
        "r2_score": metrics.get("r2_score"),
        "rmse": metrics.get("rmse"),
        "mae": metrics.get("mae"),
        "explained_variance": metrics.get("explained_variance"),
        "health_score": health_score,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_registry()
    return path


def load_model_by_version(version: str):
    """Load a model by its version string."""
    if version not in _registry:
        raise FileNotFoundError(f"Model version '{version}' not found")
    path = Path(_registry[version]["path"])
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def get_best_model_version() -> str | None:
    """Return the version string of the highest health-score model."""
    if not _registry:
        return None
    best = max(_registry.values(), key=lambda m: m.get("health_score", 0))
    return best["version"]


def get_model_path(version: str) -> Path:
    """Get the file path for a model version."""
    if version not in _registry:
        raise FileNotFoundError(f"Model version '{version}' not found")
    return Path(_registry[version]["path"])


def list_models() -> list[dict]:
    """Return all registered model versions with metadata."""
    changed = False
    models = []

    for version, meta in _registry.items():
        m = dict(meta)
        inferred_task = _infer_task_type_from_model_path(m.get("path", ""))
        task_type = m.get("task_type")

        # Correct stale task_type values from legacy runs.
        if not task_type or task_type != inferred_task:
            task_type = inferred_task
            m["task_type"] = task_type
            changed = True

        if task_type == "regression":
            # Backfill legacy regression entries that were saved before
            # regression-specific fields were added to registry.
            if m.get("r2_score") is None:
                m["r2_score"] = m.get("accuracy", 0.0)
                changed = True
            if m.get("explained_variance") is None:
                m["explained_variance"] = m.get("precision", m.get("r2_score", 0.0))
                changed = True

        if m != meta:
            _registry[version] = m

        models.append(m)

    if changed:
        _save_registry()

    return models


def _infer_task_type_from_model_path(path_str: str) -> str:
    """Infer task type by loading saved sklearn model/pipeline."""
    try:
        path = Path(path_str)
        if not path.exists():
            return "classification"

        model = joblib.load(path)
        est = model
        if hasattr(model, "named_steps"):
            if "clf" in model.named_steps:
                est = model.named_steps["clf"]
            elif "reg" in model.named_steps:
                est = model.named_steps["reg"]
            elif hasattr(model, "steps") and model.steps:
                est = model.steps[-1][1]

        if is_regressor(est):
            return "regression"
        if is_classifier(est):
            return "classification"
    except Exception:
        pass

    return "classification"


def get_next_version() -> str:
    """Generate the next version string (v1, v2, v3...)."""
    if not _registry:
        return "v1"
    existing = [k for k in _registry if k.startswith("v")]
    nums = []
    for v in existing:
        try:
            nums.append(int(v[1:]))
        except ValueError:
            continue
    return f"v{max(nums, default=0) + 1}"


# Load registry on module import
_load_registry()
