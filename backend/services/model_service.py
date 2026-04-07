"""
Model Service — load, save, list, and manage .pkl model files.
"""

from __future__ import annotations

import json
import joblib
from datetime import datetime, timezone
from pathlib import Path

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
    """Load a scikit-learn model from uploaded .pkl bytes."""
    import io
    return joblib.load(io.BytesIO(file_bytes))


def save_model(model, version: str, metrics: dict, health_score: int):
    """Save model to disk and register metadata."""
    path = MODELS_DIR / f"model_{version}.pkl"
    joblib.dump(model, path)

    _registry[version] = {
        "version": version,
        "path": str(path),
        "accuracy": metrics.get("accuracy", 0),
        "precision": metrics.get("precision", 0),
        "recall": metrics.get("recall", 0),
        "f1_score": metrics.get("f1_score", 0),
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
    return list(_registry.values())


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
