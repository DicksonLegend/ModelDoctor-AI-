"""
/download_model and /models endpoints — download best model and list all versions.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from typing import Optional

from services.model_service import (
    get_model_path,
    get_best_model_version,
    list_models,
)

router = APIRouter()


@router.get("/download_model")
async def download_model(version: Optional[str] = Query(None)):
    """
    Download a model file (.pkl).
    If no version specified, downloads the best model.
    """
    if not version:
        version = get_best_model_version()
        if not version:
            raise HTTPException(
                status_code=404,
                detail="No models available. Please analyze a model first.",
            )

    try:
        path = get_model_path(version)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not path.exists():
        raise HTTPException(status_code=404, detail="Model file not found on disk")

    return FileResponse(
        path=str(path),
        filename=f"model_{version}.pkl",
        media_type="application/octet-stream",
    )


@router.get("/models")
async def get_models():
    """Return all registered model versions with metadata."""
    models = list_models()
    if not models:
        return {"models": [], "best_version": None}

    best = get_best_model_version()
    for m in models:
        m["is_best"] = (m["version"] == best)

    return {"models": models, "best_version": best}
