"""
/predict endpoint — make predictions using a saved model.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from app.schemas import PredictRequest, PredictResponse
from core.monitor import log_prediction, log_error
from services.model_service import load_model_by_version, get_best_model_version

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Make predictions using a stored model version.
    If no version specified, uses the best model.
    """
    try:
        # ── Resolve model version ────────────────────────────────────
        version = request.model_version
        if not version:
            version = get_best_model_version()
            if not version:
                raise HTTPException(
                    status_code=404,
                    detail="No models available. Please analyze a model first.",
                )

        # ── Load model ───────────────────────────────────────────────
        try:
            model = load_model_by_version(version)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

        # ── Prepare input data ───────────────────────────────────────
        if not request.data:
            raise HTTPException(status_code=400, detail="No input data provided")

        import pandas as pd
        input_df = pd.DataFrame(request.data)
        X = input_df.values.astype(np.float64)

        # ── Predict ──────────────────────────────────────────────────
        predictions = model.predict(X).tolist()

        # ── Log prediction ───────────────────────────────────────────
        log_prediction(version, len(predictions), predictions)

        return PredictResponse(
            predictions=predictions,
            model_version=version,
        )

    except HTTPException:
        raise
    except Exception as e:
        log_error("/predict", str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
