"""
ModelDoctor AI+ — FastAPI Application Entry Point
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import FRONTEND_URL
from app.routers import analyze, retrain, predict, download
from services.mlflow_service import init_mlflow


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown events."""
    # Startup
    print("🚀 ModelDoctor AI+ starting up...")
    try:
        init_mlflow()
        print("✅ MLflow initialized")
    except Exception as e:
        print(f"⚠️  MLflow init warning: {e}")
    yield
    # Shutdown
    print("👋 ModelDoctor AI+ shutting down...")


app = FastAPI(
    title="ModelDoctor AI+",
    description=(
        "Production-level MLOps workflow for ML model diagnosis, "
        "retraining, comparison, and deployment."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_URL,
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────
app.include_router(analyze.router, tags=["Analysis"])
app.include_router(retrain.router, tags=["Retraining"])
app.include_router(predict.router, tags=["Prediction"])
app.include_router(download.router, tags=["Models"])


# ── Root ─────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "app": "ModelDoctor AI+",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/analyze", "/retrain", "/predict", "/download_model", "/models"],
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
