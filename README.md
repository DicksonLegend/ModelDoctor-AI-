# ModelDoctor AI+

Production-focused MLOps project for model diagnosis, retraining, version comparison, and operational tracking.

This project provides:
- FastAPI backend for analysis and retraining workflows
- React frontend for upload, diagnostics, comparisons, and monitoring views
- MLflow tracking for experiment metadata and metrics
- DVC integration hooks for dataset versioning
- Dockerized deployment for backend, frontend, and MLflow

---

## 1. What This Project Does

ModelDoctor helps you evaluate and improve machine learning models through a guided pipeline:

1. Upload model + dataset, or metrics + dataset
2. Evaluate performance (classification and regression supported)
3. Diagnose common ML issues (overfitting, imbalance, missing data, etc.)
4. Generate actionable suggestions (rule-based, optional Gemini enhancement)
5. Compute model health score (0-100) with factor breakdown
6. Retrain with targeted hyperparameter strategies
7. Save model versions and compare them over time
8. Track runs in MLflow and write operational logs

---

## 2. Architecture Overview

### 2.1 High-level flow

```text
Client (React)
  -> FastAPI (/analyze, /retrain, /predict, /models, /download_model)
    -> Core evaluation + diagnosis + health scoring + retraining
    -> Model registry on disk (models/_registry.json)
    -> Logs (jsonl files under backend/logs)
    -> MLflow tracking (backend/mlruns)
    -> DVC dataset add hook (if dvc installed)
```

### 2.2 Services in Docker Compose

- backend (FastAPI, port 8000)
- frontend (Nginx serving React build, port 3000)
- mlflow (MLflow server, port 5000)

Frontend calls backend via /api in Docker through Nginx reverse proxy.

---

## 3. Repository Structure

```text
MLOPS_Project/
|- backend/
|  |- app/
|  |  |- main.py                # FastAPI app, routers, CORS, health/root
|  |  |- config.py              # env-based settings and path setup
|  |  |- schemas.py             # Pydantic request/response schemas
|  |  |- routers/
|  |     |- analyze.py          # POST /analyze
|  |     |- retrain.py          # POST /retrain
|  |     |- predict.py          # POST /predict
|  |     |- download.py         # GET /download_model, GET /models
|  |- core/
|  |  |- evaluator.py           # classification + regression metrics
|  |  |- diagnosis.py           # rule-based issue detection
|  |  |- suggestions.py         # suggestions + optional Gemini enhancement
|  |  |- health_score.py        # weighted health scoring
|  |  |- retrainer.py           # targeted retraining strategies
|  |  |- monitor.py             # jsonl logs
|  |- services/
|  |  |- model_service.py       # model save/load/registry/versioning
|  |  |- data_service.py        # CSV loading + preprocessing + split
|  |  |- mlflow_service.py      # MLflow logging helpers
|  |  |- dvc_service.py         # DVC CLI wrappers
|  |- data/
|  |- logs/
|  |- mlruns/
|  |- models/
|  |- requirements.txt
|  |- Dockerfile
|- frontend/
|  |- src/
|  |  |- App.jsx
|  |  |- api/client.js          # API base uses VITE_API_BASE_URL or /api
|  |  |- pages/                 # Dashboard, Results, Compare, Monitor
|  |  |- components/
|  |- package.json
|  |- Dockerfile
|- docker-compose.yml
|- README.md
```

---

## 4. Feature Coverage

### 4.1 Input modes

- Mode A: model_file + dataset_file
- Mode B: metrics_file + dataset_file

Target column can be explicitly provided; if omitted, backend auto-detects using common target-name heuristics, then fallback to last column.

### 4.2 Supported task types

- Classification
- Regression

Task type is inferred from target column and model capabilities where possible.

### 4.3 Diagnostics produced

Examples:
- Overfitting / Mild Overfitting
- Underfitting / Low Accuracy
- Class Imbalance / Severe Class Imbalance
- High Missing Values
- Low Variance Features
- Possible Data Leakage
- Feature Scaling Needed
- Regression-specific low fit and high error diagnostics

### 4.4 Retraining behavior

- Loads an existing version from model registry
- Tries targeted candidate pipelines/hyperparameters
- Uses deterministic seeds (with retrain-round variation)
- Avoids creating a new model version when there is no user-visible metric change

### 4.5 Tracking and logs

- MLflow run logging (metrics, params, model artifact)
- JSONL operational logs:
  - backend/logs/analyses.jsonl
  - backend/logs/retraining.jsonl
  - backend/logs/predictions.jsonl
  - backend/logs/errors.jsonl

---

## 5. Requirements

### 5.1 Local development

- Python 3.11+
- Node.js 20+ (recommended)
- npm
- Git

Optional:
- Docker Desktop
- DVC CLI

### 5.2 Python dependencies (backend)

Defined in backend/requirements.txt:
- FastAPI + Uvicorn
- scikit-learn, pandas, numpy
- MLflow
- imbalanced-learn
- python-dotenv
- python-multipart
- httpx

---

## 6. Environment Configuration

Create backend/.env from backend/.env.example.

Example keys:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.0-flash
MLFLOW_EXPERIMENT_NAME=ModelDoctor
FRONTEND_URL=http://localhost:5173
APP_PORT=8000
```

Notes:
- If GEMINI_API_KEY is missing or placeholder, app still works with rule-based suggestions/fallback summaries.
- In Docker Compose, backend CORS frontend URL is set to http://localhost:3000.

---

## 7. Run Locally (Without Docker)

### 7.1 Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 7.2 Frontend

```bash
cd frontend
npm install
npm run dev
```

### 7.3 Local URLs

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- OpenAPI docs: http://localhost:8000/docs

---

## 8. Run With Docker Compose (Recommended)

### 8.1 Build and start

```bash
docker compose up --build -d
```

### 8.2 Check status

```bash
docker compose ps
```

### 8.3 View logs

```bash
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f mlflow
```

### 8.4 Stop services

```bash
docker compose down
```

### 8.5 Docker URLs

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Backend docs: http://localhost:8000/docs
- MLflow UI: http://localhost:5000

---

## 9. Docker Desktop Usage (UI-first)

Use this daily flow:

1. Open Docker Desktop
2. Go to Containers
3. Start/Stop the compose app group (not individual image Run buttons)

Important:
- Start from compose containers to preserve project network, ports, and dependencies.
- Avoid launching ad-hoc containers from Images tab for normal project operation.

---

## 10. API Reference

Base URL:
- Local backend: http://localhost:8000
- In Docker frontend: proxied via /api

### 10.1 POST /analyze

Multipart form fields:
- dataset_file (required): CSV
- model_file (optional): .pkl/.joblib
- metrics_file (optional): .json
- target_column (optional): target label column

At least one of model_file or metrics_file is required.

Example:

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "dataset_file=@data.csv" \
  -F "model_file=@model.pkl" \
  -F "target_column=target"
```

### 10.2 POST /retrain

Multipart form fields:
- dataset_file (required)
- model_version (required, e.g. v1)
- target_column (optional)

Example:

```bash
curl -X POST "http://localhost:8000/retrain" \
  -F "dataset_file=@data.csv" \
  -F "model_version=v1"
```

### 10.3 POST /predict

JSON body:

```json
{
  "model_version": "v1",
  "data": [
    {"feature1": 1.2, "feature2": 0.4},
    {"feature1": 0.7, "feature2": 2.1}
  ]
}
```

model_version is optional; if missing, backend uses best model version.

### 10.4 GET /models

Returns model list from registry, each enriched with is_best against best version.

### 10.5 GET /download_model

Query params:
- version (optional)

If omitted, backend returns best model file.

### 10.6 Utility endpoints

- GET / basic status and endpoint list
- GET /health health check

---

## 11. Metrics and Health Scoring

### 11.1 Classification metrics

- Accuracy, Precision, Recall, F1
- Macro variants
- Confusion matrix (bounded by class count threshold)
- Error rate and misclassified count
- Per-class breakdown

### 11.2 Regression metrics

- R2
- MAE, MSE, RMSE
- Explained variance

### 11.3 Health score

Classification factors:
- Accuracy (30%)
- Generalization gap (25%)
- Data quality (20%)
- Class balance (15%)
- Stability/CV (10%)

Regression factors:
- Fit quality
- Generalization
- Data quality
- Error quality
- Stability

Health status labels:
- Excellent
- Good
- Needs Tuning
- Poor

---

## 12. Model Versioning and Storage

Models and metadata are persisted under backend volume mounts:

- Model binaries: backend/models/model_vX.pkl
- Registry: backend/models/_registry.json
- MLflow artifacts and runs: backend/mlruns

Version naming:
- v1, v2, v3, ...

Best model selection:
- Highest health_score in registry.

---

## 13. DVC Integration Notes

When analyzing, dataset file is saved to backend/data/raw and passed to DVC service.

Behavior:
- If DVC CLI is installed, backend attempts dvc add <dataset>.
- If not installed, flow continues without failing core analysis.

---

## 14. Frontend Pages and Workflow

Main routes:
- / Dashboard upload form
- /results analysis results and retrain action
- /compare model version comparison with charts + download
- /monitor trend and model history dashboard

Frontend API behavior:
- Uses VITE_API_BASE_URL if provided
- Falls back to /api

This supports both:
- local Vite development (proxy in vite.config.js)
- Docker Nginx reverse-proxy deployment

---

## 15. Troubleshooting

### 15.1 Docker Desktop I/O error during build

Symptoms:
- input/output error during image pull/build
- engine stops unexpectedly

Likely cause:
- insufficient disk space on Docker storage drive (often C drive)

Fix:

1. Free disk space (especially on current Docker storage drive)
2. Docker Desktop -> Settings -> Resources -> Advanced
3. Move Docker disk image location to a drive with space (for example D)
4. Apply & restart Docker Desktop
5. Retry:

```bash
docker compose up --build -d
```

### 15.2 Compose warning about version field

If you see:
- the attribute version is obsolete

It is a harmless Compose warning with modern Docker Compose v2.

### 15.3 Frontend cannot reach backend in Docker

Ensure frontend API base is /api (already configured in frontend/src/api/client.js) and compose services are running.

### 15.4 No models available for predict/download

Run at least one successful /analyze call first to create a model version.

---

## 16. Security and Operational Notes

- Do not commit real API keys in .env.
- Validate uploaded files in production hardening phase.
- Add authentication/authorization before public deployment.
- Consider object storage for model/artifact persistence in production.

---

## 17. Suggested Next Improvements

- Add unit and integration tests for routers/core modules
- Add .dockerignore files to reduce image build context size
- Add request size limits and file-type validation
- Add async background job queue for long retraining tasks
- Add CI checks (lint, tests, security scans)

---

## 18. Quick Command Reference

```bash
# Start stack
docker compose up --build -d

# Stop stack
docker compose down

# Check running services
docker compose ps

# Tail backend logs
docker compose logs -f backend
```

---

## 19. License

MIT
