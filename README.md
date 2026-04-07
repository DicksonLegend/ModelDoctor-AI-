# 🩺 ModelDoctor AI+

> **Production-level MLOps workflow** for ML model diagnosis, retraining, comparison, and deployment.

[![CI/CD](https://github.com/your-repo/modeldoctor/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/your-repo/modeldoctor/actions)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Smart Diagnosis** | Detects overfitting, underfitting, class imbalance, missing data, scaling issues |
| 💡 **AI Suggestions** | Rule-based + optional AI-powered improvement recommendations |
| ⭐ **Health Score** | Comprehensive 0–100 model health assessment with 5-factor breakdown |
| 🔄 **Auto Retrain** | One-click retraining with automatic improvement application |
| 📊 **Version Compare** | Side-by-side model version comparison with charts |
| ⬇️ **Model Download** | Download the best performing model as `.pkl` |
| 📡 **Monitoring** | Track predictions, errors, and performance over time |
| 🧪 **MLflow Tracking** | Full experiment tracking with metrics, params, and model artifacts |
| 📦 **DVC Integration** | Dataset versioning with DVC |

## 🏗️ Architecture

```
User uploads (model + data OR metrics + data)
  → System evaluates model
  → Diagnoses issues
  → Gives suggestions (rule + AI)
  → Calculates health score
  → Applies improvements
  → Retrains model
  → Compares versions
  → Selects best model
  → Allows download
  → Logs everything (MLflow + monitoring)
```

## 📁 Project Structure

```
MLOPS_Project/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── main.py             # FastAPI entry point
│   │   ├── config.py           # Settings & env vars
│   │   ├── schemas.py          # Pydantic models
│   │   └── routers/            # API endpoints
│   │       ├── analyze.py      # POST /analyze
│   │       ├── retrain.py      # POST /retrain
│   │       ├── predict.py      # POST /predict
│   │       └── download.py     # GET  /download_model
│   ├── core/                   # Core ML logic
│   │   ├── evaluator.py        # Metrics computation
│   │   ├── diagnosis.py        # Problem detection
│   │   ├── suggestions.py      # Rule + AI suggestions
│   │   ├── health_score.py     # 0-100 health scoring
│   │   ├── retrainer.py        # Auto-retraining engine
│   │   └── monitor.py          # Logging & monitoring
│   ├── services/               # Service layer
│   │   ├── model_service.py    # Model save/load/registry
│   │   ├── data_service.py     # Dataset loading & preprocessing
│   │   ├── mlflow_service.py   # MLflow integration
│   │   └── dvc_service.py      # DVC integration
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                   # React (Vite) frontend
│   ├── src/
│   │   ├── App.jsx
│   │   ├── pages/              # Dashboard, Results, Compare, Monitor
│   │   ├── components/         # Reusable UI components
│   │   ├── api/client.js       # API client
│   │   └── styles/index.css    # Design system
│   ├── package.json
│   └── Dockerfile
│
├── docker-compose.yml          # Backend + Frontend + MLflow
├── .github/workflows/ci-cd.yml # CI/CD pipeline
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **Git**

### 1. Clone & Setup Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 2. Setup Frontend

```bash
cd frontend
npm install
npm run dev
```

### 3. Open the App

- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

### Docker (Alternative)

```bash
docker-compose up --build
```

- **Frontend:** http://localhost:3000
- **Backend:** http://localhost:8000
- **MLflow UI:** http://localhost:5000

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Upload model/metrics + dataset for full analysis |
| `POST` | `/retrain` | Retrain a model version with improvements |
| `POST` | `/predict` | Make predictions with a saved model |
| `GET`  | `/download_model` | Download a model as `.pkl` |
| `GET`  | `/models` | List all model versions |
| `GET`  | `/health` | API health check |
| `GET`  | `/docs` | Swagger API documentation |

## 📊 Input Modes

### Mode A: Model + Dataset
Upload a `.pkl` model file and `.csv` dataset. The system evaluates the model and computes all metrics.

### Mode B: Metrics + Dataset
Upload a `metrics.json` and `.csv` dataset. The system uses pre-computed metrics.

**metrics.json format:**
```json
{
  "accuracy": 0.85,
  "precision": 0.83,
  "recall": 0.87,
  "f1_score": 0.85,
  "train_accuracy": 0.95
}
```

## ⭐ Health Score

| Factor | Weight | Description |
|--------|--------|-------------|
| Accuracy | 30% | Model prediction accuracy |
| Generalization | 25% | Train vs test accuracy gap |
| Data Quality | 20% | Missing values, feature quality |
| Class Balance | 15% | Target class distribution |
| Stability | 10% | Cross-validation variance |

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React, Vite, Recharts, Framer Motion |
| Backend | FastAPI, Python |
| ML | Scikit-learn (Random Forest, Logistic Regression) |
| Tracking | MLflow |
| Data Versioning | DVC |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |

## 📄 License

MIT
