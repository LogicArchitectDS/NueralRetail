# NeuralRetail Enterprise Intelligence Platform

> **A next-generation retail analytics platform combining MLOps best practices with production-grade machine learning models for customer segmentation, demand forecasting, churn prediction, price elasticity, and inventory optimization.**

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Model Performance Metrics](#model-performance-metrics)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Dashboard Pages](#dashboard-pages)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Team](#team)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NEURALRETAIL ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐         ┌──────────────────┐                         │
│  │   Streamlit UI   │────────▶│   FastAPI Gateway│                         │
│  │   (Port 8501)    │         │   (Port 8000)    │                         │
│  │                  │         │                  │                         │
│  │  • Segmentation  │         │  /predict/churn  │                         │
│  │  • Churn SHAP    │         │  /predict/segment│                         │
│  │  • EOQ Calculator│         │  /inventory/...  │                         │
│  │  • Price Sim     │         │  /price/simulate │                         │
│  └──────────────────┘         └─────────┬────────┘                         │
│                                          │                                  │
│                                          ▼                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        ML MODEL LAYER                                │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────────┐  │   │
│  │  │   K-Means   │ │  XGBoost    │ │  LSTM +     │ │  Log-Log OLS  │  │   │
│  │  │  (F-02)     │ │  (F-04)     │ │  Prophet    │ │  (F-05)       │  │   │
│  │  │  Silhouette │ │  AUC-ROC    │ │  MAPE       │ │  R² = 0.996   │  │   │
│  │  │  = 0.609    │ │  = 0.5893   │ │  = 113%     │ │               │  │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └───────────────┘  │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │           Inventory Optimization Engine (F-06)                │  │   │
│  │  │           EOQ │ Safety Stock │ Reorder Point                  │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        DATA LAYER                                    │   │
│  │  Bronze (Raw) │ Silver (Features) │ Pickle Artifacts │ MLflow       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Backend** | FastAPI + Uvicorn | High-performance REST API |
| **Frontend** | Streamlit | Interactive multi-page dashboard |
| **ML Models** | XGBoost, K-Means, LSTM, Prophet | Predictive analytics engines |
| **Data Processing** | Polars, Pandas, PySpark | ETL and feature engineering |
| **Deep Learning** | PyTorch, PyTorch Lightning | Neural network training |
| **Experiment Tracking** | MLflow | Model registry and metrics logging |
| **Containerization** | Docker Compose | Multi-stage builds, isolated services |
| **Dependency Mgmt** | Poetry | Locked dependency resolution |

---

## Model Performance Metrics

### Completed Analytical Engines (Amdox Requirements)

| Feature ID | Engine | Model | Primary Metric | Status |
|------------|--------|-------|----------------|--------|
| **F-02** | Customer Segmentation | K-Means (k=10) | Silhouette Score = **0.609** | ✅ Production |
| **F-03** | Demand Forecasting | LSTM + Prophet Ensemble | MAPE = **113%** | ⚠️ Champion/Challenger |
| **F-04** | Churn Prediction | XGBoost + SMOTE | AUC-ROC = **0.5893** | ⚠️ Class Imbalance |
| **F-05** | Price Intelligence | Log-Log OLS Regression | R² = **0.9963**, Elasticity = **-1.73** | ✅ Production |
| **F-06** | Inventory Optimization | Deterministic OR Engine | EOQ, Safety Stock, ROP | ✅ Production |

> **Note on F-03 (Forecasting):** The elevated MAPE is expected due to sparsity in the 100k-row Olist dataset. The architecture supports Champion/Challenger comparison—users can evaluate both LSTM and Prophet predictions independently.

> **Note on F-04 (Churn):** Heavy class imbalance in the source data limits AUC-ROC. SMOTE oversampling and `scale_pos_weight` have been applied to improve recall on the minority class.

---

## Quick Start

### Prerequisites

```bash
# Python 3.12+ required
python --version

# Install Poetry for dependency management
pip install poetry

# Docker Desktop with WSL2 (Windows users: set 6GB RAM limit in .wslconfig)
docker --version
```

### Installation

```bash
# 1. Clone the repository
git clone <repository_url>
cd NeuralRetail

# 2. Install Python dependencies
poetry install

# 3. Copy environment configuration
cp .env.example .env
# Edit .env with your local credentials
```

### Launch with Docker

```bash
# Start both API and Streamlit services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### Access the Dashboard

| Service | URL | Description |
|---------|-----|-------------|
| **Streamlit Dashboard** | http://localhost:8501 | Main UI |
| **FastAPI Docs** | http://localhost:8000/docs | Swagger API explorer |
| **Health Check** | http://localhost:8000/health | Service status |

---

## API Endpoints

### Inference Endpoints

| Method | Endpoint | Description | Request Schema |
|--------|----------|-------------|----------------|
| `GET` | `/health` | Service health check | - |
| `POST` | `/predict/churn` | Churn prediction + SHAP values | `{Frequency, Monetary}` |
| `POST` | `/predict/segment` | Customer segmentation | `{Frequency, Monetary}` |
| `POST` | `/inventory/optimize` | EOQ, Safety Stock, ROP | `{annual_demand, order_cost, holding_cost, ...}` |
| `POST` | `/price/simulate` | Elasticity + revenue projection | `{historical_prices, historical_demands, current_price, ...}` |

### Example: Churn Prediction

```bash
curl -X POST http://localhost:8000/predict/churn \
  -H "Content-Type: application/json" \
  -d '{"Frequency": 10, "Monetary": 100.0}'
```

**Response:**
```json
{
  "churn_prediction": 0,
  "churn_probability": 0.23,
  "shap_values": {
    "Frequency": -0.15,
    "Monetary": -0.08
  }
}
```

---

## Dashboard Pages

The Streamlit application features **5 distinct pages** accessible via the sidebar:

1. **Executive Overview** — 4 KPI cards + 7-day revenue trend chart
2. **Customer Intelligence Hub** — CHURN + SEGMENTATION analysis with SHAP waterfall charts
3. **Inventory Health** — EOQ calculator with safety stock and reorder point outputs
4. **Price Simulator** — Elasticity coefficient display + "What-If" revenue simulator
5. **MLOps Monitor** — Model registry table with MLflow metrics

---

## Project Structure

```
NeuralRetail/
├── docker-compose.yml          # Multi-service orchestration
├── Dockerfile.api              # API container (multi-stage)
├── Dockerfile.app              # Streamlit container
├── pyproject.toml              # Poetry dependency manifest
├── .env                        # Environment variables (gitignored)
├── .env.example                # Template for environment setup
│
├── src/
│   ├── api/
│   │   └── main.py             # FastAPI application
│   ├── app/
│   │   └── streamlit_app.py    # Streamlit dashboard
│   ├── models/
│   │   ├── train_xgboost.py    # Churn model trainer
│   │   ├── train_kmeans.py     # Segmentation trainer
│   │   ├── train_lstm.py       # Demand forecaster
│   │   ├── train_shap.py       # SHAP explainer generator
│   │   ├── inventory_engine.py # EOQ math engine
│   │   └── price_engine.py     # Elasticity engine
│   ├── features/
│   │   ├── transformation.py   # Silver layer transforms
│   │   └── definitions.py      # Feature calculations
│   └── ingestion/
│       └── ingest.py           # Bronze layer ingestion
│
├── data/
│   ├── bronze/                 # Raw ingested data (Parquet)
│   ├── silver/                 # Transformed features
│   └── features/               # Model-ready datasets
│
└── models/                     # Trained model artifacts (.pkl, .pth)
    ├── xgboost_churn.pkl
    ├── kmeans_model.pkl
    ├── kmeans_scaler.pkl
    ├── lstm_scaler.pkl
    └── shap_explainer.pkl
```

---

## Deployment

### Local Development

```bash
# Full stack with hot reload
docker-compose up --build
```

### PaaS Deployment (Render / Railway)

1. **Push to GitHub** — Ensure your repository is on GitHub
2. **Connect to Render/Railway** — Link your repository
3. **Set Environment Variables:**

| Variable | Description | Example |
|----------|-------------|---------|
| `API_URL` | Backend API endpoint | `https://your-api.onrender.com` |
| `PYTHON_VERSION` | Python runtime | `3.12.0` |
| `PIP_REQUIRE_VIRTUALENV` | Disable for PaaS | `false` |

4. **Deploy Commands:**

**Render:**
```bash
# Web Service: API
pip install -r requirements.txt && uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
```

**Railway:**
```bash
# Automatically detects pyproject.toml
# Set PORT environment variable
```

### Environment Variable Migration

All hardcoded `localhost` references have been replaced with environment variables:

```python
# Before
API_URL = "http://127.0.0.1:8000"

# After
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
```

---

## Team

| Name | Role | Focus Area |
|------|------|------------|
| Sesha Sai | DevOps & MLOps | Infrastructure, CI/CD, Monitoring |
| Vaidehi | Data Engineering | Feast Feature Store, PySpark |
| Karthikeyan | ML Engineering | Demand Forecasting (LSTM/Prophet) |
| Ayush | ML Engineering | Churn Prediction (XGBoost/SHAP) |
| Pawan | Backend Engineering | FastAPI Serving Layer |
| Nithish | Frontend Engineering | Streamlit Dashboard |

---

## License

Proprietary — Amdox NeuralRetail Project

---

**Built with** ❤️ **by the NeuralRetail Team**
