# Changes Summary - NeuralRetail Optimization Sprint

**Date:** 2026-04-27  
**Engineer:** Claude Code (Senior DevOps/ML Engineer)  
**Session:** Controlled Optimization & PaaS Preparation

---

## Executive Summary

Completed 5 optimization tasks to prepare the NeuralRetail platform for enterprise deployment:

1. ✅ Fixed critical API URL mismatch in Streamlit frontend
2. ✅ Enhanced Churn model with SMOTE class imbalance handling
3. ✅ Generated enterprise-grade README.md
4. ✅ Created PaaS deployment configurations (Render, Railway)
5. ✅ Added Docker health checks and volume persistence

---

## Detailed Changes

### 1. API URL Mismatch Fix

**File:** `src/app/streamlit_app.py` (line 9)

**Before:**
```python
API_URL = os.getenv("API_URL", "http://127.0.0.1:8001")
```

**After:**
```python
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
```

**Impact:** Streamlit dashboard now correctly connects to the FastAPI backend on port 8000.

---

### 2. Churn Model Class Imbalance Enhancement

**File:** `src/models/train_xgboost.py`

**Changes:**
- Added `imbalanced-learn` SMOTE import
- Implemented SMOTE oversampling (k=5) on training data
- Added `scale_pos_weight` calculation for XGBoost
- Increased `n_estimators` from 100 → 200
- Reduced `learning_rate` from 0.1 → 0.05
- Increased `max_depth` from 5 → 6
- Added regularization params: `min_child_weight`, `gamma`, `subsample`, `colsample_bytree`
- Changed `eval_metric` from `logloss` → `auc`

**Expected Impact:** AUC-ROC improvement from 0.5893 → ~0.70+ (pending retraining)

**To apply:** Run `python src/models/train_xgboost.py` to regenerate model artifact.

---

### 3. Enterprise README.md

**File:** `README.md` (complete rewrite)

**New Sections:**
- ASCII architecture diagram
- Model performance metrics table
- Quick start commands
- API endpoint documentation with example curl
- Dashboard page descriptions
- Project structure tree
- PaaS deployment instructions
- Team roster table

**Impact:** Professional documentation suitable for stakeholder review and onboarding.

---

### 4. PaaS Deployment Configurations

**New Files Created:**

| File | Purpose |
|------|---------|
| `railway.json` | Railway.app deployment config |
| `render.yaml` | Render.com blueprint (2 services) |
| `Procfile` | Heroku/Render start command |
| `requirements.txt` | PaaS-compatible dependencies |
| `.dockerignore` | Build context optimization |
| `DEPLOYMENT.md` | Step-by-step deployment guide |

**Key Features:**
- Environment variable injection for API_URL
- Health check paths configured
- Auto-restart policies
- Port binding via $PORT env var

---

### 5. Docker Compose Enhancements

**File:** `docker-compose.yml`

**Changes:**
```yaml
# API Service
volumes:
  - ./models:/app/models          # Persist models
environment:
  - LOG_LEVEL=INFO
  - MODEL_PATH=/app/models
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
restart: unless-stopped

# App Service
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
depends_on:
  api:
    condition: service_healthy
restart: unless-stopped
```

**Impact:**
- Models persist across container restarts
- Automatic service restart on failure
- Health-based dependency ordering
- Production-ready restart policies

---

### 6. .gitignore Enhancement

**File:** `.gitignore`

**Added:**
```
# Model Artifacts
models/*.pkl
models/*.pth
models/*.h5
models/*.pt
!models/.gitkeep

# OS
.DS_Store
Thumbs.db
desktop.ini
```

**Impact:** Prevents large binary model files from being committed to git.

---

## Files Modified

| File | Change Type | Description |
|------|-------------|-------------|
| `src/app/streamlit_app.py` | Modified | Fixed API URL port |
| `src/models/train_xgboost.py` | Modified | SMOTE + class weights |
| `README.md` | Rewritten | Enterprise documentation |
| `docker-compose.yml` | Modified | Health checks, volumes |
| `.gitignore` | Modified | Model artifact exclusions |
| `requirements.txt` | Created | PaaS dependencies |
| `railway.json` | Created | Railway config |
| `render.yaml` | Created | Render blueprint |
| `Procfile` | Created | Heroku/Render start |
| `.dockerignore` | Created | Docker build optimization |
| `DEPLOYMENT.md` | Created | Deployment guide |
| `CHANGES.md` | Created | This file |

---

## Next Steps

### Immediate (Required)

1. **Retrain Churn Model:**
   ```bash
   poetry run python src/models/train_xgboost.py
   ```

2. **Test Locally:**
   ```bash
   docker-compose up -d --build
   # Visit http://localhost:8501
   ```

3. **Verify All Endpoints:**
   - Dashboard loads
   - Churn prediction works
   - Segmentation returns cluster
   - Inventory optimizer calculates
   - Price simulator displays elasticity

### PaaS Deployment (Optional)

1. Push to GitHub:
   ```bash
   git add .
   git commit -m "Prepare for PaaS deployment"
   git push origin main
   ```

2. Connect to Render or Railway (see DEPLOYMENT.md)

3. Set environment variables in PaaS dashboard

---

## Model Metrics Summary

| Engine | Previous | Target | Status |
|--------|----------|--------|--------|
| K-Means (F-02) | Silhouette = 0.609 | — | ✅ Unchanged |
| LSTM (F-03) | MAPE = 113% | — | ⚠️ Data sparsity limit |
| XGBoost (F-04) | AUC-ROC = 0.5893 | ~0.70+ | 🔁 Needs retrain |
| OLS (F-05) | R² = 0.9963 | — | ✅ Unchanged |
| EOQ (F-06) | Deterministic | — | ✅ Unchanged |

---

## Risk Assessment

| Change | Risk Level | Mitigation |
|--------|------------|------------|
| API URL fix | Low | Verified correct port |
| XGBoost retrain | Medium | Test set evaluation unchanged |
| Docker volumes | Low | Standard practice |
| PaaS configs | Low | Non-destructive additions |

---

**End of Changes Summary**
