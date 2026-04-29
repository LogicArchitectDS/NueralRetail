# NeuralRetail Deployment Guide

## Quick Reference

| Environment | Command | Access |
|-------------|---------|--------|
| **Local** | `docker-compose up -d` | localhost:8501 |
| **Render** | Push to main branch | Auto-deploys |
| **Railway** | Connect GitHub repo | Auto-deploys |

---

## 1. Local Development

### Prerequisites

```bash
# Verify Docker is running
docker --version
docker-compose --version

# Verify Python
python --version  # Must be 3.12+
```

### Start Services

```bash
# Build and start both API and Streamlit
docker-compose up -d --build

# Check service health
docker-compose ps

# View logs
docker-compose logs -f api
docker-compose logs -f app

# Stop services
docker-compose down
```

### Verify Health

```bash
# API health check
curl http://localhost:8000/health

# Streamlit health check
curl http://localhost:8501/_stcore/health
```

---

## 2. Render Deployment

### Step 1: Prepare Repository

```bash
# Ensure requirements.txt exists
cat requirements.txt

# Commit all changes
git add .
git commit -m "Prepare for PaaS deployment"
git push origin main
```

### Step 2: Configure Render

1. Go to [render.com](https://render.com) and sign in
2. Click **New +** → **Web Service**
3. Connect your GitHub repository
4. Configure the service:

**API Service:**
| Field | Value |
|-------|-------|
| Name | `neuralretail-api` |
| Environment | `Python` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT` |
| Port | `8000` |

**Environment Variables:**
```
PYTHON_VERSION=3.12.0
PORT=8000
LOG_LEVEL=INFO
MODEL_PATH=./models
```

### Step 3: Deploy Streamlit Frontend

1. Create another **Web Service**
2. Configure:

| Field | Value |
|-------|-------|
| Name | `neuralretail-app` |
| Environment | `Python` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `streamlit run src/app/streamlit_app.py --server.port $PORT --server.address 0.0.0.0` |
| Port | `8501` |

**Environment Variables:**
```
PYTHON_VERSION=3.12.0
PORT=8501
API_URL=https://neuralretail-api.onrender.com
```

---

## 3. Railway Deployment

### Step 1: Connect Repository

1. Go to [railway.app](https://railway.app)
2. Click **New Project** → **Deploy from GitHub**
3. Select your repository

### Step 2: Configure Service

Railway auto-detects `pyproject.toml` and `railway.json`.

**Environment Variables:**
```
PYTHON_VERSION=3.12.0
PORT=8000
API_URL=<will be auto-generated>
```

### Step 3: Deploy Frontend

1. Click **New** → **Empty Service**
2. Set environment variables:
```
PYTHON_VERSION=3.12.0
PORT=8501
API_URL=<your-api-url>.railway.app
```
3. Add start command:
```bash
streamlit run src/app/streamlit_app.py --server.port $PORT --server.address 0.0.0.0
```

---

## 4. Environment Variables Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `API_URL` | Backend API endpoint | `http://127.0.0.1:8000` | ✅ |
| `PORT` | Service port (PaaS) | `8000` / `8501` | ✅ |
| `LOG_LEVEL` | Logging verbosity | `INFO` | ❌ |
| `MODEL_PATH` | Path to model artifacts | `./models` | ❌ |
| `PYTHON_VERSION` | Runtime version | `3.12.0` | ✅ (PaaS) |

---

## 5. Troubleshooting

### API Connection Failed

```bash
# Check if API is running
docker-compose ps api

# Test health endpoint
curl http://localhost:8000/health

# Check API logs
docker-compose logs api
```

### Models Not Loading

```bash
# Verify models exist in container
docker-compose exec api ls -la /app/models/

# If empty, rebuild with volume mount
docker-compose down
docker-compose up -d --build
```

### Streamlit Shows Stale Data

```bash
# Force refresh by restarting
docker-compose restart app

# Or clear browser cache (Ctrl+Shift+R)
```

---

## 6. Post-Deployment Checklist

- [ ] API health endpoint returns `{"status": "ok"}`
- [ ] Streamlit dashboard loads without errors
- [ ] All 5 navigation pages are accessible
- [ ] Churn prediction returns valid JSON
- [ ] Segmentation returns cluster ID
- [ ] Inventory optimizer calculates EOQ
- [ ] Price simulator shows elasticity
- [ ] MLOps Monitor displays model table

---

## 7. Cost Estimates

| Platform | Free Tier | Starter Plan |
|----------|-----------|--------------|
| **Render** | 750 hrs/month | $7/month |
| **Railway** | $5 credit/month | $5/month |

> **Note:** Free tiers may have sleep mode. Upgrade for production use.
