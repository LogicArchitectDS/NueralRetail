# NeuralRetail Deployment Guide

## Quick Reference

| Environment | Command / Setup | Access |
|-------------|------------------|--------|
| **Local** | `docker compose up -d --build` | `http://localhost:8501` |
| **Render** | Use `render.yaml` | Public HTTPS app + API |
| **Railway** | Use root `Dockerfile` for API | Public HTTPS API |
| **Streamlit Cloud** | Use `requirements.txt` | Public HTTPS dashboard |

---

## 1. Local Development

### Prerequisites

```bash
docker --version
docker compose version
python3 --version
```

### Start Services

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f api
docker compose logs -f app
```

### Stop Services

```bash
docker compose down
```

### Verify Health

```bash
curl http://localhost:8000/health
curl http://localhost:8501/_stcore/health
```

---

## 2. Dependency Split

The repo intentionally uses two dependency files:

- `requirements-api.txt`: FastAPI + ML serving dependencies
- `requirements.txt`: lightweight Streamlit dashboard dependencies

Container files already match this split:

- `Dockerfile` and `Dockerfile.api` use `requirements-api.txt`
- `Dockerfile.app` uses `requirements.txt`

---

## 3. Render Deployment

Render is configured through `render.yaml` with two Docker services:

- `neuralretail-api` using `Dockerfile.api`
- `neuralretail-app` using `Dockerfile.app`

### Deploy Steps

1. Push the latest `main` branch to GitHub.
2. In Render, create a new Blueprint and point it at this repository.
3. Confirm both services are detected from `render.yaml`.
4. If you rename the API service, update the app service `API_URL` env var to match the final API hostname.

### Expected Endpoints

```bash
https://<api-service>.onrender.com/health
https://<app-service>.onrender.com
```

---

## 4. Railway Deployment

Railway in this repo is configured for the **API only**.

- `railway.json` and `railway.toml` point Railway at the root `Dockerfile`
- The root `Dockerfile` builds the FastAPI + ML API with `requirements-api.txt`

### Deploy API on Railway

1. Create a new Railway project from GitHub.
2. Let Railway build from the checked-in Docker configuration.
3. Expose the generated public API URL.

### Deploy Dashboard Separately

For the Streamlit dashboard, use **Streamlit Cloud** with:

- App entrypoint: `src/app/streamlit_app.py`
- Dependency file: `requirements.txt`
- Secret:

```toml
API_URL = "https://<your-railway-api>.up.railway.app"
```

---

## 5. Streamlit Cloud

### Required Settings

- Repository: `LogicArchitectDS/NueralRetail_Solo`
- Branch: `main`
- Main file path: `src/app/streamlit_app.py`

### Secrets

Use `.streamlit/secrets.example.toml` as the template.

---

## 6. Troubleshooting

### API Not Reachable

```bash
docker compose ps
docker compose logs api
curl http://localhost:8000/health
```

### Streamlit Not Updating

```bash
docker compose restart app
```

### Rebuild from a Clean Clone

```bash
docker compose down
docker compose up -d --build
```

---

## 7. Final Submission Checklist

- [ ] `README.md` updated with final public app URL
- [ ] `README.md` updated with final walkthrough video URL
- [ ] API health endpoint reachable publicly
- [ ] Streamlit dashboard reachable publicly over HTTPS
- [ ] All dashboard pages load without runtime errors
- [ ] Smoke test suite passes
