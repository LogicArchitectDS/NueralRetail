# NeuralRetail - Enterprise Execution Roadmap

**Status:** Infrastructure Initialized. Entering Data Engineering Phase.

- [x] **Phase 0: Repository Foundation** - Monorepo setup, Poetry dependency locking, Ruff/Black pre-commit hooks. *(Completed by Sesha Sai)*
- [x] **Phase 1: Local Infrastructure** - Profile-driven Docker architecture (Airflow, Marquez, Postgres, Redis, isolated MLflow on port 5002) operating under a strict 6GB WSL constraint. *(Completed by Sesha Sai)*
- [ ] **Phase 2 & 3: Data Ingestion & Feast Feature Store** - PySpark transformations, Great Expectations data quality gates, and centralized feature serving. *(Assignee: Vaidehi)*
- [ ] **Phase 4a: Demand Forecasting Modeling** - PyTorch/Prophet time-series intelligence engines. *(Assignee: Karthikeyan)*
- [ ] **Phase 4b: Classification & Churn Modeling** - XGBoost/LightGBM with SHAP segmentation. *(Assignee: Ayush)*
- [ ] **Phase 5: Serving Layer** - Secure FastAPI endpoints with Redis caching for low-latency inference. *(Assignee: Pawan)*
- [ ] **Phase 6: Intelligence Dashboard** - Streamlit interactive UI for business stakeholders. *(Assignee: Nithish)*
- [ ] **Phases 7-12: Production MLOps** - CI/CD pipelines, Kubernetes deployment, model drift monitoring, and automated retraining. *(Assignee: Sesha Sai)*