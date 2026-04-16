# Requirements: NeuralRetail

## v1: MVP Scope

### Ingestion (INGEST)
- **INGEST-01**: Daily ingestion pipeline from POS, Ecommerce, ERP, and external signals.
- **INGEST-02**: PII hashing and data quality checks during ingestion.

### Feature Engineering & MLOps (FEAT/MLOPS)
- **FEAT-01**: RFM and lag features calculation with Feast materialization.
- **MLOPS-01**: MLflow experiment tracking and model registry setup.

### Predictive Modeling (MODEL)
- **MODEL-01**: Demand forecasting (1, 7, 30 day) using Prophet/LSTM ensemble.
- **MODEL-02**: Churn detection with SHAP explanations (AUC-ROC ≥ 0.90).
- **MODEL-03**: Customer segmentation via K-Means/GMM.
- **MODEL-04**: Price elasticity estimation via DoWhy/EconML.
- **MODEL-05**: EOQ and safety stock recommendations (Inventory).

### Serving (SERVE)
- **SERVE-01**: FastAPI serving layer with caching (Redis) and rate limiting.
- **SERVE-02**: RBAC and Vault integration for secure access.

### Dashboard (DASH)
- **DASH-01**: Streamlit interactive intelligence dashboard.
- **DASH-02**: Performance: Dashboard P95 latency < 1.5s.

### Monitoring & Quality (MON)
- **MON-01**: Data drift monitoring (Evidently).
- **MON-02**: Model quality gates in CI/CD pipeline.

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INGEST-01   | Phase 1 | Pending |
| INGEST-02   | Phase 1 | Pending |
| FEAT-01     | Phase 2 | Pending |
| MLOPS-01    | Phase 2 | Pending |
| MODEL-01    | Phase 3 | Pending |
| MODEL-02    | Phase 3 | Pending |
| MODEL-03    | Phase 3 | Pending |
| MODEL-04    | Phase 3 | Pending |
| MODEL-05    | Phase 3 | Pending |
| SERVE-01    | Phase 4 | Pending |
| SERVE-02    | Phase 4 | Pending |
| DASH-01     | Phase 4 | Pending |
| DASH-02     | Phase 4 | Pending |
| MON-01      | Phase 5 | Pending |
| MON-02      | Phase 5 | Pending |
