# Roadmap: NeuralRetail

## Phases

- [ ] **Phase 1: Foundation & Ingestion** - Establish data infrastructure and automated ingestion with quality gates.
- [ ] **Phase 2: Feature Engineering & MLOps** - Centralized feature store and model lifecycle management.
- [ ] **Phase 3: Predictive Modeling** - Core intelligence engines for demand, churn, and inventory.
- [ ] **Phase 4: Serving & Dashboard** - Secure API and interactive intelligence dashboard.
- [ ] **Phase 5: Monitoring & Quality** - Drift detection and automated quality gates.

## Phase Details

### Phase 1: Foundation & Ingestion
**Goal**: Establish data infrastructure and automated ingestion from multiple sources with quality gates.
**Depends on**: Nothing
**Requirements**: INGEST-01, INGEST-02
**Success Criteria** (what must be TRUE):
  1. Multi-source data (POS, ERP, etc.) is successfully loaded into Delta Lake/S3.
  2. PII is automatically hashed and data quality checks block invalid records.
  3. Daily ingestion pipeline runs automatically via Airflow.
**Plans**: TBD

### Phase 2: Feature Engineering & MLOps
**Goal**: Implement a centralized feature store and model lifecycle management.
**Depends on**: Phase 1
**Requirements**: FEAT-01, MLOPS-01
**Success Criteria** (what must be TRUE):
  1. RFM and lag features are computed and materialized in Feast online/offline stores.
  2. Training experiments and model artifacts are tracked and versioned in MLflow.
  3. Features can be retrieved with low latency for online serving.
**Plans**: TBD

### Phase 3: Predictive Modeling
**Goal**: Build and validate the core intelligence engines for demand, churn, and inventory.
**Depends on**: Phase 2
**Requirements**: MODEL-01, MODEL-02, MODEL-03, MODEL-04, MODEL-05
**Success Criteria** (what must be TRUE):
  1. Demand forecasting model achieves MAPE ≤ 10% on test data.
  2. Churn detection model achieves AUC-ROC ≥ 0.90 with SHAP explanations.
  3. Segmentation, elasticity, and EOQ recommendations are generated and stored.
**Plans**: TBD

### Phase 4: Serving & Dashboard
**Goal**: Provide secure access to predictions through a unified API and interactive dashboard.
**Depends on**: Phase 3
**Requirements**: SERVE-01, SERVE-02, DASH-01, DASH-02
**Success Criteria** (what must be TRUE):
  1. FastAPI endpoints serve all model predictions with Redis caching enabled.
  2. Access is restricted via RBAC and Vault-managed secrets.
  3. Streamlit dashboard displays insights with P95 latency < 1.5s.
**Plans**: TBD
**UI hint**: yes

### Phase 5: Monitoring & Quality
**Goal**: Ensure long-term reliability through drift detection and CI/CD quality gates.
**Depends on**: Phase 4
**Requirements**: MON-01, MON-02
**Success Criteria** (what must be TRUE):
  1. Automated alerts trigger when data or model drift exceeds thresholds (Evidently).
  2. CI/CD pipeline fails if new models do not meet performance/quality gates.
**Plans**: TBD

## Progress Table

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation & Ingestion | 0/1 | Not started | - |
| 2. Feature Engineering & MLOps | 0/1 | Not started | - |
| 3. Predictive Modeling | 0/1 | Not started | - |
| 4. Serving & Dashboard | 0/1 | Not started | - |
| 5. Monitoring & Quality | 0/1 | Not started | - |
