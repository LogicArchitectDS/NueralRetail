# AMX-DS-2026-04 NeuralRetail

## What This Is

NeuralRetail is a production-grade, enterprise AI sales intelligence platform for Amdox Technologies - Engineering Division. It is an end-to-end data platform covering multi-source ETL, feature stores, predictive modeling (demand, churn, segmentation, elasticity), inventory optimization, and an interactive intelligence dashboard with an underlying FastAPI serving layer.

## Core Value

Empower Amdox with actionable AI sales intelligence that reduces demand forecasting MAPE to ≤ 10%, lowers stockouts by 30-50%, and provides highly accurate churn detection (AUC-ROC ≥ 0.90) through a reliable, scalable ML platform.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Run daily ingestion pipeline from POS, Ecommerce, ERP, and external signals
- [ ] Calculate RFM and lag features with Feast materialization
- [ ] Predict 1, 7, 30 day demand using a Prophet/LSTM ensemble model
- [ ] Determine customer churn risk and produce SHAP explanations
- [ ] Segment customers dynamically via K-Means and GMM
- [ ] Estimate price elasticity via DoWhy/EconML
- [ ] Compute EOQ and safety stock recommendations
- [ ] Serve all predictions via FastAPI with caching, rate limiting, and RBAC
- [ ] Expose insights via Streamlit interactive dashboard
- [ ] Enforce data drift monitoring and model quality gates in CI/CD pipeline

### Out of Scope

- [Real-time chat] — High complexity, not core to intelligence value
- [Video posts] — Storage/bandwidth costs
- [OAuth login] — JWT/API Key sufficient for internal tool
- [Mobile app] — Web-first dashboard, mobile later

## Context

Amdox Technologies needs to reduce inventory holding costs and minimize churn through predictive analytics. The system requires modern MLOps (MLflow, Feast, Airflow, Evidently) backed by Delta Lake/Spark on AWS EKS infrastructure to handle large data throughput with high availability.

## Constraints

- **Tech stack**: Python 3.12, strict version pinning
- **Infrastructure**: AWS (EKS, RDS, ElastiCache, S3)
- **Security**: No hardcoded secrets, Vault integration, strict RBAC, PII hashing, minimal container privileges
- **Performance**: Dashboard P95 latency < 1.5s, Batch throughput 15M+ < 4m

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Monorepo structure | Enables unified CI/CD, simplifies IaC | — Pending |
| AWS EKS + RDS + Redis | Fits scalable, managed enterprise requirements | — Pending |
| Airflow + Spark | Standard big data processing and orchestration | — Pending |
| Pydantic v2 + FastAPI | Fast performance and robust validation schemas | — Pending |

---
*Last updated: 2026-04-13 after initialization*
