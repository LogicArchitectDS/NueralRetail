# NeuralRetail Internship Project Report

**Project Name:** NeuralRetail - AI-Powered Retail Analytics Platform  
**Repository:** `NueralRetail_Solo`  
**Prepared For:** Internship project evaluation based on the Amdox Data Science & Analytics brief  
**Date:** 2026-05-04

## 1. Executive Summary

NeuralRetail is an end-to-end retail analytics platform built to solve business problems across customer intelligence, demand forecasting, churn prediction, price optimization, inventory planning, and MLOps monitoring.

The project combines:

- a **FastAPI backend** for prediction, KPI, export, and monitoring endpoints
- a **Streamlit dashboard** for business users
- **offline model artifacts** for segmentation, churn, forecasting, and pricing
- **data quality validation**
- **Airflow-style DAGs** for pipeline orchestration
- **metrics evidence files** for evaluator review
- **deployment assets** for Docker, Render, Railway, and Streamlit Cloud

This report explains:

- what the project does
- how the system works
- the file structure of the project
- what each important file is responsible for
- what results have been achieved
- which evaluation metrics are met
- which areas are partially met or not met, and why

## 2. Project Objective

The objective of NeuralRetail was to build a practical retail intelligence platform that supports the following eight business and engineering capabilities:

1. Multi-source data ingestion and ETL
2. Advanced customer segmentation
3. Demand forecasting
4. Churn prediction and retention intelligence
5. Revenue and price intelligence
6. Inventory optimization and reorder logic
7. Interactive analytics dashboard
8. MLOps and automated retraining

The project was not meant to be just a model notebook. It was intended to behave like a mini production analytics system where data moves through pipelines, models produce predictions, APIs serve the results, dashboards visualize the outcomes, and monitoring artifacts provide evidence of performance and health.

## 3. High-Level System Architecture

NeuralRetail is organized into five functional layers.

### 3.1 Data Layer

This layer contains the raw source data, intermediate feature datasets, parquet outputs, and artifact files used by models and dashboards.

### 3.2 Model Layer

This layer contains training scripts and model logic for:

- customer segmentation
- churn prediction
- demand forecasting
- price elasticity and simulation
- inventory optimization

### 3.3 Serving Layer

The FastAPI application exposes routes for:

- health checks
- executive KPIs
- customer scoring
- demand forecast retrieval
- inventory calculations
- pricing simulation
- data quality monitoring
- drift monitoring
- retraining decisions
- CRM export
- acceptance-metric evidence

### 3.4 Dashboard Layer

The Streamlit frontend gives business users a multi-page interface to consume the model outputs and analytics.

### 3.5 Monitoring and MLOps Layer

This layer includes:

- DQ validation
- drift detection
- retrain triggers
- Airflow-style orchestration
- MLflow evidence summaries
- artifact-based metrics reports

## 4. End-to-End Workflow of the Project

The project works in the following sequence:

1. **Raw retail data is stored in the `data/landing` layer.**
   - The project primarily uses Olist e-commerce datasets and a sample POS dataset.

2. **Ingestion scripts move raw data into a Bronze layer.**
   - Spark-based ingestion reads the landing data and writes parquet outputs into Bronze directories.

3. **Transformation scripts convert Bronze data into Silver data.**
   - Sensitive fields are anonymized.
   - Transaction-level columns are standardized for downstream processing.

4. **Feature engineering scripts create Gold-style analytical features.**
   - RFM features are generated for customer analysis.
   - Demand features and lagged revenue features are created for forecasting.

5. **Model training scripts generate analytical models.**
   - K-Means, DBSCAN, and GMM for segmentation
   - XGBoost, LightGBM, and stacked ensembles for churn
   - Prophet and LSTM for forecasting
   - OLS and DoWhy-based methods for pricing
   - EOQ, safety stock, and reorder logic for inventory

6. **Artifacts are stored locally.**
   - `.pkl`, `.pth`, `.json`, `.parquet`, and `.html` files are used as runtime evidence and cached outputs.

7. **FastAPI serves the outputs.**
   - The backend loads saved models and artifacts on startup and exposes them through REST endpoints.

8. **Streamlit displays the results.**
   - Business pages consume API responses and render charts, tables, KPIs, heatmaps, and simulators.

9. **Monitoring and MLOps functions check health and drift.**
   - DQ score, PSI drift, and retrain status are surfaced through the API and dashboard.

10. **Metrics endpoints summarize evaluation evidence.**
   - `/metrics/all` allows evaluators to inspect acceptance-criteria values in a single call.

## 5. Detailed Project Structure

Below is the meaningful project structure. Auto-generated folders such as `.venv`, `__pycache__`, `.git`, `.pytest_cache`, and CRC sidecar files are excluded from explanation because they are not authored business logic.

```text
NueralRetail_Solo/
├── artifacts/
├── config/
├── dags/
├── data/
├── docker/
├── models/
├── scripts/
├── src/
├── tests/
├── CHANGES.md
├── DEPLOYMENT.md
├── Dockerfile
├── Dockerfile.api
├── Dockerfile.app
├── INTERNSHIP_PROJECT_REPORT.md
├── Makefile
├── PERSONAL_REFLECTION.md
├── Procfile
├── README.md
├── docker-compose.yml
├── pyproject.toml
├── railway.json
├── railway.toml
├── render.yaml
├── requirements-api.txt
└── requirements.txt
```

## 6. File-by-File Reference

This section explains what each important file in the project does.

## 6.1 Top-Level Documentation and Build Files

| File | Purpose |
|---|---|
| `/home/seshu/NueralRetail_Solo/README.md` | Main project overview, quick start steps, architecture summary, API list, metrics table, and submission metadata. |
| `/home/seshu/NueralRetail_Solo/DEPLOYMENT.md` | Deployment guide for local Docker, Render, Railway, and Streamlit Cloud. |
| `/home/seshu/NueralRetail_Solo/CHANGES.md` | Historical change log for major engineering updates. |
| `/home/seshu/NueralRetail_Solo/PERSONAL_REFLECTION.md` | Reflection document describing lessons learned, MLOps challenges, and future improvements. |
| `/home/seshu/NueralRetail_Solo/pyproject.toml` | Poetry project manifest containing the main dependency graph and build configuration. |
| `/home/seshu/NueralRetail_Solo/poetry.lock` | Locked dependency versions for reproducible local builds. |
| `/home/seshu/NueralRetail_Solo/requirements.txt` | Lightweight dependency file for Streamlit dashboard deployment. |
| `/home/seshu/NueralRetail_Solo/requirements-api.txt` | Heavier dependency file for FastAPI + ML serving deployments. |
| `/home/seshu/NueralRetail_Solo/Dockerfile` | Root Docker image for API deployment, especially Railway API deployment. |
| `/home/seshu/NueralRetail_Solo/Dockerfile.api` | Dedicated Docker image for FastAPI + ML model serving. |
| `/home/seshu/NueralRetail_Solo/Dockerfile.app` | Dedicated Docker image for the Streamlit dashboard. |
| `/home/seshu/NueralRetail_Solo/docker-compose.yml` | Local two-service stack definition for API and dashboard. |
| `/home/seshu/NueralRetail_Solo/render.yaml` | Render deployment blueprint for separate API and dashboard services. |
| `/home/seshu/NueralRetail_Solo/railway.json` | Railway deployment configuration for API deployment. |
| `/home/seshu/NueralRetail_Solo/railway.toml` | Additional Railway deployment configuration. |
| `/home/seshu/NueralRetail_Solo/Procfile` | Alternative process definition for PaaS platforms. |
| `/home/seshu/NueralRetail_Solo/Makefile` | Small helper commands to bring up infrastructure profiles with Docker Compose. |
| `/home/seshu/NueralRetail_Solo/.env.example` | Environment variable template. |
| `/home/seshu/NueralRetail_Solo/.gitignore` | Git exclusions including secrets, models, cache files, and local artifacts. |
| `/home/seshu/NueralRetail_Solo/.dockerignore` | Docker build-context exclusions. |

## 6.2 Configuration Files

| File | Purpose |
|---|---|
| `/home/seshu/NueralRetail_Solo/config/auth_config.yaml` | Main authentication configuration used by the Streamlit dashboard. Includes demo users and role labels. |
| `/home/seshu/NueralRetail_Solo/config/settings.py` | Central application settings model using `pydantic-settings` for database, Redis, MLflow, Feast, auth, and secret variables. |
| `/home/seshu/NueralRetail_Solo/.streamlit/secrets.example.toml` | Safe template showing how to provide `API_URL` for Streamlit Cloud. |

## 6.3 API Layer

| File | Purpose |
|---|---|
| `/home/seshu/NueralRetail_Solo/src/api/main.py` | Core FastAPI application. Defines request schemas, loads models at startup, and exposes all major runtime endpoints including KPIs, demand forecast, churn scoring, segmentation, inventory, pricing, monitoring, export, and metrics evidence routes. |

### Important API Capabilities Implemented in `main.py`

- `/health`
- `/kpis`
- `/executive/revenue-trend`
- `/executive/demand-forecast`
- `/predict/churn`
- `/predict/churn/stack`
- `/predict/segment`
- `/predict/segment/gmm`
- `/inventory/optimize`
- `/inventory/abc_xyz`
- `/pricing/causal`
- `/price/simulate`
- `/monitoring/drift`
- `/monitoring/dq`
- `/monitoring/retrain`
- `/export/crm/high_risk`
- `/metrics/churn`
- `/metrics/forecast`
- `/metrics/segmentation`
- `/metrics/all`

## 6.4 Dashboard Layer

| File | Purpose |
|---|---|
| `/home/seshu/NueralRetail_Solo/src/app/streamlit_app.py` | Main multi-page Streamlit dashboard. Handles login, navigation, live API calls, KPI display, demand charting, churn exploration, inventory calculations, price simulation, and MLOps monitoring. |
| `/home/seshu/NueralRetail_Solo/src/app/pdf_export.py` | Utility to export KPI-style reports to PDF using ReportLab. |

### Implemented Dashboard Pages

- Executive Overview
- Demand Intelligence
- Customer Intelligence Hub
- Inventory Health
- Price Simulator
- MLOps Monitor

### Extra Dashboard Helpers Implemented

- `render_churn_heatmap()`
- `render_customer_360()`
- `render_acceptance_metrics()`

These improve evaluator visibility by showing churn-risk patterns, customer-level interpretability, and a single place to view acceptance-criteria evidence.

## 6.5 Data Quality Layer

| File | Purpose |
|---|---|
| `/home/seshu/NueralRetail_Solo/src/data/ge_validation.py` | Real data quality validation routine. Computes row-count checks, null-rate checks, non-negative monetary checks, positive frequency checks, duplicate-row checks, and returns a DQ score. |
| `/home/seshu/NueralRetail_Solo/src/quality/data_validator.py` | Great Expectations-style validator focused on churn feature data. Used as a lower-level validation utility. |

## 6.6 Explainability and Causal Layer

| File | Purpose |
|---|---|
| `/home/seshu/NueralRetail_Solo/src/explainability/causal_engine.py` | DoWhy-based causal wrapper for estimating treatment effects such as purchase frequency on churn or price on demand. |
| `/home/seshu/NueralRetail_Solo/src/models/causal_pricing.py` | Lightweight import guard for DoWhy-based pricing logic. |

## 6.7 Ingestion and Feature Engineering Layer

| File | Purpose |
|---|---|
| `/home/seshu/NueralRetail_Solo/src/ingestion/ingest.py` | Spark-based Bronze ingestion script that reads Olist landing CSV tables and writes Bronze parquet outputs. |
| `/home/seshu/NueralRetail_Solo/src/features/transformation.py` | Silver transformation script that reads Bronze POS data, hashes PII, standardizes columns, and writes Silver transactions. |
| `/home/seshu/NueralRetail_Solo/src/features/definitions.py` | Spark feature builder for RFM features and demand lag features; also prepares outputs compatible with Feast. |
| `/home/seshu/NueralRetail_Solo/src/features/build_features.py` | Pandas-based demand feature engineering script that creates temporal regressors such as weekday, holiday, month-end, and promo flags. |
| `/home/seshu/NueralRetail_Solo/src/features/generate_churn_features.py` | Generates churn training features from Olist customers, orders, and order items data. Builds Frequency, Monetary, Recency, and `is_churned`. |
| `/home/seshu/NueralRetail_Solo/src/features/feature_repo/feature_store.yaml` | Feast feature store configuration for local registry and Redis online store. |
| `/home/seshu/NueralRetail_Solo/src/features/feature_repo/features.py` | Feast entity and feature view definitions for customer RFM and store demand features. |

## 6.8 Model Training and Analytical Logic

| File | Purpose |
|---|---|
| `/home/seshu/NueralRetail_Solo/src/models/inventory_engine.py` | Core inventory math engine implementing EOQ, safety stock, and reorder point logic. |
| `/home/seshu/NueralRetail_Solo/src/models/price_engine.py` | Price intelligence engine that computes elasticity and revenue simulation behavior. |
| `/home/seshu/NueralRetail_Solo/src/models/train_kmeans.py` | K-Means segmentation training script with hyperparameter search between 6 and 10 clusters and artifact persistence. |
| `/home/seshu/NueralRetail_Solo/src/models/train_segmentation_advanced.py` | Advanced segmentation script that adds DBSCAN and GMM, logs metrics to MLflow, and saves clustering artifacts. |
| `/home/seshu/NueralRetail_Solo/src/models/train_xgboost.py` | XGBoost churn training script with SMOTE, metrics logging, and saved model output. |
| `/home/seshu/NueralRetail_Solo/src/models/train_lightgbm.py` | LightGBM churn classifier with DART boosting, MLflow logging, and artifact persistence. |
| `/home/seshu/NueralRetail_Solo/src/models/train_churn_stacked.py` | Stacked churn ensemble script using XGBoost, HistGradientBoosting, and a Logistic Regression meta-learner. Also creates a SHAP explainer artifact. |
| `/home/seshu/NueralRetail_Solo/src/models/train_prophet.py` | Prophet-based time-series forecasting pipeline with optional cross-validation and MLflow artifact logging. |
| `/home/seshu/NueralRetail_Solo/src/models/train_lstm.py` | PyTorch Lightning LSTM training script for multivariate demand forecasting. |
| `/home/seshu/NueralRetail_Solo/src/models/train_ensemble.py` | Prophet + LSTM ensemble blending script that uses Optuna to search the best forecast weight. |
| `/home/seshu/NueralRetail_Solo/src/models/train_shap.py` | Generates and stores a SHAP explainer for churn model interpretation. |
| `/home/seshu/NueralRetail_Solo/src/models/METRICS.md` | Historical model-metrics document. Some values in this file are older than the latest JSON artifacts and should be treated as secondary documentation. |
| `/home/seshu/NueralRetail_Solo/src/MODEL_CARDS.md` | Human-readable model cards summarizing purpose, inputs, outputs, metrics, and limitations of the major models. |

## 6.9 Monitoring and MLOps Layer

| File | Purpose |
|---|---|
| `/home/seshu/NueralRetail_Solo/src/monitoring/drift_report.py` | Monitoring module placeholder. The current repo surfaces drift primarily through artifact-backed API logic and summary JSON files. |
| `/home/seshu/NueralRetail_Solo/src/monitoring/auto_retrain.py` | Retraining module placeholder. Current retrain reporting is surfaced through API logic and artifact-backed status handling. |
| `/home/seshu/NueralRetail_Solo/dags/neuralretail_pipeline.py` | Standalone-testable Airflow-style DAG with implemented tasks for ingest, DQ, feature engineering, drift check, retrain decision, and stable model logging. |
| `/home/seshu/NueralRetail_Solo/src/pipelines/dags/bronze_ingestion_dag.py` | Legacy Airflow DAG for Bronze ingestion. |
| `/home/seshu/NueralRetail_Solo/src/pipelines/dags/silver_feature_dag.py` | Legacy Airflow DAG for Silver transformation, feature engineering, and Feast apply. |
| `/home/seshu/NueralRetail_Solo/src/pipelines/dags/neural_retail_master_dag.py` | Earlier master pipeline DAG coordinating ingestion, feature generation, churn training, and monitoring tasks. |

## 6.10 Tests

| File | Purpose |
|---|---|
| `/home/seshu/NueralRetail_Solo/tests/test_api_smoke.py` | Main smoke test suite covering health, KPI, monitoring, churn, segmentation, forecast, export, and inventory routes. |
| `/home/seshu/NueralRetail_Solo/tests/debug_feast.py` | Helper/debug file for feature store work. |
| `/home/seshu/NueralRetail_Solo/tests/verify_feast.py` | Verification script for Feast integration. |
| `/home/seshu/NueralRetail_Solo/tests/verify_mlflow.py` | Verification script for MLflow integration. |
| `/home/seshu/NueralRetail_Solo/tests/verify_online_store.py` | Verification script for online feature-store behavior. |

## 6.11 Data Files

### Raw and Landing Data

| Path | Purpose |
|---|---|
| `/home/seshu/NueralRetail_Solo/data/landing/olist/olist_customers_dataset.csv` | Customer master data. |
| `/home/seshu/NueralRetail_Solo/data/landing/olist/olist_orders_dataset.csv` | Order header data with timestamps and status. |
| `/home/seshu/NueralRetail_Solo/data/landing/olist/olist_order_items_dataset.csv` | Line-item sales data used in churn and pricing feature creation. |
| `/home/seshu/NueralRetail_Solo/data/landing/olist/olist_order_payments_dataset.csv` | Payment records. |
| `/home/seshu/NueralRetail_Solo/data/landing/olist/olist_order_reviews_dataset.csv` | Review data. |
| `/home/seshu/NueralRetail_Solo/data/landing/olist/olist_geolocation_dataset.csv` | Geolocation data. |
| `/home/seshu/NueralRetail_Solo/data/landing/olist/olist_products_dataset.csv` | Product master data. |
| `/home/seshu/NueralRetail_Solo/data/landing/olist/olist_sellers_dataset.csv` | Seller master data. |
| `/home/seshu/NueralRetail_Solo/data/landing/olist/product_category_name_translation.csv` | Product category name mapping. |
| `/home/seshu/NueralRetail_Solo/data/landing/pos/sample_pos.csv` | Sample POS dataset used for pipeline exercises. |

### Intermediate and Feature Data

| Path | Purpose |
|---|---|
| `/home/seshu/NueralRetail_Solo/data/bronze/bronze_pos_sales/_SUCCESS` | Bronze-ingestion success marker. |
| `/home/seshu/NueralRetail_Solo/data/silver/transactions/part-00000-f82135c8-d9cd-45ab-9968-c40cd790c592-c000.snappy.parquet` | Silver transaction parquet output. |
| `/home/seshu/NueralRetail_Solo/data/silver/sales_features.parquet` | Engineered sales-demand feature file. |
| `/home/seshu/NueralRetail_Solo/data/features/churn_features.parquet` | Main churn training feature dataset. |
| `/home/seshu/NueralRetail_Solo/data/features/rfm_features.parquet` | Customer RFM features. |
| `/home/seshu/NueralRetail_Solo/data/features/demand_features.parquet` | Demand feature set for forecasting. |
| `/home/seshu/NueralRetail_Solo/data/features/segmented_customers.parquet` | Customer dataset enriched with cluster assignments. |
| `/home/seshu/NueralRetail_Solo/data/features/demand_single.parquet` | Alternate demand dataset used by training utilities. |
| `/home/seshu/NueralRetail_Solo/data/features/rfm_single.parquet` | Alternate RFM dataset used by utilities. |

## 6.12 Model Artifact Files

| Path | Purpose |
|---|---|
| `/home/seshu/NueralRetail_Solo/models/xgboost_churn.pkl` | Saved XGBoost churn model. |
| `/home/seshu/NueralRetail_Solo/models/stacked_churn_model.pkl` | Saved stacked churn model. |
| `/home/seshu/NueralRetail_Solo/models/stacked_shap_explainer.pkl` | SHAP explainer for stacked churn output. |
| `/home/seshu/NueralRetail_Solo/models/shap_explainer.pkl` | SHAP explainer for churn tree model. |
| `/home/seshu/NueralRetail_Solo/models/kmeans_model.pkl` | Saved K-Means segmentation model. |
| `/home/seshu/NueralRetail_Solo/models/kmeans_scaler.pkl` | Scaler used before K-Means clustering. |
| `/home/seshu/NueralRetail_Solo/models/lstm_demand.pth` | Trained LSTM state dictionary for demand forecasting. |
| `/home/seshu/NueralRetail_Solo/models/lstm_scaler.pkl` | Scaling object for LSTM demand inference. |

## 6.13 Evaluation and Runtime Artifact Files

| Path | Purpose |
|---|---|
| `/home/seshu/NueralRetail_Solo/artifacts/rfm_features.parquet` | Runtime RFM feature store used by the API and dashboard. |
| `/home/seshu/NueralRetail_Solo/artifacts/churn_model_metrics.json` | Churn acceptance-metric evidence. |
| `/home/seshu/NueralRetail_Solo/artifacts/segmentation_metrics.json` | Segmentation acceptance-metric evidence. |
| `/home/seshu/NueralRetail_Solo/artifacts/price_intelligence_metrics.json` | Pricing acceptance-metric evidence. |
| `/home/seshu/NueralRetail_Solo/artifacts/inventory_metrics.json` | Inventory acceptance-metric evidence. |
| `/home/seshu/NueralRetail_Solo/artifacts/demand_forecast_summary.json` | Forecast summary including actual values, forecast values, PI coverage, and MAPE limitation notes. |
| `/home/seshu/NueralRetail_Solo/artifacts/drift_report_summary.json` | PSI drift summary and retrain recommendation state. |
| `/home/seshu/NueralRetail_Solo/artifacts/dq_report.json` | Data quality validation summary. |
| `/home/seshu/NueralRetail_Solo/artifacts/retrain_log.json` | Retrain action log. |
| `/home/seshu/NueralRetail_Solo/artifacts/mlflow_registry_summary.json` | Summary of MLflow experiments, runs, and model registry evidence. |
| `/home/seshu/NueralRetail_Solo/artifacts/dag_ingest_log.json` | Standalone DAG ingest task log. |
| `/home/seshu/NueralRetail_Solo/artifacts/dag_dq_log.json` | Standalone DAG DQ task log. |
| `/home/seshu/NueralRetail_Solo/artifacts/dag_features_log.json` | Standalone DAG feature engineering task log. |
| `/home/seshu/NueralRetail_Solo/artifacts/drift_report.html` | Full HTML drift report. |
| `/home/seshu/NueralRetail_Solo/artifacts/causal_effect_summary.json` | Causal pricing output summary. |
| `/home/seshu/NueralRetail_Solo/artifacts/lgbm_churn_model.pkl` | Saved LightGBM churn model in artifacts area. |
| `/home/seshu/NueralRetail_Solo/artifacts/dbscan_model.pkl` | Saved DBSCAN clustering model. |
| `/home/seshu/NueralRetail_Solo/artifacts/dbscan_labels.pkl` | Saved DBSCAN label outputs. |

## 6.14 Docker and Supporting Ops Files

| File | Purpose |
|---|---|
| `/home/seshu/NueralRetail_Solo/docker/airflow/Dockerfile` | Airflow image with Java and Spark installed for local orchestration experiments. |
| `/home/seshu/NueralRetail_Solo/docker/marquez/marquez.yaml` | Marquez configuration for metadata/lineage exploration. |
| `/home/seshu/NueralRetail_Solo/docker/postgres/create-marquez-db.sql` | SQL bootstrap file for Marquez Postgres database. |

## 6.15 Sync Automation Scripts

| File | Purpose |
|---|---|
| `/home/seshu/NueralRetail_Solo/scripts/sync_to_windows.sh` | Syncs the WSL repo into a Windows mirror directory while excluding large and transient files. |
| `/home/seshu/NueralRetail_Solo/scripts/watch_sync_to_windows.py` | Watches the repo for changes and automatically triggers sync. |
| `/home/seshu/NueralRetail_Solo/scripts/start_sync_watcher.sh` | Starts the sync watcher inside WSL. |
| `/home/seshu/NueralRetail_Solo/scripts/start_sync_watcher_windows.cmd` | Starts the WSL sync watcher from Windows. |

## 7. Features Implemented and How They Work

## 7.1 F-01 Multi-Source Data Ingestion and ETL

### What was implemented

- Spark-based ingestion from landing data into Bronze parquet
- Silver transformation and schema standardization
- Feature generation for RFM and demand signals
- DQ validation with scored checks
- Airflow-style DAG tasks for ingest, validate, feature engineering, drift, and retrain decision

### How it works

The ingestion pipeline starts from Olist CSV files in `data/landing/olist`. The Bronze step reads raw CSVs and writes parquet outputs. The Silver step standardizes fields and anonymizes PII. Feature scripts then produce customer-level and demand-level analytical datasets. DQ checks run over the resulting feature data and generate pass/fail metrics.

### Why this is only partially met

Although the ETL flow and DQ gates are implemented, the project does not yet demonstrate a full enterprise-grade multi-source ingestion stack with complete lineage in DataHub/OpenLineage and a fully proven sub-4-minute SLA. The architecture is present, but the exact enterprise acceptance expectations are not fully evidenced.

## 7.2 F-02 Advanced Customer Segmentation

### What was implemented

- K-Means clustering
- DBSCAN clustering
- Gaussian Mixture Model clustering
- Segment personas
- Segmentation API support
- Dashboard persona display
- Stability and silhouette evidence files

### How it works

Customer RFM features are standardized and clustered. K-Means is used for interpretable group assignment, DBSCAN identifies dense structures and potential outliers, and GMM adds probabilistic segmentation. The dashboard shows business-friendly segment names and recommended actions.

### Status

This module is one of the strongest parts of the project and satisfies the required acceptance criteria.

## 7.3 F-03 Demand Forecasting Engine

### What was implemented

- Prophet training pipeline
- LSTM training pipeline
- Prophet + LSTM ensemble blending
- Dashboard forecast page
- Forecast summary endpoint
- Confidence interval / prediction interval evidence
- Metrics artifact for forecast evaluation

### How it works

Demand-style features are engineered from historical transaction data. Prophet models overall trend and seasonality, while LSTM learns sequential patterns from lagged and calendar features. The ensemble combines both forecasts and saves the result to an artifact-backed summary used by the API and dashboard.

### Why this is not met

The architecture exists, but the core acceptance metric `MAPE <= 10%` is not achieved. The recorded MAPE is `168.07%`. This happens because the current target data behaves like RFM monetary aggregates rather than proper SKU-level daily demand series. The module is implemented, but the specific required accuracy target is not met.

## 7.4 F-04 Churn Prediction and Retention

### What was implemented

- XGBoost churn model
- LightGBM churn model
- Stacked churn ensemble
- SHAP explainability
- API endpoints for churn scoring
- CRM export of high-risk customers
- Customer 360 helper view in dashboard

### How it works

The churn pipeline uses Frequency, Monetary, and related customer features to estimate probability of churn. SMOTE is used during training to address class imbalance. The stacked model combines multiple learners to improve classification performance. High-risk customers can be exported for retention campaigns.

### Status

This module meets the required functional and metric targets.

## 7.5 F-05 Revenue and Price Intelligence

### What was implemented

- Price elasticity engine
- Revenue simulation endpoint
- DoWhy-based causal framing
- Dashboard price simulator
- Price metrics evidence

### How it works

The pricing engine estimates elasticity from historical price-demand behavior and simulates the effect of changing price on demand and projected revenue. Causal interpretation fields and simulator latency measurements are captured in artifact metrics.

### Status

This module meets the required metric targets.

## 7.6 F-06 Inventory Optimization and Reorder

### What was implemented

- EOQ calculation
- Safety stock calculation
- Reorder point calculation
- ABC-XYZ classification
- Inventory API endpoints
- Inventory dashboard page
- Dead-stock and PO-response evidence metrics

### How it works

The inventory engine accepts demand, holding-cost, and lead-time inputs to compute EOQ, reorder point, and safety stock. Portfolio classification groups items by revenue importance and demand variability. Inventory metrics artifacts provide evidence for response time and dead-stock logic quality.

### Status

This module meets the required measurable targets.

## 7.7 F-07 Interactive Analytics Dashboard

### What was implemented

- Working Streamlit dashboard with six business pages
- Role-based demo login
- KPI cards and charts
- Demand visualization
- Churn heatmap and customer 360 helper
- Segment personas table
- Inventory calculator and ABC-XYZ display
- Price simulator
- MLOps monitoring view
- CSV and Excel exports
- Acceptance-metrics table

### How it works

The dashboard uses API-backed requests to fetch KPIs, forecast data, churn scores, drift metrics, DQ results, and inventory outputs. Pages are designed for different business use cases while keeping the user flow inside a single tool.

### Why this is only partially met

The dashboard is real and usable, but several advanced brief details are still simplified or unproven. Examples include richer demand explorer controls, deeper drill-down views, formal Lighthouse/mobile performance proof, and some more advanced interaction patterns described in the brief.

## 7.8 F-08 MLOps and Automated Retraining

### What was implemented

- Drift summary artifact
- DQ summary artifact
- Retrain action endpoint
- MLflow experiment evidence summary
- Airflow-style DAG
- Acceptance-metrics API exposure

### How it works

The system computes drift and DQ summaries from artifacts, exposes them through the API, and shows them in the dashboard. The DAG captures the intended orchestration path for ingest, validate, feature engineer, check drift, and decide whether retraining should occur.

### Why this is only partially met

The MLOps layer demonstrates monitoring and orchestration concepts well, but it does not fully prove strict production criteria such as zero-downtime model swap, fully automated champion/challenger promotion, and end-to-end retrain SLAs in a live environment.

## 8. Results Achieved

The project has already produced strong measurable results in several modules.

| Module | Metric | Achieved | Target | Status |
|---|---:|---:|---:|---|
| Data Quality | DQ Score | 100.0% | >= 98% | Met |
| Segmentation | Silhouette Score | 0.609 | >= 0.55 | Met |
| Segmentation | Stability (Week-on-Week) | 0.84 | >= 0.80 | Met |
| Churn | AUC-ROC | 1.0000 | >= 0.90 | Met |
| Churn | Precision@Top20% | 1.0000 | >= 0.78 | Met |
| Price | Elasticity R² | 0.9963 | >= 0.72 | Met |
| Price | Simulator Response | 145 ms | <= 2000 ms | Met |
| Inventory | Dead-Stock Accuracy | 0.87 | >= 0.85 | Met |
| Inventory | PO Draft Response | 280 ms | <= 30000 ms | Met |
| Forecast | PI Coverage | 95.86% | >= 88% | Met |
| Forecast | MAPE | 168.07% | <= 10% | Not met |
| Monitoring | PSI | 0.0006 | < 0.2 | Met |

## 9. Evaluation-Metric Accomplishment Summary

Based on the implemented features and the currently available evidence files, the project aligns with the evaluation criteria as follows:

| Requirement | Final Assessment | Reason |
|---|---|---|
| F-01 Multi-Source Data Ingestion & ETL | Partially met | ETL, DQ, and DAG logic exist, but enterprise lineage and full SLA proof are incomplete. |
| F-02 Advanced Customer Segmentation | Exactly met | All major segmentation models exist and the required metrics are achieved. |
| F-03 Demand Forecasting Engine | Not met | PI coverage is strong, but the required MAPE target is not achieved. |
| F-04 Churn Prediction & Retention | Exactly met | Churn scoring, explainability, export workflow, and required metrics are achieved. |
| F-05 Revenue & Price Intelligence | Exactly met | Simulator, elasticity, and attribution evidence satisfy the target metrics. |
| F-06 Inventory Optimization & Reorder | Exactly met | Inventory calculations and supporting metrics satisfy the target thresholds. |
| F-07 Interactive Analytics Dashboard | Partially met | Dashboard is implemented and functional, but some advanced UI/performance proof is still missing. |
| F-08 MLOps & Automated Retraining | Partially met | Monitoring and orchestration are present, but not all production-grade acceptance behaviors are fully proven. |

## 10. Testing and Validation Status

The project includes a smoke test suite for core API functionality.

### Verified Test Outcome

- `15 passed`
- `0 failed`

### What the tests cover

- health route
- KPI route
- drift route
- DQ route
- retrain route
- pricing causal route
- churn routes
- inventory route
- revenue trend route
- CRM export route
- demand forecast route
- segmentation route

This gives confidence that the main inference and monitoring routes are functional at runtime.

## 11. Known Issues and Limitations

### 11.1 Forecasting accuracy limitation

The biggest issue in the project is the forecasting MAPE. The pipeline is implemented, but the current evaluation data is not the right kind of data for a real retail demand-forecasting MAPE target. A proper SKU-level daily time-series dataset would be needed to genuinely target `MAPE <= 10%`.

### 11.2 Mixed maturity across files

Some files in the repo are fully implemented and production-oriented, while a few others are placeholders or scaffolding retained for future extension. Examples:

- `src/monitoring/drift_report.py` is currently a placeholder
- `src/monitoring/auto_retrain.py` is currently a placeholder
- some `src/dashboard/*` and `src/serving/*` package files are empty package stubs

These do not prevent the demo from working, but they show that the repo is still a hybrid of implemented production paths and future scaffolding.

### 11.3 Documentation drift in a few older files

Some older documentation files still contain earlier metric values, especially:

- `src/models/METRICS.md`
- parts of `README.md`

The newest JSON artifacts and `/metrics/*` endpoints should be treated as the current source of truth.

## 12. Final Conclusion

NeuralRetail is a strong internship project that demonstrates a complete analytics-system mindset rather than isolated model experimentation.

The project successfully delivers:

- a working backend
- a working dashboard
- real feature engineering scripts
- real model artifacts
- exportable business outputs
- evaluator-facing metrics evidence
- MLOps-oriented monitoring artifacts
- deployment-ready container and cloud config

The strongest completed areas are:

- customer segmentation
- churn prediction and retention
- price intelligence
- inventory optimization

The partially complete areas are:

- multi-source ETL at enterprise-proof level
- dashboard richness/performance proof
- production-grade MLOps completeness

The one major unmet area is:

- demand forecasting accuracy against the required MAPE threshold

Overall, this project can honestly be presented as a **substantially implemented, practically functional retail analytics platform** with **strong evidence in most modules** and **one major documented forecasting limitation**.

## 13. Suggested Use of This Report

This report can be used as:

- the basis for an internship submission report
- a README supplement
- a viva or interview explanation document
- a final PDF or Word report after formatting

If needed, this Markdown report can be converted next into:

- a polished `.docx`
- a presentation deck
- a PDF handout
- a shorter evaluator summary
