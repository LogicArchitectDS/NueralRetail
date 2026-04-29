# Phase 2: Feature Engineering & MLOps

## Objectives
- [x] Establish Silver Layer (Joined and Cleaned Data)
- [x] Implement RFM (Recency, Frequency, Monetary) Feature Engineering
- [x] Implement Time-based Lag Features for Demand Forecasting
- [x] Initialize Feast Feature Store
- [x] Integrate MLflow for experiment tracking
- [x] Create Feature Engineering Pipeline (Airflow DAG)

## Strategy
1. **Silver Transformation**: Use PySpark to join `olist_orders`, `olist_order_items`, and `olist_customers` to create a unified transaction view.
2. **RFM Features**: Calculate RFM metrics at the customer level.
3. **Lag Features**: Create daily sales aggregates and calculate lags (1d, 7d, 30d) for demand forecasting.
4. **Feast Setup**: Define feature definitions and use a local file-based registry.
5. **Validation**: Ensure features are correctly materialized and accessible via Feast.

## Tasks
- [x] Create `src/features/transformation.py` for Silver layer.
- [x] Create `src/features/definitions.py` for RFM and Lag logic.
- [x] Initialize Feast in `src/features/feature_repo/`.
- [x] Update Airflow to include feature engineering steps.
- [x] Verify MLflow connectivity and basic logging.

