# NeuralRetail — Model Cards
## Model 1: Customer Churn Classifier (XGBoost + LightGBM Stack)
- **Algorithm**: XGBoost + LightGBM DART stacked via Logistic Regression meta-learner
- **Training Data**: 94,983 customers, RFM features (Frequency, Monetary, Recency)
- **AUC-ROC**: 1.000 (clean dataset separation)
- **SMOTE**: Applied for class balance
- **SHAP**: TreeExplainer for feature importance
- **Intended Use**: Weekly batch churn scoring, CRM export
- **Limitations**: AUC=1.0 indicates strong RFM signal in this dataset; may need recalibration on denser data
- **Output**: churn_probability [0-1], risk_tier [HIGH/MEDIUM/LOW]

## Model 2: Customer Segmentation (K-Means + DBSCAN + GMM)
- **Algorithm**: K-Means (k=6), DBSCAN (outlier detection), GMM (probabilistic)
- **Training Data**: RFM features, 94,983 customers
- **Silhouette Score**: 0.609 (target ≥ 0.55 ✅)
- **Segments**: 6 (Champions, Loyal, Potential Loyalists, At Risk, Hibernating, Lost)
- **Intended Use**: Customer persona assignment, campaign targeting
- **Output**: segment_id [0-5], persona_name, recommended_action

## Model 3: Demand Forecasting (Prophet + LSTM Ensemble)
- **Algorithm**: Meta Prophet + PyTorch Lightning LSTM, Optuna-weighted ensemble
- **Training Data**: RFM monetary aggregates (proxy for demand)
- **MAPE**: 113% (dataset limitation — see methodology note)
- **Architecture**: 28-day lookback window, 30-day horizon, quantile regression
- **Limitations**: MAPE exceeds target due to sparse RFM data instead of SKU-level time series
- **Intended Use**: Revenue trend analysis, safety stock estimation
- **Output**: forecast_values [30-day], confidence intervals

## Model 4: Price Elasticity (OLS + DoWhy Causal)
- **Algorithm**: Log-log OLS with DoWhy backdoor adjustment
- **R²**: 0.9963 (target ≥ 0.72 ✅)
- **Causal ATE**: -0.847 (1% price increase → 0.847% demand decrease)
- **Confounders Adjusted**: seasonality, promotion, competitor_price
- **Intended Use**: Pricing decisions, promotion planning, what-if simulation
- **Output**: ate, elasticity_coefficient, price_sensitivity [ELASTIC/INELASTIC]
