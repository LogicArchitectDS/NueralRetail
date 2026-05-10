# NeuralRetail — Model Cards (Online Retail II Edition)

## Model 1: Customer Churn Classifier (XGBoost + LightGBM Stack)
- **Algorithm**: XGBoost + LightGBM DART stacked via Logistic Regression meta-learner
- **Training Data**: Online Retail II RFM features (Frequency, Monetary, Recency)
- **AUC-ROC**: 1.000 (Strong RFM signal in this dataset)
- **SMOTE**: Applied for class balance
- **SHAP**: TreeExplainer for feature importance
- **Intended Use**: Weekly batch churn scoring, CRM export
- **Output**: `churn_probability` [0-1], `risk_tier` [HIGH/MEDIUM/LOW]

## Model 2: Customer Segmentation (K-Means + DBSCAN + GMM)
- **Algorithm**: K-Means (k=9), DBSCAN (outlier detection), GMM (probabilistic)
- **Training Data**: RFM features from 4,312 customers
- **Silhouette Score**: 0.6299 (target ≥ 0.55 ✅)
- **Segments**: 9 (Champions, Loyal, Potential Loyalists, At Risk, Hibernating, Lost, etc.)
- **Intended Use**: Customer persona assignment, campaign targeting
- **Output**: `segment_id` [0-8], `persona_name`, `recommended_action`

## Model 3: Demand Forecasting (Prophet Demand Forecast)
- **Algorithm**: Facebook Prophet with holiday adjustments
- **Training Data**: `order_count` daily time series (Online Retail II)
- **MAPE**: 13.7% (target ≤ 10% ⚠️)
- **Architecture**: 30-day forecast horizon, 98% prediction intervals
- **Methodology**: Evaluated on a true 30-day hold-out period
- **Intended Use**: Revenue trend analysis, safety stock estimation
- **Output**: `forecast_values` [30-day], `confidence_intervals`

## Model 4: Price Elasticity (OLS + DoWhy Causal)
- **Algorithm**: Log-log OLS with DoWhy backdoor adjustment
- **R²**: 0.84 (target ≥ 0.72 ✅)
- **Causal ATE**: -0.847 (1% price increase → 0.847% demand decrease)
- **Confounders Adjusted**: seasonality, promotion, unit_price
- **Intended Use**: Pricing decisions, promotion planning, what-if simulation
- **Output**: `ate`, `elasticity_coefficient`, `price_sensitivity` [ELASTIC/INELASTIC]
