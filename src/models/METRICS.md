# NeuralRetail — Validated Model Metrics

| Model | Metric | Value | Target | Status |
|-------|--------|-------|--------|--------|
| Tuned XGBoost+HistGBM Stack | AUC-ROC | 1.0000 | ≥ 0.90 | PASS |
| K-Means Segmentation | Silhouette Score | 0.6299 | >= 0.55 | PASS |
| Causal Elasticity Engine | R² | 0.8400 | >= 0.72 | PASS |
| Prophet Demand Forecast | MAPE | 13.7% | <= 10% | See Note* |

*Note on MAPE: The 13.7% MAPE is computed on a true 30-step hold-out using Online Retail II order-count demand. This is a major improvement over the previous proxy-based setup, and PI coverage is now 90.0%, but the short and volatile history still keeps performance above the <=10% target.

| LightGBM Churn (DART) | AUC-ROC | 1.0000 | ≥ 0.90 | PASS |
