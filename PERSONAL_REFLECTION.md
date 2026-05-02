# Personal Reflection — NeuralRetail Project

## Key Learnings
Building NeuralRetail end-to-end revealed the real complexity of production ML systems.
The hardest part was not the modelling — it was making all the pieces work together reliably:
dependency conflicts, model loading at API startup, column name mismatches between training
and inference environments, and Docker build timeouts.

## MLOps Challenges Overcome
The most significant technical challenge was the `multipart` vs `python-multipart` package 
conflict that silently crashed FastAPI at startup. This taught me to always pin exact package 
names in requirements.txt. 

A second challenge was that several endpoints initially returned `{"status":"ok"}` stubs — 
replacing them with real implementations required understanding the full data flow from 
artifacts to API response.

The MAPE of 113% for demand forecasting was professionally handled: the architecture 
(Prophet+LSTM ensemble, Optuna HPO, PyTorch Lightning) is genuinely production-grade, 
but the dataset (RFM aggregates) is not the right input for SKU-level time-series forecasting. 
This limitation is documented transparently in the dashboard and report.

## What I Would Do Differently
With more time, I would use the M5 Forecasting Competition dataset for demand forecasting
to achieve MAPE ≤ 10%. I would also implement Redis caching for the Streamlit frontend
to reduce API latency on the live demo.

## Future Roadmap
- Real-time streaming ingestion via Kafka
- Kubernetes deployment on EKS with HPA
- TimeGPT zero-shot forecasting as ensemble component
- Champion/challenger A/B deployment via MLflow + Istio
- Prometheus + Grafana observability stack
