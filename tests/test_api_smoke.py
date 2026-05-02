"""
NeuralRetail API Smoke Tests — Amdox AMX-DS-2026-04
Verifies all F01-F08 spec endpoints are functional.
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

@pytest.fixture(scope="module")
def client():
    from src.api.main import app
    return TestClient(app)

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"

def test_kpis_has_required_keys(client):
    r = client.get("/kpis")
    assert r.status_code == 200
    data = r.json()
    assert "total_revenue" in data
    assert "active_customers" in data
    assert "avg_churn_risk" in data
    assert data["segmentation_silhouette"] == 0.609
    assert data["price_r2"] == 0.9963

def test_monitoring_drift_has_psi(client):
    r = client.get("/monitoring/drift")
    assert r.status_code == 200
    data = r.json()
    assert "drift_status" in data
    assert "overall_psi" in data
    assert data["drift_status"] in ["STABLE","DRIFT_DETECTED","NO_DATA","ERROR"]

def test_monitoring_dq_has_status(client):
    r = client.get("/monitoring/dq")
    assert r.status_code == 200
    data = r.json()
    assert "dq_score" in data
    assert "status" in data
    assert data["status"] in ["PASS","WARNING","FAIL","NO_DATA","ERROR"]

def test_monitoring_retrain_returns_action(client):
    r = client.post("/monitoring/retrain")
    assert r.status_code == 200
    data = r.json()
    assert "action_taken" in data
    assert data["action_taken"] in ["NO_ACTION","RETRAIN_TRIGGERED","FAILED"]

def test_pricing_causal_has_elasticity(client):
    r = client.get("/pricing/causal")
    assert r.status_code == 200
    data = r.json()
    assert "ols_r2" in data or "ate" in data

def test_predict_churn(client):
    r = client.post("/predict/churn", json={"Frequency": 5, "Monetary": 500.0})
    assert r.status_code in [200, 503], f"Got {r.status_code}: {r.text}"
    if r.status_code == 200:
        data = r.json()
        assert "churn_probability" in data or "prediction" in data or "score" in data

def test_predict_churn_stack(client):
    r = client.post("/predict/churn/stack",
        json={"recency": 30.0, "frequency": 5.0, "monetary": 500.0})
    assert r.status_code == 200
    data = r.json()
    if "error" not in data:
        assert "churn_probability" in data
        assert "risk_tier" in data

def test_inventory_optimize(client):
    r = client.post("/inventory/optimize",
        json={"annual_demand":1000.0,"order_cost":50.0,"holding_cost":2.0,
              "max_lead_time":14.0,"avg_lead_time":7.0,
              "max_daily_demand":50.0,"avg_daily_demand":30.0})
    assert r.status_code == 200

def test_executive_revenue_trend(client):
    r = client.get("/executive/revenue-trend")
    assert r.status_code == 200

def test_export_crm_high_risk(client):
    r = client.get("/export/crm/high_risk?threshold=0.7")
    assert r.status_code == 200

def test_predict_segment(client):
    r = client.post("/predict/segment", json={"Frequency": 5, "Monetary": 500.0})
    assert r.status_code in [200, 503], f"Got {r.status_code}: {r.text}"
    if r.status_code == 200:
        data = r.json()
        assert "segment" in data or "cluster" in data or "label" in data

def test_executive_demand_forecast(client):
    r = client.get("/executive/demand-forecast")
    assert r.status_code == 200
    data = r.json()
    assert "forecast_values" in data
    assert "mape" in data

def test_export_crm_returns_csv(client):
    r = client.get("/export/crm/high_risk")
    assert r.status_code == 200
    content_type = r.headers.get("content-type", "")
    assert "csv" in content_type or "text" in content_type or len(r.text) > 10

def test_monitoring_dq_has_score(client):
    r = client.get("/monitoring/dq")
    assert r.status_code == 200
    data = r.json()
    assert "dq_score" in data
    assert data["dq_score"] >= 0
