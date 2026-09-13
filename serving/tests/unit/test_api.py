"""
Unit tests: no network calls, no real Databricks. Everything that touches
Databricks is monkeypatched. These run on every PR in CI.
"""

import io
import os
import sys

import pytest

# app/config.py reads required env vars at import time, so they must exist
# before we import anything from app.*
os.environ.setdefault("DATABRICKS_HOST", "https://fake-workspace.cloud.databricks.com")
os.environ.setdefault("DATABRICKS_TOKEN", "fake-token-for-tests")
os.environ.setdefault("SERVING_API_KEY", "test-api-key")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi.testclient import TestClient

from app import databricks_client as dbx
from app.config import settings
from app.main import app
from app.schemas import AttritionRequest, PromotionRequest

client = TestClient(app)
API_KEY_HEADERS = {"X-API-Key": settings.API_KEY}


# --------------------------------------------------------------------------
# Schema validation
# --------------------------------------------------------------------------
def test_attrition_request_valid():
    req = AttritionRequest(
        age=35, department="Engineering", job_level=2, monthly_income=45000,
        years_at_company=4, distance_from_home_km=12, job_satisfaction=2,
        work_life_balance=2, environment_satisfaction=3, overtime="Yes",
        num_companies_worked=3, training_times_last_year=1, percent_salary_hike=10,
        years_since_last_promotion=3, years_with_curr_manager=2, performance_rating=3,
    )
    assert req.overtime == "Yes"


def test_attrition_request_rejects_bad_enum():
    with pytest.raises(ValueError):
        AttritionRequest(
            age=35, department="Engineering", job_level=2, monthly_income=45000,
            years_at_company=4, distance_from_home_km=12, job_satisfaction=2,
            work_life_balance=2, environment_satisfaction=3, overtime="Maybe",  # invalid
            num_companies_worked=3, training_times_last_year=1, percent_salary_hike=10,
            years_since_last_promotion=3, years_with_curr_manager=2, performance_rating=3,
        )


def test_promotion_request_valid():
    req = PromotionRequest(
        age=30, department="Sales", education_level="Masters",
        years_in_current_role=2, avg_performance_rating_3yr=4.1,
        kpi_met_percentage=80, training_score=70, num_projects_completed=5,
        awards_won=1, manager_rating=3, years_since_last_promotion=2,
        peer_review_score=4.0,
    )
    assert req.education_level == "Masters"


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------
def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
def test_predict_rejects_missing_api_key():
    resp = client.post("/predict/attrition", json={})
    assert resp.status_code in (401, 422)  # missing header vs missing body, either is a reject


def test_predict_rejects_wrong_api_key():
    resp = client.post(
        "/predict/attrition", json={}, headers={"X-API-Key": "wrong-key"}
    )
    assert resp.status_code == 401


# --------------------------------------------------------------------------
# Real-time prediction routing (Databricks call mocked)
# --------------------------------------------------------------------------
def test_predict_attrition_success(monkeypatch):
    def fake_invoke(endpoint_url, record):
        assert "attrition-endpoint" in endpoint_url
        return {"predictions": [1]}

    monkeypatch.setattr(dbx, "invoke_serving_endpoint", fake_invoke)

    body = {
        "age": 35, "department": "Engineering", "job_level": 2, "monthly_income": 45000,
        "years_at_company": 4, "distance_from_home_km": 12, "job_satisfaction": 2,
        "work_life_balance": 2, "environment_satisfaction": 3, "overtime": "Yes",
        "num_companies_worked": 3, "training_times_last_year": 1, "percent_salary_hike": 10,
        "years_since_last_promotion": 3, "years_with_curr_manager": 2, "performance_rating": 3,
    }
    resp = client.post("/predict/attrition", json=body, headers=API_KEY_HEADERS)
    assert resp.status_code == 200
    assert resp.json() == {"prediction": 1, "model": "attrition"}


def test_predict_attrition_upstream_failure_returns_502(monkeypatch):
    def fake_invoke(endpoint_url, record):
        raise RuntimeError("simulated Databricks outage")

    monkeypatch.setattr(dbx, "invoke_serving_endpoint", fake_invoke)

    body = {
        "age": 35, "department": "Engineering", "job_level": 2, "monthly_income": 45000,
        "years_at_company": 4, "distance_from_home_km": 12, "job_satisfaction": 2,
        "work_life_balance": 2, "environment_satisfaction": 3, "overtime": "Yes",
        "num_companies_worked": 3, "training_times_last_year": 1, "percent_salary_hike": 10,
        "years_since_last_promotion": 3, "years_with_curr_manager": 2, "performance_rating": 3,
    }
    resp = client.post("/predict/attrition", json=body, headers=API_KEY_HEADERS)
    assert resp.status_code == 502


def test_predict_promotion_success(monkeypatch):
    def fake_invoke(endpoint_url, record):
        assert "promotion-endpoint" in endpoint_url
        return {"predictions": [0]}

    monkeypatch.setattr(dbx, "invoke_serving_endpoint", fake_invoke)

    body = {
        "age": 30, "department": "Sales", "education_level": "Masters",
        "years_in_current_role": 2, "avg_performance_rating_3yr": 4.1,
        "kpi_met_percentage": 80, "training_score": 70, "num_projects_completed": 5,
        "awards_won": 1, "manager_rating": 3, "years_since_last_promotion": 2,
        "peer_review_score": 4.0,
    }
    resp = client.post("/predict/promotion", json=body, headers=API_KEY_HEADERS)
    assert resp.status_code == 200
    assert resp.json() == {"prediction": 0, "model": "promotion"}


# --------------------------------------------------------------------------
# Batch upload validation (Databricks calls mocked)
# --------------------------------------------------------------------------
def test_batch_upload_rejects_non_csv(monkeypatch):
    files = {"file": ("data.txt", io.BytesIO(b"not a csv"), "text/plain")}
    resp = client.post("/predict/batch/attrition", files=files, headers=API_KEY_HEADERS)
    assert resp.status_code == 422


def test_batch_upload_rejects_empty_file(monkeypatch):
    files = {"file": ("data.csv", io.BytesIO(b""), "text/csv")}
    resp = client.post("/predict/batch/attrition", files=files, headers=API_KEY_HEADERS)
    assert resp.status_code == 422


def test_batch_upload_success(monkeypatch):
    monkeypatch.setattr(dbx, "upload_csv_to_volume", lambda contents, path: None)
    monkeypatch.setattr(dbx, "trigger_job", lambda job_id, input_path: 123456)

    files = {"file": ("data.csv", io.BytesIO(b"employee_id,age\nEMP001,30\n"), "text/csv")}
    resp = client.post("/predict/batch/attrition", files=files, headers=API_KEY_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == 123456
    assert body["status"] == "SUBMITTED"


# --------------------------------------------------------------------------
# Batch status polling (Databricks call mocked)
# --------------------------------------------------------------------------
def test_batch_status_success(monkeypatch):
    monkeypatch.setattr(
        dbx,
        "get_run_status",
        lambda run_id: {"state": {"life_cycle_state": "TERMINATED", "result_state": "SUCCESS"}},
    )
    resp = client.get("/predict/batch/status/123456", headers=API_KEY_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["life_cycle_state"] == "TERMINATED"
    assert body["result_state"] == "SUCCESS"
