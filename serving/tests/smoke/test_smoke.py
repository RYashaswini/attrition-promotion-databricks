"""
Smoke tests: run against a just-deployed, already-running instance of the
FastAPI app (staging or prod), hit over real HTTP - not TestClient - since
the point is to verify the actual deployed process is reachable and healthy,
not just that the code imports correctly.

Usage: SMOKE_TARGET_URL=https://your-deployed-host SERVING_API_KEY=... pytest tests/smoke/ -v
"""

import os
import time

import pytest
import requests

BASE_URL = os.environ.get("SMOKE_TARGET_URL", "http://localhost:8000")
API_KEY_HEADERS = {"X-API-Key": os.environ.get("SERVING_API_KEY", "")}
LATENCY_BUDGET_SECONDS = 2.0

SAMPLE_ATTRITION_PAYLOAD = {
    "age": 35, "department": "Engineering", "job_level": 2, "monthly_income": 45000,
    "years_at_company": 4, "distance_from_home_km": 12, "job_satisfaction": 2,
    "work_life_balance": 2, "environment_satisfaction": 3, "overtime": "Yes",
    "num_companies_worked": 3, "training_times_last_year": 1, "percent_salary_hike": 10,
    "years_since_last_promotion": 3, "years_with_curr_manager": 2, "performance_rating": 3,
}

SAMPLE_PROMOTION_PAYLOAD = {
    "age": 30, "department": "Sales", "education_level": "Masters",
    "years_in_current_role": 2, "avg_performance_rating_3yr": 4.1,
    "kpi_met_percentage": 80, "training_score": 70, "num_projects_completed": 5,
    "awards_won": 1, "manager_rating": 3, "years_since_last_promotion": 2,
    "peer_review_score": 4.0,
}


@pytest.mark.smoke
def test_smoke_health():
    resp = requests.get(f"{BASE_URL}/health", timeout=5)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.smoke
def test_smoke_attrition_prediction_within_latency_budget():
    start = time.monotonic()
    resp = requests.post(
        f"{BASE_URL}/predict/attrition",
        json=SAMPLE_ATTRITION_PAYLOAD,
        headers=API_KEY_HEADERS,
        timeout=10,
    )
    elapsed = time.monotonic() - start
    assert resp.status_code == 200
    assert resp.json()["prediction"] in (0, 1)
    assert elapsed < LATENCY_BUDGET_SECONDS, f"attrition prediction took {elapsed:.2f}s"


@pytest.mark.smoke
def test_smoke_promotion_prediction_within_latency_budget():
    start = time.monotonic()
    resp = requests.post(
        f"{BASE_URL}/predict/promotion",
        json=SAMPLE_PROMOTION_PAYLOAD,
        headers=API_KEY_HEADERS,
        timeout=10,
    )
    elapsed = time.monotonic() - start
    assert resp.status_code == 200
    assert resp.json()["prediction"] in (0, 1)
    assert elapsed < LATENCY_BUDGET_SECONDS, f"promotion prediction took {elapsed:.2f}s"