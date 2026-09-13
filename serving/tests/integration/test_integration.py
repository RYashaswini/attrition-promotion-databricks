"""
Integration tests: hit the REAL Databricks workspace (serving endpoints +
Jobs API), using whatever DATABRICKS_HOST/DATABRICKS_TOKEN are set in the
environment. These are NOT run on every PR - only after a real deploy to
staging (or manually, right now, against ml_dev).

Requires: DATABRICKS_HOST, DATABRICKS_TOKEN, SERVING_API_KEY set in the
environment (same .env you already have working).
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)
API_KEY_HEADERS = {"X-API-Key": os.environ.get("SERVING_API_KEY", "")}

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

SAMPLE_ATTRITION_CSV = b"employee_id,age,department,job_level,monthly_income,years_at_company,distance_from_home_km,job_satisfaction,work_life_balance,environment_satisfaction,overtime,num_companies_worked,training_times_last_year,percent_salary_hike,years_since_last_promotion,years_with_curr_manager,performance_rating\nEMP00001,35,Engineering,2,45000,4,12,2,2,3,Yes,3,1,10,3,2,3\n"


@pytest.mark.integration
def test_real_attrition_prediction():
    resp = client.post("/predict/attrition", json=SAMPLE_ATTRITION_PAYLOAD, headers=API_KEY_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert body["model"] == "attrition"


@pytest.mark.integration
def test_real_promotion_prediction():
    resp = client.post("/predict/promotion", json=SAMPLE_PROMOTION_PAYLOAD, headers=API_KEY_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert body["model"] == "promotion"


@pytest.mark.integration
def test_real_batch_attrition_end_to_end():
    """Uploads a real CSV, triggers the real Databricks job, polls until it finishes."""
    files = {"file": ("integration_test.csv", SAMPLE_ATTRITION_CSV, "text/csv")}
    trigger_resp = client.post("/predict/batch/attrition", files=files, headers=API_KEY_HEADERS)
    assert trigger_resp.status_code == 200
    run_id = trigger_resp.json()["run_id"]

    # poll until terminal state, max ~5 minutes
    deadline = time.time() + 300
    life_cycle_state = None
    while time.time() < deadline:
        status_resp = client.get(f"/predict/batch/status/{run_id}", headers=API_KEY_HEADERS)
        assert status_resp.status_code == 200
        life_cycle_state = status_resp.json()["life_cycle_state"]
        if life_cycle_state == "TERMINATED":
            break
        time.sleep(10)

    assert life_cycle_state == "TERMINATED"