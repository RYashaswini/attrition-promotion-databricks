"""
Thin wrapper around the Databricks REST APIs this service needs:
- Model Serving invocation (real-time predictions)
- Files API (upload a CSV into a Unity Catalog Volume)
- Jobs API (trigger a batch job, poll its status)

All calls retry on transient network/5xx failures via tenacity.
"""

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

HEADERS = {"Authorization": f"Bearer {settings.DATABRICKS_TOKEN}"}
TIMEOUT_SECONDS = 30


def _retryable():
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )


@_retryable()
def invoke_serving_endpoint(endpoint_url: str, record: dict) -> dict:
    payload = {"dataframe_records": [record]}
    resp = requests.post(
        endpoint_url, headers=HEADERS, json=payload, timeout=TIMEOUT_SECONDS
    )
    resp.raise_for_status()
    return resp.json()


@_retryable()
def upload_csv_to_volume(local_bytes: bytes, volume_path: str) -> None:
    """volume_path e.g. /Volumes/ml_dev/attrition_promotion/uploads/upload_xyz.csv"""
    url = f"{settings.DATABRICKS_HOST}/api/2.0/fs/files{volume_path}"
    resp = requests.put(
        url, headers=HEADERS, data=local_bytes, timeout=TIMEOUT_SECONDS
    )
    resp.raise_for_status()


@_retryable()
def download_csv_from_volume(volume_path: str) -> bytes:
    url = f"{settings.DATABRICKS_HOST}/api/2.0/fs/files{volume_path}"
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.content


@_retryable()
def trigger_job(job_id: str, input_path: str) -> int:
    url = f"{settings.DATABRICKS_HOST}/api/2.1/jobs/run-now"
    payload = {"job_id": job_id, "job_parameters": {"input_path": input_path}}
    resp = requests.post(url, headers=HEADERS, json=payload, timeout=TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()["run_id"]


@_retryable()
def get_run_status(run_id: int) -> dict:
    url = f"{settings.DATABRICKS_HOST}/api/2.1/jobs/runs/get"
    resp = requests.get(
        url, headers=HEADERS, params={"run_id": run_id}, timeout=TIMEOUT_SECONDS
    )
    resp.raise_for_status()
    return resp.json()
