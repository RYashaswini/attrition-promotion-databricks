import time
import uuid

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.security import APIKeyHeader

from app import databricks_client as dbx
from app.config import settings
from app.schemas import (
    AttritionRequest,
    BatchStatusResponse,
    BatchTriggerResponse,
    PredictionResponse,
    PromotionRequest,
)

app = FastAPI(title="Attrition & Promotion Orchestration API")

api_key_header = APIKeyHeader(name="X-API-Key")


def require_api_key(key: str = Depends(api_key_header)):
    if key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------------------------------
# Real-time inference
# --------------------------------------------------------------------------
@app.post("/predict/attrition", response_model=PredictionResponse, dependencies=[Depends(require_api_key)])
def predict_attrition(request: AttritionRequest):
    try:
        result = dbx.invoke_serving_endpoint(settings.ATTRITION_ENDPOINT_URL, request.model_dump())
        prediction = result["predictions"][0]
        return PredictionResponse(prediction=int(prediction), model="attrition")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"attrition-endpoint call failed: {exc}")


@app.post("/predict/promotion", response_model=PredictionResponse, dependencies=[Depends(require_api_key)])
def predict_promotion(request: PromotionRequest):
    try:
        result = dbx.invoke_serving_endpoint(settings.PROMOTION_ENDPOINT_URL, request.model_dump())
        prediction = result["predictions"][0]
        return PredictionResponse(prediction=int(prediction), model="promotion")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"promotion-endpoint call failed: {exc}")


# --------------------------------------------------------------------------
# Batch: upload CSV, trigger job
# --------------------------------------------------------------------------
def _handle_batch_upload(file: UploadFile, job_id: str) -> BatchTriggerResponse:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Only .csv files are accepted")

    contents = file.file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")
    if len(contents) > 50 * 1024 * 1024:  # 50MB cap
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    upload_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    volume_path = f"{settings.UPLOADS_VOLUME_PATH}/upload_{upload_id}.csv"

    try:
        dbx.upload_csv_to_volume(contents, volume_path)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"upload to Databricks Volume failed: {exc}")

    try:
        run_id = dbx.trigger_job(job_id, input_path=volume_path)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"triggering Databricks job failed: {exc}")

    return BatchTriggerResponse(run_id=run_id, job=job_id)


@app.post(
    "/predict/batch/attrition",
    response_model=BatchTriggerResponse,
    dependencies=[Depends(require_api_key)],
)
def batch_attrition(file: UploadFile = File(...)):
    return _handle_batch_upload(file, settings.ATTRITION_JOB_ID)


@app.post(
    "/predict/batch/promotion",
    response_model=BatchTriggerResponse,
    dependencies=[Depends(require_api_key)],
)
def batch_promotion(file: UploadFile = File(...)):
    return _handle_batch_upload(file, settings.PROMOTION_JOB_ID)


# --------------------------------------------------------------------------
# Batch: poll status
# --------------------------------------------------------------------------
@app.get(
    "/predict/batch/status/{run_id}",
    response_model=BatchStatusResponse,
    dependencies=[Depends(require_api_key)],
)
def batch_status(run_id: int):
    try:
        run = dbx.get_run_status(run_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"could not fetch run status: {exc}")

    state = run.get("state", {})
    life_cycle_state = state.get("life_cycle_state", "UNKNOWN")
    result_state = state.get("result_state")

    return BatchStatusResponse(
        run_id=run_id,
        life_cycle_state=life_cycle_state,
        result_state=result_state,
        output_path=None,  # caller can fetch the actual predictions separately, see below
    )
