from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ..deps import get_service_container
from ..schemas import (
    PatternMonitorPresetResponse,
    PreparationJobsResponse,
    PreparationPreviewResponse,
    PreparationRunResponse,
    PreparationUploadResponse,
)

router = APIRouter(prefix="/api/preparation", tags=["preparation"])


@router.post("/upload", response_model=PreparationUploadResponse)
async def upload(file: UploadFile = File(...), services=Depends(get_service_container)):
    content = await file.read()
    return services["preparation"].create_upload_job(file.filename or "upload.xlsx", content)


@router.post("/{upload_id}/run", response_model=PreparationRunResponse)
def run(upload_id: str, services=Depends(get_service_container)):
    resp = services["preparation"].run_preparation_job(upload_id)
    if resp.status == "failed":
        code = 404 if resp.error_message == "upload_not_found" else 400
        raise HTTPException(status_code=code, detail=resp.error_message or "preparation_failed")
    return resp


@router.post("/upload-and-run", response_model=PreparationRunResponse)
async def upload_and_run(file: UploadFile = File(...), services=Depends(get_service_container)):
    content = await file.read()
    created = services["preparation"].create_upload_job(file.filename or "upload.xlsx", content)
    resp = services["preparation"].run_preparation_job(created.upload_id)
    if resp.status == "failed":
        raise HTTPException(status_code=400, detail=resp.error_message or "preparation_failed")
    return resp


@router.get("/jobs", response_model=PreparationJobsResponse)
def jobs(status: str | None = None, limit: int = 30, services=Depends(get_service_container)):
    return PreparationJobsResponse(jobs=services["preparation"].list_preparation_jobs(status=status, limit=limit))


@router.get("/jobs/{upload_id}")
def job_detail(upload_id: str, services=Depends(get_service_container)):
    return services["preparation"].get_preparation_job(upload_id)


@router.get("/jobs/{upload_id}/preview", response_model=PreparationPreviewResponse)
def preview(upload_id: str, services=Depends(get_service_container)):
    resp = services["preparation"].get_preview(upload_id)
    if resp is None:
        return PreparationPreviewResponse(upload_id=upload_id, filename="", status="not_found", rows_total=0, available_columns=[])
    return resp


@router.post("/jobs/{upload_id}/open-pattern-monitor", response_model=PatternMonitorPresetResponse)
def open_pattern_monitor(upload_id: str, services=Depends(get_service_container)):
    return services["preparation"].build_pattern_monitor_preset(upload_id)
