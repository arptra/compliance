from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi import HTTPException

from ..deps import get_service_container
from ..schemas import RunRequest, RunResponse

router = APIRouter(prefix="/api/runs", tags=["runs"])


def _unwrap_run_response(resp: RunResponse) -> RunResponse:
    if resp.status == "success":
        return resp
    if resp.status == "busy":
        raise HTTPException(status_code=409, detail=resp.error or "Run is already in progress")
    raise HTTPException(status_code=500, detail=resp.error or "Run failed")


@router.post("/viz-build", response_model=RunResponse)
def run_viz(req: RunRequest, services=Depends(get_service_container)):
    return _unwrap_run_response(services["run"].run_viz_build(req))


@router.post("/pattern-fit", response_model=RunResponse)
def run_pf(req: RunRequest, services=Depends(get_service_container)):
    return _unwrap_run_response(services["run"].run_pattern_fit(req))


@router.post("/pattern-monitor", response_model=RunResponse)
def run_pm(req: RunRequest, services=Depends(get_service_container)):
    return _unwrap_run_response(services["run"].run_pattern_monitor(req))


@router.post("/infer-month", response_model=RunResponse)
def run_infer(req: RunRequest, services=Depends(get_service_container)):
    return _unwrap_run_response(services["run"].run_infer_month(req))


@router.get("/history/{run_id}")
def run_history(run_id: str):
    return {"run_id": run_id, "status": "not_persisted", "message": "MVP mode has no persistent run history"}
