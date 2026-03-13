from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_service_container
from ..schemas import RunRequest, RunResponse

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.post("/viz-build", response_model=RunResponse)
def run_viz(req: RunRequest, services=Depends(get_service_container)):
    return services["run"].run_viz_build(req)


@router.post("/pattern-fit", response_model=RunResponse)
def run_pf(req: RunRequest, services=Depends(get_service_container)):
    return services["run"].run_pattern_fit(req)


@router.post("/pattern-monitor", response_model=RunResponse)
def run_pm(req: RunRequest, services=Depends(get_service_container)):
    return services["run"].run_pattern_monitor(req)


@router.post("/infer-month", response_model=RunResponse)
def run_infer(req: RunRequest, services=Depends(get_service_container)):
    return services["run"].run_infer_month(req)


@router.get("/history/{run_id}")
def run_history(run_id: str):
    return {"run_id": run_id, "status": "not_persisted", "message": "MVP mode has no persistent run history"}
