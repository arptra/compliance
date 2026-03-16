from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_service_container
from ..schemas import ReportRequest, ReportResponse

router = APIRouter(prefix="/api/reports", tags=["reports"])


@router.post("/executive", response_model=ReportResponse)
def executive(req: ReportRequest, services=Depends(get_service_container)):
    return services["report"].build_report("executive", req)


@router.post("/operations", response_model=ReportResponse)
def operations(req: ReportRequest, services=Depends(get_service_container)):
    return services["report"].build_report("operations", req)


@router.post("/pattern-monitoring", response_model=ReportResponse)
def pattern_monitoring(req: ReportRequest, services=Depends(get_service_container)):
    return services["report"].build_report("pattern-monitoring", req)
