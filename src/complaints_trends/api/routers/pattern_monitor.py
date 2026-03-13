from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ..deps import get_service_container
from ..schemas import AlertRowResponse, DailyPressureResponse, OverallStateResponse, PatternMonitorSummaryResponse

router = APIRouter(prefix="/api/pattern-monitor", tags=["pattern-monitor"])


def _params(**kwargs):
    return kwargs


@router.get("/summary", response_model=PatternMonitorSummaryResponse)
def summary(pattern_tag: str = "latest", date_from: str | None = None, date_to: str | None = None, category: list[str] = Query(default_factory=list), min_score: float | None = None, threshold_mode: str | None = None, top_n: int = 200, services=Depends(get_service_container)):
    return services["pattern_monitor"].summary(pattern_tag, _params(**locals()))


@router.get("/alerts", response_model=AlertRowResponse)
def alerts(pattern_tag: str = "latest", date_from: str | None = None, date_to: str | None = None, category: list[str] = Query(default_factory=list), min_score: float | None = None, threshold_mode: str | None = None, top_n: int = 200, services=Depends(get_service_container)):
    return services["pattern_monitor"].alerts(pattern_tag, _params(**locals()))


@router.get("/pressure", response_model=DailyPressureResponse)
def pressure(pattern_tag: str = "latest", date_from: str | None = None, date_to: str | None = None, category: list[str] = Query(default_factory=list), min_score: float | None = None, threshold_mode: str | None = None, top_n: int = 200, services=Depends(get_service_container)):
    return services["pattern_monitor"].pressure(pattern_tag, _params(**locals()))


@router.get("/state", response_model=OverallStateResponse)
def state(pattern_tag: str = "latest", date_from: str | None = None, date_to: str | None = None, category: list[str] = Query(default_factory=list), min_score: float | None = None, threshold_mode: str | None = None, top_n: int = 200, services=Depends(get_service_container)):
    return services["pattern_monitor"].state(pattern_tag, _params(**locals()))


@router.get("/examples")
def examples(pattern_tag: str = "latest", date_from: str | None = None, date_to: str | None = None, category: list[str] = Query(default_factory=list), min_score: float | None = None, threshold_mode: str | None = None, top_n: int = 50, services=Depends(get_service_container)):
    return services["pattern_monitor"].examples(pattern_tag, _params(**locals()))
