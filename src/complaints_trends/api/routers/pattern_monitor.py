from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ..deps import get_service_container
from ..schemas import AlertRowResponse, DailyPressureResponse, OverallStateResponse, PatternMonitorSummaryPayload, PatternMonitorSummaryResponse

router = APIRouter(prefix="/api/pattern-monitor", tags=["pattern-monitor"])


def _params(**kwargs):
    return kwargs


def _apply_upload_preset(services: dict, params: dict) -> tuple[dict, bool, str | None]:
    upload_id = params.get("upload_id")
    if not upload_id:
        return params, True, None
    preset = services["preparation"].build_pattern_monitor_preset(upload_id)
    if not preset.allowed:
        return params, False, preset.reason
    pp = preset.pattern_monitor_preset
    if pp:
        params = dict(params)
        params["date_from"] = params.get("date_from") or pp.date_from
        params["date_to"] = params.get("date_to") or pp.date_to
    return params, True, None


@router.get("/summary", response_model=PatternMonitorSummaryResponse)
def summary(pattern_tag: str = "latest", date_from: str | None = None, date_to: str | None = None, upload_id: str | None = None, category: list[str] = Query(default_factory=list), min_score: float | None = None, threshold_mode: str | None = None, top_n: int = 200, services=Depends(get_service_container)):
    params = _params(**locals())
    params, allowed, reason = _apply_upload_preset(services, params)
    if not allowed:
        return PatternMonitorSummaryResponse(tag=pattern_tag, allowed=False, reason=reason, upload_id=upload_id, summary=PatternMonitorSummaryPayload())
    result = services["pattern_monitor"].summary(pattern_tag, params)
    result.upload_id = upload_id
    return result


@router.get("/alerts", response_model=AlertRowResponse)
def alerts(pattern_tag: str = "latest", date_from: str | None = None, date_to: str | None = None, upload_id: str | None = None, category: list[str] = Query(default_factory=list), min_score: float | None = None, threshold_mode: str | None = None, scoring_mode: str = "base", top_n: int = 200, services=Depends(get_service_container)):
    params = _params(**locals())
    params, allowed, _ = _apply_upload_preset(services, params)
    if not allowed:
        return AlertRowResponse(rows=[])
    return services["pattern_monitor"].alerts(pattern_tag, params)


@router.get("/pressure", response_model=DailyPressureResponse)
def pressure(pattern_tag: str = "latest", date_from: str | None = None, date_to: str | None = None, upload_id: str | None = None, category: list[str] = Query(default_factory=list), min_score: float | None = None, threshold_mode: str | None = None, top_n: int = 200, services=Depends(get_service_container)):
    params = _params(**locals())
    params, allowed, _ = _apply_upload_preset(services, params)
    if not allowed:
        return DailyPressureResponse(rows=[])
    return services["pattern_monitor"].pressure(pattern_tag, params)


@router.get("/state", response_model=OverallStateResponse)
def state(pattern_tag: str = "latest", date_from: str | None = None, date_to: str | None = None, upload_id: str | None = None, category: list[str] = Query(default_factory=list), min_score: float | None = None, threshold_mode: str | None = None, top_n: int = 200, services=Depends(get_service_container)):
    params = _params(**locals())
    params, allowed, _ = _apply_upload_preset(services, params)
    if not allowed:
        return OverallStateResponse(rows=[])
    return services["pattern_monitor"].state(pattern_tag, params)


@router.get("/examples")
def examples(pattern_tag: str = "latest", date_from: str | None = None, date_to: str | None = None, upload_id: str | None = None, category: list[str] = Query(default_factory=list), min_score: float | None = None, threshold_mode: str | None = None, top_n: int = 50, services=Depends(get_service_container)):
    params = _params(**locals())
    params, allowed, reason = _apply_upload_preset(services, params)
    if not allowed:
        return {"tag": pattern_tag, "allowed": False, "reason": reason, "rows": []}
    out = services["pattern_monitor"].examples(pattern_tag, params)
    out["allowed"] = True
    out["reason"] = None
    return out
