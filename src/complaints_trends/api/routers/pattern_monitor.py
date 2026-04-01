from __future__ import annotations

from pathlib import Path

import pandas as pd
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
        # If upload preset is stale/invalid, fallback to regular monitor query
        # instead of returning empty dashboard payloads.
        params = dict(params)
        params.pop("upload_id", None)
        return params, True, f"upload_preset_ignored: {preset.reason}"
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
    if reason:
        result.reason = reason
    return result


@router.get("/alerts", response_model=AlertRowResponse)
def alerts(pattern_tag: str = "latest", date_from: str | None = None, date_to: str | None = None, upload_id: str | None = None, category: list[str] = Query(default_factory=list), min_score: float | None = None, threshold_mode: str | None = None, scoring_mode: str = "base", top_n: int | None = None, services=Depends(get_service_container)):
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
def examples(pattern_tag: str = "latest", date_from: str | None = None, date_to: str | None = None, upload_id: str | None = None, category: list[str] = Query(default_factory=list), min_score: float | None = None, threshold_mode: str | None = None, top_n: int | None = None, services=Depends(get_service_container)):
    params = _params(**locals())
    params, allowed, reason = _apply_upload_preset(services, params)
    if not allowed:
        return {"tag": pattern_tag, "allowed": False, "reason": reason, "rows": []}
    out = services["pattern_monitor"].examples(pattern_tag, params)
    out["allowed"] = True
    out["reason"] = None
    return out


@router.get("/run-output", response_model=AlertRowResponse)
def run_output(pattern_tag: str = "latest", top_n: int | None = None, services=Depends(get_service_container)):
    return services["pattern_monitor"].run_output_rows(pattern_tag, top_n=top_n)


@router.get("/run-output-by-path", response_model=AlertRowResponse)
def run_output_by_path(path: str, top_n: int | None = None, services=Depends(get_service_container)):
    return services["pattern_monitor"].run_output_rows_by_path(path, top_n=top_n)


@router.get("/top-alerts-excel", response_model=AlertRowResponse)
def top_alerts_excel(pattern_tag: str = "latest", top_n: int | None = None, services=Depends(get_service_container)):
    resolved = services["loader"].resolve_tag("pattern_monitor", pattern_tag)
    export_path = Path(services["cfg"].analysis.pattern_monitoring.exports_dir) / f"pattern_monitor_{resolved}.xlsx"
    if not export_path.exists():
        return AlertRowResponse(rows=[])
    for sheet in ("top_alerts", "alert_examples"):
        try:
            df = pd.read_excel(export_path, sheet_name=sheet)
            rows = services["pattern_monitor"]._json_records(df, top_n)
            return AlertRowResponse(rows=rows)
        except Exception:
            continue
    return AlertRowResponse(rows=[])
