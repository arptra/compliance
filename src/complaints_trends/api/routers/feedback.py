from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_service_container
from ..schemas import CalibratorTrainRequest, CalibratorVersionResponse, FeedbackBulkCreate, FeedbackCreate, FeedbackItem, FeedbackSummaryResponse

router = APIRouter(prefix="/api", tags=["feedback"])


@router.post("/feedback", response_model=FeedbackItem)
def create_feedback(payload: FeedbackCreate, services=Depends(get_service_container)):
    return services["feedback"].upsert_feedback(payload.model_dump())


@router.post("/feedback/bulk")
def create_feedback_bulk(payload: FeedbackBulkCreate, services=Depends(get_service_container)):
    rows = [r.model_dump() for r in payload.rows]
    return services["feedback"].bulk_upsert(rows)


@router.get("/feedback", response_model=list[FeedbackItem])
def list_feedback(pattern_tag: str | None = None, reviewer: str | None = None, verdict: str | None = None, date_from: str | None = None, date_to: str | None = None, category: str | None = None, limit: int = 1000, services=Depends(get_service_container)):
    return services["feedback"].list_feedback(locals())


@router.get("/feedback/summary", response_model=FeedbackSummaryResponse)
def feedback_summary(pattern_tag: str | None = None, reviewer: str | None = None, verdict: str | None = None, date_from: str | None = None, date_to: str | None = None, category: str | None = None, limit: int = 10000, services=Depends(get_service_container)):
    return services["feedback"].summary(locals())


@router.post("/feedback/reset")
def reset_feedback(row_id: str, pattern_tag: str | None = None, services=Depends(get_service_container)):
    return services["feedback"].delete_feedback(row_id=row_id, pattern_tag=pattern_tag)


@router.post("/feedback/reset-all")
def reset_feedback_all(pattern_tag: str | None = None, reviewer: str | None = None, services=Depends(get_service_container)):
    return services["feedback"].delete_feedback_for_scope(pattern_tag=pattern_tag, reviewer=reviewer)


@router.post("/pattern-monitor/calibrator/train", response_model=CalibratorVersionResponse)
def train_calibrator(payload: CalibratorTrainRequest, services=Depends(get_service_container)):
    params = payload.model_dump()
    scored = services["pattern_monitor"]._filter(
        services["loader"].load_pattern_monitor_scored(services["loader"].resolve_tag("pattern_monitor", params["pattern_tag"])),
        params,
    )
    return services["calibrator"].train(params, scored)


@router.get("/pattern-monitor/calibrator/versions", response_model=list[CalibratorVersionResponse])
def list_versions(services=Depends(get_service_container)):
    return services["model_registry"].list_versions()


@router.post("/pattern-monitor/calibrator/{version_id}/activate")
def activate_version(version_id: str, services=Depends(get_service_container)):
    services["model_registry"].set_active(version_id, active=True)
    return {"status": "ok", "version_id": version_id, "active": True}


@router.post("/pattern-monitor/calibrator/{version_id}/deactivate")
def deactivate_version(version_id: str, services=Depends(get_service_container)):
    services["model_registry"].set_active(version_id, active=False)
    return {"status": "ok", "version_id": version_id, "active": False}
