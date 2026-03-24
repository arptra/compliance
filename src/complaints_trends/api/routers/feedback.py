from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse

from ..deps import get_service_container
from ..schemas import (
    CalibratorTrainRequest,
    CalibratorVersionResponse,
    FeedbackBulkCreate,
    FeedbackCreate,
    FeedbackDatasetResponse,
    FeedbackItem,
    FeedbackSummaryResponse,
    ModelQualityResponse,
    UnflaggedAuditCreateRequest,
    UnflaggedAuditReviewRequest,
    UnflaggedAuditSample,
)

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


@router.get("/feedback/dataset", response_model=FeedbackDatasetResponse)
def feedback_dataset(pattern_tag: str | None = None, reviewer: str | None = None, verdict: str | None = None, category: str | None = None, subcategory: str | None = None, reason_code: str | None = None, model_version: str | None = None, date_from: str | None = None, date_to: str | None = None, q: str | None = None, page: int = 1, page_size: int = 50, sort_by: str = "updated_at", sort_order: str = "desc", services=Depends(get_service_container)):
    return services["feedback_dataset"].list_feedback_dataset(locals())


@router.get("/feedback/export", response_class=PlainTextResponse)
def feedback_export(pattern_tag: str | None = None, reviewer: str | None = None, verdict: str | None = None, category: str | None = None, subcategory: str | None = None, reason_code: str | None = None, model_version: str | None = None, date_from: str | None = None, date_to: str | None = None, q: str | None = None, output_format: str = "csv", positives_only: bool = False, services=Depends(get_service_container)):
    content = services["feedback_dataset"].export_feedback_dataset(locals(), output_format=output_format)
    return PlainTextResponse(content)


@router.get("/feedback/{row_id}", response_model=FeedbackItem)
def feedback_item(row_id: str, services=Depends(get_service_container)):
    item = services["feedback_dataset"].get_feedback_item(row_id)
    if not item:
        raise HTTPException(status_code=404, detail="feedback item not found")
    return item


@router.get("/model-quality", response_model=ModelQualityResponse)
def model_quality(pattern_tag: str | None = None, reviewer: str | None = None, date_from: str | None = None, date_to: str | None = None, include_uncertain_as: str = "ignore", services=Depends(get_service_container)):
    return services["quality"].compute_model_quality(locals())


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


@router.post("/audit/unflagged/create", response_model=UnflaggedAuditSample)
def create_unflagged_audit(payload: UnflaggedAuditCreateRequest, services=Depends(get_service_container)):
    return services["audit"].create_unflagged_audit_sample(payload.model_dump())


@router.get("/audit/unflagged/samples", response_model=list[UnflaggedAuditSample])
def list_unflagged_audit_samples(services=Depends(get_service_container)):
    return [services["audit"].get_unflagged_sample(item["sample_id"]) for item in services["audit"].list_unflagged_audit_samples()]


@router.get("/audit/unflagged/{sample_id}", response_model=UnflaggedAuditSample)
def get_unflagged_audit_sample(sample_id: str, services=Depends(get_service_container)):
    return services["audit"].get_unflagged_sample(sample_id)


@router.post("/audit/unflagged/{sample_id}/review", response_model=UnflaggedAuditSample)
def review_unflagged_audit_sample(sample_id: str, payload: UnflaggedAuditReviewRequest, services=Depends(get_service_container)):
    return services["audit"].review_unflagged_sample(sample_id, payload.model_dump()["rows"])
