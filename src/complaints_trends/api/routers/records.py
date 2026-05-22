from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException

from ..deps import get_service_container
from ..schemas import RecordClearRequest, RecordDeleteRequest, RecordMutationResponse, RecordSearchRequest, RecordSearchResponse

router = APIRouter(prefix="/api/records", tags=["records"])


@router.post("/search", response_model=RecordSearchResponse)
def search_records(
    req: RecordSearchRequest,
    authorization: str | None = Header(default=None),
    services=Depends(get_service_container),
):
    user = services["catalog"].user_from_authorization(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    try:
        result = services["record_lake"].search_records(
            stage=req.stage,
            workspace_id=str(user.get("workspace_id") or "default"),
            filters=[item.model_dump() for item in req.filters],
            columns=req.columns or None,
            limit=max(1, min(int(req.limit), 1000)),
            offset=max(0, int(req.offset)),
        )
        return RecordSearchResponse(stage=req.stage, **result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/delete", response_model=RecordMutationResponse)
def delete_records(
    req: RecordDeleteRequest,
    authorization: str | None = Header(default=None),
    services=Depends(get_service_container),
):
    user = services["catalog"].user_from_authorization(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    try:
        result = services["record_lake"].delete_records(
            stage=req.stage,
            workspace_id=str(user.get("workspace_id") or "default"),
            record_ids=req.record_ids,
        )
        return RecordMutationResponse(stage=req.stage, **result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/clear", response_model=RecordMutationResponse)
def clear_records(
    req: RecordClearRequest,
    authorization: str | None = Header(default=None),
    services=Depends(get_service_container),
):
    user = services["catalog"].user_from_authorization(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    try:
        result = services["record_lake"].clear_stage(
            stage=req.stage,
            workspace_id=str(user.get("workspace_id") or "default"),
        )
        return RecordMutationResponse(stage=req.stage, **result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
