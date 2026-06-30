from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Header, HTTPException, Response, UploadFile

from ..deps import get_service_container
from ..schemas import (
    GigaChatAnnotatedExportRequest,
    GigaChatBackgroundTaskListResponse,
    GigaChatBackgroundTaskResultResponse,
    GigaChatBackgroundTaskStartRequest,
    GigaChatBackgroundTaskSummary,
    GigaChatFinalPromptRequest,
    GigaChatFinalPromptResponse,
    GigaChatLabRowRunRequest,
    GigaChatLabRowRunResponse,
    GigaChatLabSettingsResponse,
    GigaChatLabSettingsUpdateRequest,
    GigaChatLabSettingsVersionCreateRequest,
    GigaChatLabSettingsVersionResponse,
    GigaChatLabSettingsVersionsResponse,
    GigaChatLabSettingsVersionUpdateRequest,
    GigaChatRuleEvaluationRequest,
    GigaChatRuleEvaluationResponse,
    GigaChatTransportProbeRequest,
    GigaChatTransportProbeResponse,
    GigaChatTransportStatusResponse,
    GigaChatWorkbookLocalUploadRequest,
    GigaChatWorkbookChunkedUploadCompleteResponse,
    GigaChatWorkbookChunkedUploadStartRequest,
    GigaChatWorkbookChunkedUploadStartResponse,
    GigaChatWorkbookChunkUploadResponse,
    GigaChatWorkbookUploadTaskResponse,
    GigaChatWorkbookRowsExportRequest,
    GigaChatWorkbookSelectSheetRequest,
    GigaChatWorkbookSheetDataResponse,
    GigaChatWorkbookUploadResponse,
)

router = APIRouter(prefix="/api/gigachat", tags=["gigachat"])


def _optional_user(services: dict, authorization: str | None) -> dict:
    return services["catalog"].user_from_authorization(authorization) or services["catalog"].get_or_create_dev_user()


@router.get("/status", response_model=GigaChatTransportStatusResponse)
def status(services=Depends(get_service_container)):
    return services["gigachat_connection"].status()


@router.post("/probe", response_model=GigaChatTransportProbeResponse)
def probe(req: GigaChatTransportProbeRequest, services=Depends(get_service_container)):
    return services["gigachat_connection"].probe(req)


@router.get("/lab/settings", response_model=GigaChatLabSettingsResponse)
def lab_settings(authorization: str | None = Header(default=None), services=Depends(get_service_container)):
    user = _optional_user(services, authorization)
    return services["gigachat_lab"].get_settings()


@router.post("/lab/settings", response_model=GigaChatLabSettingsResponse)
def save_lab_settings(req: GigaChatLabSettingsUpdateRequest, authorization: str | None = Header(default=None), services=Depends(get_service_container)):
    user = _optional_user(services, authorization)
    return services["gigachat_lab"].save_settings(req)


@router.get("/lab/settings/versions", response_model=GigaChatLabSettingsVersionsResponse)
def list_lab_settings_versions(authorization: str | None = Header(default=None), services=Depends(get_service_container)):
    user = _optional_user(services, authorization)
    return services["gigachat_lab"].list_settings_versions(user=user)


@router.post("/lab/settings/versions", response_model=GigaChatLabSettingsVersionResponse)
def create_lab_settings_version(req: GigaChatLabSettingsVersionCreateRequest, authorization: str | None = Header(default=None), services=Depends(get_service_container)):
    try:
        user = _optional_user(services, authorization)
        return services["gigachat_lab"].create_settings_version(req, user=user)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/lab/settings/versions/{version_id}", response_model=GigaChatLabSettingsVersionResponse)
def get_lab_settings_version(version_id: str, authorization: str | None = Header(default=None), services=Depends(get_service_container)):
    try:
        user = _optional_user(services, authorization)
        return services["gigachat_lab"].get_settings_version(version_id, user=user)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/lab/settings/versions/{version_id}", response_model=GigaChatLabSettingsVersionResponse)
def save_lab_settings_version(version_id: str, req: GigaChatLabSettingsVersionUpdateRequest, authorization: str | None = Header(default=None), services=Depends(get_service_container)):
    try:
        user = _optional_user(services, authorization)
        return services["gigachat_lab"].save_settings_version(version_id, req, user=user)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/lab/settings/versions/{version_id}", response_model=GigaChatLabSettingsVersionsResponse)
def delete_lab_settings_version(version_id: str, authorization: str | None = Header(default=None), services=Depends(get_service_container)):
    try:
        user = _optional_user(services, authorization)
        return services["gigachat_lab"].delete_settings_version(version_id, user=user)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/lab/settings/versions/{version_id}/rule-packs/export")
def export_rule_packs(version_id: str, authorization: str | None = Header(default=None), services=Depends(get_service_container)):
    try:
        user = _optional_user(services, authorization)
        filename, content = services["gigachat_lab"].export_rule_packs_workbook(version_id, user=user)
        quoted_name = quote(filename)
        return Response(
            content=content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quoted_name}"},
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/lab/settings/versions/{version_id}/rule-packs/import", response_model=GigaChatLabSettingsVersionResponse)
async def import_rule_packs(version_id: str, file: UploadFile = File(...), authorization: str | None = Header(default=None), services=Depends(get_service_container)):
    try:
        content = await file.read()
        user = _optional_user(services, authorization)
        return services["gigachat_lab"].import_rule_packs_workbook(
            version_id,
            file.filename or "rules.xlsx",
            content,
            user=user,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/lab/final-prompt", response_model=GigaChatFinalPromptResponse)
def final_prompt_preview(req: GigaChatFinalPromptRequest, services=Depends(get_service_container)):
    return services["gigachat_lab"].build_final_prompt(req)


@router.post("/lab/rule-packs/evaluate", response_model=GigaChatRuleEvaluationResponse)
def evaluate_rule_packs(req: GigaChatRuleEvaluationRequest, services=Depends(get_service_container)):
    return services["gigachat_lab"].evaluate_rule_packs(req)


@router.post("/lab/run-row", response_model=GigaChatLabRowRunResponse)
def run_row(req: GigaChatLabRowRunRequest, services=Depends(get_service_container)):
    return services["gigachat_lab"].run_row_prompt(req)


@router.get("/lab/background-tasks", response_model=GigaChatBackgroundTaskListResponse)
def list_background_tasks(services=Depends(get_service_container)):
    return services["gigachat_lab"].list_background_tasks()


@router.post("/lab/background-tasks", response_model=GigaChatBackgroundTaskSummary)
def start_background_task(req: GigaChatBackgroundTaskStartRequest, services=Depends(get_service_container)):
    try:
        return services["gigachat_lab"].start_background_labeling(req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/lab/background-tasks/{task_id}/cancel", response_model=GigaChatBackgroundTaskSummary)
def cancel_background_task(task_id: str, services=Depends(get_service_container)):
    try:
        return services["gigachat_lab"].cancel_background_task(task_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/lab/background-tasks/{task_id}/result", response_model=GigaChatBackgroundTaskResultResponse)
def background_task_result(task_id: str, services=Depends(get_service_container)):
    try:
        return services["gigachat_lab"].load_background_result(task_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/lab/annotated/export")
def export_annotated(req: GigaChatAnnotatedExportRequest, services=Depends(get_service_container)):
    try:
        filename, content = services["gigachat_lab"].export_annotated_workbook(req)
        quoted_name = quote(filename)
        return Response(
            content=content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quoted_name}"},
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/lab/annotated/validation-export")
def export_validation(req: GigaChatAnnotatedExportRequest, services=Depends(get_service_container)):
    try:
        filename, content = services["gigachat_lab"].export_validation_workbook(req)
        quoted_name = quote(filename)
        return Response(
            content=content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quoted_name}"},
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/lab/workbooks/upload", response_model=GigaChatWorkbookUploadResponse)
async def upload_workbook(file: UploadFile = File(...), authorization: str | None = Header(default=None), services=Depends(get_service_container)):
    try:
        content = await file.read()
        user = _optional_user(services, authorization)
        return services["gigachat_lab"].upload_workbook(
            file.filename or "upload.xlsx",
            content,
            user_id=str(user["id"]),
            workspace_id=str(user.get("workspace_id") or "default"),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/lab/workbooks/upload-local", response_model=GigaChatWorkbookUploadResponse)
def upload_local_workbook(req: GigaChatWorkbookLocalUploadRequest, authorization: str | None = Header(default=None), services=Depends(get_service_container)):
    try:
        user = _optional_user(services, authorization)
        return services["gigachat_lab"].upload_local_workbook(
            req.filename,
            user_id=str(user["id"]),
            workspace_id=str(user.get("workspace_id") or "default"),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/lab/workbooks/chunked/start", response_model=GigaChatWorkbookChunkedUploadStartResponse)
def start_chunked_workbook_upload(req: GigaChatWorkbookChunkedUploadStartRequest, authorization: str | None = Header(default=None), services=Depends(get_service_container)):
    try:
        user = _optional_user(services, authorization)
        return services["gigachat_lab"].start_chunked_workbook_upload(
            req,
            user_id=str(user["id"]),
            workspace_id=str(user.get("workspace_id") or "default"),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/lab/workbooks/chunked/{session_id}/chunks/{chunk_index}", response_model=GigaChatWorkbookChunkUploadResponse)
async def upload_workbook_chunk(session_id: str, chunk_index: int, file: UploadFile = File(...), services=Depends(get_service_container)):
    try:
        content = await file.read()
        return services["gigachat_lab"].receive_chunked_workbook_chunk(session_id, chunk_index, content)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/lab/workbooks/chunked/{session_id}/complete", response_model=GigaChatWorkbookChunkedUploadCompleteResponse)
def complete_chunked_workbook_upload(session_id: str, services=Depends(get_service_container)):
    try:
        return services["gigachat_lab"].complete_chunked_workbook_upload(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/lab/workbooks/chunked/{session_id}/cancel", response_model=dict)
def cancel_chunked_workbook_upload(session_id: str, services=Depends(get_service_container)):
    try:
        return services["gigachat_lab"].cancel_chunked_workbook_upload_session(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/lab/workbooks/chunked/tasks/{task_id}", response_model=GigaChatWorkbookUploadTaskResponse)
def get_workbook_upload_task(task_id: str, services=Depends(get_service_container)):
    try:
        return services["gigachat_lab"].get_workbook_upload_task(task_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/lab/workbooks/chunked/tasks/{task_id}/cancel", response_model=GigaChatWorkbookUploadTaskResponse)
def cancel_workbook_upload_task(task_id: str, services=Depends(get_service_container)):
    try:
        return services["gigachat_lab"].cancel_workbook_upload_task(task_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/lab/workbooks/{upload_id}/select-sheet", response_model=GigaChatWorkbookSheetDataResponse)
def select_sheet(upload_id: str, req: GigaChatWorkbookSelectSheetRequest, services=Depends(get_service_container)):
    try:
        return services["gigachat_lab"].load_sheet(upload_id, req)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/lab/workbooks/export-rows")
def export_workbook_rows(req: GigaChatWorkbookRowsExportRequest, services=Depends(get_service_container)):
    try:
        filename, content = services["gigachat_lab"].export_workbook_rows(req)
        quoted_name = quote(filename)
        return Response(
            content=content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quoted_name}"},
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
