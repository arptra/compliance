from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile

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
    GigaChatWorkbookSelectSheetRequest,
    GigaChatWorkbookSheetDataResponse,
    GigaChatWorkbookUploadResponse,
)

router = APIRouter(prefix="/api/gigachat", tags=["gigachat"])


@router.get("/status", response_model=GigaChatTransportStatusResponse)
def status(services=Depends(get_service_container)):
    return services["gigachat_connection"].status()


@router.post("/probe", response_model=GigaChatTransportProbeResponse)
def probe(req: GigaChatTransportProbeRequest, services=Depends(get_service_container)):
    return services["gigachat_connection"].probe(req)


@router.get("/lab/settings", response_model=GigaChatLabSettingsResponse)
def lab_settings(services=Depends(get_service_container)):
    return services["gigachat_lab"].get_settings()


@router.post("/lab/settings", response_model=GigaChatLabSettingsResponse)
def save_lab_settings(req: GigaChatLabSettingsUpdateRequest, services=Depends(get_service_container)):
    return services["gigachat_lab"].save_settings(req)


@router.get("/lab/settings/versions", response_model=GigaChatLabSettingsVersionsResponse)
def list_lab_settings_versions(services=Depends(get_service_container)):
    return services["gigachat_lab"].list_settings_versions()


@router.post("/lab/settings/versions", response_model=GigaChatLabSettingsVersionResponse)
def create_lab_settings_version(req: GigaChatLabSettingsVersionCreateRequest, services=Depends(get_service_container)):
    try:
        return services["gigachat_lab"].create_settings_version(req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/lab/settings/versions/{version_id}", response_model=GigaChatLabSettingsVersionResponse)
def get_lab_settings_version(version_id: str, services=Depends(get_service_container)):
    try:
        return services["gigachat_lab"].get_settings_version(version_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/lab/settings/versions/{version_id}", response_model=GigaChatLabSettingsVersionResponse)
def save_lab_settings_version(version_id: str, req: GigaChatLabSettingsVersionUpdateRequest, services=Depends(get_service_container)):
    try:
        return services["gigachat_lab"].save_settings_version(version_id, req)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/lab/final-prompt", response_model=GigaChatFinalPromptResponse)
def final_prompt_preview(req: GigaChatFinalPromptRequest, services=Depends(get_service_container)):
    return services["gigachat_lab"].build_final_prompt(req, save_snapshot=False)


@router.post("/lab/final-prompt/save", response_model=GigaChatFinalPromptResponse)
def save_final_prompt(req: GigaChatFinalPromptRequest, services=Depends(get_service_container)):
    return services["gigachat_lab"].build_final_prompt(req, save_snapshot=True)


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
async def upload_workbook(file: UploadFile = File(...), services=Depends(get_service_container)):
    try:
        content = await file.read()
        return services["gigachat_lab"].upload_workbook(file.filename or "upload.xlsx", content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/lab/workbooks/{upload_id}/select-sheet", response_model=GigaChatWorkbookSheetDataResponse)
def select_sheet(upload_id: str, req: GigaChatWorkbookSelectSheetRequest, services=Depends(get_service_container)):
    try:
        return services["gigachat_lab"].load_sheet(upload_id, req)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
