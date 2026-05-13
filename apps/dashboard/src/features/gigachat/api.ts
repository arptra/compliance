import { apiGet, apiGetWithProgress, apiPost, apiPostBlob, apiPostForm } from '../../lib/api'
import type {
  GigaChatFinalPromptResponse,
  GigaChatBackgroundTaskListResponse,
  GigaChatBackgroundTaskResultResponse,
  GigaChatBackgroundTaskSummary,
  GigaChatLabRowRunResponse,
  GigaChatRuleEvaluationResponse,
  GigaChatLabSettingsResponse,
  GigaChatLabSettingsVersionResponse,
  GigaChatLabSettingsVersionsResponse,
  GigaChatSettingsVersionStatus,
  GigaChatTransportName,
  GigaChatTransportProbeResponse,
  GigaChatTransportStatusResponse,
  GigaChatWorkbookSheetDataResponse,
  GigaChatWorkbookUploadResponse,
} from './types'

export function getGigaChatStatus() {
  return apiGet<GigaChatTransportStatusResponse>('/api/gigachat/status')
}

export function probeGigaChatTransport(transport: GigaChatTransportName) {
  return apiPost<GigaChatTransportProbeResponse>('/api/gigachat/probe', { transport })
}

export function getGigaChatLabSettings() {
  return apiGet<GigaChatLabSettingsResponse>('/api/gigachat/lab/settings')
}

export function saveGigaChatLabSettings(values: Record<string, unknown>) {
  return apiPost<GigaChatLabSettingsResponse>('/api/gigachat/lab/settings', { values })
}

export function listGigaChatLabSettingsVersions() {
  return apiGet<GigaChatLabSettingsVersionsResponse>('/api/gigachat/lab/settings/versions')
}

export function getGigaChatLabSettingsVersion(versionId: string) {
  return apiGet<GigaChatLabSettingsVersionResponse>(`/api/gigachat/lab/settings/versions/${encodeURIComponent(versionId)}`)
}

export function createGigaChatLabSettingsVersion(payload: {
  title: string
  version_id?: string | null
  description: string
  status: GigaChatSettingsVersionStatus
  created_by: string
  base_version_id: string
}) {
  return apiPost<GigaChatLabSettingsVersionResponse>('/api/gigachat/lab/settings/versions', payload)
}

export function saveGigaChatLabSettingsVersion(versionId: string, payload: {
  title?: string
  description?: string
  status?: GigaChatSettingsVersionStatus
  updated_by?: string
  values: Record<string, unknown>
}) {
  return apiPost<GigaChatLabSettingsVersionResponse>(`/api/gigachat/lab/settings/versions/${encodeURIComponent(versionId)}`, payload)
}

export function uploadGigaChatWorkbook(form: FormData, signal?: AbortSignal) {
  return apiPostForm<GigaChatWorkbookUploadResponse>('/api/gigachat/lab/workbooks/upload', form, { signal })
}

export function uploadLocalGigaChatWorkbook(filename: string) {
  return apiPost<GigaChatWorkbookUploadResponse>('/api/gigachat/lab/workbooks/upload-local', { filename })
}

export function selectGigaChatWorkbookSheet(uploadId: string, sheetName: string, rowLimit = 200) {
  return apiPost<GigaChatWorkbookSheetDataResponse>(`/api/gigachat/lab/workbooks/${encodeURIComponent(uploadId)}/select-sheet`, {
    sheet_name: sheetName,
    row_limit: rowLimit,
  })
}

export function previewGigaChatFinalPrompt(values: Record<string, unknown>, columns: string[]) {
  return apiPost<GigaChatFinalPromptResponse>('/api/gigachat/lab/final-prompt', { values, columns })
}

export function evaluateGigaChatRulePacks(values: Record<string, unknown>, rows: Array<Record<string, unknown>>) {
  return apiPost<GigaChatRuleEvaluationResponse>('/api/gigachat/lab/rule-packs/evaluate', { values, rows })
}

export function runGigaChatWorkbookRow(
  transport: GigaChatTransportName,
  values: Record<string, unknown>,
  columns: string[],
  row: Record<string, unknown>,
  payloadOverride?: Record<string, unknown> | null,
  countTokens = false,
) {
  return apiPost<GigaChatLabRowRunResponse>('/api/gigachat/lab/run-row', {
    transport,
    values,
    columns,
    row,
    payload_override: payloadOverride ?? null,
    count_tokens: countTokens,
  })
}

export function listGigaChatBackgroundTasks() {
  return apiGet<GigaChatBackgroundTaskListResponse>('/api/gigachat/lab/background-tasks')
}

export function startGigaChatBackgroundTask(payload: {
  transport: GigaChatTransportName
  values: Record<string, unknown>
  columns: string[]
  rows: Array<{ row_index: number; source_row: Record<string, unknown> }>
  filename: string
  sheet_name: string
  payload_override?: Record<string, unknown> | null
  count_tokens?: boolean
}) {
  return apiPost<GigaChatBackgroundTaskSummary>('/api/gigachat/lab/background-tasks', payload)
}

export function cancelGigaChatBackgroundTask(taskId: string) {
  return apiPost<GigaChatBackgroundTaskSummary>(`/api/gigachat/lab/background-tasks/${encodeURIComponent(taskId)}/cancel`, {})
}

export function getGigaChatBackgroundTaskResult(taskId: string) {
  return apiGet<GigaChatBackgroundTaskResultResponse>(`/api/gigachat/lab/background-tasks/${encodeURIComponent(taskId)}/result`)
}

export function getGigaChatBackgroundTaskResultWithProgress(
  taskId: string,
  onProgress: (loadedBytes: number, totalBytes: number | null) => void,
) {
  return apiGetWithProgress<GigaChatBackgroundTaskResultResponse>(
    `/api/gigachat/lab/background-tasks/${encodeURIComponent(taskId)}/result`,
    onProgress,
  )
}

export function exportGigaChatAnnotatedWorkbook(
  filename: string,
  sheetName: string,
  sourceColumns: string[],
  rows: Array<{
    row_index?: number | null
    classification: string
    tags: string[]
    local_tags?: string[]
    model_added_tags?: string[]
    model_rejected_tags?: string[]
    match_type?: string | null
    evidence?: string | null
    tag_decisions?: string | null
    rule_hits?: string[]
    suggested_topics?: string[]
    confirmed_rule_hits?: string[]
    rejected_rule_hits?: string[]
    rule_decision?: string | null
    model_decision?: string | null
    reclassified_topic?: string | null
    final_topic?: string | null
    decision_source?: string | null
    source_row: Record<string, unknown>
  }>,
  onProgress?: (loadedBytes: number, totalBytes: number | null) => void,
) {
  return apiPostBlob('/api/gigachat/lab/annotated/export', {
    filename,
    sheet_name: sheetName,
    source_columns: sourceColumns,
    rows,
  }, { onProgress })
}

export function exportGigaChatValidationWorkbook(
  filename: string,
  sheetName: string,
  sourceColumns: string[],
  rows: Array<{
    row_index?: number | null
    classification: string
    tags: string[]
    local_tags?: string[]
    model_added_tags?: string[]
    model_rejected_tags?: string[]
    match_type?: string | null
    evidence?: string | null
    tag_decisions?: string | null
    rule_hits?: string[]
    suggested_topics?: string[]
    confirmed_rule_hits?: string[]
    rejected_rule_hits?: string[]
    rule_decision?: string | null
    model_decision?: string | null
    reclassified_topic?: string | null
    final_topic?: string | null
    decision_source?: string | null
    source_row: Record<string, unknown>
  }>,
  onProgress?: (loadedBytes: number, totalBytes: number | null) => void,
) {
  return apiPostBlob('/api/gigachat/lab/annotated/validation-export', {
    filename,
    sheet_name: sheetName,
    source_columns: sourceColumns,
    rows,
  }, { onProgress })
}
