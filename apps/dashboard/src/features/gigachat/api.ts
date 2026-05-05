import { apiGet, apiPost, apiPostBlob, apiPostForm } from '../../lib/api'
import type {
  GigaChatFinalPromptResponse,
  GigaChatLabRowRunResponse,
  GigaChatLabSettingsResponse,
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

export function uploadGigaChatWorkbook(form: FormData) {
  return apiPostForm<GigaChatWorkbookUploadResponse>('/api/gigachat/lab/workbooks/upload', form)
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

export function saveGigaChatFinalPrompt(values: Record<string, unknown>, columns: string[]) {
  return apiPost<GigaChatFinalPromptResponse>('/api/gigachat/lab/final-prompt/save', { values, columns })
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

export function exportGigaChatAnnotatedWorkbook(
  filename: string,
  sheetName: string,
  sourceColumns: string[],
  rows: Array<{
    row_index?: number | null
    classification: string
    tags: string[]
    source_row: Record<string, unknown>
  }>,
) {
  return apiPostBlob('/api/gigachat/lab/annotated/export', {
    filename,
    sheet_name: sheetName,
    source_columns: sourceColumns,
    rows,
  })
}
