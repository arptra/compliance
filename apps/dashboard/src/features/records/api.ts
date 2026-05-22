import { apiPost } from '../../lib/api'

export type RecordFilter = {
  column: string
  op: 'eq' | 'ne' | 'contains' | 'in' | 'between' | 'gte' | 'lte'
  value: unknown
}

export type RecordSearchResponse = {
  stage: 'raw' | 'rules' | 'gigachat'
  columns: string[]
  rows: Record<string, unknown>[]
  total: number
  engine: string
}

export type RecordMutationResponse = {
  stage: 'raw' | 'rules' | 'gigachat'
  deleted_rows: number
  affected_files: number
  message: string
}

export const GIGACHAT_LAKE_IMPORT_STORAGE_KEY = 'gigachat-lake-import-selection'

export type GigaChatLakeImportPayload = {
  upload_id: string
  filename: string
  file_format: 'csv'
  sheet_name: string
  total_rows: number
  columns: string[]
  rows: Record<string, unknown>[]
  source: {
    stage: 'raw' | 'rules' | 'gigachat'
    imported_at: string
  }
}

export function searchRecords(payload: {
  stage: 'raw' | 'rules' | 'gigachat'
  filters?: RecordFilter[]
  columns?: string[]
  limit?: number
  offset?: number
}) {
  return apiPost<RecordSearchResponse>('/api/records/search', {
    filters: [],
    columns: [],
    limit: 100,
    offset: 0,
    ...payload,
  })
}

export function deleteRecords(payload: {
  stage: 'raw' | 'rules' | 'gigachat'
  record_ids: string[]
}) {
  return apiPost<RecordMutationResponse>('/api/records/delete', payload)
}

export function clearRecords(payload: {
  stage: 'raw' | 'rules' | 'gigachat'
}) {
  return apiPost<RecordMutationResponse>('/api/records/clear', payload)
}
