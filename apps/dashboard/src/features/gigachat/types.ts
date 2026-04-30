export type GigaChatArtifact = {
  label: string
  path: string
  exists: boolean
}

export type GigaChatTransportName = 'mtls' | 'token'

export type GigaChatTransportStatus = {
  name: GigaChatTransportName
  title: string
  description: string
  active: boolean
  configured: boolean
  ready: boolean
  base_url: string
  oauth_url?: string | null
  artifacts: GigaChatArtifact[]
  message: string
}

export type GigaChatTransportStatusResponse = {
  configured_mode: 'mtls' | 'tls' | 'token'
  model: string
  transports: GigaChatTransportStatus[]
}

export type GigaChatTransportProbeResponse = {
  transport: GigaChatTransportName
  ok: boolean
  base_url: string
  oauth_url?: string | null
  model: string
  message: string
  models: string[]
}

export type GigaChatLabSettingInputType = 'text' | 'textarea' | 'number' | 'boolean' | 'select'

export type GigaChatLabSettingOption = {
  value: string
  label: string
}

export type GigaChatLabSettingField = {
  key: string
  label: string
  input_type: GigaChatLabSettingInputType
  section: string
  help_text?: string | null
  value: string | number | boolean | null
  options: GigaChatLabSettingOption[]
}

export type GigaChatLabSettingsResponse = {
  title: string
  fields: GigaChatLabSettingField[]
  saved_at?: string | null
}

export type GigaChatWorkbookSheetPreview = {
  name: string
  rows_total: number
  column_count: number
  columns: string[]
  preview_rows: Array<Record<string, unknown>>
}

export type GigaChatWorkbookUploadResponse = {
  upload_id: string
  filename: string
  file_format: 'excel' | 'csv'
  sheet_count: number
  sheets: GigaChatWorkbookSheetPreview[]
}

export type GigaChatWorkbookSheetDataResponse = {
  upload_id: string
  filename: string
  file_format: 'excel' | 'csv'
  sheet_name: string
  total_rows: number
  rendered_rows: number
  columns: string[]
  rows: Array<Record<string, unknown>>
}

export type GigaChatFinalPromptResponse = {
  generated_at: string
  source_columns: string[]
  payload: Record<string, unknown>
  saved: boolean
  saved_path?: string | null
}

export type GigaChatLabRowRunResponse = {
  transport: GigaChatTransportName
  request_payload: Record<string, unknown>
  response_raw: string
  response_json?: unknown
  parse_ok: boolean
  request_token_count?: number | null
}
