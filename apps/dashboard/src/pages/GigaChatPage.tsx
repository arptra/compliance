import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  completeGigaChatWorkbookChunkedUpload,
  cancelGigaChatWorkbookChunkedUploadSession,
  cancelGigaChatWorkbookUploadTask,
  getGigaChatBackgroundTaskResultWithProgress,
  getGigaChatWorkbookUploadTask,
  createGigaChatLabSettingsVersion,
  deleteGigaChatLabSettingsVersion,
  exportGigaChatAnnotatedWorkbook,
  exportGigaChatValidationWorkbook,
  exportGigaChatWorkbookRows,
  getGigaChatLabSettingsVersion,
  getGigaChatStatus,
  importGigaChatReclassificationRules,
  listGigaChatLabSettingsVersions,
  probeGigaChatTransport,
  previewGigaChatFinalPrompt,
  runGigaChatWorkbookRow,
  saveGigaChatLabSettingsVersion,
  selectGigaChatWorkbookSheet,
  startGigaChatBackgroundTask,
  startGigaChatWorkbookChunkedUpload,
  uploadGigaChatWorkbookChunk,
  uploadLocalGigaChatWorkbook,
  uploadGigaChatWorkbook,
} from '../features/gigachat/api'
import { GigaChatProcessingOverlay } from '../features/gigachat/GigaChatProcessingOverlay'
import { evaluateRulePacksLocally, parseRulePacks } from '../features/gigachat/rulePackMatcher'
import { RulePackEditor } from '../features/gigachat/RulePackEditor'
import { GigaChatSettingsForm } from '../features/gigachat/GigaChatSettingsForm'
import { WorkbookSheetPickerModal, WorkbookSheetsPanel } from '../features/gigachat/WorkbookSheetPickerModal'
import { WorkbookSheetTable } from '../features/gigachat/WorkbookSheetTable'
import { WorkbookStatsPanel } from '../features/gigachat/WorkbookStatsPanel'
import { GigaChatTransportCard } from '../features/gigachat/GigaChatTransportCard'
import { CellHoverPopover, useCellHoverPopover } from '../features/gigachat/useCellHoverPopover'
import { useResizableTable, type TableRowClamp } from '../features/gigachat/useResizableTable'
import { useVirtualTableRows } from '../features/gigachat/useVirtualTableRows'
import { GIGACHAT_LAKE_IMPORT_STORAGE_KEY, type GigaChatLakeImportPayload } from '../features/records/api'
import { useAuth } from '../features/auth/AuthContext'
import type {
  GigaChatFinalPromptResponse,
  GigaChatBackgroundTaskResultResponse,
  GigaChatLabSettingsVersionsResponse,
  GigaChatLabRowRunResponse,
  GigaChatRuleEvaluationResponse,
  GigaChatRulePack,
  GigaChatSettingsVersionStatus,
  GigaChatSettingsVersionVisibility,
  GigaChatRuleEvaluationRow,
  GigaChatTransportName,
  GigaChatWorkbookSheetDataResponse,
  GigaChatWorkbookUploadTaskResponse,
  GigaChatWorkbookUploadResponse,
} from '../features/gigachat/types'

type WorkbookRowLimit = 10 | 20 | 100 | 'all'
type GigaChatLabTab = 'workspace' | 'settings'
type LabSetupTab = 'prompts' | 'rules' | 'reclassification'
type ReclassificationRule = {
  name: string
  source_field: string
  context_field: string
  context_fields: string[]
  prompt: string
}
type ReclassificationDraft = {
  editingIndex: number | null
  name: string
  sourceField: string
  contextFields: string[]
  prompt: string
}
const WORKBOOK_UPLOAD_TIMEOUT_MS = 15_000
const CHUNKED_UPLOAD_THRESHOLD_BYTES = 8 * 1024 * 1024
const CHUNKED_UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024
const CHUNKED_UPLOAD_CONCURRENCY = 4
const CHUNKED_UPLOAD_POLL_MS = 1000
const RULE_PACK_EXCLUSION_SETTING_KEY = 'rule_pack_exclusion_notes'
const SETTINGS_VERSION_STORAGE_KEY = 'gigachat_lab_settings_profile_version_id'
const RULE_EVALUATION_CHUNK_COUNT = 25
const DEFAULT_RECLASSIFICATION_PROMPT = ''
const RECLASSIFICATION_DESCRIPTION_PLACEHOLDER = 'Например: обращения про задержку очередного транша по образовательному кредиту, оплату семестра или проблемы с учебным периодом.'
const EMPTY_RECLASSIFICATION_MARKERS = new Set([
  '-',
  '—',
  '–',
  'нет',
  'нет изменений',
  'без изменений',
  'не менять',
  'оставить',
  'оставить как есть',
  'none',
  'null',
  'nil',
  'n/a',
  'na',
  'no change',
  'no_change',
  'same',
  'same topic',
  'same_topic',
])
const VERSION_STATUS_LABELS: Record<GigaChatSettingsVersionStatus, string> = {
  draft: 'Черновая',
  test: 'Тестовая',
  working: 'Рабочая',
  release: 'Релизная',
  archived: 'Архивная',
}
const VERSION_VISIBILITY_LABELS: Record<GigaChatSettingsVersionVisibility, string> = {
  private: 'Приватная',
  public: 'Публичная',
}
const WORKBOOK_UPLOAD_TASK_STATUS_LABELS: Record<GigaChatWorkbookUploadTaskResponse['status'], string> = {
  queued: 'В очереди',
  running: 'В работе',
  completed: 'Готово',
  cancelled: 'Отменено',
  failed: 'Ошибка',
}

function nextFrame() {
  return new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()))
}

function formatBytes(bytes: number) {
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} МБ`
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} КБ`
  return `${bytes} Б`
}

function sleep(ms: number) {
  return new Promise<void>((resolve) => window.setTimeout(resolve, ms))
}

function canUseLocalWorkbookFallback() {
  const localHosts = new Set(['localhost', '127.0.0.1', '::1'])
  const pageHost = window.location.hostname
  let apiHost = pageHost
  const configuredBase = import.meta.env.VITE_API_BASE_URL || ''
  if (configuredBase) {
    try {
      apiHost = new URL(configuredBase, window.location.href).hostname
    } catch {
      apiHost = ''
    }
  }
  return localHosts.has(pageHost) && localHosts.has(apiHost)
}

async function evaluateRulePacksWithProgress(
  values: Record<string, unknown>,
  rows: Array<Record<string, unknown>>,
  onProgress: (processed: number, total: number) => void,
): Promise<GigaChatRuleEvaluationResponse> {
  const total = rows.length
  const chunkSize = Math.max(1, Math.ceil(total / RULE_EVALUATION_CHUNK_COUNT))
  const evaluations: GigaChatRuleEvaluationRow[] = []
  let rulePacks: GigaChatRuleEvaluationResponse['rule_packs'] = []

  onProgress(0, total)
  await nextFrame()
  for (let start = 0; start < total; start += chunkSize) {
    const chunk = rows.slice(start, start + chunkSize)
    const chunkResult = evaluateRulePacksLocally(values, chunk)
    if (!rulePacks.length) rulePacks = chunkResult.rule_packs
    evaluations.push(...chunkResult.evaluations.map((item) => ({ ...item, row_index: item.row_index + start })))
    onProgress(Math.min(total, start + chunk.length), total)
    await nextFrame()
  }

  return { rule_packs: rulePacks, evaluations }
}

type AnnotatedSheetRow = {
  rowKey: string
  rowIndex: number
  classification: string
  newClassification: string
  sourceClassification: string
  isReclassified: boolean
  tags: string[]
  localTags: string[]
  modelAddedTags: string[]
  modelRejectedTags: string[]
  matchType: string
  evidence: string
  tagDecisions: string
  ruleHits: string[]
  suggestedTopics: string[]
  confirmedRuleHits: string[]
  rejectedRuleHits: string[]
  ruleDecision: string
  modelDecision: string
  reclassifiedTopic: string
  finalTopic: string
  decisionSource: string
  responseRaw: string
  responseJson?: unknown
  sourceRow: Record<string, unknown>
}

type ProcessingOverlayState = {
  title: string
  completed: number
  total: number
  minimized: boolean
  currentLabel: string
}

type WorkbookUploadProgressState = {
  phase: string
  message: string
  progress: number
  uploadedBytes: number
  totalBytes: number
  receivedChunks: number
  totalChunks: number
  taskId?: string
  sessionId?: string
  filename?: string
}

type TokenAccountingState = {
  enabled: boolean
  pricePer1k: number
  totalTokens: number
}

type BackgroundWorkbookUpload = {
  taskId: string
  sessionId: string
  filename: string
  createdAt: string
}

type RowRunResultPage = {
  id: 'classification' | 'reclassification'
  title: string
  description: string
  result: GigaChatLabRowRunResponse
}

type ReclassificationSaveStatus = {
  type: 'idle' | 'saving' | 'saved' | 'error'
  message: string
}

type ReclassificationRuleFieldStatus = {
  valid: boolean
  missing: string[]
}

type RuleGuardIssue = {
  code: string
  missingFields: string[]
}

type PendingRuleGuardAction = (values: Record<string, unknown>) => void | Promise<void>

const TOKEN_ACCOUNTING_STORAGE_KEY = 'gigachat-lab-token-accounting'
const ASYNC_WORKERS_STORAGE_KEY = 'gigachat-lab-async-workers'
const WORKBOOK_UPLOAD_BACKGROUND_STORAGE_KEY = 'gigachat-lab-background-workbook-uploads'
const DEFAULT_ASYNC_WORKERS = 4
const MAX_ASYNC_WORKERS = 32
const ASYNC_WORKER_SLIDER_MAX = 16
const DEFAULT_RULE_PACK_PROMPT_NOTES = JSON.stringify([
  {
    code: 'DRA',
    description: 'Проставляет тег DRA/ДРПА по словам про смерть, наследство, каникулы, реструктуризацию, приставов, СВО, суд, исполнительное производство, военный контур и банкротство.',
    enabled: true,
    type: 'assign_tag',
    source_fields: ['Во. Описание', 'Обр. Результат суммаризации диалога'],
    keywords: [
      'умер',
      'погиб',
      'смерт',
      'гибел',
      'наследни',
      'наследств',
      'каникул',
      'реструктуриз',
      'пристав',
      'участник СВО',
      'участника СВО',
      'участником СВО',
      'на СВО',
      'судебное решение',
      'по решению суда',
      'исполнительное производство',
      'военн',
      'банкрот',
    ],
    filters: [],
    target_tag: 'DRA',
    target_topic: null,
  },
  {
    code: 'IPOTEKA',
    description: 'Проставляет тег ИПОТЕКА по обращениям про ипотеку, жилищный кредит, недвижимость в залоге, закладную, эскроу, обременение и ипотечное рефинансирование.',
    enabled: true,
    type: 'assign_tag',
    source_fields: ['Во. Описание', 'Обр. Результат суммаризации диалога'],
    keywords: [
      'ипотек',
      'жилищн',
      'недвижим',
      'квартир',
      'дом в залог',
      'залог недвиж',
      'закладн',
      'эскроу',
      'обременен',
      'созаемщик',
      'созаемщ',
      'рефинансир*ипот',
      'рефинансирован*ипот',
      'материнск*капитал',
      'первоначальн*взнос',
    ],
    filters: [],
    target_tag: 'ИПОТЕКА',
    target_topic: null,
  },
  {
    code: 'EDU_RECLASS_TRANCH',
    description: 'Переклассифицирует обращения в тему про очередной транш по образовательному кредиту.',
    enabled: true,
    type: 'reclass_topic',
    source_fields: ['Во. Описание', 'Обр. Результат суммаризации диалога'],
    keywords: ['транш', 'семестр'],
    filters: [
      { field: 'Трайб', op: 'eq', value: 'ПОТРЕБИТЕЛЬСКИЕ КРЕДИТЫ' },
      { field: 'драйвер', op: 'ne', value: 'ОБРАЗОВАТЕЛЬНЫЙ КРЕДИТ' },
    ],
    target_tag: null,
    target_topic: 'Проблема с выдачей очередного транша по Образовательному кредиту',
  },
  {
    code: 'EDU_RECLASS_APPLICATION',
    description: 'Переклассифицирует обращения в тему про зачисление средств, оформление или рассмотрение заявки по образовательному кредиту.',
    enabled: true,
    type: 'reclass_topic',
    source_fields: ['Во. Описание', 'Обр. Результат суммаризации диалога'],
    keywords: ['Образовательн', 'Вуз', 'Кредит на образ', 'Оплатить обучение', 'Период*обучения', 'Отчисл'],
    filters: [
      { field: 'Трайб', op: 'eq', value: 'ПОТРЕБИТЕЛЬСКИЕ КРЕДИТЫ' },
      { field: 'драйвер', op: 'ne', value: 'ОБРАЗОВАТЕЛЬНЫЙ КРЕДИТ' },
    ],
    target_tag: null,
    target_topic: 'Проблемы с зачислением средств/ оформлением-рассмотрением заявки',
  },
], null, 2)

function fieldsToValues(fields: Array<{ key: string; value: unknown }>) {
  return Object.fromEntries(fields.map((field) => [field.key, field.value]))
}

function stringifyJson(value: unknown) {
  return JSON.stringify(value, null, 2)
}

function splitReclassificationFieldList(value: unknown) {
  if (Array.isArray(value)) {
    return Array.from(new Set(value.map((item) => String(item).trim()).filter(Boolean)))
  }
  if (value === null || value === undefined) return []
  return Array.from(new Set(String(value)
    .replace(/\r/g, '\n')
    .split('\n')
    .flatMap((chunk) => chunk.split(','))
    .map((item) => item.trim())
    .filter(Boolean)))
}

function normalizeReclassificationRule(value: unknown): ReclassificationRule | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const item = value as Record<string, unknown>
  const name = String(item.name ?? item.title ?? item.code ?? '').trim()
  const sourceField = String(item.source_field ?? item.sourceField ?? '').trim()
  const contextFields = splitReclassificationFieldList(item.context_fields ?? item.contextFields)
  const legacyContextField = String(item.context_field ?? item.contextField ?? '').trim()
  if (legacyContextField && !contextFields.includes(legacyContextField)) contextFields.unshift(legacyContextField)
  const contextField = contextFields[0] ?? ''
  const prompt = String(item.prompt ?? item.description ?? '').trim()
  if (!sourceField && !contextFields.length) return null
  return {
    name,
    source_field: sourceField,
    context_field: contextField,
    context_fields: contextFields,
    prompt,
  }
}

function parseReclassificationRules(value: unknown, fallbackValues?: Record<string, unknown>): ReclassificationRule[] {
  const rawItems: unknown[] = []
  let explicitRulesConfig = false
  if (Array.isArray(value)) {
    explicitRulesConfig = true
    rawItems.push(...value)
  } else if (typeof value === 'string' && value.trim()) {
    explicitRulesConfig = true
    try {
      const parsed = JSON.parse(value) as unknown
      if (Array.isArray(parsed)) rawItems.push(...parsed)
      else rawItems.push(parsed)
    } catch {
      rawItems.length = 0
    }
  } else if (value && typeof value === 'object') {
    explicitRulesConfig = true
    rawItems.push(value)
  }

  const parsedItems = rawItems
    .map((item) => normalizeReclassificationRule(item))
    .filter((item): item is ReclassificationRule => Boolean(item))
  if (parsedItems.length) return parsedItems
  if (explicitRulesConfig) return []

  const sourceField = String(fallbackValues?.reclassification_source_field ?? '').trim()
  const contextField = String(fallbackValues?.reclassification_context_field ?? '').trim()
  if (!sourceField && !contextField) return []
  return [{
    name: '',
    source_field: sourceField,
    context_field: contextField,
    context_fields: contextField ? [contextField] : [],
    prompt: String(fallbackValues?.reclassification_prompt ?? '').trim(),
  }]
}

function serializeReclassificationRules(items: ReclassificationRule[]) {
  return JSON.stringify(items, null, 2)
}

function formatReclassificationRuleTitle(rule: ReclassificationRule) {
  const contextFields = getReclassificationRuleContextFields(rule)
  return rule.name || `${rule.source_field || 'Без поля темы'} → ${contextFields.join(', ') || 'без поля контекста'}`
}

function getReclassificationRuleContextFields(rule: ReclassificationRule) {
  return rule.context_fields?.length ? rule.context_fields : splitReclassificationFieldList(rule.context_field)
}

function canRunReclassificationRule(rule: ReclassificationRule) {
  const contextFields = getReclassificationRuleContextFields(rule)
  return Boolean(
    String(rule.name ?? '').trim()
    && String(rule.prompt ?? '').trim()
    && (String(rule.source_field ?? '').trim() || contextFields.length),
  )
}

function getReclassificationRuleFieldStatus(
  rule: ReclassificationRule,
  availableFields?: string[] | null,
): ReclassificationRuleFieldStatus {
  if (!availableFields) return { valid: true, missing: [] }
  const available = new Set(availableFields)
  const configuredFields = [rule.source_field, ...getReclassificationRuleContextFields(rule)]
    .map((field) => field.trim())
    .filter(Boolean)
  const missing = configuredFields.filter((field) => !available.has(field))
  return { valid: missing.length === 0, missing }
}

function canRunReclassificationRuleOnSheet(rule: ReclassificationRule, availableFields?: string[] | null) {
  return canRunReclassificationRule(rule) && getReclassificationRuleFieldStatus(rule, availableFields).valid
}

function withRunnableReclassificationRules(values: Record<string, unknown>, availableFields?: string[] | null) {
  const rules = parseReclassificationRules(values.reclassification_prompt_notes, values)
    .filter((rule) => canRunReclassificationRuleOnSheet(rule, availableFields))
  return {
    ...values,
    reclassification_prompt_notes: serializeReclassificationRules(rules),
  }
}

function withoutReclassificationSettings(values: Record<string, unknown>) {
  return {
    ...values,
    reclassification_prompt_notes: '[]',
    reclassification_source_field: '',
    reclassification_context_field: '',
    reclassification_prompt: DEFAULT_RECLASSIFICATION_PROMPT,
  }
}

function buildSheetRowKey(uploadId: string, sheetName: string, rowIndex: number) {
  return `${uploadId}:${sheetName}:${rowIndex}`
}

function getBlockingRuleMissingFields(rule: GigaChatRulePack, columns: string[]) {
  const available = new Set(columns)
  if (!rule.source_fields.length) return ['Source fields не выбраны']
  const missingSourceFields = Array.from(new Set(rule.source_fields.filter((field) => !available.has(field))))
  const hasAvailableSourceField = rule.source_fields.some((field) => available.has(field))
  const missingFilterFields = Array.from(new Set(
    rule.filters
      .map((filterItem) => filterItem.field)
      .filter(Boolean)
      .filter((field) => !available.has(field)),
  ))
  return [
    ...(!hasAvailableSourceField ? missingSourceFields : []),
    ...missingFilterFields,
  ]
}

function findRuleGuardIssues(values: Record<string, unknown>, columns: string[]) {
  if (!columns.length) return [] as RuleGuardIssue[]
  return parseRulePacks(values.rule_pack_prompt_notes)
    .flatMap((rule) => {
      const missingFields = getBlockingRuleMissingFields(rule, columns)
      return missingFields.length ? [{ code: rule.code || 'Без кода', missingFields }] : []
    })
}

function disableRulesWithMissingFields(values: Record<string, unknown>, columns: string[]) {
  const available = new Set(columns)
  const rules = parseRulePacks(values.rule_pack_prompt_notes)
  const nextRules = rules.map((rule): GigaChatRulePack => {
    if (!rule.source_fields.length) return rule.enabled ? { ...rule, enabled: false } : rule
    const hasAvailableSourceField = rule.source_fields.some((field) => available.has(field))
    const missingFilter = rule.filters
      .map((filterItem) => filterItem.field)
      .filter(Boolean)
      .some((field) => !available.has(field))
    const missing = !hasAvailableSourceField || missingFilter
    return rule.enabled && missing ? { ...rule, enabled: false } : rule
  })
  return {
    ...values,
    rule_pack_prompt_notes: JSON.stringify(nextRules, null, 2),
  }
}

function parseRulePackExclusions(raw: unknown) {
  const normalize = (item: unknown) => {
    if (item && typeof item === 'object' && !Array.isArray(item)) {
      const record = item as Record<string, unknown>
      return String(record.code ?? record.name ?? record.value ?? '').trim()
    }
    return String(item ?? '').trim()
  }
  const unique = (items: unknown[]) => Array.from(new Set(items.map(normalize).filter(Boolean)))

  if (Array.isArray(raw)) return unique(raw)
  const text = String(raw ?? '').trim()
  if (!text) return []
  try {
    const parsed = JSON.parse(text) as unknown
    if (Array.isArray(parsed)) return unique(parsed)
    if (parsed && typeof parsed === 'object') {
      const record = parsed as Record<string, unknown>
      const items = record.codes ?? record.items ?? record.rule_packs
      return Array.isArray(items) ? unique(items) : unique([record])
    }
    return unique([parsed])
  } catch {
    return unique(text.replace(/\r/gu, '\n').split(/\n|,/u))
  }
}

function serializeRulePackExclusions(codes: string[]) {
  return JSON.stringify(Array.from(new Set(codes.map((code) => code.trim()).filter(Boolean))), null, 2)
}

function withSerializedRulePackExclusions(values: Record<string, unknown>, codesOverride?: string[]) {
  const codes = codesOverride ?? parseRulePackExclusions(values[RULE_PACK_EXCLUSION_SETTING_KEY])
  return {
    ...values,
    [RULE_PACK_EXCLUSION_SETTING_KEY]: serializeRulePackExclusions(codes),
  }
}

function preserveRulePackExclusionValue(serverValues: Record<string, unknown>, submittedValues: Record<string, unknown>) {
  if (Object.prototype.hasOwnProperty.call(serverValues, RULE_PACK_EXCLUSION_SETTING_KEY)) return serverValues
  if (!Object.prototype.hasOwnProperty.call(submittedValues, RULE_PACK_EXCLUSION_SETTING_KEY)) return serverValues
  return {
    ...serverValues,
    [RULE_PACK_EXCLUSION_SETTING_KEY]: submittedValues[RULE_PACK_EXCLUSION_SETTING_KEY],
  }
}

function readStoredSettingsVersionId() {
  if (typeof window === 'undefined') return null
  return window.localStorage.getItem(SETTINGS_VERSION_STORAGE_KEY)
}

function loadStoredSettingsVersionId() {
  return readStoredSettingsVersionId() || ''
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

function loadTokenAccountingState(): TokenAccountingState {
  if (typeof window === 'undefined') {
    return { enabled: false, pricePer1k: 0, totalTokens: 0 }
  }
  try {
    const raw = window.localStorage.getItem(TOKEN_ACCOUNTING_STORAGE_KEY)
    if (!raw) return { enabled: false, pricePer1k: 0, totalTokens: 0 }
    const parsed = JSON.parse(raw) as Partial<TokenAccountingState>
    return {
      enabled: Boolean(parsed.enabled),
      pricePer1k: Number.isFinite(Number(parsed.pricePer1k)) ? Number(parsed.pricePer1k) : 0,
      totalTokens: Number.isFinite(Number(parsed.totalTokens)) ? Number(parsed.totalTokens) : 0,
    }
  } catch {
    return { enabled: false, pricePer1k: 0, totalTokens: 0 }
  }
}

function clampAsyncWorkerCount(value: unknown): number {
  const parsed = Number(value)
  if (!Number.isFinite(parsed)) return DEFAULT_ASYNC_WORKERS
  return Math.max(1, Math.min(MAX_ASYNC_WORKERS, Math.round(parsed)))
}

function loadAsyncWorkerCount(): number {
  if (typeof window === 'undefined') return DEFAULT_ASYNC_WORKERS
  return clampAsyncWorkerCount(window.localStorage.getItem(ASYNC_WORKERS_STORAGE_KEY))
}

function loadBackgroundWorkbookUploads(): BackgroundWorkbookUpload[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = window.localStorage.getItem(WORKBOOK_UPLOAD_BACKGROUND_STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw) as Array<Partial<BackgroundWorkbookUpload>>
    return parsed
      .map((item) => ({
        taskId: String(item.taskId || '').trim(),
        sessionId: String(item.sessionId || '').trim(),
        filename: String(item.filename || 'Фоновая загрузка').trim(),
        createdAt: String(item.createdAt || new Date().toISOString()),
      }))
      .filter((item) => item.taskId)
  } catch {
    return []
  }
}

async function runWithConcurrency<T>(
  items: T[],
  workerCount: number,
  worker: (item: T, index: number) => Promise<void>,
) {
  const parallelism = Math.min(clampAsyncWorkerCount(workerCount), items.length)
  let cursor = 0
  await Promise.all(Array.from({ length: parallelism }, async () => {
    while (cursor < items.length) {
      const index = cursor
      cursor += 1
      await worker(items[index], index)
    }
  }))
}

function extractClassification(responseJson: unknown) {
  if (!responseJson || typeof responseJson !== 'object' || Array.isArray(responseJson)) return ''
  const record = responseJson as Record<string, unknown>
  const directKeys = [
    'final_topic',
    'complaint_category',
    'complaint_subcategory',
    'category',
    'topic',
    'theme',
    'class',
    'тематика',
    'класс',
  ]
  for (const key of directKeys) {
    const value = record[key]
    if (typeof value === 'string' && value.trim()) return value.trim()
  }
  const classification = record.classification
  if (classification && typeof classification === 'object' && !Array.isArray(classification)) {
    const classificationRecord = classification as Record<string, unknown>
    for (const key of ['class', 'category', 'topic', 'theme', 'тематика', 'класс']) {
      const value = classificationRecord[key]
      if (typeof value === 'string' && value.trim()) return value.trim()
    }
  }
  return ''
}

const TAG_VALUE_KEYS = ['tag', 'target_tag', 'name', 'value', 'label', 'title', 'code']
const RULE_HIT_VALUE_KEYS = ['code', 'rule_code', 'rule', 'id', 'tag', 'value', 'label', 'name']

function extractTextValue(value: unknown, keys = TAG_VALUE_KEYS): string {
  if (value === null || value === undefined) return ''
  if (typeof value === 'string') return value.trim()
  if (typeof value === 'number' || typeof value === 'boolean') return String(value).trim()
  if (Array.isArray(value)) {
    return extractStringList(value, keys).join(', ')
  }
  if (typeof value === 'object') {
    const record = value as Record<string, unknown>
    for (const key of keys) {
      const text = extractTextValue(record[key], keys)
      if (text) return text
    }
  }
  return ''
}

function extractStringList(value: unknown, keys = TAG_VALUE_KEYS): string[] {
  if (Array.isArray(value)) {
    return Array.from(new Set(value.flatMap((item) => {
      const text = extractTextValue(item, keys)
      return text ? [text] : []
    })))
  }
  const text = extractTextValue(value, keys)
  return text ? [text] : []
}

function extractTags(responseJson: unknown) {
  if (!responseJson || typeof responseJson !== 'object' || Array.isArray(responseJson)) return [] as string[]
  const record = responseJson as Record<string, unknown>
  const assignedTags = record.assigned_tags
  const directAssignedTags = extractStringList(assignedTags)
  if (directAssignedTags.length) return directAssignedTags
  const tags = record.tags
  return extractStringList(tags)
}

function extractFirstString(record: Record<string, unknown>, keys: string[]) {
  for (const key of keys) {
    const value = record[key]
    if (typeof value === 'string' && value.trim()) return value.trim()
  }
  return ''
}

function extractConfirmedRuleHits(responseJson: unknown) {
  if (!responseJson || typeof responseJson !== 'object' || Array.isArray(responseJson)) return [] as string[]
  return extractStringList((responseJson as Record<string, unknown>).confirmed_rule_hits, RULE_HIT_VALUE_KEYS)
}

function extractRejectedRuleHits(responseJson: unknown) {
  if (!responseJson || typeof responseJson !== 'object' || Array.isArray(responseJson)) return [] as string[]
  return extractStringList((responseJson as Record<string, unknown>).rejected_rule_hits, RULE_HIT_VALUE_KEYS)
}

function extractModelAddedTags(responseJson: unknown) {
  if (!responseJson || typeof responseJson !== 'object' || Array.isArray(responseJson)) return [] as string[]
  const record = responseJson as Record<string, unknown>
  return extractStringList(record.model_added_tags)
}

function extractModelRejectedTags(responseJson: unknown) {
  if (!responseJson || typeof responseJson !== 'object' || Array.isArray(responseJson)) return [] as string[]
  const record = responseJson as Record<string, unknown>
  return [
    ...extractStringList(record.model_rejected_tags),
    ...extractStringList(record.rejected_tags),
  ].filter((item, index, arr) => arr.indexOf(item) === index)
}

function extractTagDecisionSummary(responseJson: unknown) {
  if (!responseJson || typeof responseJson !== 'object' || Array.isArray(responseJson)) return ''
  const record = responseJson as Record<string, unknown>
  const tagDecisions = record.tag_decisions
  if (Array.isArray(tagDecisions)) {
    return tagDecisions.map((item) => {
      if (!item || typeof item !== 'object' || Array.isArray(item)) return extractTextValue(item)
      const decision = item as Record<string, unknown>
      const tag = extractTextValue(decision.tag)
      const status = extractTextValue(decision.decision, ['decision', 'status', 'value', 'label', 'name'])
      const matchType = extractTextValue(decision.match_type, ['match_type', 'type', 'value', 'label', 'name'])
      const evidence = extractTextValue(decision.evidence, ['evidence', 'text', 'value', 'label', 'name'])
      return [tag, status, matchType, evidence ? `"${evidence}"` : ''].filter(Boolean).join(' · ')
    }).filter(Boolean).join('; ')
  }
  return typeof tagDecisions === 'string' ? tagDecisions.trim() : ''
}

function extractMatchType(responseJson: unknown) {
  if (!responseJson || typeof responseJson !== 'object' || Array.isArray(responseJson)) return ''
  const record = responseJson as Record<string, unknown>
  const direct = extractFirstString(record, ['match_type', 'tag_match_type'])
  if (direct) return direct
  const tagDecisions = record.tag_decisions
  if (Array.isArray(tagDecisions)) {
    return Array.from(new Set(tagDecisions.flatMap((item) => {
      if (!item || typeof item !== 'object' || Array.isArray(item)) return []
      const value = (item as Record<string, unknown>).match_type
      return typeof value === 'string' && value.trim() ? [value.trim()] : []
    }))).join(', ')
  }
  return ''
}

function extractEvidence(responseJson: unknown) {
  if (!responseJson || typeof responseJson !== 'object' || Array.isArray(responseJson)) return ''
  const record = responseJson as Record<string, unknown>
  const direct = extractFirstString(record, ['evidence', 'tag_evidence'])
  if (direct) return direct
  const tagDecisions = record.tag_decisions
  if (Array.isArray(tagDecisions)) {
    return tagDecisions.flatMap((item) => {
      if (!item || typeof item !== 'object' || Array.isArray(item)) return []
      const value = (item as Record<string, unknown>).evidence
      return typeof value === 'string' && value.trim() ? [value.trim()] : []
    }).join('; ')
  }
  return ''
}

function normalizeReclassificationTopicValue(value: unknown) {
  if (value === null || value === undefined) return ''
  const text = String(value).trim()
  if (!text) return ''
  const normalized = text.replace(/\s+/g, ' ').toLocaleLowerCase('ru-RU')
  if (EMPTY_RECLASSIFICATION_MARKERS.has(normalized)) return ''
  return text
}

function normalizeReclassificationTopicForCompare(value: unknown) {
  return normalizeReclassificationTopicValue(value).replace(/\s+/g, ' ').toLocaleLowerCase('ru-RU')
}

function extractReclassifiedTopic(responseJson: unknown) {
  if (!responseJson || typeof responseJson !== 'object' || Array.isArray(responseJson)) return ''
  const record = responseJson as Record<string, unknown>
  for (const key of ['reclassified_topic', 'suggested_topic', 'target_topic']) {
    const value = normalizeReclassificationTopicValue(record[key])
    if (value) return value
  }
  return ''
}

function isAnnotatedRowReclassified(row: AnnotatedSheetRow) {
  const newClassification = normalizeReclassificationTopicValue(row.newClassification)
  if (!row.isReclassified || !newClassification) return false
  const newComparable = normalizeReclassificationTopicForCompare(newClassification)
  const sourceComparable = normalizeReclassificationTopicForCompare(row.sourceClassification || row.classification)
  return !sourceComparable || newComparable !== sourceComparable
}

function formatLabError(error: Error | null | undefined, resourceLabel: string) {
  if (!error) return ''
  const raw = String(error.message || '').trim()
  if (error.name === 'AbortError' || raw === 'WORKBOOK_UPLOAD_TIMEOUT') {
    return `Не удалось загрузить ${resourceLabel}: браузер не получил ответ от API за ${Math.round(WORKBOOK_UPLOAD_TIMEOUT_MS / 1000)} секунд. Для больших файлов загрузка должна идти частями; если ошибка повторится, проверьте доступность API и лимиты прокси.`
  }
  try {
    const parsed = JSON.parse(raw) as { detail?: string }
    if (parsed?.detail === 'Not Found') {
      return `Не удалось загрузить ${resourceLabel}: локальный API еще не видит новые lab-endpoint'ы. Обычно это означает, что backend нужно перезапустить.`
    }
    if (parsed?.detail) {
      return `Не удалось загрузить ${resourceLabel}: ${parsed.detail}`
    }
  } catch {
    // Keep raw message fallback below.
  }
  return `Не удалось загрузить ${resourceLabel}: ${raw}`
}

async function uploadWorkbookInChunks(
  file: File,
  onProgress: (state: WorkbookUploadProgressState) => void,
  signal?: AbortSignal,
): Promise<GigaChatWorkbookUploadResponse> {
  const totalChunks = Math.max(1, Math.ceil(file.size / CHUNKED_UPLOAD_CHUNK_BYTES))
  const started = await startGigaChatWorkbookChunkedUpload({
    filename: file.name,
    total_size: file.size,
    chunk_size: CHUNKED_UPLOAD_CHUNK_BYTES,
    total_chunks: totalChunks,
  })
  let nextChunkIndex = 0
  let uploadedBytes = 0

  onProgress({
    phase: 'uploading',
    message: 'Отправляем файл частями',
    progress: 0,
    uploadedBytes: 0,
    totalBytes: file.size,
    receivedChunks: 0,
    totalChunks,
    sessionId: started.session_id,
    filename: file.name,
  })

  const uploadNextChunk = async () => {
    while (nextChunkIndex < totalChunks) {
      if (signal?.aborted) throw new DOMException('UPLOAD_ABORTED', 'AbortError')
      const chunkIndex = nextChunkIndex
      nextChunkIndex += 1
      const chunkStart = chunkIndex * CHUNKED_UPLOAD_CHUNK_BYTES
      const chunkEnd = Math.min(file.size, chunkStart + CHUNKED_UPLOAD_CHUNK_BYTES)
      const chunk = file.slice(chunkStart, chunkEnd)
      const result = await uploadGigaChatWorkbookChunk(started.session_id, chunkIndex, chunk, signal)
      uploadedBytes += chunk.size
      const uploadRatio = file.size ? Math.min(1, uploadedBytes / file.size) : result.received_chunks / totalChunks
      onProgress({
        phase: 'uploading',
        message: `Загружено ${result.received_chunks} из ${totalChunks} частей`,
        progress: uploadRatio * 0.45,
        uploadedBytes,
        totalBytes: file.size,
        receivedChunks: result.received_chunks,
        totalChunks,
        sessionId: started.session_id,
        filename: file.name,
      })
    }
  }

  await Promise.all(Array.from({ length: Math.min(CHUNKED_UPLOAD_CONCURRENCY, totalChunks) }, () => uploadNextChunk()))

  onProgress({
    phase: 'queued',
    message: 'Файл загружен, запускаем обработку на backend',
    progress: 0.46,
    uploadedBytes: file.size,
    totalBytes: file.size,
    receivedChunks: totalChunks,
    totalChunks,
    sessionId: started.session_id,
    filename: file.name,
  })
  const completed = await completeGigaChatWorkbookChunkedUpload(started.session_id)
  onProgress({
    phase: 'background',
    message: 'Файл ушел в backend-задачу, можно скрыть загрузку в фон',
    progress: 0.47,
    uploadedBytes: file.size,
    totalBytes: file.size,
    receivedChunks: totalChunks,
    totalChunks,
    taskId: completed.task_id,
    sessionId: completed.session_id,
    filename: file.name,
  })

  while (true) {
    if (signal?.aborted) throw new DOMException('UPLOAD_ABORTED', 'AbortError')
    const task = await getGigaChatWorkbookUploadTask(completed.task_id)
    onProgress({
      phase: task.phase,
      message: task.message || 'Backend обрабатывает файл',
      progress: Math.max(0, Math.min(1, task.progress)),
      uploadedBytes: task.received_bytes || file.size,
      totalBytes: task.total_size || file.size,
      receivedChunks: task.received_chunks || totalChunks,
      totalChunks: task.total_chunks || totalChunks,
      taskId: completed.task_id,
      sessionId: completed.session_id,
      filename: file.name,
    })
    if (task.status === 'completed' && task.workbook) return task.workbook
    if (task.status === 'cancelled') throw new DOMException('UPLOAD_CANCELLED', 'AbortError')
    if (task.status === 'failed') throw new Error(task.error || 'Backend не смог обработать файл.')
    await sleep(CHUNKED_UPLOAD_POLL_MS)
  }
}

export default function GigaChatPage() {
  const qc = useQueryClient()
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const { user } = useAuth()
  const hiddenLabSettingKeys = new Set([
    'rule_pack_prompt_notes',
    RULE_PACK_EXCLUSION_SETTING_KEY,
    'reclassification_prompt_notes',
    'reclassification_source_field',
    'reclassification_context_field',
    'reclassification_prompt',
  ])
  const statusQ = useQuery({
    queryKey: ['gigachat-status'],
    queryFn: getGigaChatStatus,
    staleTime: 30_000,
    refetchOnWindowFocus: false,
  })
  const [selectedSettingsVersionId, setSelectedSettingsVersionId] = useState(loadStoredSettingsVersionId)
  const [versionCreateOpen, setVersionCreateOpen] = useState(false)
  const [versionDraft, setVersionDraft] = useState({
    title: '',
    versionId: '',
    description: '',
    status: 'draft' as GigaChatSettingsVersionStatus,
    visibility: 'private' as GigaChatSettingsVersionVisibility,
    createdBy: '',
    baseVersionId: 'default',
  })
  const versionsQ = useQuery({
    queryKey: ['gigachat-lab-settings-versions'],
    queryFn: listGigaChatLabSettingsVersions,
    staleTime: 30_000,
    refetchOnWindowFocus: false,
  })
  const selectedVersionQ = useQuery({
    queryKey: ['gigachat-lab-settings-version', selectedSettingsVersionId],
    queryFn: () => getGigaChatLabSettingsVersion(selectedSettingsVersionId),
    enabled: Boolean(selectedSettingsVersionId),
    staleTime: 30_000,
    refetchOnWindowFocus: false,
  })
  const selectedVersionCanEdit = selectedVersionQ.data?.version.can_edit ?? false
  const currentUserDisplayName = user?.display_name || user?.email || ''
  const personalProfileForSelectedVersion = selectedVersionQ.data?.version.can_edit ? null : (versionsQ.data?.versions ?? []).find((version) =>
    version.can_edit
    && version.visibility === 'private'
    && version.base_version_id === selectedVersionQ.data?.version.version_id
  ) ?? null

  const [selectedTransport, setSelectedTransport] = useState<GigaChatTransportName>('mtls')
  const [heroCollapsed, setHeroCollapsed] = useState(false)
  const [activeLabTab, setActiveLabTab] = useState<GigaChatLabTab>('workspace')
  const [settingsCollapsed, setSettingsCollapsed] = useState(false)
  const [versionInfoOpen, setVersionInfoOpen] = useState(false)
  const [activeSetupTab, setActiveSetupTab] = useState<LabSetupTab>('rules')
  const [reclassificationOpen, setReclassificationOpen] = useState(false)
  const [reclassificationSaveStatus, setReclassificationSaveStatus] = useState<ReclassificationSaveStatus>({ type: 'idle', message: '' })
  const [reclassificationImporting, setReclassificationImporting] = useState(false)
  const [reclassificationImportInputVersion, setReclassificationImportInputVersion] = useState(0)
  const [rulePackExclusionDraft, setRulePackExclusionDraft] = useState('')
  const [rulePackExclusionCodesDraft, setRulePackExclusionCodesDraft] = useState<string[]>([])
  const rulePackExclusionSelectRef = useRef<HTMLSelectElement | null>(null)
  const [reclassificationDraft, setReclassificationDraft] = useState<ReclassificationDraft>({
    editingIndex: null,
    name: '',
    sourceField: '',
    contextFields: [],
    prompt: '',
  })
  const [uploadCollapsed, setUploadCollapsed] = useState(false)
  const [workbookCollapsed, setWorkbookCollapsed] = useState(false)
  const [resultCollapsed, setResultCollapsed] = useState(false)
  const [collapsedTransports, setCollapsedTransports] = useState<Record<GigaChatTransportName, boolean>>({
    mtls: false,
    token: false,
  })
  const [settingValues, setSettingValues] = useState<Record<string, unknown>>({})
  const settingValuesVersionIdRef = useRef<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [fileInputVersion, setFileInputVersion] = useState(0)
  const [workbookUploadProgress, setWorkbookUploadProgress] = useState<WorkbookUploadProgressState | null>(null)
  const [workbookUploadOverlayHidden, setWorkbookUploadOverlayHidden] = useState(false)
  const [backgroundWorkbookUploads, setBackgroundWorkbookUploads] = useState<BackgroundWorkbookUpload[]>(() => loadBackgroundWorkbookUploads())
  const [workbookMeta, setWorkbookMeta] = useState<GigaChatWorkbookUploadResponse | null>(null)
  const [sheetData, setSheetData] = useState<GigaChatWorkbookSheetDataResponse | null>(null)
  const [viewSheetData, setViewSheetData] = useState<GigaChatWorkbookSheetDataResponse | null>(null)
  const [sheetPickerOpen, setSheetPickerOpen] = useState(false)
  const [finalPromptPreview, setFinalPromptPreview] = useState<GigaChatFinalPromptResponse | null>(null)
  const [finalPromptModalOpen, setFinalPromptModalOpen] = useState(false)
  const [finalPromptDraftText, setFinalPromptDraftText] = useState('')
  const [finalPromptBaseText, setFinalPromptBaseText] = useState('')
  const [finalPromptOverrideText, setFinalPromptOverrideText] = useState<string | null>(null)
  const [rowRunModalOpen, setRowRunModalOpen] = useState(false)
  const [rowRunResultPages, setRowRunResultPages] = useState<RowRunResultPage[]>([])
  const [activeRowRunResultPage, setActiveRowRunResultPage] = useState(0)
  const [workbookRowLimit, setWorkbookRowLimit] = useState<WorkbookRowLimit>('all')
  const [includedPromptColumns, setIncludedPromptColumns] = useState<string[]>([])
  const [selectedSheetRowKeys, setSelectedSheetRowKeys] = useState<string[]>([])
  const selectedSheetRowKeySet = useMemo(() => new Set(selectedSheetRowKeys), [selectedSheetRowKeys])
  const [annotatedRows, setAnnotatedRows] = useState<AnnotatedSheetRow[]>([])
  const [annotatedClassFilter, setAnnotatedClassFilter] = useState<string[]>([])
  const [annotatedTagFilter, setAnnotatedTagFilter] = useState<string[]>([])
  const [annotatedRuleFilter, setAnnotatedRuleFilter] = useState<string[]>([])
  const [annotatedDecisionSourceFilter, setAnnotatedDecisionSourceFilter] = useState<string[]>([])
  const [ruleEvaluationMap, setRuleEvaluationMap] = useState<Record<number, GigaChatRuleEvaluationRow>>({})
  const [runRowBusyIndex, setRunRowBusyIndex] = useState<number | null>(null)
  const [batchBusy, setBatchBusy] = useState(false)
  const [rowRunError, setRowRunError] = useState<string | null>(null)
  const [batchRunError, setBatchRunError] = useState<string | null>(null)
  const [ruleEvaluationError, setRuleEvaluationError] = useState<string | null>(null)
  const [ruleEvaluationProgress, setRuleEvaluationProgress] = useState<{ processed: number; total: number } | null>(null)
  const [backgroundResultProgress, setBackgroundResultProgress] = useState<{ loadedBytes: number; totalBytes: number | null } | null>(null)
  const [exportProgress, setExportProgress] = useState<{ label: string; loadedBytes: number; totalBytes: number | null } | null>(null)
  const [backgroundTaskError, setBackgroundTaskError] = useState<string | null>(null)
  const [sendReclassificationRequests, setSendReclassificationRequests] = useState(false)
  const [processingOverlay, setProcessingOverlay] = useState<ProcessingOverlayState | null>(null)
  const [ruleGuardIssues, setRuleGuardIssues] = useState<RuleGuardIssue[]>([])
  const [pendingRuleGuardAction, setPendingRuleGuardAction] = useState<PendingRuleGuardAction | null>(null)
  const [pendingRuleGuardValues, setPendingRuleGuardValues] = useState<Record<string, unknown> | null>(null)
  const [tokenAccounting, setTokenAccounting] = useState<TokenAccountingState>(() => loadTokenAccountingState())
  const [asyncWorkerCount, setAsyncWorkerCount] = useState(() => loadAsyncWorkerCount())
  const lastAutoPreviewKeyRef = useRef<string | null>(null)
  const lastResolvedPreviewKeyRef = useRef<string | null>(null)
  const workbookUploadAbortRef = useRef<AbortController | null>(null)
  const annotatedTable = useResizableTable()
  const {
    hoveredCell: hoveredAnnotatedCell,
    showCellPopover: showAnnotatedCellPopover,
    hideCellPopover: hideAnnotatedCellPopover,
  } = useCellHoverPopover()

  useEffect(() => {
    document.title = 'GigaChat Lab'
  }, [])

  useEffect(() => {
    const raw = window.sessionStorage.getItem(GIGACHAT_LAKE_IMPORT_STORAGE_KEY)
    if (!raw) return
    try {
      const payload = JSON.parse(raw) as Partial<GigaChatLakeImportPayload>
      const columns = Array.isArray(payload.columns)
        ? payload.columns.map((column) => String(column)).filter(Boolean)
        : []
      const rows = Array.isArray(payload.rows)
        ? payload.rows.map((row) => {
          const record = row && typeof row === 'object' && !Array.isArray(row) ? row as Record<string, unknown> : {}
          return Object.fromEntries(columns.map((column) => [column, record[column] ?? '']))
        })
        : []
      if (!columns.length || !rows.length) return
      const importedSheet: GigaChatWorkbookSheetDataResponse = {
        upload_id: String(payload.upload_id || `lake-import-${Date.now()}`),
        filename: String(payload.filename || 'parquet_selection.csv'),
        file_format: 'csv',
        sheet_name: String(payload.sheet_name || 'Parquet selection'),
        total_rows: rows.length,
        rendered_rows: rows.length,
        columns,
        rows,
      }
      setWorkbookMeta({
        upload_id: importedSheet.upload_id,
        filename: importedSheet.filename,
        file_format: importedSheet.file_format,
        sheet_count: 1,
        sheets: [{
          name: importedSheet.sheet_name,
          rows_total: importedSheet.total_rows,
          column_count: importedSheet.columns.length,
          columns: importedSheet.columns,
          preview_rows: importedSheet.rows.slice(0, 5),
        }],
      })
      setSheetData(importedSheet)
      setViewSheetData(importedSheet)
      setSelectedFile(null)
      setFileInputVersion((current) => current + 1)
      setIncludedPromptColumns([...columns])
      setSelectedSheetRowKeys([])
      setAnnotatedRows([])
      setRuleEvaluationMap({})
      setWorkbookRowLimit('all')
      setActiveLabTab('workspace')
      setUploadCollapsed(true)
      setWorkbookCollapsed(false)
      window.setTimeout(() => document.getElementById('gigachat-workbook-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50)
    } finally {
      window.sessionStorage.removeItem(GIGACHAT_LAKE_IMPORT_STORAGE_KEY)
    }
  }, [])

  useEffect(() => {
    window.localStorage.setItem(TOKEN_ACCOUNTING_STORAGE_KEY, JSON.stringify(tokenAccounting))
  }, [tokenAccounting])

  useEffect(() => {
    window.localStorage.setItem(ASYNC_WORKERS_STORAGE_KEY, String(asyncWorkerCount))
  }, [asyncWorkerCount])

  useEffect(() => {
    window.localStorage.setItem(WORKBOOK_UPLOAD_BACKGROUND_STORAGE_KEY, JSON.stringify(backgroundWorkbookUploads))
  }, [backgroundWorkbookUploads])

  useEffect(() => {
    if (!selectedSettingsVersionId) return
    window.localStorage.setItem(SETTINGS_VERSION_STORAGE_KEY, selectedSettingsVersionId)
  }, [selectedSettingsVersionId])

  useEffect(() => {
    if (!currentUserDisplayName) return
    setVersionDraft((current) => current.createdBy ? current : { ...current, createdBy: currentUserDisplayName })
  }, [currentUserDisplayName])

  useEffect(() => {
    const transports = statusQ.data?.transports ?? []
    if (!transports.length) return
    setSelectedTransport((current) => {
      const currentTransport = transports.find((item) => item.name === current)
      if (currentTransport?.ready) {
        return current
      }
      const ready = transports.find((item) => item.ready)
      if (ready) {
        return ready.name
      }
      if (currentTransport) {
        return current
      }
      const configured = transports.find((item) => item.configured)
      const active = transports.find((item) => item.active)
      return configured?.name ?? active?.name ?? 'mtls'
    })
  }, [statusQ.data])

  useEffect(() => {
    const versions = versionsQ.data?.versions ?? []
    if (!versions.length) return
    if (selectedSettingsVersionId && versions.some((version) => version.version_id === selectedSettingsVersionId)) return
    const storedVersionId = readStoredSettingsVersionId()
    const storedVersion = storedVersionId ? versions.find((version) => version.version_id === storedVersionId) : null
    const latestPrivateVersion = versions
      .filter((version) => version.can_edit && version.visibility === 'private')
      .sort((left, right) => {
        const leftTime = Date.parse(left.updated_at || left.created_at || '') || 0
        const rightTime = Date.parse(right.updated_at || right.created_at || '') || 0
        return rightTime - leftTime
      })[0]
    const nextVersionId = (storedVersion ?? latestPrivateVersion ?? versions[0]).version_id
    window.localStorage.setItem(SETTINGS_VERSION_STORAGE_KEY, nextVersionId)
    setSelectedSettingsVersionId(nextVersionId)
    setVersionDraft((current) => ({ ...current, baseVersionId: nextVersionId }))
  }, [selectedSettingsVersionId, versionsQ.data])

  useEffect(() => {
    setVersionInfoOpen(false)
  }, [selectedSettingsVersionId])

  useEffect(() => {
    if (!selectedVersionQ.data?.fields?.length) return
    const versionId = selectedVersionQ.data.version.version_id
    const values = fieldsToValues(selectedVersionQ.data.fields)
    const previousVersionId = settingValuesVersionIdRef.current
    setSettingValues((current) => {
      const nextValues = previousVersionId === versionId
        ? preserveRulePackExclusionValue(values, current)
        : values
      settingValuesVersionIdRef.current = versionId
      return nextValues
    })
    if (previousVersionId !== versionId) {
      setRulePackExclusionCodesDraft(parseRulePackExclusions(values[RULE_PACK_EXCLUSION_SETTING_KEY]))
    }
  }, [selectedVersionQ.data])

  const probe = useMutation({
    mutationFn: (transport: GigaChatTransportName) => probeGigaChatTransport(transport),
  })
  const modelOptionsQ = useQuery({
    queryKey: ['gigachat-model-options', selectedTransport],
    queryFn: () => probeGigaChatTransport(selectedTransport),
    enabled: activeLabTab === 'workspace' && activeSetupTab === 'prompts',
    staleTime: 60_000,
    refetchOnWindowFocus: false,
  })

  const saveSettings = useMutation({
    mutationFn: async (valuesOverride?: Record<string, unknown>) => {
      const sourceVersion = selectedVersionQ.data?.version
      if (!sourceVersion) {
        throw new Error('Версия настроек еще не загружена.')
      }
      const values = withSerializedRulePackExclusions(valuesOverride ?? settingValues, rulePackExclusionCodesDraft)
      if (sourceVersion.can_edit) {
        const saved = await saveGigaChatLabSettingsVersion(selectedSettingsVersionId, {
          title: sourceVersion.title,
          description: sourceVersion.description,
          status: sourceVersion.status,
          visibility: sourceVersion.visibility,
          updated_by: currentUserDisplayName,
          values,
        })
        return { ...saved, values: preserveRulePackExclusionValue(saved.values, values) }
      }

      const existingPersonalVersion = (versionsQ.data?.versions ?? []).find((version) =>
        version.can_edit
        && version.visibility === 'private'
        && version.base_version_id === sourceVersion.version_id
      )
      if (existingPersonalVersion) {
        const saved = await saveGigaChatLabSettingsVersion(existingPersonalVersion.version_id, {
          title: existingPersonalVersion.title,
          description: existingPersonalVersion.description,
          status: existingPersonalVersion.status,
          visibility: existingPersonalVersion.visibility,
          updated_by: currentUserDisplayName,
          values,
        })
        return { ...saved, values: preserveRulePackExclusionValue(saved.values, values) }
      }

      const baseTitle = sourceVersion.title || selectedSettingsVersionId || 'Профиль настроек'
      const created = await createGigaChatLabSettingsVersion({
        title: `${baseTitle} · мой профиль`,
        version_id: null,
        description: `Приватный профиль настроек на основе "${baseTitle}". Создан автоматически при сохранении изменений ${new Date().toLocaleString()}.`,
        status: 'draft',
        visibility: 'private',
        created_by: currentUserDisplayName,
        base_version_id: selectedSettingsVersionId,
      })
      const saved = await saveGigaChatLabSettingsVersion(created.version.version_id, {
        title: created.version.title,
        description: created.version.description,
        status: created.version.status,
        visibility: created.version.visibility,
        updated_by: currentUserDisplayName,
        values,
      })
      return { ...saved, values: preserveRulePackExclusionValue(saved.values, values) }
    },
    onSuccess: async (data) => {
      qc.setQueryData(['gigachat-lab-settings-version', data.version.version_id], data)
      qc.setQueryData(['gigachat-lab-settings-versions'], (current: GigaChatLabSettingsVersionsResponse | undefined) => ({
        versions: [
          data.version,
          ...(current?.versions ?? []).filter((version) => version.version_id !== data.version.version_id),
        ],
      }))
      setSelectedSettingsVersionId(data.version.version_id)
      window.localStorage.setItem(SETTINGS_VERSION_STORAGE_KEY, data.version.version_id)
      settingValuesVersionIdRef.current = data.version.version_id
      setSettingValues(data.values)
      setRulePackExclusionCodesDraft(parseRulePackExclusions(data.values[RULE_PACK_EXCLUSION_SETTING_KEY]))
      await qc.invalidateQueries({ queryKey: ['gigachat-lab-settings-versions'] })
      await qc.invalidateQueries({ queryKey: ['gigachat-lab-settings-version', data.version.version_id] })
      await qc.invalidateQueries({ queryKey: ['gigachat-status'] })
    },
  })

  const persistSettingValue = (key: string, value: unknown) => {
    const nextValues = withSerializedRulePackExclusions({ ...settingValues, [key]: value }, rulePackExclusionCodesDraft)
    setSettingValues(nextValues)
    saveSettings.mutate(nextValues)
  }

  const persistRulePacks = (value: string) => {
    persistSettingValue('rule_pack_prompt_notes', value)
  }

  const updateRulePackExclusionsDraft = (codes: string[]) => {
    const nextCodes = parseRulePackExclusions(codes)
    setRulePackExclusionCodesDraft(nextCodes)
    setSettingValues((current) => withSerializedRulePackExclusions(current, nextCodes))
  }

  const addRulePackExclusion = () => {
    const code = (
      activeRulePackExclusionDraft
      || rulePackExclusionSelectRef.current?.value
      || availableRulePackExclusionOptions[0]?.code
      || ''
    ).trim()
    if (!code) return
    updateRulePackExclusionsDraft([...rulePackExclusionCodes.filter((item) => item !== code), code])
    setRulePackExclusionDraft('')
  }

  const removeRulePackExclusion = (code: string) => {
    updateRulePackExclusionsDraft(rulePackExclusionCodes.filter((item) => item !== code))
  }

  const reclassificationRules = useMemo(
    () => parseReclassificationRules(settingValues.reclassification_prompt_notes, settingValues),
    [settingValues],
  )
  const reclassificationRuleStatuses = useMemo(
    () => reclassificationRules.map((rule) => getReclassificationRuleFieldStatus(rule, sheetData?.columns ?? null)),
    [reclassificationRules, sheetData?.columns],
  )
  const reclassificationRequestsAvailable = reclassificationRules.some((rule) =>
    canRunReclassificationRuleOnSheet(rule, sheetData?.columns ?? null)
  )
  const shouldSendReclassificationRequest = (valuesForRun: Record<string, unknown>) => (
    sendReclassificationRequests
    && parseReclassificationRules(valuesForRun.reclassification_prompt_notes, valuesForRun)
      .some((rule) => canRunReclassificationRuleOnSheet(rule, sheetData?.columns ?? null))
  )

  const openReclassificationRule = (index: number | null) => {
    const rule = index === null ? null : reclassificationRules[index]
    const contextFields = rule ? getReclassificationRuleContextFields(rule) : []
    setReclassificationSaveStatus({ type: 'idle', message: '' })
    setReclassificationDraft({
      editingIndex: index,
      name: rule?.name ?? '',
      sourceField: rule?.source_field ?? '',
      contextFields,
      prompt: rule?.prompt ?? '',
    })
    setReclassificationOpen(true)
  }

  const persistReclassificationRules = (
    nextRules: ReclassificationRule[],
    closeModal = false,
    messages?: {
      saving?: string
      saved?: (profileTitle: string) => string
    },
  ) => {
    const nextValues = {
      ...settingValues,
      reclassification_prompt_notes: serializeReclassificationRules(nextRules),
      reclassification_source_field: '',
      reclassification_context_field: '',
      reclassification_prompt: DEFAULT_RECLASSIFICATION_PROMPT,
    }
    setSettingValues(nextValues)
    setReclassificationSaveStatus({
      type: 'saving',
      message: messages?.saving ?? (selectedVersionCanEdit
        ? 'Сохраняем переклассификации в профиль...'
        : personalProfileForSelectedVersion
          ? `Записываем переклассификации в профиль "${personalProfileForSelectedVersion.title}"...`
          : 'Создаем мой профиль и записываем переклассификации...'),
    })
    saveSettings.mutate(nextValues, {
      onSuccess: (data) => {
        setReclassificationSaveStatus({
          type: 'saved',
          message: messages?.saved?.(data.version.title) ?? `Переклассификации записаны в профиль "${data.version.title}".`,
        })
        setActiveSetupTab('reclassification')
        if (closeModal) setReclassificationOpen(false)
      },
      onError: (error) => {
        setReclassificationSaveStatus({
          type: 'error',
          message: formatLabError(error as Error, 'переклассификацию'),
        })
      },
    })
  }

  const saveReclassificationRule = () => {
    const contextFields = Array.from(new Set(reclassificationDraftContextFields.map((field) => field.trim()).filter(Boolean)))
    const nextRule: ReclassificationRule = {
      name: reclassificationDraft.name.trim(),
      source_field: reclassificationDraft.sourceField.trim(),
      context_field: contextFields[0] ?? '',
      context_fields: contextFields,
      prompt: reclassificationDraft.prompt.trim(),
    }
    if (!nextRule.name || !nextRule.prompt || (!nextRule.source_field && !nextRule.context_fields.length)) return
    const fieldStatus = getReclassificationRuleFieldStatus(nextRule, sheetData?.columns ?? null)
    if (!fieldStatus.valid) {
      setReclassificationSaveStatus({
        type: 'error',
        message: `Нельзя сохранить активное правило: в загруженной таблице нет колонок ${fieldStatus.missing.join(', ')}.`,
      })
      return
    }
    const nextRules = [...reclassificationRules]
    if (reclassificationDraft.editingIndex === null) {
      nextRules.push(nextRule)
    } else {
      nextRules[reclassificationDraft.editingIndex] = nextRule
    }
    persistReclassificationRules(nextRules, true)
  }

  const removeReclassificationRule = (index: number) => {
    persistReclassificationRules(reclassificationRules.filter((_, currentIndex) => currentIndex !== index))
  }

  const importReclassificationRulesFromFile = async (file: File | null | undefined) => {
    if (!file || reclassificationImporting || saveSettings.isPending) return
    setReclassificationImporting(true)
    setReclassificationSaveStatus({ type: 'saving', message: `Читаем правила переклассификации из "${file.name}"...` })
    try {
      const data = await importGigaChatReclassificationRules(file)
      const importedRules = data.rules.map((rule) => {
        const contextFields = splitReclassificationFieldList(rule.context_fields?.length ? rule.context_fields : rule.context_field)
        return {
          name: rule.name.trim(),
          source_field: rule.source_field.trim(),
          context_field: contextFields[0] ?? '',
          context_fields: contextFields,
          prompt: rule.prompt.trim(),
        }
      })
      persistReclassificationRules(importedRules, false, {
        saving: `Excel прочитан: ${data.imported_count}. Записываем правила переклассификации в профиль...`,
        saved: (profileTitle) => `Из Excel загружено ${data.imported_count} правил и записано в профиль "${profileTitle}".`,
      })
    } catch (error) {
      setReclassificationSaveStatus({
        type: 'error',
        message: formatLabError(error as Error, 'загрузку правил переклассификации'),
      })
    } finally {
      setReclassificationImporting(false)
      setReclassificationImportInputVersion((current) => current + 1)
    }
  }

  const createSettingsVersion = useMutation({
    mutationFn: () => createGigaChatLabSettingsVersion({
      title: versionDraft.title,
      version_id: versionDraft.versionId || null,
      description: versionDraft.description,
      status: versionDraft.status,
      visibility: versionDraft.visibility,
      created_by: versionDraft.createdBy,
      base_version_id: versionDraft.baseVersionId,
    }),
    onSuccess: async (data) => {
      setSelectedSettingsVersionId(data.version.version_id)
      window.localStorage.setItem(SETTINGS_VERSION_STORAGE_KEY, data.version.version_id)
      settingValuesVersionIdRef.current = data.version.version_id
      setSettingValues(data.values)
      setRulePackExclusionCodesDraft(parseRulePackExclusions(data.values[RULE_PACK_EXCLUSION_SETTING_KEY]))
      setVersionCreateOpen(false)
      setVersionDraft((current) => ({ ...current, title: '', versionId: '', description: '', visibility: 'private', createdBy: currentUserDisplayName }))
      await qc.invalidateQueries({ queryKey: ['gigachat-lab-settings-versions'] })
    },
  })

  const deleteSettingsVersion = useMutation({
    mutationFn: (versionId: string) => deleteGigaChatLabSettingsVersion(versionId),
    onSuccess: async (data, deletedVersionId) => {
      qc.removeQueries({ queryKey: ['gigachat-lab-settings-version', deletedVersionId] })
      qc.setQueryData(['gigachat-lab-settings-versions'], data)
      const nextVersionId = data.versions[0]?.version_id ?? 'default'
      window.localStorage.setItem(SETTINGS_VERSION_STORAGE_KEY, nextVersionId)
      setSelectedSettingsVersionId(nextVersionId)
      setVersionDraft((current) => ({ ...current, baseVersionId: nextVersionId }))
      setVersionInfoOpen(false)
      await qc.invalidateQueries({ queryKey: ['gigachat-lab-settings-versions'] })
      await qc.invalidateQueries({ queryKey: ['gigachat-lab-settings-version', nextVersionId] })
    },
  })

  const selectSheet = useMutation({
    mutationFn: ({ uploadId, sheetName, rowLimit }: { uploadId: string; sheetName: string; rowLimit: number }) => selectGigaChatWorkbookSheet(uploadId, sheetName, rowLimit),
    onSuccess: (data) => {
      const sameSheet = sheetData?.upload_id === data.upload_id && sheetData?.sheet_name === data.sheet_name
      setIncludedPromptColumns((current) => {
        if (!sameSheet || !current.length) return [...data.columns]
        const filtered = current.filter((column) => data.columns.includes(column))
        return filtered.length ? filtered : [...data.columns]
      })
      if (!sameSheet) {
        setSelectedSheetRowKeys([])
        setRuleEvaluationMap({})
        setAnnotatedRows([])
        setAnnotatedClassFilter([])
        setAnnotatedTagFilter([])
        setAnnotatedRuleFilter([])
        setAnnotatedDecisionSourceFilter([])
        setRowRunResultPages([])
        setActiveRowRunResultPage(0)
        setRowRunError(null)
        setBatchRunError(null)
      }
      setSheetData(data)
      setViewSheetData(data)
      setWorkbookCollapsed(false)
      setUploadCollapsed(true)
      setSheetPickerOpen(false)
    },
  })

  const viewSheet = useMutation({
    mutationFn: ({ uploadId, sheetName, rowLimit }: { uploadId: string; sheetName: string; rowLimit: number }) => selectGigaChatWorkbookSheet(uploadId, sheetName, rowLimit),
    onSuccess: (data) => {
      setViewSheetData(data)
      setWorkbookCollapsed(false)
    },
  })

  const selectWorkbookSheet = (sheetName: string) => {
    if (!workbookMeta) return
    const previewSheet = workbookMeta.sheets.find((sheet) => sheet.name === sheetName)
    const rowLimit = workbookRowLimit === 'all' ? (previewSheet?.rows_total ?? 200) : workbookRowLimit
    selectSheet.mutate({ uploadId: workbookMeta.upload_id, sheetName, rowLimit })
  }

  const viewWorkbookSheet = (sheetName: string) => {
    if (!workbookMeta) return
    const previewSheet = workbookMeta.sheets.find((sheet) => sheet.name === sheetName)
    const rowLimit = workbookRowLimit === 'all' ? (previewSheet?.rows_total ?? 200) : workbookRowLimit
    viewSheet.mutate({ uploadId: workbookMeta.upload_id, sheetName, rowLimit })
  }

  const activateUploadedWorkbook = (data: GigaChatWorkbookUploadResponse) => {
    setWorkbookMeta(data)
    setSheetData(null)
    setViewSheetData(null)
    setSelectedFile(null)
    setFileInputVersion((current) => current + 1)
    setWorkbookCollapsed(false)
    if (data.sheet_count <= 1 && data.sheets[0]) {
      const rowLimit = workbookRowLimit === 'all' ? data.sheets[0].rows_total : workbookRowLimit
      selectSheet.mutate({ uploadId: data.upload_id, sheetName: data.sheets[0].name, rowLimit })
    } else {
      setSheetPickerOpen(true)
    }
  }

  const addBackgroundWorkbookUpload = (task: BackgroundWorkbookUpload) => {
    setBackgroundWorkbookUploads((current) => {
      const filtered = current.filter((item) => item.taskId !== task.taskId)
      return [task, ...filtered].slice(0, 12)
    })
  }

  const backgroundUploadTasksQ = useQuery({
    queryKey: ['gigachat-workbook-background-uploads', backgroundWorkbookUploads.map((item) => item.taskId).join('|')],
    enabled: backgroundWorkbookUploads.length > 0,
    queryFn: async () => Promise.all(backgroundWorkbookUploads.map(async (item) => {
      try {
        const task = await getGigaChatWorkbookUploadTask(item.taskId)
        return { item, task, error: null as string | null }
      } catch (error) {
        return { item, task: null as GigaChatWorkbookUploadTaskResponse | null, error: (error as Error).message }
      }
    })),
    refetchInterval: backgroundWorkbookUploads.length ? 2000 : false,
  })

  const backgroundUploadTaskMap = useMemo(() => {
    const entries = backgroundUploadTasksQ.data ?? []
    return new Map(entries.map((entry) => [entry.item.taskId, entry]))
  }, [backgroundUploadTasksQ.data])

  const sendCurrentUploadToBackground = () => {
    const taskId = workbookUploadProgress?.taskId
    if (!taskId) return
    addBackgroundWorkbookUpload({
      taskId,
      sessionId: workbookUploadProgress.sessionId || '',
      filename: workbookUploadProgress.filename || selectedFile?.name || 'Фоновая загрузка',
      createdAt: new Date().toISOString(),
    })
    setWorkbookUploadOverlayHidden(true)
    setUploadCollapsed(true)
  }

  const uploadWorkbook = useMutation({
    mutationFn: async (fileArg?: File) => {
      const file = fileArg ?? selectedFile
      if (!file) throw new Error('Выберите Excel или CSV файл')
      workbookUploadAbortRef.current?.abort()
      const controller = new AbortController()
      workbookUploadAbortRef.current = controller
      setWorkbookUploadOverlayHidden(false)
      setWorkbookUploadProgress({
        phase: 'starting',
        message: 'Готовим загрузку файла',
        progress: 0,
        uploadedBytes: 0,
        totalBytes: file.size,
        receivedChunks: 0,
        totalChunks: Math.max(1, Math.ceil(file.size / CHUNKED_UPLOAD_CHUNK_BYTES)),
        filename: file.name,
      })

      const allowLocalFallback = canUseLocalWorkbookFallback()

      if (!allowLocalFallback || file.size >= CHUNKED_UPLOAD_THRESHOLD_BYTES) {
        return uploadWorkbookInChunks(file, setWorkbookUploadProgress, controller.signal)
      }

      if (allowLocalFallback) {
        try {
          return await uploadLocalGigaChatWorkbook(file.name)
        } catch {
          // Fall back to browser multipart for files that are not present in local project folders.
        }
      }

      const timeoutId = window.setTimeout(() => {
        controller.abort(new DOMException('WORKBOOK_UPLOAD_TIMEOUT', 'AbortError'))
      }, WORKBOOK_UPLOAD_TIMEOUT_MS)
      const form = new FormData()
      form.append('file', file)
      try {
        return await uploadGigaChatWorkbook(form, controller.signal)
      } catch (error) {
        if (allowLocalFallback && error instanceof DOMException && error.name === 'AbortError') {
          return uploadLocalGigaChatWorkbook(file.name)
        }
        throw error
      } finally {
        window.clearTimeout(timeoutId)
        if (workbookUploadAbortRef.current === controller) {
          workbookUploadAbortRef.current = null
        }
      }
    },
    onSuccess: (data) => {
      setWorkbookUploadProgress(null)
      if (workbookUploadOverlayHidden) {
        setSelectedFile(null)
        setFileInputVersion((current) => current + 1)
      } else {
        activateUploadedWorkbook(data)
      }
      setWorkbookUploadOverlayHidden(false)
    },
    onError: () => {
      workbookUploadAbortRef.current = null
      setWorkbookUploadOverlayHidden(false)
    },
    onSettled: () => {
      setWorkbookUploadProgress(null)
    },
  })

  const cancelWorkbookUploadTaskMutation = useMutation({
    mutationFn: cancelGigaChatWorkbookUploadTask,
    onSuccess: async (task) => {
      setBackgroundWorkbookUploads((current) => current.filter((item) => item.taskId !== task.task_id))
      await backgroundUploadTasksQ.refetch()
    },
  })

  const cancelCurrentWorkbookUpload = async () => {
    const progress = workbookUploadProgress
    workbookUploadAbortRef.current?.abort()
    workbookUploadAbortRef.current = null
    try {
      if (progress?.taskId) {
        await cancelGigaChatWorkbookUploadTask(progress.taskId)
        setBackgroundWorkbookUploads((current) => current.filter((item) => item.taskId !== progress.taskId))
      } else if (progress?.sessionId) {
        await cancelGigaChatWorkbookChunkedUploadSession(progress.sessionId)
      }
    } finally {
      setWorkbookUploadOverlayHidden(false)
      setWorkbookUploadProgress(null)
      uploadWorkbook.reset()
    }
  }

  const finalPromptColumns = useMemo(
    () => (sheetData ? sheetData.columns.filter((column) => includedPromptColumns.includes(column)) : []),
    [sheetData, includedPromptColumns],
  )
  const currentSheetColumnSet = useMemo(() => new Set(sheetData?.columns ?? []), [sheetData?.columns])
  const isReclassificationFieldMissing = (field: string) => (
    Boolean(sheetData && String(field ?? '').trim() && !currentSheetColumnSet.has(String(field ?? '').trim()))
  )
  const reclassificationDraftContextFields = splitReclassificationFieldList(reclassificationDraft.contextFields)
  const reclassificationDraftSourceMissing = isReclassificationFieldMissing(reclassificationDraft.sourceField)
  const reclassificationDraftMissingContextFields = reclassificationDraftContextFields.filter((field) => isReclassificationFieldMissing(field))
  const reclassificationDraftContextMissing = reclassificationDraftMissingContextFields.length > 0
  const reclassificationColumnOptions = useMemo(() => {
    const columns = sheetData?.columns ?? []
    return Array.from(new Set([
      ...columns,
      ...reclassificationRules.flatMap((rule) => [rule.source_field, ...getReclassificationRuleContextFields(rule)]),
      String(settingValues.reclassification_source_field ?? '').trim(),
      String(settingValues.reclassification_context_field ?? '').trim(),
      reclassificationDraft.sourceField.trim(),
      ...reclassificationDraftContextFields.map((field) => field.trim()),
    ].filter(Boolean)))
  }, [
    sheetData?.columns,
    reclassificationRules,
    settingValues.reclassification_source_field,
    settingValues.reclassification_context_field,
    reclassificationDraft.sourceField,
    reclassificationDraftContextFields,
  ])
  const reclassificationAvailableColumnOptions = useMemo(
    () => reclassificationColumnOptions.filter((column) => !isReclassificationFieldMissing(column)),
    [reclassificationColumnOptions, sheetData, currentSheetColumnSet],
  )
  const reclassificationMissingColumnOptions = useMemo(
    () => reclassificationColumnOptions.filter((column) => isReclassificationFieldMissing(column)),
    [reclassificationColumnOptions, sheetData, currentSheetColumnSet],
  )
  const hasReclassificationDraftMissingFields = reclassificationDraftSourceMissing || reclassificationDraftContextMissing
  const canSaveReclassificationDraft = (
    Boolean(reclassificationDraft.name.trim())
    && Boolean(reclassificationDraft.prompt.trim())
    && Boolean(reclassificationDraft.sourceField.trim() || reclassificationDraftContextFields.length)
    && !hasReclassificationDraftMissingFields
  )
  const renderReclassificationColumnOptions = (prefix: string) => <>
    <option value=''>Не выбрано</option>
    {reclassificationAvailableColumnOptions.length ? <optgroup label='Есть в загруженной таблице'>
      {reclassificationAvailableColumnOptions.map((column) => <option
        key={`${prefix}-available-${column}`}
        value={column}
      >
        {column}
      </option>)}
    </optgroup> : null}
    {reclassificationMissingColumnOptions.length ? <optgroup label='Нет в загруженной таблице'>
      {reclassificationMissingColumnOptions.map((column) => <option
        key={`${prefix}-missing-${column}`}
        value={column}
        className='missing-field-option'
        disabled
      >
        {column} (нет в таблице)
      </option>)}
    </optgroup> : null}
  </>
  const finalPromptRequestKey = useMemo(
    () => JSON.stringify({ values: settingValues, columns: finalPromptColumns }),
    [settingValues, finalPromptColumns],
  )

  const previewFinalPrompt = useMutation({
    mutationFn: ({ requestKey }: { requestKey: string }) => previewGigaChatFinalPrompt(settingValues, finalPromptColumns).then((data) => ({ data, requestKey })),
    onSuccess: ({ data, requestKey }) => {
      setFinalPromptPreview(data)
      lastResolvedPreviewKeyRef.current = requestKey
      const nextBaseText = stringifyJson(data.payload)
      setFinalPromptBaseText(nextBaseText)
      setFinalPromptDraftText(nextBaseText)
      setFinalPromptOverrideText(null)
    },
  })

  const exportWorkbookRows = useMutation({
    mutationFn: async (rows: Array<{ rowIndex: number; row: Record<string, unknown>; evaluation?: GigaChatRuleEvaluationRow }>) => {
      if (!sheetData) throw new Error('Сначала загрузите рабочую таблицу.')
      return exportGigaChatWorkbookRows(
        sheetData.filename,
        sheetData.sheet_name,
        sheetData.columns,
        rows.map(({ rowIndex, row, evaluation }) => ({
          row_index: rowIndex,
          source_row: row,
          rule_hits: evaluation?.hits.map((hit) => hit.code) ?? [],
          suggested_actions: [
            ...(evaluation?.suggested_tags ?? []).map((item) => `tag:${item}`),
            ...(evaluation?.suggested_topics ?? []).map((item) => `topic:${item}`),
          ],
          matched_keywords: Array.from(new Set(evaluation?.hits.flatMap((hit) => hit.matched_keywords) ?? [])),
          matched_fields: Array.from(new Set(evaluation?.hits.flatMap((hit) => hit.matched_fields) ?? [])),
          suggested_topics: evaluation?.suggested_topics ?? [],
        })),
        (loadedBytes, totalBytes) => setExportProgress({ label: 'Выгружаем рабочую таблицу', loadedBytes, totalBytes }),
      )
    },
    onSuccess: ({ blob, filename }) => {
      const sourceName = sheetData?.filename ?? 'workbook.xlsx'
      const fallbackName = `${sourceName.replace(/\.[^.]+$/u, '') || 'workbook'}_${sheetData?.sheet_name || 'sheet'}_rows.xlsx`
      downloadBlob(blob, filename || fallbackName)
    },
    onSettled: () => setExportProgress(null),
  })

  const buildAnnotatedExportRow = (row: AnnotatedSheetRow) => ({
    row_index: row.rowIndex,
    classification: row.classification,
    new_class: isAnnotatedRowReclassified(row) ? normalizeReclassificationTopicValue(row.newClassification) : '',
    source_classification: row.sourceClassification,
    is_reclassified: isAnnotatedRowReclassified(row),
    tags: row.tags,
    local_tags: row.localTags,
    model_added_tags: row.modelAddedTags,
    model_rejected_tags: row.modelRejectedTags,
    match_type: row.matchType || null,
    evidence: row.evidence || null,
    tag_decisions: row.tagDecisions || null,
    rule_hits: row.ruleHits,
    suggested_topics: row.suggestedTopics,
    confirmed_rule_hits: row.confirmedRuleHits,
    rejected_rule_hits: row.rejectedRuleHits,
    rule_decision: row.ruleDecision,
    model_decision: row.modelDecision,
    reclassified_topic: row.reclassifiedTopic || null,
    final_topic: row.finalTopic || null,
    decision_source: row.decisionSource,
    source_row: row.sourceRow,
  })

  const exportAnnotatedRows = useMutation({
    mutationFn: async (mode: 'filtered' | 'reclassified' = 'filtered') => {
      if (!sheetData) throw new Error('Сначала загрузите рабочую таблицу.')
      const rowsForExport = mode === 'reclassified'
        ? filteredAnnotatedRows.filter((row) => isAnnotatedRowReclassified(row))
        : filteredAnnotatedRows
      if (!rowsForExport.length) {
        throw new Error(mode === 'reclassified' ? 'Нет строк с новой подтематикой для выгрузки.' : 'Нет строк для выгрузки.')
      }
      const baseName = sheetData.filename.replace(/\.[^.]+$/u, '') || 'annotated'
      const exportFilename = mode === 'reclassified'
        ? `${baseName}_reclassified.xlsx`
        : sheetData.filename
      return exportGigaChatAnnotatedWorkbook(
        exportFilename,
        sheetData.sheet_name,
        sheetData.columns,
        rowsForExport.map(buildAnnotatedExportRow),
        (loadedBytes, totalBytes) => {
          setExportProgress({
            label: mode === 'reclassified' ? 'Выгружаем строки с новой подтематикой' : 'Выгружаем Excel',
            loadedBytes,
            totalBytes,
          })
        },
      )
    },
    onSuccess: ({ blob, filename }, mode) => {
      const sourceName = sheetData?.filename ?? 'annotated.xlsx'
      const fallbackStem = sourceName.replace(/\.[^.]+$/u, '') || 'annotated'
      const fallbackName = mode === 'reclassified'
        ? `${fallbackStem}_reclassified_annotated.xlsx`
        : `${fallbackStem}_annotated.xlsx`
      downloadBlob(blob, filename || fallbackName)
    },
    onSettled: () => setExportProgress(null),
  })

  const exportValidationRows = useMutation({
    mutationFn: async () => {
      if (!sheetData) throw new Error('Сначала загрузите рабочую таблицу.')
      return exportGigaChatValidationWorkbook(
        sheetData.filename,
        sheetData.sheet_name,
        sheetData.columns,
        filteredAnnotatedRows.map(buildAnnotatedExportRow),
        (loadedBytes, totalBytes) => {
          setExportProgress({ label: 'Готовим validation-файл', loadedBytes, totalBytes })
        },
      )
    },
    onSuccess: ({ blob, filename }) => {
      const sourceName = sheetData?.filename ?? 'validation.xlsx'
      const fallbackName = `${sourceName.replace(/\.[^.]+$/u, '') || 'annotated'}_validation.xlsx`
      downloadBlob(blob, filename || fallbackName)
    },
    onSettled: () => setExportProgress(null),
  })

  const evaluateRulePacks = useMutation({
    mutationFn: async (rows: Array<Record<string, unknown>>) => {
      return evaluateRulePacksWithProgress(settingValues, rows, (processed, total) => {
        setRuleEvaluationProgress({ processed, total })
      })
    },
    onSuccess: (data) => {
      setRuleEvaluationMap(Object.fromEntries(data.evaluations.map((item) => [item.row_index, item])))
      setRuleEvaluationError(null)
    },
    onError: (error) => {
      setRuleEvaluationError(formatLabError(error as Error, 'rule packs'))
    },
    onSettled: () => {
      setRuleEvaluationProgress(null)
    },
  })

  const buildAnnotatedRow = (
    data: GigaChatLabRowRunResponse,
    rowIndex: number,
    row: Record<string, unknown>,
    workbook: GigaChatWorkbookSheetDataResponse,
    reclassificationData?: GigaChatLabRowRunResponse | null,
  ): AnnotatedSheetRow => {
    const rowKey = buildSheetRowKey(workbook.upload_id, workbook.sheet_name, rowIndex)
    const localRuleHits = data.rule_evaluation?.hits.map((item) => item.code) ?? []
    const localTags = data.rule_evaluation?.suggested_tags ?? []
    const confirmedRuleHits = extractConfirmedRuleHits(data.response_json)
    const rejectedRuleHits = extractRejectedRuleHits(data.response_json)
    const modelAddedTags = extractModelAddedTags(data.response_json)
    const modelRejectedTags = extractModelRejectedTags(data.response_json)
    const matchType = extractMatchType(data.response_json)
    const evidence = extractEvidence(data.response_json)
    const tagDecisions = extractTagDecisionSummary(data.response_json)
    const reclassificationWasRun = Boolean(reclassificationData)
    const secondPassReclassifiedTopic = extractReclassifiedTopic(reclassificationData?.response_json)
    const reclassifiedTopic = secondPassReclassifiedTopic || extractReclassifiedTopic(data.response_json)
    const primaryClassification = extractClassification(data.response_json)
    const sourceClassificationField = (
      reclassificationRules.find((rule) => rule.source_field.trim())?.source_field
      || ''
    )
    const sourceClassification = sourceClassificationField
      ? String(row[sourceClassificationField] ?? '').trim()
      : primaryClassification
    const newClassification = normalizeReclassificationTopicValue(reclassifiedTopic)
    const isReclassified = Boolean(newClassification)
    const classification = primaryClassification || sourceClassification || ''
    const finalTopic = reclassifiedTopic || classification
    const confirmedLocalTags = localTags.filter((tag) => {
      if (confirmedRuleHits.includes(tag)) return true
      return data.rule_evaluation?.hits.some((hit) => hit.target_tag === tag && confirmedRuleHits.includes(hit.code)) ?? false
    })
    const explicitTags = extractTags(data.response_json)
    const computedFinalTags = Array.from(new Set([...confirmedLocalTags, ...modelAddedTags]))
      .filter((tag) => !modelRejectedTags.includes(tag))
    const finalTags = explicitTags.length ? explicitTags : computedFinalTags
    const ruleDecision = confirmedRuleHits.length || rejectedRuleHits.length
      ? [
        confirmedRuleHits.length ? `confirmed: ${confirmedRuleHits.join(', ')}` : '',
        rejectedRuleHits.length ? `rejected: ${rejectedRuleHits.join(', ')}` : '',
      ].filter(Boolean).join('; ')
      : localRuleHits.length ? 'pending_in_model_response' : 'no_rule_hits'
    const modelDecision = isReclassified
      ? 'reclassified_by_second_request'
      : reclassificationWasRun
      ? 'reclassification_no_change'
      : !data.parse_ok
      ? 'raw_response'
      : modelAddedTags.length
        ? 'semantic_added'
        : modelRejectedTags.length
          ? 'semantic_rejected'
          : confirmedRuleHits.length || reclassifiedTopic
            ? 'confirmed'
            : rejectedRuleHits.length
              ? 'rejected'
              : finalTopic
                ? 'classified'
                : 'empty_result'
    const decisionSource = secondPassReclassifiedTopic
      ? 'llm_reclassification'
      : reclassificationWasRun
      ? 'llm_reclassification'
      : modelAddedTags.length
      ? 'llm_semantic'
      : modelRejectedTags.length
        ? 'llm_rejected_rule'
        : confirmedRuleHits.length || reclassifiedTopic
          ? 'rule+llm'
          : localRuleHits.length
            ? 'llm_with_rule_precheck'
            : 'llm_only'
    return {
      rowKey,
      rowIndex,
      classification,
      newClassification,
      sourceClassification,
      isReclassified,
      tags: finalTags,
      localTags,
      modelAddedTags,
      modelRejectedTags,
      matchType,
      evidence,
      tagDecisions,
      ruleHits: localRuleHits,
      suggestedTopics: data.rule_evaluation?.suggested_topics ?? [],
      confirmedRuleHits,
      rejectedRuleHits,
      ruleDecision,
      modelDecision,
      reclassifiedTopic,
      finalTopic,
      decisionSource,
      responseRaw: reclassificationData
        ? `${data.response_raw}\n\n--- reclassification ---\n${reclassificationData.response_raw}`
        : data.response_raw,
      responseJson: reclassificationData
        ? { classification: data.response_json, reclassification: reclassificationData.response_json }
        : data.response_json,
      sourceRow: row,
    }
  }

  const applyBackgroundTaskResult = (data: GigaChatBackgroundTaskResultResponse) => {
    setSheetData(data.workbook)
    setViewSheetData(data.workbook)
    setWorkbookMeta({
      upload_id: data.workbook.upload_id,
      filename: data.workbook.filename,
      file_format: data.workbook.file_format,
      sheet_count: 1,
      sheets: [{
        name: data.workbook.sheet_name,
        rows_total: data.workbook.total_rows,
        column_count: data.workbook.columns.length,
        columns: data.workbook.columns,
        preview_rows: data.workbook.rows.slice(0, 5),
      }],
    })
    setIncludedPromptColumns([...data.workbook.columns])
    setWorkbookRowLimit('all')
    setWorkbookCollapsed(false)
    setUploadCollapsed(true)
    setAnnotatedRows(data.row_runs.flatMap((rowRun) => rowRun.result
      ? [buildAnnotatedRow(rowRun.result, rowRun.row_index, rowRun.source_row, data.workbook, rowRun.reclassification_result ?? null)]
      : []))
    setBackgroundTaskError(null)
    window.setTimeout(() => document.getElementById('gigachat-workbook-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50)
  }

  const buildExcludedRuleHitMap = (
    rows: Array<{ rowIndex: number; row: Record<string, unknown> }>,
    valuesForRun: Record<string, unknown>,
  ) => {
    const excludedCodes = new Set(parseRulePackExclusions(valuesForRun[RULE_PACK_EXCLUSION_SETTING_KEY]))
    if (!excludedCodes.size || !rows.length) return new Map<number, string[]>()
    const evaluations = evaluateRulePacksLocally(valuesForRun, rows.map((item) => item.row)).evaluations
    const excludedByRow = new Map<number, string[]>()
    evaluations.forEach((evaluation, localIndex) => {
      const excludedHits = evaluation.hits
        .map((hit) => hit.code)
        .filter((code) => excludedCodes.has(code))
      if (excludedHits.length) {
        excludedByRow.set(rows[localIndex].rowIndex, Array.from(new Set(excludedHits)))
      }
    })
    return excludedByRow
  }

  const loadBackgroundTaskResult = useMutation({
    mutationFn: (taskId: string) => {
      setBackgroundResultProgress({ loadedBytes: 0, totalBytes: null })
      return getGigaChatBackgroundTaskResultWithProgress(taskId, (loadedBytes, totalBytes) => {
        setBackgroundResultProgress({ loadedBytes, totalBytes })
      })
    },
    onSuccess: (data) => {
      applyBackgroundTaskResult(data)
      setSearchParams((current) => {
        current.delete('backgroundTaskId')
        return current
      }, { replace: true })
    },
    onError: (error) => setBackgroundTaskError(formatLabError(error as Error, 'результат фоновой задачи')),
    onSettled: () => setBackgroundResultProgress(null),
  })

  const startBackgroundTask = useMutation({
    mutationFn: async (valuesForRun?: Record<string, unknown>) => {
      if (!selected?.ready) throw new Error(`Транспорт ${selected?.title ?? selectedTransport} сейчас не готов к отправке.`)
      if (!sheetData) throw new Error('Сначала загрузите рабочую таблицу.')
      const selectedRows = sheetData.rows
        .map((row, index) => ({ row_index: index, source_row: row }))
        .filter((item) => selectedSheetRowKeySet.has(buildSheetRowKey(sheetData.upload_id, sheetData.sheet_name, item.row_index)))
      if (!selectedRows.length) throw new Error('Сначала выберите хотя бы одну строку.')
      const baseValues = valuesForRun ?? settingValues
      const excludedRows = buildExcludedRuleHitMap(
        selectedRows.map((item) => ({ rowIndex: item.row_index, row: item.source_row })),
        baseValues,
      )
      const rowsForGigaChat = selectedRows.filter((item) => !excludedRows.has(item.row_index))
      if (!rowsForGigaChat.length) {
        throw new Error('Все выбранные строки исключены из отправки в GigaChat по настройкам исключений.')
      }
      const useReclassification = shouldSendReclassificationRequest(baseValues)
      const runValues = useReclassification
        ? withRunnableReclassificationRules(baseValues, sheetData.columns)
        : baseValues
      return startGigaChatBackgroundTask({
        transport: selectedTransport,
        values: runValues,
        columns: finalPromptColumns,
        rows: rowsForGigaChat,
        filename: sheetData.filename,
        sheet_name: sheetData.sheet_name,
        payload_override: useReclassification ? null : resolvePayloadOverride(),
        count_tokens: tokenAccounting.enabled,
        reclassification_enabled: useReclassification,
        async_workers: asyncWorkerCount,
      })
    },
    onSuccess: () => {
      setBackgroundTaskError(null)
      void qc.invalidateQueries({ queryKey: ['gigachat-background-tasks'] })
      navigate('/gigachat/background')
    },
    onError: (error) => setBackgroundTaskError(formatLabError(error as Error, 'запуск фоновой задачи')),
  })

  useEffect(() => {
    if (!selectedVersionQ.data || selectedVersionQ.isLoading) return
    if (lastAutoPreviewKeyRef.current === finalPromptRequestKey) return
    const timer = window.setTimeout(() => {
      lastAutoPreviewKeyRef.current = finalPromptRequestKey
      previewFinalPrompt.mutate({ requestKey: finalPromptRequestKey })
    }, 350)
    return () => window.clearTimeout(timer)
  }, [finalPromptRequestKey, selectedVersionQ.data, selectedVersionQ.isLoading])

  useEffect(() => {
    if (!sheetData) return
    if (!includedPromptColumns.length) {
      setIncludedPromptColumns([...sheetData.columns])
    }
  }, [sheetData, includedPromptColumns.length])

  useEffect(() => {
    if (!reclassificationRequestsAvailable && sendReclassificationRequests) {
      setSendReclassificationRequests(false)
    }
  }, [reclassificationRequestsAvailable, sendReclassificationRequests])

  useEffect(() => {
    if (!sheetData) {
      setSelectedSheetRowKeys([])
      setRuleEvaluationMap({})
      return
    }
    const available = new Set(sheetData.rows.map((_, index) => buildSheetRowKey(sheetData.upload_id, sheetData.sheet_name, index)))
    setSelectedSheetRowKeys((current) => current.filter((key) => available.has(key)))
  }, [sheetData])

  useEffect(() => {
    if (!sheetData) return
    const timer = window.setTimeout(() => {
      evaluateRulePacks.mutate(sheetData.rows)
    }, 250)
    return () => window.clearTimeout(timer)
  }, [sheetData, settingValues.rule_pack_prompt_notes])

  useEffect(() => {
    const taskId = searchParams.get('backgroundTaskId')
    if (taskId && !loadBackgroundTaskResult.isPending) {
      loadBackgroundTaskResult.mutate(taskId)
    }
  }, [searchParams])

  const appendAnnotatedResult = (
    data: GigaChatLabRowRunResponse,
    rowIndex: number,
    row: Record<string, unknown>,
    reclassificationData?: GigaChatLabRowRunResponse | null,
  ) => {
    if (!sheetData) return
    const annotatedRow = buildAnnotatedRow(data, rowIndex, row, sheetData, reclassificationData)
    setAnnotatedRows((current) => {
      const next = current.filter((item) => item.rowKey !== annotatedRow.rowKey)
      next.unshift(annotatedRow)
      return next
    })
  }

  const buildRowRunResultPages = (
    data: GigaChatLabRowRunResponse,
    reclassificationData?: GigaChatLabRowRunResponse | null,
  ): RowRunResultPage[] => [
    {
      id: 'classification',
      title: '1. Основной запрос',
      description: 'Классификация строки, rule hits и теги.',
      result: data,
    },
    ...(reclassificationData ? [{
      id: 'reclassification' as const,
      title: '2. Переклассификация',
      description: 'Отдельный второй запрос только для проверки и замены темы.',
      result: reclassificationData,
    }] : []),
  ]

  const resolvePayloadOverride = () => (
    finalPromptOverrideText
      ? (JSON.parse(finalPromptOverrideText) as Record<string, unknown>)
      : finalPromptPreview?.payload ?? null
  )

  const startProcessingOverlay = (title: string, total: number) => {
    setProcessingOverlay({
      title,
      completed: 0,
      total,
      minimized: false,
      currentLabel: total > 1 ? `Подготовили 0 из ${total}` : 'Готовим запрос к GigaChat...',
    })
  }

  const updateProcessingOverlay = (completed: number, total: number, currentLabel: string) => {
    setProcessingOverlay((current) => current ? {
      ...current,
      completed,
      total,
      currentLabel,
    } : current)
  }

  const finishProcessingOverlay = () => {
    setProcessingOverlay(null)
  }

  const accumulateTokenCount = (tokenCount: number | null | undefined) => {
    if (!tokenAccounting.enabled || !tokenCount || tokenCount <= 0) return
    setTokenAccounting((current) => ({
      ...current,
      totalTokens: current.totalTokens + tokenCount,
    }))
  }

  const runWorkbookRowWithOptionalReclassification = async (
    row: Record<string, unknown>,
    valuesForRun: Record<string, unknown>,
    payloadOverride: Record<string, unknown> | null,
    onPrimaryComplete?: (data: GigaChatLabRowRunResponse) => void,
  ): Promise<{
    data: GigaChatLabRowRunResponse
    reclassificationData: GigaChatLabRowRunResponse | null
    reclassificationEnabled: boolean
  }> => {
    const reclassificationEnabled = shouldSendReclassificationRequest(valuesForRun)
    const reclassificationValues = reclassificationEnabled
      ? withRunnableReclassificationRules(valuesForRun, sheetData?.columns ?? null)
      : valuesForRun
    const primaryValues = reclassificationEnabled ? withoutReclassificationSettings(reclassificationValues) : valuesForRun
    const primaryPayloadOverride = reclassificationEnabled ? null : payloadOverride
    const data = await runGigaChatWorkbookRow(
      selectedTransport,
      primaryValues,
      finalPromptColumns,
      row,
      primaryPayloadOverride,
      tokenAccounting.enabled,
    )
    accumulateTokenCount(data.request_token_count)
    onPrimaryComplete?.(data)

    let reclassificationData: GigaChatLabRowRunResponse | null = null
    if (reclassificationEnabled) {
      reclassificationData = await runGigaChatWorkbookRow(
        selectedTransport,
        reclassificationValues,
        finalPromptColumns,
        row,
        null,
        tokenAccounting.enabled,
        true,
      )
      accumulateTokenCount(reclassificationData.request_token_count)
    }

    return { data, reclassificationData, reclassificationEnabled }
  }

  const runWithRuleGuard = (action: PendingRuleGuardAction) => {
    if (!sheetData) {
      void action(settingValues)
      return
    }
    const issues = findRuleGuardIssues(settingValues, sheetData.columns)
    if (!issues.length) {
      void action(settingValues)
      return
    }
    setRuleGuardIssues(issues)
    setPendingRuleGuardValues(disableRulesWithMissingFields(settingValues, sheetData.columns))
    setPendingRuleGuardAction(() => action)
  }

  const confirmRuleGuard = () => {
    if (!pendingRuleGuardAction || !pendingRuleGuardValues) return
    const action = pendingRuleGuardAction
    const values = pendingRuleGuardValues
    setSettingValues(values)
    setRuleGuardIssues([])
    setPendingRuleGuardAction(null)
    setPendingRuleGuardValues(null)
    void action(values)
  }

  const cancelRuleGuard = () => {
    setRuleGuardIssues([])
    setPendingRuleGuardAction(null)
    setPendingRuleGuardValues(null)
  }

  const handleRunRow = async (row: Record<string, unknown>, rowIndex: number, valuesForRun = settingValues) => {
    if (!selected?.ready) {
      setRowRunError(`Транспорт ${selected?.title ?? selectedTransport} сейчас не готов к отправке. Сначала выберите ready-вариант подключения.`)
      return
    }
    setRowRunError(null)
    const excludedHits = buildExcludedRuleHitMap([{ rowIndex, row }], valuesForRun).get(rowIndex) ?? []
    if (excludedHits.length) {
      setRowRunError(`Строка не отправлена в GigaChat: сработали исключенные rule packs ${excludedHits.join(', ')}.`)
      return
    }
    setRunRowBusyIndex(rowIndex)
    const useReclassification = shouldSendReclassificationRequest(valuesForRun)
    const totalSteps = useReclassification ? 2 : 1
    startProcessingOverlay('Обработка одной жалобы', totalSteps)
    try {
      const { data, reclassificationData } = await runWorkbookRowWithOptionalReclassification(
        row,
        valuesForRun,
        resolvePayloadOverride(),
        () => {
          if (useReclassification) {
            updateProcessingOverlay(1, totalSteps, 'Классификация готова. Переклассифицируем тему...')
          }
        },
      )
      appendAnnotatedResult(data, rowIndex, row, reclassificationData)
      updateProcessingOverlay(totalSteps, totalSteps, 'Жалоба обработана. Открываем результат...')
      setRowRunResultPages(buildRowRunResultPages(data, reclassificationData))
      setActiveRowRunResultPage(0)
      setRowRunModalOpen(true)
    } catch (error) {
      setRowRunError(formatLabError(error as Error, 'строку'))
    } finally {
      setRunRowBusyIndex(null)
      finishProcessingOverlay()
    }
  }

  const handleRunRowIndexes = async (rowIndexes: number[], emptyMessage: string, errorContext: string, valuesForRun = settingValues) => {
    if (!selected?.ready) {
      setBatchRunError(`Транспорт ${selected?.title ?? selectedTransport} сейчас не готов к отправке. Сначала выберите ready-вариант подключения.`)
      return
    }
    if (!sheetData) {
      setBatchRunError('Сначала загрузите рабочую таблицу.')
      return
    }

    if (!rowIndexes.length) {
      setBatchRunError(emptyMessage)
      return
    }

    const excludedRows = buildExcludedRuleHitMap(
      rowIndexes.map((rowIndex) => ({ rowIndex, row: sheetData.rows[rowIndex] })),
      valuesForRun,
    )
    const runnableRowIndexes = rowIndexes.filter((rowIndex) => !excludedRows.has(rowIndex))
    if (!runnableRowIndexes.length) {
      setBatchRunError('Все выбранные строки исключены из отправки в GigaChat по настройкам исключений.')
      return
    }

    setBatchRunError(null)
    setBatchBusy(true)
    const useReclassification = shouldSendReclassificationRequest(valuesForRun)
    const stepsPerRow = useReclassification ? 2 : 1
    const totalSteps = runnableRowIndexes.length * stepsPerRow
    startProcessingOverlay('Пакетная разметка жалоб', totalSteps)

    try {
      const payloadOverride = resolvePayloadOverride()
      let lastResultPages: RowRunResultPage[] = []
      let completedSteps = 0
      let completedRows = 0
      let firstError: Error | null = null
      const workerCount = Math.min(asyncWorkerCount, runnableRowIndexes.length)
      await runWithConcurrency(runnableRowIndexes, asyncWorkerCount, async (rowIndex, idx) => {
        if (firstError) return
        const row = sheetData.rows[rowIndex]
        updateProcessingOverlay(
          completedSteps,
          totalSteps,
          `В работе ${workerCount} workers. Обрабатываем запись ${idx + 1} из ${runnableRowIndexes.length}`,
        )
        try {
          const { data, reclassificationData } = await runWorkbookRowWithOptionalReclassification(
            row,
            valuesForRun,
            payloadOverride,
            () => {
              completedSteps += 1
              updateProcessingOverlay(
                completedSteps,
                totalSteps,
                useReclassification
                  ? `Основной запрос готов. Переклассифицируем тему ${idx + 1} из ${runnableRowIndexes.length}`
                  : `Готово запросов: ${completedSteps} из ${totalSteps}`,
              )
            },
          )
          if (reclassificationData) {
            completedSteps += 1
          }
          appendAnnotatedResult(data, rowIndex, row, reclassificationData)
          lastResultPages = buildRowRunResultPages(data, reclassificationData)
          completedRows += 1
          updateProcessingOverlay(
            completedSteps,
            totalSteps,
            tokenAccounting.enabled && data.request_token_count
              ? `Готово ${completedRows} из ${runnableRowIndexes.length}. Workers: ${workerCount}. Последний основной запрос: ${data.request_token_count} токенов`
              : `Готово ${completedRows} из ${runnableRowIndexes.length}. Workers: ${workerCount}`,
          )
        } catch (error) {
          if (!firstError) firstError = error as Error
        }
      })
      if (firstError) throw firstError
      if (lastResultPages.length) {
        setRowRunResultPages(lastResultPages)
        setActiveRowRunResultPage(0)
      }
    } catch (error) {
      setBatchRunError(formatLabError(error as Error, errorContext))
    } finally {
      setBatchBusy(false)
      finishProcessingOverlay()
    }
  }

  const handleRunSelectedRows = async (valuesForRun = settingValues) => {
    if (!sheetData) {
      setBatchRunError('Сначала загрузите рабочую таблицу.')
      return
    }
    const rowIndexes = sheetData.rows
      .map((_, index) => index)
      .filter((index) => selectedSheetRowKeySet.has(buildSheetRowKey(sheetData.upload_id, sheetData.sheet_name, index)))
    await handleRunRowIndexes(rowIndexes, 'Сначала выберите хотя бы одну строку.', 'выбранные строки', valuesForRun)
  }

  const handleRunAllRows = async (valuesForRun = settingValues) => {
    if (!sheetData) {
      setBatchRunError('Сначала загрузите рабочую таблицу.')
      return
    }
    await handleRunRowIndexes(
      sheetData.rows.map((_, index) => index),
      'В текущем листе нет строк для отправки.',
      'все строки',
      valuesForRun,
    )
  }

  const toggleTransportCard = (name: GigaChatTransportName) => {
    setCollapsedTransports((current) => ({ ...current, [name]: !current[name] }))
  }

  const transports = statusQ.data?.transports ?? []
  const selected = transports.find((item) => item.name === selectedTransport) ?? transports[0]
  const hasMultipleSheets = (workbookMeta?.sheet_count ?? 0) > 1
  const selectedSheetName = sheetData?.sheet_name ?? null
  const viewSheetName = viewSheetData?.sheet_name ?? null
  const displayedSheetData = viewSheetData ?? sheetData
  const displayedSheetIsWorking = Boolean(displayedSheetData && sheetData && displayedSheetData.upload_id === sheetData.upload_id && displayedSheetData.sheet_name === sheetData.sheet_name)
  const ruleEvaluationTotal = ruleEvaluationProgress?.total ?? sheetData?.rows.length ?? 0
  const ruleEvaluationProcessed = ruleEvaluationProgress?.processed ?? 0
  const ruleEvaluationRemaining = Math.max(0, ruleEvaluationTotal - ruleEvaluationProcessed)
  const ruleEvaluationPercent = ruleEvaluationTotal ? Math.round((ruleEvaluationProcessed / ruleEvaluationTotal) * 100) : 0
  const ruleEvaluationBusy = evaluateRulePacks.isPending || Boolean(ruleEvaluationProgress)
  const backgroundLoadedBytes = backgroundResultProgress?.loadedBytes ?? 0
  const backgroundTotalBytes = backgroundResultProgress?.totalBytes ?? null
  const backgroundLoadPercent = backgroundTotalBytes ? Math.min(100, Math.round((backgroundLoadedBytes / backgroundTotalBytes) * 100)) : null
  const exportLoadedBytes = exportProgress?.loadedBytes ?? 0
  const exportTotalBytes = exportProgress?.totalBytes ?? null
  const exportPercent = exportTotalBytes ? Math.min(100, Math.round((exportLoadedBytes / exportTotalBytes) * 100)) : null
  const uploadPercent = workbookUploadProgress ? Math.min(100, Math.max(3, Math.round(workbookUploadProgress.progress * 100))) : null
  const exportBusy = exportWorkbookRows.isPending || exportAnnotatedRows.isPending || exportValidationRows.isPending
  const modelOptionNames = Array.from(new Set([
    ...((modelOptionsQ.data?.ok ? modelOptionsQ.data.models : []) ?? []),
    ...((probe.data?.ok ? probe.data.models : []) ?? []),
    String(settingValues.model ?? '').trim(),
    String(selectedVersionQ.data?.values?.model ?? '').trim(),
    String(statusQ.data?.model ?? '').trim(),
  ].filter(Boolean)))
  const requestSettingsFields = (selectedVersionQ.data?.fields ?? [])
    .filter((field) => !hiddenLabSettingKeys.has(field.key))
    .map((field) => {
      if (field.key !== 'model') return field
      const modelsHelpText = modelOptionsQ.isFetching
        ? 'Загружаем актуальный список моделей из API...'
        : modelOptionsQ.isError
          ? 'Не удалось загрузить список моделей из API. Оставлена текущая модель.'
          : 'Список моделей загружается из API выбранного transport.'
      return {
        ...field,
        input_type: 'select' as const,
        help_text: modelsHelpText,
        options: modelOptionNames.map((modelName) => ({ value: modelName, label: modelName })),
      }
    })
  const finalPromptDraftDirty = finalPromptDraftText !== finalPromptBaseText
  const finalPromptDraftValidation = useMemo(() => {
    const text = finalPromptDraftText.trim()
    if (!text) return { error: 'JSON payload пустой.', parsed: null as Record<string, unknown> | null }
    try {
      const parsed = JSON.parse(text) as unknown
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        return { error: 'Нужен валидный JSON-объект.', parsed: null }
      }
      return { error: null as string | null, parsed: parsed as Record<string, unknown> }
    } catch (error) {
      return { error: `JSON невалиден: ${(error as Error).message}`, parsed: null }
    }
  }, [finalPromptDraftText])
  const finalPromptSynchronized = lastResolvedPreviewKeyRef.current === finalPromptRequestKey && !previewFinalPrompt.isPending
  const visibleSheetRowKeys = useMemo(
    () => (sheetData ? sheetData.rows.map((_, index) => buildSheetRowKey(sheetData.upload_id, sheetData.sheet_name, index)) : []),
    [sheetData],
  )
  const allVisibleRowsSelected = visibleSheetRowKeys.length > 0 && visibleSheetRowKeys.every((key) => selectedSheetRowKeySet.has(key))
  const annotatedClassOptions = useMemo(
    () => Array.from(new Set(annotatedRows.map((row) => row.classification).filter(Boolean))).sort(),
    [annotatedRows],
  )
  const annotatedTagOptions = useMemo(
    () => Array.from(new Set(annotatedRows.flatMap((row) => row.tags))).sort(),
    [annotatedRows],
  )
  const annotatedRuleOptions = useMemo(
    () => Array.from(new Set(annotatedRows.flatMap((row) => row.ruleHits))).sort(),
    [annotatedRows],
  )
  const annotatedDecisionSourceOptions = useMemo(
    () => Array.from(new Set(annotatedRows.map((row) => row.decisionSource).filter(Boolean))).sort(),
    [annotatedRows],
  )
  const filteredAnnotatedRows = useMemo(
    () => annotatedRows.filter((row) => {
      const classMatch = !annotatedClassFilter.length || annotatedClassFilter.includes(row.classification)
      const tagMatch = !annotatedTagFilter.length || row.tags.some((tag) => annotatedTagFilter.includes(tag))
      const ruleMatch = !annotatedRuleFilter.length || row.ruleHits.some((ruleCode) => annotatedRuleFilter.includes(ruleCode))
      const sourceMatch = !annotatedDecisionSourceFilter.length || annotatedDecisionSourceFilter.includes(row.decisionSource)
      return classMatch && tagMatch && ruleMatch && sourceMatch
    }),
    [annotatedRows, annotatedClassFilter, annotatedTagFilter, annotatedRuleFilter, annotatedDecisionSourceFilter],
  )
  const reclassifiedAnnotatedRows = useMemo(
    () => filteredAnnotatedRows.filter((row) => isAnnotatedRowReclassified(row)),
    [filteredAnnotatedRows],
  )
  const virtualAnnotatedTable = useVirtualTableRows(filteredAnnotatedRows, annotatedTable.rowClamp === 'all' ? 148 : 112)
  const annotatedTableColumnCount = 14 + (sheetData?.columns.length ?? 0)
  const totalTokenCost = useMemo(
    () => (tokenAccounting.totalTokens / 1000) * tokenAccounting.pricePer1k,
    [tokenAccounting.totalTokens, tokenAccounting.pricePer1k],
  )
  const ruleEvaluationSummary = useMemo(() => {
    const summary = new Map<string, {
      code: string
      count: number
      keywords: Set<string>
      fields: Set<string>
      suggested: Set<string>
    }>()
    Object.values(ruleEvaluationMap).forEach((evaluation) => {
      evaluation.hits.forEach((hit) => {
        const current = summary.get(hit.code) ?? {
          code: hit.code,
          count: 0,
          keywords: new Set<string>(),
          fields: new Set<string>(),
          suggested: new Set<string>(),
        }
        current.count += 1
        hit.matched_keywords.forEach((keyword) => current.keywords.add(keyword))
        hit.matched_fields.forEach((field) => current.fields.add(field))
        if (hit.target_tag) current.suggested.add(`tag:${hit.target_tag}`)
        if (hit.target_topic) current.suggested.add(`topic:${hit.target_topic}`)
        summary.set(hit.code, current)
      })
    })
    return Array.from(summary.values()).sort((left, right) => right.count - left.count || left.code.localeCompare(right.code))
  }, [ruleEvaluationMap])
  const ruleHitRowCount = useMemo(
    () => Object.values(ruleEvaluationMap).filter((evaluation) => evaluation.hits.length).length,
    [ruleEvaluationMap],
  )
  const parsedRulePacks = useMemo(
    () => parseRulePacks(settingValues.rule_pack_prompt_notes),
    [settingValues.rule_pack_prompt_notes],
  )
  const rulePackOptions = useMemo(
    () => parsedRulePacks.map((item) => item.code).filter(Boolean),
    [parsedRulePacks],
  )
  const rulePackByCode = useMemo(
    () => new Map(parsedRulePacks.map((item) => [item.code, item])),
    [parsedRulePacks],
  )
  const rulePackExclusionCodes = useMemo(
    () => rulePackExclusionCodesDraft,
    [rulePackExclusionCodesDraft],
  )
  const rulePackExclusionSet = useMemo(() => new Set(rulePackExclusionCodes), [rulePackExclusionCodes])
  const availableRulePackExclusionOptions = useMemo(
    () => parsedRulePacks.filter((item) => item.code && !rulePackExclusionSet.has(item.code)),
    [parsedRulePacks, rulePackExclusionSet],
  )
  const activeRulePackExclusionDraft = rulePackExclusionDraft
  const excludedRulePackEntries = useMemo(
    () => rulePackExclusionCodes.map((code) => ({ code, rule: rulePackByCode.get(code) ?? null })),
    [rulePackByCode, rulePackExclusionCodes],
  )
  useEffect(() => {
    const nextDraft = availableRulePackExclusionOptions[0]?.code ?? ''
    if (!rulePackExclusionDraft) {
      if (nextDraft) setRulePackExclusionDraft(nextDraft)
      return
    }
    if (availableRulePackExclusionOptions.some((item) => item.code === rulePackExclusionDraft)) return
    setRulePackExclusionDraft(nextDraft)
  }, [availableRulePackExclusionOptions, rulePackExclusionDraft])
  const ruleValidationSummary = useMemo(() => {
    if (!sheetData) return null
    const hasExpectedHits = sheetData.columns.includes('Expected rule hits')
    const hasExpectedTags = sheetData.columns.includes('Expected tags')
    const hasExpectedTopic = sheetData.columns.includes('Expected final topic')
    if (!hasExpectedHits && !hasExpectedTags && !hasExpectedTopic) {
      return {
        available: false,
        total: sheetData.rows.length,
        passed: 0,
        failed: 0,
        reason: 'В текущем листе нет эталонных колонок Expected rule hits / Expected tags / Expected final topic.',
      }
    }
    const splitExpected = (value: unknown) => String(value ?? '')
      .replace(/,/gu, ';')
      .split(';')
      .map((item) => item.trim())
      .filter(Boolean)
    let passed = 0
    sheetData.rows.forEach((row, index) => {
      const evaluation = ruleEvaluationMap[index]
      const actualHits = evaluation?.hits.map((hit) => hit.code) ?? []
      const actualTags = evaluation?.suggested_tags ?? []
      const actualTopic = evaluation?.suggested_topics?.[0] ?? ''
      const expectedHits = hasExpectedHits ? splitExpected(row['Expected rule hits']) : actualHits
      const expectedTags = hasExpectedTags ? splitExpected(row['Expected tags']) : actualTags
      const expectedTopic = hasExpectedTopic ? String(row['Expected final topic'] ?? '').trim() : actualTopic
      const sameHits = actualHits.join('|') === expectedHits.join('|')
      const sameTags = actualTags.join('|') === expectedTags.join('|')
      const sameTopic = actualTopic === expectedTopic
      if (sameHits && sameTags && sameTopic) passed += 1
    })
    return {
      available: true,
      total: sheetData.rows.length,
      passed,
      failed: sheetData.rows.length - passed,
      reason: '',
    }
  }, [ruleEvaluationMap, sheetData])
  const handleEvaluateRules = () => {
    if (!sheetData) {
      setRuleEvaluationError('Сначала загрузите лист Excel, чтобы прогнать rule packs по строкам.')
      return
    }
    setRuleEvaluationProgress({ processed: 0, total: sheetData.rows.length })
    window.setTimeout(() => evaluateRulePacks.mutate(sheetData.rows), 0)
  }

  const activeRowRunPageIndex = Math.min(activeRowRunResultPage, Math.max(0, rowRunResultPages.length - 1))
  const activeRowRunPage = rowRunResultPages[activeRowRunPageIndex] ?? null
  const activeRowRunResult = activeRowRunPage?.result ?? null

  return <div className='transport-page'>
    {loadBackgroundTaskResult.isPending ? <div className='background-result-lock' role='status' aria-live='polite'>
      <div className='card background-result-lock-card'>
        <div className='spinner workbook-upload-spinner' aria-hidden='true' />
        <div className='giga-processing-progress-copy'>
          <div className='giga-processing-progress-title'>Загружаем рабочую тетрадь</div>
          <div className='lab-muted'>
            {backgroundLoadPercent !== null
              ? `Получено ${formatBytes(backgroundLoadedBytes)} из ${formatBytes(backgroundTotalBytes ?? 0)}. Осталось ${formatBytes(Math.max(0, (backgroundTotalBytes ?? 0) - backgroundLoadedBytes))}.`
              : `Получено ${formatBytes(backgroundLoadedBytes)}. Размер ответа уточняется.`}
          </div>
        </div>
        <div className='giga-processing-progressbar' aria-label='background result loading progress'>
          <div
            className='giga-processing-progressbar-fill'
            style={{ width: `${backgroundLoadPercent ?? Math.min(95, Math.max(8, Math.round(backgroundLoadedBytes / 80_000)))}%` }}
          />
        </div>
      </div>
    </div> : null}

    <section className='transport-hero card'>
      <div className='transport-section-head'>
        <div className='transport-section-title'>
          <h2>Лаборатория разметки GigaChat</h2>
          <p>Страница для экспериментов с разметкой через API: выбор транспорта, настройка промптов и контекста, загрузка Excel и просмотр данных перед запуском разметки.</p>
        </div>
        <button className='transport-collapse-button' onClick={() => setHeroCollapsed((current) => !current)}>
          {heroCollapsed ? 'Развернуть' : 'Свернуть'}
        </button>
      </div>
      {!heroCollapsed ? <>
        <div className='transport-summary'>
          <div><b>Configured mode:</b> <code>{statusQ.data?.configured_mode ?? '—'}</code></div>
          <div><b>Model:</b> <code>{statusQ.data?.model ?? '—'}</code></div>
          <div><b>Selected transport:</b> <code>{selected?.title ?? selectedTransport}</code></div>
        </div>
      </> : null}
    </section>

    <div className='lab-main-tabs' role='tablist' aria-label='GigaChat Lab tabs'>
      <button
        type='button'
        className={activeLabTab === 'workspace' ? 'active' : ''}
        onClick={() => setActiveLabTab('workspace')}
      >
        Рабочая тетрадь
      </button>
      <button
        type='button'
        className={activeLabTab === 'settings' ? 'active' : ''}
        onClick={() => setActiveLabTab('settings')}
      >
        Настройки
      </button>
    </div>

    {activeLabTab === 'settings' ? <>
    <section className='card transport-result'>
      <div className='transport-section-head'>
        <div className='transport-section-title'>
          <h3>Токены и стоимость</h3>
          <p>Можно включить подсчет токенов через API GigaChat перед каждой отправкой. Счетчик суммирует все успешные запросы на текущей странице.</p>
        </div>
      </div>

      <div className='token-accounting-grid'>
        <label className='lab-checkbox token-accounting-toggle'>
          <input
            type='checkbox'
            checked={tokenAccounting.enabled}
            onChange={(e) => setTokenAccounting((current) => ({ ...current, enabled: e.target.checked }))}
          />
          <span>
            <strong>Подсчитывать токены перед отправкой</strong>
            <small>Перед каждым запросом в GigaChat будет вызван API подсчета токенов для текущего payload.</small>
          </span>
        </label>

        <label className='lab-field'>
          <span>Цена за 1000 токенов</span>
          <input
            type='number'
            min='0'
            step='0.0001'
            value={String(tokenAccounting.pricePer1k)}
            onChange={(e) => setTokenAccounting((current) => ({
              ...current,
              pricePer1k: Number.isFinite(Number(e.target.value)) ? Number(e.target.value) : 0,
            }))}
          />
          <small className='lab-field-help'>Используется только для пересчета в деньги. Валюта любая, которую вы для себя принимаете.</small>
        </label>

        <div className='token-accounting-stats'>
          <div className='token-accounting-stat card'>
            <div className='lab-muted'>Потрачено токенов</div>
            <strong>{tokenAccounting.totalTokens.toLocaleString('ru-RU')}</strong>
          </div>
          <div className='token-accounting-stat card'>
            <div className='lab-muted'>Потрачено денег</div>
            <strong>{totalTokenCost.toLocaleString('ru-RU', { minimumFractionDigits: 2, maximumFractionDigits: 4 })}</strong>
          </div>
        </div>

        <div className='transport-actions'>
          <button
            type='button'
            onClick={() => setTokenAccounting((current) => ({ ...current, totalTokens: 0 }))}
            disabled={!tokenAccounting.totalTokens}
          >
            Сбросить счетчик токенов
          </button>
        </div>
      </div>
    </section>

    {statusQ.isLoading ? <div className='card'>Загружаем состояние транспортов...</div> : null}
    {statusQ.isError ? <div className='card'>Ошибка загрузки transport status: {(statusQ.error as Error).message}</div> : null}

    {transports.length > 0 ? <div className='transport-grid'>
      {transports.map((transport) => <GigaChatTransportCard
        key={transport.name}
        status={transport}
        selected={transport.name === selectedTransport}
        busy={probe.isPending && probe.variables === transport.name}
        collapsed={collapsedTransports[transport.name]}
        onSelect={setSelectedTransport}
        onProbe={(name) => probe.mutate(name)}
        onToggle={toggleTransportCard}
      />)}
    </div> : null}

    <section className='card transport-result async-requests-card'>
      <div className='transport-section-head'>
        <div className='transport-section-title'>
          <h3>Асинхронные запросы</h3>
          <p>Количество параллельных запросов к GigaChat для пакетной и фоновой разметки.</p>
        </div>
        <span className='async-workers-badge'>{asyncWorkerCount} workers</span>
      </div>
      <div className='async-workers-grid'>
        <label className='lab-field async-workers-field'>
          <span>Воркеров</span>
          <input
            type='number'
            min={1}
            max={MAX_ASYNC_WORKERS}
            step={1}
            value={asyncWorkerCount}
            onChange={(event) => setAsyncWorkerCount(clampAsyncWorkerCount(event.target.value))}
          />
          <small className='lab-field-help'>1 = последовательный режим. Настройка применяется к выбранному транспорту: {selected?.title ?? selectedTransport}.</small>
        </label>
        <label className='lab-field async-workers-slider'>
          <span>Параллельность</span>
          <input
            type='range'
            min={1}
            max={ASYNC_WORKER_SLIDER_MAX}
            step={1}
            value={Math.min(asyncWorkerCount, ASYNC_WORKER_SLIDER_MAX)}
            onChange={(event) => setAsyncWorkerCount(clampAsyncWorkerCount(event.target.value))}
          />
          <small className='lab-field-help'>В обычном режиме запросы идут асинхронно. При HTTP 429 backend переводит ожидающие запросы в одну общую очередь с concurrency = 1.</small>
        </label>
        <div className='async-workers-summary'>
          <strong>{asyncWorkerCount === 1 ? 'Последовательно' : 'Параллельно'}</strong>
          <span>{asyncWorkerCount === 1 ? 'Следующий запрос стартует после завершения предыдущего.' : `До ${asyncWorkerCount} строк могут быть в работе одновременно. HTTP 429 включает общую последовательную очередь и записывается в backend log.`}</span>
        </div>
      </div>
    </section>

    <section className='card transport-result' id='gigachat-workbook-section'>
      <div className='transport-section-head'>
        <div className='transport-section-title'>
          <h3>Результат проверки</h3>
          <p>Последний ответ по выбранному транспорту, включая сообщение API и список доступных моделей.</p>
        </div>
        <button className='transport-collapse-button' onClick={() => setResultCollapsed((current) => !current)}>
          {resultCollapsed ? 'Развернуть' : 'Свернуть'}
        </button>
      </div>

      {!resultCollapsed ? <>
        {!probe.data && !probe.isPending ? <p>Нажмите на `Проверить ...`, чтобы запросить список моделей и проверить transport.</p> : null}
        {probe.isError ? <div className='transport-error'>{(probe.error as Error).message}</div> : null}
        {probe.data ? <div className='transport-result-grid'>
          <div><b>Transport:</b> <code>{probe.data.transport}</code></div>
          <div><b>API base URL:</b> <code>{probe.data.base_url}</code></div>
          {probe.data.oauth_url ? <div><b>OAuth URL:</b> <code>{probe.data.oauth_url}</code></div> : null}
          <div><b>Model:</b> <code>{probe.data.model}</code></div>
          <div className={probe.data.ok ? '' : 'transport-error'}><b>Message:</b> {probe.data.message}</div>
          <div>
            <b>Models:</b>
            {probe.data.ok && probe.data.models.length ? <div className='transport-model-list'>
              {probe.data.models.map((modelName) => <span key={modelName} className='transport-model-pill'>{modelName}</span>)}
            </div> : <span className={probe.data.ok ? '' : 'transport-error'}>
              {probe.data.ok ? 'API вернул пустой список моделей' : 'недоступны: проверка соединения завершилась ошибкой'}
            </span>}
          </div>
        </div> : null}
      </> : null}
    </section>

    </> : <>
    <section className='card transport-result lab-setup-card'>
      <div className='transport-section-head'>
        <div className='transport-section-title'>
          <h3>Настройки Lab</h3>
          <p>Три рабочие зоны: промпты, переклассификация и локальные правила, которые проверяются до отправки в GigaChat.</p>
        </div>
        <button className='transport-collapse-button' onClick={() => setSettingsCollapsed((current) => !current)}>
          {settingsCollapsed ? 'Развернуть' : 'Свернуть'}
        </button>
      </div>

      {!settingsCollapsed ? <>
        {versionsQ.isLoading || selectedVersionQ.isLoading ? <div>Загружаем версии настроек...</div> : null}
        {versionsQ.isError ? <div className='transport-error'>{formatLabError(versionsQ.error as Error, 'версии настроек')}</div> : null}
        {selectedVersionQ.isError ? <div className='transport-error'>{formatLabError(selectedVersionQ.error as Error, 'версию настроек')}</div> : null}
        {selectedVersionQ.data ? <>
          <div className='settings-version-panel'>
            <label className='lab-field'>
              <span>Версия настроек</span>
              <select
                value={selectedSettingsVersionId}
                onChange={(event) => setSelectedSettingsVersionId(event.target.value)}
              >
                {(versionsQ.data?.versions ?? []).map((version) => <option key={version.version_id} value={version.version_id}>
                  {version.title} · {VERSION_STATUS_LABELS[version.status]} · {VERSION_VISIBILITY_LABELS[version.visibility]} · Автор: {version.created_by || '—'}
                </option>)}
              </select>
            </label>
            <div className='settings-version-actions'>
              <button type='button' onClick={() => {
                setVersionDraft((current) => ({ ...current, baseVersionId: selectedSettingsVersionId, createdBy: current.createdBy || currentUserDisplayName }))
                setVersionCreateOpen((current) => !current)
              }}>
                Создать версию
              </button>
              <button type='button' onClick={() => saveSettings.mutate(undefined)} disabled={saveSettings.isPending || !selectedVersionQ.data}>
                {saveSettings.isPending ? 'Сохраняем...' : selectedVersionCanEdit ? 'Сохранить профиль' : 'Сохранить в мой профиль'}
              </button>
              <button
                type='button'
                className='danger'
                onClick={() => {
                  const version = selectedVersionQ.data?.version
                  if (!version) return
                  if (window.confirm(`Удалить версию "${version.title}"?`)) {
                    deleteSettingsVersion.mutate(version.version_id)
                  }
                }}
                disabled={!selectedVersionCanEdit || deleteSettingsVersion.isPending || saveSettings.isPending}
                title={selectedVersionCanEdit ? 'Удалить выбранную версию настроек' : 'Можно удалять только свой приватный профиль'}
              >
                {deleteSettingsVersion.isPending ? 'Удаляем...' : 'Удалить версию'}
              </button>
            </div>
          </div>
          {saveSettings.isError ? <div className='transport-error'>{formatLabError(saveSettings.error as Error, 'сохранение профиля')}</div> : null}
          {deleteSettingsVersion.isError ? <div className='transport-error'>{formatLabError(deleteSettingsVersion.error as Error, 'удаление версии')}</div> : null}
          {!selectedVersionCanEdit ? <div className='lab-muted'>
            {personalProfileForSelectedVersion
              ? `Эта версия доступна для просмотра и использования. При сохранении изменений будет обновлен ваш приватный профиль "${personalProfileForSelectedVersion.title}".`
              : 'Эта версия доступна для просмотра и использования. При первом сохранении изменений будет создан приватный профиль на основе текущей версии.'}
          </div> : null}

          {versionCreateOpen ? <div className='settings-version-create'>
            <label className='lab-field'>
              <span>Название</span>
              <input value={versionDraft.title} onChange={(e) => setVersionDraft((current) => ({ ...current, title: e.target.value }))} />
            </label>
            <label className='lab-field'>
              <span>ID файла</span>
              <input value={versionDraft.versionId} placeholder='автоматически из названия' onChange={(e) => setVersionDraft((current) => ({ ...current, versionId: e.target.value }))} />
            </label>
            <label className='lab-field'>
              <span>Автор</span>
              <input value={versionDraft.createdBy} onChange={(e) => setVersionDraft((current) => ({ ...current, createdBy: e.target.value }))} />
            </label>
            <label className='lab-field'>
              <span>Статус</span>
              <select value={versionDraft.status} onChange={(e) => setVersionDraft((current) => ({ ...current, status: e.target.value as GigaChatSettingsVersionStatus }))}>
                {Object.entries(VERSION_STATUS_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </select>
            </label>
            <label className='lab-field'>
              <span>Видимость</span>
              <select value={versionDraft.visibility} onChange={(e) => setVersionDraft((current) => ({ ...current, visibility: e.target.value as GigaChatSettingsVersionVisibility }))}>
                <option value='private'>Приватная — только я</option>
                <option value='public'>Публичная — видят все пользователи</option>
              </select>
            </label>
            <label className='lab-field'>
              <span>Создать на основе</span>
              <select value={versionDraft.baseVersionId} onChange={(e) => setVersionDraft((current) => ({ ...current, baseVersionId: e.target.value }))}>
                {(versionsQ.data?.versions ?? []).map((version) => <option key={version.version_id} value={version.version_id}>{version.title} · {VERSION_VISIBILITY_LABELS[version.visibility]}</option>)}
              </select>
            </label>
            <label className='lab-field wide'>
              <span>Описание</span>
              <textarea value={versionDraft.description} onChange={(e) => setVersionDraft((current) => ({ ...current, description: e.target.value }))} />
            </label>
            <div className='lab-settings-actions'>
              <button type='button' onClick={() => createSettingsVersion.mutate()} disabled={!versionDraft.title.trim() || createSettingsVersion.isPending}>
                {createSettingsVersion.isPending ? 'Создаём...' : 'Создать'}
              </button>
              <button type='button' onClick={() => setVersionCreateOpen(false)}>Отмена</button>
              {createSettingsVersion.isError ? <span className='transport-error'>{formatLabError(createSettingsVersion.error as Error, 'создание версии')}</span> : null}
            </div>
          </div> : null}

          <div className='settings-info-panel'>
            <div className='settings-info-head'>
              <div>
                <b>Информация</b>
                <span>
                  {selectedVersionQ.data.version.title} · {VERSION_STATUS_LABELS[selectedVersionQ.data.version.status]} · {VERSION_VISIBILITY_LABELS[selectedVersionQ.data.version.visibility]}
                </span>
              </div>
              <button type='button' onClick={() => setVersionInfoOpen((current) => !current)}>
                {versionInfoOpen ? 'Свернуть' : 'Развернуть'}
              </button>
            </div>
            {versionInfoOpen ? <div className='lab-settings-meta'>
              <div><b>Профиль:</b> <code>{selectedVersionQ.data.version.title}</code></div>
              <div><b>Статус:</b> {VERSION_STATUS_LABELS[selectedVersionQ.data.version.status]}</div>
              <div><b>Видимость:</b> {VERSION_VISIBILITY_LABELS[selectedVersionQ.data.version.visibility]}</div>
              <div><b>Автор:</b> {selectedVersionQ.data.version.created_by || '—'}</div>
              <div><b>Владелец:</b> <code>{selectedVersionQ.data.version.owner_user_id || '—'}</code></div>
              <div><b>Основана на:</b> <code>{selectedVersionQ.data.version.base_version_id || '—'}</code></div>
              <div><b>Создана:</b> {selectedVersionQ.data.version.created_at ? new Date(selectedVersionQ.data.version.created_at).toLocaleString() : '—'}</div>
              <div><b>Последнее сохранение:</b> {selectedVersionQ.data.version.updated_at ? new Date(selectedVersionQ.data.version.updated_at).toLocaleString() : 'еще не сохраняли'}</div>
              <div className='lab-field wide'><b>Описание:</b> {selectedVersionQ.data.version.description || '—'}</div>
              <div className='lab-field wide'><b>Файл:</b> <code>{selectedVersionQ.data.version.path || 'виртуальная default-версия'}</code></div>
            </div> : null}
          </div>

          <div className='lab-setup-tabs' role='tablist' aria-label='GigaChat Lab settings tabs'>
            <button
              type='button'
              className={activeSetupTab === 'rules' ? 'active' : ''}
              onClick={() => setActiveSetupTab('rules')}
            >
              Правила
            </button>
            <button
              type='button'
              className={activeSetupTab === 'prompts' ? 'active' : ''}
              onClick={() => setActiveSetupTab('prompts')}
            >
              Промпты
            </button>
            <button
              type='button'
              className={activeSetupTab === 'reclassification' ? 'active' : ''}
              onClick={() => setActiveSetupTab('reclassification')}
            >
              Переклассификация
            </button>
          </div>

          {reclassificationOpen ? <div className='sheet-modal-backdrop' onClick={() => setReclassificationOpen(false)}>
            <div className='card sheet-modal reclassification-settings-modal' onClick={(event) => event.stopPropagation()}>
              <div className='transport-section-head'>
                <div className='transport-section-title'>
                  <h3>{reclassificationDraft.editingIndex === null ? 'Добавить правило переклассификации' : 'Редактировать правило переклассификации'}</h3>
                  <p>Настройте поля и инструкцию, которые попадут в итоговый prompt для проверки темы.</p>
                </div>
                <button className='transport-collapse-button' type='button' onClick={() => setReclassificationOpen(false)}>Закрыть</button>
              </div>

              <div className='labeling-edit-modal-form reclassification-settings-form'>
                <label className='lab-field wide'>
                  <span>Название переклассификации</span>
                  <input
                    type='text'
                    value={reclassificationDraft.name}
                    onChange={(event) => setReclassificationDraft((current) => ({ ...current, name: event.target.value }))}
                    placeholder='Проблема с выдачей очередного транша по образовательному кредиту'
                  />
                  <span className='lab-field-help'>Это точное значение, которое GigaChat сможет вернуть в `reclassified_topic`.</span>
                </label>

                <label className={`lab-field ${reclassificationDraftSourceMissing ? 'missing-field' : ''}`}>
                  <span>Поле исходной темы</span>
                  <select
                    className={reclassificationDraftSourceMissing ? 'missing-field-control' : ''}
                    value={reclassificationDraft.sourceField}
                    onChange={(event) => {
                      setReclassificationDraft((current) => ({ ...current, sourceField: event.target.value }))
                      setReclassificationSaveStatus({ type: 'idle', message: '' })
                    }}
                  >
                    {renderReclassificationColumnOptions('reclassification-source')}
                  </select>
                  <span className={reclassificationDraftSourceMissing ? 'lab-field-help missing-field-help' : 'lab-field-help'}>
                    {reclassificationDraftSourceMissing
                      ? `Колонки "${reclassificationDraft.sourceField}" нет в текущем рабочем листе. Правило будет неактивным.`
                      : 'Колонка, где хранится текущая/исходная тема обращения.'}
                  </span>
                </label>

                <div className={`lab-field ${reclassificationDraftContextMissing ? 'missing-field' : ''}`}>
                  <span>Поле контекста</span>
                  <div className={`rule-source-field-picker ${reclassificationDraftContextMissing ? 'missing-field-control' : ''}`}>
                    <div className='rule-source-selected'>
                      {reclassificationDraftContextFields.length ? reclassificationDraftContextFields.map((field) => {
                        const missing = isReclassificationFieldMissing(field)
                        return <button
                          key={`reclassification-context-selected-${field}`}
                          type='button'
                          className={missing ? 'missing' : ''}
                          title='Убрать поле из контекста'
                          onClick={() => {
                            setReclassificationDraft((current) => ({
                              ...current,
                              contextFields: current.contextFields.filter((item) => item !== field),
                            }))
                            setReclassificationSaveStatus({ type: 'idle', message: '' })
                          }}
                        >
                          {field} ×
                        </button>
                      }) : <span className='lab-muted'>Не выбрано</span>}
                    </div>
                    <div className='rule-source-options'>
                      {reclassificationAvailableColumnOptions.map((column) => {
                        const selected = reclassificationDraftContextFields.includes(column)
                        return <button
                          key={`reclassification-context-option-${column}`}
                          type='button'
                          className={selected ? 'selected' : ''}
                          disabled={selected}
                          onClick={() => {
                            if (selected) return
                            setReclassificationDraft((current) => ({
                              ...current,
                              contextFields: [...current.contextFields, column],
                            }))
                            setReclassificationSaveStatus({ type: 'idle', message: '' })
                          }}
                        >
                          <span className='rule-source-field-option'>
                            <span>{selected ? '✓' : '+'}</span>
                            <span>{column}</span>
                          </span>
                        </button>
                      })}
                    </div>
                  </div>
                  <span className={reclassificationDraftContextMissing ? 'lab-field-help missing-field-help' : 'lab-field-help'}>
                    {reclassificationDraftContextMissing
                      ? `Колонок ${reclassificationDraftMissingContextFields.join(', ')} нет в текущем рабочем листе. Правило будет неактивным.`
                      : 'Колонки с текстом, суммаризацией или другим контекстом для проверки темы.'}
                  </span>
                </div>

                <label className='lab-field wide'>
                  <span>Описание переклассификации</span>
                  <textarea
                    value={reclassificationDraft.prompt}
                    onChange={(event) => setReclassificationDraft((current) => ({ ...current, prompt: event.target.value }))}
                    placeholder={RECLASSIFICATION_DESCRIPTION_PLACEHOLDER}
                  />
                  <span className='lab-field-help'>Кратко опишите, по каким признакам выбирать эту тему. В prompt уйдут только название и это описание.</span>
                </label>
              </div>

              {!sheetData ? <div className='lab-muted'>После загрузки рабочей тетради здесь появятся колонки текущего листа.</div> : null}
              {reclassificationSaveStatus.message ? <div className={reclassificationSaveStatus.type === 'error' ? 'transport-error' : 'lab-muted'}>
                {reclassificationSaveStatus.message}
              </div> : null}

              <div className='lab-settings-actions'>
                <button
                  className='primary'
                  type='button'
                  onClick={saveReclassificationRule}
                  disabled={saveSettings.isPending || reclassificationImporting || !canSaveReclassificationDraft}
                >
                  {saveSettings.isPending || reclassificationImporting ? 'Записываем...' : 'Записать в профиль'}
                </button>
                <button type='button' onClick={() => setReclassificationOpen(false)} disabled={saveSettings.isPending || reclassificationImporting}>Отмена</button>
              </div>
            </div>
          </div> : null}

          {activeSetupTab === 'reclassification' ? <div className='lab-tab-panel'>
            <section className='labeling-panel reclassification-rules-panel'>
              <div className='labeling-panel-head'>
                <h4>Переклассификация</h4>
                <p>Правила, которые отправляют в GigaChat исходную тему, контекст и отдельную инструкцию для проверки, нужно ли заменить тему строки.</p>
              </div>

              <div className='transport-actions'>
                <button type='button' onClick={() => openReclassificationRule(null)} disabled={saveSettings.isPending || reclassificationImporting}>+ Добавить правило</button>
                <label className={`rule-pack-file-button ${saveSettings.isPending || reclassificationImporting ? 'disabled' : ''}`}>
                  <span>{reclassificationImporting ? 'Читаем Excel...' : 'Загрузить из Excel'}</span>
                  <input
                    key={reclassificationImportInputVersion}
                    type='file'
                    accept='.xlsx,.xls,.xlsm,.csv'
                    disabled={saveSettings.isPending || reclassificationImporting}
                    onChange={(event) => importReclassificationRulesFromFile(event.target.files?.[0])}
                  />
                </label>
              </div>
              <div className='lab-muted'>Ожидаемые колонки: name, src_field, context_field, prompt_field. В context_field можно указать несколько колонок через запятую или перенос строки.</div>
              {reclassificationSaveStatus.message ? <div className={reclassificationSaveStatus.type === 'error' ? 'transport-error' : 'lab-muted'}>
                {reclassificationSaveStatus.message}
              </div> : null}

              {reclassificationRules.length ? <div className='labeling-rules-list'>
                {reclassificationRules.map((rule, index) => {
                  const status = reclassificationRuleStatuses[index] ?? { valid: true, missing: [] }
                  const active = canRunReclassificationRule(rule) && status.valid
                  const contextFields = getReclassificationRuleContextFields(rule)
                  return <div
                    key={`${rule.name}-${rule.source_field}-${contextFields.join('|')}-${index}`}
                    className={`labeling-rule-card reclassification-rule-card ${active ? 'active' : 'inactive'}${status.valid ? '' : ' missing-fields'}`}
                  >
                    <div className='labeling-rule-copy'>
                      <div className='labeling-rule-name'>{formatReclassificationRuleTitle(rule)}</div>
                      {!status.valid ? <div className='rule-pack-missing-fields'>Нет колонок: {status.missing.join(', ')}</div> : null}
                      <div className='labeling-rule-description'>{rule.prompt}</div>
                      <div className='labeling-rule-meta'>
                        <span className={rule.source_field && status.missing.includes(rule.source_field) ? 'reclassification-field-missing' : ''}>Поле темы: {rule.source_field || '—'}</span>
                        <span className={contextFields.some((field) => status.missing.includes(field)) ? 'reclassification-field-missing' : ''}>Контекст: {contextFields.join(', ') || '—'}</span>
                        <span className={`reclassification-status-badge ${active ? 'active' : 'inactive'}`}>{active ? 'Активно' : 'Неактивно'}</span>
                      </div>
                    </div>
                    <div className='labeling-rule-actions'>
                      <button className='labeling-remove-button' type='button' onClick={() => openReclassificationRule(index)} disabled={saveSettings.isPending || reclassificationImporting}>Редактировать</button>
                      <button className='labeling-remove-button' type='button' onClick={() => removeReclassificationRule(index)} disabled={saveSettings.isPending || reclassificationImporting}>Удалить</button>
                    </div>
                  </div>
                })}
              </div> : <div className='lab-muted'>Правила переклассификации пока не добавлены.</div>}
            </section>
          </div> : null}

          {activeSetupTab === 'prompts' ? <div className='lab-tab-panel'>
            <GigaChatSettingsForm
              fields={requestSettingsFields}
              values={settingValues}
              busy={saveSettings.isPending}
              saveError={saveSettings.isError ? formatLabError(saveSettings.error as Error, 'настройки') : null}
              onChange={(key, value) => setSettingValues((current) => ({ ...current, [key]: value }))}
              onPersist={persistSettingValue}
              onSave={() => saveSettings.mutate(undefined)}
            />
          </div> : null}

          {activeSetupTab === 'rules' ? <div className='lab-tab-panel'>
            <RulePackEditor
              value={settingValues.rule_pack_prompt_notes}
              onChange={(value) => setSettingValues((current) => ({ ...current, rule_pack_prompt_notes: value }))}
              onPersist={persistRulePacks}
              persistBusy={saveSettings.isPending}
              persistError={saveSettings.isError ? formatLabError(saveSettings.error as Error, 'правила') : null}
              availableFields={sheetData?.columns ?? []}
              defaultValue={DEFAULT_RULE_PACK_PROMPT_NOTES}
              versionId={selectedSettingsVersionId}
              canEdit={selectedVersionCanEdit}
              currentVersionTitle={selectedVersionQ.data?.version.title}
              currentUserDisplayName={currentUserDisplayName}
              onImportComplete={async (data) => {
                setSelectedSettingsVersionId(data.version.version_id)
                window.localStorage.setItem(SETTINGS_VERSION_STORAGE_KEY, data.version.version_id)
                settingValuesVersionIdRef.current = data.version.version_id
                setSettingValues(data.values)
                setRulePackExclusionCodesDraft(parseRulePackExclusions(data.values[RULE_PACK_EXCLUSION_SETTING_KEY]))
                await qc.invalidateQueries({ queryKey: ['gigachat-lab-settings-versions'] })
                await qc.invalidateQueries({ queryKey: ['gigachat-lab-settings-version', data.version.version_id] })
              }}
            />
          </div> : null}
        </> : null}
      </> : null}

      <div className='lab-settings-actions'>
        <button
          onClick={() => {
            setFinalPromptModalOpen(true)
            if (!finalPromptPreview || lastResolvedPreviewKeyRef.current !== finalPromptRequestKey) {
              lastAutoPreviewKeyRef.current = finalPromptRequestKey
              previewFinalPrompt.mutate({ requestKey: finalPromptRequestKey })
            }
          }}
          disabled={previewFinalPrompt.isPending && !finalPromptPreview}
        >
          {previewFinalPrompt.isPending && !finalPromptPreview ? 'Собираем итоговый...' : 'Показать итоговый промпт'}
        </button>
        <span className='lab-settings-status'>
          {previewFinalPrompt.isPending && !finalPromptSynchronized
            ? 'Итоговый промпт обновляется автоматически...'
            : finalPromptPreview && finalPromptSynchronized
              ? 'Итоговый промпт синхронизирован с текущими полями формы.'
              : 'Итоговый промпт будет собран автоматически после первого изменения формы.'}
        </span>
        {previewFinalPrompt.isError ? <span className='transport-error'>{formatLabError(previewFinalPrompt.error as Error, 'итоговый промпт')}</span> : null}
      </div>
    </section>

    {backgroundWorkbookUploads.length ? <section className='card transport-result background-upload-panel'>
      <div className='transport-section-head'>
        <div className='transport-section-title'>
          <h3>Фоновые загрузки</h3>
          <p>Файлы, которые уже переданы backend и сейчас собираются, читаются или пишутся в parquet. Готовый файл можно открыть в Lab отсюда.</p>
        </div>
        <button type='button' className='transport-collapse-button' onClick={() => backgroundUploadTasksQ.refetch()} disabled={backgroundUploadTasksQ.isFetching}>
          {backgroundUploadTasksQ.isFetching ? 'Обновляем...' : 'Обновить'}
        </button>
      </div>
      <div className='background-upload-list'>
        {backgroundWorkbookUploads.map((upload) => {
          const entry = backgroundUploadTaskMap.get(upload.taskId)
          const task = entry?.task
          const percent = Math.min(100, Math.max(0, Math.round((task?.progress ?? 0) * 100)))
          const status = task?.status ?? 'queued'
          const canOpen = task?.status === 'completed' && Boolean(task.workbook)
          return <article key={upload.taskId} className='background-upload-card'>
            <div className='background-upload-head'>
              <div>
                <strong>{upload.filename}</strong>
                <span>Создана: {new Date(upload.createdAt).toLocaleString()} · ID: <code>{upload.taskId}</code></span>
              </div>
              <span className={`background-task-status ${status}`}>{WORKBOOK_UPLOAD_TASK_STATUS_LABELS[status]}</span>
            </div>
            <div className='giga-processing-progressbar' aria-label='workbook background upload progress'>
              <div className='giga-processing-progressbar-fill' style={{ width: `${percent || 3}%` }} />
            </div>
            <div className='lab-muted'>
              {entry?.error
                ? entry.error
                : task
                  ? `${percent}% · ${task.message || 'Backend обрабатывает файл'}`
                  : 'Ждем статус backend-задачи...'}
            </div>
            <div className='transport-actions'>
              <button
                type='button'
                className='primary'
                disabled={!canOpen}
                onClick={() => {
                  if (!task?.workbook) return
                  activateUploadedWorkbook(task.workbook)
                  setBackgroundWorkbookUploads((current) => current.filter((item) => item.taskId !== upload.taskId))
                }}
              >
                Открыть в Lab
              </button>
              {status === 'queued' || status === 'running' ? <button
                type='button'
                disabled={cancelWorkbookUploadTaskMutation.isPending}
                onClick={() => cancelWorkbookUploadTaskMutation.mutate(upload.taskId)}
              >
                {cancelWorkbookUploadTaskMutation.isPending ? 'Отменяем...' : 'Отменить'}
              </button> : null}
              <button
                type='button'
                onClick={() => setBackgroundWorkbookUploads((current) => current.filter((item) => item.taskId !== upload.taskId))}
              >
                Убрать из списка
              </button>
            </div>
          </article>
        })}
      </div>
    </section> : null}

    <section className='card transport-result workbook-upload-card'>
      <div className='transport-section-head'>
        <div className='transport-section-title'>
          <h3>Excel для разметки</h3>
          <p>Загрузите локальный Excel, CSV или ZIP-архив с ними, затем выберите рабочий лист для правил и GigaChat.</p>
        </div>
        <button className='transport-collapse-button' onClick={() => setUploadCollapsed((current) => !current)}>
          {uploadCollapsed ? 'Развернуть' : 'Свернуть'}
        </button>
      </div>

      {uploadWorkbook.isPending && !workbookUploadOverlayHidden ? <div className='workbook-upload-loader'>
        <div className='card workbook-upload-loader-card'>
          <div className='spinner workbook-upload-spinner' aria-label='uploading workbook' />
          <div className='workbook-upload-loader-title'>{workbookUploadProgress?.message || 'Загружаем файл...'}</div>
          <div className='giga-processing-progressbar' aria-label='workbook upload progress'>
            <div className='giga-processing-progressbar-fill' style={{ width: `${uploadPercent ?? 8}%` }} />
          </div>
          <div className='lab-muted'>
            {workbookUploadProgress
              ? `${uploadPercent ?? 0}% · ${formatBytes(workbookUploadProgress.uploadedBytes)} из ${formatBytes(workbookUploadProgress.totalBytes)} · ${workbookUploadProgress.receivedChunks} из ${workbookUploadProgress.totalChunks} частей`
              : 'Во время загрузки выбор файла и кнопка заблокированы.'}
          </div>
          <div className='lab-muted'>
            Большие файлы отправляются частями, затем backend собирает их и пишет parquet в фоне.
          </div>
          <div className='transport-actions workbook-upload-loader-actions'>
            <button
              type='button'
              className='primary'
              onClick={sendCurrentUploadToBackground}
              disabled={!workbookUploadProgress?.taskId}
              title={workbookUploadProgress?.taskId ? 'Скрыть загрузку и продолжить работу. Задача останется в фоновых загрузках.' : 'Будет доступно после передачи всех частей файла на backend.'}
            >
              В фон
            </button>
            <button
              type='button'
              onClick={() => void cancelCurrentWorkbookUpload()}
            >
              Отменить загрузку
            </button>
          </div>
          {!workbookUploadProgress?.taskId ? <div className='lab-muted'>
            Кнопка “В фон” включится после того, как файл полностью передан backend.
          </div> : null}
        </div>
      </div> : null}

      {!uploadCollapsed ? <>
        <div className='workbook-upload-row'>
          <label className={`workbook-file-button ${uploadWorkbook.isPending ? 'disabled' : ''}`}>
            <input
              key={fileInputVersion}
              type='file'
              accept='.xlsx,.xls,.xlsm,.csv,.zip'
              disabled={uploadWorkbook.isPending}
              onChange={(e) => {
                const file = e.target.files?.[0] ?? null
                setSelectedFile(file)
                if (file) uploadWorkbook.mutate(file)
              }}
            />
            <span>{uploadWorkbook.isPending ? 'Загружаем файл...' : 'Выбрать Excel / CSV / ZIP'}</span>
          </label>
          {selectedFile ? <span className='lab-muted'>Выбран файл: <code>{selectedFile.name}</code></span> : <span className='lab-muted'>Файл пока не выбран.</span>}
        </div>

        {uploadWorkbook.isError ? <div className='transport-error'>{formatLabError(uploadWorkbook.error as Error, 'файл')}</div> : null}
        {selectSheet.isError ? <div className='transport-error'>{formatLabError(selectSheet.error as Error, 'лист')}</div> : null}
        {viewSheet.isError ? <div className='transport-error'>{formatLabError(viewSheet.error as Error, 'лист для просмотра')}</div> : null}
        {workbookMeta ? <div className='lab-settings-meta'>
          <div><b>Файл:</b> <code>{workbookMeta.filename}</code></div>
          <div><b>Формат:</b> <code>{workbookMeta.file_format}</code></div>
          <div><b>Листов:</b> {workbookMeta.sheet_count}</div>
          <div><b>Рабочий лист:</b> <code>{selectedSheetName ?? 'еще не выбран'}</code></div>
        </div> : null}
        {workbookMeta && !selectedSheetName && hasMultipleSheets ? <div className='transport-actions'>
          <button type='button' onClick={() => setSheetPickerOpen(true)}>
            Выбрать рабочий лист
          </button>
        </div> : null}
      </> : null}
    </section>

    {selectedVersionQ.data ? <section className='card transport-result rule-pack-exclusions-card'>
      <div className='transport-section-head'>
        <div className='transport-section-title'>
          <h3>Исключения GigaChat</h3>
          <p>Эти rule packs остаются в локальной проверке и рабочей таблице, но не попадают в prompt и API GigaChat.</p>
        </div>
      </div>

      <div className='rule-pack-exclusion-controls'>
        <label className='lab-field rule-pack-exclusion-field'>
          <span>Добавить правило в исключения</span>
          <select
            ref={rulePackExclusionSelectRef}
            value={activeRulePackExclusionDraft}
            onChange={(event) => setRulePackExclusionDraft(event.target.value)}
            disabled={!availableRulePackExclusionOptions.length || saveSettings.isPending}
          >
            {availableRulePackExclusionOptions.length
              ? availableRulePackExclusionOptions.map((rule) => <option key={rule.code} value={rule.code}>{rule.code}</option>)
              : <option value=''>Нет доступных rule packs</option>}
          </select>
        </label>
        <button
          className='rule-pack-exclusion-add-button'
          type='button'
          onClick={addRulePackExclusion}
          disabled={!availableRulePackExclusionOptions.length || saveSettings.isPending}
        >
          Добавить
        </button>
      </div>

      {saveSettings.isError ? <div className='transport-error'>{formatLabError(saveSettings.error as Error, 'исключения')}</div> : null}

      <div className='rule-pack-exclusions-summary'>
        <div className='rule-pack-exclusions-summary-head'>
          <h4>Правила без отправки в GigaChat</h4>
          <p>Записи, где сработают эти rule packs, не будут отправляться в GigaChat; в локальной проверке и рабочей таблице они останутся видимыми.</p>
        </div>
        {excludedRulePackEntries.length ? <div className='labeling-rules-list rule-pack-exclusions-list'>
          {excludedRulePackEntries.map(({ code, rule }) => {
            const actionText = rule
              ? rule.type === 'assign_tag'
                ? rule.target_tag
                : rule.target_topic
              : ''
            return <div key={code} className={`labeling-rule-card rule-pack-exclusion-card ${rule ? 'active' : 'missing-fields'}`}>
              <div className='labeling-rule-copy'>
                <div className='labeling-rule-name'>{code}</div>
                {!rule ? <div className='rule-pack-missing-fields'>Rule pack не найден в текущем списке правил.</div> : null}
                <div className='labeling-rule-description'>
                  {rule?.description || 'Исключение будет записано в профиль при сохранении. Локальная проверка продолжит показывать срабатывания этого rule pack.'}
                </div>
                <div className='labeling-rule-meta'>
                  <span>В API GigaChat: не отправляется</span>
                  <span>В таблице: показывается</span>
                  {rule ? <span>{rule.type}{actionText ? ` -> ${actionText}` : ''}</span> : null}
                </div>
              </div>
              <div className='labeling-rule-actions'>
                <button
                  className='labeling-remove-button'
                  type='button'
                  onClick={() => removeRulePackExclusion(code)}
                  disabled={saveSettings.isPending}
                >
                  Удалить
                </button>
              </div>
            </div>
          })}
        </div> : <div className='rule-pack-exclusions-empty'>Исключения пока не добавлены. Все активные rule packs попадают в prompt и API GigaChat.</div>}
      </div>
    </section> : null}

    <section className='card transport-result rule-validation-card'>
      <div className='transport-section-head'>
        <div className='transport-section-title'>
          <h3>Проверка правил</h3>
          <p>Локальный прогон rule packs по рабочему листу до GigaChat: здесь видно, какие правила сработали, по каким полям и сколько строк они нашли.</p>
        </div>
        <div className='transport-actions'>
          <button type='button' onClick={handleEvaluateRules} disabled={!sheetData || ruleEvaluationBusy}>
            {ruleEvaluationBusy ? 'Прогоняем правила...' : 'Прогнать правила на рабочем листе'}
          </button>
        </div>
      </div>

      {!sheetData ? <p>Загрузите Excel и выберите лист, чтобы проверить rule packs.</p> : <>
        {ruleEvaluationBusy ? <div className='rule-progress-panel'>
          <div className='rule-progress-copy'>
            <strong>Проверяем правила</strong>
            <span>{ruleEvaluationProcessed} из {ruleEvaluationTotal} строк обработано, осталось {ruleEvaluationRemaining}</span>
          </div>
          <div className='giga-processing-progressbar' aria-label='rule evaluation progress'>
            <div className='giga-processing-progressbar-fill' style={{ width: `${ruleEvaluationPercent}%` }} />
          </div>
        </div> : null}

        <div className='rule-validation-summary'>
          <div className='rule-validation-kpi'>
            <span>Строк с rule hits</span>
            <strong>{ruleHitRowCount}</strong>
            <small>из {sheetData.rows.length} показанных строк</small>
          </div>
          <div className='rule-validation-kpi'>
            <span>Активных срабатываний</span>
            <strong>{ruleEvaluationSummary.reduce((sum, item) => sum + item.count, 0)}</strong>
            <small>по всем rule packs</small>
          </div>
          <div className='rule-validation-kpi'>
            <span>Rule packs с матчами</span>
            <strong>{ruleEvaluationSummary.length}</strong>
            <small>в текущем листе</small>
          </div>
        </div>

        {ruleEvaluationSummary.length ? <div className='rule-validation-list'>
          {ruleEvaluationSummary.map((item) => <div key={item.code} className='rule-validation-row'>
            <div>
              <strong>{item.code}</strong>
              <div className='lab-muted'>{item.count} совпадений</div>
            </div>
            <div>
              <span className='lab-muted'>Поля</span>
              <div>{Array.from(item.fields).join(', ') || '—'}</div>
            </div>
            <div>
              <span className='lab-muted'>Keywords</span>
              <div>{Array.from(item.keywords).slice(0, 8).join(', ') || '—'}</div>
            </div>
            <div>
              <span className='lab-muted'>Suggested action</span>
              <div>{Array.from(item.suggested).join(', ') || '—'}</div>
            </div>
          </div>)}
        </div> : <div className='lab-muted'>Пока совпадений нет. Если лист уже выбран, можно нажать “Прогнать правила” или проверить keywords/фильтры во вкладке “Правила”.</div>}

        {ruleEvaluationError ? <div className='transport-error'>{ruleEvaluationError}</div> : null}
      </>}
    </section>

    <section className='card transport-result rule-verification-card'>
      <div className='transport-section-head'>
        <div className='transport-section-title'>
          <h3>Валидация и верификация</h3>
          <p>Проверка качества rule packs на тестовом наборе с эталонными колонками. Это отдельный слой от GigaChat: он сверяет локальные rule hits с ожидаемым результатом.</p>
        </div>
      </div>

      {!sheetData ? <p>Загрузите тестовый лист, чтобы увидеть статус валидации.</p> : <>
        <div className='rule-validation-summary'>
          <div className='rule-validation-kpi'>
            <span>Статус</span>
            <strong>{ruleValidationSummary?.available ? 'Активна' : 'Нет эталона'}</strong>
            <small>{ruleValidationSummary?.available ? 'есть expected-колонки' : 'нужен validation-набор'}</small>
          </div>
          <div className='rule-validation-kpi'>
            <span>PASS</span>
            <strong>{ruleValidationSummary?.passed ?? 0}</strong>
            <small>строк совпало с эталоном</small>
          </div>
          <div className='rule-validation-kpi'>
            <span>FAIL</span>
            <strong>{ruleValidationSummary?.failed ?? 0}</strong>
            <small>строк требует разбора</small>
          </div>
        </div>
        {!ruleValidationSummary?.available ? <div className='lab-muted'>
          {ruleValidationSummary?.reason} Для обычного январского файла это нормально: он нужен для разметки. Для валидации нужен отдельный golden-test файл или audit export с expected-колонками.
        </div> : null}
      </>}
    </section>

    <section className='card transport-result workbook-table-card'>
      <div className='transport-section-head'>
        <div className='transport-section-title'>
          <h3>Рабочая тетрадь</h3>
          <p>Просмотр листов файла как таблиц. Правила и GigaChat работают по рабочему листу.</p>
        </div>
        <button className='transport-collapse-button' onClick={() => setWorkbookCollapsed((current) => !current)}>
          {workbookCollapsed ? 'Развернуть' : 'Свернуть'}
        </button>
      </div>

      {ruleEvaluationBusy && displayedSheetIsWorking ? <div className='workbook-rule-lock'>
        <div className='card workbook-rule-lock-card'>
          <div className='spinner workbook-upload-spinner' aria-hidden='true' />
          <div className='giga-processing-progress-copy'>
            <div className='giga-processing-progress-title'>Проверяем правила</div>
            <div className='lab-muted'>
              Обработано {ruleEvaluationProcessed} из {ruleEvaluationTotal} строк. Осталось {ruleEvaluationRemaining}.
            </div>
          </div>
          <div className='giga-processing-progressbar' aria-label='workbook lock rule progress'>
            <div className='giga-processing-progressbar-fill' style={{ width: `${ruleEvaluationPercent}%` }} />
          </div>
        </div>
      </div> : null}

      {!workbookCollapsed ? <>
        {exportWorkbookRows.isPending ? <div className='export-progress-panel'>
          <div className='rule-progress-copy'>
            <strong>{exportProgress?.label ?? 'Выгружаем рабочую таблицу'}</strong>
            <span>
              {exportPercent !== null
                ? `${formatBytes(exportLoadedBytes)} из ${formatBytes(exportTotalBytes ?? 0)}`
                : exportLoadedBytes
                  ? `${formatBytes(exportLoadedBytes)} получено`
                  : 'Собираем файл на backend...'}
            </span>
          </div>
          <div className='giga-processing-progressbar' aria-label='workbook export progress'>
            <div
              className='giga-processing-progressbar-fill'
              style={{ width: `${exportPercent ?? (exportLoadedBytes ? Math.min(95, Math.max(8, Math.round(exportLoadedBytes / 40_000))) : 8)}%` }}
            />
          </div>
          <div className='lab-muted'>Выгрузка полного выбранного листа идёт в фоне, кнопки экспорта временно заблокированы.</div>
        </div> : null}
        {!workbookMeta ? <p>Сначала загрузите Excel или CSV файл.</p> : null}
        {workbookMeta && hasMultipleSheets ? <WorkbookSheetsPanel
          workbook={workbookMeta}
          selectedSheetName={selectedSheetName}
          viewSheetName={viewSheetName}
          busySheetName={selectSheet.isPending ? selectSheet.variables?.sheetName : null}
          viewBusySheetName={viewSheet.isPending ? viewSheet.variables?.sheetName : null}
          onSelect={selectWorkbookSheet}
          onView={viewWorkbookSheet}
        /> : null}
        {workbookMeta && !displayedSheetData && !selectSheet.isPending && !viewSheet.isPending ? <div className='transport-actions'>
          <button
            onClick={() => {
              if (!workbookMeta) return
              if (hasMultipleSheets) {
                setSheetPickerOpen(true)
                return
              }
              const onlySheet = workbookMeta.sheets[0]
              if (onlySheet) {
                selectWorkbookSheet(onlySheet.name)
              }
            }}
            disabled={!hasMultipleSheets && !workbookMeta.sheets[0]}
          >
            {hasMultipleSheets ? 'Выбрать лист' : 'Загрузить лист'}
          </button>
        </div> : null}
        {workbookMeta && !displayedSheetData && !hasMultipleSheets && workbookMeta.sheets[0] ? <p className='lab-muted'>В файле найден один лист. Нажмите `Загрузить лист`, если автозагрузка не успела завершиться.</p> : null}
        {selectSheet.isPending ? <div>Загружаем рабочий лист <code>{selectSheet.variables?.sheetName ?? ''}</code>...</div> : null}
        {viewSheet.isPending ? <div>Открываем лист <code>{viewSheet.variables?.sheetName ?? ''}</code> в тетради...</div> : null}
        {displayedSheetData ? <WorkbookSheetTable
          data={displayedSheetData}
          canChooseAnotherSheet={false}
          rowLimit={workbookRowLimit}
          onRowLimitChange={(value) => {
            setWorkbookRowLimit(value)
            if (!displayedSheetData) return
            const rowLimit = value === 'all' ? displayedSheetData.total_rows : value
            if (displayedSheetIsWorking) {
              selectSheet.mutate({ uploadId: displayedSheetData.upload_id, sheetName: displayedSheetData.sheet_name, rowLimit })
              return
            }
            viewSheet.mutate({ uploadId: displayedSheetData.upload_id, sheetName: displayedSheetData.sheet_name, rowLimit })
          }}
          includedPromptColumns={displayedSheetIsWorking ? includedPromptColumns : displayedSheetData.columns}
          onTogglePromptColumn={(column) => {
            if (!displayedSheetIsWorking) return
            setIncludedPromptColumns((current) => {
              if (current.includes(column)) return current.filter((item) => item !== column)
              return [...current, column]
            })
          }}
          onRunRow={(row, rowIndex) => runWithRuleGuard((valuesForRun) => handleRunRow(row, rowIndex, valuesForRun))}
          runRowBusyIndex={runRowBusyIndex}
          selectedRowKeys={selectedSheetRowKeys}
          selectedRowKeySet={selectedSheetRowKeySet}
          allVisibleRowsSelected={allVisibleRowsSelected}
          onToggleRowSelection={(rowIndex) => {
            if (!sheetData) return
            const key = buildSheetRowKey(sheetData.upload_id, sheetData.sheet_name, rowIndex)
            setSelectedSheetRowKeys((current) => {
              const next = new Set(current)
              if (next.has(key)) next.delete(key)
              else next.add(key)
              return Array.from(next)
            })
          }}
          onToggleAllRows={(checked) => {
            if (!sheetData) return
            const keys = sheetData.rows.map((_, index) => buildSheetRowKey(sheetData.upload_id, sheetData.sheet_name, index))
            const keySet = new Set(keys)
            setSelectedSheetRowKeys((current) => {
              if (!checked) return current.filter((key) => !keySet.has(key))
              const next = new Set(current)
              keys.forEach((key) => next.add(key))
              return Array.from(next)
            })
          }}
          onPickRandomRows={() => {
            if (!sheetData) return
            const keys = sheetData.rows.map((_, index) => buildSheetRowKey(sheetData.upload_id, sheetData.sheet_name, index))
            const keySet = new Set(keys)
            const shuffled = [...keys].sort(() => Math.random() - 0.5)
            setSelectedSheetRowKeys((current) => {
              const rest = current.filter((key) => !keySet.has(key))
              return [...rest, ...shuffled.slice(0, Math.min(5, shuffled.length))]
            })
          }}
          onSelectRuleHitRows={() => {
            if (!sheetData) return
            const keys = sheetData.rows
              .map((_, index) => ({ index, evaluation: ruleEvaluationMap[index] }))
              .filter(({ evaluation }) => evaluation?.hits?.length)
              .map(({ index }) => buildSheetRowKey(sheetData.upload_id, sheetData.sheet_name, index))
            setSelectedSheetRowKeys((current) => {
              const visible = new Set(sheetData.rows.map((_, index) => buildSheetRowKey(sheetData.upload_id, sheetData.sheet_name, index)))
              const rest = current.filter((key) => !visible.has(key))
              return [...rest, ...keys]
            })
          }}
          onSelectNoRuleHitRows={() => {
            if (!sheetData) return
            const keys = sheetData.rows
              .map((_, index) => ({ index, evaluation: ruleEvaluationMap[index] }))
              .filter(({ evaluation }) => !(evaluation?.hits?.length))
              .map(({ index }) => buildSheetRowKey(sheetData.upload_id, sheetData.sheet_name, index))
            setSelectedSheetRowKeys((current) => {
              const visible = new Set(sheetData.rows.map((_, index) => buildSheetRowKey(sheetData.upload_id, sheetData.sheet_name, index)))
              const rest = current.filter((key) => !visible.has(key))
              return [...rest, ...keys]
            })
          }}
          reclassificationRequestsEnabled={sendReclassificationRequests}
          reclassificationRequestsAvailable={reclassificationRequestsAvailable}
          onToggleReclassificationRequests={setSendReclassificationRequests}
          onRunSelectedRows={() => runWithRuleGuard((valuesForRun) => handleRunSelectedRows(valuesForRun))}
          onRunAllRows={() => runWithRuleGuard((valuesForRun) => handleRunAllRows(valuesForRun))}
          onRunSelectedRowsInBackground={() => runWithRuleGuard((valuesForRun) => startBackgroundTask.mutate(valuesForRun))}
          onExportRows={(rows) => exportWorkbookRows.mutate(rows)}
          batchBusy={batchBusy}
          exportBusy={exportBusy}
          exportPending={exportWorkbookRows.isPending}
          busy={displayedSheetIsWorking ? ruleEvaluationBusy || batchBusy || runRowBusyIndex !== null || startBackgroundTask.isPending : viewSheet.isPending}
          ruleEvaluations={displayedSheetIsWorking ? ruleEvaluationMap : {}}
          rulePackOptions={rulePackOptions}
          readOnly={!displayedSheetIsWorking}
          workingSheetName={selectedSheetName}
          onMakeCurrentSheetWorking={displayedSheetIsWorking ? undefined : () => selectWorkbookSheet(displayedSheetData.sheet_name)}
          makeCurrentSheetWorkingBusy={selectSheet.isPending && selectSheet.variables?.sheetName === displayedSheetData.sheet_name}
        /> : null}
        {rowRunError ? <div className='transport-error'>{rowRunError}</div> : null}
        {batchRunError ? <div className='transport-error'>{batchRunError}</div> : null}
        {backgroundTaskError ? <div className='transport-error'>{backgroundTaskError}</div> : null}
        {loadBackgroundTaskResult.isPending ? <div className='lab-muted'>Подгружаем рабочую тетрадь из фоновой задачи...</div> : null}
      </> : null}
    </section>

    <WorkbookStatsPanel
      data={displayedSheetData}
      ruleEvaluations={displayedSheetIsWorking ? ruleEvaluationMap : {}}
    />

    <section className='card transport-result annotated-table-card'>
      <div className='transport-section-head'>
        <div className='transport-section-title'>
          <h3>Размеченная таблица</h3>
          <p>Сюда сразу попадают ответы GigaChat. Слева добавлены колонки класса и тегов, дальше идут исходные колонки жалобы.</p>
        </div>
        {annotatedRows.length ? <div className='transport-actions'>
          <button
            type='button'
            onClick={() => exportAnnotatedRows.mutate('filtered')}
            disabled={exportBusy}
          >
            {exportAnnotatedRows.isPending ? 'Выгружаем Excel...' : 'Выгрузить в Excel'}
          </button>
          <button
            type='button'
            onClick={() => exportAnnotatedRows.mutate('reclassified')}
            disabled={exportBusy || !reclassifiedAnnotatedRows.length}
          >
            Выгрузить с новой подтематикой ({reclassifiedAnnotatedRows.length})
          </button>
          <button
            type='button'
            onClick={() => exportValidationRows.mutate()}
            disabled={exportBusy}
          >
            {exportValidationRows.isPending ? 'Готовим validation...' : 'Сделать validation-файл'}
          </button>
        </div> : null}
      </div>

      {!annotatedRows.length ? <p>Пока здесь пусто. Отправьте одну строку или выбранные строки в GigaChat, и результаты появятся в этой таблице.</p> : <>
        {exportBusy ? <div className='export-progress-panel'>
          <div className='rule-progress-copy'>
            <strong>{exportProgress?.label ?? 'Фоновая выгрузка'}</strong>
            <span>
              {exportPercent !== null
                ? `${formatBytes(exportLoadedBytes)} из ${formatBytes(exportTotalBytes ?? 0)}`
                : exportLoadedBytes
                  ? `${formatBytes(exportLoadedBytes)} получено`
                  : 'Собираем файл на backend...'}
            </span>
          </div>
          <div className='giga-processing-progressbar' aria-label='export progress'>
            <div
              className='giga-processing-progressbar-fill'
              style={{ width: `${exportPercent ?? (exportLoadedBytes ? Math.min(95, Math.max(8, Math.round(exportLoadedBytes / 40_000))) : 8)}%` }}
            />
          </div>
          <div className='lab-muted'>Можно продолжать смотреть страницу, выгрузка идёт в фоне. Кнопки экспорта временно заблокированы.</div>
        </div> : null}

        <div className='annotated-filters'>
          <label className='annotated-filter'>
            <span>Фильтр по классу</span>
            <select
              multiple
              value={annotatedClassFilter}
              onChange={(e) => setAnnotatedClassFilter(Array.from(e.target.selectedOptions).map((option) => option.value))}
            >
              {annotatedClassOptions.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </label>
          <label className='annotated-filter'>
            <span>Фильтр по тегам</span>
            <select
              multiple
              value={annotatedTagFilter}
              onChange={(e) => setAnnotatedTagFilter(Array.from(e.target.selectedOptions).map((option) => option.value))}
            >
              {annotatedTagOptions.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </label>
          <label className='annotated-filter'>
            <span>Фильтр по rule hits</span>
            <select
              multiple
              value={annotatedRuleFilter}
              onChange={(e) => setAnnotatedRuleFilter(Array.from(e.target.selectedOptions).map((option) => option.value))}
            >
              {annotatedRuleOptions.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </label>
          <label className='annotated-filter'>
            <span>Фильтр по source</span>
            <select
              multiple
              value={annotatedDecisionSourceFilter}
              onChange={(e) => setAnnotatedDecisionSourceFilter(Array.from(e.target.selectedOptions).map((option) => option.value))}
            >
              {annotatedDecisionSourceOptions.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </label>
          <div className='transport-actions'>
            <button type='button' onClick={() => {
              setAnnotatedClassFilter([])
              setAnnotatedTagFilter([])
              setAnnotatedRuleFilter([])
              setAnnotatedDecisionSourceFilter([])
            }}>Сбросить фильтры</button>
          </div>
        </div>

        <div className='workbook-table-toolbar'>
          <label className='workbook-row-limit-control'>
            <span>Высота строк:</span>
            <select
              value={String(annotatedTable.rowClamp)}
              onChange={(e) => {
                const value = e.target.value
                if (value === '2' || value === '4' || value === '8') {
                  annotatedTable.setRowClamp(Number(value) as Exclude<TableRowClamp, 'all'>)
                  return
                }
                annotatedTable.setRowClamp('all')
              }}
            >
              <option value='2'>2 строки</option>
              <option value='4'>4 строки</option>
              <option value='8'>8 строк</option>
              <option value='all'>Полный текст</option>
            </select>
          </label>
          <span className='lab-muted'>В таблице: {filteredAnnotatedRows.length} из {annotatedRows.length}</span>
          <span className='lab-muted'>С новой подтематикой: {reclassifiedAnnotatedRows.length}</span>
          <span className='lab-muted'>Отрисовано сейчас: {virtualAnnotatedTable.virtualRows.length}</span>
        </div>

        <div className='workbook-table-wrap' ref={virtualAnnotatedTable.scrollRef} onScroll={virtualAnnotatedTable.onScroll} onMouseLeave={hideAnnotatedCellPopover}>
          <table className='table workbook-table'>
            <thead>
              <tr>
                <th style={annotatedTable.getColumnStyle('__classification')}>
                  <div className='table-simple-header-cell'>
                    <span>Класс</span>
                    <button
                      type='button'
                      className='table-column-resizer'
                      title='Потяните, чтобы изменить ширину колонки. Двойной клик сбрасывает ширину.'
                      aria-label='Изменить ширину колонки Класс'
                      onMouseDown={(e) => annotatedTable.startColumnResize(e, '__classification')}
                      onDoubleClick={() => annotatedTable.resetColumnWidth('__classification')}
                    />
                  </div>
                </th>
                <th style={annotatedTable.getColumnStyle('__new_class')}>
                  <div className='table-simple-header-cell'>
                    <span>Новая подтематика</span>
                    <button
                      type='button'
                      className='table-column-resizer'
                      title='Потяните, чтобы изменить ширину колонки. Двойной клик сбрасывает ширину.'
                      aria-label='Изменить ширину колонки Новая подтематика'
                      onMouseDown={(e) => annotatedTable.startColumnResize(e, '__new_class')}
                      onDoubleClick={() => annotatedTable.resetColumnWidth('__new_class')}
                    />
                  </div>
                </th>
                <th style={annotatedTable.getColumnStyle('__tags')}>
                  <div className='table-simple-header-cell'>
                    <span>Теги</span>
                    <button
                      type='button'
                      className='table-column-resizer'
                      title='Потяните, чтобы изменить ширину колонки. Двойной клик сбрасывает ширину.'
                      aria-label='Изменить ширину колонки Теги'
                      onMouseDown={(e) => annotatedTable.startColumnResize(e, '__tags')}
                      onDoubleClick={() => annotatedTable.resetColumnWidth('__tags')}
                    />
                  </div>
                </th>
                <th style={annotatedTable.getColumnStyle('__local_tags')}>
                  <div className='table-simple-header-cell'>
                    <span>Local tags</span>
                    <button
                      type='button'
                      className='table-column-resizer'
                      title='Потяните, чтобы изменить ширину колонки. Двойной клик сбрасывает ширину.'
                      aria-label='Изменить ширину колонки Local tags'
                      onMouseDown={(e) => annotatedTable.startColumnResize(e, '__local_tags')}
                      onDoubleClick={() => annotatedTable.resetColumnWidth('__local_tags')}
                    />
                  </div>
                </th>
                <th style={annotatedTable.getColumnStyle('__model_added_tags')}>
                  <div className='table-simple-header-cell'>
                    <span>Model added tags</span>
                    <button
                      type='button'
                      className='table-column-resizer'
                      title='Потяните, чтобы изменить ширину колонки. Двойной клик сбрасывает ширину.'
                      aria-label='Изменить ширину колонки Model added tags'
                      onMouseDown={(e) => annotatedTable.startColumnResize(e, '__model_added_tags')}
                      onDoubleClick={() => annotatedTable.resetColumnWidth('__model_added_tags')}
                    />
                  </div>
                </th>
                <th style={annotatedTable.getColumnStyle('__model_rejected_tags')}>
                  <div className='table-simple-header-cell'>
                    <span>Rejected tags</span>
                    <button
                      type='button'
                      className='table-column-resizer'
                      title='Потяните, чтобы изменить ширину колонки. Двойной клик сбрасывает ширину.'
                      aria-label='Изменить ширину колонки Rejected tags'
                      onMouseDown={(e) => annotatedTable.startColumnResize(e, '__model_rejected_tags')}
                      onDoubleClick={() => annotatedTable.resetColumnWidth('__model_rejected_tags')}
                    />
                  </div>
                </th>
                <th style={annotatedTable.getColumnStyle('__match_type')}>
                  <div className='table-simple-header-cell'>
                    <span>Match type</span>
                    <button
                      type='button'
                      className='table-column-resizer'
                      title='Потяните, чтобы изменить ширину колонки. Двойной клик сбрасывает ширину.'
                      aria-label='Изменить ширину колонки Match type'
                      onMouseDown={(e) => annotatedTable.startColumnResize(e, '__match_type')}
                      onDoubleClick={() => annotatedTable.resetColumnWidth('__match_type')}
                    />
                  </div>
                </th>
                <th style={annotatedTable.getColumnStyle('__evidence')}>
                  <div className='table-simple-header-cell'>
                    <span>Evidence</span>
                    <button
                      type='button'
                      className='table-column-resizer'
                      title='Потяните, чтобы изменить ширину колонки. Двойной клик сбрасывает ширину.'
                      aria-label='Изменить ширину колонки Evidence'
                      onMouseDown={(e) => annotatedTable.startColumnResize(e, '__evidence')}
                      onDoubleClick={() => annotatedTable.resetColumnWidth('__evidence')}
                    />
                  </div>
                </th>
                <th style={annotatedTable.getColumnStyle('__rule_hits')}>
                  <div className='table-simple-header-cell'>
                    <span>Rule hits</span>
                    <button
                      type='button'
                      className='table-column-resizer'
                      title='Потяните, чтобы изменить ширину колонки. Двойной клик сбрасывает ширину.'
                      aria-label='Изменить ширину колонки Rule hits'
                      onMouseDown={(e) => annotatedTable.startColumnResize(e, '__rule_hits')}
                      onDoubleClick={() => annotatedTable.resetColumnWidth('__rule_hits')}
                    />
                  </div>
                </th>
                <th style={annotatedTable.getColumnStyle('__rule_decision')}>
                  <div className='table-simple-header-cell'>
                    <span>Rule decision</span>
                    <button
                      type='button'
                      className='table-column-resizer'
                      title='Потяните, чтобы изменить ширину колонки. Двойной клик сбрасывает ширину.'
                      aria-label='Изменить ширину колонки Rule decision'
                      onMouseDown={(e) => annotatedTable.startColumnResize(e, '__rule_decision')}
                      onDoubleClick={() => annotatedTable.resetColumnWidth('__rule_decision')}
                    />
                  </div>
                </th>
                <th style={annotatedTable.getColumnStyle('__model_decision')}>
                  <div className='table-simple-header-cell'>
                    <span>Model decision</span>
                    <button
                      type='button'
                      className='table-column-resizer'
                      title='Потяните, чтобы изменить ширину колонки. Двойной клик сбрасывает ширину.'
                      aria-label='Изменить ширину колонки Model decision'
                      onMouseDown={(e) => annotatedTable.startColumnResize(e, '__model_decision')}
                      onDoubleClick={() => annotatedTable.resetColumnWidth('__model_decision')}
                    />
                  </div>
                </th>
                <th style={annotatedTable.getColumnStyle('__reclassified_topic')}>
                  <div className='table-simple-header-cell'>
                    <span>Reclassified topic</span>
                    <button
                      type='button'
                      className='table-column-resizer'
                      title='Потяните, чтобы изменить ширину колонки. Двойной клик сбрасывает ширину.'
                      aria-label='Изменить ширину колонки Reclassified topic'
                      onMouseDown={(e) => annotatedTable.startColumnResize(e, '__reclassified_topic')}
                      onDoubleClick={() => annotatedTable.resetColumnWidth('__reclassified_topic')}
                    />
                  </div>
                </th>
                <th style={annotatedTable.getColumnStyle('__final_topic')}>
                  <div className='table-simple-header-cell'>
                    <span>Final topic</span>
                    <button
                      type='button'
                      className='table-column-resizer'
                      title='Потяните, чтобы изменить ширину колонки. Двойной клик сбрасывает ширину.'
                      aria-label='Изменить ширину колонки Final topic'
                      onMouseDown={(e) => annotatedTable.startColumnResize(e, '__final_topic')}
                      onDoubleClick={() => annotatedTable.resetColumnWidth('__final_topic')}
                    />
                  </div>
                </th>
                <th style={annotatedTable.getColumnStyle('__decision_source')}>
                  <div className='table-simple-header-cell'>
                    <span>Decision source</span>
                    <button
                      type='button'
                      className='table-column-resizer'
                      title='Потяните, чтобы изменить ширину колонки. Двойной клик сбрасывает ширину.'
                      aria-label='Изменить ширину колонки Decision source'
                      onMouseDown={(e) => annotatedTable.startColumnResize(e, '__decision_source')}
                      onDoubleClick={() => annotatedTable.resetColumnWidth('__decision_source')}
                    />
                  </div>
                </th>
                {sheetData?.columns.map((column) => <th key={`annotated-head-${column}`} style={annotatedTable.getColumnStyle(column)}>
                  <div className='table-simple-header-cell'>
                    <span>{column}</span>
                    <button
                      type='button'
                      className='table-column-resizer'
                      title='Потяните, чтобы изменить ширину колонки. Двойной клик сбрасывает ширину.'
                      aria-label={`Изменить ширину колонки ${column}`}
                      onMouseDown={(e) => annotatedTable.startColumnResize(e, column)}
                      onDoubleClick={() => annotatedTable.resetColumnWidth(column)}
                    />
                  </div>
                </th>)}
              </tr>
            </thead>
            <tbody>
              {virtualAnnotatedTable.topSpacerHeight ? <tr aria-hidden='true' className='virtual-table-spacer-row'>
                <td colSpan={annotatedTableColumnCount} style={{ height: virtualAnnotatedTable.topSpacerHeight }} />
              </tr> : null}
              {virtualAnnotatedTable.virtualRows.map(({ item: row }) => <tr key={row.rowKey}>
                <td style={annotatedTable.getColumnStyle('__classification')}>
                  <div
                    className={annotatedTable.cellClampClassName}
                    style={annotatedTable.cellClampStyle}
                    onMouseEnter={(e) => showAnnotatedCellPopover(e, 'Класс', row.classification || '—')}
                    onMouseLeave={hideAnnotatedCellPopover}
                  >
                    {row.classification || '—'}
                  </div>
                </td>
                <td
                  className={isAnnotatedRowReclassified(row) ? 'annotated-new-class-cell reclassified' : 'annotated-new-class-cell'}
                  style={annotatedTable.getColumnStyle('__new_class')}
                >
                  <div
                    className={annotatedTable.cellClampClassName}
                    style={annotatedTable.cellClampStyle}
                    onMouseEnter={(e) => showAnnotatedCellPopover(
                      e,
                      'Новая подтематика',
                      normalizeReclassificationTopicValue(row.newClassification)
                        ? `${normalizeReclassificationTopicValue(row.newClassification)}\nТекущий класс: ${row.sourceClassification || row.classification || '—'}`
                        : '—',
                    )}
                    onMouseLeave={hideAnnotatedCellPopover}
                  >
                    {normalizeReclassificationTopicValue(row.newClassification) || '—'}
                  </div>
                </td>
                <td style={annotatedTable.getColumnStyle('__tags')}>
                  <div
                    className={annotatedTable.cellClampClassName}
                    style={annotatedTable.cellClampStyle}
                    onMouseEnter={(e) => showAnnotatedCellPopover(e, 'Теги', row.tags.length ? row.tags.join(', ') : '—')}
                    onMouseLeave={hideAnnotatedCellPopover}
                  >
                    {row.tags.length ? row.tags.join(', ') : '—'}
                  </div>
                </td>
                <td style={annotatedTable.getColumnStyle('__local_tags')}>
                  <div
                    className={annotatedTable.cellClampClassName}
                    style={annotatedTable.cellClampStyle}
                    onMouseEnter={(e) => showAnnotatedCellPopover(e, 'Local tags', row.localTags.length ? row.localTags.join(', ') : '—')}
                    onMouseLeave={hideAnnotatedCellPopover}
                  >
                    {row.localTags.length ? row.localTags.join(', ') : '—'}
                  </div>
                </td>
                <td style={annotatedTable.getColumnStyle('__model_added_tags')}>
                  <div
                    className={annotatedTable.cellClampClassName}
                    style={annotatedTable.cellClampStyle}
                    onMouseEnter={(e) => showAnnotatedCellPopover(e, 'Model added tags', row.modelAddedTags.length ? row.modelAddedTags.join(', ') : '—')}
                    onMouseLeave={hideAnnotatedCellPopover}
                  >
                    {row.modelAddedTags.length ? row.modelAddedTags.join(', ') : '—'}
                  </div>
                </td>
                <td style={annotatedTable.getColumnStyle('__model_rejected_tags')}>
                  <div
                    className={annotatedTable.cellClampClassName}
                    style={annotatedTable.cellClampStyle}
                    onMouseEnter={(e) => showAnnotatedCellPopover(e, 'Rejected tags', row.modelRejectedTags.length ? row.modelRejectedTags.join(', ') : '—')}
                    onMouseLeave={hideAnnotatedCellPopover}
                  >
                    {row.modelRejectedTags.length ? row.modelRejectedTags.join(', ') : '—'}
                  </div>
                </td>
                <td style={annotatedTable.getColumnStyle('__match_type')}>
                  <div
                    className={annotatedTable.cellClampClassName}
                    style={annotatedTable.cellClampStyle}
                    onMouseEnter={(e) => showAnnotatedCellPopover(e, 'Match type', row.matchType || '—')}
                    onMouseLeave={hideAnnotatedCellPopover}
                  >
                    {row.matchType || '—'}
                  </div>
                </td>
                <td style={annotatedTable.getColumnStyle('__evidence')}>
                  <div
                    className={annotatedTable.cellClampClassName}
                    style={annotatedTable.cellClampStyle}
                    onMouseEnter={(e) => showAnnotatedCellPopover(e, 'Evidence', row.evidence || row.tagDecisions || '—')}
                    onMouseLeave={hideAnnotatedCellPopover}
                  >
                    {row.evidence || row.tagDecisions || '—'}
                  </div>
                </td>
                <td style={annotatedTable.getColumnStyle('__rule_hits')}>
                  <div
                    className={annotatedTable.cellClampClassName}
                    style={annotatedTable.cellClampStyle}
                    onMouseEnter={(e) => showAnnotatedCellPopover(e, 'Rule hits', row.ruleHits.length ? row.ruleHits.join(', ') : '—')}
                    onMouseLeave={hideAnnotatedCellPopover}
                  >
                    {row.ruleHits.length ? row.ruleHits.join(', ') : '—'}
                  </div>
                </td>
                <td style={annotatedTable.getColumnStyle('__rule_decision')}>
                  <div
                    className={annotatedTable.cellClampClassName}
                    style={annotatedTable.cellClampStyle}
                    onMouseEnter={(e) => showAnnotatedCellPopover(e, 'Rule decision', row.ruleDecision || '—')}
                    onMouseLeave={hideAnnotatedCellPopover}
                  >
                    {row.ruleDecision || '—'}
                  </div>
                </td>
                <td style={annotatedTable.getColumnStyle('__model_decision')}>
                  <div
                    className={annotatedTable.cellClampClassName}
                    style={annotatedTable.cellClampStyle}
                    onMouseEnter={(e) => showAnnotatedCellPopover(e, 'Model decision', row.modelDecision || '—')}
                    onMouseLeave={hideAnnotatedCellPopover}
                  >
                    {row.modelDecision || '—'}
                  </div>
                </td>
                <td style={annotatedTable.getColumnStyle('__reclassified_topic')}>
                  <div
                    className={annotatedTable.cellClampClassName}
                    style={annotatedTable.cellClampStyle}
                    onMouseEnter={(e) => showAnnotatedCellPopover(e, 'Reclassified topic', row.reclassifiedTopic || '—')}
                    onMouseLeave={hideAnnotatedCellPopover}
                  >
                    {row.reclassifiedTopic || '—'}
                  </div>
                </td>
                <td style={annotatedTable.getColumnStyle('__final_topic')}>
                  <div
                    className={annotatedTable.cellClampClassName}
                    style={annotatedTable.cellClampStyle}
                    onMouseEnter={(e) => showAnnotatedCellPopover(e, 'Final topic', row.finalTopic || '—')}
                    onMouseLeave={hideAnnotatedCellPopover}
                  >
                    {row.finalTopic || '—'}
                  </div>
                </td>
                <td style={annotatedTable.getColumnStyle('__decision_source')}>
                  <div
                    className={annotatedTable.cellClampClassName}
                    style={annotatedTable.cellClampStyle}
                    onMouseEnter={(e) => showAnnotatedCellPopover(e, 'Decision source', row.decisionSource || '—')}
                    onMouseLeave={hideAnnotatedCellPopover}
                  >
                    {row.decisionSource || '—'}
                  </div>
                </td>
                {sheetData?.columns.map((column) => {
                  const text = String(row.sourceRow[column] ?? '')
                  return <td key={`${row.rowKey}-${column}`} style={annotatedTable.getColumnStyle(column)}>
                    <div
                      className={annotatedTable.cellClampClassName}
                      style={annotatedTable.cellClampStyle}
                      onMouseEnter={(e) => showAnnotatedCellPopover(e, column, text)}
                      onMouseLeave={hideAnnotatedCellPopover}
                    >
                      {text}
                    </div>
                  </td>
                })}
              </tr>)}
              {virtualAnnotatedTable.bottomSpacerHeight ? <tr aria-hidden='true' className='virtual-table-spacer-row'>
                <td colSpan={annotatedTableColumnCount} style={{ height: virtualAnnotatedTable.bottomSpacerHeight }} />
              </tr> : null}
            </tbody>
          </table>
        </div>
        <CellHoverPopover hoveredCell={hoveredAnnotatedCell} />
        {exportAnnotatedRows.isError ? <div className='transport-error'>{formatLabError(exportAnnotatedRows.error as Error, 'размеченную таблицу')}</div> : null}
        {exportValidationRows.isError ? <div className='transport-error'>{formatLabError(exportValidationRows.error as Error, 'validation-файл')}</div> : null}
      </>}
    </section>
    </>}

    <WorkbookSheetPickerModal
      workbook={workbookMeta}
      open={sheetPickerOpen}
      selectedSheetName={selectedSheetName}
      busySheetName={selectSheet.isPending ? selectSheet.variables?.sheetName : null}
      onClose={() => setSheetPickerOpen(false)}
      onSelect={(sheetName) => {
        selectWorkbookSheet(sheetName)
      }}
    />

    {ruleGuardIssues.length ? <div className='sheet-modal-backdrop' onClick={cancelRuleGuard}>
      <div className='card sheet-modal rule-guard-modal' onClick={(e) => e.stopPropagation()}>
        <div className='transport-section-head'>
          <div className='transport-section-title'>
            <h3>Часть правил будет выключена</h3>
            <p>В текущей таблице нет колонок для этих правил. Чтобы отправить данные в GigaChat, мы выключим только проблемные правила и продолжим с остальными.</p>
          </div>
          <button className='transport-collapse-button' type='button' onClick={cancelRuleGuard}>Отмена</button>
        </div>
        <div className='rule-guard-list'>
          {ruleGuardIssues.map((issue) => <div className='rule-guard-item' key={issue.code}>
            <strong>{issue.code}</strong>
            <span>Нет колонок: {issue.missingFields.join(', ')}</span>
          </div>)}
        </div>
        <div className='transport-actions'>
          <button className='primary' type='button' onClick={confirmRuleGuard}>Продолжить и выключить</button>
          <button type='button' onClick={cancelRuleGuard}>Не отправлять</button>
        </div>
      </div>
    </div> : null}

    {finalPromptModalOpen ? <div className='sheet-modal-backdrop' onClick={() => setFinalPromptModalOpen(false)}>
      <div className='card sheet-modal' onClick={(e) => e.stopPropagation()}>
        <div className='transport-section-head'>
          <div className='transport-section-title'>
            <h3>Итоговый JSON для GigaChat</h3>
            <p>Это payload, который будет отправляться в GigaChat для одной жалобы. Колонки из таблицы подставлены как плейсхолдеры.</p>
          </div>
          <button className='transport-collapse-button' onClick={() => setFinalPromptModalOpen(false)}>Закрыть</button>
        </div>
        {previewFinalPrompt.isPending && !finalPromptPreview ? <div>Собираем актуальный итоговый промпт...</div> : null}
        {finalPromptPreview ? <>
          <div className='lab-settings-meta'>
            <div><b>Generated at:</b> {new Date(finalPromptPreview.generated_at).toLocaleString()}</div>
            <div><b>Колонки:</b> {finalPromptPreview.source_columns.length}</div>
            {finalPromptPreview.saved_path ? <div><b>Сохранено в:</b> <code>{finalPromptPreview.saved_path}</code></div> : null}
          </div>
          <textarea
            className='final-prompt-editor'
            value={finalPromptDraftText}
            onChange={(e) => setFinalPromptDraftText(e.target.value)}
            spellCheck={false}
          />
          {finalPromptDraftValidation.error ? <div className='transport-error'>{finalPromptDraftValidation.error}</div> : <div className='lab-muted'>JSON валиден. Сохраненные правки будут использованы при отправке строки в GigaChat.</div>}
          <div className='lab-settings-actions final-prompt-editor-actions'>
            <button
              className={finalPromptDraftDirty ? 'primary' : ''}
              onClick={() => {
                if (!finalPromptDraftValidation.parsed) return
                setFinalPromptBaseText(finalPromptDraftText)
                setFinalPromptOverrideText(finalPromptDraftText)
              }}
              disabled={!finalPromptDraftDirty || Boolean(finalPromptDraftValidation.error)}
            >
              Сохранить
            </button>
            <button
              onClick={() => {
                setFinalPromptDraftText(finalPromptBaseText)
                setFinalPromptOverrideText(null)
              }}
              disabled={!finalPromptDraftDirty}
            >
              Сбросить
            </button>
          </div>
        </> : null}
      </div>
    </div> : null}

    {rowRunModalOpen && activeRowRunPage && activeRowRunResult ? <div className='sheet-modal-backdrop' onClick={() => setRowRunModalOpen(false)}>
      <div className='card sheet-modal' onClick={(e) => e.stopPropagation()}>
        <div className='transport-section-head'>
          <div className='transport-section-title'>
            <h3>Запрос в GigaChat по строке</h3>
            <p>Здесь видно, какие payload были отправлены в GigaChat и что модель вернула в ответ.</p>
          </div>
          <button className='transport-collapse-button' onClick={() => setRowRunModalOpen(false)}>Закрыть</button>
        </div>
        <div className='row-run-pages'>
          <div className='row-run-page-tabs' role='tablist' aria-label='Запросы GigaChat по строке'>
            {rowRunResultPages.map((page, index) => <button
              key={page.id}
              type='button'
              className={index === activeRowRunPageIndex ? 'active' : ''}
              onClick={() => setActiveRowRunResultPage(index)}
            >
              {page.title}
            </button>)}
          </div>
          <div className='row-run-page-actions'>
            <button
              type='button'
              onClick={() => setActiveRowRunResultPage((current) => Math.max(0, current - 1))}
              disabled={activeRowRunPageIndex <= 0}
            >
              Назад
            </button>
            <span>{activeRowRunPageIndex + 1} из {rowRunResultPages.length}</span>
            <button
              type='button'
              onClick={() => setActiveRowRunResultPage((current) => Math.min(rowRunResultPages.length - 1, current + 1))}
              disabled={activeRowRunPageIndex >= rowRunResultPages.length - 1}
            >
              Дальше
            </button>
          </div>
        </div>
        <div className='row-run-page-summary'>
          <strong>{activeRowRunPage.title}</strong>
          <span>{activeRowRunPage.description}</span>
        </div>
        <div className='lab-settings-meta'>
          <div><b>Transport:</b> <code>{activeRowRunResult.transport}</code></div>
          <div><b>JSON parse:</b> {activeRowRunResult.parse_ok ? 'успешно' : 'не удалось распарсить'}</div>
          <div><b>Токены запроса:</b> {activeRowRunResult.request_token_count ?? 'не считали'}</div>
          <div><b>Rule hits:</b> {activeRowRunResult.rule_evaluation?.hits.length ? activeRowRunResult.rule_evaluation.hits.map((item) => item.code).join(', ') : 'нет'}</div>
        </div>
        <div className='row-run-grid'>
          <section className='row-run-panel'>
            <h4>Что ушло в GigaChat</h4>
            <pre className='final-prompt-json'>{stringifyJson(activeRowRunResult.request_payload)}</pre>
          </section>
          <section className='row-run-panel'>
            <h4>Что вернул GigaChat</h4>
            <pre className='final-prompt-json'>{activeRowRunResult.parse_ok ? stringifyJson(activeRowRunResult.response_json) : activeRowRunResult.response_raw}</pre>
          </section>
        </div>
      </div>
    </div> : null}

    <GigaChatProcessingOverlay
      active={Boolean(processingOverlay)}
      minimized={Boolean(processingOverlay?.minimized)}
      title={processingOverlay?.title ?? ''}
      completed={processingOverlay?.completed ?? 0}
      total={processingOverlay?.total ?? 0}
      currentLabel={processingOverlay?.currentLabel}
      onBackground={() => setProcessingOverlay((current) => current ? { ...current, minimized: true } : current)}
      onRestore={() => setProcessingOverlay((current) => current ? { ...current, minimized: false } : current)}
    />
  </div>
}
