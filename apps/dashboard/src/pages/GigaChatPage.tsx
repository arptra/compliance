import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  getGigaChatBackgroundTaskResultWithProgress,
  createGigaChatLabSettingsVersion,
  exportGigaChatAnnotatedWorkbook,
  exportGigaChatValidationWorkbook,
  getGigaChatLabSettingsVersion,
  getGigaChatStatus,
  listGigaChatLabSettingsVersions,
  probeGigaChatTransport,
  previewGigaChatFinalPrompt,
  runGigaChatWorkbookRow,
  saveGigaChatLabSettingsVersion,
  saveGigaChatFinalPrompt,
  selectGigaChatWorkbookSheet,
  startGigaChatBackgroundTask,
  uploadLocalGigaChatWorkbook,
  uploadGigaChatWorkbook,
} from '../features/gigachat/api'
import { GigaChatProcessingOverlay } from '../features/gigachat/GigaChatProcessingOverlay'
import { LabelingRulesEditor } from '../features/gigachat/LabelingRulesEditor'
import { evaluateRulePacksLocally, parseRulePacks } from '../features/gigachat/rulePackMatcher'
import { RulePackEditor } from '../features/gigachat/RulePackEditor'
import { GigaChatSettingsForm } from '../features/gigachat/GigaChatSettingsForm'
import { WorkbookSheetPickerModal } from '../features/gigachat/WorkbookSheetPickerModal'
import { WorkbookSheetTable } from '../features/gigachat/WorkbookSheetTable'
import { GigaChatTransportCard } from '../features/gigachat/GigaChatTransportCard'
import { CellHoverPopover, useCellHoverPopover } from '../features/gigachat/useCellHoverPopover'
import { useResizableTable, type TableRowClamp } from '../features/gigachat/useResizableTable'
import { useVirtualTableRows } from '../features/gigachat/useVirtualTableRows'
import type {
  GigaChatFinalPromptResponse,
  GigaChatBackgroundTaskResultResponse,
  GigaChatLabRowRunResponse,
  GigaChatRuleEvaluationResponse,
  GigaChatRulePack,
  GigaChatSettingsVersionStatus,
  GigaChatRuleEvaluationRow,
  GigaChatTransportName,
  GigaChatWorkbookSheetDataResponse,
  GigaChatWorkbookUploadResponse,
} from '../features/gigachat/types'

type WorkbookRowLimit = 10 | 20 | 100 | 'all'
type GigaChatLabTab = 'workspace' | 'settings'
type LabSetupTab = 'prompts' | 'labels' | 'rules'
const WORKBOOK_UPLOAD_TIMEOUT_MS = 15_000
const RULE_EVALUATION_CHUNK_COUNT = 25
const VERSION_STATUS_LABELS: Record<GigaChatSettingsVersionStatus, string> = {
  draft: 'Черновая',
  test: 'Тестовая',
  working: 'Рабочая',
  release: 'Релизная',
  archived: 'Архивная',
}

function nextFrame() {
  return new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()))
}

function formatBytes(bytes: number) {
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} МБ`
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} КБ`
  return `${bytes} Б`
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

type TokenAccountingState = {
  enabled: boolean
  pricePer1k: number
  totalTokens: number
}

type RuleGuardIssue = {
  code: string
  missingFields: string[]
}

type PendingRuleGuardAction = (values: Record<string, unknown>) => void | Promise<void>

const TOKEN_ACCOUNTING_STORAGE_KEY = 'gigachat-lab-token-accounting'
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

function hasRulePacks(value: unknown) {
  const text = String(value ?? '').trim()
  if (!text) return false
  try {
    const parsed = JSON.parse(text) as unknown
    return Array.isArray(parsed) && parsed.length > 0
  } catch {
    return false
  }
}

function stringifyJson(value: unknown) {
  return JSON.stringify(value, null, 2)
}

function buildSheetRowKey(uploadId: string, sheetName: string, rowIndex: number) {
  return `${uploadId}:${sheetName}:${rowIndex}`
}

function findRuleGuardIssues(values: Record<string, unknown>, columns: string[]) {
  if (!columns.length) return [] as RuleGuardIssue[]
  const available = new Set(columns)
  return parseRulePacks(values.rule_pack_prompt_notes)
    .flatMap((rule) => {
      if (!rule.source_fields.length) {
        return [{ code: rule.code || 'Без кода', missingFields: ['Source fields не выбраны'] }]
      }
      const ruleFields = [
        ...rule.source_fields,
        ...rule.filters.map((filterItem) => filterItem.field).filter(Boolean),
      ]
      const missingFields = Array.from(new Set(ruleFields.filter((field) => !available.has(field))))
      return missingFields.length ? [{ code: rule.code || 'Без кода', missingFields }] : []
    })
}

function disableRulesWithMissingFields(values: Record<string, unknown>, columns: string[]) {
  const available = new Set(columns)
  const rules = parseRulePacks(values.rule_pack_prompt_notes)
  const nextRules = rules.map((rule): GigaChatRulePack => {
    if (!rule.source_fields.length) return rule.enabled ? { ...rule, enabled: false } : rule
    const ruleFields = [
      ...rule.source_fields,
      ...rule.filters.map((filterItem) => filterItem.field).filter(Boolean),
    ]
    const missing = ruleFields.some((field) => !available.has(field))
    return rule.enabled && missing ? { ...rule, enabled: false } : rule
  })
  return {
    ...values,
    rule_pack_prompt_notes: JSON.stringify(nextRules, null, 2),
  }
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

function extractTags(responseJson: unknown) {
  if (!responseJson || typeof responseJson !== 'object' || Array.isArray(responseJson)) return [] as string[]
  const record = responseJson as Record<string, unknown>
  const assignedTags = record.assigned_tags
  if (Array.isArray(assignedTags)) {
    return assignedTags.map((item) => String(item ?? '').trim()).filter(Boolean)
  }
  const tags = record.tags
  if (Array.isArray(tags)) {
    return tags.flatMap((item) => {
      if (typeof item === 'string') return item.trim() ? [item.trim()] : []
      if (item && typeof item === 'object' && !Array.isArray(item)) {
        const tagValue = (item as Record<string, unknown>).tag
        return typeof tagValue === 'string' && tagValue.trim() ? [tagValue.trim()] : []
      }
      return []
    })
  }
  return []
}

function extractStringList(value: unknown) {
  return Array.isArray(value)
    ? value.map((item) => String(item ?? '').trim()).filter(Boolean)
    : []
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
  return extractStringList((responseJson as Record<string, unknown>).confirmed_rule_hits)
}

function extractRejectedRuleHits(responseJson: unknown) {
  if (!responseJson || typeof responseJson !== 'object' || Array.isArray(responseJson)) return [] as string[]
  return extractStringList((responseJson as Record<string, unknown>).rejected_rule_hits)
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
      if (!item || typeof item !== 'object' || Array.isArray(item)) return String(item ?? '').trim()
      const decision = item as Record<string, unknown>
      const tag = String(decision.tag ?? '').trim()
      const status = String(decision.decision ?? '').trim()
      const matchType = String(decision.match_type ?? '').trim()
      const evidence = String(decision.evidence ?? '').trim()
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

function extractReclassifiedTopic(responseJson: unknown) {
  if (!responseJson || typeof responseJson !== 'object' || Array.isArray(responseJson)) return ''
  const record = responseJson as Record<string, unknown>
  for (const key of ['reclassified_topic', 'suggested_topic', 'target_topic']) {
    const value = record[key]
    if (typeof value === 'string' && value.trim()) return value.trim()
  }
  return ''
}

function formatLabError(error: Error | null | undefined, resourceLabel: string) {
  if (!error) return ''
  const raw = String(error.message || '').trim()
  if (error.name === 'AbortError' || raw === 'WORKBOOK_UPLOAD_TIMEOUT') {
    return `Не удалось загрузить ${resourceLabel}: браузер не получил ответ от локального API за ${Math.round(WORKBOOK_UPLOAD_TIMEOUT_MS / 1000)} секунд, fallback по имени файла тоже не сработал.`
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

export default function GigaChatPage() {
  const qc = useQueryClient()
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const labelingFieldKeys = new Set(['classification_prompt_notes', 'tagging_prompt_notes', 'rule_pack_prompt_notes'])
  const statusQ = useQuery({
    queryKey: ['gigachat-status'],
    queryFn: getGigaChatStatus,
    staleTime: 30_000,
    refetchOnWindowFocus: false,
  })
  const [selectedSettingsVersionId, setSelectedSettingsVersionId] = useState('default')
  const [versionCreateOpen, setVersionCreateOpen] = useState(false)
  const [versionDraft, setVersionDraft] = useState({
    title: '',
    versionId: '',
    description: '',
    status: 'draft' as GigaChatSettingsVersionStatus,
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

  const [selectedTransport, setSelectedTransport] = useState<GigaChatTransportName>('mtls')
  const [heroCollapsed, setHeroCollapsed] = useState(false)
  const [activeLabTab, setActiveLabTab] = useState<GigaChatLabTab>('workspace')
  const [settingsCollapsed, setSettingsCollapsed] = useState(false)
  const [activeSetupTab, setActiveSetupTab] = useState<LabSetupTab>('prompts')
  const [uploadCollapsed, setUploadCollapsed] = useState(false)
  const [workbookCollapsed, setWorkbookCollapsed] = useState(false)
  const [resultCollapsed, setResultCollapsed] = useState(false)
  const [collapsedTransports, setCollapsedTransports] = useState<Record<GigaChatTransportName, boolean>>({
    mtls: false,
    token: false,
  })
  const [settingValues, setSettingValues] = useState<Record<string, unknown>>({})
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [fileInputVersion, setFileInputVersion] = useState(0)
  const [workbookMeta, setWorkbookMeta] = useState<GigaChatWorkbookUploadResponse | null>(null)
  const [sheetData, setSheetData] = useState<GigaChatWorkbookSheetDataResponse | null>(null)
  const [sheetPickerOpen, setSheetPickerOpen] = useState(false)
  const [finalPromptPreview, setFinalPromptPreview] = useState<GigaChatFinalPromptResponse | null>(null)
  const [finalPromptModalOpen, setFinalPromptModalOpen] = useState(false)
  const [finalPromptDraftText, setFinalPromptDraftText] = useState('')
  const [finalPromptBaseText, setFinalPromptBaseText] = useState('')
  const [finalPromptOverrideText, setFinalPromptOverrideText] = useState<string | null>(null)
  const [rowRunModalOpen, setRowRunModalOpen] = useState(false)
  const [rowRunResult, setRowRunResult] = useState<GigaChatLabRowRunResponse | null>(null)
  const [workbookRowLimit, setWorkbookRowLimit] = useState<WorkbookRowLimit>(10)
  const [includedPromptColumns, setIncludedPromptColumns] = useState<string[]>([])
  const [selectedSheetRowKeys, setSelectedSheetRowKeys] = useState<string[]>([])
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
  const [processingOverlay, setProcessingOverlay] = useState<ProcessingOverlayState | null>(null)
  const [ruleGuardIssues, setRuleGuardIssues] = useState<RuleGuardIssue[]>([])
  const [pendingRuleGuardAction, setPendingRuleGuardAction] = useState<PendingRuleGuardAction | null>(null)
  const [pendingRuleGuardValues, setPendingRuleGuardValues] = useState<Record<string, unknown> | null>(null)
  const [tokenAccounting, setTokenAccounting] = useState<TokenAccountingState>(() => loadTokenAccountingState())
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
    window.localStorage.setItem(TOKEN_ACCOUNTING_STORAGE_KEY, JSON.stringify(tokenAccounting))
  }, [tokenAccounting])

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
    if (!versions.some((version) => version.version_id === selectedSettingsVersionId)) {
      setSelectedSettingsVersionId(versions[0].version_id)
      setVersionDraft((current) => ({ ...current, baseVersionId: versions[0].version_id }))
    }
  }, [selectedSettingsVersionId, versionsQ.data])

  useEffect(() => {
    if (!selectedVersionQ.data?.fields?.length) return
    const values = fieldsToValues(selectedVersionQ.data.fields)
    if (!hasRulePacks(values.rule_pack_prompt_notes)) {
      values.rule_pack_prompt_notes = DEFAULT_RULE_PACK_PROMPT_NOTES
    }
    setSettingValues(values)
  }, [selectedVersionQ.data])

  const probe = useMutation({
    mutationFn: (transport: GigaChatTransportName) => probeGigaChatTransport(transport),
  })

  const saveSettings = useMutation({
    mutationFn: (valuesOverride?: Record<string, unknown>) => saveGigaChatLabSettingsVersion(selectedSettingsVersionId, {
      title: selectedVersionQ.data?.version.title,
      description: selectedVersionQ.data?.version.description,
      status: selectedVersionQ.data?.version.status,
      values: valuesOverride ?? settingValues,
    }),
    onSuccess: async (data) => {
      setSettingValues(data.values)
      await qc.invalidateQueries({ queryKey: ['gigachat-lab-settings-versions'] })
      await qc.invalidateQueries({ queryKey: ['gigachat-lab-settings-version', selectedSettingsVersionId] })
      await qc.invalidateQueries({ queryKey: ['gigachat-status'] })
    },
  })

  const persistRulePacks = (value: string) => {
    const nextValues = { ...settingValues, rule_pack_prompt_notes: value }
    setSettingValues(nextValues)
    saveSettings.mutate(nextValues)
  }

  const createSettingsVersion = useMutation({
    mutationFn: () => createGigaChatLabSettingsVersion({
      title: versionDraft.title,
      version_id: versionDraft.versionId || null,
      description: versionDraft.description,
      status: versionDraft.status,
      created_by: versionDraft.createdBy,
      base_version_id: versionDraft.baseVersionId,
    }),
    onSuccess: async (data) => {
      setSelectedSettingsVersionId(data.version.version_id)
      setSettingValues(data.values)
      setVersionCreateOpen(false)
      setVersionDraft((current) => ({ ...current, title: '', versionId: '', description: '' }))
      await qc.invalidateQueries({ queryKey: ['gigachat-lab-settings-versions'] })
    },
  })

  const selectSheet = useMutation({
    mutationFn: ({ uploadId, sheetName, rowLimit }: { uploadId: string; sheetName: string; rowLimit: number }) => selectGigaChatWorkbookSheet(uploadId, sheetName, rowLimit),
    onSuccess: (data) => {
      setIncludedPromptColumns((current) => {
        const sameSheet = sheetData?.upload_id === data.upload_id && sheetData?.sheet_name === data.sheet_name
        if (!sameSheet || !current.length) return [...data.columns]
        const filtered = current.filter((column) => data.columns.includes(column))
        return filtered.length ? filtered : [...data.columns]
      })
      setSheetData(data)
      setWorkbookCollapsed(false)
      setSheetPickerOpen(false)
    },
  })

  const uploadWorkbook = useMutation({
    mutationFn: async () => {
      if (!selectedFile) throw new Error('Выберите Excel или CSV файл')
      try {
        return await uploadLocalGigaChatWorkbook(selectedFile.name)
      } catch {
        // Fall back to browser multipart for files that are not present in local project folders.
      }
      workbookUploadAbortRef.current?.abort()
      const controller = new AbortController()
      workbookUploadAbortRef.current = controller
      const timeoutId = window.setTimeout(() => {
        controller.abort(new DOMException('WORKBOOK_UPLOAD_TIMEOUT', 'AbortError'))
      }, WORKBOOK_UPLOAD_TIMEOUT_MS)
      const form = new FormData()
      form.append('file', selectedFile)
      try {
        return await uploadGigaChatWorkbook(form, controller.signal)
      } catch (error) {
        if (error instanceof DOMException && error.name === 'AbortError') {
          return uploadLocalGigaChatWorkbook(selectedFile.name)
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
      setWorkbookMeta(data)
      setSheetData(null)
      setSelectedFile(null)
      setFileInputVersion((current) => current + 1)
      setWorkbookCollapsed(false)
      if (data.sheet_count <= 1 && data.sheets[0]) {
        const rowLimit = workbookRowLimit === 'all' ? data.sheets[0].rows_total : workbookRowLimit
        selectSheet.mutate({ uploadId: data.upload_id, sheetName: data.sheets[0].name, rowLimit })
      } else {
        setSheetPickerOpen(true)
      }
    },
    onError: () => {
      workbookUploadAbortRef.current = null
    },
  })

  const finalPromptColumns = useMemo(
    () => (sheetData ? sheetData.columns.filter((column) => includedPromptColumns.includes(column)) : []),
    [sheetData, includedPromptColumns],
  )
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

  const saveFinalPrompt = useMutation({
    mutationFn: () => saveGigaChatFinalPrompt(settingValues, finalPromptColumns),
    onSuccess: (data) => {
      setFinalPromptPreview(data)
      const nextBaseText = stringifyJson(data.payload)
      setFinalPromptBaseText(nextBaseText)
      setFinalPromptDraftText(nextBaseText)
      setFinalPromptOverrideText(null)
      lastResolvedPreviewKeyRef.current = finalPromptRequestKey
      setFinalPromptModalOpen(true)
      void qc.invalidateQueries({ queryKey: ['gigachat-lab-settings'] })
    },
  })

  const exportAnnotatedRows = useMutation({
    mutationFn: async () => {
      if (!sheetData) throw new Error('Сначала загрузите рабочую таблицу.')
      return exportGigaChatAnnotatedWorkbook(
        sheetData.filename,
        sheetData.sheet_name,
        sheetData.columns,
        filteredAnnotatedRows.map((row) => ({
          row_index: row.rowIndex,
          classification: row.classification,
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
        })),
        (loadedBytes, totalBytes) => {
          setExportProgress({ label: 'Выгружаем Excel', loadedBytes, totalBytes })
        },
      )
    },
    onSuccess: ({ blob, filename }) => {
      const sourceName = sheetData?.filename ?? 'annotated.xlsx'
      const fallbackName = `${sourceName.replace(/\.[^.]+$/u, '') || 'annotated'}_annotated.xlsx`
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
        filteredAnnotatedRows.map((row) => ({
          row_index: row.rowIndex,
          classification: row.classification,
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
        })),
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
    const reclassifiedTopic = extractReclassifiedTopic(data.response_json)
    const classification = extractClassification(data.response_json)
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
    const modelDecision = !data.parse_ok
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
    const decisionSource = modelAddedTags.length
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
      responseRaw: data.response_raw,
      responseJson: data.response_json,
      sourceRow: row,
    }
  }

  const applyBackgroundTaskResult = (data: GigaChatBackgroundTaskResultResponse) => {
    setSheetData(data.workbook)
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
    setAnnotatedRows(data.row_runs.flatMap((rowRun) => rowRun.result ? [buildAnnotatedRow(rowRun.result, rowRun.row_index, rowRun.source_row, data.workbook)] : []))
    setBackgroundTaskError(null)
    window.setTimeout(() => document.getElementById('gigachat-workbook-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50)
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
        .filter((item) => selectedSheetRowKeys.includes(buildSheetRowKey(sheetData.upload_id, sheetData.sheet_name, item.row_index)))
      if (!selectedRows.length) throw new Error('Сначала выберите хотя бы одну строку.')
      return startGigaChatBackgroundTask({
        transport: selectedTransport,
        values: valuesForRun ?? settingValues,
        columns: finalPromptColumns,
        rows: selectedRows,
        filename: sheetData.filename,
        sheet_name: sheetData.sheet_name,
        payload_override: resolvePayloadOverride(),
        count_tokens: tokenAccounting.enabled,
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

  const appendAnnotatedResult = (data: GigaChatLabRowRunResponse, rowIndex: number, row: Record<string, unknown>) => {
    if (!sheetData) return
    const annotatedRow = buildAnnotatedRow(data, rowIndex, row, sheetData)
    setAnnotatedRows((current) => {
      const next = current.filter((item) => item.rowKey !== annotatedRow.rowKey)
      next.unshift(annotatedRow)
      return next
    })
  }

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
    setRunRowBusyIndex(rowIndex)
    startProcessingOverlay('Обработка одной жалобы', 1)
    try {
      const data = await runGigaChatWorkbookRow(
        selectedTransport,
        valuesForRun,
        finalPromptColumns,
        row,
        resolvePayloadOverride(),
        tokenAccounting.enabled,
      )
      accumulateTokenCount(data.request_token_count)
      appendAnnotatedResult(data, rowIndex, row)
      updateProcessingOverlay(1, 1, 'Жалоба обработана. Открываем результат...')
      setRowRunResult(data)
      setRowRunModalOpen(true)
    } catch (error) {
      setRowRunError(formatLabError(error as Error, 'строку'))
    } finally {
      setRunRowBusyIndex(null)
      finishProcessingOverlay()
    }
  }

  const handleRunSelectedRows = async (valuesForRun = settingValues) => {
    if (!selected?.ready) {
      setBatchRunError(`Транспорт ${selected?.title ?? selectedTransport} сейчас не готов к отправке. Сначала выберите ready-вариант подключения.`)
      return
    }
    if (!sheetData) {
      setBatchRunError('Сначала загрузите рабочую таблицу.')
      return
    }

    const rowIndexes = sheetData.rows
      .map((_, index) => index)
      .filter((index) => selectedSheetRowKeys.includes(buildSheetRowKey(sheetData.upload_id, sheetData.sheet_name, index)))

    if (!rowIndexes.length) {
      setBatchRunError('Сначала выберите хотя бы одну строку.')
      return
    }

    setBatchRunError(null)
    setBatchBusy(true)
    startProcessingOverlay('Пакетная разметка жалоб', rowIndexes.length)

    try {
      const payloadOverride = resolvePayloadOverride()
      let lastResult: GigaChatLabRowRunResponse | null = null
      for (let idx = 0; idx < rowIndexes.length; idx += 1) {
        const rowIndex = rowIndexes[idx]
        const row = sheetData.rows[rowIndex]
        updateProcessingOverlay(idx, rowIndexes.length, `Обрабатываем запись ${idx + 1} из ${rowIndexes.length}`)
        const data = await runGigaChatWorkbookRow(selectedTransport, valuesForRun, finalPromptColumns, row, payloadOverride, tokenAccounting.enabled)
        accumulateTokenCount(data.request_token_count)
        appendAnnotatedResult(data, rowIndex, row)
        lastResult = data
        updateProcessingOverlay(
          idx + 1,
          rowIndexes.length,
          tokenAccounting.enabled && data.request_token_count
            ? `Готово ${idx + 1} из ${rowIndexes.length}. Последний запрос: ${data.request_token_count} токенов`
            : `Готово ${idx + 1} из ${rowIndexes.length}`,
        )
      }
      if (lastResult) {
        setRowRunResult(lastResult)
      }
    } catch (error) {
      setBatchRunError(formatLabError(error as Error, 'выбранные строки'))
    } finally {
      setBatchBusy(false)
      finishProcessingOverlay()
    }
  }

  const toggleTransportCard = (name: GigaChatTransportName) => {
    setCollapsedTransports((current) => ({ ...current, [name]: !current[name] }))
  }

  const transports = statusQ.data?.transports ?? []
  const selected = transports.find((item) => item.name === selectedTransport) ?? transports[0]
  const hasMultipleSheets = (workbookMeta?.sheet_count ?? 0) > 1
  const selectedSheetName = sheetData?.sheet_name ?? null
  const ruleEvaluationTotal = ruleEvaluationProgress?.total ?? sheetData?.rows.length ?? 0
  const ruleEvaluationProcessed = ruleEvaluationProgress?.processed ?? 0
  const ruleEvaluationRemaining = Math.max(0, ruleEvaluationTotal - ruleEvaluationProcessed)
  const ruleEvaluationPercent = ruleEvaluationTotal ? Math.round((ruleEvaluationProcessed / ruleEvaluationTotal) * 100) : 0
  const backgroundLoadedBytes = backgroundResultProgress?.loadedBytes ?? 0
  const backgroundTotalBytes = backgroundResultProgress?.totalBytes ?? null
  const backgroundLoadPercent = backgroundTotalBytes ? Math.min(100, Math.round((backgroundLoadedBytes / backgroundTotalBytes) * 100)) : null
  const exportLoadedBytes = exportProgress?.loadedBytes ?? 0
  const exportTotalBytes = exportProgress?.totalBytes ?? null
  const exportPercent = exportTotalBytes ? Math.min(100, Math.round((exportLoadedBytes / exportTotalBytes) * 100)) : null
  const exportBusy = exportAnnotatedRows.isPending || exportValidationRows.isPending
  const requestSettingsFields = (selectedVersionQ.data?.fields ?? []).filter((field) => !labelingFieldKeys.has(field.key))
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
  const allVisibleRowsSelected = visibleSheetRowKeys.length > 0 && visibleSheetRowKeys.every((key) => selectedSheetRowKeys.includes(key))
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
  const virtualAnnotatedTable = useVirtualTableRows(filteredAnnotatedRows, annotatedTable.rowClamp === 'all' ? 148 : 112)
  const annotatedTableColumnCount = 13 + (sheetData?.columns.length ?? 0)
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
  const rulePackOptions = useMemo(
    () => parseRulePacks(settingValues.rule_pack_prompt_notes).map((item) => item.code).filter(Boolean),
    [settingValues.rule_pack_prompt_notes],
  )
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
    evaluateRulePacks.mutate(sheetData.rows)
  }

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
          <p>Три рабочие зоны: промпты, справочники классов/тегов и локальные правила, которые проверяются до отправки в GigaChat.</p>
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
                  {version.title} · {VERSION_STATUS_LABELS[version.status]}
                </option>)}
              </select>
            </label>
            <div className='settings-version-actions'>
              <button type='button' onClick={() => {
                setVersionDraft((current) => ({ ...current, baseVersionId: selectedSettingsVersionId }))
                setVersionCreateOpen((current) => !current)
              }}>
                Создать версию
              </button>
              <button type='button' onClick={() => saveSettings.mutate(undefined)} disabled={saveSettings.isPending}>
                {saveSettings.isPending ? 'Сохраняем...' : 'Сохранить версию'}
              </button>
            </div>
          </div>

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
              <span>Создать на основе</span>
              <select value={versionDraft.baseVersionId} onChange={(e) => setVersionDraft((current) => ({ ...current, baseVersionId: e.target.value }))}>
                {(versionsQ.data?.versions ?? []).map((version) => <option key={version.version_id} value={version.version_id}>{version.title}</option>)}
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

          <div className='lab-settings-meta'>
            <div><b>Профиль:</b> <code>{selectedVersionQ.data.version.title}</code></div>
            <div><b>Статус:</b> {VERSION_STATUS_LABELS[selectedVersionQ.data.version.status]}</div>
            <div><b>Автор:</b> {selectedVersionQ.data.version.created_by || '—'}</div>
            <div><b>Основана на:</b> <code>{selectedVersionQ.data.version.base_version_id || '—'}</code></div>
            <div><b>Создана:</b> {selectedVersionQ.data.version.created_at ? new Date(selectedVersionQ.data.version.created_at).toLocaleString() : '—'}</div>
            <div><b>Последнее сохранение:</b> {selectedVersionQ.data.version.updated_at ? new Date(selectedVersionQ.data.version.updated_at).toLocaleString() : 'еще не сохраняли'}</div>
            <div className='lab-field wide'><b>Описание:</b> {selectedVersionQ.data.version.description || '—'}</div>
            <div className='lab-field wide'><b>Файл:</b> <code>{selectedVersionQ.data.version.path || 'виртуальная default-версия'}</code></div>
          </div>

          <div className='lab-setup-tabs' role='tablist' aria-label='GigaChat Lab settings tabs'>
            <button
              type='button'
              className={activeSetupTab === 'prompts' ? 'active' : ''}
              onClick={() => setActiveSetupTab('prompts')}
            >
              Промпты
            </button>
            <button
              type='button'
              className={activeSetupTab === 'labels' ? 'active' : ''}
              onClick={() => setActiveSetupTab('labels')}
            >
              Классы и теги
            </button>
            <button
              type='button'
              className={activeSetupTab === 'rules' ? 'active' : ''}
              onClick={() => setActiveSetupTab('rules')}
            >
              Правила
            </button>
          </div>

          {activeSetupTab === 'prompts' ? <div className='lab-tab-panel'>
            <GigaChatSettingsForm
              fields={requestSettingsFields}
              values={settingValues}
              busy={saveSettings.isPending}
              saveError={saveSettings.isError ? formatLabError(saveSettings.error as Error, 'настройки') : null}
              onChange={(key, value) => setSettingValues((current) => ({ ...current, [key]: value }))}
              onSave={() => saveSettings.mutate(undefined)}
            />
          </div> : null}

          {activeSetupTab === 'labels' ? <div className='lab-tab-panel labeling-grid two-columns'>
            <LabelingRulesEditor
              title='Классификации'
              description='Список классов для классификации: корзины, категории, подкатегории и краткие правила, как их выбирать.'
              addLabel='Добавить класс'
              clearLabel='Сбросить все классы'
              nameLabel='Класс'
              value={settingValues.classification_prompt_notes}
              onChange={(value) => setSettingValues((current) => ({ ...current, classification_prompt_notes: value }))}
              onClear={() => setSettingValues((current) => ({ ...current, classification_prompt_notes: '[]' }))}
            />

            <LabelingRulesEditor
              title='Теги'
              description='Список тегов и краткие описания, когда тег должен присваиваться обращению.'
              addLabel='Добавить тег'
              clearLabel='Сбросить все теги'
              nameLabel='Тег'
              value={settingValues.tagging_prompt_notes}
              onChange={(value) => setSettingValues((current) => ({ ...current, tagging_prompt_notes: value }))}
              onClear={() => setSettingValues((current) => ({ ...current, tagging_prompt_notes: '[]' }))}
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
            />
          </div> : null}
        </> : null}
      </> : null}

      <div className='lab-settings-actions'>
        <button onClick={() => saveFinalPrompt.mutate()} disabled={saveFinalPrompt.isPending}>
          {saveFinalPrompt.isPending ? 'Сохраняем итоговый промпт...' : 'Сохранить итоговый промпт'}
        </button>
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
          {previewFinalPrompt.isPending && !finalPromptPreview ? 'Собираем итоговый...' : 'Показать итоговый'}
        </button>
        <span className='lab-settings-status'>
          {previewFinalPrompt.isPending && !finalPromptSynchronized
            ? 'Итоговый промпт обновляется автоматически...'
            : finalPromptPreview && finalPromptSynchronized
              ? 'Итоговый промпт синхронизирован с текущими полями формы.'
              : 'Итоговый промпт будет собран автоматически после первого изменения формы.'}
        </span>
        {saveFinalPrompt.isError ? <span className='transport-error'>{formatLabError(saveFinalPrompt.error as Error, 'итоговый промпт')}</span> : null}
        {previewFinalPrompt.isError ? <span className='transport-error'>{formatLabError(previewFinalPrompt.error as Error, 'итоговый промпт')}</span> : null}
      </div>
    </section>

    <section className='card transport-result workbook-upload-card'>
      <div className='transport-section-head'>
        <div className='transport-section-title'>
          <h3>Excel для разметки</h3>
          <p>Загрузите локальный Excel, CSV или ZIP-архив с ними, затем выберите нужный лист и откройте его как рабочую таблицу прямо на странице.</p>
        </div>
        <button className='transport-collapse-button' onClick={() => setUploadCollapsed((current) => !current)}>
          {uploadCollapsed ? 'Развернуть' : 'Свернуть'}
        </button>
      </div>

      {uploadWorkbook.isPending ? <div className='workbook-upload-loader'>
        <div className='card workbook-upload-loader-card'>
          <div className='spinner workbook-upload-spinner' aria-label='uploading workbook' />
          <div><b>Загружаем файл...</b></div>
          <div className='lab-muted'>Во время загрузки выбор файла и кнопка заблокированы.</div>
          <button
            type='button'
            onClick={() => {
              workbookUploadAbortRef.current?.abort()
              workbookUploadAbortRef.current = null
              uploadWorkbook.reset()
            }}
          >
            Отменить загрузку
          </button>
        </div>
      </div> : null}

      {!uploadCollapsed ? <>
        <div className='workbook-upload-row'>
          <input
            key={fileInputVersion}
            type='file'
            accept='.xlsx,.xls,.xlsm,.csv,.zip'
            disabled={uploadWorkbook.isPending}
            onChange={(e) => setSelectedFile(e.target.files?.[0] ?? null)}
          />
          <button onClick={() => uploadWorkbook.mutate()} disabled={!selectedFile || uploadWorkbook.isPending}>
            {uploadWorkbook.isPending ? 'Загружаем...' : 'Загрузить'}
          </button>
          {selectedFile ? <span className='lab-muted'>Выбран файл: <code>{selectedFile.name}</code></span> : <span className='lab-muted'>Файл пока не выбран.</span>}
        </div>

        {uploadWorkbook.isError ? <div className='transport-error'>{formatLabError(uploadWorkbook.error as Error, 'файл')}</div> : null}
        {selectSheet.isError ? <div className='transport-error'>{formatLabError(selectSheet.error as Error, 'лист')}</div> : null}
        {workbookMeta ? <div className='lab-settings-meta'>
          <div><b>Файл:</b> <code>{workbookMeta.filename}</code></div>
          <div><b>Формат:</b> <code>{workbookMeta.file_format}</code></div>
          <div><b>Листов:</b> {workbookMeta.sheet_count}</div>
          <div><b>Выбранный лист:</b> <code>{selectedSheetName ?? 'еще не выбран'}</code></div>
        </div> : null}
      </> : null}
    </section>

    <section className='card transport-result rule-validation-card'>
      <div className='transport-section-head'>
        <div className='transport-section-title'>
          <h3>Проверка правил</h3>
          <p>Локальный прогон rule packs по текущему листу до GigaChat: здесь видно, какие правила сработали, по каким полям и сколько строк они нашли.</p>
        </div>
        <div className='transport-actions'>
          <button type='button' onClick={handleEvaluateRules} disabled={!sheetData || evaluateRulePacks.isPending}>
            {evaluateRulePacks.isPending ? 'Прогоняем правила...' : 'Прогнать правила на текущем листе'}
          </button>
        </div>
      </div>

      {!sheetData ? <p>Загрузите Excel и выберите лист, чтобы проверить rule packs.</p> : <>
        {evaluateRulePacks.isPending ? <div className='rule-progress-panel'>
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
          <h3>Рабочая таблица</h3>
          <p>Ниже отображаются колонки выбранного листа, строки для проверки структуры и быстрые действия по отправке одной записи в GigaChat.</p>
        </div>
        <button className='transport-collapse-button' onClick={() => setWorkbookCollapsed((current) => !current)}>
          {workbookCollapsed ? 'Развернуть' : 'Свернуть'}
        </button>
      </div>

      {evaluateRulePacks.isPending ? <div className='workbook-rule-lock'>
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
        {!workbookMeta ? <p>Сначала загрузите Excel или CSV файл.</p> : null}
        {workbookMeta && !sheetData && !selectSheet.isPending ? <div className='transport-actions'>
          <button
            onClick={() => {
              if (!workbookMeta) return
              if (hasMultipleSheets) {
                setSheetPickerOpen(true)
                return
              }
              const onlySheet = workbookMeta.sheets[0]
              if (onlySheet) {
                const rowLimit = workbookRowLimit === 'all' ? onlySheet.rows_total : workbookRowLimit
                selectSheet.mutate({ uploadId: workbookMeta.upload_id, sheetName: onlySheet.name, rowLimit })
              }
            }}
            disabled={!hasMultipleSheets && !workbookMeta.sheets[0]}
          >
            {hasMultipleSheets ? 'Выбрать лист' : 'Загрузить лист'}
          </button>
        </div> : null}
        {workbookMeta && !sheetData && !hasMultipleSheets && workbookMeta.sheets[0] ? <p className='lab-muted'>В файле найден один лист. Нажмите `Загрузить лист`, если автозагрузка не успела завершиться.</p> : null}
        {selectSheet.isPending ? <div>Загружаем лист <code>{selectSheet.variables?.sheetName ?? ''}</code>...</div> : null}
        {sheetData ? <WorkbookSheetTable
          data={sheetData}
          canChooseAnotherSheet={hasMultipleSheets}
          onChooseAnotherSheet={() => setSheetPickerOpen(true)}
          rowLimit={workbookRowLimit}
          onRowLimitChange={(value) => {
            setWorkbookRowLimit(value)
            if (!sheetData) return
            const rowLimit = value === 'all' ? sheetData.total_rows : value
            selectSheet.mutate({ uploadId: sheetData.upload_id, sheetName: sheetData.sheet_name, rowLimit })
          }}
          includedPromptColumns={includedPromptColumns}
          onTogglePromptColumn={(column) => {
            setIncludedPromptColumns((current) => {
              if (current.includes(column)) return current.filter((item) => item !== column)
              return [...current, column]
            })
          }}
          onRunRow={(row, rowIndex) => runWithRuleGuard((valuesForRun) => handleRunRow(row, rowIndex, valuesForRun))}
          runRowBusyIndex={runRowBusyIndex}
          selectedRowKeys={selectedSheetRowKeys}
          allVisibleRowsSelected={allVisibleRowsSelected}
          onToggleRowSelection={(rowIndex) => {
            if (!sheetData) return
            const key = buildSheetRowKey(sheetData.upload_id, sheetData.sheet_name, rowIndex)
            setSelectedSheetRowKeys((current) => current.includes(key) ? current.filter((item) => item !== key) : [...current, key])
          }}
          onToggleAllRows={(checked) => {
            if (!sheetData) return
            const keys = sheetData.rows.map((_, index) => buildSheetRowKey(sheetData.upload_id, sheetData.sheet_name, index))
            setSelectedSheetRowKeys((current) => {
              const rest = current.filter((key) => !keys.includes(key))
              return checked ? [...rest, ...keys] : rest
            })
          }}
          onPickRandomRows={() => {
            if (!sheetData) return
            const keys = sheetData.rows.map((_, index) => buildSheetRowKey(sheetData.upload_id, sheetData.sheet_name, index))
            const shuffled = [...keys].sort(() => Math.random() - 0.5)
            setSelectedSheetRowKeys((current) => {
              const rest = current.filter((key) => !keys.includes(key))
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
          onRunSelectedRows={() => runWithRuleGuard((valuesForRun) => handleRunSelectedRows(valuesForRun))}
          onRunSelectedRowsInBackground={() => runWithRuleGuard((valuesForRun) => startBackgroundTask.mutate(valuesForRun))}
          batchBusy={batchBusy}
          busy={batchBusy || runRowBusyIndex !== null || startBackgroundTask.isPending}
          ruleEvaluations={ruleEvaluationMap}
          rulePackOptions={rulePackOptions}
        /> : null}
        {rowRunError ? <div className='transport-error'>{rowRunError}</div> : null}
        {batchRunError ? <div className='transport-error'>{batchRunError}</div> : null}
        {backgroundTaskError ? <div className='transport-error'>{backgroundTaskError}</div> : null}
        {loadBackgroundTaskResult.isPending ? <div className='lab-muted'>Подгружаем рабочую тетрадь из фоновой задачи...</div> : null}
      </> : null}
    </section>

    <section className='card transport-result'>
      <div className='transport-section-head'>
        <div className='transport-section-title'>
          <h3>Размеченная таблица</h3>
          <p>Сюда сразу попадают ответы GigaChat. Слева добавлены колонки класса и тегов, дальше идут исходные колонки жалобы.</p>
        </div>
        {annotatedRows.length ? <div className='transport-actions'>
          <button
            type='button'
            onClick={() => exportAnnotatedRows.mutate()}
            disabled={exportBusy}
          >
            {exportAnnotatedRows.isPending ? 'Выгружаем Excel...' : 'Выгрузить в Excel'}
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
      busySheetName={selectSheet.isPending ? selectSheet.variables?.sheetName : null}
      onClose={() => setSheetPickerOpen(false)}
      onSelect={(sheetName) => {
        if (!workbookMeta) return
        const previewSheet = workbookMeta.sheets.find((sheet) => sheet.name === sheetName)
        const rowLimit = workbookRowLimit === 'all' ? (previewSheet?.rows_total ?? 200) : workbookRowLimit
        selectSheet.mutate({ uploadId: workbookMeta.upload_id, sheetName, rowLimit })
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

    {rowRunModalOpen && rowRunResult ? <div className='sheet-modal-backdrop' onClick={() => setRowRunModalOpen(false)}>
      <div className='card sheet-modal' onClick={(e) => e.stopPropagation()}>
        <div className='transport-section-head'>
          <div className='transport-section-title'>
            <h3>Запрос в GigaChat по строке</h3>
            <p>Здесь видно, какой payload был отправлен в GigaChat и что модель вернула в ответ.</p>
          </div>
          <button className='transport-collapse-button' onClick={() => setRowRunModalOpen(false)}>Закрыть</button>
        </div>
        <div className='lab-settings-meta'>
          <div><b>Transport:</b> <code>{rowRunResult.transport}</code></div>
          <div><b>JSON parse:</b> {rowRunResult.parse_ok ? 'успешно' : 'не удалось распарсить'}</div>
          <div><b>Токены запроса:</b> {rowRunResult.request_token_count ?? 'не считали'}</div>
          <div><b>Rule hits:</b> {rowRunResult.rule_evaluation?.hits.length ? rowRunResult.rule_evaluation.hits.map((item) => item.code).join(', ') : 'нет'}</div>
        </div>
        <div className='row-run-grid'>
          <section className='row-run-panel'>
            <h4>Что ушло в GigaChat</h4>
            <pre className='final-prompt-json'>{stringifyJson(rowRunResult.request_payload)}</pre>
          </section>
          <section className='row-run-panel'>
            <h4>Что вернул GigaChat</h4>
            <pre className='final-prompt-json'>{rowRunResult.parse_ok ? stringifyJson(rowRunResult.response_json) : rowRunResult.response_raw}</pre>
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
