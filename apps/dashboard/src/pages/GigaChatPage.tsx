import { useEffect, useMemo, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  exportGigaChatAnnotatedWorkbook,
  getGigaChatLabSettings,
  getGigaChatStatus,
  probeGigaChatTransport,
  previewGigaChatFinalPrompt,
  runGigaChatWorkbookRow,
  saveGigaChatLabSettings,
  saveGigaChatFinalPrompt,
  selectGigaChatWorkbookSheet,
  uploadGigaChatWorkbook,
} from '../features/gigachat/api'
import { GigaChatProcessingOverlay } from '../features/gigachat/GigaChatProcessingOverlay'
import { LabelingRulesEditor } from '../features/gigachat/LabelingRulesEditor'
import { GigaChatSettingsForm } from '../features/gigachat/GigaChatSettingsForm'
import { WorkbookSheetPickerModal } from '../features/gigachat/WorkbookSheetPickerModal'
import { WorkbookSheetTable } from '../features/gigachat/WorkbookSheetTable'
import { GigaChatTransportCard } from '../features/gigachat/GigaChatTransportCard'
import type {
  GigaChatFinalPromptResponse,
  GigaChatLabRowRunResponse,
  GigaChatTransportName,
  GigaChatWorkbookSheetDataResponse,
  GigaChatWorkbookUploadResponse,
} from '../features/gigachat/types'

type WorkbookRowLimit = 10 | 20 | 100 | 'all'
type AnnotatedSheetRow = {
  rowKey: string
  rowIndex: number
  classification: string
  tags: string[]
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

const TOKEN_ACCOUNTING_STORAGE_KEY = 'gigachat-lab-token-accounting'

function fieldsToValues(fields: Array<{ key: string; value: unknown }>) {
  return Object.fromEntries(fields.map((field) => [field.key, field.value]))
}

function stringifyJson(value: unknown) {
  return JSON.stringify(value, null, 2)
}

function buildSheetRowKey(uploadId: string, sheetName: string, rowIndex: number) {
  return `${uploadId}:${sheetName}:${rowIndex}`
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
  if (typeof record.complaint_category === 'string') return record.complaint_category
  const classification = record.classification
  if (classification && typeof classification === 'object' && !Array.isArray(classification)) {
    const cls = (classification as Record<string, unknown>).class
    if (typeof cls === 'string') return cls
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

function formatLabError(error: Error | null | undefined, resourceLabel: string) {
  if (!error) return ''
  const raw = String(error.message || '').trim()
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
  const labelingFieldKeys = new Set(['classification_prompt_notes', 'tagging_prompt_notes'])
  const statusQ = useQuery({
    queryKey: ['gigachat-status'],
    queryFn: getGigaChatStatus,
    staleTime: 30_000,
    refetchOnWindowFocus: false,
  })
  const settingsQ = useQuery({
    queryKey: ['gigachat-lab-settings'],
    queryFn: getGigaChatLabSettings,
    staleTime: 30_000,
    refetchOnWindowFocus: false,
  })

  const [selectedTransport, setSelectedTransport] = useState<GigaChatTransportName>('mtls')
  const [heroCollapsed, setHeroCollapsed] = useState(false)
  const [settingsCollapsed, setSettingsCollapsed] = useState(false)
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
  const [runRowBusyIndex, setRunRowBusyIndex] = useState<number | null>(null)
  const [batchBusy, setBatchBusy] = useState(false)
  const [rowRunError, setRowRunError] = useState<string | null>(null)
  const [batchRunError, setBatchRunError] = useState<string | null>(null)
  const [processingOverlay, setProcessingOverlay] = useState<ProcessingOverlayState | null>(null)
  const [tokenAccounting, setTokenAccounting] = useState<TokenAccountingState>(() => loadTokenAccountingState())
  const lastAutoPreviewKeyRef = useRef<string | null>(null)
  const lastResolvedPreviewKeyRef = useRef<string | null>(null)

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
      if (transports.some((item) => item.name === current)) {
        return current
      }
      const active = transports.find((item) => item.active)
      return active?.name ?? 'mtls'
    })
  }, [statusQ.data])

  useEffect(() => {
    if (!settingsQ.data?.fields?.length) return
    setSettingValues(fieldsToValues(settingsQ.data.fields))
  }, [settingsQ.data])

  const probe = useMutation({
    mutationFn: (transport: GigaChatTransportName) => probeGigaChatTransport(transport),
  })

  const saveSettings = useMutation({
    mutationFn: () => saveGigaChatLabSettings(settingValues),
    onSuccess: async (data) => {
      setSettingValues(fieldsToValues(data.fields))
      await qc.invalidateQueries({ queryKey: ['gigachat-lab-settings'] })
      await qc.invalidateQueries({ queryKey: ['gigachat-status'] })
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
      const form = new FormData()
      form.append('file', selectedFile)
      return uploadGigaChatWorkbook(form)
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
          classification: row.classification,
          tags: row.tags,
          source_row: row.sourceRow,
        })),
      )
    },
    onSuccess: ({ blob, filename }) => {
      const sourceName = sheetData?.filename ?? 'annotated.xlsx'
      const fallbackName = `${sourceName.replace(/\.[^.]+$/u, '') || 'annotated'}_annotated.xlsx`
      downloadBlob(blob, filename || fallbackName)
    },
  })

  useEffect(() => {
    if (!settingsQ.data || settingsQ.isLoading) return
    if (lastAutoPreviewKeyRef.current === finalPromptRequestKey) return
    const timer = window.setTimeout(() => {
      lastAutoPreviewKeyRef.current = finalPromptRequestKey
      previewFinalPrompt.mutate({ requestKey: finalPromptRequestKey })
    }, 350)
    return () => window.clearTimeout(timer)
  }, [finalPromptRequestKey, settingsQ.data, settingsQ.isLoading])

  useEffect(() => {
    if (!sheetData) return
    if (!includedPromptColumns.length) {
      setIncludedPromptColumns([...sheetData.columns])
    }
  }, [sheetData, includedPromptColumns.length])

  useEffect(() => {
    if (!sheetData) {
      setSelectedSheetRowKeys([])
      return
    }
    const available = new Set(sheetData.rows.map((_, index) => buildSheetRowKey(sheetData.upload_id, sheetData.sheet_name, index)))
    setSelectedSheetRowKeys((current) => current.filter((key) => available.has(key)))
  }, [sheetData])

  const appendAnnotatedResult = (data: GigaChatLabRowRunResponse, rowIndex: number, row: Record<string, unknown>) => {
    if (!sheetData) return
    const rowKey = buildSheetRowKey(sheetData.upload_id, sheetData.sheet_name, rowIndex)
    const annotatedRow: AnnotatedSheetRow = {
      rowKey,
      rowIndex,
      classification: extractClassification(data.response_json),
      tags: extractTags(data.response_json),
      responseRaw: data.response_raw,
      responseJson: data.response_json,
      sourceRow: row,
    }
    setAnnotatedRows((current) => {
      const next = current.filter((item) => item.rowKey !== rowKey)
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

  const handleRunRow = async (row: Record<string, unknown>, rowIndex: number) => {
    setRowRunError(null)
    setRunRowBusyIndex(rowIndex)
    startProcessingOverlay('Обработка одной жалобы', 1)
    try {
      const data = await runGigaChatWorkbookRow(
        selectedTransport,
        settingValues,
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

  const handleRunSelectedRows = async () => {
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
        const data = await runGigaChatWorkbookRow(selectedTransport, settingValues, finalPromptColumns, row, payloadOverride, tokenAccounting.enabled)
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
  const requestSettingsFields = (settingsQ.data?.fields ?? []).filter((field) => !labelingFieldKeys.has(field.key))
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
  const filteredAnnotatedRows = useMemo(
    () => annotatedRows.filter((row) => {
      const classMatch = !annotatedClassFilter.length || annotatedClassFilter.includes(row.classification)
      const tagMatch = !annotatedTagFilter.length || row.tags.some((tag) => annotatedTagFilter.includes(tag))
      return classMatch && tagMatch
    }),
    [annotatedRows, annotatedClassFilter, annotatedTagFilter],
  )
  const totalTokenCost = useMemo(
    () => (tokenAccounting.totalTokens / 1000) * tokenAccounting.pricePer1k,
    [tokenAccounting.totalTokens, tokenAccounting.pricePer1k],
  )

  return <div className='transport-page'>
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
        <div className='transport-toggle'>
          <button className={selectedTransport === 'mtls' ? 'active' : ''} onClick={() => setSelectedTransport('mtls')}>mTLS</button>
          <button className={selectedTransport === 'token' ? 'active' : ''} onClick={() => setSelectedTransport('token')}>Token</button>
        </div>
        <div className='transport-summary'>
          <div><b>Configured mode:</b> <code>{statusQ.data?.configured_mode ?? '—'}</code></div>
          <div><b>Model:</b> <code>{statusQ.data?.model ?? '—'}</code></div>
          <div><b>Selected transport:</b> <code>{selected?.title ?? selectedTransport}</code></div>
        </div>
        <div className='transport-actions'>
          <button onClick={() => selected && probe.mutate(selected.name)} disabled={!selected || probe.isPending}>
            {probe.isPending ? 'Проверяем соединение...' : `Проверить ${selected?.title ?? ''}`}
          </button>
        </div>
      </> : null}
    </section>

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

    <section className='card transport-result'>
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

    <section className='card transport-result'>
      <div className='transport-section-head'>
        <div className='transport-section-title'>
          <h3>Настройки GigaChat</h3>
          <p>Параметры самого запроса: model, sampling и всё, что относится к system/user/context prompt перед отправкой в GigaChat.</p>
        </div>
        <button className='transport-collapse-button' onClick={() => setSettingsCollapsed((current) => !current)}>
          {settingsCollapsed ? 'Развернуть' : 'Свернуть'}
        </button>
      </div>

      {!settingsCollapsed ? <>
        {settingsQ.isLoading ? <div>Загружаем настройки...</div> : null}
        {settingsQ.isError ? <div className='transport-error'>{formatLabError(settingsQ.error as Error, 'настройки')}</div> : null}
        {settingsQ.data ? <>
          <div className='lab-settings-meta'>
            <div><b>Профиль:</b> <code>{settingsQ.data.title}</code></div>
            <div><b>Последнее сохранение:</b> {settingsQ.data.saved_at ? new Date(settingsQ.data.saved_at).toLocaleString() : 'еще не сохраняли'}</div>
          </div>
          <GigaChatSettingsForm
            fields={requestSettingsFields}
            values={settingValues}
            busy={saveSettings.isPending}
            saveError={saveSettings.isError ? formatLabError(saveSettings.error as Error, 'настройки') : null}
            onChange={(key, value) => setSettingValues((current) => ({ ...current, [key]: value }))}
            onSave={() => saveSettings.mutate()}
          />
        </> : null}
      </> : null}
    </section>

    <section className='card transport-result'>
      <div className='transport-section-head'>
        <div className='transport-section-title'>
          <h3>Классификация и разметка</h3>
          <p>Две отдельные области для правил классификации и правил тегирования, которые добавляются в prompt-контекст эксперимента.</p>
        </div>
      </div>

      <div className='labeling-grid'>
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
      </div>

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
          <p>Загрузите локальный Excel или CSV, затем выберите нужный лист и откройте его как рабочую таблицу прямо на странице.</p>
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
        </div>
      </div> : null}

      {!uploadCollapsed ? <>
        <div className='workbook-upload-row'>
          <input
            key={fileInputVersion}
            type='file'
            accept='.xlsx,.xls,.csv'
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

    <section className='card transport-result'>
      <div className='transport-section-head'>
        <div className='transport-section-title'>
          <h3>Рабочая таблица</h3>
          <p>Ниже отображаются колонки выбранного листа, строки для проверки структуры и быстрые действия по отправке одной записи в GigaChat.</p>
        </div>
        <button className='transport-collapse-button' onClick={() => setWorkbookCollapsed((current) => !current)}>
          {workbookCollapsed ? 'Развернуть' : 'Свернуть'}
        </button>
      </div>

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
          onRunRow={handleRunRow}
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
          onRunSelectedRows={handleRunSelectedRows}
          batchBusy={batchBusy}
          busy={batchBusy || runRowBusyIndex !== null}
        /> : null}
        {rowRunError ? <div className='transport-error'>{rowRunError}</div> : null}
        {batchRunError ? <div className='transport-error'>{batchRunError}</div> : null}
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
            disabled={exportAnnotatedRows.isPending}
          >
            {exportAnnotatedRows.isPending ? 'Выгружаем Excel...' : 'Выгрузить в Excel'}
          </button>
        </div> : null}
      </div>

      {!annotatedRows.length ? <p>Пока здесь пусто. Отправьте одну строку или выбранные строки в GigaChat, и результаты появятся в этой таблице.</p> : <>
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
          <div className='transport-actions'>
            <button type='button' onClick={() => {
              setAnnotatedClassFilter([])
              setAnnotatedTagFilter([])
            }}>Сбросить фильтры</button>
          </div>
        </div>

        <div className='workbook-table-wrap'>
          <table className='table workbook-table'>
            <thead>
              <tr>
                <th>Класс</th>
                <th>Теги</th>
                {sheetData?.columns.map((column) => <th key={`annotated-head-${column}`}>{column}</th>)}
              </tr>
            </thead>
            <tbody>
              {filteredAnnotatedRows.map((row) => <tr key={row.rowKey}>
                <td>{row.classification || '—'}</td>
                <td>{row.tags.length ? row.tags.join(', ') : '—'}</td>
                {sheetData?.columns.map((column) => <td key={`${row.rowKey}-${column}`}>{String(row.sourceRow[column] ?? '')}</td>)}
              </tr>)}
            </tbody>
          </table>
        </div>
        {exportAnnotatedRows.isError ? <div className='transport-error'>{formatLabError(exportAnnotatedRows.error as Error, 'размеченную таблицу')}</div> : null}
      </>}
    </section>

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
