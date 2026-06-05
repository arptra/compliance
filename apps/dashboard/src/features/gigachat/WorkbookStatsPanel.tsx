import { useEffect, useMemo, useState } from 'react'
import type * as echarts from 'echarts'
import { EChart } from '../../components/EChart'
import type { GigaChatRuleEvaluationRow, GigaChatWorkbookSheetDataResponse } from './types'

type TopLimit = 8 | 12 | 20 | 'all'
type DateBucket = 'day' | 'week' | 'month'
type DimensionDataType = 'category' | 'date' | 'number'

type DimensionOption = {
  key: string
  label: string
  dataType: DimensionDataType
  column?: string
  listLike?: boolean
}

type DistributionItem = {
  label: string
  count: number
}

type StatsChartKey = 'pie' | 'histogram'

type StatRow = {
  row: Record<string, unknown>
  rowIndex: number
  hitCount: number
  ruleTags: string[]
  ruleCodes: string[]
  ruleKeywords: string[]
  suggestedTopics: string[]
}

export type WorkbookStatsPanelProps = {
  data: GigaChatWorkbookSheetDataResponse | null
  ruleEvaluations: Record<number, GigaChatRuleEvaluationRow>
}

const EMPTY_LABEL = 'Без значения'
const NO_RULE_LABEL = 'Без rule hits'
const TOP_LIMITS: TopLimit[] = [8, 12, 20, 'all']
const DATE_BUCKETS: Array<{ value: DateBucket; label: string }> = [
  { value: 'day', label: 'Дни' },
  { value: 'week', label: 'Недели' },
  { value: 'month', label: 'Месяцы' },
]
const AXIS_LABEL_LINE_LENGTH = 14
const AXIS_LABEL_MAX_LINES = 3

function stringifyValue(value: unknown): string {
  if (value === null || value === undefined) return ''
  if (Array.isArray(value)) return value.map((item) => stringifyValue(item)).filter(Boolean).join(', ')
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value).trim()
}

function unique(values: string[]) {
  return Array.from(new Set(values.map((value) => value.trim()).filter(Boolean)))
}

function parseNumberValue(value: unknown) {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null
  const text = stringifyValue(value).replace(/\s+/g, '').replace(',', '.')
  if (!text) return null
  const number = Number(text)
  return Number.isFinite(number) ? number : null
}

function parseDateValue(value: unknown) {
  const text = stringifyValue(value)
  if (!text) return null
  const ruDate = text.match(/^(\d{1,2})\.(\d{1,2})\.(\d{4})(?:\s+.*)?$/u)
  if (ruDate) {
    const date = new Date(Number(ruDate[3]), Number(ruDate[2]) - 1, Number(ruDate[1]))
    return Number.isNaN(date.getTime()) ? null : date
  }
  const isoDate = text.match(/^(\d{4})-(\d{1,2})-(\d{1,2})(?:[T\s].*)?$/u)
  if (isoDate) {
    const date = new Date(Number(isoDate[1]), Number(isoDate[2]) - 1, Number(isoDate[3]))
    return Number.isNaN(date.getTime()) ? null : date
  }
  const timestamp = Date.parse(text)
  return Number.isNaN(timestamp) ? null : new Date(timestamp)
}

function formatDateKey(date: Date, bucket: DateBucket) {
  const normalized = new Date(date.getFullYear(), date.getMonth(), date.getDate())
  if (bucket === 'month') {
    return `${normalized.getFullYear()}-${String(normalized.getMonth() + 1).padStart(2, '0')}`
  }
  if (bucket === 'week') {
    const day = normalized.getDay() || 7
    normalized.setDate(normalized.getDate() - day + 1)
    return `${normalized.getFullYear()}-${String(normalized.getMonth() + 1).padStart(2, '0')}-${String(normalized.getDate()).padStart(2, '0')}`
  }
  return `${normalized.getFullYear()}-${String(normalized.getMonth() + 1).padStart(2, '0')}-${String(normalized.getDate()).padStart(2, '0')}`
}

function splitListValue(value: unknown, listLike: boolean) {
  const text = stringifyValue(value)
  if (!text) return []
  if (!listLike) return [text]
  return unique(text.split(/[,;|\n]+/u))
}

function looksListLike(column: string) {
  return /tag|тег|rule|правил|keyword|ключ|topic|топик|класс|катег|categor|label|метк/i.test(column)
}

function inferColumnType(rows: Array<Record<string, unknown>>, column: string): DimensionDataType {
  const values = rows.map((row) => row[column]).map(stringifyValue).filter(Boolean).slice(0, 80)
  if (!values.length) return 'category'
  const dateScore = values.filter((value) => parseDateValue(value)).length / values.length
  if (dateScore >= 0.7 || (/date|дата|created|updated|published|time|timestamp/i.test(column) && dateScore >= 0.45)) {
    return 'date'
  }
  const numberScore = values.filter((value) => parseNumberValue(value) !== null).length / values.length
  if (numberScore >= 0.8) return 'number'
  return 'category'
}

function formatNumber(value: number) {
  return new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 2 }).format(value)
}

function wrapChartLabel(value: string, lineLength = AXIS_LABEL_LINE_LENGTH, maxLines = AXIS_LABEL_MAX_LINES) {
  const text = value.trim()
  if (text.length <= lineLength) return text

  const words = text.split(/\s+/u)
  const lines: string[] = []
  let currentLine = ''

  words.forEach((word) => {
    const nextLine = currentLine ? `${currentLine} ${word}` : word
    if (nextLine.length <= lineLength) {
      currentLine = nextLine
      return
    }
    if (currentLine) lines.push(currentLine)
    if (word.length <= lineLength) {
      currentLine = word
      return
    }
    const chunks = word.match(new RegExp(`.{1,${lineLength}}`, 'gu')) ?? [word]
    lines.push(...chunks.slice(0, -1))
    currentLine = chunks[chunks.length - 1] ?? ''
  })

  if (currentLine) lines.push(currentLine)
  if (lines.length <= maxLines) return lines.join('\n')

  const visibleLines = lines.slice(0, maxLines)
  visibleLines[maxLines - 1] = `${visibleLines[maxLines - 1].slice(0, Math.max(1, lineLength - 1))}…`
  return visibleLines.join('\n')
}

function buildNumericBins(values: number[], limit: TopLimit): DistributionItem[] {
  const clean = values.filter(Number.isFinite)
  if (!clean.length) return []
  const min = Math.min(...clean)
  const max = Math.max(...clean)
  if (min === max) return [{ label: formatNumber(min), count: clean.length }]
  const requestedBins = limit === 'all' ? 14 : Number(limit)
  const binCount = Math.max(4, Math.min(18, requestedBins, Math.ceil(Math.sqrt(clean.length)) + 2))
  const width = (max - min) / binCount
  const bins = Array.from({ length: binCount }, (_, index) => {
    const start = min + width * index
    const end = index === binCount - 1 ? max : min + width * (index + 1)
    return {
      label: `${formatNumber(start)}-${formatNumber(end)}`,
      count: 0,
    }
  })
  clean.forEach((value) => {
    const index = Math.min(binCount - 1, Math.max(0, Math.floor((value - min) / width)))
    bins[index].count += 1
  })
  return bins
}

function applyTopLimit(items: DistributionItem[], limit: TopLimit) {
  const sorted = [...items].sort((a, b) => b.count - a.count || a.label.localeCompare(b.label, 'ru'))
  if (limit === 'all' || sorted.length <= limit) return sorted
  const top = sorted.slice(0, limit)
  const rest = sorted.slice(limit).reduce((sum, item) => sum + item.count, 0)
  return rest ? [...top, { label: 'Другие', count: rest }] : top
}

function emptyChartOption(title: string): echarts.EChartsCoreOption {
  return {
    title: {
      text: title,
      left: 'center',
      top: 'middle',
      textStyle: { color: '#64748b', fontSize: 14, fontWeight: 600 },
    },
  }
}

export function WorkbookStatsPanel({ data, ruleEvaluations }: WorkbookStatsPanelProps) {
  const [collapsed, setCollapsed] = useState(true)
  const [dimensionKey, setDimensionKey] = useState('__rule_tag')
  const [topLimit, setTopLimit] = useState<TopLimit>(12)
  const [dateBucket, setDateBucket] = useState<DateBucket>('day')
  const [includeEmpty, setIncludeEmpty] = useState(true)
  const [expandedChart, setExpandedChart] = useState<StatsChartKey | null>(null)

  useEffect(() => {
    if (!expandedChart) return undefined

    const previousOverflow = document.body.style.overflow
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setExpandedChart(null)
    }
    document.body.style.overflow = 'hidden'
    window.addEventListener('keydown', closeOnEscape)

    return () => {
      document.body.style.overflow = previousOverflow
      window.removeEventListener('keydown', closeOnEscape)
    }
  }, [expandedChart])

  const rows = data?.rows ?? []
  const statRows = useMemo<StatRow[]>(() => rows.map((row, rowIndex) => {
    const evaluation = ruleEvaluations[rowIndex]
    const hits = evaluation?.hits ?? []
    const ruleTags = unique([
      ...(evaluation?.suggested_tags ?? []),
      ...hits.flatMap((hit) => hit.target_tag ? [hit.target_tag] : []),
    ])
    const suggestedTopics = unique([
      ...(evaluation?.suggested_topics ?? []),
      ...hits.flatMap((hit) => hit.target_topic ? [hit.target_topic] : []),
    ])
    return {
      row,
      rowIndex,
      hitCount: hits.length,
      ruleTags,
      ruleCodes: unique(hits.map((hit) => hit.code)),
      ruleKeywords: unique(hits.flatMap((hit) => hit.matched_keywords ?? [])),
      suggestedTopics,
    }
  }), [rows, ruleEvaluations])

  const dimensionOptions = useMemo<DimensionOption[]>(() => {
    const columnOptions = (data?.columns ?? []).map((column) => ({
      key: `col:${column}`,
      label: column,
      column,
      dataType: inferColumnType(rows, column),
      listLike: looksListLike(column),
    }))
    return [
      { key: '__rule_tag', label: 'Rule tags', dataType: 'category', listLike: true },
      { key: '__rule_keyword', label: 'Ключевые слова', dataType: 'category', listLike: true },
      { key: '__rule_code', label: 'Rule codes', dataType: 'category', listLike: true },
      { key: '__rule_topic', label: 'Suggested topics', dataType: 'category', listLike: true },
      { key: '__rule_presence', label: 'Наличие rule hits', dataType: 'category' },
      ...columnOptions,
    ]
  }, [data?.columns, rows])

  useEffect(() => {
    if (!dimensionOptions.some((option) => option.key === dimensionKey)) {
      setDimensionKey(dimensionOptions[0]?.key ?? '__rule_tag')
    }
  }, [dimensionKey, dimensionOptions])

  const selectedDimension = dimensionOptions.find((option) => option.key === dimensionKey) ?? dimensionOptions[0]

  const getDimensionValues = (item: StatRow) => {
    if (!selectedDimension) return []
    if (selectedDimension.key === '__rule_tag') return item.ruleTags.length ? item.ruleTags : [NO_RULE_LABEL]
    if (selectedDimension.key === '__rule_keyword') return item.ruleKeywords.length ? item.ruleKeywords : [NO_RULE_LABEL]
    if (selectedDimension.key === '__rule_code') return item.ruleCodes.length ? item.ruleCodes : [NO_RULE_LABEL]
    if (selectedDimension.key === '__rule_topic') return item.suggestedTopics.length ? item.suggestedTopics : [NO_RULE_LABEL]
    if (selectedDimension.key === '__rule_presence') return [item.hitCount ? 'Есть rule hits' : NO_RULE_LABEL]
    if (!selectedDimension.column) return []
    const raw = item.row[selectedDimension.column]
    if (selectedDimension.dataType === 'date') {
      const date = parseDateValue(raw)
      return date ? [formatDateKey(date, dateBucket)] : []
    }
    return splitListValue(raw, Boolean(selectedDimension.listLike))
  }

  const distribution = useMemo(() => {
    const counts = new Map<string, number>()
    statRows.forEach((item) => {
      const values = getDimensionValues(item).filter((value) => includeEmpty || (value && value !== EMPTY_LABEL && value !== NO_RULE_LABEL))
      const normalized = values.length ? values : (includeEmpty ? [EMPTY_LABEL] : [])
      normalized.forEach((value) => counts.set(value, (counts.get(value) ?? 0) + 1))
    })
    return Array.from(counts.entries()).map(([label, count]) => ({ label, count }))
  }, [dateBucket, dimensionKey, includeEmpty, statRows])

  const visibleDistribution = useMemo(() => applyTopLimit(distribution, topLimit), [distribution, topLimit])

  const numericDimensionValues = useMemo(() => {
    if (!selectedDimension?.column || selectedDimension.dataType !== 'number') return []
    return statRows.flatMap((item) => {
      const value = parseNumberValue(item.row[selectedDimension.column ?? ''])
      return value === null ? [] : [value]
    })
  }, [selectedDimension, statRows])

  const histogramData = selectedDimension?.dataType === 'number'
    ? buildNumericBins(numericDimensionValues, topLimit)
    : selectedDimension?.dataType === 'date'
      ? [...visibleDistribution].sort((a, b) => a.label.localeCompare(b.label))
      : visibleDistribution

  const rowsWithHits = statRows.filter((item) => item.hitCount > 0).length
  const uniqueTags = new Set(statRows.flatMap((item) => item.ruleTags)).size
  const topSegment = visibleDistribution[0]
  const averageHits = statRows.length ? statRows.reduce((sum, item) => sum + item.hitCount, 0) / statRows.length : 0

  const pieOption = useMemo<echarts.EChartsCoreOption>(() => {
    if (!visibleDistribution.length) return emptyChartOption('Нет данных')
    return {
      tooltip: { trigger: 'item' },
      legend: {
        bottom: 0,
        type: 'scroll',
        formatter: (name: string) => wrapChartLabel(name, 18, 2),
      },
      series: [{
        name: selectedDimension?.label ?? 'Разрез',
        type: 'pie',
        radius: ['38%', '64%'],
        center: ['50%', '42%'],
        avoidLabelOverlap: true,
        itemStyle: { borderColor: '#fff', borderWidth: 2 },
        label: {
          formatter: (params: { name?: string; percent?: number }) => `${wrapChartLabel(params.name ?? '', 16, 2)}\n${params.percent ?? 0}%`,
          lineHeight: 14,
          width: 110,
          overflow: 'break',
        },
        data: visibleDistribution.map((item) => ({ name: item.label, value: item.count })),
      }],
    }
  }, [selectedDimension?.label, visibleDistribution])

  const histogramOption = useMemo<echarts.EChartsCoreOption>(() => {
    if (!histogramData.length) return emptyChartOption('Нет данных')
    return {
      tooltip: { trigger: 'axis' },
      grid: { left: 8, right: 12, top: 24, bottom: 8, containLabel: true },
      xAxis: {
        type: 'category',
        data: histogramData.map((item) => item.label),
        axisLabel: {
          interval: 0,
          formatter: (value: string) => wrapChartLabel(value),
          lineHeight: 14,
          margin: 12,
        },
      },
      yAxis: { type: 'value', minInterval: 1 },
      series: [{
        name: 'Строк',
        type: 'bar',
        data: histogramData.map((item) => item.count),
        itemStyle: { color: '#2563eb', borderRadius: [5, 5, 0, 0] },
        barMaxWidth: 42,
      }],
    }
  }, [histogramData])

  const chartCards: Array<{ key: StatsChartKey; title: string; meta: string; option: echarts.EChartsCoreOption }> = [
    {
      key: 'pie',
      title: 'Доли',
      meta: topSegment ? `${topSegment.label}: ${topSegment.count}` : 'нет данных',
      option: pieOption,
    },
    {
      key: 'histogram',
      title: 'Гистограмма',
      meta: selectedDimension?.label ?? '',
      option: histogramOption,
    },
  ]
  const expandedChartCard = chartCards.find((chart) => chart.key === expandedChart) ?? null

  return <section className='card transport-result workbook-stats-card'>
    <div className='transport-section-head'>
      <div className='transport-section-title'>
        <h3>Статистика rule hits</h3>
        <p>Распределения по тегам, ключевым словам и колонкам рабочей тетради.</p>
      </div>
      <button className='transport-collapse-button' onClick={() => setCollapsed((current) => !current)}>
        {collapsed ? 'Развернуть' : 'Свернуть'}
      </button>
    </div>

    {!collapsed && (!data ? <p className='lab-muted'>Загрузите рабочий лист, чтобы увидеть статистику.</p> : <>
      <div className='workbook-stats-controls'>
        <label className='workbook-stats-field'>
          <span>Разрез</span>
          <select value={dimensionKey} onChange={(event) => setDimensionKey(event.target.value)}>
            {dimensionOptions.map((option) => <option key={option.key} value={option.key}>{option.label}</option>)}
          </select>
        </label>
        <div className='workbook-stats-setting'>
          <span>Top</span>
          <div className='workbook-stats-segment' role='group' aria-label='Top segments'>
            {TOP_LIMITS.map((limit) => <button
              key={String(limit)}
              type='button'
              className={topLimit === limit ? 'active' : ''}
              onClick={() => setTopLimit(limit)}
            >
              {limit === 'all' ? 'Все' : limit}
            </button>)}
          </div>
        </div>
        {selectedDimension?.dataType === 'date' ? <div className='workbook-stats-setting'>
          <span>Даты</span>
          <div className='workbook-stats-segment' role='group' aria-label='Date bucket'>
            {DATE_BUCKETS.map((bucket) => <button
              key={bucket.value}
              type='button'
              className={dateBucket === bucket.value ? 'active' : ''}
              onClick={() => setDateBucket(bucket.value)}
            >
              {bucket.label}
            </button>)}
          </div>
        </div> : null}
        <label className='lab-checkbox workbook-stats-checkbox'>
          <input type='checkbox' checked={includeEmpty} onChange={(event) => setIncludeEmpty(event.target.checked)} />
          <span>Показывать пустые</span>
        </label>
      </div>

      <div className='workbook-stats-kpi-grid'>
        <div className='workbook-stats-kpi'><span>Строк</span><strong>{data.rows.length}</strong><small>{data.sheet_name}</small></div>
        <div className='workbook-stats-kpi'><span>С rule hits</span><strong>{rowsWithHits}</strong><small>{data.rows.length ? `${Math.round((rowsWithHits / data.rows.length) * 100)}%` : '0%'}</small></div>
        <div className='workbook-stats-kpi'><span>Rule tags</span><strong>{uniqueTags}</strong><small>уникальных тегов</small></div>
        <div className='workbook-stats-kpi'><span>Среднее hits</span><strong>{formatNumber(averageHits)}</strong><small>на строку</small></div>
      </div>

      <div className='workbook-stats-chart-grid'>
        {chartCards.map((chart) => <article
          key={chart.key}
          className='workbook-stats-panel interactive'
          role='button'
          tabIndex={0}
          aria-label={`${chart.title}: открыть на весь экран`}
          onClick={() => setExpandedChart(chart.key)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' || event.key === ' ') {
              event.preventDefault()
              setExpandedChart(chart.key)
            }
          }}
        >
          <div className='workbook-stats-panel-head'>
            <h4>{chart.title}</h4>
            <span>{chart.meta}</span>
          </div>
          <EChart option={chart.option} height={330} />
        </article>)}
      </div>

      <div className='workbook-stats-strip'>
        {visibleDistribution.slice(0, 6).map((item) => <div key={item.label} className='workbook-stats-strip-item'>
          <span>{item.label}</span>
          <strong>{item.count}</strong>
        </div>)}
      </div>

      {expandedChartCard ? <div className='workbook-chart-modal-backdrop' onClick={() => setExpandedChart(null)}>
        <section className='workbook-chart-modal' onClick={(event) => event.stopPropagation()}>
          <div className='workbook-chart-modal-head'>
            <div>
              <h3>{expandedChartCard.title}</h3>
              <p>{expandedChartCard.meta}</p>
            </div>
            <button type='button' className='transport-collapse-button' onClick={() => setExpandedChart(null)}>Закрыть</button>
          </div>
          <div className='workbook-chart-modal-body'>
            <EChart option={expandedChartCard.option} height='100%' />
          </div>
        </section>
      </div> : null}
    </>)}
  </section>
}
