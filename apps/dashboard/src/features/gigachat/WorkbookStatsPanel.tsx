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

type MetricOption = {
  key: string
  label: string
  column?: string
}

type DistributionItem = {
  label: string
  count: number
}

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

function quantile(sorted: number[], q: number) {
  if (!sorted.length) return 0
  const position = (sorted.length - 1) * q
  const base = Math.floor(position)
  const rest = position - base
  const next = sorted[base + 1]
  return next === undefined ? sorted[base] : sorted[base] + rest * (next - sorted[base])
}

function boxStats(values: number[]) {
  const sorted = [...values].filter(Number.isFinite).sort((a, b) => a - b)
  if (!sorted.length) return null
  return [
    sorted[0],
    quantile(sorted, 0.25),
    quantile(sorted, 0.5),
    quantile(sorted, 0.75),
    sorted[sorted.length - 1],
  ]
}

function formatNumber(value: number) {
  return new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 2 }).format(value)
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
  const [dimensionKey, setDimensionKey] = useState('__rule_tag')
  const [metricKey, setMetricKey] = useState('__hit_count')
  const [topLimit, setTopLimit] = useState<TopLimit>(12)
  const [dateBucket, setDateBucket] = useState<DateBucket>('day')
  const [includeEmpty, setIncludeEmpty] = useState(true)

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

  const metricOptions = useMemo<MetricOption[]>(() => [
    { key: '__hit_count', label: 'Rule hits на строку' },
    ...(data?.columns ?? [])
      .filter((column) => inferColumnType(rows, column) === 'number')
      .map((column) => ({ key: `num:${column}`, label: column, column })),
  ], [data?.columns, rows])

  useEffect(() => {
    if (!dimensionOptions.some((option) => option.key === dimensionKey)) {
      setDimensionKey(dimensionOptions[0]?.key ?? '__rule_tag')
    }
  }, [dimensionKey, dimensionOptions])

  useEffect(() => {
    if (!metricOptions.some((option) => option.key === metricKey)) {
      setMetricKey(metricOptions[0]?.key ?? '__hit_count')
    }
  }, [metricKey, metricOptions])

  const selectedDimension = dimensionOptions.find((option) => option.key === dimensionKey) ?? dimensionOptions[0]
  const selectedMetric = metricOptions.find((option) => option.key === metricKey) ?? metricOptions[0]

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

  const getMetricValue = (item: StatRow) => {
    if (!selectedMetric) return null
    if (selectedMetric.key === '__hit_count') return item.hitCount
    if (!selectedMetric.column) return null
    return parseNumberValue(item.row[selectedMetric.column])
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

  const boxPlotGroups = useMemo(() => {
    const topLabels = new Set(visibleDistribution.filter((item) => item.label !== 'Другие').map((item) => item.label))
    const valuesByLabel = new Map<string, number[]>()
    statRows.forEach((item) => {
      const metric = getMetricValue(item)
      if (metric === null) return
      const values = getDimensionValues(item)
      const normalized = values.length ? values : (includeEmpty ? [EMPTY_LABEL] : [])
      normalized.forEach((label) => {
        if (!topLabels.has(label)) return
        const list = valuesByLabel.get(label) ?? []
        list.push(metric)
        valuesByLabel.set(label, list)
      })
    })
    return Array.from(valuesByLabel.entries())
      .map(([label, values]) => ({ label, values, stats: boxStats(values) }))
      .filter((item): item is { label: string; values: number[]; stats: number[] } => Boolean(item.stats))
  }, [dateBucket, dimensionKey, includeEmpty, metricKey, statRows, visibleDistribution])

  const rowsWithHits = statRows.filter((item) => item.hitCount > 0).length
  const uniqueTags = new Set(statRows.flatMap((item) => item.ruleTags)).size
  const topSegment = visibleDistribution[0]
  const averageHits = statRows.length ? statRows.reduce((sum, item) => sum + item.hitCount, 0) / statRows.length : 0

  const pieOption = useMemo<echarts.EChartsCoreOption>(() => {
    if (!visibleDistribution.length) return emptyChartOption('Нет данных')
    return {
      tooltip: { trigger: 'item' },
      legend: { bottom: 0, type: 'scroll' },
      series: [{
        name: selectedDimension?.label ?? 'Разрез',
        type: 'pie',
        radius: ['42%', '70%'],
        center: ['50%', '43%'],
        avoidLabelOverlap: true,
        itemStyle: { borderColor: '#fff', borderWidth: 2 },
        label: { formatter: '{b}: {d}%' },
        data: visibleDistribution.map((item) => ({ name: item.label, value: item.count })),
      }],
    }
  }, [selectedDimension?.label, visibleDistribution])

  const histogramOption = useMemo<echarts.EChartsCoreOption>(() => {
    if (!histogramData.length) return emptyChartOption('Нет данных')
    return {
      tooltip: { trigger: 'axis' },
      grid: { left: 44, right: 16, top: 24, bottom: 76 },
      xAxis: {
        type: 'category',
        data: histogramData.map((item) => item.label),
        axisLabel: { interval: 0, rotate: histogramData.some((item) => item.label.length > 12) ? 30 : 0 },
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

  const boxPlotOption = useMemo<echarts.EChartsCoreOption>(() => {
    if (!boxPlotGroups.length) return emptyChartOption('Нет числовой метрики')
    return {
      tooltip: {
        trigger: 'item',
        formatter: (params: unknown) => {
          const data = (params as { data?: number[]; name?: string }).data ?? []
          return [
            `<b>${(params as { name?: string }).name ?? ''}</b>`,
            `min: ${formatNumber(data[0] ?? 0)}`,
            `Q1: ${formatNumber(data[1] ?? 0)}`,
            `median: ${formatNumber(data[2] ?? 0)}`,
            `Q3: ${formatNumber(data[3] ?? 0)}`,
            `max: ${formatNumber(data[4] ?? 0)}`,
          ].join('<br/>')
        },
      },
      grid: { left: 50, right: 16, top: 24, bottom: 78 },
      xAxis: {
        type: 'category',
        data: boxPlotGroups.map((item) => item.label),
        axisLabel: { interval: 0, rotate: boxPlotGroups.some((item) => item.label.length > 12) ? 30 : 0 },
      },
      yAxis: { type: 'value', scale: true, name: selectedMetric?.label ?? '' },
      series: [{
        name: selectedMetric?.label ?? 'Метрика',
        type: 'boxplot',
        data: boxPlotGroups.map((item) => item.stats),
        itemStyle: { color: '#e0f2fe', borderColor: '#0284c7' },
      }],
    }
  }, [boxPlotGroups, selectedMetric?.label])

  return <section className='card transport-result workbook-stats-card'>
    <div className='transport-section-head'>
      <div className='transport-section-title'>
        <h3>Статистика rule hits</h3>
        <p>Распределения по тегам, ключевым словам и колонкам рабочей тетради.</p>
      </div>
    </div>

    {!data ? <p className='lab-muted'>Загрузите рабочий лист, чтобы увидеть статистику.</p> : <>
      <div className='workbook-stats-controls'>
        <label className='workbook-stats-field'>
          <span>Разрез</span>
          <select value={dimensionKey} onChange={(event) => setDimensionKey(event.target.value)}>
            {dimensionOptions.map((option) => <option key={option.key} value={option.key}>{option.label}</option>)}
          </select>
        </label>
        <label className='workbook-stats-field'>
          <span>Метрика для ящика</span>
          <select value={metricKey} onChange={(event) => setMetricKey(event.target.value)}>
            {metricOptions.map((option) => <option key={option.key} value={option.key}>{option.label}</option>)}
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
        <article className='workbook-stats-panel'>
          <div className='workbook-stats-panel-head'>
            <h4>Доли</h4>
            <span>{topSegment ? `${topSegment.label}: ${topSegment.count}` : 'нет данных'}</span>
          </div>
          <EChart option={pieOption} height={310} />
        </article>
        <article className='workbook-stats-panel'>
          <div className='workbook-stats-panel-head'>
            <h4>Гистограмма</h4>
            <span>{selectedDimension?.label ?? ''}</span>
          </div>
          <EChart option={histogramOption} height={310} />
        </article>
        <article className='workbook-stats-panel'>
          <div className='workbook-stats-panel-head'>
            <h4>Ящик с усами</h4>
            <span>{selectedMetric?.label ?? ''}</span>
          </div>
          <EChart option={boxPlotOption} height={310} />
        </article>
      </div>

      <div className='workbook-stats-strip'>
        {visibleDistribution.slice(0, 6).map((item) => <div key={item.label} className='workbook-stats-strip-item'>
          <span>{item.label}</span>
          <strong>{item.count}</strong>
        </div>)}
      </div>
    </>}
  </section>
}
