import { useMemo, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { apiPost } from '../lib/api'
import { executiveReportSchema, type ExecutiveReportResponse } from '../types/api'
import { ExecutiveKpiCards } from '../components/reports/ExecutiveKpiCards'
import { ActualExpectedPanel } from '../components/reports/ActualExpectedPanel'
import { ContributionPanel } from '../components/reports/ContributionPanel'
import { CategoryPriorityTable } from '../components/reports/CategoryPriorityTable'
import { AlertExamplesPanel } from '../components/reports/AlertExamplesPanel'
import { ExecutiveSummaryPanel } from '../components/reports/ExecutiveSummaryPanel'

const today = new Date().toISOString().slice(0, 10)

export default function ReportsPage() {
  const [date_from, setDateFrom] = useState(today)
  const [date_to, setDateTo] = useState(today)
  const [compare_mode, setCompareMode] = useState<'previous_period' | 'same_weekday' | 'seasonal' | 'custom_range'>('previous_period')
  const [baseline_date_from, setBaselineDateFrom] = useState('')
  const [baseline_date_to, setBaselineDateTo] = useState('')
  const [categoryScope, setCategoryScope] = useState<'all_categories' | 'selected_categories'>('all_categories')
  const [categories, setCategories] = useState('')
  const [include_examples, setIncludeExamples] = useState(true)
  const [include_ownership, setIncludeOwnership] = useState(true)

  const report = useMutation({
    mutationFn: async () => {
      const payload = await apiPost<ExecutiveReportResponse>('/api/reports/executive', {
        date_from,
        date_to,
        compare_mode,
        baseline_date_from: compare_mode === 'custom_range' ? baseline_date_from || undefined : undefined,
        baseline_date_to: compare_mode === 'custom_range' ? baseline_date_to || undefined : undefined,
        categories: categoryScope === 'selected_categories' ? categories.split(',').map((x) => x.trim()).filter(Boolean) : undefined,
        include_examples,
        include_ownership,
      })
      return executiveReportSchema.parse(payload)
    },
  })

  const data = report.data
  const generatedAt = useMemo(() => data?.meta.generated_at as string | undefined, [data])

  const reset = () => {
    setDateFrom(today)
    setDateTo(today)
    setCompareMode('previous_period')
    setBaselineDateFrom('')
    setBaselineDateTo('')
    setCategoryScope('all_categories')
    setCategories('')
    setIncludeExamples(true)
    setIncludeOwnership(true)
  }

  const copySummary = async () => {
    if (!data) return
    const text = [data.summary.headline, ...data.summary.bullets, 'Рекомендуемые действия:', ...data.summary.recommended_actions].join('\n')
    await navigator.clipboard.writeText(text)
  }

  const download = (content: string, filename: string, type: string) => {
    const blob = new Blob([content], { type })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }

  return <div>
    <div className='card'>
      <h2>Executive report</h2>
      <div>Ключевые метрики и зоны внимания</div>
      <small>Last updated: {generatedAt ?? '—'}</small>
    </div>

    <div className='card' style={{ marginTop: 12 }}>
      <div className='filters'>
        <label>Period from <input type='date' value={date_from} onChange={(e) => setDateFrom(e.target.value)} /></label>
        <label>to <input type='date' value={date_to} onChange={(e) => setDateTo(e.target.value)} /></label>
        <label>Compare mode <select value={compare_mode} onChange={(e) => setCompareMode(e.target.value as never)}><option value='previous_period'>previous_period</option><option value='same_weekday'>same_weekday</option><option value='seasonal'>seasonal</option><option value='custom_range'>custom_range</option></select></label>
        {compare_mode === 'custom_range' && <>
          <label>Baseline from <input type='date' value={baseline_date_from} onChange={(e) => setBaselineDateFrom(e.target.value)} /></label>
          <label>to <input type='date' value={baseline_date_to} onChange={(e) => setBaselineDateTo(e.target.value)} /></label>
        </>}
        <label>Category scope <select value={categoryScope} onChange={(e) => setCategoryScope(e.target.value as never)}><option value='all_categories'>all categories</option><option value='selected_categories'>selected categories</option></select></label>
        {categoryScope === 'selected_categories' && <label>Categories <input value={categories} onChange={(e) => setCategories(e.target.value)} placeholder='A, B, C' /></label>}
        <label><input type='checkbox' checked={include_examples} onChange={(e) => setIncludeExamples(e.target.checked)} /> include examples</label>
        <label><input type='checkbox' checked={include_ownership} onChange={(e) => setIncludeOwnership(e.target.checked)} /> include ownership / area summary</label>
        <button onClick={() => report.mutate()} disabled={report.isPending}>Build report</button>
        <button onClick={reset}>Reset</button>
      </div>
    </div>

    {report.isError && <div className='card' style={{ marginTop: 12 }}>Ошибка загрузки отчета</div>}

    {data && <>
      <div className='card-grid' style={{ marginTop: 12 }}>
        <div className='card'><div>Total complaints</div><strong>{data.kpis.total_complaints}</strong></div>
        <div className='card'><div>Delta</div><strong>{data.kpis.delta_abs >= 0 ? '+' : ''}{data.kpis.delta_abs}</strong></div>
        <div className='card'><div>Alert rows</div><strong>{data.meta.alert_rows ?? 0}</strong></div>
        <div className='card'>
          <div>Pattern Risk <span className='term-help' title='Индикатор того, насколько вероятно, что в текущих жалобах сохраняется ранее выявленный нетипичный проблемный сценарий.'>ⓘ</span></div>
          <strong><span className={`badge ${data.kpis.pattern_risk.status === 'danger' ? 'high' : data.kpis.pattern_risk.status === 'warning' ? 'medium' : data.kpis.pattern_risk.status === 'success' ? 'low' : 'neutral'}`}>{data.kpis.pattern_risk.display_label}</span></strong>
          <div>{data.kpis.pattern_risk.score == null ? 'Недостаточно данных pattern monitoring' : `score ${data.kpis.pattern_risk.score.toFixed(2)}`}</div>
        </div>
      </div>
      <div style={{ marginTop: 12 }}><ExecutiveKpiCards kpis={data.kpis} /></div>
      <div className='executive-grid' style={{ marginTop: 12 }}>
        <ActualExpectedPanel rows={data.charts.actual_expected} />
        <ContributionPanel rows={data.charts.category_contribution} />
        <CategoryPriorityTable rows={data.charts.category_priority} />
        <AlertExamplesPanel rows={include_examples ? data.charts.alert_examples : []} />
      </div>
      <div style={{ marginTop: 12 }}><ExecutiveSummaryPanel summary={data.summary} /></div>
      <div className='card' style={{ marginTop: 12 }}>
        <button onClick={copySummary}>Copy summary</button>{' '}
        <button onClick={() => download(data.export.html, 'executive-report.html', 'text/html')}>Download HTML</button>{' '}
        <button onClick={() => download(data.export.markdown, 'executive-report.md', 'text/markdown')}>Download Markdown</button>{' '}
        <button onClick={() => window.print()}>Print view</button>
      </div>
    </>}

    {!data && !report.isPending && <div className='card' style={{ marginTop: 12 }}>Нажмите Build report, чтобы сформировать executive dashboard.</div>}
  </div>
}
