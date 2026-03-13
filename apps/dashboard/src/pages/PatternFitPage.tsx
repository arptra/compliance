import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiGet, apiPost } from '../lib/api'
import { useFilters } from '../state/filters'

type TagsResp = { pattern_fit_tags: string[] }
type FitSummary = { tag: string; categories: string[]; growth_summary: Array<Record<string, unknown>> }
type CategoryRows = { tag: string; category: string; rows: Array<Record<string, unknown>> }

type RunResp = { status: string; outputs?: Record<string, string>; error?: string }

export default function PatternFitPage() {
  const f = useFilters()
  const qc = useQueryClient()
  const [normalFrom, setNormalFrom] = useState('')
  const [normalTo, setNormalTo] = useState('')
  const [eventFrom, setEventFrom] = useState('')
  const [eventTo, setEventTo] = useState('')

  const tag = f.pattern_tag || 'latest'

  const tagsQ = useQuery({ queryKey: ['meta-tags'], queryFn: () => apiGet<TagsResp>('/api/meta/tags') })
  const summaryQ = useQuery({ queryKey: ['pf-summary', tag], queryFn: () => apiGet<FitSummary>(`/api/pattern-fit/summary?tag=${encodeURIComponent(tag)}`) })

  const [selectedCategory, setSelectedCategory] = useState('')
  const effectiveCategory = selectedCategory || summaryQ.data?.categories?.[0] || ''
  const rowsQ = useQuery({
    queryKey: ['pf-category', tag, effectiveCategory],
    enabled: Boolean(effectiveCategory),
    queryFn: () => apiGet<CategoryRows>(`/api/pattern-fit/category/${encodeURIComponent(effectiveCategory)}?tag=${encodeURIComponent(tag)}`),
  })

  const runFit = useMutation({
    mutationFn: () => apiPost<RunResp>('/api/runs/pattern-fit', {
      params: {
        tag,
        normal_period: normalFrom && normalTo ? `${normalFrom}:${normalTo}` : undefined,
        event_period: eventFrom && eventTo ? `${eventFrom}:${eventTo}` : undefined,
      },
    }),
    onSuccess: async () => {
      await qc.invalidateQueries({ queryKey: ['pf-summary'] })
      await qc.invalidateQueries({ queryKey: ['meta-tags'] })
    },
  })

  const growth = summaryQ.data?.growth_summary ?? []
  const growthRows = useMemo(() => growth.slice(0, 20), [growth])

  return <div>
    <div className='card'>
      <h3>Pattern Fit: переобучение модели</h3>
      <div className='filters'>
        <select value={tag} onChange={(e) => f.set({ pattern_tag: e.target.value })}>
          <option value='latest'>latest</option>
          {(tagsQ.data?.pattern_fit_tags ?? []).map((t) => <option key={t} value={t}>{t}</option>)}
        </select>
        <label>Normal from <input type='date' value={normalFrom} onChange={(e) => setNormalFrom(e.target.value)} /></label>
        <label>Normal to <input type='date' value={normalTo} onChange={(e) => setNormalTo(e.target.value)} /></label>
        <label>Anomaly from <input type='date' value={eventFrom} onChange={(e) => setEventFrom(e.target.value)} /></label>
        <label>Anomaly to <input type='date' value={eventTo} onChange={(e) => setEventTo(e.target.value)} /></label>
        <button onClick={() => runFit.mutate()} disabled={runFit.isPending}>Запустить pattern-fit</button>
      </div>
      {runFit.data && <p>Run status: <b>{runFit.data.status}</b> {runFit.data.error ? `(${runFit.data.error})` : ''}</p>}
    </div>

    <div className='card' style={{ marginTop: 12 }}>
      <h3>Топ выросших категорий</h3>
      {summaryQ.isLoading && <div>Загрузка...</div>}
      {summaryQ.error && <div>Ошибка: {(summaryQ.error as Error).message}</div>}
      {!summaryQ.isLoading && <table className='table'>
        <thead><tr><th>Category</th><th>Δ</th></tr></thead>
        <tbody>
          {growthRows.map((r, i) => <tr key={i}><td>{String(r.category ?? r.complaint_category_llm ?? 'UNKNOWN')}</td><td>{String(r.delta ?? r.delta_abs ?? '')}</td></tr>)}
        </tbody>
      </table>}
    </div>

    <div className='card' style={{ marginTop: 12 }}>
      <h3>Жалобы по выбранной категории ({effectiveCategory || '—'})</h3>
      <select value={effectiveCategory} onChange={(e) => setSelectedCategory(e.target.value)}><option value=''>auto</option>{(summaryQ.data?.categories ?? []).map((c) => <option key={c} value={c}>{c}</option>)}</select>
      {rowsQ.isLoading && <div>Загрузка...</div>}
      {rowsQ.error && <div>Ошибка: {(rowsQ.error as Error).message}</div>}
      {!rowsQ.isLoading && <table className='table'>
        <thead><tr><th>#</th><th>Date</th><th>Category</th><th>Text</th></tr></thead>
        <tbody>
          {(rowsQ.data?.rows ?? []).slice(0, 100).map((r, i) => (
            <tr key={i}>
              <td>{i + 1}</td>
              <td>{String(r.date ?? r.event_time ?? '')}</td>
              <td>{String(r.category ?? r.complaint_category_llm ?? '')}</td>
              <td>{String(r.client_first_message ?? r.dialog_text ?? r.text ?? '')}</td>
            </tr>
          ))}
        </tbody>
      </table>}
    </div>
  </div>
}
