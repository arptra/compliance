import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiGet, apiPost } from '../lib/api'
import { useFilters } from '../state/filters'
import { useShallow } from 'zustand/react/shallow'

type TagsResp = { pattern_fit_tags: string[]; pattern_monitor_tags: string[] }
type SummaryResp = { tag: string; summary: Record<string, number> }
type RowsResp = { rows: Array<Record<string, unknown>>; tag?: string }
type RunResp = { status: string; outputs?: Record<string, string>; error?: string }

export default function PatternMonitorPage() {
  const f = useFilters(useShallow((s) => ({
    date_from: s.date_from,
    date_to: s.date_to,
    categories: s.categories,
    pattern_tag: s.pattern_tag,
    set: s.set,
  })))
  const qc = useQueryClient()
  const [dialogPreview, setDialogPreview] = useState<string | null>(null)
  const patternTag = f.pattern_tag || 'latest'

  const tagsQ = useQuery({ queryKey: ['meta-tags'], queryFn: () => apiGet<TagsResp>('/api/meta/tags') })

  const qs = useMemo(() => {
    const q = new URLSearchParams()
    q.set('pattern_tag', patternTag)
    if (f.date_from) q.set('date_from', f.date_from)
    if (f.date_to) q.set('date_to', f.date_to)
    for (const c of f.categories) q.append('category', c)
    return q.toString()
  }, [patternTag, f.date_from, f.date_to, f.categories])

  const summaryQ = useQuery({ queryKey: ['pm-summary', qs], queryFn: () => apiGet<SummaryResp>(`/api/pattern-monitor/summary?${qs}`), staleTime: 30_000, refetchOnWindowFocus: false })
  const alertsQ = useQuery({ queryKey: ['pm-alerts', qs], queryFn: () => apiGet<RowsResp>(`/api/pattern-monitor/alerts?${qs}&top_n=300`), staleTime: 30_000, refetchOnWindowFocus: false })
  const examplesQ = useQuery({ queryKey: ['pm-examples', qs], queryFn: () => apiGet<RowsResp>(`/api/pattern-monitor/examples?${qs}&top_n=300`), staleTime: 30_000, refetchOnWindowFocus: false })

  const runMonitor = useMutation({
    mutationFn: () => apiPost<RunResp>('/api/runs/pattern-monitor', {
      params: {
        tag: patternTag,
        date_from: f.date_from,
        date_to: f.date_to,
        fit_tag: patternTag,
      },
    }),
    onSuccess: async () => {
      await qc.invalidateQueries({ queryKey: ['pm-summary'] })
      await qc.invalidateQueries({ queryKey: ['pm-alerts'] })
      await qc.invalidateQueries({ queryKey: ['pm-examples'] })
      await qc.invalidateQueries({ queryKey: ['meta-tags'] })
    },
  })

  return <div>
    <div className='card'>
      <h3>Pattern Monitor: модель и период</h3>
      <div className='filters'>
        <select value={patternTag} onChange={(e) => f.set({ pattern_tag: e.target.value })}>
          <option value='latest'>latest</option>
          {(tagsQ.data?.pattern_monitor_tags ?? []).map((t) => <option key={t} value={t}>{t}</option>)}
          {(tagsQ.data?.pattern_fit_tags ?? []).map((t) => <option key={`fit-${t}`} value={t}>{t} (fit)</option>)}
        </select>
        <button onClick={() => runMonitor.mutate()} disabled={runMonitor.isPending}>Запустить pattern-monitor на выбранной модели</button>
      </div>
      {runMonitor.data && <p>Run status: <b>{runMonitor.data.status}</b> {runMonitor.data.error ? `(${runMonitor.data.error})` : ''}</p>}
    </div>

    <div className='card' style={{ marginTop: 12 }}>
      <h3>Сводка</h3>
      {summaryQ.isLoading && <div>Загрузка...</div>}
      {summaryQ.error && <div>Ошибка: {(summaryQ.error as Error).message}</div>}
      {!!summaryQ.data && <ul>
        <li>Tag: {summaryQ.data.tag}</li>
        <li>Scored rows: {summaryQ.data.summary.scored_rows ?? 0}</li>
        <li>Alert rows: {summaryQ.data.summary.alert_rows ?? 0}</li>
      </ul>}
    </div>

    <div className='card' style={{ marginTop: 12 }}>
      <h3>Найденные аномальные жалобы за выбранный период</h3>
      <p style={{ marginTop: 0, color: '#475569' }}>Период берётся из глобальных фильтров даты. Это аналог листа <b>alert_examples</b> из Excel-выгрузки.</p>
      {examplesQ.isLoading && <div>Загрузка...</div>}
      {examplesQ.error && <div>Ошибка: {(examplesQ.error as Error).message}</div>}
      {!examplesQ.isLoading && <table className='table'>
        <thead><tr><th>#</th><th>Date</th><th>Category</th><th>Subcategory</th><th>Score</th><th>row_dialog</th></tr></thead>
        <tbody>
          {(examplesQ.data?.rows ?? []).map((r, i) => {
            const dialog = String(r.row_dialog ?? '')
            const preview = dialog.length > 160 ? `${dialog.slice(0, 160)}…` : dialog
            return (
              <tr key={i}>
                <td>{i + 1}</td>
                <td>{String(r.date ?? '')}</td>
                <td>{String(r.category ?? 'UNKNOWN')}</td>
                <td>{String(r.subcategory ?? 'UNKNOWN')}</td>
                <td>{String(r.score ?? '')}</td>
                <td>
                  <button onClick={() => setDialogPreview(dialog)} style={{ border: 'none', background: 'transparent', color: '#1d4ed8', cursor: 'pointer', textAlign: 'left' }}>
                    {preview || '—'}
                  </button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>}
    </div>

    <div className='card' style={{ marginTop: 12 }}>
      <h3>Alerts</h3>
      {alertsQ.isLoading && <div>Загрузка...</div>}
      {!alertsQ.isLoading && <div>Alerts rows: {(alertsQ.data?.rows ?? []).length}</div>}
    </div>

    {dialogPreview !== null && (
      <div style={{ position: 'fixed', inset: 0, background: 'rgba(2,6,23,0.45)', display: 'flex', alignItems: 'flex-start', justifyContent: 'center', paddingTop: 60, zIndex: 40 }} onClick={() => setDialogPreview(null)}>
        <div className='card' style={{ width: 'min(1000px, 92vw)', maxHeight: '80vh', overflow: 'auto' }} onClick={(e) => e.stopPropagation()}>
          <h3 style={{ marginTop: 0 }}>Полный row_dialog</h3>
          <pre style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{dialogPreview}</pre>
          <div style={{ marginTop: 12 }}><button onClick={() => setDialogPreview(null)}>Закрыть</button></div>
        </div>
      </div>
    )}
  </div>
}
