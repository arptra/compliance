import { useMemo } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiGet, apiPost } from '../lib/api'
import { useFilters } from '../state/filters'

type TagsResp = { pattern_fit_tags: string[]; pattern_monitor_tags: string[] }
type SummaryResp = { tag: string; summary: Record<string, number> }
type RowsResp = { rows: Array<Record<string, unknown>>; tag?: string }
type RunResp = { status: string; outputs?: Record<string, string>; error?: string }

export default function PatternMonitorPage() {
  const f = useFilters()
  const qc = useQueryClient()
  const patternTag = f.pattern_tag || 'latest'

  const tagsQ = useQuery({ queryKey: ['meta-tags'], queryFn: () => apiGet<TagsResp>('/api/meta/tags') })

  const qs = useMemo(() => {
    const q = new URLSearchParams()
    q.set('pattern_tag', patternTag)
    if (f.date_from) q.set('date_from', f.date_from)
    if (f.date_to) q.set('date_to', f.date_to)
    return q.toString()
  }, [patternTag, f.date_from, f.date_to])

  const summaryQ = useQuery({ queryKey: ['pm-summary', qs], queryFn: () => apiGet<SummaryResp>(`/api/pattern-monitor/summary?${qs}`) })
  const alertsQ = useQuery({ queryKey: ['pm-alerts', qs], queryFn: () => apiGet<RowsResp>(`/api/pattern-monitor/alerts?${qs}&top_n=300`) })
  const examplesQ = useQuery({ queryKey: ['pm-examples', qs], queryFn: () => apiGet<RowsResp>(`/api/pattern-monitor/examples?${qs}&top_n=300`) })

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
      <h3>Все найденные жалобы за выбранный период</h3>
      {examplesQ.isLoading && <div>Загрузка...</div>}
      {examplesQ.error && <div>Ошибка: {(examplesQ.error as Error).message}</div>}
      {!examplesQ.isLoading && <table className='table'>
        <thead><tr><th>#</th><th>Date</th><th>Category</th><th>Score</th><th>Text</th></tr></thead>
        <tbody>
          {(examplesQ.data?.rows ?? []).map((r, i) => (
            <tr key={i}>
              <td>{i + 1}</td>
              <td>{String(r.date ?? '')}</td>
              <td>{String(r.category ?? 'UNKNOWN')}</td>
              <td>{String(r.row_score ?? r.score ?? '')}</td>
              <td>{String(r.client_first_message ?? r.dialog_text ?? r.complaint_text ?? '')}</td>
            </tr>
          ))}
        </tbody>
      </table>}
    </div>

    <div className='card' style={{ marginTop: 12 }}>
      <h3>Alerts</h3>
      {alertsQ.isLoading && <div>Загрузка...</div>}
      {!alertsQ.isLoading && <div>Alerts rows: {(alertsQ.data?.rows ?? []).length}</div>}
    </div>
  </div>
}
