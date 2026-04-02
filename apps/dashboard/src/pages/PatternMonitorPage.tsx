import { useCallback, useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiGet, apiPost } from '../lib/api'
import { useFilters } from '../state/filters'
import { useShallow } from 'zustand/react/shallow'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { ReviewVerdictControl } from '../components/pattern-monitor/ReviewVerdictControl'
import { ReviewReasonSelect } from '../components/pattern-monitor/ReviewReasonSelect'
import { ReviewCommentDialog } from '../components/pattern-monitor/ReviewCommentDialog'
import { ReviewSummaryBar } from '../components/pattern-monitor/ReviewSummaryBar'
import { FeedbackMetricsPanel } from '../components/pattern-monitor/FeedbackMetricsPanel'
import { ScoringModeSwitch } from '../components/pattern-monitor/ScoringModeSwitch'
import { ModelVersionBadge } from '../components/pattern-monitor/ModelVersionBadge'

type TagsResp = { pattern_fit_tags: string[]; pattern_monitor_tags: string[] }
type SummaryResp = { tag: string; allowed?: boolean; reason?: string | null; upload_id?: string | null; summary: Record<string, number | string | null> }
type AlertsResp = { rows: Array<Record<string, unknown>>; scoring_mode_requested: 'base'|'calibrated'|'reranked'; scoring_mode_effective: 'base'|'calibrated'|'reranked'; reranker_available: boolean; active_calibrator_version?: string | null }
type ExamplesResp = { rows: Array<Record<string, unknown>>; allowed?: boolean; reason?: string | null }
type RunResp = { status: string; outputs?: Record<string, string>; error?: string }
type FeedbackSummary = Record<string, number | string | null>
type VersionRow = { version_id: string; status: string; created_at: string; train_rows?: number; metrics_json?: Record<string, unknown> | null; active: number }

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
  const [reviewMode, setReviewMode] = useState(false)
  const [scoringMode, setScoringMode] = useState<'base'|'calibrated'|'reranked'>('base')
  const [lastRunInfo, setLastRunInfo] = useState<string>('not_started')
  const [lastRunScoredPath, setLastRunScoredPath] = useState<string>('')
  const [tableLimit, setTableLimit] = useState<'all' | 10 | 20 | 100>('all')
  const [sp, setSp] = useSearchParams()
  const navigate = useNavigate()

  const uploadId = sp.get('uploadId') || undefined
  const sourceFilename = sp.get('sourceFilename') || undefined
  const autoDateFrom = sp.get('autoDateFrom') || undefined
  const autoDateTo = sp.get('autoDateTo') || undefined
  const autoMonth = sp.get('autoMonth') || undefined
  const autoPatternTag = sp.get('autoPatternTag') || undefined
  const fromPreparation = sp.get('fromPreparation') === '1'
  const presetError = sp.get('presetError') || undefined
  const patternTag = uploadId ? (autoPatternTag || f.pattern_tag || 'latest') : (f.pattern_tag || 'latest')

  useEffect(() => {
    if (uploadId && autoPatternTag && f.pattern_tag !== autoPatternTag) {
      f.set({ pattern_tag: autoPatternTag })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uploadId, autoPatternTag])

  const tagsQ = useQuery({ queryKey: ['meta-tags'], queryFn: () => apiGet<TagsResp>('/api/meta/tags') })

  const qs = useMemo(() => {
    const q = new URLSearchParams()
    q.set('pattern_tag', patternTag)
    const dateFrom = f.date_from || autoDateFrom
    const dateTo = f.date_to || autoDateTo
    q.set('date_from', dateFrom || '')
    q.set('date_to', dateTo || '')
    q.set('scoring_mode', scoringMode)
    if (uploadId) q.set('upload_id', uploadId)
    for (const c of f.categories) q.append('category', c)
    return q.toString()
  }, [patternTag, f.date_from, f.date_to, f.categories, uploadId, autoDateFrom, autoDateTo, scoringMode])

  const summaryUrl = `/api/pattern-monitor/summary?${qs}`
  const alertsUrl = `/api/pattern-monitor/alerts?${qs}`
  const examplesUrl = `/api/pattern-monitor/examples?${qs}`
  const runOutputUrl = `/api/pattern-monitor/run-output?pattern_tag=${encodeURIComponent(patternTag)}`
  const runOutputByPathUrl = lastRunScoredPath ? `/api/pattern-monitor/run-output-by-path?path=${encodeURIComponent(lastRunScoredPath)}` : ''
  const topAlertsExcelUrl = `/api/pattern-monitor/top-alerts-excel?${qs}`

  const summaryQ = useQuery({ queryKey: ['pm-summary', qs], queryFn: () => apiGet<SummaryResp>(summaryUrl), staleTime: 30_000, refetchOnWindowFocus: false })
  const alertsQ = useQuery({ queryKey: ['pm-alerts', qs], queryFn: () => apiGet<AlertsResp>(alertsUrl), staleTime: 30_000, refetchOnWindowFocus: false, enabled: summaryQ.data?.allowed !== false })
  const examplesQ = useQuery({
    queryKey: ['pm-examples', qs],
    queryFn: () => apiGet<ExamplesResp>(examplesUrl),
    staleTime: 30_000,
    refetchOnWindowFocus: false,
    enabled: summaryQ.data?.allowed !== false && !alertsQ.isLoading && (alertsQ.data?.rows?.length ?? 0) === 0,
  })
  const runOutputQ = useQuery({
    queryKey: ['pm-run-output', patternTag],
    queryFn: () => apiGet<AlertsResp>(runOutputUrl),
    staleTime: 30_000,
    refetchOnWindowFocus: false,
    enabled: !alertsQ.isLoading && (alertsQ.data?.rows?.length ?? 0) === 0 && !examplesQ.isLoading && (examplesQ.data?.rows?.length ?? 0) === 0,
  })
  const runOutputByPathQ = useQuery({
    queryKey: ['pm-run-output-path', lastRunScoredPath],
    queryFn: () => apiGet<AlertsResp>(runOutputByPathUrl),
    staleTime: 30_000,
    refetchOnWindowFocus: false,
    enabled: Boolean(lastRunScoredPath),
  })
  const topAlertsExcelQ = useQuery({
    queryKey: ['pm-top-alerts-excel', topAlertsExcelUrl],
    queryFn: () => apiGet<AlertsResp>(topAlertsExcelUrl),
    staleTime: 0,
    refetchOnWindowFocus: false,
  })
  const feedbackSummaryQ = useQuery({ queryKey: ['feedback-summary', patternTag], queryFn: () => apiGet<FeedbackSummary>(`/api/feedback/summary?pattern_tag=${patternTag}`), staleTime: 10_000 })
  const versionsQ = useQuery({ queryKey: ['calibrator-versions'], queryFn: () => apiGet<VersionRow[]>('/api/pattern-monitor/calibrator/versions') })

  const runMonitor = useMutation({
    mutationFn: (trigger: 'manual_click') => {
      const params = fromPreparation
        ? ((f.date_from || autoDateFrom || f.date_to || autoDateTo)
            ? { tag: patternTag, date_from: f.date_from || autoDateFrom, date_to: f.date_to || autoDateTo, label_source: 'llm', force_materialize: true, fit_tag: 'latest', categories: f.categories, trigger }
            : { tag: patternTag, month: autoMonth, label_source: 'llm', force_materialize: true, fit_tag: 'latest', categories: f.categories, trigger })
        : { tag: patternTag, date_from: f.date_from || autoDateFrom, date_to: f.date_to || autoDateTo, fit_tag: 'latest', categories: f.categories, trigger }
      console.info('[PatternMonitorPage] run pattern-monitor', params)
      return apiPost<RunResp>('/api/runs/pattern-monitor', { params })
    },
    onSuccess: async (resp) => {
      setLastRunInfo(`success @ ${new Date().toISOString()}`)
      setLastRunScoredPath(String(resp.outputs?.scored ?? ''))
      await qc.invalidateQueries({ queryKey: ['pm-summary'] }); await qc.invalidateQueries({ queryKey: ['pm-alerts'] }); await qc.invalidateQueries({ queryKey: ['meta-tags'] })
    },
    onError: (e) => {
      setLastRunInfo(`error @ ${new Date().toISOString()} :: ${String(e)}`)
    },
  })

  const triggerRun = useCallback((trigger: 'manual_click') => {
    if (!runMonitor.isPending) runMonitor.mutate(trigger)
  }, [runMonitor])

  const hasRunFilter = useMemo(() => {
    const hasDateRange = Boolean((f.date_from || autoDateFrom) && (f.date_to || autoDateTo))
    const hasCategories = f.categories.length > 0
    const hasUploadContext = Boolean(uploadId)
    return hasDateRange || hasCategories || hasUploadContext
  }, [f.date_from, f.date_to, f.categories, autoDateFrom, autoDateTo, uploadId])

  const saveFeedback = useMutation({ mutationFn: (payload: Record<string, unknown>) => apiPost('/api/feedback', payload), onSuccess: () => qc.invalidateQueries({ queryKey: ['feedback-summary'] }) })
  const resetFeedbackOne = useMutation({ mutationFn: (payload: { row_id: string, pattern_tag: string }) => apiPost(`/api/feedback/reset?row_id=${encodeURIComponent(payload.row_id)}&pattern_tag=${encodeURIComponent(payload.pattern_tag)}`, {}), onSuccess: async () => { await qc.invalidateQueries({ queryKey: ['feedback-summary'] }); await qc.invalidateQueries({ queryKey: ['pm-alerts'] }) } })
  const resetFeedbackAll = useMutation({ mutationFn: () => apiPost(`/api/feedback/reset-all?pattern_tag=${encodeURIComponent(patternTag)}`, {}), onSuccess: async () => { await qc.invalidateQueries({ queryKey: ['feedback-summary'] }); await qc.invalidateQueries({ queryKey: ['pm-alerts'] }) } })
  const trainCalibrator = useMutation({ mutationFn: () => apiPost('/api/pattern-monitor/calibrator/train', { pattern_tag: patternTag, date_from: f.date_from || autoDateFrom, date_to: f.date_to || autoDateTo, activate_if_better: true }), onSuccess: async () => { await qc.invalidateQueries({ queryKey: ['calibrator-versions'] }); await qc.invalidateQueries({ queryKey: ['pm-alerts'] }) } })
  const activateVersion = useMutation({ mutationFn: (v: string) => apiPost(`/api/pattern-monitor/calibrator/${v}/activate`, {}), onSuccess: () => { qc.invalidateQueries({ queryKey: ['calibrator-versions'] }); qc.invalidateQueries({ queryKey: ['pm-alerts'] }) } })

  const clearUploadFilter = () => { const next = new URLSearchParams(sp); ['uploadId','sourceFilename','autoDateFrom','autoDateTo','autoMonth','autoPatternTag','fromPreparation'].forEach((k) => next.delete(k)); setSp(next); navigate(`/pattern-monitor${next.toString() ? `?${next.toString()}` : ''}`) }

  const onVerdict = (row: Record<string, unknown>, verdict: 'true'|'false'|'uncertain', reason_code?: string, comment?: string) => {
    saveFeedback.mutate({ row_id: String(row.row_id ?? ''), pattern_tag: patternTag, verdict, reason_code, comment, category: row.category, subcategory: row.subcategory, base_score: row.pattern_like_score ?? row.row_score, rerank_score: row.rerank_score ?? row.calibrated_score })
  }
  const alertRowsFallback = (alertsQ.data?.rows ?? []).filter((row) => {
    const raw = row.is_pattern_alert
    if (typeof raw === 'boolean') return raw
    return ['true', '1', 'yes', 'y', 't'].includes(String(raw ?? '').trim().toLowerCase())
  })
  const tableRows = (topAlertsExcelQ.data?.rows?.length ?? 0) > 0 ? (topAlertsExcelQ.data?.rows ?? []) : alertRowsFallback
  const visibleRows = tableLimit === 'all' ? tableRows : tableRows.slice(0, tableLimit)

  return <div>
    {runMonitor.isPending && (
      <div style={{ position: 'fixed', inset: 0, background: 'rgba(2,6,23,0.45)', zIndex: 60, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div className='card' style={{ width: 420, textAlign: 'center' }}>
          <div style={{ fontSize: 18, fontWeight: 600, marginBottom: 10 }}>Идет поиск аномальных жалоб…</div>
          <div style={{ height: 8, borderRadius: 999, background: '#e2e8f0', overflow: 'hidden' }}>
            <div style={{ width: '40%', height: '100%', background: '#2563eb', animation: 'pmPulse 1.1s ease-in-out infinite' }} />
          </div>
          <div style={{ marginTop: 10, fontSize: 12, opacity: 0.8 }}>Пожалуйста подождите, обновляем Pattern Monitor</div>
        </div>
      </div>
    )}
    <style>{`@keyframes pmPulse {0%{transform:translateX(-120%)}100%{transform:translateX(320%)}}`}</style>
    {uploadId && <div className='card' style={{ marginBottom: 12 }}><b>Вы анализируете новый файл:</b> {sourceFilename ?? uploadId}. Диапазон дат: {(f.date_from || autoDateFrom || '—')} .. {(f.date_to || autoDateTo || '—')}.{presetError ? <div style={{ color: '#991b1b', marginTop: 6 }}>Preset warning: {presetError}</div> : null}<div style={{ marginTop: 8 }}><button onClick={clearUploadFilter}>Сбросить фильтр файла</button></div></div>}

    <div className='card'>
      <h3>Pattern Monitor: модель и период</h3>
      <div className='filters'>
        <select value={patternTag} onChange={(e) => f.set({ pattern_tag: e.target.value })}><option value='latest'>latest</option>{(tagsQ.data?.pattern_monitor_tags ?? []).map((t) => <option key={t} value={t}>{t}</option>)}{(tagsQ.data?.pattern_fit_tags ?? []).map((t) => <option key={`fit-${t}`} value={t}>{t} (fit)</option>)}</select>
        <ScoringModeSwitch value={scoringMode} disabledModes={alertsQ.data?.reranker_available ? [] : ['calibrated', 'reranked']} onChange={setScoringMode} />
        <label><input type='checkbox' checked={reviewMode} onChange={(e) => setReviewMode(e.target.checked)} /> Review mode</label>
        <button onClick={() => triggerRun('manual_click')} disabled={runMonitor.isPending || !hasRunFilter}>Старт pattern-monitor</button>
        <button onClick={() => trainCalibrator.mutate()} disabled={trainCalibrator.isPending}>Train calibrator</button>
        {reviewMode && <button onClick={() => resetFeedbackAll.mutate()} disabled={resetFeedbackAll.isPending}>Сбросить все review</button>}
      </div>
      {!hasRunFilter && <div style={{ marginTop: 8, fontSize: 12, opacity: 0.8 }}>Сначала выставьте фильтр (даты и/или категории), затем нажмите «Старт pattern-monitor».</div>}
      <div style={{ marginTop: 8, fontSize: 12, opacity: 0.8 }}>Run status: {runMonitor.isPending ? 'running…' : lastRunInfo}</div>
      <div style={{ marginTop: 8, fontSize: 11, opacity: 0.7 }}>
        API: <code>{alertsUrl}</code>
        {alertsQ.error ? <span style={{ color: '#991b1b' }}> | alerts error: {String(alertsQ.error)}</span> : null}
        {(topAlertsExcelQ.data?.rows?.length ?? 0) > 0 ? <span> | source: /top-alerts-excel</span> : null}
        {(topAlertsExcelQ.data?.rows?.length ?? 0) === 0 && alertRowsFallback.length > 0 ? <span> | fallback: /alerts (is_pattern_alert=true)</span> : null}
      </div>
      <div style={{ marginTop: 8 }}>Active version: <ModelVersionBadge version={alertsQ.data?.active_calibrator_version} /></div>
    </div>

    <ReviewSummaryBar summary={feedbackSummaryQ.data ?? {}} mode={alertsQ.data?.scoring_mode_effective ?? scoringMode} version={alertsQ.data?.active_calibrator_version} />
    <FeedbackMetricsPanel summary={feedbackSummaryQ.data} />

    <div className='card' style={{ marginTop: 12 }}>
      <h3>Model versions</h3>
      {(versionsQ.data ?? []).map((v) => <div key={v.version_id} style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
        <code>{v.version_id}</code><span>{v.status}</span><span>rows:{v.train_rows ?? 0}</span><span>precision:{String(v.metrics_json?.precision_reviewed ?? 'n/a')}</span>
        {v.active ? <b>active</b> : <button onClick={() => activateVersion.mutate(v.version_id)}>Activate</button>}
      </div>)}
    </div>

    <div className='card' style={{ marginTop: 12 }}>
      <h3>Найденные аномальные жалобы</h3>
      <div className='filters' style={{ marginBottom: 8 }}>
        <label>Показывать строк:</label>
        <select value={String(tableLimit)} onChange={(e) => setTableLimit(e.target.value === 'all' ? 'all' : Number(e.target.value) as 10 | 20 | 100)}>
          <option value='all'>Все</option>
          <option value='10'>10</option>
          <option value='20'>20</option>
          <option value='100'>100</option>
        </select>
      </div>
      {!alertsQ.isLoading && !topAlertsExcelQ.isLoading && <table className='table'>
        <thead><tr><th>#</th><th>Date</th><th>Category</th><th>Subcategory</th><th>Base</th><th>Rerank</th><th>dialog</th>{reviewMode && <><th>Verdict</th><th>Reason</th><th>Comment</th><th>Reset</th></>}</tr></thead>
        <tbody>
          {visibleRows.map((r, i) => {
            const dialog = String(r.row_dialog ?? r.dialog_text ?? '')
            const preview = dialog.length > 120 ? `${dialog.slice(0, 120)}…` : dialog
            const reasonValue = String(r.reason_code ?? '')
            const commentValue = String(r.comment ?? '')
            return <tr key={String(r.row_id ?? i)}>
              <td>{i + 1}</td><td>{String(r.event_time ?? r.date ?? '')}</td><td>{String(r.category_label_ru ?? r.category ?? 'UNKNOWN')}</td><td>{String(r.subcategory_label_ru ?? r.subcategory ?? 'UNKNOWN')}</td><td>{String(r.pattern_like_score ?? r.row_score ?? '')}</td><td>{String(r.rerank_score ?? r.calibrated_score ?? '')}</td>
              <td><button onClick={() => setDialogPreview(dialog)} style={{ border: 'none', background: 'transparent', color: '#1d4ed8', cursor: 'pointer', textAlign: 'left' }}>{preview || '—'}</button></td>
              {reviewMode && <><td><ReviewVerdictControl value={String(r.feedback_verdict ?? '')} onChange={(v) => onVerdict(r, v, reasonValue, commentValue)} /></td><td><ReviewReasonSelect value={reasonValue} onChange={(v) => onVerdict(r, (String(r.feedback_verdict ?? 'uncertain') as 'true'|'false'|'uncertain'), v, commentValue)} /></td><td><ReviewCommentDialog value={commentValue} onChange={(v) => onVerdict(r, (String(r.feedback_verdict ?? 'uncertain') as 'true'|'false'|'uncertain'), reasonValue, v)} /></td><td><button onClick={() => resetFeedbackOne.mutate({ row_id: String(r.row_id ?? ''), pattern_tag: patternTag })}>Сбросить</button></td></>}
            </tr>
          })}
        </tbody>
      </table>}
    </div>

    {dialogPreview !== null && <div style={{ position: 'fixed', inset: 0, background: 'rgba(2,6,23,0.45)', display: 'flex', alignItems: 'flex-start', justifyContent: 'center', paddingTop: 60, zIndex: 40 }} onClick={() => setDialogPreview(null)}><div className='card' style={{ width: 'min(1000px, 92vw)', maxHeight: '80vh', overflow: 'auto' }} onClick={(e) => e.stopPropagation()}><h3 style={{ marginTop: 0 }}>Полный row_dialog</h3><pre style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{dialogPreview}</pre><div style={{ marginTop: 12 }}><button onClick={() => setDialogPreview(null)}>Закрыть</button></div></div></div>}
  </div>
}
