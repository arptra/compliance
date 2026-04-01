import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiGet, apiPost, apiPostForm } from '../lib/api'
import { useNavigate } from 'react-router-dom'

type Job = {
  upload_id: string
  original_filename: string
  uploaded_at: string
  status: 'uploaded' | 'queued' | 'running' | 'succeeded' | 'failed'
  rows_total: number
  prepared_rows: number
  date_min?: string | null
  date_max?: string | null
  available_for_pattern_monitor: boolean
  error_message?: string | null
}

type JobsResp = { jobs: Job[] }
type UploadResp = { upload_id: string; filename: string; uploaded_at: string; status: string }
type RunResp = { upload_id: string; status: string; error_message?: string | null }
type PresetResp = {
  allowed: boolean
  reason?: string | null
  pattern_monitor_preset?: { date_from?: string | null; date_to?: string | null; month?: string | null; upload_id: string; pattern_tag?: string | null; label_source: string; source_filename?: string | null } | null
}

export default function PreparationPage() {
  const qc = useQueryClient()
  const navigate = useNavigate()
  const [file, setFile] = useState<File | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)

  const upsertLocalJob = (job: Job) => {
    qc.setQueryData<JobsResp>(['prep-jobs'], (prev) => {
      const rows = prev?.jobs ?? []
      const rest = rows.filter((r) => r.upload_id !== job.upload_id)
      return { jobs: [job, ...rest] }
    })
  }

  const jobsQ = useQuery({
    queryKey: ['prep-jobs'],
    queryFn: () => apiGet<JobsResp>('/api/preparation/jobs?limit=40'),
    refetchInterval: (q) => {
      const jobs = (q.state.data as JobsResp | undefined)?.jobs ?? []
      return jobs.some((j) => j.status === 'running' || j.status === 'queued') ? 4000 : false
    },
  })

  const upload = useMutation({
    mutationFn: async () => {
      if (!file) throw new Error('Выберите файл')
      const form = new FormData()
      form.append('file', file)
      return apiPostForm<UploadResp>('/api/preparation/upload', form)
    },
    onSuccess: async (d) => {
      upsertLocalJob({
        upload_id: d.upload_id,
        original_filename: d.filename,
        uploaded_at: d.uploaded_at,
        status: 'uploaded',
        rows_total: 0,
        prepared_rows: 0,
        available_for_pattern_monitor: false,
      })
      setSelectedId(d.upload_id)
      await qc.invalidateQueries({ queryKey: ['prep-jobs'] })
      await jobsQ.refetch()
    },
  })

  const uploadAndRun = useMutation({
    mutationFn: async () => {
      if (!file) throw new Error('Выберите файл')
      const form = new FormData()
      form.append('file', file)
      const run = await apiPostForm<RunResp>('/api/preparation/upload-and-run', form)
      if (run.status !== 'succeeded') {
        throw new Error(run.error_message || `Разметка не завершена (status=${run.status})`)
      }
      return run
    },
    onSuccess: async (d) => {
      setSelectedId(d.upload_id)
      await qc.invalidateQueries({ queryKey: ['prep-jobs'] })
      await jobsQ.refetch()
    },
  })

  const run = useMutation({
    mutationFn: async (upload_id: string) => {
      const resp = await apiPost<RunResp>(`/api/preparation/${encodeURIComponent(upload_id)}/run`, {})
      if (resp.status !== 'succeeded') {
        throw new Error(resp.error_message || `Разметка не завершена (status=${resp.status})`)
      }
      return resp
    },
    onSuccess: async () => {
      await qc.invalidateQueries({ queryKey: ['prep-jobs'] })
      await jobsQ.refetch()
    },
  })

  const selected = useMemo(() => (jobsQ.data?.jobs ?? []).find((j) => j.upload_id === selectedId) ?? (jobsQ.data?.jobs ?? [])[0], [jobsQ.data, selectedId])

  const openInMonitor = useMutation({
    mutationFn: (upload_id: string) => apiPost<PresetResp>(`/api/preparation/jobs/${encodeURIComponent(upload_id)}/open-pattern-monitor`, {}),
    onSuccess: (d) => {
      const q = new URLSearchParams()
      const p = d.pattern_monitor_preset
      if (p?.upload_id) q.set('uploadId', p.upload_id)
      if (p?.date_from) q.set('autoDateFrom', p.date_from)
      if (p?.date_to) q.set('autoDateTo', p.date_to)
      if (p?.month) q.set('autoMonth', p.month)
      if (p?.pattern_tag) q.set('autoPatternTag', p.pattern_tag)
      if (p?.source_filename) q.set('sourceFilename', p.source_filename)
      if (!d.allowed) q.set('presetError', d.reason ?? 'pattern_monitor_preset_failed')
      q.set('fromPreparation', '1')
      navigate(`/pattern-monitor?${q.toString()}`)
    },
    onError: () => {
      navigate('/pattern-monitor?fromPreparation=1&presetError=open_pattern_monitor_failed')
    },
  })

  return <div>
    <div className='card'>
      <h2>Подготовка данных</h2>
      <p style={{ marginTop: 0 }}>Загрузка нового Excel и разметка через GigaChat.</p>
      <div className='filters'>
        <input type='file' accept='.xlsx,.xls,.csv' onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        <button onClick={() => upload.mutate()} disabled={!file || upload.isPending}>Загрузить файл</button>
        <button onClick={() => uploadAndRun.mutate()} disabled={!file || uploadAndRun.isPending}>
          {uploadAndRun.isPending ? 'Разметка...' : 'Разметить файл'}
        </button>
      </div>
      {upload.error && <div>Ошибка upload: {(upload.error as Error).message}</div>}
      {uploadAndRun.error && <div>Ошибка upload/run: {(uploadAndRun.error as Error).message}</div>}
    </div>

    <div className='card' style={{ marginTop: 12 }}>
      <h3>Recent uploads</h3>
      {jobsQ.isLoading && <div>Загрузка...</div>}
      {!jobsQ.isLoading && <table className='table'>
        <thead><tr><th>Filename</th><th>Uploaded</th><th>Status</th><th>Date range</th><th>Rows</th><th>Available</th><th>Actions</th></tr></thead>
        <tbody>
          {(jobsQ.data?.jobs ?? []).map((j) => <tr key={j.upload_id} onClick={() => setSelectedId(j.upload_id)} style={{ cursor: 'pointer', background: selected?.upload_id === j.upload_id ? '#f8fafc' : 'transparent' }}>
            <td>{j.original_filename}</td>
            <td>{new Date(j.uploaded_at).toLocaleString()}</td>
            <td><span className={`badge ${j.status === 'failed' ? 'high' : j.status === 'succeeded' ? 'low' : j.status === 'running' ? 'medium' : 'neutral'}`}>{j.status}</span></td>
            <td>{j.date_min ?? '—'} .. {j.date_max ?? '—'}</td>
            <td>{j.prepared_rows || j.rows_total || 0}</td>
            <td>{j.available_for_pattern_monitor ? 'yes' : 'no'}</td>
            <td>
              <button onClick={(e) => { e.stopPropagation(); run.mutate(j.upload_id) }} disabled={run.isPending || j.status === 'running' || j.status === 'queued'}>
                {run.isPending && run.variables === j.upload_id ? 'Разметка...' : 'Запустить разметку'}
              </button>{' '}
              <button onClick={(e) => { e.stopPropagation(); openInMonitor.mutate(j.upload_id) }} disabled={!j.available_for_pattern_monitor || openInMonitor.isPending}>
                {openInMonitor.isPending ? 'Открываем...' : 'Открыть в Pattern Monitor'}
              </button>
            </td>
          </tr>)}
        </tbody>
      </table>}
    </div>

    {selected && <div className='card' style={{ marginTop: 12 }}>
      <h3>Job details</h3>
      <ul>
        <li>upload_id: {selected.upload_id}</li>
        <li>status: <b>{selected.status}</b></li>
        <li>date range: {selected.date_min ?? '—'} .. {selected.date_max ?? '—'}</li>
        <li>prepared_rows: {selected.prepared_rows}</li>
        <li>available_for_pattern_monitor: {selected.available_for_pattern_monitor ? 'true' : 'false'}</li>
      </ul>
      {selected.status === 'running' || selected.status === 'queued' ? <p>Файл ещё размечается, анализ станет доступен после завершения подготовки.</p> : null}
      {selected.status === 'succeeded' ? <p><b>Файл обработан.</b> Диапазон дат: {selected.date_min ?? '—'} .. {selected.date_max ?? '—'}.</p> : null}
      {selected.error_message ? <p style={{ color: '#991b1b' }}>Error: {selected.error_message}</p> : null}
    </div>}
  </div>
}
