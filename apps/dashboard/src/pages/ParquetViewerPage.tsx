import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { apiGet } from '../lib/api'
import type { PreparePreviewResponse } from '../types/api'

export default function ParquetViewerPage() {
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(50)
  const [q, setQ] = useState('')

  const qs = useMemo(() => {
    const p = new URLSearchParams()
    p.set('page', String(page))
    p.set('page_size', String(pageSize))
    if (q.trim()) p.set('q', q.trim())
    return p.toString()
  }, [page, pageSize, q])

  const previewQ = useQuery({
    queryKey: ['prepare-preview', qs],
    queryFn: () => apiGet<PreparePreviewResponse>(`/api/meta/prepare-preview?${qs}`),
  })

  const data = previewQ.data
  const rows = data?.items ?? []
  const cols = data?.columns ?? []
  const totalPages = Math.max(1, Math.ceil((data?.total ?? 0) / (data?.page_size ?? 1)))

  return <div>
    <div className='card'>
      <h2>Prepare parquet viewer</h2>
      <div>Просмотр исходного parquet, из которого строятся данные дашборда.</div>
      <div style={{ marginTop: 8, fontSize: 12, opacity: 0.8 }}><code>{data?.path ?? '—'}</code></div>
    </div>

    <div className='card' style={{ marginTop: 12 }}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
        <input
          placeholder='Поиск по row_id / text / category / subcategory'
          value={q}
          onChange={(e) => { setQ(e.target.value); setPage(1) }}
          style={{ minWidth: 360 }}
        />
        <label style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          Page size
          <select value={pageSize} onChange={(e) => { setPageSize(Number(e.target.value)); setPage(1) }}>
            {[25, 50, 100, 200].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
      </div>

      {previewQ.isLoading ? <div>Loading…</div> : null}
      {previewQ.isError ? <div>Failed to load parquet preview.</div> : null}

      {!previewQ.isLoading && !previewQ.isError && rows.length === 0 ? <div>No rows found.</div> : null}

      {!previewQ.isLoading && !previewQ.isError && rows.length > 0 ? <div style={{ overflowX: 'auto' }}>
        <table>
          <thead>
            <tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((r, i) => <tr key={String((r.row_id as string | undefined) ?? i)}>
              {cols.map((c) => <td key={c}>{String((r[c] as string | number | boolean | null | undefined) ?? '')}</td>)}
            </tr>)}
          </tbody>
        </table>
      </div> : null}

      <div style={{ display: 'flex', gap: 8, marginTop: 8, alignItems: 'center' }}>
        <button disabled={page <= 1} onClick={() => setPage((v) => Math.max(1, v - 1))}>Prev</button>
        <div>Page {data?.page ?? page} / {totalPages}, rows: {data?.total ?? 0}</div>
        <button disabled={page >= totalPages} onClick={() => setPage((v) => v + 1)}>Next</button>
      </div>
    </div>
  </div>
}
