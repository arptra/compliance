import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { apiGet } from '../lib/api'
import type { FeedbackDatasetItem, FeedbackDatasetResponse, ReviewDatasetFiltersState } from '../types/api'
import { ReviewDatasetFilters } from '../components/review-dataset/ReviewDatasetFilters'
import { ReviewDatasetTable } from '../components/review-dataset/ReviewDatasetTable'
import { ReviewDetailsDrawer } from '../components/review-dataset/ReviewDetailsDrawer'

export default function ReviewDatasetPage() {
  const [filters, setFilters] = useState<ReviewDatasetFiltersState>({ page: 1, page_size: 25, sort_by: 'updated_at', sort_order: 'desc' })
  const [selected, setSelected] = useState<FeedbackDatasetItem | null>(null)
  const qs = useMemo(() => {
    const p = new URLSearchParams()
    for (const [k, v] of Object.entries(filters)) {
      if (v !== undefined && v !== '') p.set(k, String(v))
    }
    return p.toString()
  }, [filters])

  const datasetQ = useQuery({ queryKey: ['review-dataset', qs], queryFn: () => apiGet<FeedbackDatasetResponse>(`/api/feedback/dataset?${qs}`) })
  const data = datasetQ.data

  return <div>
    <div className='card'>
      <h2>Review Dataset</h2>
      <div>Накопленная разметка аналитиков для второго слоя</div>
    </div>

    <div className='card-grid' style={{ marginTop: 12 }}>
      <div className='card'><b>total reviewed rows</b><div className='kpi-value'>{data?.summary.reviewed_rows ?? 0}</div></div>
      <div className='card'><b>true / false / uncertain</b><div className='kpi-value'>{data?.summary.true_count ?? 0} / {data?.summary.false_count ?? 0} / {data?.summary.uncertain_count ?? 0}</div></div>
      <div className='card'><b>precision reviewed</b><div className='kpi-value'>{data?.summary.precision_reviewed ?? '—'}</div><div>active model: {data?.summary.active_model_version ?? '—'}</div></div>
    </div>

    <ReviewDatasetFilters filters={filters} onChange={(patch) => setFilters((prev) => ({ ...prev, ...patch, page: 1 }))} />

    <div className='card' style={{ marginTop: 12 }}>
      <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
        <a href={`http://localhost:8000/api/feedback/export?${qs}&output_format=csv`} target='_blank' rel='noreferrer'><button>Export CSV</button></a>
        <a href={`http://localhost:8000/api/feedback/export?${qs}&output_format=json`} target='_blank' rel='noreferrer'><button>Export JSON</button></a>
        <a href={`http://localhost:8000/api/feedback/export?${qs}&output_format=csv&positives_only=true`} target='_blank' rel='noreferrer'><button>Export positives only</button></a>
      </div>
      {!data?.items?.length ? <div>No feedback data yet.</div> : <ReviewDatasetTable rows={data.items} onSelect={setSelected} />}
      <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
        <button disabled={(filters.page ?? 1) <= 1} onClick={() => setFilters((p) => ({ ...p, page: Math.max((p.page ?? 1) - 1, 1) }))}>Prev</button>
        <div>Page {data?.page ?? 1} / {Math.max(1, Math.ceil((data?.total ?? 0) / (data?.page_size ?? 1)))}</div>
        <button disabled={(data?.page ?? 1) >= Math.ceil((data?.total ?? 0) / (data?.page_size ?? 1))} onClick={() => setFilters((p) => ({ ...p, page: (p.page ?? 1) + 1 }))}>Next</button>
      </div>
    </div>

    <ReviewDetailsDrawer item={selected} onClose={() => setSelected(null)} />
  </div>
}
