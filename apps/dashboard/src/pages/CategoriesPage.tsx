import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { EChart } from '../components/EChart'
import { apiGet } from '../lib/api'
import { useFilters } from '../state/filters'

type Row = {
  category: string
  count: number
  share: number
  baseline_count?: number | null
  delta_abs?: number | null
  delta_pct?: number | null
}

type SubRow = { subcategory: string; count: number; share?: number }

export default function CategoriesPage() {
  const f = useFilters()
  const [selected, setSelected] = useState<string>('')

  const qs = useMemo(() => {
    const q = new URLSearchParams()
    if (f.date_from) q.set('date_from', f.date_from)
    if (f.date_to) q.set('date_to', f.date_to)
    if (f.viz_tag) q.set('viz_tag', f.viz_tag)
    q.set('baseline_mode', f.baseline_mode)
    q.set('category_mode', f.categoryMode)
    q.set('top_n', String(f.topN))
    q.set('include_other', String(f.includeOther))
    for (const c of f.categories) q.append('category', c)
    if (f.baseline_date_from) q.set('baseline_date_from', f.baseline_date_from)
    if (f.baseline_date_to) q.set('baseline_date_to', f.baseline_date_to)
    return q.toString()
  }, [f])

  const q = useQuery({
    queryKey: ['categories', qs],
    queryFn: () => apiGet<{ rows: Row[] }>(`/api/categories?${qs}`),
  })

  const rows = q.data?.rows ?? []
  const topRows = rows.slice(0, 15)
  const growthRows = [...rows]
    .filter((r) => (r.delta_abs ?? 0) > 0)
    .sort((a, b) => (b.delta_abs ?? 0) - (a.delta_abs ?? 0))
    .slice(0, 10)

  const selectedCategory = selected || (rows[0]?.category ?? '')

  const subQ = useQuery({
    queryKey: ['subcategories', selectedCategory, qs],
    enabled: Boolean(selectedCategory),
    queryFn: () => apiGet<{ category: string; subcategories: SubRow[] }>(`/api/categories/${encodeURIComponent(selectedCategory)}/subcategories?${qs}`),
  })

  if (q.isLoading) return <div className='card'>Загрузка категорий...</div>
  if (q.error) return <div className='card'>Ошибка категорий: {(q.error as Error).message}</div>

  return <div>
    <div className='card'>
      <h3>Топ категорий по объему</h3>
      <EChart option={{
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'value' },
        yAxis: { type: 'category', data: topRows.map((r) => r.category) },
        series: [{ name: 'count', type: 'bar', data: topRows.map((r) => r.count) }],
      }} height={420} />
    </div>

    <div className='card' style={{ marginTop: 12 }}>
      <h3>Топ выросших относительно baseline</h3>
      <EChart option={{
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'value' },
        yAxis: { type: 'category', data: growthRows.map((r) => r.category) },
        series: [{ name: 'delta_abs', type: 'bar', data: growthRows.map((r) => r.delta_abs ?? 0) }],
      }} height={360} />
    </div>

    <table className='table' style={{ marginTop: 12 }}>
      <thead>
        <tr><th>Category</th><th>Count</th><th>Baseline</th><th>Δ abs</th><th>Δ %</th><th>Share</th></tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.category} onClick={() => setSelected(r.category)} style={{ cursor: 'pointer', background: selectedCategory === r.category ? '#eef2ff' : 'white' }}>
            <td>{r.category}</td>
            <td>{r.count.toFixed(0)}</td>
            <td>{(r.baseline_count ?? 0).toFixed(0)}</td>
            <td>{(r.delta_abs ?? 0).toFixed(0)}</td>
            <td>{r.delta_pct == null ? '—' : `${(r.delta_pct * 100).toFixed(1)}%`}</td>
            <td>{(r.share * 100).toFixed(1)}%</td>
          </tr>
        ))}
      </tbody>
    </table>

    <div className='card' style={{ marginTop: 12 }}>
      <h3>Подкатегории: {selectedCategory || '—'}</h3>
      {subQ.isLoading && <div>Загрузка подкатегорий...</div>}
      {subQ.error && <div>Ошибка подкатегорий: {(subQ.error as Error).message}</div>}
      {!!subQ.data && <EChart option={{
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'value' },
        yAxis: { type: 'category', data: subQ.data.subcategories.map((s) => s.subcategory) },
        series: [{ type: 'bar', data: subQ.data.subcategories.map((s) => s.count) }],
      }} height={360} />}
    </div>
  </div>
}
