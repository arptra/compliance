import { useQuery } from '@tanstack/react-query'
import { EChart } from '../components/EChart'
import { apiGet } from '../lib/api'

type Row = { category: string; count: number; share: number; delta_abs?: number | null }

export default function CategoriesPage() {
  const q = useQuery({ queryKey: ['categories'], queryFn: () => apiGet<{ rows: Row[] }>('/api/categories') })
  if (q.isLoading) return <div className='card'>Загрузка категорий...</div>
  if (q.error) return <div className='card'>Ошибка категорий: {(q.error as Error).message}</div>

  const rows = q.data?.rows ?? []

  return <div>
    <div className='card'>
      <h3>Категории по объему</h3>
      <EChart option={{
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'value' },
        yAxis: { type: 'category', data: rows.map((r) => r.category) },
        series: [{ type: 'bar', data: rows.map((r) => r.count) }]
      }} height={420} />
    </div>

    <table className='table' style={{ marginTop: 12 }}>
      <thead><tr><th>Category</th><th>Count</th><th>Share</th></tr></thead>
      <tbody>{rows.map(r => <tr key={r.category}><td>{r.category}</td><td>{r.count}</td><td>{(r.share*100).toFixed(1)}%</td></tr>)}</tbody>
    </table>
  </div>
}
