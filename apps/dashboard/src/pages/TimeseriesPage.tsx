import { useQuery } from '@tanstack/react-query'
import { EChart } from '../components/EChart'
import { apiGet } from '../lib/api'

type TsPoint = { date: string; actual: number }
type ByCategoryRow = { date: string; category: string; count: number }

export default function TimeseriesPage() {
  const overall = useQuery({ queryKey: ['ts-overall'], queryFn: () => apiGet<TsPoint[]>('/api/timeseries/overall') })
  const byCategory = useQuery({ queryKey: ['ts-by-category'], queryFn: () => apiGet<{ rows: ByCategoryRow[] }>('/api/timeseries/by-category') })

  if (overall.isLoading || byCategory.isLoading) return <div className='card'>Загрузка timeseries...</div>
  if (overall.error) return <div className='card'>Ошибка overall: {(overall.error as Error).message}</div>
  if (byCategory.error) return <div className='card'>Ошибка by-category: {(byCategory.error as Error).message}</div>

  const overallRows = overall.data ?? []
  const catRows = byCategory.data?.rows ?? []

  const categories = Array.from(new Set(catRows.map((r) => r.category))).slice(0, 8)
  const dates = Array.from(new Set(catRows.map((r) => r.date))).sort()
  const series = categories.map((c) => ({
    name: c,
    type: 'line' as const,
    stack: 'total',
    areaStyle: {},
    data: dates.map((d) => catRows.filter((r) => r.date === d && r.category === c).reduce((s, r) => s + r.count, 0)),
  }))

  return <div>
    <div className='card'>
      <h3>Overall динамика</h3>
      <EChart option={{
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'category', data: overallRows.map((x) => x.date) },
        yAxis: { type: 'value' },
        series: [{ type: 'line', smooth: true, data: overallRows.map((x) => x.actual) }],
      }} />
    </div>

    <div className='card' style={{ marginTop: 12 }}>
      <h3>By-category stacked area</h3>
      <EChart option={{ tooltip: { trigger: 'axis' }, legend: { data: categories }, xAxis: { type: 'category', data: dates }, yAxis: { type: 'value' }, series }} height={380} />
    </div>
  </div>
}
