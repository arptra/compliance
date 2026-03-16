import { EChart } from '../components/EChart'
import { useOverview } from '../hooks/useOverview'

export default function OverviewPage() {
  const q = useOverview()
  if (q.isLoading) return <div className='card'>Загрузка overview...</div>
  if (q.error) return <div className='card'>Ошибка: {(q.error as Error).message}</div>
  const data = q.data!

  const dates = data.actual_vs_expected.map((x) => x.date)
  const actual = data.actual_vs_expected.map((x) => x.actual)
  const expected = data.actual_vs_expected.map((x) => x.expected ?? null)

  return <div>
    <div className='card-grid'>
      {data.kpis.map((k) => <div className='card' key={k.key}><div>{k.key}</div><strong>{k.value.toFixed(0)}</strong></div>)}
    </div>

    <div className='card' style={{ marginTop: 12 }}>
      <h3>Actual vs Expected</h3>
      <EChart
        option={{
          tooltip: { trigger: 'axis' },
          legend: { data: ['actual', 'expected'] },
          xAxis: { type: 'category', data: dates },
          yAxis: { type: 'value' },
          series: [
            { name: 'actual', type: 'line', smooth: true, data: actual, areaStyle: {} },
            { name: 'expected', type: 'line', smooth: true, data: expected }
          ]
        }}
      />
    </div>

    <div className='card' style={{ marginTop: 12 }}>
      <h3>Top growth categories</h3>
      <EChart
        option={{
          tooltip: { trigger: 'axis' },
          xAxis: { type: 'value' },
          yAxis: { type: 'category', data: data.top_growth_categories.map((x) => x.category) },
          series: [{ type: 'bar', data: data.top_growth_categories.map((x) => x.delta_abs ?? 0) }]
        }}
        height={360}
      />
    </div>

    <div className='card' style={{ marginTop: 12 }}><h3>Executive summary</h3><p>{data.executive_summary}</p></div>
  </div>
}
